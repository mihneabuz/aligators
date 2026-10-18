use std::{
    alloc::{AllocError, Allocator, Global, Layout, handle_alloc_error},
    cell::Cell,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::utils::Leaky;

// A cursor which keeps track of the amount of bytes currently allocated from the underlying buffer.
trait Cursor {
    fn new() -> Self;
    fn get(&self) -> usize;
    fn reset(&mut self);
    fn try_update<F>(&self, update: F) -> Option<usize>
    where
        F: FnMut(usize) -> Option<usize>;
}

impl Cursor for Cell<usize> {
    fn new() -> Self {
        Self::new(0)
    }

    fn get(&self) -> usize {
        self.get()
    }

    fn reset(&mut self) {
        self.set(0);
    }

    fn try_update<F>(&self, mut update: F) -> Option<usize>
    where
        F: FnMut(usize) -> Option<usize>,
    {
        let old = self.get();
        let new = update(old)?;
        self.set(new);
        Some(old)
    }
}

impl Cursor for AtomicUsize {
    fn new() -> Self {
        Self::new(0)
    }

    fn get(&self) -> usize {
        self.load(Ordering::Relaxed)
    }

    fn reset(&mut self) {
        *self.get_mut() = 0;
    }

    fn try_update<F>(&self, update: F) -> Option<usize>
    where
        F: FnMut(usize) -> Option<usize>,
    {
        self.try_update(Ordering::AcqRel, Ordering::Acquire, update)
            .ok()
    }
}

const BASE_ALIGN: usize = 8;

fn layout(size: usize) -> Option<Layout> {
    Layout::from_size_align(size, BASE_ALIGN).ok()
}

fn is_aligned(ptr: NonNull<u8>, align: usize) -> bool {
    ptr.addr().get().is_multiple_of(align)
}

#[derive(Debug)]
struct BumpInner<C, A: Allocator> {
    base: NonNull<[u8]>,
    cursor: C,
    backing: A,
}

// SAFETY: `base` is owned (or exclusively borrowed), so moving it across threads is fine.
unsafe impl<C, A: Allocator + Send> Send for BumpInner<C, A> {}

// SAFETY: `base` is owned (or exclusively borrowed), and through `&self` it is only handed out in
// disjoint blocks reserved by the cursor, so the allocator is `Sync` when the `Cursor` is `Sync`.
// `backing` is only used by the constructors and `drop`, so `A` doesn't need to be `Sync`.
unsafe impl<C: Sync, A: Allocator> Sync for BumpInner<C, A> {}

impl<'a, C: Cursor> BumpInner<C, Leaky<'a>> {
    fn with_buffer(buf: &'a mut [u8]) -> Self {
        Self {
            base: NonNull::from_mut(buf),
            cursor: C::new(),
            backing: Leaky::new(),
        }
    }
}

impl<C: Cursor> BumpInner<C, Global> {
    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl<C: Cursor, A: Allocator> BumpInner<C, A> {
    fn with_capacity_in(capacity: usize, backing: A) -> Self {
        match Self::try_with_capacity_in(capacity, backing) {
            Ok(bump) => bump,
            Err(_) => match layout(capacity) {
                Some(layout) => handle_alloc_error(layout),
                None => panic!("capacity overflow"),
            },
        }
    }

    fn try_with_capacity_in(capacity: usize, backing: A) -> Result<Self, AllocError> {
        let layout = layout(capacity).ok_or(AllocError)?;
        let base = backing.allocate(layout)?;
        Ok(Self {
            base,
            cursor: C::new(),
            backing,
        })
    }

    fn remaining_capacity(&self) -> usize {
        self.base.len() - self.cursor.get()
    }

    fn reset(&mut self) {
        self.cursor.reset();
    }

    fn try_fit_new(&self, layout: Layout) -> Option<NonNull<[u8]>> {
        let (base, size) = (self.base.cast::<u8>(), self.base.len());
        let mut offset = 0;

        let old = self.cursor.try_update(|cursor| {
            offset = base
                .as_ptr()
                .wrapping_add(cursor)
                .align_offset(layout.align());
            let end = cursor.checked_add(offset)?.checked_add(layout.size())?;
            (end <= size).then_some(end)
        })?;

        // SAFETY: the update only succeeded if `old + offset + layout.size() <= base.len()`, so the
        // block lies within `base`, and moving the cursor past it reserved it for this caller.
        Some(unsafe { NonNull::slice_from_raw_parts(base.add(old + offset), layout.size()) })
    }

    fn try_update_last<F>(&self, block: NonNull<u8>, layout: Layout, update: F) -> bool
    where
        F: Fn(usize) -> Option<usize>,
    {
        let base = self.base.cast::<u8>();
        let block_end = block.addr().get().wrapping_add(layout.size());

        self.cursor
            .try_update(|cursor| {
                if block_end != base.addr().get().wrapping_add(cursor) {
                    return None;
                }

                update(cursor)
            })
            .is_some()
    }

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.try_fit_new(layout).ok_or(AllocError)
    }

    /// # Safety
    ///
    /// Same as [`Allocator::deallocate`].
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.try_update_last(ptr, layout, |cursor| Some(cursor - layout.size()));
    }

    /// # Safety
    ///
    /// Same as [`Allocator::grow`].
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let (new_size, old_size) = (new_layout.size(), old_layout.size());
        let diff = new_size - old_size;

        if is_aligned(ptr, new_layout.align())
            && self.try_update_last(ptr, old_layout, |cursor| {
                (cursor + diff <= self.base.len()).then_some(cursor + diff)
            })
        {
            return Ok(NonNull::slice_from_raw_parts(ptr, new_size));
        }

        let new = self.allocate(new_layout)?;
        // SAFETY: `new` is a fresh block, disjoint from `ptr`, and `new_size >= old_size`.
        unsafe { ptr.copy_to_nonoverlapping(new.cast(), old_size) };
        Ok(new)
    }

    /// # Safety
    ///
    /// Same as [`Allocator::grow_zeroed`].
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let (new_size, old_size) = (new_layout.size(), old_layout.size());
        let diff = new_size - old_size;

        if is_aligned(ptr, new_layout.align())
            && self.try_update_last(ptr, old_layout, |cursor| {
                (cursor + diff <= self.base.len()).then_some(cursor + diff)
            })
        {
            // SAFETY: the cursor was just moved past `ptr + new.size`, so `old_size..new_size`
            // lies inside `base` and belongs to this block.
            unsafe {
                ptr.add(old_size).write_bytes(0, diff);
            }

            return Ok(NonNull::slice_from_raw_parts(ptr, new_size));
        }

        let new = self.allocate(new_layout)?;
        // SAFETY: `new` is a fresh block, disjoint from `ptr` and `new_size >= old_size`.
        // `new` is at least `new_size` long, so `old_size..new_size` is inside it.
        unsafe {
            ptr.copy_to_nonoverlapping(new.cast(), old_size);
            new.cast::<u8>().add(old_size).write_bytes(0, diff);
        }
        Ok(new)
    }

    /// # Safety
    ///
    /// Same as [`Allocator::shrink`].
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let (new_size, old_size) = (new_layout.size(), old_layout.size());
        let diff = old_size - new_size;

        if is_aligned(ptr, new_layout.align()) {
            if self.try_update_last(ptr, old_layout, |cursor| Some(cursor - diff)) {
                return Ok(NonNull::slice_from_raw_parts(ptr, new_size));
            }

            return Ok(NonNull::slice_from_raw_parts(ptr, old_size));
        }

        let new = self.allocate(new_layout)?;
        // SAFETY: `new` is a fresh block, disjoint from `ptr`, and `new_size <= old_size`, so
        // both ranges are valid for `new_size` bytes.
        unsafe { ptr.copy_to_nonoverlapping(new.cast(), new_size) };
        Ok(new)
    }
}

impl<C, A: Allocator> Drop for BumpInner<C, A> {
    fn drop(&mut self) {
        if let Some(layout) = layout(self.base.len()) {
            // SAFETY: `base` came from `backing.allocate` (or is borrowed, with `Leaky` doing
            // nothing), and `base.len()` is in the fitting range for the requested layout.
            unsafe { self.backing.deallocate(self.base.cast(), layout) };
        }
    }
}

/// Defines a public newtype around [`BumpInner`] with the given cursor, forwarding the whole API.
macro_rules! bump {
    ($(#[$meta:meta])* $name:ident, $cursor:ty) => {
        $(#[$meta])*
        #[derive(Debug)]
        pub struct $name<A: Allocator = Global>(BumpInner<$cursor, A>);

        impl<'a> $name<Leaky<'a>> {
            /// Creates a new allocator backed by `buf`.
            pub fn with_buffer(buf: &'a mut [u8]) -> Self {
                Self(BumpInner::with_buffer(buf))
            }
        }

        impl $name<Global> {
            /// Creates a new allocator backed by the [`Global`] allocator, with at least
            /// `capacity` bytes.
            ///
            /// # Panics
            ///
            /// If the [`Global`] allocator fails.
            #[must_use]
            pub fn with_capacity(capacity: usize) -> Self {
                Self(BumpInner::with_capacity(capacity))
            }
        }

        impl<A: Allocator> $name<A> {
            /// Creates a new allocator backed by `backing`, with at least `capacity` bytes.
            ///
            /// # Panics
            ///
            /// If `backing` fails.
            #[must_use]
            pub fn with_capacity_in(capacity: usize, backing: A) -> Self {
                Self(BumpInner::with_capacity_in(capacity, backing))
            }

            /// Creates a new allocator backed by `backing`, with at least `capacity` bytes.
            ///
            /// # Errors
            ///
            /// Returns [`AllocError`] if `backing` fails.
            pub fn try_with_capacity_in(capacity: usize, backing: A) -> Result<Self, AllocError> {
                BumpInner::try_with_capacity_in(capacity, backing).map(Self)
            }

            /// Returns the number of bytes left past the last allocation.
            pub fn remaining_capacity(&self) -> usize {
                self.0.remaining_capacity()
            }

            /// Frees every allocation at once.
            pub fn reset(&mut self) {
                self.0.reset();
            }
        }

        // SAFETY: live blocks never overlap, since the cursor only rewinds past the last block.
        // They stay valid until `reset` (needs `&mut self`) or drop, and moving the allocator
        // doesn't move `base`.
        unsafe impl<A: Allocator> Allocator for $name<A> {
            fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
                self.0.allocate(layout)
            }

            unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
                // SAFETY: forwarded from the caller.
                unsafe { self.0.deallocate(ptr, layout) }
            }

            unsafe fn grow(
                &self,
                ptr: NonNull<u8>,
                old_layout: Layout,
                new_layout: Layout,
            ) -> Result<NonNull<[u8]>, AllocError> {
                // SAFETY: forwarded from the caller.
                unsafe { self.0.grow(ptr, old_layout, new_layout) }
            }

            unsafe fn grow_zeroed(
                &self,
                ptr: NonNull<u8>,
                old_layout: Layout,
                new_layout: Layout,
            ) -> Result<NonNull<[u8]>, AllocError> {
                // SAFETY: forwarded from the caller.
                unsafe { self.0.grow_zeroed(ptr, old_layout, new_layout) }
            }

            unsafe fn shrink(
                &self,
                ptr: NonNull<u8>,
                old_layout: Layout,
                new_layout: Layout,
            ) -> Result<NonNull<[u8]>, AllocError> {
                // SAFETY: forwarded from the caller.
                unsafe { self.0.shrink(ptr, old_layout, new_layout) }
            }
        }
    };
}

bump! {
    /// Bump allocator for a single thread.
    ///
    /// The allocator hands out blocks from a single fixed-size buffer, which is either borrowed
    /// ([`Bump::with_buffer`]) or allocated from a backing allocator ([`Bump::with_capacity_in`]).
    /// It never grows: once the buffer is full, allocations fail with [`AllocError`].
    ///
    /// A cursor tracks how many bytes of the buffer are in use. Every allocation moves the cursor
    /// forward, so allocating is just a pointer bump. Only the most recent allocation can give
    /// memory back:
    ///
    /// - Deallocating the most recent block moves the cursor back; deallocating any other block
    ///   does nothing.
    /// - Growing the most recent block happens in place, as long as it fits. Any other block is
    ///   copied into a new allocation.
    /// - Shrinking happens in place whenever the pointer is already aligned for the new layout.
    ///   For the most recent block, it also moves the cursor back.
    ///
    /// The memory of all other blocks is reclaimed at once with [`Bump::reset`], or when the
    /// allocator is dropped. Use [`AtomicBump`] to share a bump allocator between threads.
    ///
    /// ```
    /// use aligators::Bump;
    ///
    /// let mut buf = [0u8; 1024];
    /// let mut bump = Bump::with_buffer(&mut buf);
    ///
    /// {
    ///     let mut vec = Vec::new_in(&bump);
    ///     vec.extend([1, 2, 3]);
    ///     let boxed = Box::new_in(42, &bump);
    ///     assert_eq!(vec.iter().sum::<i32>(), 6);
    ///     assert_eq!(*boxed, 42);
    /// }
    ///
    /// // Every block has been dropped, so all of the memory can be reused.
    /// bump.reset();
    /// assert_eq!(bump.remaining_capacity(), 1024);
    /// ```
    Bump, Cell<usize>
}

bump! {
    /// Bump allocator which can be shared between threads.
    ///
    /// This is the thread-safe version of [`Bump`] and behaves the same way, but its cursor is an
    /// atomic, so `&AtomicBump` can be used as an allocator from several threads at once.
    /// Allocating is a compare-and-swap loop on the cursor, so it is a bit slower than [`Bump`]
    /// and can retry under contention.
    ///
    /// The allocator hands out blocks from a single fixed-size buffer, which is either borrowed
    /// ([`AtomicBump::with_buffer`]) or allocated from a backing allocator
    /// ([`AtomicBump::with_capacity_in`]). It never grows: once the buffer is full, allocations
    /// fail with [`AllocError`].
    ///
    /// As with [`Bump`], only the most recent allocation can give memory back. With several
    /// threads, "most recent" means the last allocation made by any thread. A block that another
    /// thread has since allocated past can only be reclaimed by [`AtomicBump::reset`], or by
    /// dropping the allocator. `reset` takes `&mut self`, so no thread can be using the allocator
    /// at that point.
    ///
    /// For situations with high contention or collection which try to grow often, consider giving
    /// each thread it's own [`Bump`] instead.
    ///
    /// ```
    /// use aligators::AtomicBump;
    /// use std::thread;
    ///
    /// let mut bump = AtomicBump::with_capacity(4096);
    ///
    /// thread::scope(|scope| {
    ///     for i in 0..4 {
    ///         let bump = &bump;
    ///         scope.spawn(move || {
    ///             let mut vec = Vec::new_in(bump);
    ///             vec.extend(0..i);
    ///             assert_eq!(vec.len(), i);
    ///         });
    ///     }
    /// });
    ///
    /// // Every thread is done, so all of the memory can be reused.
    /// bump.reset();
    /// assert_eq!(bump.remaining_capacity(), 4096);
    /// ```
    AtomicBump, AtomicUsize
}

#[cfg(test)]
mod tests {
    use crate::page::Page;

    use super::*;

    #[test]
    fn moving() {
        let mut buf = [0u8; 64];

        let v = {
            let bump = Bump::with_buffer(&mut buf);
            let mut v = Vec::with_capacity_in(8, bump);
            v.extend(0..8u8);
            v
        };

        let moved = Box::new(v);
        assert!(moved.iter().copied().eq(0..8));
    }

    #[test]
    fn page_backing() {
        let mut alloc = Bump::with_capacity_in(512, Page);
        assert_eq!(alloc.remaining_capacity(), Page::page_size());

        {
            let box1 = Box::new_in([1_usize; 64], &alloc);

            let mut vec = Vec::new_in(&alloc);
            for i in 0..100_usize {
                vec.push(i * 10);
            }

            let box2 = Box::new_in(7_usize, &alloc);

            vec.extend_from_slice(&box1[..]);
            *vec.first_mut().unwrap() = *box2;
        }

        alloc.reset();
        assert_eq!(alloc.remaining_capacity(), Page::page_size());
    }
}
