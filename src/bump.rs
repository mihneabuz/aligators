use std::{
    alloc::{AllocError, Allocator, Global, Layout, handle_alloc_error}, cell::Cell, ptr::NonNull,
};

use crate::utils::Leaky;

const BASE_ALIGN: usize = 8;

fn layout(size: usize) -> Option<Layout> {
    Layout::from_size_align(size, BASE_ALIGN).ok()
}

fn is_aligned(ptr: NonNull<u8>, align: usize) -> bool {
    ptr.addr().get().is_multiple_of(align)
}

#[derive(Debug)]
pub struct BumpAllocator<A: Allocator = Global> {
    base: NonNull<[u8]>,
    cursor: Cell<usize>,
    backing: A,
}

// SAFETY: `base` is owned (or exclusively borrowed), so moving it across threads is fine.
unsafe impl<A: Allocator + Send> Send for BumpAllocator<A> {}

impl<'a> BumpAllocator<Leaky<'a>> {
    /// Create a new allocator backed by `buf`.
    pub fn with_buffer(buf: &'a mut [u8]) -> Self {
        Self {
            base: NonNull::from_mut(buf),
            cursor: Cell::new(0),
            backing: Leaky::new(),
        }
    }
}

impl BumpAllocator<Global> {
    /// Create a new allocator backed by the `Global` allocator and with at least `capacity`.
    ///
    /// # Panics
    /// When the `Global` allocator returns `AllocError`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl<A: Allocator> BumpAllocator<A> {
    /// Create a new allocator backed by the `backing` allocator and with at least `capacity`.
    ///
    /// # Panics
    /// When the `backing` allocator returns `AllocError`.
    pub fn with_capacity_in(capacity: usize, backing: A) -> Self {
        match Self::try_with_capacity_in(capacity, backing) {
            Ok(bump) => bump,
            Err(_) => match layout(capacity) {
                Some(layout) => handle_alloc_error(layout),
                None => panic!("capacity overflow"),
            },
        }
    }

    /// Create a new allocator backed by the `backing` allocator and with at least `capacity`.
    ///
    /// # Errors
    /// Returns `AllocError` when the `backing` allocator fails.
    pub fn try_with_capacity_in(capacity: usize, backing: A) -> Result<Self, AllocError> {
        let layout = layout(capacity).ok_or(AllocError)?;
        let base = backing.allocate(layout)?;
        Ok(Self {
            base,
            cursor: Cell::new(0),
            backing,
        })
    }

    pub fn remaining_capacity(&self) -> usize {
        self.base.len() - self.cursor.get()
    }

    pub fn reset(&mut self) {
        self.cursor.set(0);
    }

    fn fit_block(&self, layout: Layout) -> Option<NonNull<[u8]>> {
        // SAFETY: `cursor <= base.len()`, so this is in bounds or one past the end.
        let free = unsafe { self.base.cast::<u8>().add(self.cursor.get()) };
        let available = self.base.len() - self.cursor.get();

        let offset = free.align_offset(layout.align());
        if offset + layout.size() > available {
            return None;
        }

        self.cursor.update(|pos| pos + offset + layout.size());

        // SAFETY: `offset + size <= available`, so the block lies within `base`.
        Some(unsafe { NonNull::slice_from_raw_parts(free.add(offset), layout.size()) })
    }

    fn last_block(&self, ptr: NonNull<u8>, layout: Layout) -> bool {
        let end = self.base.addr().get().wrapping_add(self.cursor.get());
        ptr.addr().get().wrapping_add(layout.size()) == end
    }
}

impl<A: Allocator> Drop for BumpAllocator<A> {
    fn drop(&mut self) {
        if let Some(layout) = layout(self.base.len()) {
            // SAFETY: `base` was allocated by `backing` with this same layout.
            unsafe { self.backing.deallocate(self.base.cast(), layout) };
        }
    }
}

// SAFETY: live blocks never overlap, since the cursor only rewinds past the last block.
// They stay valid until `reset` (needs `&mut self`) or drop.
unsafe impl<A: Allocator> Allocator for BumpAllocator<A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.fit_block(layout).ok_or(AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if self.last_block(ptr, layout) {
            self.cursor.update(|pos| pos - layout.size());
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if !self.last_block(ptr, old_layout) || !is_aligned(ptr, new_layout.align()) {
            let new = self.allocate(new_layout)?;
            // SAFETY: `new` is a fresh block past the cursor, disjoint from `ptr`,
            // and `new_layout.size() >= old_layout.size()`.
            unsafe { ptr.copy_to_nonoverlapping(new.cast(), old_layout.size()) };
            return Ok(new);
        }

        let diff = new_layout.size() - old_layout.size();
        if diff > self.base.len() - self.cursor.get() {
            return Err(AllocError);
        }

        self.cursor.update(|pos| pos + diff);

        Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()))
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if !self.last_block(ptr, old_layout) || !is_aligned(ptr, new_layout.align()) {
            let new = self.allocate(new_layout)?;
            // SAFETY: `new` is disjoint from `ptr` and holds `new_layout.size()` bytes,
            // covering both the copy and the zeroed tail.
            unsafe {
                ptr.copy_to_nonoverlapping(new.cast(), old_layout.size());
                let diff = new_layout.size() - old_layout.size();
                new.cast::<u8>().add(old_layout.size()).write_bytes(0, diff);
            }
            return Ok(new);
        }

        let diff = new_layout.size() - old_layout.size();
        if diff > self.base.len() - self.cursor.get() {
            return Err(AllocError);
        }

        self.cursor.update(|pos| pos + diff);

        // SAFETY: `ptr` is the last block and the cursor was just advanced by `diff`,
        // so the tail lies within `base`.
        unsafe {
            ptr.add(old_layout.size()).write_bytes(0, diff);
        }

        Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()))
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if !self.last_block(ptr, old_layout) || !is_aligned(ptr, new_layout.align()) {
            let new = self.allocate(new_layout)?;
            // SAFETY: `new` is disjoint from `ptr`, and we copy only the `new_layout.size()`
            // bytes that fit.
            unsafe { ptr.copy_to_nonoverlapping(new.cast(), new_layout.size()) };
            return Ok(new);
        }

        let diff = old_layout.size() - new_layout.size();
        self.cursor.update(|pos| pos - diff);

        Ok(NonNull::slice_from_raw_parts(ptr, new_layout.size()))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use crate::page::PageAllocator;

    use super::*;

    fn layout(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).unwrap()
    }

    fn bytes<'a>(block: NonNull<[u8]>) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(block.cast::<u8>().as_ptr(), block.len()) }
    }

    fn check_block(block: NonNull<[u8]>, layout: Layout, buffer: &Range<*const u8>) {
        let start = block.cast::<u8>().as_ptr().cast_const();

        assert!(block.len() >= layout.size());
        assert!(start.addr().is_multiple_of(layout.align()));
        assert!(buffer.start.addr() <= start.addr());
        assert!(start.addr() + block.len() <= buffer.end.addr());

        bytes(block).fill(0xAB);
    }

    #[test]
    fn allocate() {
        let mut buf = [0u8; 64];
        let range = buf.as_ptr_range();
        let alloc = BumpAllocator::with_buffer(&mut buf);

        let block = alloc.allocate(layout(16, 1)).unwrap();
        check_block(block, layout(16, 1), &range);
        assert_eq!(block.len(), 16);

        unsafe { alloc.deallocate(block.cast(), layout(16, 1)) };
    }

    #[test]
    fn allocate_too_large() {
        let mut buf = [0u8; 64];
        let alloc = BumpAllocator::with_buffer(&mut buf);

        assert!(alloc.allocate(layout(32, 1)).is_ok());
        assert!(alloc.allocate(layout(33, 1)).is_err());
        assert!(alloc.allocate(layout(32, 1)).is_ok());
    }

    #[test]
    fn alignment() {
        #[repr(align(128))]
        struct Aligned([u8; 256]);
        let mut storage = Aligned([0; 256]);

        let buf = &mut storage.0[1..];
        let range = buf.as_ptr_range();

        let alloc = BumpAllocator::with_buffer(buf);
        for align in [1, 2, 4, 8, 16, 32, 64, 128] {
            let block = alloc.allocate(layout(8, align)).unwrap();
            check_block(block, layout(8, align), &range);
            assert_eq!(block.len(), 8);
            unsafe { alloc.deallocate(block.cast(), layout(8, align)) };
        }

        assert!(alloc.allocate(layout(1, 4096)).is_err());
    }

    #[test]
    fn empty_buffer() {
        let alloc = BumpAllocator::with_buffer(&mut []);

        assert!(alloc.allocate(layout(1, 1)).is_err());
        assert!(alloc.allocate(layout(1, 8)).is_err());
    }

    #[test]
    fn zero_size() {
        let mut buf = [0u8; 64];
        let alloc = BumpAllocator::with_buffer(&mut buf);

        let empty = alloc.allocate(layout(0, 8)).unwrap();
        assert_eq!(empty.len(), 0);
        assert!(empty.cast::<u8>().as_ptr().addr().is_multiple_of(8));
    }

    #[test]
    fn grow_in_place() {
        let mut buf = [0u8; 64];
        let range = buf.as_ptr_range();
        let alloc = BumpAllocator::with_buffer(&mut buf);

        let block = alloc.allocate(layout(8, 1)).unwrap();
        bytes(block)[..8].copy_from_slice(b"aligator");

        let grown = unsafe { alloc.grow(block.cast(), layout(8, 1), layout(48, 1)) }.unwrap();
        assert_eq!(grown.cast::<u8>(), block.cast::<u8>());
        assert_eq!(&bytes(grown)[..8], b"aligator");
        check_block(grown, layout(48, 1), &range);

        unsafe { alloc.deallocate(grown.cast(), layout(48, 1)) };
    }

    #[test]
    fn grow_too_large() {
        let mut buf = [0u8; 64];
        let alloc = BumpAllocator::with_buffer(&mut buf);

        let block = alloc.allocate(layout(8, 1)).unwrap();
        bytes(block)[..8].copy_from_slice(b"aligator");

        assert!(unsafe { alloc.grow(block.cast(), layout(8, 1), layout(65, 1)) }.is_err());

        assert_eq!(&bytes(block)[..8], b"aligator");
        unsafe { alloc.deallocate(block.cast(), layout(8, 1)) };
    }

    #[test]
    fn grow_zeroed() {
        let mut buf = [0u8; 64];
        let alloc = BumpAllocator::with_buffer(&mut buf);

        let block = alloc.allocate(layout(8, 1)).unwrap();
        bytes(block).fill(0xFF);

        let grown =
            unsafe { alloc.grow_zeroed(block.cast(), layout(8, 1), layout(32, 1)) }.unwrap();
        assert!(bytes(grown)[..8].iter().all(|&b| b == 0xFF));
        assert!(bytes(grown)[8..].iter().all(|&b| b == 0));

        unsafe { alloc.deallocate(grown.cast(), layout(32, 1)) };
    }

    #[test]
    fn grow_align() {
        let mut buf = [0u8; 64];
        let alloc = BumpAllocator::with_buffer(&mut buf);

        let block = alloc.allocate(layout(8, 4)).unwrap();
        bytes(block)[..8].copy_from_slice(b"aligator");

        // Lower alignment, should reuse pointer
        let grown = unsafe { alloc.grow(block.cast(), layout(8, 4), layout(16, 2)) }.unwrap();
        assert_eq!(grown.cast::<u8>(), block.cast::<u8>());
        assert_eq!(&bytes(grown)[..8], b"aligator");

        // Higher alignment, might have to move
        let growner = unsafe { alloc.grow(grown.cast(), layout(16, 2), layout(32, 8)) }.unwrap();
        assert_eq!(&bytes(growner)[..8], b"aligator");

        unsafe { alloc.deallocate(growner.cast(), layout(32, 8)) };
    }

    #[test]
    fn shrink_in_place() {
        let mut buf = [0u8; 64];
        let alloc = BumpAllocator::with_buffer(&mut buf);

        let block = alloc.allocate(layout(32, 1)).unwrap();
        bytes(block)[..8].copy_from_slice(b"aligator");

        let shrunk = unsafe { alloc.shrink(block.cast(), layout(32, 1), layout(8, 1)) }.unwrap();
        assert_eq!(shrunk.cast::<u8>(), block.cast::<u8>());
        assert_eq!(&bytes(shrunk)[..8], b"aligator");

        unsafe { alloc.deallocate(shrunk.cast(), layout(8, 1)) };
    }

    #[test]
    fn shrink_align() {
        let mut buf = [0u8; 64];
        let alloc = BumpAllocator::with_buffer(&mut buf);

        let block = alloc.allocate(layout(32, 4)).unwrap();
        bytes(block)[..8].copy_from_slice(b"aligator");

        // Lower alignment, should reuse pointer
        let shrunk = unsafe { alloc.shrink(block.cast(), layout(32, 4), layout(16, 1)) }.unwrap();
        assert_eq!(shrunk.cast::<u8>(), block.cast::<u8>());
        assert_eq!(&bytes(shrunk)[..8], b"aligator");

        // Higher alignment, might have to move
        let shrunker = unsafe { alloc.shrink(shrunk.cast(), layout(16, 1), layout(8, 8)) }.unwrap();
        assert_eq!(&bytes(shrunker)[..8], b"aligator");

        unsafe { alloc.deallocate(shrunker.cast(), layout(8, 8)) };
    }

    #[test]
    fn vec() {
        let mut buf = [0u8; 512];
        let bump = BumpAllocator::with_buffer(&mut buf);

        let mut vec = Vec::new_in(&bump);
        for i in 0..50 {
            vec.push(i * 4);
        }
        vec.truncate(20);
        vec.shrink_to_fit();

        assert_eq!(&vec[..], &(0..20).map(|i| i * 4).collect::<Vec<_>>());
    }

    #[test]
    fn boxed() {
        let mut buf = [0u8; 512];
        let mut bump = BumpAllocator::with_buffer(&mut buf);

        {
            let mut box1 = Box::new_in(10, &bump);
            let mut box2 = Box::new_in(20, &bump);
            let mut box3 = Box::new_in(30, &bump);

            assert_eq!(*box1, 10);
            assert_eq!(*box2, 20);
            assert_eq!(*box3, 30);

            *box1 += 1;
            *box2 += 2;
            *box3 += 3;

            drop(box1);
            drop(box3);

            assert_eq!(*box2, 22);
        }

        bump.reset();

        let big = Box::new_in([8u8; 100], &bump);
        assert_eq!(&*big, &[8u8; 100]);
    }

    #[test]
    fn moving() {
        let mut buf = [0u8; 64];

        let v = {
            let bump = BumpAllocator::with_buffer(&mut buf);
            let mut v = Vec::with_capacity_in(8, bump);
            v.extend(0..8u8);
            v
        };

        let moved = Box::new(v);
        assert!(moved.iter().copied().eq(0..8));
    }

    #[test]
    fn page_backing() {
        let mut alloc = BumpAllocator::with_capacity_in(512, PageAllocator);
        assert_eq!(alloc.remaining_capacity(), PageAllocator::page_size());

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
        assert_eq!(alloc.remaining_capacity(), PageAllocator::page_size());
    }
}
