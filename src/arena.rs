use std::{
    alloc::{AllocError, Allocator, Global, Layout, handle_alloc_error},
    cell::Cell,
    mem,
    ptr::{self, NonNull},
};

const DEFAULT_INITIAL_CAPACITY: usize = 4096;

const HEADER_SIZE: usize = size_of::<BlockHeader>();
const HEADER_ALIGN: usize = align_of::<BlockHeader>();

fn is_aligned(ptr: NonNull<u8>, align: usize) -> bool {
    ptr.addr().get().is_multiple_of(align)
}

fn layout(size: usize) -> Option<Layout> {
    Layout::from_size_align(size, HEADER_ALIGN).ok()
}

struct BlockHeader {
    prev: NonNull<BlockHeader>,
    size: usize,
}

struct EmptyBlock(BlockHeader);

// SAFETY: the empty block is never written to, since nothing fits in it.
unsafe impl Sync for EmptyBlock {}

// Sentinel block for an arena which didn't allocate yet.
static EMPTY_BLOCK: EmptyBlock = EmptyBlock(BlockHeader {
    prev: NonNull::from_ref(&EMPTY_BLOCK.0),
    size: HEADER_SIZE,
});

fn empty_block() -> NonNull<BlockHeader> {
    NonNull::from_ref(&EMPTY_BLOCK.0)
}

fn is_empty_block(ptr: NonNull<BlockHeader>) -> bool {
    ptr::eq(ptr.as_ptr(), &raw const EMPTY_BLOCK.0)
}

/// Growable bump allocator for a single thread.
///
/// The allocator hands out blocks from a list of buffers allocated from a backing allocator. When
/// the current buffer is full, it allocates a new one of four times the size (or larger, if needed
/// to fit the allocation) and continues from there, so allocations only fail if `backing` does.
/// [`Arena::new`] doesn't allocate until the first allocation is made, while
/// [`Arena::with_capacity_in`] allocates the first buffer right away.
///
/// Inside the current buffer it works like a [`Bump`](crate::Bump): a cursor tracks how many
/// bytes are in use, and every allocation moves it forward. Only the most recent allocation in the
/// current buffer can give memory back:
///
/// - Deallocating the most recent block moves the cursor back; deallocating any other block does
///   nothing.
/// - Growing the most recent block happens in place, as long as it fits in the current buffer. Any
///   other block is copied into a new allocation, which may start a new buffer.
/// - Shrinking happens in place whenever the pointer is already aligned for the new layout. For
///   the most recent block, it also moves the cursor back.
///
/// Blocks never move, so they stay valid when a new buffer is started. Their memory is reclaimed
/// at once with [`Arena::reset`], which keeps only the most recent buffer around for reuse, or
/// when the allocator is dropped.
///
/// ```
/// use aligators::Arena;
///
/// let mut arena = Arena::new();
///
/// {
///     // Grows past the first buffer, so the arena allocates more of them.
///     let mut vec = Vec::new_in(&arena);
///     vec.extend(0..10_000);
///     let boxed = Box::new_in(42, &arena);
///     assert_eq!(vec.len(), 10_000);
///     assert_eq!(*boxed, 42);
/// }
///
/// // Every block has been dropped, so the memory can be reused.
/// arena.reset();
/// let boxed = Box::new_in(7, &arena);
/// assert_eq!(*boxed, 7);
/// ```
///
/// An arena can't be shared between threads. Give each thread its own arena instead. They can
/// still share a backing allocator, since `&A` is an allocator too:
///
/// ```
/// use aligators::Arena;
/// use std::{alloc::Global, thread};
///
/// let backing = Global;
///
/// thread::scope(|scope| {
///     for i in 0..4 {
///         let backing = &backing;
///         scope.spawn(move || {
///             let arena = Arena::with_capacity_in(1024, backing);
///             let mut vec = Vec::new_in(&arena);
///             vec.extend(0..i);
///             assert_eq!(vec.len(), i);
///         });
///     }
/// });
/// ```
#[derive(Debug)]
pub struct Arena<A: Allocator = Global> {
    block: Cell<NonNull<BlockHeader>>,
    cursor: Cell<usize>,
    backing: A,
}

// SAFETY: the arena owns all of its blocks, and the empty block is never written to, so moving it
// across threads is fine when `backing` can be moved too.
unsafe impl<A: Allocator + Send> Send for Arena<A> {}

impl<A: Allocator + Default> Default for Arena<A> {
    fn default() -> Self {
        Self::new_in(A::default())
    }
}

impl Arena {
    /// Creates a new allocator backed by the [`Global`] allocator.
    ///
    /// It doesn't allocate until the first allocation is made.
    #[must_use]
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    /// Creates a new allocator backed by the [`Global`] allocator, with a first block of at least
    /// `capacity` bytes.
    ///
    /// # Panics
    ///
    /// If the [`Global`] allocator fails.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl<A: Allocator> Arena<A> {
    /// Creates a new allocator backed by `backing`.
    ///
    /// It doesn't allocate until the first allocation is made.
    #[must_use]
    pub fn new_in(backing: A) -> Self {
        Self {
            block: Cell::new(empty_block()),
            cursor: Cell::new(HEADER_SIZE),
            backing,
        }
    }

    /// Creates a new allocator backed by `backing`, with a first block of at least `capacity`
    /// bytes.
    ///
    /// # Panics
    ///
    /// If `backing` fails.
    #[must_use]
    pub fn with_capacity_in(capacity: usize, backing: A) -> Self {
        match Self::try_with_capacity_in(capacity, backing) {
            Ok(arena) => arena,
            Err(_) => match HEADER_SIZE.checked_add(capacity).and_then(layout) {
                Some(layout) => handle_alloc_error(layout),
                None => panic!("capacity overflow"),
            },
        }
    }

    /// Creates a new allocator backed by `backing`, with a first block of at least `capacity`
    /// bytes.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if `backing` fails.
    pub fn try_with_capacity_in(capacity: usize, backing: A) -> Result<Self, AllocError> {
        let arena = Self::new_in(backing);
        if capacity > 0 {
            arena.new_block(capacity)?;
        }
        Ok(arena)
    }

    /// Returns the number of bytes left past the last allocation in the current buffer.
    ///
    /// An allocation larger than this (including the padding needed to align it) starts a new
    /// buffer. Before the first buffer is allocated, this is `0`.
    pub fn remaining_capacity(&self) -> usize {
        // SAFETY: `block` is either the empty block or a live block, both starting with a header.
        unsafe { self.block.get().as_ref() }.size - self.cursor.get()
    }

    /// Returns the total number of bytes allocated from `backing`, over all buffers.
    ///
    /// This includes memory that is no longer used, like the end of a buffer that was too small for
    /// the next allocation, and a small header at the start of each buffer.
    pub fn allocated_bytes(&self) -> usize {
        let mut block = self.block.get();
        let mut total = 0;

        while !is_empty_block(block) {
            // SAFETY: every block other than the empty one is live and starts with a header.
            let header = unsafe { block.as_ref() };
            total += header.size;
            block = header.prev;
        }

        total
    }

    /// Frees every allocation at once.
    ///
    /// Every buffer except the most recent one is returned to `backing`. The most recent buffer is
    /// usually the largest, and is kept so the arena doesn't need to allocate again until it fills
    /// up.
    pub fn reset(&mut self) {
        let mut block = *self.block.get_mut();
        if is_empty_block(block) {
            return;
        }

        // SAFETY: `block` is live, since it isn't the empty one, and `&mut self` means no other
        // reference to its header exists.
        let prev = mem::replace(unsafe { &mut block.as_mut().prev }, empty_block());

        // SAFETY: `&mut self` means no allocation is still in use, and the current block no
        // longer links to the freed ones.
        unsafe { self.free_blocks(prev) };
        *self.cursor.get_mut() = HEADER_SIZE;
    }

    // Returns the capacity of the next block, so that it fits `layout`.
    fn next_capacity(&self, layout: Layout) -> Option<usize> {
        // SAFETY: `block` is either the empty block or a live block, both starting with a header.
        let size = unsafe { self.block.get().as_ref() }.size;
        let next = (size - HEADER_SIZE)
            .saturating_mul(4)
            .max(DEFAULT_INITIAL_CAPACITY);

        // Worst-case padding needed to align the allocation.
        let needed = layout.size().checked_add(layout.align() - 1)?;
        Some(next.max(needed))
    }

    // Bumps a block for `layout` from the current block, if it fits.
    #[inline]
    fn try_fit(&self, layout: Layout) -> Option<NonNull<[u8]>> {
        let block = self.block.get();
        // SAFETY: `block` is either the empty block or a live block, both starting with a header.
        let size = unsafe { block.as_ref() }.size;
        let (base, cursor) = (block.cast::<u8>(), self.cursor.get());

        let offset = base
            .as_ptr()
            .wrapping_add(cursor)
            .align_offset(layout.align());
        let end = cursor.checked_add(offset)?.checked_add(layout.size())?;
        if end > size {
            return None;
        }

        self.cursor.set(end);
        // SAFETY: `cursor + offset + layout.size() <= size`, so the block lies within the current
        // block, and moving the cursor past it reserved it. For the empty block this only happens
        // for a zero-sized layout, which gets the address just past its header.
        Some(unsafe { NonNull::slice_from_raw_parts(base.add(cursor + offset), layout.size()) })
    }

    // Moves the cursor to `update(cursor, size)` if `ptr` is the last block bumped from the current
    // block, returning whether it did.
    fn try_update_last<F>(&self, ptr: NonNull<u8>, layout: Layout, update: F) -> bool
    where
        F: FnOnce(usize, usize) -> Option<usize>,
    {
        let block = self.block.get();
        // SAFETY: `block` is either the empty block or a live block, both starting with a header.
        let size = unsafe { block.as_ref() }.size;
        let cursor = self.cursor.get();

        // Blocks never overlap, so only the last block of the current block ends at the cursor.
        let block_end = ptr.addr().get().wrapping_add(layout.size());
        if block_end != block.addr().get().wrapping_add(cursor) {
            return false;
        }

        match update(cursor, size) {
            Some(cursor) => {
                self.cursor.set(cursor);
                true
            }
            None => false,
        }
    }

    // Tries to resize `ptr` in place, if it is the last block and aligned for `new_layout`.
    fn try_resize_last(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> bool {
        is_aligned(ptr, new_layout.align())
            && self.try_update_last(ptr, old_layout, |cursor, size| {
                let end = (cursor - old_layout.size()).checked_add(new_layout.size())?;
                (end <= size).then_some(end)
            })
    }

    // Frees `block` and every block before it, up to the empty block.
    //
    // # Safety
    //
    // `block` must be the empty block or a live block of this arena, and no allocation from the
    // freed blocks may be used afterwards.
    unsafe fn free_blocks(&self, mut block: NonNull<BlockHeader>) {
        while !is_empty_block(block) {
            // SAFETY: every block other than the empty one is live and starts with a header.
            let BlockHeader { prev, size } = unsafe { block.read() };

            if let Some(layout) = layout(size) {
                // SAFETY: `block` came from `backing.allocate` with a layout of `size` bytes or
                // less, and `size` is the length it returned, so it is in the fitting range.
                unsafe { self.backing.deallocate(block.cast(), layout) };
            }

            block = prev;
        }
    }

    // Slow path of `allocate`, kept out of line so the fast path stays small enough to inline.
    #[cold]
    #[inline(never)]
    fn allocate_in_new_block(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.new_block(self.next_capacity(layout).ok_or(AllocError)?)?;
        // The new block has room for `layout` even with the worst-case padding.
        self.try_fit(layout).ok_or(AllocError)
    }

    // Allocates a new block with room for at least `capacity` bytes and makes it current.
    fn new_block(&self, capacity: usize) -> Result<(), AllocError> {
        let size = HEADER_SIZE.checked_add(capacity).ok_or(AllocError)?;
        let new = self.backing.allocate(layout(size).ok_or(AllocError)?)?;
        let header = new.cast::<BlockHeader>();

        // SAFETY: `new` is at least `HEADER_SIZE` bytes long and aligned for `BlockHeader`.
        unsafe {
            header.write(BlockHeader {
                prev: self.block.get(),
                size: new.len(),
            });
        }

        self.block.set(header);
        self.cursor.set(HEADER_SIZE);

        Ok(())
    }
}

// SAFETY: live blocks never overlap, since the cursor only rewinds past the last block of the
// current block. Blocks are only freed on drop, and moving the arena doesn't move them.
unsafe impl<A: Allocator> Allocator for Arena<A> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match self.try_fit(layout) {
            Some(block) => Ok(block),
            None => self.allocate_in_new_block(layout),
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.try_update_last(ptr, layout, |cursor, _| Some(cursor - layout.size()));
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let (new_size, old_size) = (new_layout.size(), old_layout.size());

        if self.try_resize_last(ptr, old_layout, new_layout) {
            return Ok(NonNull::slice_from_raw_parts(ptr, new_size));
        }

        let new = self.allocate(new_layout)?;
        // SAFETY: `new` is a fresh block, disjoint from `ptr`, and `new_size >= old_size`.
        unsafe { ptr.copy_to_nonoverlapping(new.cast(), old_size) };
        Ok(new)
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let (new_size, old_size) = (new_layout.size(), old_layout.size());
        let diff = new_size - old_size;

        if self.try_resize_last(ptr, old_layout, new_layout) {
            // SAFETY: the cursor was just moved to `ptr + new_size`, so `old_size..new_size` lies
            // inside the current block and belongs to this block.
            unsafe { ptr.add(old_size).write_bytes(0, diff) };
            return Ok(NonNull::slice_from_raw_parts(ptr, new_size));
        }

        let new = self.allocate(new_layout)?;
        // SAFETY: `new` is a fresh block, disjoint from `ptr`, and `new_size >= old_size`.
        // `new` is exactly `new_size` long, so `old_size..new_size` is inside it.
        unsafe {
            ptr.copy_to_nonoverlapping(new.cast(), old_size);
            new.cast::<u8>().add(old_size).write_bytes(0, diff);
        }
        Ok(new)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let (new_size, old_size) = (new_layout.size(), old_layout.size());

        if self.try_resize_last(ptr, old_layout, new_layout) {
            return Ok(NonNull::slice_from_raw_parts(ptr, new_size));
        }

        if is_aligned(ptr, new_layout.align()) {
            return Ok(NonNull::slice_from_raw_parts(ptr, old_size));
        }

        let new = self.allocate(new_layout)?;
        // SAFETY: `new` is a fresh block, disjoint from `ptr`, and `new_size <= old_size`, so
        // both ranges are valid for `new_size` bytes.
        unsafe { ptr.copy_to_nonoverlapping(new.cast(), new_size) };
        Ok(new)
    }
}

impl<A: Allocator> Drop for Arena<A> {
    fn drop(&mut self) {
        let block = *self.block.get_mut();
        // SAFETY: the arena is being dropped, so no block is used after this.
        unsafe { self.free_blocks(block) };
    }
}

#[cfg(test)]
mod tests {
    use crate::{Leaky, debug::Tracking};

    use super::*;

    #[test]
    fn reset() {
        let mut arena = Arena::with_capacity(16);
        arena.reset();

        {
            let mut vec = Vec::new_in(&arena);
            vec.extend(0..1000_usize);
            let boxed = Box::new_in([7_u8; 100], &arena);
            assert_eq!(vec.iter().sum::<usize>(), 499_500);
            assert_eq!(boxed[99], 7);
        }

        let newest = arena.block.get();
        arena.reset();

        // Only the newest block is kept, emptied.
        assert_eq!(arena.block.get(), newest);
        assert!(is_empty_block(unsafe { newest.as_ref() }.prev));
        assert_eq!(arena.cursor.get(), HEADER_SIZE);

        let boxed = Box::new_in(42_usize, &arena);
        assert_eq!(*boxed, 42);
    }

    #[test]
    fn reset_empty() {
        let mut arena = Arena::new();
        arena.reset();
        assert!(is_empty_block(arena.block.get()));
    }

    #[test]
    fn capacity() {
        let arena = Arena::new();
        assert_eq!(arena.remaining_capacity(), 0);
        assert_eq!(arena.allocated_bytes(), 0);

        let arena = Arena::with_capacity(100);
        assert!(arena.remaining_capacity() >= 100);
        let first = arena.allocated_bytes();
        assert_eq!(first, HEADER_SIZE + arena.remaining_capacity());

        let block = arena.allocate(Layout::new::<[u8; 40]>()).unwrap();
        assert!(arena.remaining_capacity() <= first - HEADER_SIZE - 40);
        unsafe { arena.deallocate(block.cast(), Layout::new::<[u8; 40]>()) };
        assert_eq!(arena.remaining_capacity(), first - HEADER_SIZE);

        // Doesn't fit, so a second block is added.
        let _block = arena.allocate(Layout::new::<[u8; 200]>()).unwrap();
        assert!(arena.allocated_bytes() >= first + HEADER_SIZE + DEFAULT_INITIAL_CAPACITY);
    }

    #[test]
    fn growth() {
        let arena = Arena::new();
        arena.allocate(Layout::new::<u8>()).unwrap();
        let first = arena.allocated_bytes();
        assert!(first >= HEADER_SIZE + DEFAULT_INITIAL_CAPACITY);

        // Overflowing the first block adds one four times as big.
        let rest = Layout::from_size_align(arena.remaining_capacity() + 1, 1).unwrap();
        arena.allocate(rest).unwrap();
        assert!(arena.allocated_bytes() >= first + 4 * (first - HEADER_SIZE));
    }

    #[test]
    fn huge() {
        let arena = Arena::new();
        let layout = Layout::new::<[u8; 1 << 20]>();
        arena.allocate(layout).unwrap();
        assert!(arena.allocated_bytes() >= HEADER_SIZE + layout.size());
    }

    #[test]
    fn stable() {
        let arena = Arena::with_capacity(16);
        let first = Box::new_in(7_u64, &arena);
        let addr = &raw const *first;
        let _many = (0..1000_u64)
            .map(|i| Box::new_in(i, &arena))
            .collect::<Vec<_>>();
        assert_eq!(&raw const *first, addr);
        assert_eq!(*first, 7);
    }

    #[test]
    fn no_leak() {
        let tracking = Tracking::new(Global);
        let mut arena = Arena::with_capacity_in(16, &tracking);
        for _ in 0..3 {
            let mut vec = Vec::new_in(&arena);
            vec.extend(0..1000_u64);
            drop(vec);
            arena.reset();
        }
        drop(arena);
        assert!(tracking.finish().is_empty());
    }

    #[test]
    fn try_capacity() {
        assert!(Arena::try_with_capacity_in(16, Leaky::new()).is_err());
        assert!(Arena::try_with_capacity_in(usize::MAX, Global).is_err());

        // Lazy, so only the first allocation fails.
        let arena = Arena::new_in(Leaky::new());
        assert!(arena.allocate(Layout::new::<u8>()).is_err());
    }
}
