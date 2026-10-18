use std::{
    alloc::{AllocError, Allocator, Global, Layout, handle_alloc_error},
    cell::Cell,
    ptr::NonNull,
};

const DEFAULT_INITIAL_CAPACITY: usize = 2048;

fn layout(size: usize) -> Option<Layout> {
    Layout::from_size_align(size, align_of::<BlockHeader>()).ok()
}

fn is_aligned(ptr: NonNull<u8>, align: usize) -> bool {
    ptr.addr().get().is_multiple_of(align)
}

pub struct BlockHeader {
    prev: Option<NonNull<u8>>,
    size: usize,
}

#[derive(Debug)]
pub struct ArenaAllocator<A: Allocator = Global> {
    block: NonNull<u8>,
    size: usize,
    cursor: Cell<usize>,
    backing: A,
}

impl ArenaAllocator {
    /// Create a new allocator with default initial capacity.
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    /// Create a new allocator with initial `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl Default for ArenaAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Allocator> ArenaAllocator<A> {
    /// Create a new allocator backed by the `backing` allocator.
    ///
    /// # Panics
    /// When the `backing` allocator returns `AllocError`.
    pub fn new_in(backing: A) -> Self {
        Self::with_capacity_in(DEFAULT_INITIAL_CAPACITY, backing)
    }

    /// Create a new allocator backed by the `backing` allocator and with initial `capacity`.
    ///
    /// # Panics
    /// When the `backing` allocator returns `AllocError`.
    pub fn with_capacity_in(capacity: usize, backing: A) -> Self {
        match Self::try_with_capacity_in(capacity, backing) {
            Ok(arena) => arena,
            Err(_) => match layout(capacity) {
                Some(layout) => handle_alloc_error(layout),
                None => panic!("capacity overflow"),
            },
        }
    }

    /// Create a new allocator backed by the `backing` allocator and with initial `capacity`.
    ///
    /// # Errors
    /// Returns `AllocError` when the `backing` allocator fails.
    pub fn try_with_capacity_in(capacity: usize, backing: A) -> Result<Self, AllocError> {
        let layout = layout(capacity).ok_or(AllocError)?;
        let block = backing.allocate(layout)?;

        let alloc = Self {
            block: block.cast(),
            size: block.len(),
            cursor: Cell::new(size_of::<BlockHeader>()),
            backing,
        };

        alloc.header().prev = None;
        alloc.header().size = alloc.size;

        Ok(alloc)
    }

    fn header(&self) -> &mut BlockHeader {
        unsafe { self.block.cast::<BlockHeader>().as_mut() }
    }

    fn new_block(&self) {
        todo!()
    }
}

impl<A: Allocator> Drop for ArenaAllocator<A> {
    fn drop(&mut self) {
        while let Some(prev) = self.header().prev {
            if let Some(layout) = layout(self.size) {
                unsafe { self.backing.deallocate(self.block, layout) }
            }

            self.block = prev;
            self.size = self.header().size;
        }

        if let Some(layout) = layout(self.header().size) {
            unsafe { self.backing.deallocate(self.block, layout) }
        }
    }
}
