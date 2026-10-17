use std::{
    alloc::{AllocError, Allocator, Layout},
    marker::PhantomData,
    ptr::NonNull,
};

#[derive(Clone, Copy, Debug)]
pub struct Leaky<'a> {
    _phantom: PhantomData<&'a mut [u8]>,
}

impl Default for Leaky<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl Leaky<'_> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

unsafe impl Allocator for Leaky<'_> {
    fn allocate(&self, _layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}
