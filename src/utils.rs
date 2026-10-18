use std::{
    alloc::{AllocError, Allocator, Layout},
    marker::PhantomData,
    ptr::NonNull,
};

/// Allocator which never allocates and never frees anything.
///
/// Every allocation fails with [`AllocError`], and deallocating does nothing, so any block passed
/// to it is leaked. It is used as the backing allocator for memory that is only borrowed, like the
/// buffer of [`Bump::with_buffer`](crate::Bump::with_buffer): the lifetime `'a` ties it to that
/// borrow, and dropping the owner frees nothing.
///
/// ```
/// use std::alloc::{Allocator, Layout};
/// use aligators::Leaky;
///
/// let leaky = Leaky::new();
/// assert!(leaky.allocate(Layout::new::<u64>()).is_err());
/// ```
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
    /// Creates a new `Leaky` allocator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

// SAFETY: it never hands out a block, so there are no blocks to keep valid.
unsafe impl Allocator for Leaky<'_> {
    fn allocate(&self, _layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError)
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}
