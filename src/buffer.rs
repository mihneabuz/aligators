use std::{
    alloc::{AllocError, Allocator, Layout},
    cell::Cell,
    marker::PhantomData,
    ptr::{self, NonNull},
};

// We use the most significant bit of the len field to keep track of when the buffer is in use.
const USED_BIT: usize = 1 << (usize::BITS - 1);
const LEN_MASK: usize = !USED_BIT;

/// Allocator backed by a buffer which provides a single allocation.
///
/// This allocator can only provide a single non-zero-sized allocation at a time. Trying to allocate
/// again without freeing the first block will result in an [`AllocError`], however it is possible
/// to resize (grow/shrink) the initial allocation. This can be useful for placing a single growable
/// collection (like a [`Vec`]) on the stack.
///
/// ```
/// use aligators::Buffer;
///
/// let mut buf = [0u8; 128];
/// let mut vec = Vec::new_in(Buffer::new(&mut buf));
/// vec.push(10);
/// vec.push(20);
/// ```
///
/// If you need to allocate multiple blocks at the same time, consider using a more sophisticated
/// allocator like a [`Bump`](crate::Bump) which allows for multiple allocations and can also be
/// backed by a stack buffer.
#[derive(Debug)]
pub struct Buffer<'a> {
    buf: NonNull<u8>,
    len: Cell<usize>,
    _marker: PhantomData<&'a mut [u8]>,
}

impl<'a> Buffer<'a> {
    /// Creates a new allocator backed by `buf`.
    ///
    /// # Panics
    ///
    /// If the size of `buf` is equal to or bigger than `1 << (usize::BITS - 1)`.
    pub fn new(buf: &'a mut [u8]) -> Self {
        assert!(buf.len() < USED_BIT, "Buffer is too big");

        Self {
            buf: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: Cell::new(buf.len()),
            _marker: PhantomData,
        }
    }

    /// Returns the total size of the buffer which backs this allocator.
    pub fn size(&self) -> usize {
        self.len.get() & LEN_MASK
    }

    /// Returns true while a non-zero-sized block is allocated.
    pub fn is_used(&self) -> bool {
        self.len.get() & USED_BIT > 0
    }

    fn set_used(&self, used: bool) {
        self.len.set(if used {
            self.len.get() | USED_BIT
        } else {
            self.len.get() & LEN_MASK
        });
    }

    fn align(&self, layout: Layout) -> Option<NonNull<[u8]>> {
        let (base, len) = (self.buf.cast::<u8>(), self.size());
        let offset = base.align_offset(layout.align());
        if offset + layout.size() > len {
            return None;
        }
        // SAFETY: offset is in `0..len` so the slice fits in `buf`
        Some(unsafe { NonNull::slice_from_raw_parts(base.byte_add(offset), len - offset) })
    }

    fn alloc(&self, layout: Layout, zero: bool) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0));
        }

        if self.is_used() {
            return Err(AllocError);
        }

        let aligned = self.align(layout).ok_or(AllocError)?;

        self.set_used(true);

        if zero {
            // SAFETY: `aligned` is a valid slice of the buffer
            unsafe {
                aligned.cast::<u8>().write_bytes(0, aligned.len());
            }
        }

        Ok(aligned)
    }

    /// # Safety
    ///
    /// Same as [`Allocator::grow`] / [`Allocator::shrink`], for a block returned by this allocator.
    unsafe fn resize(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
        zero: bool,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if old.size() == 0 {
            return self.alloc(new, zero);
        }

        if new.size() == 0 {
            self.set_used(false);
            return Ok(NonNull::slice_from_raw_parts(new.dangling_ptr(), 0));
        }

        // SAFETY: a non-empty block always lies within the buffer, at or after its start.
        let offset = unsafe { ptr.offset_from_unsigned(self.buf.cast::<u8>()) };
        let in_place =
            ptr.addr().get().is_multiple_of(new.align()) && self.size() - offset >= new.size();

        let block = if in_place {
            NonNull::slice_from_raw_parts(ptr, self.size() - offset)
        } else {
            let block = self.align(new).ok_or(AllocError)?;
            // SAFETY: both ranges are inside the buffer; `copy` handles the overlap.
            unsafe {
                ptr::copy(
                    ptr.as_ptr(),
                    block.cast::<u8>().as_ptr(),
                    old.size().min(new.size()),
                );
            };
            block
        };

        if zero && block.len() > old.size() {
            // SAFETY: `old.size()..block.len()` is a valid slice of `block`
            unsafe {
                block
                    .cast::<u8>()
                    .add(old.size())
                    .write_bytes(0, block.len() - old.size());
            }
        }

        Ok(block)
    }
}

unsafe impl Allocator for Buffer<'_> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc(layout, false)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc(layout, true)
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, layout: Layout) {
        if layout.size() > 0 {
            self.set_used(false);
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded from the caller.
        unsafe { self.resize(ptr, old_layout, new_layout, false) }
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded from the caller.
        unsafe { self.resize(ptr, old_layout, new_layout, true) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded from the caller.
        unsafe { self.resize(ptr, old_layout, new_layout, false) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single() {
        let mut buf = [0u8; 64];
        let buffer = Buffer::new(&mut buf);
        let layout = Layout::new::<u64>();

        let block = buffer.allocate(layout).unwrap();
        assert!(buffer.is_used());
        assert!(buffer.allocate(layout).is_err());
        // Zero-sized blocks don't need the buffer.
        assert!(buffer.allocate(Layout::new::<()>()).is_ok());

        unsafe { buffer.deallocate(block.cast(), layout) };
        assert!(!buffer.is_used());
        assert!(buffer.allocate(layout).is_ok());
    }

    #[test]
    fn full() {
        let mut buf = [0u8; 16];
        let buffer = Buffer::new(&mut buf);
        assert_eq!(buffer.size(), 16);
        assert!(buffer.allocate(Layout::new::<[u8; 32]>()).is_err());
        assert!(!buffer.is_used());
    }
}
