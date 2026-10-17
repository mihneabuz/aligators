use std::{
    alloc::{AllocError, Allocator, Layout},
    cell::Cell,
    marker::PhantomData,
    ptr::{self, NonNull},
};

// Use the most significant bit of the len field to keep track of
// wether the buffer is currently in use.
const USED_BIT: usize = 1 << 63;
const LEN_MASK: usize = !USED_BIT;

#[derive(Debug)]
pub struct BufferAllocator<'a> {
    buf: NonNull<u8>,
    len: Cell<usize>,
    _marker: PhantomData<&'a mut [u8]>,
}

impl<'a> BufferAllocator<'a> {
    /// Create a new allocator backed by `buf`.
    ///
    /// # Panics
    /// When `buf` is bigger than `1 << 63`.
    pub fn new(buf: &'a mut [u8]) -> Self {
        assert!(buf.len() < USED_BIT, "Buffer is too big");

        Self {
            buf: NonNull::new(buf.as_mut_ptr()).unwrap(),
            len: Cell::new(buf.len()),
            _marker: PhantomData,
        }
    }

    fn len(&self) -> usize {
        self.len.get() & LEN_MASK
    }

    fn is_used(&self) -> bool {
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
        let (base, len) = (self.buf.cast::<u8>(), self.len());
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
    /// Same as [`Allocator::deallocate`], for a block returned by this mapping.
    unsafe fn dealloc(&self, layout: Layout) {
        if layout.size() > 0 {
            self.set_used(false);
        }
    }

    /// # Safety
    ///
    /// Same as [`Allocator::grow`] / [`Allocator::shrink`], for a block returned by this mapping.
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
            ptr.addr().get().is_multiple_of(new.align()) && self.len() - offset >= new.size();

        let block = if in_place {
            NonNull::slice_from_raw_parts(ptr, self.len() - offset)
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
                ptr.cast::<u8>()
                    .add(old.size())
                    .write_bytes(0, block.len() - old.size());
            }
        }

        Ok(block)
    }
}

unsafe impl Allocator for BufferAllocator<'_> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc(layout, false)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc(layout, true)
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded from the caller.
        unsafe { self.dealloc(layout) };
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
        unsafe { self.resize(ptr, old_layout, new_layout, true) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

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
        let alloc = BufferAllocator::new(&mut buf);

        let block = alloc.allocate(layout(16, 1)).unwrap();
        check_block(block, layout(16, 1), &range);
        assert_eq!(block.len(), 64);

        unsafe { alloc.deallocate(block.cast(), layout(16, 1)) };
    }

    #[test]
    fn allocate_too_large() {
        let mut buf = [0u8; 64];
        let alloc = BufferAllocator::new(&mut buf);

        assert!(alloc.allocate(layout(65, 1)).is_err());
        assert!(alloc.allocate(layout(64, 1)).is_ok());
    }

    #[test]
    fn alignment() {
        #[repr(align(128))]
        struct Aligned([u8; 256]);
        let mut storage = Aligned([0; 256]);

        let buf = &mut storage.0[1..];
        let range = buf.as_ptr_range();

        let alloc = BufferAllocator::new(buf);
        for align in [1, 2, 4, 8, 16, 32, 64, 128] {
            let block = alloc.allocate(layout(8, align)).unwrap();
            check_block(block, layout(8, align), &range);
            assert_eq!(block.len(), 255 - (align - 1));
            unsafe { alloc.deallocate(block.cast(), layout(8, align)) };
        }

        assert!(alloc.allocate(layout(1, 4096)).is_err());
    }

    #[test]
    fn empty_buffer() {
        let alloc = BufferAllocator::new(&mut []);

        assert!(alloc.allocate(layout(1, 1)).is_err());
        assert!(alloc.allocate(layout(1, 8)).is_err());
    }

    #[test]
    fn single_slot() {
        let mut buf = [0u8; 64];
        let alloc = BufferAllocator::new(&mut buf);

        let block = alloc.allocate(layout(8, 1)).unwrap();
        assert!(alloc.allocate(layout(8, 1)).is_err());

        unsafe { alloc.deallocate(block.cast(), layout(8, 1)) };
        let block = alloc.allocate(layout(8, 1)).unwrap();
        unsafe { alloc.deallocate(block.cast(), layout(8, 1)) };
    }

    #[test]
    fn zero_size() {
        let mut buf = [0u8; 64];
        let alloc = BufferAllocator::new(&mut buf);

        let empty = alloc.allocate(layout(0, 8)).unwrap();
        assert_eq!(empty.len(), 0);
        assert!(empty.cast::<u8>().as_ptr().addr().is_multiple_of(8));

        let block = alloc.allocate(layout(16, 1)).unwrap();

        unsafe { alloc.deallocate(empty.cast(), layout(0, 8)) };
        assert!(alloc.allocate(layout(16, 1)).is_err());

        unsafe { alloc.deallocate(block.cast(), layout(16, 1)) };
    }

    #[test]
    fn grow_in_place() {
        let mut buf = [0u8; 64];
        let range = buf.as_ptr_range();
        let alloc = BufferAllocator::new(&mut buf);

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
        let alloc = BufferAllocator::new(&mut buf);

        let block = alloc.allocate(layout(8, 1)).unwrap();
        bytes(block)[..8].copy_from_slice(b"aligator");

        assert!(unsafe { alloc.grow(block.cast(), layout(8, 1), layout(65, 1)) }.is_err());

        assert_eq!(&bytes(block)[..8], b"aligator");
        unsafe { alloc.deallocate(block.cast(), layout(8, 1)) };
    }

    #[test]
    fn grow_zeroed() {
        let mut buf = [0u8; 64];
        let alloc = BufferAllocator::new(&mut buf);

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
        let alloc = BufferAllocator::new(&mut buf);

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
        let alloc = BufferAllocator::new(&mut buf);

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
        let alloc = BufferAllocator::new(&mut buf);

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
    fn boxed() {
        let mut buf = [0u8; 64];
        let alloc = BufferAllocator::new(&mut buf);

        let mut b = Box::new_in(5u32, &alloc);
        *b += 1;
        assert_eq!(*b, 6);
        drop(b);

        let b = Box::new_in([1u64; 4], &alloc);
        assert_eq!(b.iter().sum::<u64>(), 4);
    }

    #[test]
    fn vec() {
        let mut buf = [0u8; 64];
        let alloc = BufferAllocator::new(&mut buf);
        let mut v = Vec::new_in(&alloc);

        for i in 0..64u8 {
            v.try_reserve(1).expect("grow within the buffer failed");
            v.push(i);
        }
        assert!(v.iter().copied().eq(0..64));

        assert!(v.try_reserve(1).is_err());
    }

    #[test]
    fn moving() {
        let mut buf = [0u8; 64];

        let v = {
            let mut v = Vec::with_capacity_in(8, BufferAllocator::new(&mut buf));
            v.extend(0..8u8);
            v
        };

        let moved = Box::new(v);
        assert!(moved.iter().copied().eq(0..8));
    }
}
