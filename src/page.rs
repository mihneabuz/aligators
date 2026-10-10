use std::{
    alloc::{AllocError, Allocator, Layout},
    ptr::{self, NonNull},
    sync::OnceLock,
};

fn page_size() -> usize {
    static PAGE_SIZE: OnceLock<usize> = OnceLock::new();
    // SAFETY: sysconf has no preconditions
    *PAGE_SIZE.get_or_init(|| unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize })
}

fn mmap_alloc(size: usize, flags: i32) -> Option<NonNull<[u8]>> {
    // SAFETY: the mapping is private and anonymous
    let ptr = unsafe {
        libc::mmap(
            ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | flags,
            -1,
            0,
        )
    };

    if ptr == libc::MAP_FAILED {
        return None;
    }

    NonNull::new(ptr as *mut u8).map(|ptr| NonNull::slice_from_raw_parts(ptr, size))
}

/// # Safety
///
/// `ptr` must have been returned by `mmap_alloc` with the same `size` as `old`, and it must not
/// be used to access memory after this call. `new` must be > 0.
unsafe fn mmap_resize(
    ptr: NonNull<u8>,
    old: usize,
    new: usize,
    flags: i32,
) -> Option<NonNull<[u8]>> {
    // SAFETY: the caller guarantees `ptr..ptr + old` is exactly one mapping we created.
    let ptr = unsafe { libc::mremap(ptr.as_ptr() as *mut _, old, new, flags) };

    if ptr == libc::MAP_FAILED {
        return None;
    }

    NonNull::new(ptr as *mut u8).map(|ptr| NonNull::slice_from_raw_parts(ptr, new))
}

/// # Safety
///
/// `ptr` must have been returned by `mmap_alloc` with the same `size`, and it must not be used
/// to access memory after this call.
unsafe fn mmap_free(ptr: NonNull<u8>, size: usize) {
    if size > 0 {
        // SAFETY: the caller guarantees ptr..ptr + size is exactly one mapping we created.
        unsafe { libc::munmap(ptr.as_ptr() as *mut _, size) };
    }
}

#[derive(Clone, Copy)]
struct Mapping {
    unit: usize,
    flags: i32,
}

impl Mapping {
    fn round(&self, size: usize) -> usize {
        size.next_multiple_of(self.unit)
    }

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0));
        }

        if layout.align() > self.unit {
            return Err(AllocError);
        }

        mmap_alloc(self.round(layout.size()), self.flags).ok_or(AllocError)
    }

    /// # Safety
    ///
    /// Same as [`Allocator::deallocate`], for a block returned by this mapping.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the `Allocator` contract guarantees `ptr` comes from a valid allocation.
        unsafe { mmap_free(ptr, self.round(layout.size())) }
    }

    /// # Safety
    ///
    /// Same as [`Allocator::grow`] / [`Allocator::shrink`], for a block returned by this mapping.
    ///
    /// When manual is true, skip trying the `mremap` and do a fresh `mmap`.
    /// Also returns wether the new allocation is a fresh `mmap`.
    unsafe fn resize_inner(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
        manual: bool,
    ) -> Result<(NonNull<[u8]>, bool), AllocError> {
        if new.align() > self.unit {
            return Err(AllocError);
        }

        if old.size() == 0 {
            return self.allocate(new).map(|ptr| (ptr, true));
        }

        if new.size() == 0 {
            // SAFETY: the `Allocator` contract guarantees `ptr` comes from a valid allocation.
            unsafe { self.deallocate(ptr, old) };
            return Ok((NonNull::slice_from_raw_parts(new.dangling_ptr(), 0), true));
        }

        let (old_round, new_round) = (self.round(old.size()), self.round(new.size()));
        if old_round == new_round {
            return Ok((NonNull::slice_from_raw_parts(ptr, new_round), false));
        }

        if !manual {
            // SAFETY: the `Allocator` contract guarantees `ptr` comes from a valid allocation.
            if let Some(ptr) =
                unsafe { mmap_resize(ptr, old_round, new_round, libc::MREMAP_MAYMOVE) }
            {
                return Ok((ptr, false));
            }
        }

        let block = self.allocate(new)?;
        // SAFETY: `ptr` and `block` come from different allocations and the size is within bounds.
        unsafe {
            ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                block.cast().as_ptr(),
                old.size().min(new.size()),
            );
            self.deallocate(ptr, old);
        }

        Ok((block, true))
    }

    /// # Safety
    ///
    /// Same as [`Allocator::grow`] / [`Allocator::shrink`], for a block returned by this mapping.
    unsafe fn resize(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
        manual: bool,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded from the caller.
        unsafe { self.resize_inner(ptr, old, new, manual).map(|(ptr, _)| ptr) }
    }

    /// # Safety
    ///
    /// Same as [`Allocator::grow_zeroed`], for a block returned by this mapping.
    ///
    /// When manual is true, skip trying the `mremap` and do a fresh `mmap`.
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
        manual: bool,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded from the caller.
        let (block, fresh) = unsafe { self.resize_inner(ptr, old, new, manual)? };

        if !fresh {
            // In case callers wrote past the original layout.size()
            // SAFETY: `old.size()..round(old.size())` is inside the new block
            unsafe {
                block
                    .cast::<u8>()
                    .add(old.size())
                    .write_bytes(0, self.round(old.size()) - old.size())
            };
        }

        Ok(block)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PageAllocator;

impl PageAllocator {
    fn mapping() -> Mapping {
        Mapping {
            unit: page_size(),
            flags: 0,
        }
    }
}

unsafe impl Allocator for PageAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Self::mapping().allocate(layout)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Self::mapping().allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded from the caller.
        unsafe { Self::mapping().deallocate(ptr, layout) }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded from the caller.
        unsafe { Self::mapping().resize(ptr, old_layout, new_layout, false) }
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded from the caller.
        unsafe { Self::mapping().grow_zeroed(ptr, old_layout, new_layout, false) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded from the caller.
        unsafe { Self::mapping().resize(ptr, old_layout, new_layout, false) }
    }
}

pub const HUGE_PAGE_2MB: u32 = 21;
pub const HUGE_PAGE_1GB: u32 = 30;

pub type HugePageAllocator2MB = HugePageAllocator<HUGE_PAGE_2MB>;
pub type HugePageAllocator1GB = HugePageAllocator<HUGE_PAGE_1GB>;

#[derive(Clone, Copy, Debug, Default)]
pub struct HugePageAllocator<const SHIFT: u32 = HUGE_PAGE_2MB>;

impl<const SHIFT: u32> HugePageAllocator<SHIFT> {
    const SIZE: usize = 1 << SHIFT;
    const MAPPING: Mapping = Mapping {
        unit: Self::SIZE,
        flags: libc::MAP_HUGETLB | ((SHIFT as i32) << libc::MAP_HUGE_SHIFT),
    };
}

unsafe impl<const SHIFT: u32> Allocator for HugePageAllocator<SHIFT> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Self::MAPPING.allocate(layout)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Self::MAPPING.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded from the caller.
        unsafe { Self::MAPPING.deallocate(ptr, layout) }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded from the caller.
        unsafe { Self::MAPPING.resize(ptr, old_layout, new_layout, true) }
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded from the caller.
        unsafe { Self::MAPPING.grow_zeroed(ptr, old_layout, new_layout, true) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: forwarded from the caller.
        unsafe { Self::MAPPING.resize(ptr, old_layout, new_layout, false) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_aligned<T>(ptr: *const T, align: usize) -> bool {
        ptr.addr().is_multiple_of(align)
    }

    fn layout(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).unwrap()
    }

    fn bytes<'a>(block: NonNull<[u8]>, len: usize) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(block.cast::<u8>().as_ptr(), len) }
    }

    fn check_alloc(alloc: impl Allocator, layout: Layout, granularity: usize) {
        let block = alloc.allocate(layout).expect("allocation failed");
        let ptr = block.cast::<u8>();

        assert!(block.len() >= layout.size());
        assert_eq!(block.len() % granularity, 0);
        assert!(is_aligned(ptr.as_ptr(), layout.align()));

        for byte in bytes(block, block.len()).iter_mut() {
            assert_eq!(*byte, 0);
            *byte = 0xAB;
        }

        unsafe { alloc.deallocate(ptr, layout) };
    }

    #[test]
    fn allocate() {
        let ps = page_size();
        for size in [1, 7, ps - 1, ps, ps + 1, 3 * ps, 10 * ps + 123] {
            check_alloc(PageAllocator, layout(size, 1), ps);
        }
    }

    #[test]
    fn alignment() {
        let ps = page_size();
        let mut align = 1;
        while align <= ps {
            check_alloc(PageAllocator, layout(100, align), ps);
            align *= 2;
        }

        assert!(PageAllocator.allocate(layout(16, page_size() * 2)).is_err());
    }

    #[test]
    fn zero_size() {
        for align in [1, 2, 8, 64, 4096] {
            let block = PageAllocator.allocate(layout(0, align)).unwrap();

            assert_eq!(block.len(), 0);
            assert!(is_aligned(block.cast::<u8>().as_ptr(), align));

            unsafe { PageAllocator.deallocate(block.cast(), layout(0, align)) };
        }
    }

    #[test]
    fn deallocate() {
        let block = PageAllocator.allocate(layout(10, 1)).unwrap();
        unsafe { PageAllocator.deallocate(block.cast(), layout(10, 1)) };

        let block = PageAllocator.allocate(layout(10, 1)).unwrap();
        unsafe { PageAllocator.deallocate(block.cast(), layout(block.len(), 1)) };
    }

    #[test]
    fn grow_in_place() {
        let block = PageAllocator.allocate_zeroed(layout(10, 1)).unwrap();
        assert!(bytes(block, block.len()).iter().all(|&b| b == 0));
        bytes(block, 10).fill(0x5A);

        let same = unsafe {
            PageAllocator
                .grow(block.cast(), layout(10, 1), layout(100, 1))
                .unwrap()
        };
        assert_eq!(same.cast::<u8>(), block.cast::<u8>());
        assert_eq!(bytes(same, 10), &[0x5A; 10]);

        unsafe { PageAllocator.deallocate(same.cast(), layout(100, 1)) };
    }

    #[test]
    fn grow_zeroed() {
        let block = PageAllocator.allocate(layout(10, 1)).unwrap();
        bytes(block, block.len()).fill(0xFF);

        let grown = unsafe {
            PageAllocator
                .grow_zeroed(block.cast(), layout(10, 1), layout(page_size() + 1, 1))
                .unwrap()
        };
        let grown_bytes = bytes(grown, grown.len());
        assert_eq!(&grown_bytes[0..10], &[0xFFu8; 10]);
        assert_eq!(&grown_bytes[10..], &vec![0u8; grown_bytes.len() - 10]);

        unsafe { PageAllocator.deallocate(grown.cast(), layout(page_size() + 1, 1)) };
    }

    #[test]
    fn grow_from_empty() {
        let block = PageAllocator.allocate(layout(0, 1)).unwrap();
        assert_eq!(block.len(), 0);

        let grown = unsafe {
            PageAllocator
                .grow(block.cast(), layout(0, 1), layout(100, 1))
                .unwrap()
        };
        bytes(grown, 100).fill(0x5A);

        unsafe { PageAllocator.deallocate(grown.cast(), layout(100, 1)) };
    }

    #[test]
    fn shrink_in_place() {
        let block = PageAllocator.allocate_zeroed(layout(100, 1)).unwrap();
        assert!(bytes(block, block.len()).iter().all(|&b| b == 0));
        bytes(block, 100).fill(0x5A);

        let same = unsafe {
            PageAllocator
                .shrink(block.cast(), layout(100, 1), layout(10, 1))
                .unwrap()
        };
        assert_eq!(same.cast::<u8>(), block.cast::<u8>());
        assert_eq!(bytes(same, 10), &[0x5A; 10]);

        unsafe { PageAllocator.deallocate(same.cast(), layout(10, 1)) };
    }

    #[test]
    fn shrink_to_empty() {
        let block = PageAllocator.allocate(layout(100, 1)).unwrap();
        bytes(block, 100).fill(0x5A);

        let empty = unsafe {
            PageAllocator
                .shrink(block.cast(), layout(100, 1), layout(0, 1))
                .unwrap()
        };

        unsafe { PageAllocator.deallocate(empty.cast(), layout(0, 1)) };
    }

    #[test]
    fn boxed() {
        let b = Box::new_in([7u8; 10_000], PageAllocator);
        assert!(b.iter().all(|&x| x == 7));

        #[repr(align(256))]
        struct Aligned(u8);
        let b = Box::new_in(Aligned(1), PageAllocator);
        assert!((&raw const *b).is_aligned());
        assert_eq!(b.0, 1);
    }

    #[test]
    fn vec() {
        let mut v = Vec::new_in(PageAllocator);
        for i in 0..10_000u64 {
            v.push(i);
        }
        assert!(v.iter().copied().eq(0..10_000));

        v.truncate(10);
        v.shrink_to_fit();
        assert_eq!(&v[..], &(0..10).collect::<Vec<_>>()[..]);
    }

    #[test]
    fn vec_zst() {
        let mut v = Vec::new_in(PageAllocator);
        for _ in 0..1000 {
            v.push(());
        }
        assert_eq!(v.len(), 1000);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn vec_large() {
        let mut v = Vec::with_capacity_in(1 << 20, PageAllocator);
        v.extend(0..1u64 << 20);
        v.reserve(1 << 24);
        assert!(v.iter().copied().eq(0..1 << 20));
    }

    fn check_huge<const SHIFT: u32>() {
        let alloc = HugePageAllocator::<SHIFT>;
        let size = HugePageAllocator::<SHIFT>::SIZE;
        let layout = Layout::from_size_align(100, 8).unwrap();

        match alloc.allocate(layout) {
            Ok(block) => {
                assert_eq!(block.len(), size);
                assert!(is_aligned(block.cast::<u8>().as_ptr(), size));
                unsafe { alloc.deallocate(block.cast(), layout) };

                check_alloc(alloc, Layout::from_size_align(size + 1, 1).unwrap(), size);
            }
            Err(AllocError) => eprintln!("no {size}-byte huge pages available, skipping"),
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn huge() {
        check_huge::<HUGE_PAGE_2MB>();
        check_huge::<HUGE_PAGE_1GB>();
    }
}
