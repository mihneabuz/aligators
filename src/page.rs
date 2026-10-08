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

    fn check_alloc(alloc: impl Allocator, layout: Layout, granularity: usize) {
        let block = alloc.allocate(layout).expect("allocation failed");
        let ptr = block.cast::<u8>();

        assert!(block.len() >= layout.size());
        assert_eq!(block.len() % granularity, 0);
        assert!(is_aligned(ptr.as_ptr(), layout.align()));

        let bytes = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), block.len()) };
        for byte in bytes.iter_mut() {
            assert_eq!(*byte, 0);
            *byte = 0xAB;
        }

        unsafe { alloc.deallocate(ptr, layout) };
    }

    #[test]
    fn page_sizes() {
        let ps = page_size();
        for size in [1, 7, ps - 1, ps, ps + 1, 3 * ps, 10 * ps + 123] {
            check_alloc(PageAllocator, Layout::from_size_align(size, 1).unwrap(), ps);
        }
    }

    #[test]
    fn page_alignments() {
        let ps = page_size();
        let mut align = 1;
        while align <= ps {
            check_alloc(
                PageAllocator,
                Layout::from_size_align(100, align).unwrap(),
                ps,
            );
            align *= 2;
        }
    }

    #[test]
    fn page_align_too_large() {
        let layout = Layout::from_size_align(16, page_size() * 2).unwrap();
        assert!(PageAllocator.allocate(layout).is_err());
    }

    #[test]
    fn page_zero_size() {
        for align in [1, 2, 8, 64, 4096] {
            let layout = Layout::from_size_align(0, align).unwrap();
            let block = PageAllocator.allocate(layout).unwrap();

            assert_eq!(block.len(), 0);
            assert!(is_aligned(block.cast::<u8>().as_ptr(), align));

            unsafe { PageAllocator.deallocate(block.cast(), layout) };
        }
    }

    #[test]
    fn page_dealloc_with_reported_size() {
        let layout = Layout::from_size_align(10, 1).unwrap();
        let block = PageAllocator.allocate(layout).unwrap();
        let fitted = Layout::from_size_align(block.len(), 1).unwrap();

        unsafe { PageAllocator.deallocate(block.cast(), fitted) };
    }

    #[test]
    fn page_vec() {
        let mut v = Vec::new_in(PageAllocator);

        v.push(1);
        v.push(2);
        v.push(3);

        assert_eq!(&v[0..3], &[1, 2, 3]);
    }

    #[test]
    fn page_vec_grow_and_shrink() {
        let mut v = Vec::new_in(PageAllocator);
        for i in 0..100_000u64 {
            v.push(i);
        }
        assert!(v.iter().copied().eq(0..100_000));

        v.truncate(10);
        v.shrink_to_fit();
        assert_eq!(&v[..], &(0..10).collect::<Vec<_>>()[..]);
    }

    #[test]
    fn page_box() {
        let b = Box::new_in([7u8; 10_000], PageAllocator);
        assert!(b.iter().all(|&x| x == 7));

        #[repr(align(256))]
        struct Aligned(u8);
        let b = Box::new_in(Aligned(1), PageAllocator);
        assert!((&raw const *b).is_aligned());
        assert_eq!(b.0, 1);
    }

    #[test]
    fn page_zst() {
        let mut v = Vec::new_in(PageAllocator);
        for _ in 0..1000 {
            v.push(());
        }
        assert_eq!(v.len(), 1000);
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
    fn huge_2mb() {
        check_huge::<HUGE_PAGE_2MB>();
    }

    #[test]
    fn huge_1gb() {
        check_huge::<HUGE_PAGE_1GB>();
    }

    #[test]
    fn huge_zero_size() {
        let layout = Layout::from_size_align(0, 64).unwrap();
        let block = HugePageAllocator2MB::default().allocate(layout).unwrap();

        assert_eq!(block.len(), 0);
        assert!(is_aligned(block.cast::<u8>().as_ptr(), 64));

        unsafe { HugePageAllocator2MB::default().deallocate(block.cast(), layout) };
    }

    fn layout(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).unwrap()
    }

    fn bytes<'a>(block: NonNull<[u8]>, len: usize) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(block.cast::<u8>().as_ptr(), len) }
    }

    /// Exercises `allocate_zeroed`, `grow`, `grow_zeroed` and `shrink` with sizes expressed in
    /// multiples of the allocator's `unit`. Needs at most 3 units mapped at once.
    fn check_resize(alloc: impl Allocator + Copy, unit: usize) {
        unsafe {
            // allocate_zeroed: whole rounded block is zero.
            let block = alloc.allocate_zeroed(layout(10, 1)).unwrap();
            assert_eq!(block.len(), unit);
            assert!(bytes(block, unit).iter().all(|&b| b == 0));

            // Growing within the same unit keeps the pointer and doesn't remap.
            bytes(block, 10).fill(0x5A);
            let same = alloc
                .grow(block.cast(), layout(10, 1), layout(unit - 1, 1))
                .unwrap();
            assert_eq!(same.cast::<u8>(), block.cast::<u8>());
            assert_eq!(same.len(), unit);

            // Growing across units preserves the data and reports the rounded length.
            bytes(same, unit - 1).fill(0x5A);
            let grown = alloc
                .grow(same.cast(), layout(unit - 1, 1), layout(unit + 1, 1))
                .unwrap();
            assert_eq!(grown.len(), 2 * unit);
            assert!(bytes(grown, unit - 1).iter().all(|&b| b == 0x5A));

            // Shrinking within the same unit count is a no-op.
            let same = alloc
                .shrink(grown.cast(), layout(unit + 1, 1), layout(unit + 1, 1))
                .unwrap();
            assert_eq!(same.cast::<u8>(), grown.cast::<u8>());

            // Shrinking across units keeps the pointer and the data.
            let shrunk = alloc
                .shrink(same.cast(), layout(unit + 1, 1), layout(100, 1))
                .unwrap();
            assert_eq!(shrunk.cast::<u8>(), grown.cast::<u8>());
            assert_eq!(shrunk.len(), unit);
            assert!(bytes(shrunk, 100).iter().all(|&b| b == 0x5A));

            // Over-aligned requests fail and leave the block intact.
            assert!(
                alloc
                    .grow(shrunk.cast(), layout(100, 1), layout(200, 2 * unit))
                    .is_err()
            );
            assert!(
                alloc
                    .shrink(shrunk.cast(), layout(100, 1), layout(50, 2 * unit))
                    .is_err()
            );
            assert!(bytes(shrunk, 100).iter().all(|&b| b == 0x5A));

            // Shrinking to zero frees the block and returns an aligned dangling pointer.
            let empty = alloc
                .shrink(shrunk.cast(), layout(100, 1), layout(0, 64))
                .unwrap();
            assert_eq!(empty.len(), 0);
            assert!(is_aligned(empty.cast::<u8>().as_ptr(), 64));

            // Growing from zero is a fresh allocation.
            let fresh = alloc
                .grow(empty.cast(), layout(0, 64), layout(50, 64))
                .unwrap();
            assert_eq!(fresh.len(), unit);
            assert!(is_aligned(fresh.cast::<u8>().as_ptr(), 64));
            alloc.deallocate(fresh.cast(), layout(50, 64));

            let fresh = alloc
                .grow_zeroed(empty.cast(), layout(0, 64), layout(50, 64))
                .unwrap();
            assert!(bytes(fresh, unit).iter().all(|&b| b == 0));
            alloc.deallocate(fresh.cast(), layout(50, 64));
        }
    }

    /// `grow_zeroed` must clear the slack past the old size, which callers may have written to
    /// since we report the rounded length.
    fn check_grow_zeroed_slack(alloc: impl Allocator + Copy, unit: usize) {
        for new_size in [100, unit, unit + 1] {
            unsafe {
                let block = alloc.allocate(layout(10, 1)).unwrap();
                bytes(block, block.len()).fill(0xFF);

                let grown = alloc
                    .grow_zeroed(block.cast(), layout(10, 1), layout(new_size, 1))
                    .unwrap();
                let grown_bytes = bytes(grown, grown.len());
                assert!(grown_bytes[..10].iter().all(|&b| b == 0xFF));
                assert!(
                    grown_bytes[10..].iter().all(|&b| b == 0),
                    "dirty slack growing to {new_size}"
                );

                alloc.deallocate(grown.cast(), layout(new_size, 1));
            }
        }
    }

    #[test]
    fn page_resize() {
        check_resize(PageAllocator, page_size());
    }

    #[test]
    fn page_grow_zeroed_slack() {
        check_grow_zeroed_slack(PageAllocator, page_size());
    }

    #[test]
    fn page_grow_large() {
        // Large enough that a copying grow would be noticeably slow; `mremap` just moves page
        // table entries.
        let mut v = Vec::with_capacity_in(1 << 20, PageAllocator);
        v.extend(0..1u64 << 20);
        v.reserve(1 << 24);
        assert!(v.iter().copied().eq(0..1 << 20));
    }

    /// Runs the resize suites if enough huge pages are reserved, otherwise skips.
    fn check_huge_resize<const SHIFT: u32>() {
        let alloc = HugePageAllocator::<SHIFT>;
        let unit = HugePageAllocator::<SHIFT>::SIZE;

        match alloc.allocate(layout(3 * unit, 1)) {
            Ok(probe) => unsafe { alloc.deallocate(probe.cast(), layout(3 * unit, 1)) },
            Err(AllocError) => {
                eprintln!("fewer than 3 {unit}-byte huge pages available, skipping");
                return;
            }
        }

        check_resize(alloc, unit);
        check_grow_zeroed_slack(alloc, unit);
    }

    #[test]
    fn huge_2mb_resize() {
        check_huge_resize::<HUGE_PAGE_2MB>();
    }

    #[test]
    fn huge_1gb_resize() {
        check_huge_resize::<HUGE_PAGE_1GB>();
    }
}
