use std::{
    alloc::{AllocError, Allocator, Layout},
    ptr::{self, NonNull},
    sync::OnceLock,
};

mod sys {
    use std::ptr::{self, NonNull};

    pub fn get_page_size() -> usize {
        // SAFETY: sysconf(SC_PAGESIZE) is available on any Linux and has no preconditions.
        unsafe {
            usize::try_from(libc::sysconf(libc::_SC_PAGESIZE)).expect("Failed to get page size")
        }
    }

    pub fn mmap(size: usize, flags: i32) -> Option<NonNull<[u8]>> {
        // SAFETY: the mapping is private and anonymous, with no backing file.
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

        NonNull::new(ptr.cast()).map(|ptr| NonNull::slice_from_raw_parts(ptr, size))
    }

    /// # Safety
    ///
    /// `ptr` must have been returned by `mmap` with the same `size` as `old`, and it must
    /// not be used to access memory after this call. `new` must be > 0.
    pub unsafe fn mremap(ptr: NonNull<u8>, old: usize, new: usize) -> Option<NonNull<[u8]>> {
        // SAFETY: the caller guarantees `ptr..ptr + old` is exactly one mapping we created.
        let new_ptr = unsafe { libc::mremap(ptr.as_ptr().cast(), old, new, libc::MREMAP_MAYMOVE) };

        if new_ptr == libc::MAP_FAILED {
            return None;
        }

        NonNull::new(new_ptr.cast()).map(|ptr| NonNull::slice_from_raw_parts(ptr, new))
    }

    /// # Safety
    ///
    /// `ptr` must have been returned by `mmap` with the same `size`, and it must not be used
    /// to access memory after this call.
    pub unsafe fn munmap(ptr: NonNull<u8>, size: usize) {
        if size > 0 {
            // SAFETY: the caller guarantees ptr..ptr + size is exactly one mapping we created.
            unsafe { libc::munmap(ptr.as_ptr().cast(), size) };
        }
    }
}

/// Generic type used to allocate both normal and huge pages.
///
/// # Safety
///
/// New allocations use `mmap` with private, anonymous flags, which will always returns disjoint
/// pages of memory for the process. This memory is also guaranteed to be zeroed.
///
/// Deallocation is done with `munmap`, and resizing (grow/shrink) use the `mremap` if possible,
/// falling back to manual reallocation and copy if that fails. Both `munmap` and `mremap` must be
/// called with the same size as the initial `mmap` which returned the pointer.
///
/// The [`Allocator`] contract only guarantees that the size of the layour will be between the
/// size requested by the caller and the actual size returned by the allocator. We can get exact
/// size from this range by rounding up to the page size, which is constant for all page allocators.
///
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
        // ZST don't require an allocation.
        if layout.size() == 0 {
            return Ok(NonNull::slice_from_raw_parts(layout.dangling_ptr(), 0));
        }

        // Check if the required alignment is greater than the page size.
        if layout.align() > self.unit {
            return Err(AllocError);
        }

        sys::mmap(self.round(layout.size()), self.flags).ok_or(AllocError)
    }

    /// # Safety
    ///
    /// Same as [`Allocator::deallocate`], for a block returned by this mapping.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the `Allocator` contract guarantees `ptr` comes from a valid allocation.
        unsafe { sys::munmap(ptr, self.round(layout.size())) }
    }

    /// # Safety
    ///
    /// Same as [`Allocator::grow`] / [`Allocator::shrink`], for a block returned by this mapping.
    ///
    /// When manual is true, skip trying the `mremap` and do a fresh `mmap`.
    /// Also returns Ok((_, true)) when the new allocation is a fresh `mmap`, or a dangling pointer
    /// for a ZST allocation.
    unsafe fn resize_inner(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
        manual: bool,
    ) -> Result<(NonNull<[u8]>, bool), AllocError> {
        // Check if the old allocation was a ZST.
        if old.size() == 0 {
            return self.allocate(new).map(|ptr| (ptr, true));
        }

        // Check if the required alignment is greater than the page size.
        if new.align() > self.unit {
            return Err(AllocError);
        }

        // Check if the new allocation is a ZST.
        if new.size() == 0 {
            // SAFETY: the `Allocator` contract guarantees `ptr` comes from a valid allocation.
            unsafe {
                self.deallocate(ptr, old);
            }

            return Ok((NonNull::slice_from_raw_parts(new.dangling_ptr(), 0), true));
        }

        // Check if the new size requested actually fits in the same block.
        let (old_rounded, new_rounded) = (self.round(old.size()), self.round(new.size()));
        if old_rounded == new_rounded {
            return Ok((NonNull::slice_from_raw_parts(ptr, new_rounded), false));
        }

        if !manual {
            // SAFETY: the `Allocator` contract guarantees `ptr` comes from a valid allocation.
            if let Some(ptr) = unsafe { sys::mremap(ptr, old_rounded, new_rounded) } {
                return Ok((ptr, false));
            }
        }

        // Fallback to manual reallocation.
        let block = self.allocate(new)?;

        // SAFETY: `ptr` and `block` come from different allocations and the size is within bounds.
        unsafe {
            ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                block.cast().as_ptr(),
                old.size().min(new.size()),
            );
        }

        // SAFETY: the `Allocator` contract guarantees `ptr` comes from a valid allocation.
        unsafe {
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
                    .write_bytes(0, self.round(old.size()) - old.size());
            };
        }

        Ok(block)
    }
}

/// Simple page allocator based on `mmap` family of syscalls.
///
/// New allocations use `mmap`, grow/shrink always try to use `mremap` to attempt in-place resize,
/// and deallocations go through `munmap`.
///
/// All allocations will be multiples of [`Page::page_size()`], which can be quite large for most
/// uses, so it is recommended to combine it with a [`Bump`](crate::Bump) or [`Arena`](crate::Arena)
/// when doing many smaller allocations.
///
#[derive(Clone, Copy, Debug, Default)]
pub struct Page;

impl Page {
    fn mapping() -> Mapping {
        Mapping {
            unit: Self::page_size(),
            flags: 0,
        }
    }
}

impl Page {
    /// Returns the size of a page of memory on the current system.
    /// This is also the smallest unit of memory that this allocator can provide.
    ///
    #[must_use]
    pub fn page_size() -> usize {
        static PAGE_SIZE: OnceLock<usize> = OnceLock::new();
        *PAGE_SIZE.get_or_init(sys::get_page_size)
    }
}

/// Huge page allocator based on `mmap` family of syscalls.
///
/// Same as [`Page`] except it allocates huge pages instead of regular pages. These pages are
/// much bigger in size, depending on the target system. On common hardware, the [`HUGE_PAGE_2MB`]
/// and [`HUGE_PAGE_1GB`] are most probable to be supported.
///
/// Having hardware support for huge pages is not sufficient for this allocator to work, you
/// must also have some huge pages reserved by the operating system. Failing to do this will
/// result in all allocations returning `AllocError`.
///
/// Huge pages do not support growing in place, so [`Allocator::grow`] and
/// [`Allocator::grow_zeroed`] will have to reallocate and copy the contents, unless the new
/// size fits in the same huge pages.
///
/// For convenience, it is recommended to use these types instead:
///  - [`HugePage2MB`]
///  - [`HugePage1GB`]
///
#[derive(Clone, Copy, Debug, Default)]
pub struct HugePage<const SHIFT: u32 = HUGE_PAGE_2MB>;

pub const HUGE_PAGE_2MB: u32 = 21;
pub const HUGE_PAGE_1GB: u32 = 30;

/// Huge page allocator, with 2 MB huge pages, based on `mmap` family of syscalls.
///
/// For more details, please see [`crate::page::HugePage`].
///
pub type HugePage2MB = HugePage<HUGE_PAGE_2MB>;

/// Huge page allocator, with 1 GB huge pages, based on `mmap` family of syscalls.
///
/// For more details, please see [`crate::page::HugePage`].
///
pub type HugePage1GB = HugePage<HUGE_PAGE_1GB>;

impl<const SHIFT: u32> HugePage<SHIFT> {
    const SIZE: usize = 1 << SHIFT;
    const MAPPING: Mapping = Mapping {
        unit: Self::SIZE,
        flags: libc::MAP_HUGETLB | (SHIFT.cast_signed() << libc::MAP_HUGE_SHIFT),
    };
}

impl<const SHIFT: u32> HugePage<SHIFT> {
    /// Returns the size of a huge page for this specific allocator.
    /// Note that the target system might not actually support this page size, or might not have
    /// any pages of this size allocated by the operating system. This function does not guarantee
    /// that allocations from this allocator will succeed.
    ///
    #[must_use]
    pub fn page_size() -> usize {
        Self::SIZE
    }
}

// SAFETY: See safety docs for `Mapping`.
unsafe impl Allocator for Page {
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

// SAFETY: See safety docs for `Mapping`.
unsafe impl<const SHIFT: u32> Allocator for HugePage<SHIFT> {
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
