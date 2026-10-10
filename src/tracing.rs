use std::{
    alloc::{AllocError, Allocator, Layout},
    ptr::NonNull,
};

pub const TRACE: u8 = 1;
pub const DEBUG: u8 = 2;
pub const INFO: u8 = 3;
pub const WARN: u8 = 4;
pub const ERROR: u8 = 5;

macro_rules! event {
    ($level:expr, $($arg:tt)+) => {
        match $level {
            ERROR => ::tracing::error!($($arg)+),
            WARN => ::tracing::warn!($($arg)+),
            INFO => ::tracing::info!($($arg)+),
            DEBUG => ::tracing::debug!($($arg)+),
            _ => ::tracing::trace!($($arg)+),
        }
    };
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Tracing<A, const LEVEL: u8 = DEBUG> {
    inner: A,
}

impl<A: Allocator, const LEVEL: u8> Tracing<A, LEVEL> {
    fn alloc(&self, layout: Layout, zeroed: bool) -> Result<NonNull<[u8]>, AllocError> {
        event!(LEVEL, ?layout, ?zeroed, "Allocating");

        let res = if zeroed {
            self.inner.allocate_zeroed(layout)
        } else {
            self.inner.allocate(layout)
        };

        match res {
            Ok(block) => {
                event!(ERROR, ptr =? block.as_ptr(), len =? block.len(), "Allocated block");
                Ok(block)
            }
            Err(err) => {
                event!(ERROR, ?err, "Failed to allocate");
                Err(err)
            }
        }
    }

    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        event!(LEVEL, ptr =? ptr.as_ptr(), ?layout, "Deallocating");
        unsafe { self.inner.deallocate(ptr, layout) };
    }

    unsafe fn resize(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
        grow: bool,
        zeroed: bool,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if grow {
            event!(LEVEL, ptr =? ptr.as_ptr(), ?new, ?old, ?zeroed, "Growing allocation");
        } else {
            event!(LEVEL, ptr =? ptr.as_ptr(), ?new, ?old, "Shrinking allocating");
        }

        let res = match (grow, zeroed) {
            (true, false) => unsafe { self.inner.grow(ptr, old, new) },
            (true, true) => unsafe { self.inner.grow_zeroed(ptr, old, new) },
            (false, false) => unsafe { self.inner.shrink(ptr, old, new) },
            _ => unreachable!(),
        };

        match res {
            Ok(block) => {
                event!(ERROR, ptr =? block.as_ptr(), len =? block.len(), "Resized block");
                Ok(block)
            }
            Err(err) => {
                event!(ERROR, ?err, "Failed to resize");
                Err(err)
            }
        }
    }
}

unsafe impl<A: Allocator, const LEVEL: u8> Allocator for Tracing<A, LEVEL> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc(layout, false)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc(layout, true)
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { self.resize(ptr, old_layout, new_layout, true, false) }
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { self.resize(ptr, old_layout, new_layout, true, true) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { self.resize(ptr, old_layout, new_layout, false, false) }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { self.dealloc(ptr, layout) }
    }
}
