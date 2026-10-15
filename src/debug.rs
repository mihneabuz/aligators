use std::{
    alloc::{AllocError, Allocator, Layout},
    backtrace::{Backtrace, BacktraceStatus},
    collections::HashMap,
    fmt,
    ptr::NonNull,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Violation {
    UnknownPointer {
        ptr: usize,
        layout: Layout,
    },

    LayoutMismatch {
        ptr: usize,
        expected: Layout,
        len: usize,
        got: Layout,
    },

    Leak {
        ptr: usize,
        layout: Layout,
        len: usize,
    },
}

#[derive(Debug)]
pub struct Report {
    pub violation: Violation,
    pub allocated_at: Option<Arc<Backtrace>>,
    pub detected_at: Option<Backtrace>,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.violation)?;
        let backtraces = [
            ("allocated at", self.allocated_at.as_deref()),
            ("detected at", self.detected_at.as_ref()),
        ];
        for (label, backtrace) in backtraces {
            if let Some(backtrace) = backtrace
                && backtrace.status() == BacktraceStatus::Captured
            {
                write!(f, "\n{label}:\n{backtrace}")?;
            }
        }
        Ok(())
    }
}

struct Block {
    layout: Layout,
    len: usize,
    allocated_at: Arc<Backtrace>,
}

impl Block {
    fn fits(&self, layout: Layout) -> bool {
        layout.align() == self.layout.align()
            && (self.layout.size()..=self.len).contains(&layout.size())
    }
}

#[derive(Default)]
struct State {
    blocks: HashMap<usize, Block>,
    reports: Vec<Report>,
}

/// Records every live block of the inner allocator and checks the layouts passed to `deallocate`,
/// `grow` and `shrink`.
///
/// Misuses are recorded as [`Report`]s instead of being forwarded: a bad `deallocate` leaks
/// the block, and a bad `grow` / `shrink` fails with [`AllocError`]. Zero-sized blocks aren't
/// tracked, since their dangling pointers aren't unique.
///
/// Use it by reference (`&Tracking<A>` is an allocator too) and call [`Tracking::finish`] once
/// every user is gone, to also get the leaked blocks.
pub struct Tracking<A> {
    inner: A,
    state: Mutex<State>,
}

impl<A> Tracking<A> {
    pub fn new(inner: A) -> Self {
        Self {
            inner,
            state: Mutex::default(),
        }
    }

    pub fn inner(&self) -> &A {
        &self.inner
    }

    pub fn violations(&self) -> Vec<Violation> {
        self.state()
            .reports
            .iter()
            .map(|report| report.violation)
            .collect()
    }

    pub fn take_violations(&self) -> Vec<Violation> {
        self.take_reports()
            .into_iter()
            .map(|report| report.violation)
            .collect()
    }

    pub fn take_reports(&self) -> Vec<Report> {
        std::mem::take(&mut self.state().reports)
    }

    pub fn finish(self) -> Vec<Report> {
        let State {
            blocks,
            mut reports,
        } = self
            .state
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner);

        let mut leaks = blocks.into_iter().collect::<Vec<_>>();
        leaks.sort_unstable_by_key(|&(ptr, _)| ptr);
        reports.extend(leaks.into_iter().map(|(ptr, block)| Report {
            violation: Violation::Leak {
                ptr,
                layout: block.layout,
                len: block.len,
            },
            allocated_at: Some(block.allocated_at),
            detected_at: None,
        }));
        reports
    }

    fn state(&self) -> MutexGuard<'_, State> {
        // The map is never left half-updated, so a panic elsewhere doesn't invalidate it.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn insert(&self, block: NonNull<[u8]>, layout: Layout) {
        if layout.size() > 0 {
            let entry = Block {
                layout,
                len: block.len(),
                allocated_at: Arc::new(Backtrace::capture()),
            };
            self.state().blocks.insert(block.addr().get(), entry);
        }
    }

    fn check(&self, ptr: NonNull<u8>, layout: Layout) -> bool {
        let ptr = ptr.addr().get();
        let (violation, allocated_at) = match self.state().blocks.get(&ptr) {
            Some(block) if block.fits(layout) => return true,
            Some(block) => (
                Violation::LayoutMismatch {
                    ptr,
                    expected: block.layout,
                    len: block.len,
                    got: layout,
                },
                Some(Arc::clone(&block.allocated_at)),
            ),
            None if layout.size() == 0 => return true,
            None => (Violation::UnknownPointer { ptr, layout }, None),
        };

        let report = Report {
            violation,
            allocated_at,
            detected_at: Some(Backtrace::capture()),
        };
        self.state().reports.push(report);
        false
    }

    fn remove(&self, ptr: NonNull<u8>) {
        self.state().blocks.remove(&ptr.addr().get());
    }

    fn replace(&self, ptr: NonNull<u8>, block: NonNull<[u8]>, layout: Layout) {
        self.remove(ptr);
        self.insert(block, layout);
    }
}

unsafe impl<A: Allocator> Allocator for Tracking<A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let block = self.inner.allocate(layout)?;
        self.insert(block, layout);
        Ok(block)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let block = self.inner.allocate_zeroed(layout)?;
        self.insert(block, layout);
        Ok(block)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let fits = self.check(ptr, layout);
        self.remove(ptr);
        if fits {
            // SAFETY: `check` verified `ptr` is a live block of `inner` fitting `layout`.
            unsafe { self.inner.deallocate(ptr, layout) };
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if !self.check(ptr, old_layout) {
            return Err(AllocError);
        }
        // SAFETY: `check` verified `ptr` is a live block of `inner` fitting `old_layout`; the
        // size precondition is forwarded from the caller.
        let block = unsafe { self.inner.grow(ptr, old_layout, new_layout) }?;
        self.replace(ptr, block, new_layout);
        Ok(block)
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if !self.check(ptr, old_layout) {
            return Err(AllocError);
        }
        // SAFETY: as in `grow`.
        let block = unsafe { self.inner.grow_zeroed(ptr, old_layout, new_layout) }?;
        self.replace(ptr, block, new_layout);
        Ok(block)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if !self.check(ptr, old_layout) {
            return Err(AllocError);
        }
        // SAFETY: as in `grow`.
        let block = unsafe { self.inner.shrink(ptr, old_layout, new_layout) }?;
        self.replace(ptr, block, new_layout);
        Ok(block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Global;

    fn layout(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).unwrap()
    }

    fn violations(reports: Vec<Report>) -> Vec<Violation> {
        reports.into_iter().map(|report| report.violation).collect()
    }

    #[test]
    fn clean() {
        let tracking = Tracking::new(Global);

        let mut v = Vec::new_in(&tracking);
        for i in 0..1000 {
            v.push(i);
        }
        v.truncate(10);
        v.shrink_to_fit();
        drop(v);
        assert_eq!(tracking.take_violations(), []);

        let zeroed = tracking.allocate_zeroed(layout(128, 8)).unwrap();
        unsafe { tracking.deallocate(zeroed.cast(), layout(128, 8)) };
        assert_eq!(tracking.take_violations(), []);
    }

    #[test]
    fn leak() {
        let tracking = Tracking::new(Global);
        let block1 = tracking.allocate(layout(256, 8)).unwrap();
        let block2 = tracking.allocate(layout(64, 8)).unwrap();

        unsafe { tracking.deallocate(block1.cast(), layout(256, 8)) };
        assert_eq!(tracking.take_violations(), []);

        let reports = tracking.finish();
        assert_eq!(
            violations(reports),
            vec![Violation::Leak {
                ptr: block2.addr().get(),
                layout: layout(64, 8),
                len: block2.len()
            }]
        );

        unsafe { Global.deallocate(block2.cast(), layout(64, 8)) };
    }

    #[test]
    fn double_free() {
        let tracking = Tracking::new(Global);
        let block = tracking.allocate(layout(256, 8)).unwrap();

        unsafe { tracking.deallocate(block.cast(), layout(256, 8)) };
        assert_eq!(tracking.take_violations(), []);

        unsafe { tracking.deallocate(block.cast(), layout(256, 8)) };
        assert_eq!(
            tracking.take_violations(),
            vec![Violation::UnknownPointer {
                ptr: block.addr().get(),
                layout: layout(256, 8)
            }]
        );
    }

    #[test]
    fn bad_layout() {
        let tracking = Tracking::new(Global);

        let block = tracking.allocate(layout(256, 8)).unwrap();
        unsafe { tracking.deallocate(block.cast(), layout(256, 4)) };
        assert_eq!(
            tracking.take_violations(),
            vec![Violation::LayoutMismatch {
                ptr: block.addr().get(),
                expected: layout(256, 8),
                len: block.len(),
                got: layout(256, 4)
            }]
        );
        unsafe { Global.deallocate(block.cast(), layout(256, 8)) };

        let block = tracking.allocate(layout(256, 8)).unwrap();
        unsafe { tracking.deallocate(block.cast(), layout(256, 64)) };
        assert_eq!(
            tracking.take_violations(),
            vec![Violation::LayoutMismatch {
                ptr: block.addr().get(),
                expected: layout(256, 8),
                len: block.len(),
                got: layout(256, 64)
            }]
        );
        unsafe { Global.deallocate(block.cast(), layout(256, 8)) };

        let block = tracking.allocate(layout(256, 8)).unwrap();
        unsafe { tracking.deallocate(block.cast(), layout(16, 8)) };
        assert_eq!(
            tracking.take_violations(),
            vec![Violation::LayoutMismatch {
                ptr: block.addr().get(),
                expected: layout(256, 8),
                len: block.len(),
                got: layout(16, 8)
            }]
        );
        unsafe { Global.deallocate(block.cast(), layout(256, 8)) };
    }
}
