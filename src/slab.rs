use std::{
    alloc::{AllocError, Allocator, Global, Layout, handle_alloc_error},
    cell::Cell,
    mem,
    ptr::{self, NonNull},
};

const DEFAULT_INITIAL_CAPACITY: usize = 32;

struct BlockHeader {
    free: Option<NonNull<u8>>,
    base: NonNull<u8>,
    prev: NonNull<BlockHeader>,
    live: usize,
    slots: usize,
}

struct EmptySlot {
    next: Option<NonNull<u8>>,
}

fn slot_layout(item: Layout) -> Option<Layout> {
    let empty = Layout::new::<EmptySlot>();
    let size = item.size().max(empty.size());
    let align = item.align().max(empty.align());
    Some(Layout::from_size_align(size, align).ok()?.pad_to_align())
}

fn fits(slot: Layout, item: Layout) -> bool {
    item.size() <= slot.size() && item.align() <= slot.align()
}

fn layout(slot: Layout, count: usize) -> Option<(Layout, usize)> {
    let slots = slot.repeat(count).ok()?.0;
    Layout::new::<BlockHeader>().extend(slots).ok()
}

struct EmptyBlock(BlockHeader);

// SAFETY: the empty block is never written to, since it has no slots to hand out or take back.
unsafe impl Sync for EmptyBlock {}

// Sentinel block for a slab which didn't allocate yet.
static EMPTY_BLOCK: EmptyBlock = EmptyBlock(BlockHeader {
    free: None,
    base: NonNull::from_ref(&EMPTY_BLOCK.0).cast(),
    prev: NonNull::from_ref(&EMPTY_BLOCK.0),
    live: 0,
    slots: 0,
});

fn empty_block() -> NonNull<BlockHeader> {
    NonNull::from_ref(&EMPTY_BLOCK.0)
}

fn is_empty_block(ptr: NonNull<BlockHeader>) -> bool {
    ptr::eq(ptr.as_ptr(), &raw const EMPTY_BLOCK.0)
}

/// Fixed-size slot allocator for a single thread.
///
/// The allocator hands out slots of a single layout from a list of blocks allocated from a backing
/// allocator. The slot layout is picked by the first allocation, or by the `item` passed to
/// [`Slab::with_capacity_in`], and is rounded up to fit a pointer, which free slots use to link to
/// each other. Any allocation which fits in a slot takes a whole one, so a slab with 256 byte
/// slots can hold values of different types, as long as none is bigger or more aligned than that.
/// Allocations which don't fit fail.
///
/// - Allocating takes a free slot from the current block. When the current block is full, a new
///   one with four times as many slots is allocated and becomes current. Free slots in older
///   blocks are not reused until those blocks are freed.
/// - Deallocating puts the slot back on its block's free list. Finding the block means walking
///   the list of blocks, so it gets slower the more blocks there are. Blocks with no slots in use
///   are given back to `backing`, except for the current one so alloc/free cycles don't thrash.
/// - Growing and shrinking happen in place, as long as the new layout still fits in the slot, and
///   fail otherwise.
///
/// Slots never move, so they stay valid when a new block is allocated. All memory is reclaimed at
/// once with [`Slab::reset`], which also lets the next allocation pick a new slot layout, or when
/// the allocator is dropped.
///
/// ```
/// use aligators::Slab;
/// use std::alloc::{Allocator, Layout};
///
/// let mut slab = Slab::with_capacity(Layout::new::<[u64; 4]>(), 16);
///
/// {
///     // Different types share the slab, as long as they fit a slot.
///     let numbers = Box::new_in([1_u64, 2, 3, 4], &slab);
///     let byte = Box::new_in(7_u8, &slab);
///     assert_eq!(slab.used_slots(), 2);
///     assert_eq!(numbers[3] + u64::from(*byte), 11);
/// }
///
/// // The slots were given back, and are reused for the next allocations.
/// assert_eq!(slab.used_slots(), 0);
/// assert_eq!(slab.remaining_slots(), 16);
///
/// // Too big for a slot.
/// assert!(slab.allocate(Layout::new::<[u64; 5]>()).is_err());
/// ```
pub struct Slab<A: Allocator = Global> {
    block: Cell<NonNull<BlockHeader>>,
    slot: Cell<Layout>,
    backing: A,
}

impl<A: Allocator + Default> Default for Slab<A> {
    fn default() -> Self {
        Self::new_in(A::default())
    }
}

impl Slab {
    /// Creates a new allocator backed by the [`Global`] allocator.
    ///
    /// It doesn't allocate until the first allocation is made, which also picks the slot layout.
    #[must_use]
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    /// Creates a new allocator backed by the [`Global`] allocator, with slots that fit `item` and
    /// a first block of `capacity` slots.
    ///
    /// With a `capacity` of `0` this is the same as [`Slab::new`], and the first allocation picks
    /// the slot layout instead.
    ///
    /// # Panics
    ///
    /// If the [`Global`] allocator fails.
    #[must_use]
    pub fn with_capacity(item: Layout, capacity: usize) -> Self {
        Self::with_capacity_in(item, capacity, Global)
    }
}

impl<A: Allocator> Slab<A> {
    /// Creates a new allocator backed by `backing`.
    ///
    /// It doesn't allocate until the first allocation is made, which also picks the slot layout.
    #[must_use]
    pub fn new_in(backing: A) -> Self {
        Self {
            block: Cell::new(empty_block()),
            slot: Cell::new(Layout::new::<EmptySlot>()),
            backing,
        }
    }

    /// Creates a new allocator backed by `backing`, with slots that fit `item` and a first block
    /// of `capacity` slots.
    ///
    /// With a `capacity` of `0` this is the same as [`Slab::new_in`], and the first allocation
    /// picks the slot layout instead.
    ///
    /// # Panics
    ///
    /// If `backing` fails.
    #[must_use]
    pub fn with_capacity_in(item: Layout, capacity: usize, backing: A) -> Self {
        match Self::try_with_capacity_in(item, capacity, backing) {
            Ok(alloc) => alloc,
            Err(_) => match slot_layout(item).and_then(|slot| layout(slot, capacity)) {
                Some((layout, _)) => handle_alloc_error(layout),
                None => panic!("capacity overflow"),
            },
        }
    }

    /// Creates a new allocator backed by `backing`, with slots that fit `item` and a first block
    /// of `capacity` slots.
    ///
    /// With a `capacity` of `0` this is the same as [`Slab::new_in`], and the first allocation
    /// picks the slot layout instead.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if `backing` fails.
    pub fn try_with_capacity_in(
        item: Layout,
        capacity: usize,
        backing: A,
    ) -> Result<Self, AllocError> {
        let slab = Self::new_in(backing);
        if capacity > 0 {
            slab.new_block(item, capacity)?;
        }
        Ok(slab)
    }

    // Allocates a new block of `capacity` slots and makes it current. The first block picks the
    // slot layout from `wanted`, later ones fail if `wanted` doesn't fit.
    #[cold]
    #[inline(never)]
    fn new_block(&self, wanted: Layout, capacity: usize) -> Result<(), AllocError> {
        if is_empty_block(self.block.get()) {
            self.slot.set(slot_layout(wanted).ok_or(AllocError)?);
        } else if !fits(self.slot.get(), wanted) {
            return Err(AllocError);
        }

        let (layout, offset) = layout(self.slot.get(), capacity).ok_or(AllocError)?;
        let new = self.backing.allocate(layout)?;
        let header = new.cast::<BlockHeader>();

        // SAFETY: `layout` puts the slots at `offset`, inside the block.
        let base = unsafe { new.byte_add(offset).cast::<u8>() };
        let stride = self.slot.get().size();

        for i in 0..capacity {
            // SAFETY: slot `i + 1` exists, so it is in bounds.
            let next = (i + 1 < capacity).then(|| unsafe { base.byte_add((i + 1) * stride) });
            // SAFETY: slot `i` is in bounds, and aligned for `EmptySlot` since slots fit it.
            unsafe {
                base.byte_add(i * stride)
                    .cast::<EmptySlot>()
                    .write(EmptySlot { next });
            }
        }

        // SAFETY: `layout` starts with a `BlockHeader`, so the block is big and aligned enough.
        unsafe {
            header.write(BlockHeader {
                free: Some(base),
                base,
                prev: self.block.get(),
                live: 0,
                slots: capacity,
            });
        }

        self.block.set(header);

        Ok(())
    }

    /// Returns the number of slots in use, over all blocks.
    pub fn used_slots(&self) -> usize {
        let mut block = self.block.get();
        let mut total = 0;

        while !is_empty_block(block) {
            // SAFETY: every block other than the empty one is live and starts with a header.
            let header = unsafe { block.as_ref() };
            total += header.live;
            block = header.prev;
        }

        total
    }

    /// Returns the number of free slots in the current block.
    ///
    /// Allocations come from the current block until it fills up, then a new block is started.
    /// Before the first block is allocated, this is `0`.
    pub fn remaining_slots(&self) -> usize {
        // SAFETY: `block` is either the empty block or a live block, both starting with a header.
        let header = unsafe { self.block.get().as_ref() };
        header.slots - header.live
    }

    /// Frees every allocation at once, and returns all blocks to `backing`.
    ///
    /// The slab is left as if newly created, so the next allocation picks the slot layout again.
    pub fn reset(&mut self) {
        let mut block = mem::replace(self.block.get_mut(), empty_block());

        while !is_empty_block(block) {
            // SAFETY: every block other than the empty one is live and starts with a header.
            let BlockHeader { prev, slots, .. } = unsafe { block.read() };

            if let Some((layout, _)) = layout(self.slot.get(), slots) {
                // SAFETY: `&mut self` means no allocation is still in use, and the block was
                // allocated by `backing` with this layout.
                unsafe { self.backing.deallocate(block.cast(), layout) };
            }

            block = prev;
        }

        *self.slot.get_mut() = Layout::new::<EmptySlot>();
    }

    // Slots for the next block, four times as many as the current one has.
    fn next_capacity(&self) -> usize {
        // SAFETY: `block` is either the empty block or a live block, both starting with a header.
        unsafe { self.block.get().as_ref() }.slots.saturating_mul(4)
    }

    // Resizes in place, which works as long as the new layout still fits the slot.
    #[inline]
    fn resize(&self, ptr: NonNull<u8>, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let slot = self.slot.get();
        if !fits(slot, layout) {
            return Err(AllocError);
        }
        Ok(NonNull::slice_from_raw_parts(ptr, slot.size()))
    }

    // Takes a free slot from the current block, which must not be the empty block.
    fn take_slot(&self) -> Option<NonNull<[u8]>> {
        debug_assert!(!is_empty_block(self.block.get()));
        // SAFETY: the current block is live, and no other reference to it exists.
        let block = unsafe { self.block.get().as_mut() };
        let free = block.free.take()?;
        // SAFETY: slots on the free list hold an `EmptySlot`.
        block.free = unsafe { free.cast::<EmptySlot>().as_ref() }.next;
        block.live += 1;
        Some(NonNull::slice_from_raw_parts(free, self.slot.get().size()))
    }
}

impl<A: Allocator> Drop for Slab<A> {
    fn drop(&mut self) {
        self.reset();
    }
}

// SAFETY: slots are only handed out once until they are deallocated, never move, and stay valid
// until then since only blocks with no slots in use are freed. `Slab` is not `Clone`.
unsafe impl<A: Allocator> Allocator for Slab<A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if is_empty_block(self.block.get()) {
            self.new_block(layout, DEFAULT_INITIAL_CAPACITY)?;
        }

        if !fits(self.slot.get(), layout) {
            return Err(AllocError);
        }

        if let Some(block) = self.take_slot() {
            return Ok(block);
        }

        self.new_block(layout, self.next_capacity())?;
        self.take_slot().ok_or(AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        let stride = self.slot.get().size();
        let mut newer = None;
        let mut block = self.block.get();

        // Find the block holding `ptr`, remembering the newer one so it can be unlinked.
        loop {
            debug_assert!(!is_empty_block(block), "pointer not allocated by this slab");
            // SAFETY: every block other than the empty one is live and starts with a header.
            let header = unsafe { block.as_ref() };
            // Wraps around for pointers below `base`, so one compare covers both bounds.
            let offset = ptr.addr().get().wrapping_sub(header.base.addr().get());
            if offset < header.slots * stride {
                break;
            }
            newer = Some(block);
            block = header.prev;
        }

        // SAFETY: as above, `block` is live.
        let header = unsafe { block.as_mut() };
        // SAFETY: `ptr` is a slot of this block, and slots fit an `EmptySlot`.
        unsafe {
            ptr.cast::<EmptySlot>()
                .write(EmptySlot { next: header.free });
        }
        header.free = Some(ptr);
        header.live -= 1;

        // Empty blocks are freed, except the current one so alloc/free cycles don't thrash.
        if let Some(mut newer) = newer
            && header.live == 0
        {
            let BlockHeader { prev, slots, .. } = *header;
            // SAFETY: `newer` is live, and no other reference to it exists.
            unsafe { newer.as_mut() }.prev = prev;
            if let Some((layout, _)) = layout(self.slot.get(), slots) {
                // SAFETY: the block was allocated by `backing` with this layout.
                unsafe { self.backing.deallocate(block.cast(), layout) };
            }
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        _old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.resize(ptr, new_layout)
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let block = self.resize(ptr, new_layout)?;
        // SAFETY: the slot is `block.len()` bytes long, and `old_layout` fits in it.
        unsafe {
            ptr.add(old_layout.size())
                .write_bytes(0, block.len() - old_layout.size());
        }
        Ok(block)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        _old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.resize(ptr, new_layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::Tracking;

    #[test]
    fn reuse() {
        let slab = Slab::with_capacity(Layout::new::<u64>(), 4);
        let a = Box::new_in(1_u64, &slab);
        let addr = &raw const *a;
        drop(a);
        let b = Box::new_in(2_u64, &slab);
        assert_eq!(&raw const *b, addr);
    }

    #[test]
    fn mixed() {
        let slab = Slab::with_capacity(Layout::from_size_align(256, 8).unwrap(), 4);
        let a = Box::new_in([1_u8; 200], &slab);
        let b = Box::new_in(2_u64, &slab);
        assert!(slab.allocate(Layout::new::<[u8; 257]>()).is_err());
        assert!(
            slab.allocate(Layout::from_size_align(8, 16).unwrap())
                .is_err()
        );
        assert_eq!((*a, *b), ([1; 200], 2));
    }

    #[test]
    fn reset() {
        let tracking = Tracking::new(Global);
        let mut slab = Slab::with_capacity_in(Layout::new::<u64>(), 2, &tracking);
        (0..10_u64).for_each(|i| mem::forget(Box::new_in(i, &slab)));
        slab.reset();
        // The slot layout is picked again.
        drop(Box::new_in([0_u8; 64], &slab));
        drop(slab);
        assert!(tracking.finish().is_empty());
    }

    #[test]
    fn resize() {
        let slab = Slab::with_capacity(Layout::from_size_align(64, 8).unwrap(), 2);
        let mut vec = Vec::with_capacity_in(2, &slab);
        vec.extend([1_u64, 2]);
        let addr = vec.as_ptr();
        vec.extend(3..=8);
        assert_eq!(vec.as_ptr(), addr);
        assert!(vec.try_reserve(1).is_err());
        vec.truncate(1);
        vec.shrink_to_fit();
        assert_eq!((vec.as_ptr(), &vec[..]), (addr, &[1][..]));
    }

    #[test]
    fn counts() {
        let slab = Slab::new();
        assert_eq!((slab.used_slots(), slab.remaining_slots()), (0, 0));
        let a = Box::new_in(1_u64, &slab);
        let b = Box::new_in(2_u64, &slab);
        assert_eq!(slab.used_slots(), 2);
        assert_eq!(slab.remaining_slots(), DEFAULT_INITIAL_CAPACITY - 2);
        drop(a);
        assert_eq!(slab.used_slots(), 1);
        assert_eq!(slab.remaining_slots(), DEFAULT_INITIAL_CAPACITY - 1);
        drop(b);
    }

    #[test]
    fn no_leak() {
        let tracking = Tracking::new(Global);
        let slab = Slab::with_capacity_in(Layout::new::<u64>(), 2, &tracking);
        for round in 0..3 {
            // Spans several blocks, then frees them out of order.
            let mut boxes: Vec<_> = (0..100_u64).map(|i| Box::new_in(i, &slab)).collect();
            boxes.reverse();
            boxes.retain(|b| (**b + round) % 2 == 0);
            drop(boxes);
        }
        drop(slab);
        assert!(tracking.finish().is_empty());
    }
}
