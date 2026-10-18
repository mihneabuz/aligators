use std::{
    alloc::{Allocator, Layout},
    ptr::NonNull,
};

use aligators::{AtomicBump, Buffer, Bump};

#[cfg(target_os = "linux")]
use aligators::page::Page;

fn layout(size: usize, align: usize) -> Layout {
    Layout::from_size_align(size, align).unwrap()
}

fn bytes<'a>(block: NonNull<[u8]>) -> &'a mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(block.cast::<u8>().as_ptr(), block.len()) }
}

/// Checks that `block` fits `layout`, then writes all of it so Miri catches out of bounds blocks.
fn check_block(block: NonNull<[u8]>, layout: Layout) {
    assert!(block.len() >= layout.size());
    assert!(
        block
            .cast::<u8>()
            .addr()
            .get()
            .is_multiple_of(layout.align())
    );
    bytes(block).fill(0xAB);
}

/// Like `check_block`, but first checks the block still starts with `aligator`, and writes it
/// back afterwards for the next resize.
fn check_block_keeping(block: NonNull<[u8]>, layout: Layout) {
    assert_eq!(&bytes(block)[..8], b"aligator");
    check_block(block, layout);
    bytes(block)[..8].copy_from_slice(b"aligator");
}

/// Fills a block of `size` bytes with `0xFF` and frees it, so allocators that reuse memory
/// (like a bump allocator rewinding its last block) hand out dirty memory next.
fn dirty(alloc: &dyn Allocator, size: usize) {
    let block = alloc.allocate(layout(size, 8)).unwrap();
    bytes(block).fill(0xFF);
    unsafe { alloc.deallocate(block.cast(), layout(size, 8)) };
}

fn allocate(alloc: &dyn Allocator) {
    for size in [1, 7, 16, 100, 127, 129] {
        let block = alloc.allocate(layout(size, 1)).unwrap();
        check_block(block, layout(size, 1));
        unsafe { alloc.deallocate(block.cast(), layout(size, 1)) };
    }
}

fn allocate_zeroed(alloc: &dyn Allocator) {
    let block = alloc.allocate_zeroed(layout(64, 8)).unwrap();
    assert!(bytes(block).iter().all(|&b| b == 0));
    check_block(block, layout(64, 8));
    unsafe { alloc.deallocate(block.cast(), layout(64, 8)) };
}

fn alignment(alloc: &dyn Allocator) {
    for align in [1, 2, 4, 8, 16, 32, 64, 128] {
        let block = alloc.allocate(layout(8, align)).unwrap();
        check_block(block, layout(8, align));
        unsafe { alloc.deallocate(block.cast(), layout(8, align)) };
    }
}

fn zero_size(alloc: &dyn Allocator) {
    let empty = alloc.allocate(layout(0, 8)).unwrap();
    assert!(empty.cast::<u8>().addr().get().is_multiple_of(8));
    unsafe { alloc.deallocate(empty.cast(), layout(0, 8)) };
}

fn grow(alloc: &dyn Allocator) {
    let block = alloc.allocate(layout(8, 1)).unwrap();
    bytes(block)[..8].copy_from_slice(b"aligator");

    let grown = unsafe { alloc.grow(block.cast(), layout(8, 1), layout(48, 1)) }.unwrap();
    assert_eq!(&bytes(grown)[..8], b"aligator");
    check_block(grown, layout(48, 1));

    unsafe { alloc.deallocate(grown.cast(), layout(48, 1)) };
}

fn grow_zeroed(alloc: &dyn Allocator) {
    let block = alloc.allocate(layout(8, 1)).unwrap();
    bytes(block).fill(0xFF);

    let grown = unsafe { alloc.grow_zeroed(block.cast(), layout(8, 1), layout(32, 1)) }.unwrap();
    assert!(bytes(grown)[..8].iter().all(|&b| b == 0xFF));
    // Everything past the old size is zeroed, up to the returned length.
    assert!(bytes(grown)[8..].iter().all(|&b| b == 0));

    unsafe { alloc.deallocate(grown.cast(), layout(32, 1)) };
}

fn grow_align(alloc: &dyn Allocator) {
    let block = alloc.allocate(layout(8, 4)).unwrap();
    bytes(block)[..8].copy_from_slice(b"aligator");

    let grown = unsafe { alloc.grow(block.cast(), layout(8, 4), layout(16, 2)) }.unwrap();
    check_block_keeping(grown, layout(16, 2));

    let growner = unsafe { alloc.grow(grown.cast(), layout(16, 2), layout(32, 8)) }.unwrap();
    check_block_keeping(growner, layout(32, 8));

    unsafe { alloc.deallocate(growner.cast(), layout(32, 8)) };
}

fn grow_misaligned(alloc: &dyn Allocator) {
    let pad = alloc.allocate(layout(1, 1)).unwrap();
    bytes(pad).fill(0xCD);
    let block = alloc.allocate(layout(8, 4)).unwrap();
    bytes(block)[..8].copy_from_slice(b"aligator");

    let grown = unsafe { alloc.grow(block.cast(), layout(8, 4), layout(32, 8)) }.unwrap();
    check_block_keeping(grown, layout(32, 8));

    assert!(bytes(pad).iter().all(|&b| b == 0xCD));
    unsafe {
        alloc.deallocate(grown.cast(), layout(32, 8));
        alloc.deallocate(pad.cast(), layout(1, 1));
    }
}

fn grow_not_last(alloc: &dyn Allocator) {
    dirty(alloc, 512);

    let block = alloc.allocate(layout(8, 8)).unwrap();
    bytes(block)[..8].copy_from_slice(b"aligator");
    let next = alloc.allocate(layout(8, 8)).unwrap();
    bytes(next).fill(0xCD);

    let grown = unsafe { alloc.grow(block.cast(), layout(8, 8), layout(64, 8)) }.unwrap();
    check_block_keeping(grown, layout(64, 8));

    let zeroed = unsafe { alloc.grow_zeroed(grown.cast(), layout(64, 8), layout(128, 8)) }.unwrap();
    assert_eq!(&bytes(zeroed)[..8], b"aligator");
    assert!(bytes(zeroed)[64..].iter().all(|&b| b == 0));
    check_block(zeroed, layout(128, 8));

    assert!(bytes(next).iter().all(|&b| b == 0xCD));
    unsafe {
        alloc.deallocate(zeroed.cast(), layout(128, 8));
        alloc.deallocate(next.cast(), layout(8, 8));
    }
}

fn grow_from_zero(alloc: &dyn Allocator) {
    let empty = alloc.allocate(layout(0, 8)).unwrap();
    let grown = unsafe { alloc.grow(empty.cast(), layout(0, 8), layout(16, 8)) }.unwrap();
    check_block(grown, layout(16, 8));
    unsafe { alloc.deallocate(grown.cast(), layout(16, 8)) };

    let empty = alloc.allocate(layout(0, 8)).unwrap();
    let grown = unsafe { alloc.grow_zeroed(empty.cast(), layout(0, 8), layout(16, 8)) }.unwrap();
    assert!(bytes(grown).iter().all(|&b| b == 0));
    check_block(grown, layout(16, 8));
    unsafe { alloc.deallocate(grown.cast(), layout(16, 8)) };
}

fn shrink(alloc: &dyn Allocator) {
    let block = alloc.allocate(layout(32, 1)).unwrap();
    bytes(block)[..8].copy_from_slice(b"aligator");

    let shrunk = unsafe { alloc.shrink(block.cast(), layout(32, 1), layout(8, 1)) }.unwrap();
    check_block_keeping(shrunk, layout(8, 1));

    unsafe { alloc.deallocate(shrunk.cast(), layout(8, 1)) };
}

fn shrink_align(alloc: &dyn Allocator) {
    let block = alloc.allocate(layout(32, 4)).unwrap();
    bytes(block)[..8].copy_from_slice(b"aligator");

    let shrunk = unsafe { alloc.shrink(block.cast(), layout(32, 4), layout(16, 1)) }.unwrap();
    check_block_keeping(shrunk, layout(16, 1));

    let shrunker = unsafe { alloc.shrink(shrunk.cast(), layout(16, 1), layout(8, 8)) }.unwrap();
    check_block_keeping(shrunker, layout(8, 8));

    unsafe { alloc.deallocate(shrunker.cast(), layout(8, 8)) };
}

fn shrink_misaligned(alloc: &dyn Allocator) {
    let pad = alloc.allocate(layout(1, 1)).unwrap();
    bytes(pad).fill(0xCD);
    let block = alloc.allocate(layout(32, 4)).unwrap();
    bytes(block)[..8].copy_from_slice(b"aligator");

    let shrunk = unsafe { alloc.shrink(block.cast(), layout(32, 4), layout(8, 8)) }.unwrap();
    check_block_keeping(shrunk, layout(8, 8));

    assert!(bytes(pad).iter().all(|&b| b == 0xCD));
    unsafe {
        alloc.deallocate(shrunk.cast(), layout(8, 8));
        alloc.deallocate(pad.cast(), layout(1, 1));
    }
}

fn shrink_to_zero(alloc: &dyn Allocator) {
    let block = alloc.allocate(layout(16, 8)).unwrap();
    let empty = unsafe { alloc.shrink(block.cast(), layout(16, 8), layout(0, 8)) }.unwrap();
    assert!(empty.cast::<u8>().addr().get().is_multiple_of(8));
    unsafe { alloc.deallocate(empty.cast(), layout(0, 8)) };

    // Whatever the shrink did with the old block, the allocator still has room for it.
    let block = alloc.allocate(layout(16, 8)).unwrap();
    check_block(block, layout(16, 8));
    unsafe { alloc.deallocate(block.cast(), layout(16, 8)) };
}

fn dirty_zeroed(alloc: &dyn Allocator) {
    dirty(alloc, 256);
    let block = alloc.allocate_zeroed(layout(256, 8)).unwrap();
    // The whole returned block is zeroed, not just the requested size.
    assert!(bytes(block).iter().all(|&b| b == 0));
    unsafe { alloc.deallocate(block.cast(), layout(256, 8)) };

    dirty(alloc, 256);
    let block = alloc.allocate(layout(8, 8)).unwrap();
    bytes(block).fill(0xAB);
    let grown = unsafe { alloc.grow_zeroed(block.cast(), layout(8, 8), layout(256, 8)) }.unwrap();
    assert!(bytes(grown)[..8].iter().all(|&b| b == 0xAB));
    assert!(bytes(grown)[8..].iter().all(|&b| b == 0));
    unsafe { alloc.deallocate(grown.cast(), layout(256, 8)) };
}

fn too_large(alloc: &dyn Allocator) {
    let huge = layout(isize::MAX as usize / 2, 8);
    assert!(alloc.allocate(huge).is_err());
    assert!(alloc.allocate_zeroed(huge).is_err());

    let block = alloc.allocate(layout(8, 8)).unwrap();
    bytes(block)[..8].copy_from_slice(b"aligator");

    // A failed grow leaves the old block valid and untouched.
    unsafe {
        assert!(alloc.grow(block.cast(), layout(8, 8), huge).is_err());
        assert!(alloc.grow_zeroed(block.cast(), layout(8, 8), huge).is_err());
    }
    check_block_keeping(block, layout(8, 8));

    unsafe { alloc.deallocate(block.cast(), layout(8, 8)) };
}

fn vec(alloc: &dyn Allocator) {
    let mut vec = Vec::new_in(alloc);
    for i in 0..50 {
        vec.push(i * 4);
    }
    vec.truncate(20);
    vec.shrink_to_fit();

    assert!(vec.iter().copied().eq((0..20).map(|i| i * 4)));
}

fn boxed(alloc: &dyn Allocator) {
    let mut box1 = Box::new_in(10, alloc);
    let mut box2 = Box::new_in(20, alloc);
    let mut box3 = Box::new_in(30, alloc);

    *box1 += 1;
    *box2 += 2;
    *box3 += 3;

    drop(box1);
    drop(box3);
    assert_eq!(*box2, 22);

    let big = Box::new_in([8u8; 100], alloc);
    assert_eq!(&*big, &[8u8; 100]);
}

macro_rules! sanity {
    ($($(#[$meta:meta])* $name:ident => $make:expr; [$($(#[$check_meta:meta])* $check:ident),* $(,)?])*) => {
        $(
            $(#[$meta])*
            mod $name {
                #[allow(unused_imports)]
                use super::*;
                $(
                    #[test]
                    $(#[$check_meta])*
                    fn $check() {
                        super::$check(&$make);
                    }
                )*
            }
        )*
    };
}

mod sanity {
    use super::*;

    sanity! {
        #[cfg(target_os = "linux")]
        page => Page;
            [allocate, alignment, zero_size, allocate_zeroed, grow, grow_zeroed, grow_align, shrink,
             shrink_align, vec, boxed, grow_not_last, grow_misaligned, shrink_misaligned, dirty_zeroed,
             grow_from_zero, shrink_to_zero, #[cfg_attr(miri, ignore)] too_large]

        buffer => Buffer::new(&mut [0u8; 2048]);
            [allocate, alignment, zero_size, allocate_zeroed, grow, grow_zeroed, grow_align, shrink,
             shrink_align, vec, dirty_zeroed, too_large, grow_from_zero, shrink_to_zero]

        bump => Bump::with_buffer(&mut [0u8; 2048]);
            [allocate, alignment, zero_size, allocate_zeroed, grow, grow_zeroed, grow_align, shrink,
             shrink_align, vec, boxed, grow_not_last, grow_misaligned, shrink_misaligned, dirty_zeroed,
             too_large, grow_from_zero, shrink_to_zero]

        bump_atomic => AtomicBump::with_buffer(&mut [0u8; 2048]);
            [allocate, alignment, zero_size, allocate_zeroed, grow, grow_zeroed, grow_align, shrink,
             shrink_align, vec, boxed, grow_not_last, grow_misaligned, shrink_misaligned, dirty_zeroed,
             too_large, grow_from_zero, shrink_to_zero]

        #[cfg(target_os = "linux")]
        bump_on_page => Bump::with_capacity_in(2048, Page);
            [allocate, alignment, zero_size, allocate_zeroed, grow, grow_zeroed, grow_align, shrink,
             shrink_align, vec, boxed, grow_not_last, grow_misaligned, shrink_misaligned, dirty_zeroed,
             too_large, grow_from_zero, shrink_to_zero]
    }
}
