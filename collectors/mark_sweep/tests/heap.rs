use std::sync::Arc;

use core::alloc::Layout;
use core::ptr::NonNull;

use heap_api::{
    AllocError, CLEARED, GcHost, HeapBackend, RawCell, STRONG_PTR, Visitor, WEAK_PTR, Word,
};

use mark_sweep::block::{self, BlockHeader};
use mark_sweep::heap::{MarkSweep, MarkSweepConfig, MarkSweepLocal, MarkSweepState};

struct Roots {
    slots: Vec<RawCell>,
}

impl Roots {
    fn empty() -> Self {
        Self { slots: Vec::new() }
    }

    fn install(self, state: &MarkSweepState) -> Box<Self> {
        let boxed = Box::new(self);
        state.set_host(host_for(boxed.as_ref()));
        boxed
    }

    fn strong(&mut self, ptr: NonNull<u8>) {
        self.slots
            .push(unsafe { RawCell::from_word(ptr.as_ptr() as Word | STRONG_PTR) });
    }

    fn weak(&mut self, ptr: NonNull<u8>) -> usize {
        self.slots
            .push(unsafe { RawCell::from_word(ptr.as_ptr() as Word | WEAK_PTR) });
        self.slots.len() - 1
    }
}

fn host_for(roots: &Roots) -> GcHost {
    fn visit_roots(ctx: *const (), visitor: &mut dyn Visitor) {
        let roots = unsafe { &*(ctx as *const Roots) };
        for slot in &roots.slots {
            visitor.visit(slot);
        }
    }
    fn visit_object(_addr: NonNull<()>, _visitor: &mut dyn Visitor) {}
    fn layout_of(_addr: NonNull<()>) -> Layout {
        unreachable!()
    }
    GcHost {
        ctx: roots as *const Roots as *const (),
        visit_roots,
        layout_of,
        visit_object,
    }
}

fn backend(
    heap_size: usize,
    roots: Roots,
) -> (Arc<MarkSweepState>, Box<MarkSweepLocal>, Box<Roots>) {
    let state = MarkSweepState::new(MarkSweepConfig { heap_size }).unwrap();
    let roots = roots.install(&state);
    let local = MarkSweepLocal::new(Arc::clone(&state));
    (state, local, roots)
}

#[test]
fn vtable_roundtrip_smoke() {
    let ms = MarkSweep::new(MarkSweepConfig {
        heap_size: 64 * 1024,
    })
    .unwrap();
    let (shared, vtable) = ms.into_global();
    let local = (vtable.new_local)(shared);
    let ptr = (vtable.local_vtable.allocate_raw)(local, Layout::new::<u64>()).unwrap();
    assert!((vtable.contains)(shared, ptr.as_ptr() as Word));
    assert_eq!(
        (vtable.stats)(shared).capacity,
        (64 * 1024usize).next_multiple_of(4096)
    );
    (vtable.local_vtable.drop_local)(local);
    (vtable.drop_shared)(shared);
}

#[test]
fn allocations_are_aligned_and_accounted() {
    let (state, local, _roots) = backend(64 * 1024, Roots::empty());
    let a = local
        .allocate(Layout::from_size_align(5, 8).unwrap())
        .unwrap();
    let b = local.allocate(Layout::new::<u64>()).unwrap();
    assert_eq!(a.as_ptr() as usize % 16, 0);
    assert_eq!(b.as_ptr() as usize % 16, 0);
    assert_ne!(a, b);
    assert_eq!(state.stats().used, 64);
}

#[test]
fn garbage_is_reclaimed_and_coalesced() {
    let (state, local, _roots) = backend(64 * 1024, Roots::empty());
    for _ in 0..16 {
        local
            .allocate(Layout::from_size_align(512, 8).unwrap())
            .unwrap();
    }
    assert_eq!(state.stats().used, 16 * 528);

    local.collect();

    assert_eq!(state.stats().used, 0);
    // the whole arena must be one free block again
    let big = local
        .allocate(Layout::from_size_align(60 * 1024, 8).unwrap())
        .unwrap();
    assert_eq!(
        state.stats().used,
        (60 * 1024usize).next_multiple_of(16) + 16
    );
    let _ = big;
}

#[test]
fn oom_when_live_data_exhausts_arena() {
    let (state, local, mut roots) = backend(8 * 1024, Roots::empty());
    let mut live = Vec::new();
    loop {
        match local.allocate(Layout::new::<u64>()) {
            Ok(ptr) => {
                roots.strong(ptr);
                live.push(ptr);
            }
            Err(AllocError::OutOfMemory(_)) => break,
        }
    }
    assert!(state.stats().used > 0);
    // freeing the roots lets the next allocation trigger a cycle and succeed
    roots.slots.clear();
    drop(live);
    local.allocate(Layout::new::<u64>()).unwrap();
}

#[test]
fn collect_keeps_rooted_objects_only() {
    let (state, local, mut roots) = backend(64 * 1024, Roots::empty());
    let layout = Layout::from_size_align(1024, 8).unwrap();
    let a = local.allocate(layout).unwrap();
    let b = local.allocate(layout).unwrap();
    let c = local.allocate(layout).unwrap();
    roots.strong(a);
    roots.strong(c);
    let cycle_before = state.cycles();

    local.collect();

    assert_eq!(state.cycles(), cycle_before + 1);
    let block_size = 1024 + 16;
    assert_eq!(state.stats().used, 2 * block_size);
    // rooted payloads sit at their original addresses, the garbage one is free
    let blocks: Vec<(*mut u8, bool)> = block::blocks(state.base(), state.capacity())
        .map(|b| (BlockHeader::payload(b).as_ptr(), BlockHeader::is_free(b)))
        .collect();
    assert!(blocks.contains(&(a.as_ptr(), false)));
    assert!(blocks.contains(&(c.as_ptr(), false)));
    assert!(blocks.contains(&(b.as_ptr(), true)));
}

#[test]
fn sweep_coalesces_dead_block_with_surrounding_free_space() {
    let (state, local, mut roots) = backend(64 * 1024, Roots::empty());
    let layout = Layout::from_size_align(1024, 8).unwrap();
    let first = local.allocate(layout).unwrap();
    let middle = local.allocate(layout).unwrap();
    let tail = local.allocate(layout).unwrap();
    roots.strong(first);
    roots.strong(middle);

    local.collect();

    // first and middle survive; the dead tail block merges with the
    // trailing free space into a single free block
    let blocks: Vec<*mut BlockHeader> = block::blocks(state.base(), state.capacity()).collect();
    assert_eq!(blocks.len(), 3);
    assert_eq!(BlockHeader::payload(blocks[0]).as_ptr(), first.as_ptr());
    assert!(!BlockHeader::is_free(blocks[0]));
    assert_eq!(BlockHeader::payload(blocks[1]).as_ptr(), middle.as_ptr());
    assert!(!BlockHeader::is_free(blocks[1]));
    assert!(BlockHeader::is_free(blocks[2]));
    assert_eq!(BlockHeader::size(blocks[0]), 1024 + 16);
    assert_eq!(
        BlockHeader::size(blocks[2]),
        state.capacity() - 2 * (1024 + 16)
    );
    let _ = tail;
}

#[test]
fn weak_refs_to_dead_objects_are_cleared() {
    let (state, local, mut roots) = backend(64 * 1024, Roots::empty());
    let layout = Layout::new::<u64>();
    let live = local.allocate(layout).unwrap();
    let dead = local.allocate(layout).unwrap();
    roots.strong(live);
    let weak_dead = roots.weak(dead);
    let weak_live = roots.weak(live);

    local.collect();

    assert_eq!(roots.slots[weak_dead].load(), CLEARED);
    assert_eq!(roots.slots[weak_live].load(), weak_word(live));
    assert_eq!(state.stats().used, 32);
}

fn weak_word(target: NonNull<u8>) -> Word {
    target.as_ptr() as Word | WEAK_PTR
}

#[test]
fn block_walk_covers_arena_exactly() {
    let (state, local, mut roots) = backend(64 * 1024, Roots::empty());
    let layout = Layout::from_size_align(48, 8).unwrap();
    for i in 0..5 {
        if i % 2 == 0 {
            let ptr = local.allocate(layout).unwrap();
            roots.strong(ptr);
        } else {
            local.allocate(layout).unwrap();
        }
    }
    local.collect();
    local.collect();

    let mut total = 0;
    for b in block::blocks(state.base(), state.capacity()) {
        let size = BlockHeader::size(b);
        assert_eq!(size % 16, 0);
        total += size;
    }
    assert_eq!(total, state.capacity());
}
