use core::alloc::Layout;
use core::ptr::NonNull;

use heap_api::{CLEARED, GcHost, RawCell, STRONG_PTR, TAG_MASK, Visitor, WEAK_PTR, Word};

use mark_sweep::heap::{MarkSweepConfig, MarkSweepLocal, MarkSweepState};

use std::sync::Arc;

/// Node size for the fake host: fixed, so `layout_of` works identically on
/// originals and evacuated copies.
const NODE_SIZE: usize = 32;

struct Roots {
    slots: Vec<RawCell>,
}

impl Roots {
    fn empty() -> Self {
        Self { slots: Vec::new() }
    }

    fn install(self, state: &MarkSweepState) -> Box<Self> {
        let boxed = Box::new(self);
        state.set_host(GcHost {
            ctx: boxed.as_ref() as *const Roots as *const (),
            visit_roots,
            layout_of,
            visit_object,
        });
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

fn visit_roots(ctx: *const (), visitor: &mut dyn Visitor) {
    let roots = unsafe { &*(ctx as *const Roots) };
    for slot in &roots.slots {
        visitor.visit(slot);
    }
}

fn layout_of(_addr: NonNull<()>) -> Layout {
    Layout::from_size_align(NODE_SIZE, 8).unwrap()
}

/// Traces the first payload word as a tagged link so chains form a real
/// object graph. Passes the in-heap cell so moving visitors can rewrite it.
fn visit_object(addr: NonNull<()>, visitor: &mut dyn Visitor) {
    let cell = unsafe { &*(addr.as_ptr().cast::<Word>().add(1) as *const RawCell) };
    if cell.load() & TAG_MASK == STRONG_PTR {
        visitor.visit(cell);
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

/// Allocates a node whose link word is `link`.
fn node(local: &MarkSweepLocal, link: Word) -> NonNull<u8> {
    let ptr = local.allocate(layout_of(NonNull::dangling())).unwrap();
    unsafe { *(ptr.as_ptr().cast::<Word>().add(1)) = link };
    ptr
}

fn link_word(ptr: NonNull<u8>) -> Word {
    unsafe { *(ptr.as_ptr().cast::<Word>().add(1)) }
}

fn strong_word(ptr: NonNull<u8>) -> Word {
    ptr.as_ptr() as Word | STRONG_PTR
}

fn link_cell(ptr: NonNull<u8>) -> &'static RawCell {
    unsafe { &*(ptr.as_ptr().cast::<Word>().add(1) as *const RawCell) }
}

#[test]
fn minor_moves_rooted_survivors_and_frees_young_chunks() {
    let (state, local, mut roots) = backend(8 * 1024 * 1024, Roots::empty());
    let a = node(&local, 0);
    let b = node(&local, strong_word(a));
    roots.strong(a);
    roots.strong(b);

    let before_b = roots.slots[1].load();
    local.collect_minor();

    assert_eq!(state.minor_cycles(), 1);
    // both survivors were evacuated: the root cells were rewritten
    let moved_b = roots.slots[1].load();
    assert_ne!(moved_b, before_b);
    assert!(state.contains((moved_b & !TAG_MASK) as usize));
    // b's link followed a to its new address
    let moved_a = roots.slots[0].load();
    assert_eq!(
        link_word(NonNull::new((moved_b & !TAG_MASK) as *mut u8).unwrap()),
        moved_a
    );
    let _ = moved_a;
}

#[test]
fn remembered_edge_survives_full_gc() {
    let (state, local, mut roots) = backend(8 * 1024 * 1024, Roots::empty());

    // promote `old` out of the young set with one minor cycle
    let old = node(&local, 0);
    roots.strong(old);
    local.collect_minor();
    let old_moved = (roots.slots[0].load() & !TAG_MASK) as usize;
    let old_ptr = NonNull::new(old_moved as *mut u8).unwrap();

    // a young target referenced ONLY by the old node's link slot, recorded
    // by the write barrier
    let young = node(&local, 0);
    let young_word = strong_word(young);
    unsafe { *(old_moved as *mut Word).add(1) = young_word };
    state.write_barrier(link_cell(old_ptr), young_word);

    // a full collection must keep the recorded entry intact
    local.collect();
    local.collect_minor();

    let updated = link_word(old_ptr);
    assert_ne!(updated, young_word);
    assert!(state.contains((updated & !TAG_MASK) as usize));
    assert_eq!(state.minor_cycles(), 2);
}

#[test]
fn write_barrier_keeps_old_to_young_target_alive() {
    let (state, local, mut roots) = backend(8 * 1024 * 1024, Roots::empty());

    // promote `old` out of the young set with one minor cycle
    let old = node(&local, 0);
    roots.strong(old);
    local.collect_minor();
    let old_moved = (roots.slots[0].load() & !TAG_MASK) as usize;
    let old_ptr = NonNull::new(old_moved as *mut u8).unwrap();

    // a young target referenced ONLY by the old node's link slot, recorded
    // by the write barrier exactly like GcSlot::set would
    let young = node(&local, 0);
    let young_word = strong_word(young);
    unsafe { *(old_moved as *mut Word).add(1) = young_word };
    state.write_barrier(link_cell(old_ptr), young_word);

    local.collect_minor();

    // the remembered slot was found and rewritten to the evacuated copy
    let updated = link_word(old_ptr);
    assert_ne!(updated, young_word);
    assert!(state.contains((updated & !TAG_MASK) as usize));
    assert_eq!(state.minor_cycles(), 2);
}

#[test]
fn weak_young_refs_settle_after_the_closure() {
    let (state, local, mut roots) = backend(8 * 1024 * 1024, Roots::empty());

    let strongly_rooted = node(&local, 0);
    let before = strong_word(strongly_rooted);
    roots.strong(strongly_rooted);
    let weak_alive = roots.weak(strongly_rooted);
    let weak_dead = roots.weak(node(&local, 0));

    local.collect_minor();

    // the target really moved, and the weak cell followed it to the new
    // address (kept alive by the separate strong root, not by the cell)
    assert_ne!(roots.slots[0].load(), before);
    assert_eq!(
        roots.slots[weak_alive].load(),
        roots.slots[0].load() | WEAK_PTR
    );
    // never evacuated: the weak-only target died with its young chunk
    assert_eq!(roots.slots[weak_dead].load(), CLEARED);
    let _ = state;
}

#[test]
fn chains_relocate_consistently() {
    let (state, local, mut roots) = backend(8 * 1024 * 1024, Roots::empty());
    const LEN: usize = 512;
    let mut link: Word = 0;
    for _ in 0..LEN {
        link = strong_word(node(&local, link));
    }
    roots.slots.push(unsafe { RawCell::from_word(link) });

    local.collect_minor();

    // walk the relocated chain end to end
    let mut word = roots.slots[0].load();
    let mut count = 0usize;
    while word & TAG_MASK == STRONG_PTR {
        let ptr = NonNull::new((word & !TAG_MASK) as *mut u8).unwrap();
        assert!(state.contains(ptr.as_ptr() as usize));
        word = link_word(ptr);
        count += 1;
    }
    assert_eq!(count, LEN);
}

#[test]
fn minor_then_full_conserves_arena() {
    let (state, local, mut roots) = backend(8 * 1024 * 1024, Roots::empty());
    let keep = node(&local, 0);
    roots.strong(keep);
    for _ in 0..64 {
        node(&local, 0);
    }

    local.collect_minor();
    local.collect();

    assert!(state.minor_cycles() >= 1);
    assert_eq!(
        state.stats().used + state.free_bytes(),
        state.stats().capacity
    );
    assert_eq!(state.stats().used, NODE_SIZE.next_multiple_of(16));
}

#[test]
fn minors_fire_from_the_allocation_path() {
    let (state, local, _roots) = backend(8 * 1024 * 1024, Roots::empty());
    // pure garbage: every minor empties the young set again. Several MiB
    // of churn so the young-set budget (chunks, not bytes) is crossed.
    for _ in 0..(2 * 1024 * 1024 / NODE_SIZE) {
        node(&local, 0);
    }
    assert!(state.minor_cycles() > 0);
    // and the heap did not grow to hold all the garbage
    assert!(state.active_chunks() < 4);
}

#[test]
fn starved_minor_runs_full_collection_instead() {
    // two chunks of rooted survivors: no fresh chunks exist for promotion,
    // so the minor's pre-flight gate must decline to evacuate and run a
    // full (non-moving) collection instead — before any partial state
    let (state, local, mut roots) = backend(512 * 1024, Roots::empty());
    // rooted nodes filling both chunks' usable regions exactly
    // (allocation-path minors starve the same way during the loop)
    let fill = 2 * (256 * 1024 - mark_sweep::chunk::HEADER_SIZE);
    for _ in 0..(fill / NODE_SIZE) {
        roots.strong(node(&local, 0));
    }
    assert_eq!(state.active_chunks(), 2);

    let keepers: Vec<Word> = roots.slots.iter().map(|c| c.load()).collect();

    local.collect_minor();

    assert_eq!(state.minor_cycles(), 0);
    assert!(state.cycles() >= 1);
    // the full collection was non-moving: every rooted object stayed put
    for (i, word) in keepers.iter().enumerate() {
        assert_eq!(roots.slots[i].load(), *word);
    }
    assert_eq!(state.active_chunks(), 2);
}
