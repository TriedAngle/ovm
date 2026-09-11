use core::alloc::Layout;
use core::ptr::NonNull;

use heap_api::{
    AllocError, GcHost, HeapBackend, RawCell, TAG_MASK, Visitor, Word, CLEARED, STRONG_PTR,
    WEAK_PTR,
};

use mark_sweep::heap::{MarkSweep, MarkSweepConfig, MarkSweepLocal, MarkSweepState};

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

static SIZES: LazyLock<Mutex<HashMap<usize, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn register(addr: NonNull<u8>, size: usize) {
    SIZES.lock().unwrap().insert(addr.as_ptr() as usize, size);
}

fn registered_size(addr: usize) -> usize {
    *SIZES.lock()
        .unwrap()
        .get(&addr)
        .expect("layout requested for unregistered object")
}

struct Roots {
    slots: Vec<RawCell>,
}

impl Roots {
    fn empty() -> Self {
        Self {
            slots: Vec::new(),
        }
    }

    fn install(self, state: &MarkSweepState) -> Box<Self> {
        self.install_with(state, host_for)
    }

    fn install_linked(self, state: &MarkSweepState) -> Box<Self> {
        self.install_with(state, linked_host_for)
    }

    fn install_with(
        self,
        state: &MarkSweepState,
        host: fn(&Roots) -> GcHost,
    ) -> Box<Self> {
        let boxed = Box::new(self);
        state.set_host(host(boxed.as_ref()));
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

/// Allocates and roots; registered so the fake `layout_of` can size the
/// object during the sweep.
fn rooted(
    local: &MarkSweepLocal,
    roots: &mut Roots,
    layout: Layout,
) -> NonNull<u8> {
    let ptr = local.allocate(layout).unwrap();
    register(ptr, layout.size());
    roots.strong(ptr);
    ptr
}

fn host_with(roots: &Roots, visit_object: fn(NonNull<()>, &mut dyn Visitor)) -> GcHost {
    fn visit_roots(ctx: *const (), visitor: &mut dyn Visitor) {
        let roots = unsafe { &*(ctx as *const Roots) };
        for slot in &roots.slots {
            visitor.visit(slot);
        }
    }
    fn layout_of(addr: NonNull<()>) -> Layout {
        let size = registered_size(addr.as_ptr() as usize);
        Layout::from_size_align(size, 8).unwrap()
    }
    GcHost {
        ctx: roots as *const Roots as *const (),
        visit_roots,
        layout_of,
        visit_object,
    }
}

fn host_for(roots: &Roots) -> GcHost {
    fn visit_object(_addr: NonNull<()>, _visitor: &mut dyn Visitor) {}
    host_with(roots, visit_object)
}

/// Traces the first payload word as a tagged next pointer, so chains built
/// by the test form a real object graph.
fn linked_host_for(roots: &Roots) -> GcHost {
    fn visit_object(addr: NonNull<()>, visitor: &mut dyn Visitor) {
        let word = unsafe { *(addr.as_ptr() as *const Word) };
        if word & TAG_MASK == STRONG_PTR {
            let cell = unsafe { RawCell::from_word(word) };
            visitor.visit(&cell);
        }
    }
    host_with(roots, visit_object)
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
    let ms = MarkSweep::new(MarkSweepConfig { heap_size: 64 * 1024 }).unwrap();
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
    let a = local.allocate(Layout::from_size_align(5, 8).unwrap()).unwrap();
    let b = local.allocate(Layout::new::<u64>()).unwrap();
    assert_eq!(a.as_ptr() as usize % 16, 0);
    assert_eq!(b.as_ptr() as usize % 16, 0);
    assert_ne!(a, b);
    // both served from one 32KB tlab refill
    assert_eq!(state.stats().used, 32 * 1024);
    // tlab bumps are sequential
    let diff = b.as_ptr() as usize - a.as_ptr() as usize;
    assert_eq!(diff, 16);
}

#[test]
fn garbage_is_reclaimed_and_coalesced() {
    let (state, local, _roots) = backend(64 * 1024, Roots::empty());
    for _ in 0..16 {
        local
            .allocate(Layout::from_size_align(512, 8).unwrap())
            .unwrap();
    }
    assert_eq!(state.stats().used, 32 * 1024);

    local.collect();

    assert_eq!(state.stats().used, 0);
    // the whole arena must be one free run again
    let big = local
        .allocate(Layout::from_size_align(60 * 1024, 8).unwrap())
        .unwrap();
    assert_eq!(state.stats().used, 60 * 1024);
    let _ = big;
}

#[test]
fn oom_when_live_data_exhausts_arena() {
    let (state, local, mut roots) = backend(8 * 1024, Roots::empty());
    let layout = Layout::new::<u64>();
    let mut live = Vec::new();
    loop {
        match local.allocate(layout) {
            Ok(ptr) => {
                register(ptr, 8);
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
    local.allocate(layout).unwrap();
}

#[test]
fn collect_keeps_rooted_objects_only() {
    let (state, local, mut roots) = backend(64 * 1024, Roots::empty());
    // 16KB objects bypass the tlab (direct free-list allocations)
    let layout = Layout::from_size_align(16 * 1024, 8).unwrap();
    let a = rooted(&local, &mut roots, layout);
    let b = local.allocate(layout).unwrap();
    let c = rooted(&local, &mut roots, layout);
    let cycle_before = state.cycles();

    local.collect();

    assert_eq!(state.cycles(), cycle_before + 1);
    assert_eq!(state.stats().used, 2 * 16 * 1024);
    // rooted payloads sit at their original addresses, and the garbage in
    // between is the first free run: a fresh allocation reuses b's address
    let reused = local.allocate(layout).unwrap();
    assert_eq!(reused.as_ptr(), b.as_ptr());
    let _ = (a, c);
}

#[test]
fn sweep_coalesces_dead_block_with_surrounding_free_space() {
    let (state, local, mut roots) = backend(64 * 1024, Roots::empty());
    let layout = Layout::from_size_align(16 * 1024, 8).unwrap();
    rooted(&local, &mut roots, layout);
    rooted(&local, &mut roots, layout);
    let tail = local.allocate(layout).unwrap();

    local.collect();

    assert_eq!(state.stats().used, 2 * 16 * 1024);
    // the first free run starts where the dead tail began
    let next = local.allocate(layout).unwrap();
    assert_eq!(next.as_ptr(), tail.as_ptr());
    // the tail merged with all trailing free space: everything but the two
    // live objects fits in one allocation
    let big = local
        .allocate(Layout::from_size_align(state.stats().capacity - 3 * 16 * 1024, 8).unwrap())
        .unwrap();
    assert_eq!(big.as_ptr() as usize, tail.as_ptr() as usize + 16 * 1024);
    assert_eq!(state.stats().used, state.stats().capacity);
}

#[test]
fn weak_refs_to_dead_objects_are_cleared() {
    let (state, local, mut roots) = backend(64 * 1024, Roots::empty());
    let layout = Layout::new::<u64>();
    let live = rooted(&local, &mut roots, layout);
    let dead = local.allocate(layout).unwrap();
    let weak_dead = roots.weak(dead);
    let weak_live = roots.weak(live);

    local.collect();

    assert_eq!(roots.slots[weak_dead].load(), CLEARED);
    assert_eq!(roots.slots[weak_live].load(), weak_word(live));
    assert_eq!(state.stats().used, 16);
}

fn weak_word(target: NonNull<u8>) -> Word {
    target.as_ptr() as Word | WEAK_PTR
}

#[test]
fn used_plus_free_conserve_arena_after_cycles() {
    let (state, local, mut roots) = backend(64 * 1024, Roots::empty());
    let layout = Layout::from_size_align(48, 8).unwrap();
    let mut keepers = Vec::new();
    for i in 0..5 {
        if i % 2 == 0 {
            keepers.push(rooted(&local, &mut roots, layout));
        } else {
            local.allocate(layout).unwrap();
        }
    }
    local.collect();
    local.collect();

    assert_eq!(state.stats().used + state.free_bytes(), state.stats().capacity);
    for ptr in keepers {
        assert!(state.contains(ptr.as_ptr() as usize));
    }
}

#[test]
fn allocation_continues_across_external_cycles() {
    let state = MarkSweepState::new(MarkSweepConfig { heap_size: 64 * 1024 }).unwrap();
    let _roots = Roots::empty().install(&state);
    let running = Arc::new(AtomicBool::new(true));
    let allocations = Arc::new(AtomicUsize::new(0));

    let mut threads = Vec::new();
    for _ in 0..2 {
        let state = Arc::clone(&state);
        let running = Arc::clone(&running);
        let allocations = Arc::clone(&allocations);
        threads.push(std::thread::spawn(move || {
            let local = MarkSweepLocal::new(state);
            let layout = Layout::from_size_align(64, 8).unwrap();
            while running.load(Ordering::Relaxed) {
                local.allocate(layout).unwrap();
                allocations.fetch_add(1, Ordering::Relaxed);
            }
            local.collect();
        }));
    }
    let collector = {
        let state = Arc::clone(&state);
        let running = Arc::clone(&running);
        std::thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                state.collect_now();
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        })
    };
    std::thread::sleep(std::time::Duration::from_millis(50));
    running.store(false, Ordering::Relaxed);
    for t in threads {
        t.join().unwrap();
    }
    collector.join().unwrap();

    assert!(allocations.load(Ordering::Relaxed) > 0);
    state.collect_now();
    assert_eq!(state.stats().used + state.free_bytes(), state.stats().capacity);
}

#[test]
fn heap_grows_and_shrinks_by_chunks() {
    let (state, local, mut roots) = backend(1024 * 1024, Roots::empty());
    let layout = Layout::from_size_align(16 * 1024, 8).unwrap();
    assert_eq!(state.stats().capacity, 0);

    let mut keepers = Vec::new();
    for _ in 0..20 {
        keepers.push(rooted(&local, &mut roots, layout));
    }
    // 20 x 16KB live spans two 256KB chunks
    assert_eq!(state.active_chunks(), 2);
    assert_eq!(state.stats().capacity, 512 * 1024);

    roots.slots.clear();
    drop(keepers);
    local.collect();

    // every chunk emptied: decommitted back to the pool
    assert_eq!(state.active_chunks(), 0);
    assert_eq!(state.stats().capacity, 0);
    assert_eq!(state.stats().used, 0);

    // allocation re-activates a pooled chunk
    rooted(&local, &mut roots, layout);
    assert_eq!(state.active_chunks(), 1);
    assert_eq!(state.stats().capacity, 256 * 1024);
}

#[test]
fn sweep_clears_its_own_chunk_bitmap() {
    use mark_sweep::chunk::ChunkedHeap;

    fn chunk_bitmap_is_clear(heap: &ChunkedHeap, addr: usize) -> bool {
        let chunk = heap.chunk_of(addr);
        let base = chunk.base();
        (base..base + chunk.size())
            .step_by(16)
            .all(|granule| !chunk.bitmap.is_set(granule))
    }

    let mut heap = ChunkedHeap::new(64 * 1024).unwrap();
    let layout = Layout::from_size_align(48, 8).unwrap();
    let host = host_for(&Roots::empty());
    let mut objects = Vec::new();
    for _ in 0..4 {
        let (ptr, claimed) = heap.allocate(layout, None);
        assert!(claimed.is_none());
        let ptr = ptr.unwrap();
        register(ptr, layout.size());
        objects.push(ptr.as_ptr() as usize);
    }

    // simulate a mark phase: half the objects are live
    for &addr in &objects[..2] {
        assert!(heap.chunk_of(addr).bitmap.set(addr));
    }

    heap.flag_all_pending();
    while let Some(chunk) = heap.claim_pending() {
        let live_bytes = ChunkedHeap::sweep_claimed(&chunk, &host);
        heap.publish_swept(&chunk, live_bytes);
    }

    // every sweep consumed and cleared its own chunk's bits
    assert!(chunk_bitmap_is_clear(&heap, objects[0]));
    assert_eq!(heap.live_bytes(), 2 * 48);
}

fn build_chain(local: &MarkSweepLocal, layout: Layout, len: usize) -> Word {
    let mut head: Word = 0;
    for _ in 0..len {
        let node = local.allocate(layout).unwrap();
        register(node, layout.size());
        unsafe { *(node.as_ptr() as *mut Word) = head };
        head = node.as_ptr() as Word | STRONG_PTR;
    }
    head
}

#[test]
fn marking_traces_linked_chains_exactly() {
    const CHAINS: usize = 512;
    const CHAIN_LEN: usize = 256;
    let state = MarkSweepState::new(MarkSweepConfig { heap_size: 8 * 1024 * 1024 }).unwrap();
    let mut roots = Roots::empty();
    let local = MarkSweepLocal::new(Arc::clone(&state));
    let layout = Layout::from_size_align(16, 8).unwrap();

    for _ in 0..CHAINS {
        let head = build_chain(&local, layout, CHAIN_LEN);
        roots.slots.push(unsafe { RawCell::from_word(head) });
    }
    let mut roots = roots.install_linked(&state);

    local.collect();

    assert_eq!(state.stats().used, CHAINS * CHAIN_LEN * 16);

    roots.slots.clear();
    local.collect();

    assert_eq!(state.stats().used, 0);
}

#[test]
fn parked_mutators_assist_marking() {
    const CHAINS: usize = 512;
    const CHAIN_LEN: usize = 256;
    let state = MarkSweepState::new(MarkSweepConfig { heap_size: 8 * 1024 * 1024 }).unwrap();
    let mut roots = Roots::empty();
    let local = MarkSweepLocal::new(Arc::clone(&state));
    let layout = Layout::from_size_align(16, 8).unwrap();

    for _ in 0..CHAINS {
        let head = build_chain(&local, layout, CHAIN_LEN);
        roots.slots.push(unsafe { RawCell::from_word(head) });
    }
    let _roots = roots.install_linked(&state);
    // the builder's node must not be part of upcoming cycles
    drop(local);

    // mutators poll the safepoint through allocation; their allocations
    // are unrooted garbage, dropped the moment they are made
    let running = Arc::new(AtomicBool::new(true));
    let mut threads = Vec::new();
    for _ in 0..3 {
        let state = Arc::clone(&state);
        let running = Arc::clone(&running);
        threads.push(std::thread::spawn(move || {
            let local = MarkSweepLocal::new(state);
            let layout = Layout::from_size_align(16, 8).unwrap();
            while running.load(Ordering::Relaxed) {
                local.allocate(layout).unwrap();
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        }));
    }
    std::thread::sleep(std::time::Duration::from_millis(50));

    for _ in 0..3 {
        state.collect_now();
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    running.store(false, Ordering::Relaxed);
    for t in threads {
        t.join().unwrap();
    }

    assert!(state.cycles() >= 3);
    assert!(state.mark_assists() > 0);

    state.collect_now();

    assert_eq!(state.stats().used, CHAINS * CHAIN_LEN * 16);
    assert_eq!(state.stats().used + state.free_bytes(), state.stats().capacity);
}
