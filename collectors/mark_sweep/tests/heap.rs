use core::alloc::Layout;
use core::ptr::NonNull;

use heap_api::{
    AllocError, GcHost, HeapBackend, RawCell, Visitor, Word, CLEARED, STRONG_PTR, WEAK_PTR,
};

use mark_sweep::heap::{MarkSweep, MarkSweepConfig, MarkSweepLocal, MarkSweepState};

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

thread_local! {
    static SIZES: RefCell<Vec<(usize, usize)>> = RefCell::new(Vec::new());
}

fn register(addr: NonNull<u8>, size: usize) {
    SIZES.with(|sizes| sizes.borrow_mut().push((addr.as_ptr() as usize, size)));
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

fn host_for(roots: &Roots) -> GcHost {
    fn visit_roots(ctx: *const (), visitor: &mut dyn Visitor) {
        let roots = unsafe { &*(ctx as *const Roots) };
        for slot in &roots.slots {
            visitor.visit(slot);
        }
    }
    fn visit_object(_addr: NonNull<()>, _visitor: &mut dyn Visitor) {}
    fn layout_of(addr: NonNull<()>) -> Layout {
        let addr = addr.as_ptr() as usize;
        let size = SIZES
            .with(|sizes| sizes.borrow().iter().find(|(a, _)| *a == addr).map(|(_, s)| *s))
            .expect("layout requested for unregistered object");
        Layout::from_size_align(size, 8).unwrap()
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
        .allocate(Layout::from_size_align(state.capacity() - 3 * 16 * 1024, 8).unwrap())
        .unwrap();
    assert_eq!(big.as_ptr() as usize, tail.as_ptr() as usize + 16 * 1024);
    assert_eq!(state.stats().used, state.capacity());
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

    assert_eq!(state.stats().used + state.free_bytes(), state.capacity());
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
    assert_eq!(state.stats().used + state.free_bytes(), state.capacity());
}
