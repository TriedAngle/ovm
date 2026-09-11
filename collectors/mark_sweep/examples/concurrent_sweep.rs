use core::alloc::Layout;
use core::ptr::NonNull;

use heap_api::{GcHost, Visitor};

use mark_sweep::heap::{MarkSweepConfig, MarkSweepLocal, MarkSweepState};

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn empty_host() -> GcHost {
    fn visit_roots(_ctx: *const (), _visitor: &mut dyn Visitor) {}
    fn visit_object(_addr: NonNull<()>, _visitor: &mut dyn Visitor) {}
    fn layout_of(_addr: NonNull<()>) -> Layout {
        Layout::from_size_align(16 * 1024, 8).unwrap()
    }
    GcHost {
        ctx: core::ptr::null(),
        visit_roots,
        layout_of,
        visit_object,
    }
}

fn main() {
    let state = MarkSweepState::new(MarkSweepConfig { heap_size: 64 * 1024 * 1024 }).unwrap();
    state.set_host(empty_host());

    let running = Arc::new(AtomicBool::new(true));
    let allocations = Arc::new(AtomicUsize::new(0));
    let during_sweeping = Arc::new(AtomicUsize::new(0));

    let mut mutators = Vec::new();
    for _ in 0..2 {
        let state = Arc::clone(&state);
        let running = Arc::clone(&running);
        let allocations = Arc::clone(&allocations);
        let during_sweeping = Arc::clone(&during_sweeping);
        mutators.push(std::thread::spawn(move || {
            let local = MarkSweepLocal::new(Arc::clone(&state));
            let layout = Layout::from_size_align(16 * 1024, 8).unwrap();
            let mut since_poll = 0;
            while running.load(Ordering::Relaxed) {
                local.allocate(layout).unwrap();
                allocations.fetch_add(1, Ordering::Relaxed);
                since_poll += 1;
                if since_poll == 32 {
                    since_poll = 0;
                    if state.pending_chunks() > 0 {
                        during_sweeping.fetch_add(32, Ordering::Relaxed);
                    }
                }
            }
        }));
    }
    let collector = {
        let state = Arc::clone(&state);
        let running = Arc::clone(&running);
        std::thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                state.collect_now();
                std::thread::sleep(Duration::from_millis(1));
            }
        })
    };

    std::thread::sleep(Duration::from_millis(300));
    running.store(false, Ordering::Relaxed);
    for mutator in mutators {
        mutator.join().unwrap();
    }
    collector.join().unwrap();
    state.collect_now();

    println!("cycles: {}", state.cycles());
    println!("allocations: {} (16KB each)", allocations.load(Ordering::Relaxed));
    println!(
        "allocations while sweeping was in flight: {}",
        during_sweeping.load(Ordering::Relaxed)
    );
    println!(
        "chunks swept by the background thread: {}",
        state.background_sweeps()
    );
    assert!(state.background_sweeps() > 0);
    assert!(during_sweeping.load(Ordering::Relaxed) > 0);
}
