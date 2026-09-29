use heap_utils::{LocalNode, Safepoint};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

fn spin_until(mut f: impl FnMut() -> bool) {
    while !f() {
        std::thread::yield_now();
    }
}

#[test]
fn single_node_cycle_runs_gc_immediately() {
    let sync = Safepoint::new();
    let node = LocalNode::detached();
    sync.attach(&node);

    let ran = AtomicUsize::new(0);
    sync.stop_the_world(Some(&node), || {
        ran.fetch_add(1, Ordering::Relaxed);
    });

    assert_eq!(ran.load(Ordering::Relaxed), 1);
    assert!(!sync.is_armed());
    assert!(!node.requested());
    sync.detach(&node);
}

#[test]
fn mutators_park_before_gc_runs() {
    let sync = Safepoint::new();
    let attached = AtomicUsize::new(0);
    let ran = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..3 {
            s.spawn(|| {
                let node = LocalNode::detached();
                sync.attach(&node);
                attached.fetch_add(1, Ordering::Relaxed);
                spin_until(|| node.requested());
                sync.park_for_collection(&node);
            });
        }
        spin_until(|| attached.load(Ordering::Relaxed) == 3);
        s.spawn(|| {
            sync.stop_the_world(None, || {
                ran.fetch_add(1, Ordering::Relaxed);
            });
        });
    });

    assert_eq!(ran.load(Ordering::Relaxed), 1);
    assert!(!sync.is_armed());
}

#[test]
fn detach_without_parking_completes_cycle() {
    let sync = Safepoint::new();
    let node = LocalNode::detached();
    sync.attach(&node);

    let ran = AtomicUsize::new(0);
    std::thread::scope(|s| {
        let leaver = s.spawn(|| {
            spin_until(|| node.requested());
            sync.detach(&node);
        });
        sync.stop_the_world(None, || {
            ran.fetch_add(1, Ordering::Relaxed);
        });
        leaver.join().unwrap();
    });

    assert_eq!(ran.load(Ordering::Relaxed), 1);
    assert!(!sync.is_armed());
}

#[test]
fn attach_during_cycle_joins_and_is_released() {
    let sync = Safepoint::new();
    let first = LocalNode::detached();
    sync.attach(&first);
    let late = LocalNode::detached();

    let ran = AtomicUsize::new(0);
    std::thread::scope(|s| {
        s.spawn(|| {
            sync.stop_the_world(None, || {
                ran.fetch_add(1, Ordering::Relaxed);
            });
        });
        spin_until(|| first.requested());

        s.spawn(|| {
            sync.attach(&late);
            assert!(!late.requested());
            sync.detach(&late);
        });
        // the cycle cannot complete before `first` parks, so `late` is
        // guaranteed to join the armed cycle
        spin_until(|| late.requested());
        sync.park_for_collection(&first);
    });

    assert_eq!(ran.load(Ordering::Relaxed), 1);
    sync.detach(&first);
}

#[test]
fn second_requester_parks_instead_of_running_gc() {
    let sync = Safepoint::new();
    let a = LocalNode::detached();
    let b = LocalNode::detached();
    sync.attach(&a);
    sync.attach(&b);

    let first_runs = AtomicUsize::new(0);
    let second_runs = AtomicUsize::new(0);
    std::thread::scope(|s| {
        s.spawn(|| {
            sync.stop_the_world(None, || {
                first_runs.fetch_add(1, Ordering::Relaxed);
            });
        });
        spin_until(|| b.requested());
        s.spawn(|| {
            sync.stop_the_world(Some(&b), || {
                second_runs.fetch_add(1, Ordering::Relaxed);
            });
        });
        sync.park_for_collection(&a);
    });

    assert_eq!(first_runs.load(Ordering::Relaxed), 1);
    assert_eq!(second_runs.load(Ordering::Relaxed), 0);
    sync.detach(&a);
    sync.detach(&b);
}

#[test]
fn panicking_gc_still_disarms() {
    let sync = Safepoint::new();
    let node = LocalNode::detached();
    sync.attach(&node);

    std::thread::scope(|s| {
        s.spawn(|| {
            spin_until(|| node.requested());
            sync.park_for_collection(&node);
        });
        let panicked = s.spawn(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                sync.stop_the_world(None, || panic!("gc exploded"));
            }))
            .unwrap_err();
        });
        assert!(panicked.join().is_ok());
    });

    assert!(!sync.is_armed());
    assert!(!node.requested());
    sync.detach(&node);
}

#[test]
fn churn_stress() {
    let sync = Safepoint::new();
    let cycles = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                let node = LocalNode::detached();
                sync.attach(&node);
                for _ in 0..2000 {
                    sync.park_for_collection(&node);
                    std::thread::yield_now();
                }
                sync.detach(&node);
            });
        }
        s.spawn(|| {
            for _ in 0..100 {
                sync.stop_the_world(None, || {
                    cycles.fetch_add(1, Ordering::Relaxed);
                });
            }
        });
    });

    assert_eq!(cycles.load(Ordering::Relaxed), 100);
}

#[test]
fn cancel_executions_releases_threads_into_halt() {
    let sync = Safepoint::new();
    let requester = LocalNode::detached();
    sync.attach(&requester);

    let attached = AtomicUsize::new(0);
    let cancelled = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..2 {
            s.spawn(|| {
                let node = LocalNode::detached();
                sync.attach(&node);
                attached.fetch_add(1, Ordering::Relaxed);
                // a mutator: run until requested, then pause
                spin_until(|| node.requested());
                let paused = sync.park_for_collection(&node);
                // the pause carried the cancel; taking it consumes it
                let took = node.take_cancel();
                let took_again = node.take_cancel();
                cancelled.fetch_add((paused && took && !took_again) as usize, Ordering::Relaxed);
                // a cancelled node still detaches cleanly
                sync.detach(&node);
            });
        }
        spin_until(|| attached.load(Ordering::Relaxed) == 2);

        let protocol = AtomicUsize::new(0);
        sync.cancel_executions(&requester, || {
            protocol.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(protocol.load(Ordering::Relaxed), 1);
        assert!(!sync.is_armed());
        // the requester itself carries no cancel
        assert!(!requester.pending());
        // cancelling again is a no-op cycle, not an error
        sync.cancel_executions(&requester, || {});

        spin_until(|| cancelled.load(Ordering::Relaxed) == 2);
    });
}

#[test]
fn cancel_survives_collection_cycles() {
    let sync = Safepoint::new();
    let requester = LocalNode::detached();
    let node = LocalNode::detached();
    sync.attach(&requester);

    let cancelled = AtomicUsize::new(0);
    let attached = AtomicBool::new(false);
    std::thread::scope(|s| {
        s.spawn(|| {
            sync.attach(&node);
            attached.store(true, Ordering::Relaxed);
            spin_until(|| node.requested());
            // first pause: the cancel cycle itself (the cancel is NOT
            // taken yet — it must survive what comes next)
            let paused = sync.park_for_collection(&node);
            cancelled.fetch_add(paused as usize, Ordering::Relaxed);
            // an ordinary collection cycle arrives: the thread parks for
            // it, wakes, and the cancel is still observable afterwards
            spin_until(|| sync.is_armed());
            let paused_again = sync.park_for_collection(&node);
            let took = node.take_cancel();
            cancelled.fetch_add((paused_again && took) as usize, Ordering::Relaxed);
            sync.detach(&node);
        });
        // the cancel protocol completes once the mutator paused once...
        spin_until(|| attached.load(Ordering::Relaxed));
        sync.cancel_executions(&requester, || {});
        spin_until(|| cancelled.load(Ordering::Relaxed) == 1);
        // ...and an ordinary cycle right after it runs normally (the
        // pending cancel does not block barrier traffic) while the mutator
        // is between pauses
        let cycles = AtomicUsize::new(0);
        sync.stop_the_world(Some(&requester), || {
            cycles.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(cycles.load(Ordering::Relaxed), 1);
        spin_until(|| cancelled.load(Ordering::Relaxed) == 2);
    });

    sync.detach(&requester);
}

#[test]
fn attach_after_cancel_starts_fresh() {
    let sync = Safepoint::new();
    let requester = LocalNode::detached();
    sync.attach(&requester);
    sync.cancel_executions(&requester, || {});

    // a fresh node is not cancelled: shutdown only kills executions that
    // were attached when the protocol ran
    let paused = AtomicUsize::new(1);
    std::thread::scope(|s| {
        s.spawn(|| {
            let node = LocalNode::detached();
            sync.attach(&node);
            let paused_here = sync.park_for_collection(&node);
            paused.store(paused_here as usize, Ordering::Relaxed);
            sync.detach(&node);
        });
    });

    assert_eq!(paused.load(Ordering::Relaxed), 0);
    sync.detach(&requester);
}
