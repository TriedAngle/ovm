use heap_utils::{LocalNode, Safepoint};
use std::sync::atomic::{AtomicUsize, Ordering};

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
