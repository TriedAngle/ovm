use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

const PARKED: u8 = 0b01;
const REQUESTED: u8 = 0b10;

/// a node is:
/// - `PARKED` while its thread sleeps inside the barrier
/// - `REQUESTED` while an armed cycle waits for it to reach a safepoint.
/// - A running node with `REQUESTED` set is part of the armed cycle's target.
pub struct LocalNode {
    state: AtomicU8,
    linked: AtomicBool,
    prev: *mut LocalNode,
    next: *mut LocalNode,
}

unsafe impl Send for LocalNode {}
unsafe impl Sync for LocalNode {}

impl LocalNode {
    pub const fn detached() -> Self {
        Self {
            state: AtomicU8::new(0),
            linked: AtomicBool::new(false),
            prev: ptr::null_mut(),
            next: ptr::null_mut(),
        }
    }

    pub fn requested(&self) -> bool {
        self.state.load(Ordering::Relaxed) & REQUESTED != 0
    }
}

struct ListState {
    head: *mut LocalNode,
    count: usize,
}

struct BarrierState {
    armed: bool,
    stopped: usize,
    target: usize,
    work_generation: usize,
}

pub struct Safepoint {
    list: Mutex<ListState>,
    barrier: Mutex<BarrierState>,
    cond_stopped: Condvar,
    cond_resume: Condvar,
    work: OnceLock<Box<dyn Fn() + Send + Sync>>,
}

unsafe impl Send for Safepoint {}
unsafe impl std::marker::Sync for Safepoint {}

impl Default for Safepoint {
    fn default() -> Self {
        Self::new()
    }
}

impl Safepoint {
    pub const fn new() -> Self {
        Self {
            list: Mutex::new(ListState {
                head: ptr::null_mut(),
                count: 0,
            }),
            barrier: Mutex::new(BarrierState {
                armed: false,
                stopped: 0,
                target: 0,
                work_generation: 0,
            }),
            cond_stopped: Condvar::new(),
            cond_resume: Condvar::new(),
            work: OnceLock::new(),
        }
    }

    pub fn is_armed(&self) -> bool {
        self.barrier.lock().unwrap().armed
    }

    /// Installs the closure parked threads run each time the armer publishes
    /// work. Installed once, before any thread parks.
    pub fn set_work(&self, work: Box<dyn Fn() + Send + Sync>) {
        let _ = self.work.set(work);
    }

    /// Wakes parked threads to run the installed work closure.
    pub fn publish_work(&self) {
        let mut barrier = self.barrier.lock().unwrap();
        barrier.work_generation += 1;
        self.cond_resume.notify_all();
    }

    fn work_pending(&self, barrier: &MutexGuard<'_, BarrierState>, serviced: &mut usize) -> bool {
        if self.work.get().is_none() {
            return false;
        }
        let generation = barrier.work_generation;
        if generation == *serviced {
            return false;
        }
        *serviced = generation;
        true
    }

    pub fn attach(&self, node: &LocalNode) {
        debug_assert!(!node.linked.load(Ordering::Relaxed));
        let joined = {
            let mut list = self.list.lock().unwrap();
            let mut barrier = self.barrier.lock().unwrap();
            unsafe {
                let node = ptr::from_ref(node) as *mut LocalNode;
                (*node).prev = ptr::null_mut();
                (*node).next = list.head;
                if !list.head.is_null() {
                    (*list.head).prev = node;
                }
                list.head = node;
                (*node).linked.store(true, Ordering::Relaxed);
            }
            list.count += 1;
            if barrier.armed {
                // Relaxed: this thread owns the node and re-reads the bit
                // itself in `park_for_collection`; the barrier lock orders it
                // against the armer's `target` bookkeeping.
                node.state.fetch_or(REQUESTED, Ordering::Relaxed);
                barrier.target += 1;
                true
            } else {
                false
            }
        };
        if joined {
            self.park_for_collection(node);
        }
    }

    pub fn detach(&self, node: &LocalNode) {
        loop {
            if node.requested() {
                self.park_for_collection(node);
            }
            let mut list = self.list.lock().unwrap();
            if node.requested() {
                drop(list);
                continue;
            }
            debug_assert!(node.linked.load(Ordering::Relaxed));
            unsafe {
                let node = ptr::from_ref(node) as *mut LocalNode;
                if !(*node).prev.is_null() {
                    (*(*node).prev).next = (*node).next;
                } else {
                    list.head = (*node).next;
                }
                if !(*node).next.is_null() {
                    (*(*node).next).prev = (*node).prev;
                }
                (*node).prev = ptr::null_mut();
                (*node).next = ptr::null_mut();
                (*node).linked.store(false, Ordering::Relaxed);
            }
            list.count -= 1;
            return;
        }
    }

    pub fn park_for_collection(&self, node: &LocalNode) {
        loop {
            // Fast path: nothing pending. PARKED is only ever written by this
            // thread, so an exact `0` cannot hide our own parked state; a
            // stale REQUESTED only means the armer already counted us as a
            // target, and we will observe it at our next safepoint.
            if node.state.load(Ordering::Relaxed) == 0 {
                return;
            }
            // Mark ourselves parked. `prev` comes from an RMW, so it is the
            // current value: the decision below can never be made on a stale
            // REQUESTED read (the old load-then-act fast path could, and then
            // wait for a disarm that a counted target never performs).
            let prev = node.state.fetch_or(PARKED, Ordering::AcqRel);
            if prev & REQUESTED == 0 {
                // No cycle pending: leave the barrier. If one armed while we
                // were deciding, the CAS sees REQUESTED and fails, and we
                // re-evaluate from the fresh value.
                if node
                    .state
                    .compare_exchange(PARKED, 0, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return;
                }
                continue;
            }
            if prev & PARKED != 0 {
                self.wait_disarmed();
            } else {
                self.count_in_and_wait();
            }
            // PARKED stays set across the wait, then this loop clears it (or
            // handles a newly armed cycle).
        }
    }

    pub fn stop_the_world(&self, requester: Option<&LocalNode>, gc: impl FnOnce()) {
        {
            let list = self.list.lock().unwrap();
            let mut barrier = self.barrier.lock().unwrap();
            if !barrier.armed {
                debug_assert!(
                    requester
                        .map(|n| n.linked.load(Ordering::Relaxed))
                        .unwrap_or(true)
                );
                barrier.armed = true;
                barrier.stopped = 0;
                barrier.target = 0;
                let req = requester.map(ptr::from_ref);
                let mut node = list.head;
                while !node.is_null() {
                    let n = unsafe { &*node };
                    if Some(ptr::from_ref(n)) != req {
                        let old = n.state.fetch_or(REQUESTED, Ordering::AcqRel);
                        if old & PARKED == 0 {
                            barrier.target += 1;
                        }
                        debug_assert_eq!(old & REQUESTED, 0);
                    }
                    node = n.next;
                }
                drop(barrier);
                drop(list);
                let mut barrier = self.barrier.lock().unwrap();
                while barrier.stopped < barrier.target {
                    barrier = self.cond_stopped.wait(barrier).unwrap();
                }
                drop(barrier);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(gc));
                {
                    let list = self.list.lock().unwrap();
                    let mut barrier = self.barrier.lock().unwrap();
                    barrier.armed = false;
                    barrier.stopped = 0;
                    let mut node = list.head;
                    while !node.is_null() {
                        let n = unsafe { &*node };
                        n.state.fetch_and(!REQUESTED, Ordering::Release);
                        node = n.next;
                    }
                    drop(barrier);
                    self.cond_resume.notify_all();
                }
                if let Err(panic) = result {
                    std::panic::resume_unwind(panic);
                }
                return;
            }
        }
        if let Some(node) = requester {
            self.park_for_collection(node);
        }
    }

    fn count_in_and_wait(&self) {
        let mut barrier = self.barrier.lock().unwrap();
        barrier.stopped += 1;
        self.cond_stopped.notify_one();
        let mut serviced = 0;
        while barrier.armed {
            if self.work_pending(&barrier, &mut serviced) {
                let work = self.work.get().unwrap();
                drop(barrier);
                work();
                barrier = self.barrier.lock().unwrap();
                continue;
            }
            barrier = self.cond_resume.wait(barrier).unwrap();
        }
    }

    fn wait_disarmed(&self) {
        let mut barrier = self.barrier.lock().unwrap();
        let mut serviced = 0;
        while barrier.armed {
            if self.work_pending(&barrier, &mut serviced) {
                let work = self.work.get().unwrap();
                drop(barrier);
                work();
                barrier = self.barrier.lock().unwrap();
                continue;
            }
            barrier = self.cond_resume.wait(barrier).unwrap();
        }
    }
}
