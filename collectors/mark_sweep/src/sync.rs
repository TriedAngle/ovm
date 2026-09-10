use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Condvar, Mutex};

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
}

pub struct Safepoint {
    list: Mutex<ListState>,
    barrier: Mutex<BarrierState>,
    cond_stopped: Condvar,
    cond_resume: Condvar,
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
            }),
            cond_stopped: Condvar::new(),
            cond_resume: Condvar::new(),
        }
    }

    pub fn is_armed(&self) -> bool {
        self.barrier.lock().unwrap().armed
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
            let state = node.state.load(Ordering::Relaxed);
            if state & REQUESTED == 0 {
                if state & PARKED == 0 {
                    return;
                }
                let current = node.state.fetch_and(!PARKED, Ordering::Relaxed);
                if current & REQUESTED != 0 {
                    self.wait_disarmed();
                }
                continue;
            }
            if state & PARKED != 0 {
                self.wait_disarmed();
                continue;
            }
            let old = node.state.fetch_or(PARKED, Ordering::Relaxed);
            if old & REQUESTED != 0 {
                self.count_in_and_wait();
            }
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
                        let old = n.state.fetch_or(REQUESTED, Ordering::Relaxed);
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
                        n.state.fetch_and(!REQUESTED, Ordering::Relaxed);
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
        while barrier.armed {
            barrier = self.cond_resume.wait(barrier).unwrap();
        }
    }

    fn wait_disarmed(&self) {
        let mut barrier = self.barrier.lock().unwrap();
        while barrier.armed {
            barrier = self.cond_resume.wait(barrier).unwrap();
        }
    }
}
