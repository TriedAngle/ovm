use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::{self, NonNull};

use heap_api::{
    AllocError, CLEARED, GcHost, GlobalVtable, HeapBackend, HeapStats, HeapVtable, RawCell,
    STRONG_PTR, TAG_MASK, Visitor, WEAK_PTR, Word,
};

use heap_utils::{LocalNode, Safepoint};

use crate::block::{need_for, ALIGN};
use crate::chunk::ChunkedHeap;

#[derive(Debug, Clone, Copy)]
pub struct MarkSweepConfig {
    pub heap_size: usize,
}

impl Default for MarkSweepConfig {
    fn default() -> Self {
        Self {
            heap_size: 1024 * 1024 * 1024,
        }
    }
}

pub struct MarkSweepState {
    alloc: Mutex<ChunkedHeap>,
    safepoint: Safepoint,
    host: Mutex<Option<GcHost>>,
    mark_job: Mutex<Option<Arc<MarkJob>>>,
    mark_assists: AtomicUsize,
    cycles: AtomicUsize,
    gc_threshold: AtomicUsize,
}

unsafe impl Send for MarkSweepState {}
unsafe impl Sync for MarkSweepState {}

impl MarkSweepState {
    pub fn new(config: MarkSweepConfig) -> Result<Arc<Self>, AllocError> {
        let this = Arc::new(Self {
            alloc: Mutex::new(ChunkedHeap::new(config.heap_size)?),
            safepoint: Safepoint::new(),
            host: Mutex::new(None),
            mark_job: Mutex::new(None),
            mark_assists: AtomicUsize::new(0),
            cycles: AtomicUsize::new(0),
            gc_threshold: AtomicUsize::new(MIN_GC_THRESHOLD),
        });
        let weak = Arc::downgrade(&this);
        this.safepoint.set_work(Box::new(move || {
            let Some(state) = weak.upgrade() else { return };
            let Some(job) = state.mark_job.lock().unwrap().clone() else {
                return;
            };
            if !job.done.load(Ordering::Acquire) {
                state.mark_assists.fetch_add(1, Ordering::Relaxed);
                join_marking(&job, false);
            }
        }));
        Ok(this)
    }

    pub fn set_host(&self, host: GcHost) {
        *self.host.lock().unwrap() = Some(host);
    }

    pub fn cycles(&self) -> usize {
        self.cycles.load(Ordering::Relaxed)
    }

    pub fn mark_assists(&self) -> usize {
        self.mark_assists.load(Ordering::Relaxed)
    }

    pub fn active_chunks(&self) -> usize {
        self.alloc.lock().unwrap().active_chunks()
    }

    pub fn contains(&self, addr: usize) -> bool {
        self.alloc.lock().unwrap().in_heap(addr)
    }

    pub fn stats(&self) -> HeapStats {
        let alloc = self.alloc.lock().unwrap();
        HeapStats {
            used: alloc.live_bytes(),
            capacity: alloc.committed_bytes(),
        }
    }

    pub fn free_bytes(&self) -> usize {
        self.alloc.lock().unwrap().free_bytes()
    }

    pub fn collect_now(&self) {
        self.collect_internal(None);
    }

    fn finish_sweeping(&self, host: &GcHost) {
        {
            let mut heap = self.alloc.lock().unwrap();
            heap.set_sweep_block(true);
        }
        loop {
            let ready = {
                let mut heap = self.alloc.lock().unwrap();
                heap.finish_pending(host)
            };
            if ready {
                return;
            }
            std::thread::yield_now();
        }
    }

    fn collect_internal(&self, requester: Option<&MarkSweepLocal>) {
        let host = self.host();
        self.safepoint
            .stop_the_world(requester.map(|local| &local.node), || {
                self.finish_sweeping(&host);
                let completed = {
                    let mut heap = self.alloc.lock().unwrap();
                    self.mark(&heap, &host);
                    self.clear_weaks(&heap, &host);
                    heap.flag_all_pending();
                    heap.set_sweep_block(false);
                    heap.take_completed_live()
                };
                if let Some(live) = completed {
                    self.update_threshold(live);
                }
            });
        self.cycles.fetch_add(1, Ordering::Relaxed);
        self.sweep_pending(&host, requester);
    }

    fn sweep_pending(&self, host: &GcHost, requester: Option<&MarkSweepLocal>) {
        loop {
            let claimed = self.alloc.lock().unwrap().claim_pending();
            let Some(chunk) = claimed else { return };
            let live = ChunkedHeap::sweep_claimed(&chunk, host);
            let completed = {
                let mut heap = self.alloc.lock().unwrap();
                heap.publish_swept(&chunk, live);
                heap.take_completed_live()
            };
            if let Some(live) = completed {
                self.update_threshold(live);
            }
            if let Some(local) = requester {
                local.park_if_requested();
            }
        }
    }

    fn update_threshold(&self, live: usize) {
        let reserve = self.alloc.lock().unwrap().reserve_size();
        let threshold = (live * 2).max(MIN_GC_THRESHOLD).min(reserve);
        self.gc_threshold.store(threshold, Ordering::Relaxed);
    }

    fn collect_terminal(
        &self,
        requester: Option<&LocalNode>,
        layout: Layout,
    ) -> TerminalAllocation {
        let host = self.host();
        let mut executed = false;
        let mut allocated = None;
        self.safepoint.stop_the_world(requester, || {
            executed = true;
            self.finish_sweeping(&host);
            let (ptr, completed) = {
                let mut heap = self.alloc.lock().unwrap();
                self.mark(&heap, &host);
                self.clear_weaks(&heap, &host);
                heap.flag_all_pending();
                heap.finish_pending(&host);
                let (ptr, _) = heap.allocate(layout, Some(&host));
                heap.set_sweep_block(false);
                (ptr, heap.take_completed_live())
            };
            if let Some(live) = completed {
                self.update_threshold(live);
            }
            allocated = ptr;
        });
        if executed {
            self.cycles.fetch_add(1, Ordering::Relaxed);
            TerminalAllocation::Done(allocated)
        } else {
            TerminalAllocation::Absorbed
        }
    }

    fn used_exceeds_threshold(&self) -> bool {
        self.alloc.lock().unwrap().live_bytes() > self.gc_threshold.load(Ordering::Relaxed)
    }

    fn alloc_block(&self, layout: Layout) -> Option<NonNull<u8>> {
        let host = self.try_host();
        loop {
            let (ptr, claimed, completed) = {
                let mut heap = self.alloc.lock().unwrap();
                let (ptr, claimed) = heap.allocate(layout, host.as_ref());
                (ptr, claimed, heap.take_completed_live())
            };
            if let Some(live) = completed {
                self.update_threshold(live);
            }
            let Some(ptr) = ptr else {
                let Some(chunk) = claimed else { return None };
                let host = host.expect("claimed pending chunk without host");
                let live = ChunkedHeap::sweep_claimed(&chunk, &host);
                let completed = {
                    let mut heap = self.alloc.lock().unwrap();
                    heap.publish_swept(&chunk, live);
                    heap.take_completed_live()
                };
                if let Some(live) = completed {
                    self.update_threshold(live);
                }
                continue;
            };
            return Some(ptr);
        }
    }

    fn try_host(&self) -> Option<GcHost> {
        self.host.lock().unwrap().as_ref().copied()
    }

    fn host(&self) -> GcHost {
        self.host
            .lock()
            .unwrap()
            .as_ref()
            .copied()
            .expect("host not registered before collection")
    }

    fn mark(&self, heap: &ChunkedHeap, host: &GcHost) {
        let job = Arc::new(MarkJob::new(heap, *host));
        *self.mark_job.lock().unwrap() = Some(Arc::clone(&job));
        self.safepoint.publish_work();
        {
            let mut scanner = RootScanner { job: &job };
            (host.visit_roots)(host.ctx, &mut scanner);
        }
        job.roots_scanned.store(true, Ordering::Release);
        self.safepoint.publish_work();
        join_marking(&job, true);
        *self.mark_job.lock().unwrap() = None;
    }

    fn clear_weaks(&self, heap: &ChunkedHeap, host: &GcHost) {
        let mut clearer = WeakClearer { heap };
        (host.visit_roots)(host.ctx, &mut clearer);
        heap.for_each_live(|addr| {
            let object = unsafe { NonNull::new_unchecked(addr as *mut ()) };
            (host.visit_object)(object, &mut clearer);
        });
    }
}

const MARK_BATCH: usize = 32;
const MARK_SPILL: usize = 512;
const ASSIST_SPINS: usize = 256;

struct MarkJob {
    heap: *const ChunkedHeap,
    host: GcHost,
    worklist: Mutex<Vec<usize>>,
    roots_scanned: AtomicBool,
    active: AtomicUsize,
    done: AtomicBool,
}

unsafe impl Send for MarkJob {}
unsafe impl Sync for MarkJob {}

impl MarkJob {
    fn new(heap: &ChunkedHeap, host: GcHost) -> Self {
        Self {
            heap,
            host,
            worklist: Mutex::new(Vec::new()),
            roots_scanned: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            done: AtomicBool::new(false),
        }
    }

    fn heap(&self) -> &ChunkedHeap {
        unsafe { &*self.heap }
    }

    fn claim(&self, cell: &RawCell) -> Option<usize> {
        let word = cell.load();
        if word & TAG_MASK != STRONG_PTR {
            return None;
        }
        let addr = (word & !TAG_MASK) as usize;
        let heap = self.heap();
        (heap.in_heap(addr) && heap.chunk_of(addr).bitmap.set(addr)).then_some(addr)
    }

    fn steal(&self, local: &mut Vec<usize>) {
        let mut worklist = self.worklist.lock().unwrap();
        let take = MARK_BATCH.min(worklist.len());
        if take > 0 {
            let split = worklist.len() - take;
            local.append(&mut worklist.split_off(split));
        }
    }

    fn spill(&self, local: &mut Vec<usize>) {
        let mut worklist = self.worklist.lock().unwrap();
        let split = local.len() / 2;
        worklist.append(&mut local.split_off(split));
    }
}

struct RootScanner<'a> {
    job: &'a MarkJob,
}

impl Visitor for RootScanner<'_> {
    fn visit(&mut self, cell: &RawCell) {
        if let Some(addr) = self.job.claim(cell) {
            self.job.worklist.lock().unwrap().push(addr);
        }
    }
}

struct Marker<'a> {
    job: &'a MarkJob,
    local: Vec<usize>,
}

impl Visitor for Marker<'_> {
    fn visit(&mut self, cell: &RawCell) {
        if let Some(addr) = self.job.claim(cell) {
            self.local.push(addr);
            if self.local.len() > MARK_SPILL {
                self.job.spill(&mut self.local);
            }
        }
    }
}

fn join_marking(job: &MarkJob, persistent: bool) {
    let mut marker = Marker {
        job,
        local: Vec::new(),
    };
    let mut spins = 0;
    job.active.fetch_add(1, Ordering::AcqRel);
    loop {
        if job.done.load(Ordering::Acquire) {
            break;
        }
        if marker.local.is_empty() {
            job.steal(&mut marker.local);
        }
        if marker.local.is_empty() {
            if job.roots_scanned.load(Ordering::Acquire)
                && job.active.load(Ordering::Acquire) == 1
                && job.worklist.lock().unwrap().is_empty()
            {
                job.done.store(true, Ordering::Release);
                break;
            }
            if !persistent && spins >= ASSIST_SPINS {
                break;
            }
            spins += 1;
            std::thread::yield_now();
            continue;
        }
        spins = 0;
        let addr = marker.local.pop().unwrap();
        let object = unsafe { NonNull::new_unchecked(addr as *mut ()) };
        (job.host.visit_object)(object, &mut marker);
    }
    job.active.fetch_sub(1, Ordering::Release);
}

struct WeakClearer<'a> {
    heap: &'a ChunkedHeap,
}

impl Visitor for WeakClearer<'_> {
    fn visit(&mut self, cell: &RawCell) {
        let word = cell.load();
        if word & TAG_MASK == WEAK_PTR && word != CLEARED {
            let addr = (word & !TAG_MASK) as usize;
            let dead = !self.heap.in_heap(addr)
                || !self.heap.chunk_of(addr).bitmap.is_set(addr);
            if dead {
                cell.store_raw(CLEARED);
            }
        }
    }
}

/// Per-thread local heap of the mark-sweep collector.
pub struct MarkSweepLocal {
    state: Arc<MarkSweepState>,
    node: LocalNode,
    tlab: Tlab,
}

struct Tlab {
    cursor: Cell<*mut u8>,
    end: Cell<*mut u8>,
}

const TLAB_SIZES: [usize; 2] = [32 * 1024, 8 * 1024];
const MIN_GC_THRESHOLD: usize = 16 * 1024 * 1024;
const TRANSIENT_ATTEMPTS: usize = 2;

enum TerminalAllocation {
    Absorbed,
    Done(Option<NonNull<u8>>),
}

impl Tlab {
    const fn empty() -> Self {
        Self {
            cursor: Cell::new(ptr::null_mut()),
            end: Cell::new(ptr::null_mut()),
        }
    }
}

impl MarkSweepLocal {
    pub fn new(state: Arc<MarkSweepState>) -> Box<Self> {
        let local = Box::new(Self {
            state,
            node: LocalNode::detached(),
            tlab: Tlab::empty(),
        });
        local.state.safepoint.attach(&local.node);
        local
    }

    pub fn state(&self) -> &MarkSweepState {
        &self.state
    }

    pub fn collect(&self) {
        self.invalidate_tlab();
        self.state.collect_internal(Some(self));
    }

    pub fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.park_if_requested();
        let need = need_for(layout);
        if let Some(ptr) = self.tlab_bump(need) {
            return Ok(ptr);
        }
        if self.state.used_exceeds_threshold() {
            self.collect();
        }
        if let Some(ptr) = self.tlab_refill(need) {
            return Ok(ptr);
        }
        // Contended slow path. A collect may be absorbed by another
        // thread's cycle whose memory siblings consume before the retry
        for _ in 0..TRANSIENT_ATTEMPTS {
            if let Some(ptr) = self.state.alloc_block(layout) {
                return Ok(ptr);
            }
            self.collect();
            if let Some(ptr) = self.tlab_bump(need) {
                return Ok(ptr);
            }
            if let Some(ptr) = self.tlab_refill(need) {
                return Ok(ptr);
            }
        }
        loop {
            if let Some(ptr) = self.state.alloc_block(layout) {
                return Ok(ptr);
            }
            match self.state.collect_terminal(Some(&self.node), layout) {
                TerminalAllocation::Done(Some(ptr)) => return Ok(ptr),
                TerminalAllocation::Done(None) => {
                    return Err(AllocError::OutOfMemory(layout))
                }
                TerminalAllocation::Absorbed => continue,
            }
        }
    }

    fn park_if_requested(&self) {
        if self.node.requested() {
            self.invalidate_tlab();
            self.state.safepoint.park_for_collection(&self.node);
        }
    }

    fn tlab_bump(&self, need: usize) -> Option<NonNull<u8>> {
        let cursor = self.tlab.cursor.get();
        if cursor.is_null() {
            return None;
        }
        let next = cursor.wrapping_add(need);
        if next > self.tlab.end.get() {
            return None;
        }
        self.tlab.cursor.set(next);
        Some(unsafe { NonNull::new_unchecked(cursor) })
    }

    fn tlab_refill(&self, need: usize) -> Option<NonNull<u8>> {
        if need > *TLAB_SIZES.last().unwrap() {
            return None;
        }
        self.invalidate_tlab();
        for &size in &TLAB_SIZES {
            let layout = Layout::from_size_align(size, ALIGN).unwrap();
            if let Some(block) = self.state.alloc_block(layout) {
                unsafe {
                    self.tlab.cursor.set(block.as_ptr());
                    self.tlab.end.set(block.as_ptr().add(size));
                }
                return self.tlab_bump(need);
            }
        }
        None
    }

    fn invalidate_tlab(&self) {
        self.tlab.cursor.set(ptr::null_mut());
        self.tlab.end.set(ptr::null_mut());
    }
}

impl Drop for MarkSweepLocal {
    fn drop(&mut self) {
        self.invalidate_tlab();
        self.state.safepoint.detach(&self.node);
    }
}

fn erased_allocate_raw(local: *mut (), layout: Layout) -> Result<NonNull<u8>, AllocError> {
    unsafe { &*local.cast::<MarkSweepLocal>() }.allocate(layout)
}

fn erased_write_barrier(_local: *const (), _host: Word, _slot: &RawCell, _value: Word) {}

fn erased_collection_requested(local: *const ()) -> bool {
    unsafe { (*local.cast::<MarkSweepLocal>()).node.requested() }
}

fn erased_park_for_collection(local: *const ()) {
    let local = unsafe { &*local.cast::<MarkSweepLocal>() };
    local.park_if_requested();
}

fn erased_force_collect(local: *const ()) {
    let local = unsafe { &*local.cast::<MarkSweepLocal>() };
    local.collect();
}

fn erased_gc_in_progress(local: *const ()) -> bool {
    let local = unsafe { &*local.cast::<MarkSweepLocal>() };
    local.state.safepoint.is_armed()
}

fn erased_drop_local(local: *mut ()) {
    unsafe { drop(Box::from_raw(local.cast::<MarkSweepLocal>())) };
}

fn erased_global_new_local(shared: *const ()) -> *mut () {
    let state = shared as *const MarkSweepState;
    unsafe { Arc::increment_strong_count(state) };
    let local = MarkSweepLocal::new(unsafe { Arc::from_raw(state) });
    Box::into_raw(local) as *mut ()
}

fn erased_set_host(shared: *const (), host: GcHost) {
    unsafe { (*shared.cast::<MarkSweepState>()).set_host(host) };
}

fn erased_global_iterate_roots(_shared: *const (), _roots: &mut dyn Visitor) {}

fn erased_global_should_collect(_shared: *const ()) -> bool {
    false
}

fn erased_global_gc_in_progress(shared: *const ()) -> bool {
    unsafe { (*shared.cast::<MarkSweepState>()).safepoint.is_armed() }
}

fn erased_global_force_collect(shared: *const ()) {
    unsafe { (*shared.cast::<MarkSweepState>()).collect_now() };
}

fn erased_global_contains(shared: *const (), addr: Word) -> bool {
    unsafe { (*shared.cast::<MarkSweepState>()).contains(addr as usize) }
}

fn erased_global_is_young(_shared: *const (), _value: Word) -> bool {
    false
}

fn erased_global_stats(shared: *const ()) -> HeapStats {
    unsafe { (*shared.cast::<MarkSweepState>()).stats() }
}

fn erased_global_drop_shared(shared: *mut ()) {
    unsafe { Arc::decrement_strong_count(shared.cast::<MarkSweepState>()) };
}

static MARK_SWEEP_HEAP_VTABLE: HeapVtable = HeapVtable {
    allocate_raw: erased_allocate_raw,
    write_barrier: erased_write_barrier,
    collection_requested: erased_collection_requested,
    park_for_collection: erased_park_for_collection,
    force_collect: erased_force_collect,
    gc_in_progress: erased_gc_in_progress,
    drop_local: erased_drop_local,
};

static MARK_SWEEP_GLOBAL_VTABLE: GlobalVtable = GlobalVtable {
    local_vtable: &MARK_SWEEP_HEAP_VTABLE,
    new_local: erased_global_new_local,
    set_host: erased_set_host,
    iterate_roots: erased_global_iterate_roots,
    should_collect: erased_global_should_collect,
    gc_in_progress: erased_global_gc_in_progress,
    force_collect: erased_global_force_collect,
    contains: erased_global_contains,
    is_young: erased_global_is_young,
    stats: erased_global_stats,
    drop_shared: erased_global_drop_shared,
};

pub struct MarkSweep {
    inner: Arc<MarkSweepState>,
}

impl MarkSweep {
    pub fn new(config: MarkSweepConfig) -> Result<Self, AllocError> {
        Ok(Self {
            inner: MarkSweepState::new(config)?,
        })
    }

    pub fn state(&self) -> &MarkSweepState {
        &self.inner
    }
}

impl HeapBackend for MarkSweep {
    type Config = MarkSweepConfig;

    fn new(config: Self::Config) -> Result<Self, AllocError> {
        MarkSweep::new(config)
    }

    fn into_global(self) -> (*mut (), &'static GlobalVtable) {
        let state = Arc::into_raw(self.inner) as *mut ();
        (state, &MARK_SWEEP_GLOBAL_VTABLE)
    }
}
