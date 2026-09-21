use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::{self, NonNull};
use std::thread::JoinHandle;

use heap_api::{
    AllocError, CLEARED, GcHost, HeapBackend, HeapStats, LocalHeap, RawCell, STRONG_PTR,
    SharedHeap, TAG_MASK, Visitor, WEAK_PTR, Word,
};

use heap_utils::{LocalNode, Safepoint};

use crate::block::{ALIGN, need_for};
use crate::chunk::{Chunk, ChunkHeader, ChunkedHeap};

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

const FORWARD_TAG: Word = 0b10;

const MIN_YOUNG_CHUNKS: usize = 2;
const MAX_YOUNG_CHUNKS: usize = 32;

pub struct MarkSweepState {
    /// `[base, end)` of the reservation
    heap_base: usize,
    heap_end: usize,
    alloc: Mutex<ChunkedHeap>,
    safepoint: Safepoint,
    host: Mutex<Option<GcHost>>,
    mark_job: Mutex<Option<Arc<MarkJob>>>,
    mark_assists: AtomicUsize,
    cycles: AtomicUsize,
    minor_cycles: AtomicUsize,
    young_limit: AtomicUsize,
    gc_threshold: AtomicUsize,
    sweep_signal: Mutex<SweepSignal>,
    sweep_cond: Condvar,
    sweeper: Mutex<Option<JoinHandle<()>>>,
    background_sweeps: AtomicUsize,
}

struct SweepSignal {
    epoch: usize,
    shutdown: bool,
}

unsafe impl Send for MarkSweepState {}
unsafe impl Sync for MarkSweepState {}

impl MarkSweepState {
    pub fn new(config: MarkSweepConfig) -> Result<Arc<Self>, AllocError> {
        let alloc = ChunkedHeap::new(config.heap_size)?;
        let (heap_base, heap_end) = alloc.reservation();
        let this = Arc::new(Self {
            heap_base,
            heap_end,
            alloc: Mutex::new(alloc),
            safepoint: Safepoint::new(),
            host: Mutex::new(None),
            mark_job: Mutex::new(None),
            mark_assists: AtomicUsize::new(0),
            cycles: AtomicUsize::new(0),
            minor_cycles: AtomicUsize::new(0),
            young_limit: AtomicUsize::new(MIN_YOUNG_CHUNKS),
            gc_threshold: AtomicUsize::new(MIN_GC_THRESHOLD),
            sweep_signal: Mutex::new(SweepSignal {
                epoch: 0,
                shutdown: false,
            }),
            sweep_cond: Condvar::new(),
            sweeper: Mutex::new(None),
            background_sweeps: AtomicUsize::new(0),
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
        let sweeper = std::thread::Builder::new()
            .name("mark-sweep-sweeper".into())
            .spawn({
                let state = Arc::as_ptr(&this) as usize;
                move || sweeper_loop(state as *const MarkSweepState)
            })
            .ok();
        *this.sweeper.lock().unwrap() = sweeper;
        Ok(this)
    }

    pub fn set_host(&self, host: GcHost) {
        *self.host.lock().unwrap() = Some(host);
    }

    fn in_reservation(&self, addr: usize) -> bool {
        addr >= self.heap_base && addr < self.heap_end
    }

    fn header_of(&self, addr: usize) -> &ChunkHeader {
        ChunkHeader::at(self.heap_base, addr)
    }

    pub fn is_young_addr(&self, addr: usize) -> bool {
        debug_assert!(
            self.in_reservation(addr),
            "young-check on a non-heap address {addr:#x}"
        );
        self.header_of(addr).young()
    }

    pub fn write_barrier(&self, slot: &RawCell, value: Word) {
        let value_addr = (value & !TAG_MASK) as usize;
        debug_assert!(self.in_reservation(value_addr));
        let slot_addr = slot as *const RawCell as usize;
        let slot_header = self.header_of(slot_addr);
        if !slot_header.young() && self.header_of(value_addr).young() {
            slot_header.remember(slot_addr);
        }
    }

    pub fn cycles(&self) -> usize {
        self.cycles.load(Ordering::Relaxed)
    }

    pub fn mark_assists(&self) -> usize {
        self.mark_assists.load(Ordering::Relaxed)
    }

    pub fn background_sweeps(&self) -> usize {
        self.background_sweeps.load(Ordering::Relaxed)
    }

    pub fn active_chunks(&self) -> usize {
        self.alloc.lock().unwrap().active_chunks()
    }

    pub fn pending_chunks(&self) -> usize {
        self.alloc.lock().unwrap().pending_chunks()
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

    pub fn collect_minor_now(&self) {
        self.collect_minor_internal(None);
    }

    pub fn minor_cycles(&self) -> usize {
        self.minor_cycles.load(Ordering::Relaxed)
    }

    pub fn should_minor_gc(&self) -> bool {
        let heap = self.alloc.lock().unwrap();
        heap.young_chunk_count() >= self.young_limit.load(Ordering::Relaxed)
    }

    fn update_young_limit(&self, survivors: usize, chunk_size: usize) {
        let want = survivors
            .saturating_mul(2)
            .div_ceil(chunk_size)
            .max(MIN_YOUNG_CHUNKS)
            .min(MAX_YOUNG_CHUNKS);
        self.young_limit.store(want, Ordering::Relaxed);
    }

    fn full_collect_stw(&self, host: &GcHost) -> Option<usize> {
        let mut heap = self.alloc.lock().unwrap();
        self.mark(&heap, host);
        heap.flag_all_pending();
        heap.finish_pending(host);
        heap.set_sweep_block(false);
        heap.take_completed_live()
    }

    fn collect_minor_internal(&self, requester: Option<&MarkSweepLocal>) {
        {
            let heap = self.alloc.lock().unwrap();
            if heap.young_chunk_count() == 0 {
                return;
            }
        }
        let host = self.host();
        self.safepoint
            .stop_the_world(requester.map(|local| &local.node), || {
                self.collect_minor_stw(&host, true);
            });
    }

    /// Evacuate the nursery. Must be called while the world is stopped.
    ///
    /// When there are too few free chunks to promote into, `fallback_to_full`
    /// selects between bailing out to a full collection (a plain minor) and
    /// simply leaving the nursery for the caller's own full collection.
    fn collect_minor_stw(&self, host: &GcHost, fallback_to_full: bool) {
        self.finish_sweeping(host);
        let young = self.alloc.lock().unwrap().young_indices();
        if young.is_empty() {
            self.alloc.lock().unwrap().set_sweep_block(false);
            return;
        }
        let starved = {
            let heap = self.alloc.lock().unwrap();
            heap.available_chunks() < 2 * young.len()
        };
        if starved {
            if fallback_to_full {
                let completed = self.full_collect_stw(host);
                if let Some(live) = completed {
                    self.update_threshold(live);
                }
                self.cycles.fetch_add(1, Ordering::Relaxed);
            }
            return;
        }

        let mut job = MinorJob {
            state: self,
            host: *host,
            promote: Vec::new(),
            worklist: Vec::new(),
            deferred_weak: Vec::new(),
            survivors: 0,
        };
        {
            let mut scanner = MinorScanner { job: &mut job };
            (host.visit_roots)(host.ctx, &mut scanner);
        }
        // bind first: the temporary guard must not outlive the
        // statement, the scanner below re-enters the alloc lock
        let remembered = self.alloc.lock().unwrap().remembered_slots();
        for slot_addr in remembered {
            let cell = unsafe { &*(slot_addr as *const RawCell) };
            let mut scanner = MinorScanner { job: &mut job };
            scanner.visit(cell);
        }
        job.drain();

        // weak references to young objects settle only after the
        // strong closure: forwarded targets move the cell, dead
        // targets clear it
        for cell in job.deferred_weak {
            let cell = unsafe { &*cell };
            let word = cell.load();
            if word & TAG_MASK == WEAK_PTR && word != CLEARED {
                let addr = (word & !TAG_MASK) as usize;
                let young = self.is_young_addr(addr);
                if young {
                    let header = unsafe { *(addr as *const Word) };
                    if header & TAG_MASK == FORWARD_TAG {
                        cell.store_raw((header & !TAG_MASK) as Word | WEAK_PTR);
                    } else {
                        cell.store_raw(CLEARED);
                    }
                }
            }
        }

        let survivors;
        {
            let mut heap = self.alloc.lock().unwrap();
            for (chunk, cursor) in &job.promote {
                heap.finish_promotion(chunk, *cursor);
            }
            heap.remove_chunks(&young);
            heap.clear_remembered_sets();
            survivors = job.survivors;
            let chunk_size = heap.chunk_size();
            heap.set_sweep_block(false);
            self.update_young_limit(survivors, chunk_size);
        }
        self.minor_cycles.fetch_add(1, Ordering::Relaxed);
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
                // Evacuate the nursery first, under the same stop-the-world:
                // this empties the young generation and clears the remembered
                // set, so the major below cannot trip over stale old->young
                // remembered bits left behind by an earlier sweep.
                self.collect_minor_stw(&host, false);
                self.finish_sweeping(&host);
                let completed = {
                    let mut heap = self.alloc.lock().unwrap();
                    self.mark(&heap, &host);
                    heap.flag_all_pending();
                    heap.set_sweep_block(false);
                    heap.take_completed_live()
                };
                if let Some(live) = completed {
                    self.update_threshold(live);
                }
            });
        self.cycles.fetch_add(1, Ordering::Relaxed);
        self.wake_sweeper();
        self.sweep_to_completion(&host, requester);
    }

    /// The initiating thread sweeps alongside the background sweeper (and
    /// any lazy-sweeping mutators) until the cycle's pendings are drained.
    fn sweep_to_completion(&self, host: &GcHost, requester: Option<&MarkSweepLocal>) {
        loop {
            if let Some(local) = requester {
                local.park_if_requested();
            }
            let chunk = self.alloc.lock().unwrap().claim_pending(false);
            if let Some(chunk) = chunk {
                let live = ChunkedHeap::sweep_claimed(&chunk, host);
                let completed = {
                    let mut heap = self.alloc.lock().unwrap();
                    heap.publish_swept(&chunk, live);
                    heap.take_completed_live()
                };
                if let Some(live) = completed {
                    self.update_threshold(live);
                }
                continue;
            }
            let drained = {
                let heap = self.alloc.lock().unwrap();
                heap.pending_chunks() == 0 && !heap.any_sweeping()
            };
            if drained {
                return;
            }
            std::thread::yield_now();
        }
    }

    fn wake_sweeper(&self) {
        let mut signal = self.sweep_signal.lock().unwrap();
        signal.epoch += 1;
        self.sweep_cond.notify_all();
    }

    fn drain_pending(&self) {
        let Some(host) = self.try_host() else { return };
        loop {
            let chunk = self.alloc.lock().unwrap().claim_pending(false);
            let Some(chunk) = chunk else { return };
            let live = ChunkedHeap::sweep_claimed(&chunk, &host);
            let completed = {
                let mut heap = self.alloc.lock().unwrap();
                heap.publish_swept(&chunk, live);
                heap.take_completed_live()
            };
            self.background_sweeps.fetch_add(1, Ordering::Relaxed);
            if let Some(live) = completed {
                self.update_threshold(live);
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

    fn alloc_block(&self, layout: Layout, young: bool) -> Option<NonNull<u8>> {
        let host = self.try_host();
        loop {
            let (ptr, claimed, completed) = {
                let mut heap = self.alloc.lock().unwrap();
                let (ptr, claimed) = if young {
                    heap.allocate_young(layout, host.as_ref())
                } else {
                    heap.allocate(layout, host.as_ref())
                };
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
        job.settle_weaks();
        *self.mark_job.lock().unwrap() = None;
    }
}

fn sweeper_loop(state: *const MarkSweepState) {
    let state = unsafe { &*state };
    let mut seen_epoch = 0;
    loop {
        {
            let mut signal = state.sweep_signal.lock().unwrap();
            while signal.epoch == seen_epoch && !signal.shutdown {
                signal = state.sweep_cond.wait(signal).unwrap();
            }
            if signal.shutdown {
                return;
            }
            seen_epoch = signal.epoch;
        }
        state.drain_pending();
    }
}

impl Drop for MarkSweepState {
    fn drop(&mut self) {
        {
            let mut signal = self.sweep_signal.lock().unwrap();
            signal.shutdown = true;
            self.sweep_cond.notify_all();
        }
        if let Some(handle) = self.sweeper.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

const MARK_BATCH: usize = 32;
const MARK_SPILL: usize = 512;
const ASSIST_SPINS: usize = 256;

struct MarkJob {
    heap_base: usize,
    heap_end: usize,
    host: GcHost,
    worklist: Mutex<Vec<usize>>,
    /// Weak cells seen while tracing (roots and live objects alike):
    /// liveness cannot be decided mid-closure, so they are recorded here
    /// and settled against the mark bits once the closure drains.
    deferred_weak: Mutex<Vec<*const RawCell>>,
    roots_scanned: AtomicBool,
    active: AtomicUsize,
    done: AtomicBool,
}

unsafe impl Send for MarkJob {}
unsafe impl Sync for MarkJob {}

impl MarkJob {
    fn new(heap: &ChunkedHeap, host: GcHost) -> Self {
        let (heap_base, heap_end) = heap.reservation();
        Self {
            heap_base,
            heap_end,
            host,
            worklist: Mutex::new(Vec::new()),
            deferred_weak: Mutex::new(Vec::new()),
            roots_scanned: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            done: AtomicBool::new(false),
        }
    }

    fn in_heap(&self, addr: usize) -> bool {
        addr >= self.heap_base && addr < self.heap_end
    }

    fn header_of(&self, addr: usize) -> &ChunkHeader {
        ChunkHeader::at(self.heap_base, addr)
    }

    fn claim_word(&self, word: Word) -> Option<usize> {
        if word & TAG_MASK != STRONG_PTR {
            return None;
        }
        let addr = (word & !TAG_MASK) as usize;
        self.in_heap(addr)
            .then(|| self.header_of(addr))
            .filter(|header| header.mark_set(addr))
            .map(|_| addr)
    }

    fn claim(&self, cell: &RawCell) -> Option<usize> {
        self.claim_word(cell.load())
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

    fn settle_weaks(&self) {
        let cells = self
            .deferred_weak
            .lock()
            .unwrap()
            .drain(..)
            .collect::<Vec<_>>();
        for cell in cells {
            let cell = unsafe { &*cell };
            let word = cell.load();
            if word & TAG_MASK == WEAK_PTR && word != CLEARED {
                let addr = (word & !TAG_MASK) as usize;
                let dead = !self.in_heap(addr) || !self.header_of(addr).mark_is_set(addr);
                if dead {
                    cell.store_raw(CLEARED);
                }
            }
        }
    }
}

struct RootScanner<'a> {
    job: &'a MarkJob,
}

impl Visitor for RootScanner<'_> {
    fn visit(&mut self, cell: &RawCell) {
        let word = cell.load();
        let tag = word & TAG_MASK;
        if tag == STRONG_PTR {
            if let Some(addr) = self.job.claim(cell) {
                self.job.worklist.lock().unwrap().push(addr);
            }
        } else if tag == WEAK_PTR && word != CLEARED {
            self.job
                .deferred_weak
                .lock()
                .unwrap()
                .push(cell as *const RawCell);
        }
    }
}

struct Marker<'a> {
    job: &'a MarkJob,
    local: Vec<usize>,
}

impl Visitor for Marker<'_> {
    fn visit(&mut self, cell: &RawCell) {
        let word = cell.load();
        let tag = word & TAG_MASK;
        if tag == STRONG_PTR {
            if let Some(addr) = self.job.claim_word(word) {
                self.local.push(addr);
                if self.local.len() > MARK_SPILL {
                    self.job.spill(&mut self.local);
                }
            }
        } else if tag == WEAK_PTR && word != CLEARED {
            self.job
                .deferred_weak
                .lock()
                .unwrap()
                .push(cell as *const RawCell);
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

struct MinorJob<'a> {
    state: &'a MarkSweepState,
    host: GcHost,
    /// Promotion chunks with their bump cursors
    promote: Vec<(Arc<Chunk>, usize)>,
    worklist: Vec<usize>,
    /// Weak cells whose young targets settle after the strong closure.
    deferred_weak: Vec<*const RawCell>,
    survivors: usize,
}

impl MinorJob<'_> {
    fn is_young(&self, addr: usize) -> bool {
        self.state.is_young_addr(addr)
    }

    fn evacuate(&mut self, addr: usize) -> usize {
        let header = unsafe { *(addr as *const Word) };
        if header & TAG_MASK == FORWARD_TAG {
            return (header & !TAG_MASK) as usize;
        }
        let layout = (self.host.layout_of)(unsafe { NonNull::new_unchecked(addr as *mut ()) });
        let size = need_for(layout);
        let new = self.promote_alloc(size);
        unsafe {
            core::ptr::copy_nonoverlapping(addr as *const u8, new as *mut u8, layout.size());
            *(addr as *mut Word) = new as Word | FORWARD_TAG;
        }
        self.survivors += size;
        self.worklist.push(new);
        new
    }

    fn promote_alloc(&mut self, size: usize) -> usize {
        if let Some((chunk, cursor)) = self.promote.last_mut() {
            let next = *cursor + size;
            if next <= chunk.object_size() {
                let at = chunk.object_base() + *cursor;
                *cursor = next;
                return at;
            }
        }
        let chunk = self
            .state
            .alloc
            .lock()
            .unwrap()
            .activate_promotion_chunk()
            .expect("promotion chunks guaranteed by the pre-flight gate");
        let at = chunk.object_base();
        self.promote.push((chunk, size));
        at
    }

    fn drain(&mut self) {
        let visit_object = self.host.visit_object;
        while let Some(addr) = self.worklist.pop() {
            let object = unsafe { NonNull::new_unchecked(addr as *mut ()) };
            let mut scanner = MinorScanner { job: self };
            visit_object(object, &mut scanner);
        }
    }
}

struct MinorScanner<'a, 'b> {
    job: &'a mut MinorJob<'b>,
}

impl Visitor for MinorScanner<'_, '_> {
    fn visit(&mut self, cell: &RawCell) {
        let word = cell.load();
        let tag = word & TAG_MASK;
        if tag == STRONG_PTR {
            let addr = (word & !TAG_MASK) as usize;
            if self.job.is_young(addr) {
                cell.store_raw(self.job.evacuate(addr) as Word | STRONG_PTR);
            }
        } else if tag == WEAK_PTR && word != CLEARED {
            let addr = (word & !TAG_MASK) as usize;
            if self.job.is_young(addr) {
                self.job.deferred_weak.push(cell as *const RawCell);
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

// SAFETY: the cursors are plain addresses into this thread's bump region;
// a Tlab is only ever touched by the thread owning the local heap.
unsafe impl Send for Tlab {}

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

    pub fn collect_minor(&self) {
        self.invalidate_tlab();
        self.state.collect_minor_internal(Some(self));
    }

    pub fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.park_if_requested();
        let need = need_for(layout);
        if let Some(ptr) = self.tlab_bump(need) {
            return Ok(ptr);
        }
        if self.state.should_minor_gc() {
            self.collect_minor();
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
            if let Some(ptr) = self.state.alloc_block(layout, false) {
                return Ok(ptr);
            }
            if self.state.should_minor_gc() {
                self.collect_minor();
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
            if let Some(ptr) = self.state.alloc_block(layout, false) {
                return Ok(ptr);
            }
            match self.state.collect_terminal(Some(&self.node), layout) {
                TerminalAllocation::Done(Some(ptr)) => return Ok(ptr),
                TerminalAllocation::Done(None) => return Err(AllocError::OutOfMemory(layout)),
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
            if let Some(block) = self.state.alloc_block(layout, true) {
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

impl LocalHeap for MarkSweepLocal {
    fn allocate_raw(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.allocate(layout)
    }

    fn write_barrier(&self, _host: Word, slot: &RawCell, value: Word) {
        self.state.write_barrier(slot, value);
    }

    fn collection_requested(&self) -> bool {
        self.node.requested()
    }

    fn park_for_collection(&self) {
        self.park_if_requested();
    }

    fn force_collect(&self) {
        self.collect();
    }

    fn collect_minor(&self) {
        self.collect_minor();
    }

    fn gc_in_progress(&self) -> bool {
        self.state.safepoint.is_armed()
    }
}

impl SharedHeap for MarkSweep {
    fn new_local(&self) -> Box<dyn LocalHeap> {
        MarkSweepLocal::new(Arc::clone(&self.inner))
    }

    fn set_host(&self, host: GcHost) {
        self.inner.set_host(host);
    }

    fn iterate_roots(&self, _roots: &mut dyn Visitor) {}

    fn should_collect(&self) -> bool {
        false
    }

    fn gc_in_progress(&self) -> bool {
        self.inner.safepoint.is_armed()
    }

    fn force_collect(&self) {
        self.inner.collect_now();
    }

    fn contains(&self, addr: Word) -> bool {
        self.inner.contains(addr as usize)
    }

    fn is_young(&self, value: Word) -> bool {
        if value & TAG_MASK != STRONG_PTR {
            return false;
        }
        let addr = (value & !TAG_MASK) as usize;
        self.inner.in_reservation(addr) && self.inner.header_of(addr).young()
    }

    fn stats(&self) -> HeapStats {
        self.inner.stats()
    }
}

impl HeapBackend for MarkSweep {
    type Config = MarkSweepConfig;

    fn new(config: Self::Config) -> Result<Self, AllocError> {
        MarkSweep::new(config)
    }

    fn into_shared(self) -> Arc<dyn SharedHeap> {
        Arc::new(self)
    }
}
