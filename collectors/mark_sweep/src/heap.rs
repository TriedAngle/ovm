use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::{self, NonNull};

use heap_api::{
    AllocError, CLEARED, GcHost, GlobalVtable, HeapBackend, HeapStats, HeapVtable, RawCell,
    STRONG_PTR, TAG_MASK, Visitor, WEAK_PTR, Word,
};

use heap_utils::{Bitmap, LocalNode, MMapBuffer, Safepoint};

use crate::block::{ALIGN, FreeList};

#[derive(Debug, Clone, Copy)]
pub struct MarkSweepConfig {
    pub heap_size: usize,
}

impl Default for MarkSweepConfig {
    fn default() -> Self {
        Self {
            heap_size: 256 * 1024 * 1024,
        }
    }
}

pub struct MarkSweepState {
    buffer: MMapBuffer,
    bitmap: Bitmap,
    alloc: Mutex<FreeList>,
    safepoint: Safepoint,
    host: Mutex<Option<GcHost>>,
    cycles: AtomicUsize,
    gc_threshold: AtomicUsize,
}

unsafe impl Send for MarkSweepState {}
unsafe impl Sync for MarkSweepState {}

impl MarkSweepState {
    pub fn new(config: MarkSweepConfig) -> Result<Arc<Self>, AllocError> {
        let buffer = MMapBuffer::new(config.heap_size)?;
        let base = buffer.start();
        let size = buffer.size();
        Ok(Arc::new(Self {
            bitmap: Bitmap::new(base.as_ptr() as usize, size, ALIGN),
            alloc: Mutex::new(FreeList::covering(base, size)),
            buffer,
            safepoint: Safepoint::new(),
            host: Mutex::new(None),
            cycles: AtomicUsize::new(0),
            gc_threshold: AtomicUsize::new(MIN_GC_THRESHOLD),
        }))
    }

    pub fn set_host(&self, host: GcHost) {
        *self.host.lock().unwrap() = Some(host);
    }

    pub fn base(&self) -> NonNull<u8> {
        self.buffer.start()
    }

    pub fn capacity(&self) -> usize {
        self.buffer.size()
    }

    pub fn cycles(&self) -> usize {
        self.cycles.load(Ordering::Relaxed)
    }

    pub fn contains(&self, addr: usize) -> bool {
        self.buffer.contains(addr)
    }

    pub fn stats(&self) -> HeapStats {
        let alloc = self.alloc.lock().unwrap();
        HeapStats {
            used: alloc.live_bytes(),
            capacity: self.buffer.size(),
        }
    }

    pub fn free_bytes(&self) -> usize {
        self.alloc.lock().unwrap().free_bytes()
    }

    pub fn collect_now(&self) {
        self.safepoint.stop_the_world(None, || self.collect());
    }

    fn alloc_block(&self, layout: Layout) -> Option<NonNull<u8>> {
        self.alloc.lock().unwrap().allocate(layout)
    }

    fn collect(&self) {
        self.bitmap.clear_all();
        self.mark();
        self.clear_weaks();
        let host = self.host();
        let live = {
            let mut alloc = self.alloc.lock().unwrap();
            alloc.sweep(self.buffer.start(), self.buffer.size(), &self.bitmap, &host);
            alloc.live_bytes()
        };
        let threshold = (live * 2).max(MIN_GC_THRESHOLD).min(self.buffer.size());
        self.gc_threshold.store(threshold, Ordering::Relaxed);
        self.cycles.fetch_add(1, Ordering::Relaxed);
    }

    fn used_exceeds_threshold(&self) -> bool {
        self.alloc.lock().unwrap().live_bytes() > self.gc_threshold.load(Ordering::Relaxed)
    }

    fn host(&self) -> GcHost {
        self.host
            .lock()
            .unwrap()
            .as_ref()
            .copied()
            .expect("host not registered before collection")
    }

    fn mark(&self) {
        let host = self.host();
        let mut marker = Marker {
            state: self,
            worklist: Vec::new(),
        };
        (host.visit_roots)(host.ctx, &mut marker);
        while let Some(addr) = marker.worklist.pop() {
            let object = unsafe { NonNull::new_unchecked(addr as *mut ()) };
            (host.visit_object)(object, &mut marker);
        }
    }

    fn clear_weaks(&self) {
        let host = self.host();
        let mut clearer = WeakClearer { state: self };
        (host.visit_roots)(host.ctx, &mut clearer);
        for addr in self.bitmap.iter_set() {
            let object = unsafe { NonNull::new_unchecked(addr as *mut ()) };
            (host.visit_object)(object, &mut clearer);
        }
    }
}

struct Marker<'a> {
    state: &'a MarkSweepState,
    worklist: Vec<usize>,
}

impl Visitor for Marker<'_> {
    fn visit(&mut self, cell: &RawCell) {
        let word = cell.load();
        if word & TAG_MASK == STRONG_PTR {
            let addr = (word & !TAG_MASK) as usize;
            if self.state.contains(addr) && self.state.bitmap.set(addr) {
                self.worklist.push(addr);
            }
        }
    }
}

struct WeakClearer<'a> {
    state: &'a MarkSweepState,
}

impl Visitor for WeakClearer<'_> {
    fn visit(&mut self, cell: &RawCell) {
        let word = cell.load();
        if word & TAG_MASK == WEAK_PTR && word != CLEARED {
            let addr = (word & !TAG_MASK) as usize;
            if !self.state.contains(addr) || !self.state.bitmap.is_set(addr) {
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
        self.state
            .safepoint
            .stop_the_world(Some(&self.node), || self.state.collect());
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
        if let Some(ptr) = self.state.alloc_block(layout) {
            return Ok(ptr);
        }
        Err(AllocError::OutOfMemory(layout))
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

fn need_for(layout: Layout) -> usize {
    debug_assert!(layout.size() > 0, "zero-sized allocation");
    debug_assert!(layout.align() <= ALIGN, "alignment above {ALIGN} unsupported");
    layout.size().next_multiple_of(ALIGN)
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
