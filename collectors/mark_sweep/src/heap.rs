use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use core::alloc::Layout;
use core::ptr::NonNull;

use heap_api::{
    AllocError, CLEARED, GcHost, GlobalVtable, HeapBackend, HeapStats, HeapVtable, RawCell,
    STRONG_PTR, TAG_MASK, Visitor, WEAK_PTR, Word,
};

use heap_utils::{Bitmap, LocalNode, MMapBuffer, Safepoint};

use crate::block::{self, ALIGN, BlockHeader, FreeList};

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
        let base = self.buffer.start();
        let size = self.buffer.size();
        self.alloc.lock().unwrap().sweep(base, size, &self.bitmap);
        self.cycles.fetch_add(1, Ordering::Relaxed);
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
        for block in block::blocks(self.buffer.start(), self.buffer.size()) {
            let keep = !BlockHeader::is_free(block)
                && self
                    .bitmap
                    .is_set(BlockHeader::payload(block).as_ptr() as usize);
            if keep {
                let object = BlockHeader::payload(block).cast::<()>();
                (host.visit_object)(object, &mut clearer);
            }
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
}

impl MarkSweepLocal {
    /// Boxed: the intrusive safepoint node must not move after attach.
    pub fn new(state: Arc<MarkSweepState>) -> Box<Self> {
        let local = Box::new(Self {
            state,
            node: LocalNode::detached(),
        });
        local.state.safepoint.attach(&local.node);
        local
    }

    pub fn state(&self) -> &MarkSweepState {
        &self.state
    }

    pub fn collect(&self) {
        self.state
            .safepoint
            .stop_the_world(Some(&self.node), || self.state.collect());
    }

    pub fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.state.safepoint.park_for_collection(&self.node);
        if let Some(block) = self.state.alloc_block(layout) {
            return Ok(block);
        }
        self.state
            .safepoint
            .stop_the_world(Some(&self.node), || self.state.collect());
        if let Some(block) = self.state.alloc_block(layout) {
            return Ok(block);
        }
        Err(AllocError::OutOfMemory(layout))
    }
}

impl Drop for MarkSweepLocal {
    fn drop(&mut self) {
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
    local.state.safepoint.park_for_collection(&local.node);
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
