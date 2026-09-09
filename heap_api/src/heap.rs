use core::{alloc::Layout, cell::UnsafeCell, ptr::NonNull};

use crate::Word;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocError {
    OutOfMemory(Layout),
}

impl core::fmt::Display for AllocError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfMemory(layout) => write!(
                f,
                "out of memory: failed to allocate {} bytes (align {})",
                layout.size(),
                layout.align()
            ),
        }
    }
}

impl std::error::Error for AllocError {}

/// A single GC-traced slot: one tagged word. The VM layers typed cells
/// (`GcSlot<T>` etc.) on top; collectors only see raw words.
#[repr(transparent)]
pub struct RawCell {
    raw: UnsafeCell<Word>,
}

impl RawCell {
    pub unsafe fn from_word(w: Word) -> Self {
        Self {
            raw: UnsafeCell::new(w),
        }
    }

    pub fn load(&self) -> Word {
        unsafe { *self.raw.get() }
    }

    pub fn store_raw(&self, w: Word) {
        unsafe { *self.raw.get() = w };
    }
}

pub trait Visitor {
    fn visit(&mut self, cell: &RawCell);
}

impl<V: Visitor + ?Sized> Visitor for &mut V {
    fn visit(&mut self, cell: &RawCell) {
        (**self).visit(cell)
    }
}

/// Statistics reported by a global heap for introspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeapStats {
    pub used: usize,
    pub capacity: usize,
}

/// VM services a collector needs, registered once via
/// `GlobalVtable::set_host`. Plain function pointers in the crate's
/// vtable style; `ctx` is opaque VM state passed back to `visit_roots`.
///
/// Contract: the host (and everything `ctx` points to) must outlive the
/// heap.
pub struct GcHost {
    pub ctx: *const (),
    /// Enumerate every root: handle blocks, stacks, caches, interner,
    /// well-known table. Called once per cycle, at quiescence.
    pub visit_roots: unsafe fn(ctx: *const (), visitor: &mut dyn Visitor),
    /// Size and alignment of the object starting at `addr` (reads its map).
    ///
    /// # Safety: `addr` must be the start of a live object.
    pub layout_of: unsafe fn(addr: NonNull<()>) -> Layout,
    /// Trace the object at `addr` (read its map, dispatch its edges).
    ///
    /// # Safety: `addr` must be the start of a live object.
    pub visit_object: unsafe fn(addr: NonNull<()>, visitor: &mut dyn Visitor),
}

/// Function table of per-thread (local) heap operations.
pub struct HeapVtable {
    /// Allocate `layout` bytes. Implementations may run a full collection
    /// cycle (stopping the world via `collection_requested`/
    /// `park_for_collection` and marking through the registered
    /// [`GcHost`]) before failing with [`AllocError::OutOfMemory`].
    ///
    /// Contract: called only where the caller holds no unrooted objects
    /// (guaranteed by the `&mut Heap` / no-GC-scope discipline).
    pub allocate_raw: fn(local: *mut (), layout: Layout) -> Result<NonNull<u8>, AllocError>,
    pub write_barrier: fn(local: *const (), host: Word, slot: &RawCell, value: Word),
    pub collection_requested: fn(local: *const ()) -> bool,
    pub park_for_collection: fn(local: *const ()),
    pub gc_in_progress: fn(local: *const ()) -> bool,
    pub drop_local: fn(local: *mut ()),
}

/// Function table of shared (global) heap operations. The VM-side roots
/// reach the backend only through `iterate_roots`.
pub struct GlobalVtable {
    /// Vtable used for local heaps created by the global heap.
    pub local_vtable: &'static HeapVtable,
    pub new_local: fn(shared: *const ()) -> *mut (),
    pub set_host: fn(shared: *const (), host: GcHost),
    pub iterate_roots: fn(shared: *const (), roots: &mut dyn Visitor),
    pub should_collect: fn(shared: *const ()) -> bool,
    pub gc_in_progress: fn(shared: *const ()) -> bool,
    pub contains: fn(shared: *const (), addr: Word) -> bool,
    pub is_young: fn(shared: *const (), value: Word) -> bool,
    pub stats: fn(shared: *const ()) -> HeapStats,
    pub drop_shared: fn(shared: *mut ()),
}

pub trait HeapBackend: Sized + Send + Sync {
    type Config;

    fn new(config: Self::Config) -> Result<Self, AllocError>;

    fn into_global(self) -> (*mut (), &'static GlobalVtable);
}
