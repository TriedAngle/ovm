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

/// VM services a collector needs, registered once via `GlobalVtable::set_host`.
///
/// Callable from any thread: `visit_roots` runs once per cycle on the
/// initiating thread while all mutators are stopped; `visit_object` may run
/// concurrently on several marker threads and must only read.
#[derive(Clone, Copy)]
pub struct GcHost {
    pub ctx: *const (),
    /// Enumerate every root
    pub visit_roots: fn(ctx: *const (), visitor: &mut dyn Visitor),
    /// Size and alignment of the object starting at `addr`
    pub layout_of: fn(addr: NonNull<()>) -> Layout,
    /// Trace the object at `addr`
    pub visit_object: fn(addr: NonNull<()>, visitor: &mut dyn Visitor),
}

unsafe impl Send for GcHost {}
unsafe impl Sync for GcHost {}

pub struct HeapVtable {
    pub allocate_raw: fn(local: *mut (), layout: Layout) -> Result<NonNull<u8>, AllocError>,
    pub write_barrier: fn(local: *const (), host: Word, slot: &RawCell, value: Word),
    pub collection_requested: fn(local: *const ()) -> bool,
    pub park_for_collection: fn(local: *const ()),
    pub force_collect: fn(local: *const ()),
    pub collect_minor: fn(local: *const ()),
    pub gc_in_progress: fn(local: *const ()) -> bool,
    pub drop_local: fn(local: *mut ()),
}

pub struct GlobalVtable {
    /// Vtable used for local heaps created by the global heap.
    pub local_vtable: &'static HeapVtable,
    pub new_local: fn(shared: *const ()) -> *mut (),
    pub set_host: fn(shared: *const (), host: GcHost),
    pub iterate_roots: fn(shared: *const (), roots: &mut dyn Visitor),
    pub should_collect: fn(shared: *const ()) -> bool,
    pub gc_in_progress: fn(shared: *const ()) -> bool,
    /// Run one full collection cycle synchronously; returns when complete.
    /// Must not be called from a thread that owns a local heap.
    pub force_collect: fn(shared: *const ()),
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
