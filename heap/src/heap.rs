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

pub trait RootVisitor: Visitor {}

impl<R: RootVisitor + ?Sized> RootVisitor for &mut R {}

/// Statistics reported by a global heap for introspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeapStats {
    pub used: usize,
    pub capacity: usize,
}

/// Function table of per-thread (local) heap operations.
pub struct HeapVtable {
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
    pub iterate_roots: fn(shared: *const (), roots: &mut dyn RootVisitor),
    pub collect: fn(shared: *const ()),
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
