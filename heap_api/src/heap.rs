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

/// VM services a collector needs, registered once via [`SharedHeap::set_host`].
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

pub trait LocalHeap: Send {
    fn allocate_raw(&self, layout: Layout) -> Result<NonNull<u8>, AllocError>;
    fn write_barrier(&self, host: Word, slot: &RawCell, value: Word);
    fn collection_requested(&self) -> bool;
    fn park_for_collection(&self);
    fn force_collect(&self);
    fn collect_minor(&self);
    fn gc_in_progress(&self) -> bool;
}

pub trait SharedHeap: Send + Sync {
    fn new_local(&self) -> Box<dyn LocalHeap>;
    fn set_host(&self, host: GcHost);
    fn iterate_roots(&self, roots: &mut dyn Visitor);
    fn should_collect(&self) -> bool;
    fn gc_in_progress(&self) -> bool;
    /// Run one full collection cycle synchronously; returns when complete.
    /// Must not be called from a thread that owns a local heap.
    fn force_collect(&self);
    fn contains(&self, addr: Word) -> bool;
    fn is_young(&self, value: Word) -> bool;
    fn stats(&self) -> HeapStats;
}

pub trait HeapBackend: Sized + Send + Sync {
    type Config;

    fn new(config: Self::Config) -> Result<Self, AllocError>;

    fn into_shared(self) -> std::sync::Arc<dyn SharedHeap>;
}
