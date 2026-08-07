use core::{alloc::Layout, cell::UnsafeCell, marker::PhantomData, ops::FnOnce, ptr::NonNull};

use crate::{FromValue, HeapObject, HeapPtr, IntoValue, Smi, Tagged, Value, Word};

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

/// Trait the Core Heap must implement
/// this VM is designed to be multithreaded
/// so a each thread must have its LocalHeap and a Global/SharedHeap must exist.
pub trait SharedHeap: Sized + Send + Sync {
    type Config;

    fn new(config: Self::Config) -> Result<Self, AllocError>;

    fn collect(&mut self, roots: &mut dyn RootVisitor);
    fn should_collect(&self) -> bool;
    fn gc_in_progress(&self) -> bool;

    fn contains(&self, addr: Word) -> bool;
    fn is_young(&self, _value: Value) -> bool;
}

pub trait LocalHeap: Sized {
    fn allocate_raw(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocError>;

    fn allocate<T: HeapObject>(&mut self, layout: Layout) -> Result<HeapPtr<T>, AllocError> {
        let ptr = self.allocate_raw(layout)?;
        Ok(unsafe { HeapPtr::new_unchecked(ptr.cast::<T>().as_ptr()) })
    }

    fn write_barrier(&self, host: Value, slot: &GcSlot, value: Value);

    fn collection_requested(&self) -> bool;
    fn park_for_collection(&self);
    fn gc_in_progress(&self) -> bool;

    fn safepoint_enter(&self);
    fn safepoint_exit(&self);

    fn poll_safepoint(&mut self) {
        if self.collection_requested() {
            self.park_for_collection();
        }
    }

    fn no_gc<R>(&mut self, f: impl for<'a> FnOnce(&'a mut NoGc<'a>, &'a Self) -> R) -> R {
        self.safepoint_enter();

        let mut guard = NoGc {
            _phantom: PhantomData,
        };
        let result = f(&mut guard, self);

        self.safepoint_exit();
        result
    }
}

pub struct NoGc<'a> {
    _phantom: PhantomData<&'a mut &'a ()>,
}

pub trait WordType: 'static {
    const IS_HEAP: bool;
}

impl WordType for Value {
    const IS_HEAP: bool = true;
}

impl WordType for Smi {
    const IS_HEAP: bool = false;
}

impl<T: HeapObject + 'static> WordType for Tagged<T> {
    const IS_HEAP: bool = true;
}

#[repr(transparent)]
pub struct GcSlot<T = Value> {
    raw: UnsafeCell<Value>,
    _phantom: PhantomData<T>,
}

impl<T: WordType> GcSlot<T>
where
    T: FromValue + IntoValue,
{
    pub fn get(&self) -> T {
        T::from_value(unsafe { *self.raw.get() }).expect("GcSlot invariant violated")
    }

    pub fn set(&self, heap: &impl LocalHeap, host: Value, value: T) {
        let v = value.into_value();
        if T::IS_HEAP {
            heap.write_barrier(host, self.ereased(), v);
        }
        unsafe { *self.raw.get() = v };
    }

    pub fn ereased(&self) -> &GcSlot {
        unsafe { &*(self as *const GcSlot<T> as *const GcSlot) }
    }

    pub const fn raw_get(this: *const Self) -> *mut T {
        this as *const T as *mut T
    }
}

pub trait RootVisitor {
    fn visit_slot(&mut self, slot: &GcSlot);

    fn visit_weak_slot(&mut self, slot: &GcSlot) {
        self.visit_slot(slot);
    }
}

pub trait EdgeVisitable {
    fn visit_edges(&self, visitor: &mut impl RootVisitor);
}
