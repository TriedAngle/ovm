use core::{alloc::Layout, cell::UnsafeCell, marker::PhantomData, ops::FnOnce, ptr::NonNull};

use crate::{Handle, HandleScope, HeapObject, HeapPtr, Smi, Tagged, Value, Word};

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

pub trait Heap: Sized + Send + Sync {
    type Config;
    type Local: LocalHeap;

    fn new(config: Self::Config) -> Result<Self, AllocError>;

    fn new_local(&self) -> Self::Local;

    fn iterate_roots(&self, roots: &mut dyn RootVisitor);

    fn collect(&self);

    fn should_collect(&self) -> bool;
    fn gc_in_progress(&self) -> bool;

    fn contains(&self, addr: Word) -> bool;
    fn is_young(&self, _value: Value) -> bool;
}

pub trait LocalHeap: Sized + Send {
    fn allocate_raw(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocError>;

    fn allocate<T: HeapObject>(&mut self, layout: Layout) -> Fresh<'_, T> {
        let ptr = self.allocate_raw(layout);
        debug_assert!(ptr.is_ok(), "allocation must not fail");
        let ptr = unsafe { ptr.unwrap_unchecked() };
        Fresh {
            ptr: ptr.cast::<T>(),
            _phantom: PhantomData,
        }
    }

    fn allocate_handle<'s, T: HeapObject>(
        &mut self,
        layout: Layout,
        scope: &'s HandleScope<'_>,
    ) -> Handle<'s, T> {
        self.allocate(layout).into_handle(scope)
    }

    fn write_barrier(&self, host: Value, slot: &GcSlot, value: Value);

    fn collection_requested(&self) -> bool;
    fn park_for_collection(&self);
    fn gc_in_progress(&self) -> bool;

    fn safepoint_poll(&mut self) {
        if self.collection_requested() {
            self.park_for_collection();
        }
    }

    fn no_gc<R>(&mut self, f: impl for<'a> FnOnce(&'a mut NoGc<'a>, &'a Self) -> R) -> R {
        let mut guard = NoGc {
            _phantom: PhantomData,
        };
        let result = f(&mut guard, self);

        result
    }
}

pub struct NoGc<'a> {
    _phantom: PhantomData<&'a mut &'a ()>,
}

impl<'a> NoGc<'a> {
    pub fn get<T: HeapObject>(&'a self, slot: &'a GcSlot<T>) -> HeapRef<'a, T> {
        unsafe { self.get_unchecked(slot.get()) }
    }

    pub unsafe fn get_unchecked<T: HeapObject>(&'a self, v: Tagged<T>) -> HeapRef<'a, T> {
        HeapRef {
            ptr: v.into(),
            _phantom: PhantomData,
        }
    }
}

/// Direct reference to a heap object, valid only within a no-GC scope.
/// Carries no strong/weak semantics. Mutation of GC-pointer fields should
/// go through `GcSlot::set` to preserve the write barrier.
pub struct HeapRef<'scope, T: HeapObject> {
    ptr: HeapPtr<T>,
    _phantom: PhantomData<&'scope T>,
}

impl<T: HeapObject> core::ops::Deref for HeapRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }
}

pub struct Fresh<'scope, T> {
    ptr: NonNull<T>,
    _phantom: PhantomData<&'scope T>,
}

impl<T> Clone for Fresh<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Fresh<'_, T> {}

impl<T> core::fmt::Debug for Fresh<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Fresh({:#x})", self.ptr.as_ptr() as Word)
    }
}

impl<'scope, T> Fresh<'scope, T> {
    pub const fn as_ptr(self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<'scope, T: HeapObject> Fresh<'scope, T> {
    /// Demote to a raw heap pointer.
    pub fn into_ptr(self) -> HeapPtr<T> {
        unsafe { HeapPtr::new_unchecked(self.ptr.as_ptr()) }
    }

    pub fn into_handle<'s>(self, scope: &'s HandleScope<'_>) -> Handle<'s, T> {
        let value = Value::from_bits(self.ptr.as_ptr() as Word | crate::value::STRONG_PTR);
        scope.create_handle(unsafe { Tagged::from_value_unchecked(value) })
    }

    /// Promote to a direct reference within a no-GC scope.
    pub fn heap_ref<'a>(self, _guard: &'a NoGc<'a>) -> HeapRef<'a, T> {
        HeapRef {
            ptr: self.into_ptr(),
            _phantom: PhantomData,
        }
    }
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

/// A GC-tracked slot holding a tagged word with a type hint.
/// Values flow in and out as `Tagged<T>`; the slot never decodes.
impl<T> GcSlot<T> {
    pub fn get(&self) -> Tagged<T> {
        unsafe { Tagged::from_value_unchecked(*self.raw.get()) }
    }

    pub fn set(&self, heap: &impl LocalHeap, host: Value, value: impl Into<Tagged<T>>) {
        let v = value.into().erase();
        if v.is_ptr() {
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
