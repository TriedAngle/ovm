use core::{
    alloc::Layout,
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    ops::FnOnce,
    ptr::NonNull,
};

use crate::{
    Global, Handle, HandleScope, HeapObject, HeapPtr, Map, MapInit, RootHandles, SlotsObject,
    SlotsObjectInit, Smi, Tagged, Value, Word,
};

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

pub struct WellKnown {
    pub roots: RootHandles,
    pub meta_map: Global<Map>,
    /// The hole: fill for not-yet-written slots
    pub void: Global<SlotsObject>,
    pub smi_map: Global<Map>,
    pub float_map: Global<Map>,
    pub array_map: Global<Map>,
    pub byte_array_map: Global<Map>,
    pub string_map: Global<Map>,
    pub symbol_map: Global<Map>,
    pub accessor_pair_map: Global<Map>,
}

pub trait Heap: Sized + Send + Sync {
    type Config;
    type Local: LocalHeap;

    fn new(config: Self::Config) -> Result<Self, AllocError>;

    fn new_local(&self) -> Self::Local;

    fn set_known(&self, known: WellKnown);

    fn install_well_known_maps(&self) {
        let mut local = self.new_local();

        let meta_map = local
            .allocate::<Map>(MapInit {
                meta_map: None,
                value_slot_count: 0,
                descriptors: &[],
            })
            .into_tagged();
        let void_map = local
            .allocate::<Map>(MapInit {
                meta_map: Some(meta_map),
                value_slot_count: 0,
                descriptors: &[],
            })
            .into_tagged();
        let void = local
            .allocate::<SlotsObject>(SlotsObjectInit {
                map: void_map,
                values: &[],
            })
            .into_tagged();

        let roots = unsafe { RootHandles::new(9, void.erase()) };

        let mut new_map = |meta_map: Option<Tagged<Map>>| {
            let fresh = local.allocate::<Map>(MapInit {
                meta_map,
                value_slot_count: 0,
                descriptors: &[],
            });
            roots.create_handle(fresh.into_tagged())
        };

        let meta_map = roots.create_handle(meta_map);
        let meta = meta_map.as_tagged();
        let known = WellKnown {
            meta_map,
            void: roots.create_handle(void),
            smi_map: new_map(Some(meta)),
            float_map: new_map(Some(meta)),
            array_map: new_map(Some(meta)),
            byte_array_map: new_map(Some(meta)),
            string_map: new_map(Some(meta)),
            symbol_map: new_map(Some(meta)),
            accessor_pair_map: new_map(Some(meta)),
            roots,
        };
        self.set_known(known);
    }

    fn iterate_roots(&self, roots: &mut dyn RootVisitor);

    fn collect(&self);

    fn should_collect(&self) -> bool;
    fn gc_in_progress(&self) -> bool;

    fn contains(&self, addr: Word) -> bool;
    fn is_young(&self, _value: Value) -> bool;
}

pub trait LocalHeap: Sized + Send {
    fn allocate_raw(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocError>;

    fn known(&self) -> &WellKnown;

    fn allocate<T: HeapObject>(&mut self, config: T::Init<'_>) -> Fresh<'_, T> {
        let raw = self.allocate_raw(T::layout_for(&config));
        debug_assert!(raw.is_ok(), "allocation must not fail");
        let raw = unsafe { raw.unwrap_unchecked() };
        let mut ptr = raw.cast::<T>();
        unsafe { ptr.as_mut() }.init(self, &config);
        Fresh {
            ptr,
            _phantom: PhantomData,
        }
    }

    fn allocate_handle<'s, T: HeapObject>(
        &mut self,
        config: T::Init<'_>,
        scope: &'s HandleScope<'_>,
    ) -> Handle<'s, T> {
        self.allocate(config).into_handle(scope)
    }

    fn allocate_enter_nogc<T: HeapObject, R>(
        &mut self,
        config: T::Init<'_>,
        f: impl for<'a> FnOnce(HeapRef<'a, T>, &'a mut NoGc<'a>, &'a Self) -> R,
    ) -> R {
        let ptr = self.allocate(config).into_ptr();
        self.no_gc(move |nogc, heap| {
            let r = HeapRef {
                ptr,
                _phantom: PhantomData,
            };
            f(r, nogc, heap)
        })
    }

    fn allocate_token(&mut self, total: Layout) -> AllocToken<'_, Self> {
        let raw = self.allocate_raw(total);
        debug_assert!(raw.is_ok(), "allocation must not fail");
        let raw = unsafe { raw.unwrap_unchecked() };
        AllocToken {
            heap: self,
            next: Cell::new(raw.as_ptr()),
            end: unsafe { raw.as_ptr().add(total.size()) },
        }
    }

    fn allocate_token_enter_nogc<R>(
        &mut self,
        total: Layout,
        f: impl for<'a> FnOnce(&AllocToken<'_, Self>, &'a mut NoGc<'a>, &Self) -> R,
    ) -> R {
        let token = self.allocate_token(total);
        token.enter_no_gc(|nogc, heap| f(&token, nogc, heap))
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

    pub fn get_as<T: HeapObject>(
        &'a self,
        v: Value,
        expected: Global<Map>,
    ) -> Option<HeapRef<'a, T>> {
        let ptr = HeapPtr::<T>::decode_strong(v)?;
        let map = unsafe { ptr.as_ref() }.header().map.get();
        if !map.ptr_eq(expected.as_tagged()) {
            return None;
        }
        Some(unsafe { HeapRef::from_ptr(ptr) })
    }

    pub unsafe fn get_unchecked<T: HeapObject>(&'a self, v: Tagged<T>) -> HeapRef<'a, T> {
        HeapRef {
            ptr: v.into(),
            _phantom: PhantomData,
        }
    }
}

/// Direct reference to a heap object, valid only within a no-GC scope.
pub struct HeapRef<'scope, T: HeapObject> {
    ptr: HeapPtr<T>,
    _phantom: PhantomData<&'scope T>,
}

impl<T: HeapObject> Clone for HeapRef<'_, T> {
    fn clone(&self) -> Self {
        HeapRef {
            ptr: self.ptr,
            _phantom: PhantomData,
        }
    }
}

impl<'scope, T: HeapObject> HeapRef<'scope, T> {
    pub fn from_ref(r: &'scope T) -> Self {
        HeapRef {
            ptr: unsafe { HeapPtr::new_unchecked(r as *const T as *mut T) },
            _phantom: PhantomData,
        }
    }

    pub unsafe fn from_ptr(ptr: HeapPtr<T>) -> Self {
        HeapRef {
            ptr,
            _phantom: PhantomData,
        }
    }

    pub fn as_ref(&self) -> &'scope T {
        unsafe { self.ptr.as_ref() }
    }

    pub fn into_ptr(self) -> HeapPtr<T> {
        self.ptr
    }

    pub fn into_tagged(self) -> Tagged<T> {
        Tagged::from_ptr(self.ptr)
    }

    pub fn into_handle<'s>(self, scope: &'s HandleScope<'_>) -> Handle<'s, T> {
        unsafe { scope.create_handle_unchecked(self.into_tagged()) }
    }
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
    pub fn into_ptr(self) -> HeapPtr<T> {
        unsafe { HeapPtr::new_unchecked(self.ptr.as_ptr()) }
    }

    pub fn into_tagged(self) -> Tagged<T> {
        Tagged::from_ptr(self.into_ptr())
    }

    pub fn erase(self) -> Value {
        self.into_tagged().erase()
    }

    pub fn into_handle<'s>(self, scope: &'s HandleScope<'_>) -> Handle<'s, T> {
        unsafe { scope.create_handle_unchecked(self.into_tagged()) }
    }

    pub fn heap_ref<'a>(self, _guard: &'a NoGc<'a>) -> HeapRef<'a, T> {
        HeapRef {
            ptr: self.into_ptr(),
            _phantom: PhantomData,
        }
    }
}

pub struct AllocToken<'heap, H: LocalHeap> {
    heap: &'heap mut H,
    next: Cell<*mut u8>,
    end: *mut u8,
}

impl<'heap, H: LocalHeap> AllocToken<'heap, H> {
    pub fn allocate<T: HeapObject>(&self, config: T::Init<'_>) -> Fresh<'_, T> {
        let mut ptr = self.bump(T::layout_for(&config)).cast::<T>();
        unsafe { ptr.as_mut() }.init(&*self.heap, &config);
        Fresh {
            ptr,
            _phantom: PhantomData,
        }
    }

    pub fn allocate_ref<'g, T: HeapObject>(
        &self,
        config: T::Init<'_>,
        _guard: &'g NoGc<'g>,
    ) -> HeapRef<'g, T> {
        HeapRef {
            ptr: self.allocate(config).into_ptr(),
            _phantom: PhantomData,
        }
    }

    pub fn enter_no_gc<R>(&self, f: impl for<'a> FnOnce(&'a mut NoGc<'a>, &'a H) -> R) -> R {
        let mut guard = NoGc {
            _phantom: PhantomData,
        };
        f(&mut guard, &*self.heap)
    }

    pub fn remaining(&self) -> usize {
        self.end as usize - self.next.get() as usize
    }

    pub fn heap(&self) -> &H {
        &*self.heap
    }

    fn bump(&self, layout: Layout) -> NonNull<u8> {
        let aligned = (self.next.get() as usize).next_multiple_of(layout.align().max(4));
        let end = aligned + layout.size();
        assert!(end <= self.end as usize, "allocation token exhausted");
        self.next.set(end as *mut u8);
        unsafe { NonNull::new_unchecked(aligned as *mut u8) }
    }
}

impl<H: LocalHeap> Drop for AllocToken<'_, H> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let remaining = self.remaining();
            assert_eq!(
                remaining, 0,
                "allocation token dropped with {remaining} bytes unused"
            );
        }
    }
}

pub struct WeakGcCell<T: HeapObject> {
    raw: UnsafeCell<Value>,
    _phantom: PhantomData<T>,
}

unsafe impl<T: HeapObject> Send for WeakGcCell<T> {}
unsafe impl<T: HeapObject> Sync for WeakGcCell<T> {}

impl<T: HeapObject> WeakGcCell<T> {
    pub fn new(ptr: HeapPtr<T>) -> Self {
        Self {
            raw: UnsafeCell::new(ptr.encode_weak()),
            _phantom: PhantomData,
        }
    }

    pub fn is_cleared(&self) -> bool {
        unsafe { *self.raw.get() }.is_cleared()
    }

    pub fn upgrade<'a>(&self, guard: &'a NoGc<'a>) -> Option<HeapRef<'a, T>> {
        let word = unsafe { *self.raw.get() };
        if word.is_cleared() {
            return None;
        }
        let strong = Value::from_bits(word.raw_addr() | crate::value::STRONG_PTR);
        Some(unsafe { guard.get_unchecked(Tagged::from_value_unchecked(strong)) })
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
/// Values flow in and out as `Tagged<T>`
impl<T> GcSlot<T> {
    pub unsafe fn from_value(v: Value) -> Self {
        Self {
            raw: UnsafeCell::new(v),
            _phantom: PhantomData,
        }
    }

    pub fn get(&self) -> Tagged<T> {
        unsafe { Tagged::from_value_unchecked(*self.raw.get()) }
    }

    pub fn inner(&self) -> Value {
        unsafe { *self.raw.get() }
    }

    pub fn set(&self, heap: &impl LocalHeap, host: Value, value: impl Into<Tagged<T>>) {
        let v = value.into().erase();
        debug_assert!(!v.is_weak_ptr(), "weak value stored into a strong slot");
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

impl GcSlot<Smi> {
    pub fn to_smi(&self) -> Smi {
        Smi::decode(self.inner()).expect("GcSlot invariant violated")
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
