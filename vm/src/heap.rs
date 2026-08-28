use core::{
    alloc::Layout,
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    ops::FnOnce,
    ptr::NonNull,
};
use crate::{
    FixedArray, Global, Handle, HandleScope, HandleSet, HeapObject, HeapPtr, Map, MapKind, Object,
    ObjectInit, ObjectSlotsInit, RootHandles, Smi, STRONG_PTR, Tagged, TransitionLock, Value, Word,
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
    pub map_map: Global<Map>,
    /// "hole"
    pub void: Global<Object>,
    pub smi_map: Global<Map>,
    pub float_map: Global<Map>,
    pub array_map: Global<Map>,
    pub byte_array_map: Global<Map>,
    pub string_map: Global<Map>,
    pub symbol_map: Global<Map>,
    pub accessor_pair_map: Global<Map>,
    pub callable_map: Global<Map>,
}

pub trait Heap: Sized + Send + Sync {
    type Config;
    type Local: LocalHeap;

    fn new(config: Self::Config) -> Result<Self, AllocError>;

    fn new_local(&self) -> Self::Local;

    fn set_known(&self, known: WellKnown);

    fn install_well_known_maps(&self) {
        let mut local = self.new_local();
        let roots = unsafe { RootHandles::new(64, Smi::new(0).encode()) };

        fn bootstrap_map(
            local: &mut impl LocalHeap,
            roots: &RootHandles,
            map_map: Global<Map>,
            kind: MapKind,
        ) -> Global<Map> {
            let raw = local
                .allocate_raw(Map::layout_for(0))
                .expect("bootstrap map allocation");
            let ptr = unsafe { HeapPtr::<Map>::new(raw.as_ptr().cast()) };
            let global = roots.create_handle(Tagged::from_ptr(ptr));
            let map = unsafe { global.get().as_mut() };
            let host = map.erase();
            map.header.map.set(local, host, map_map.as_tagged());
            map.value_slot_count.set(local, host, Smi::new(0));
            map.descriptor_count.set(local, host, Smi::new(0));
            map.kind.set(local, host, Smi::new(kind.bits() as i64));
            // placeholder: `void` does not exist yet, fixed up below
            map.transitions.as_raw().store_raw(Smi::new(0).encode());
            global
        }

        let map_map = {
            let raw = local
                .allocate_raw(Map::layout_for(0))
                .expect("bootstrap map map allocation");
            let ptr = unsafe { HeapPtr::<Map>::new(raw.as_ptr().cast()) };
            let tagged = Tagged::from_ptr(ptr);
            let map = unsafe { ptr.as_mut() };
            let host = map.erase();
            map.header.map.set(&local, host, tagged);
            map.value_slot_count.set(&local, host, Smi::new(0));
            map.descriptor_count.set(&local, host, Smi::new(0));
            map.kind
                .set(&local, host, Smi::new(MapKind::MAP.bits() as i64));
            // placeholder: `void` does not exist yet, fixed up below
            map.transitions.as_raw().store_raw(Smi::new(0).encode());
            roots.create_handle(tagged)
        };
        let void_map = bootstrap_map(&mut local, &roots, map_map, MapKind::OBJECT);

        let void = {
            let raw = local
                .allocate_raw(Object::layout_for())
                .expect("bootstrap void allocation");
            let ptr = unsafe { HeapPtr::<Object>::new(raw.as_ptr().cast()) };
            let tagged = Tagged::from_ptr(ptr);
            let obj = unsafe { ptr.as_mut() };
            let host = obj.erase();
            obj.header.map.set(&local, host, void_map.as_tagged());
            obj.slots.set(&local, host, unsafe { tagged.cast() });
            obj.elements.set(&local, host, tagged.erase_tagged());
            obj.length.set(&local, host, Smi::new(0));
            roots.create_handle(tagged)
        };

        let smi_map = bootstrap_map(&mut local, &roots, map_map, MapKind::OBJECT);
        let float_map = bootstrap_map(&mut local, &roots, map_map, MapKind::FLOAT);
        let array_map = bootstrap_map(&mut local, &roots, map_map, MapKind::FIXED_ARRAY);
        let byte_array_map =
            bootstrap_map(&mut local, &roots, map_map, MapKind::FIXED_BYTE_ARRAY);
        let string_map = bootstrap_map(&mut local, &roots, map_map, MapKind::VM_STRING);
        let symbol_map = bootstrap_map(&mut local, &roots, map_map, MapKind::SYMBOL);
        let accessor_pair_map =
            bootstrap_map(&mut local, &roots, map_map, MapKind::ACCESSOR_PAIR);
        let callable_map = bootstrap_map(&mut local, &roots, map_map, MapKind::CALLABLE_INFO);

        // Fix up the bootstrap maps' transition slots now that `void` exists:
        // from here on a map's transitions are either empty (`void`) or a
        // `FixedArray` of `[name, target_map]` pairs.
        local.no_gc(|nogc, _| {
            for map in [
                &map_map,
                &void_map,
                &smi_map,
                &float_map,
                &array_map,
                &byte_array_map,
                &string_map,
                &symbol_map,
                &accessor_pair_map,
                &callable_map,
            ] {
                map.heap_ref(nogc).transitions.clear(void.value());
            }
        });

        let known = WellKnown {
            map_map,
            void,
            smi_map,
            float_map,
            array_map,
            byte_array_map,
            string_map,
            symbol_map,
            accessor_pair_map,
            callable_map,
            roots,
        };
        self.set_known(known);
    }

    fn iterate_roots(&self, roots: &mut impl RootVisitor);

    fn collect(&self);

    fn should_collect(&self) -> bool;
    fn gc_in_progress(&self) -> bool;

    fn contains(&self, addr: Word) -> bool;
    fn is_young(&self, _value: Value) -> bool;
}

pub trait LocalHeap: Sized + Send {
    fn allocate_raw(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocError>;

    fn known(&self) -> &WellKnown;

    fn transition_lock(&self) -> TransitionLock;

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

    // TODO: potentially remove this in favor of a better allocate function
    fn allocate_object<'a>(
        &mut self,
        handles: &'a impl HandleSet,
        config: ObjectSlotsInit<'a, '_>,
    ) -> Fresh<'_, Object> {
        let slots = handles.create_handle(self.allocate::<FixedArray>(config.values).into_tagged());
        self.allocate::<Object>(ObjectInit {
            map: config.map,
            slots,
            elements: config.elements,
            length: config.length,
        })
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

    fn write_barrier(&self, host: Value, slot: &RawCell, value: Value);

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
            ptr: unsafe { HeapPtr::new(r as *const T as *mut T) },
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
        unsafe { HeapPtr::new(self.ptr.as_ptr()) }
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

    pub fn into_global(self, roots: &RootHandles) -> Global<T> {
        roots.create_handle(self.into_tagged())
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
    pub fn allocate<T: HeapObject>(&self, config: T::Init<'_>) -> Fresh<'heap, T> {
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

#[repr(transparent)]
pub struct RawCell {
    raw: UnsafeCell<Value>,
}

impl RawCell {
    pub unsafe fn from_value(v: Value) -> Self {
        Self {
            raw: UnsafeCell::new(v),
        }
    }

    pub fn load(&self) -> Value {
        unsafe { *self.raw.get() }
    }

    pub fn store_raw(&self, v: Value) {
        unsafe { *self.raw.get() = v };
    }

    pub unsafe fn heap_ref<'a, T: HeapObject>(&self, _nogc: &'a NoGc<'a>) -> HeapRef<'a, T> {
        unsafe { HeapRef::from_ptr(Tagged::from_value_unchecked(self.load()).into()) }
    }
}

/// A strong cell holding a weak reference: does not keep the target
/// alive and is cleared by the GC once the target dies.
/// TODO: potentially remove this in favor of weak collections (vm intern table)
/// and having "MaybeWeakGcCell" for VM objects that have weak semantics
#[repr(transparent)]
pub struct WeakGcCell<T: HeapObject> {
    cell: RawCell,
    _phantom: PhantomData<T>,
}

unsafe impl<T: HeapObject> Send for WeakGcCell<T> {}
unsafe impl<T: HeapObject> Sync for WeakGcCell<T> {}

impl<T: HeapObject> WeakGcCell<T> {
    pub fn new(ptr: HeapPtr<T>) -> Self {
        Self {
            cell: unsafe { RawCell::from_value(ptr.encode_weak()) },
            _phantom: PhantomData,
        }
    }

    pub fn as_raw(&self) -> &RawCell {
        &self.cell
    }

    pub fn is_cleared(&self) -> bool {
        self.cell.load().is_cleared()
    }

    // TODO: this maybe doens't make much sense
    // if a WeakGcCell is always weak, then upgrading it doesn't actually upgrade but only pretend
    pub fn upgrade<'a>(&self, _nogc: &'a NoGc<'a>) -> Option<HeapRef<'a, T>> {
        let word = self.cell.load();
        if word.is_cleared() {
            return None;
        }
        let strong = unsafe { Value::from_bits(word.raw_addr() | STRONG_PTR) };
        Some(unsafe { HeapRef::from_ptr(Tagged::from_value_unchecked(strong).into()) })
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
    cell: RawCell,
    _phantom: PhantomData<T>,
}
impl<T> GcSlot<T> {
    pub unsafe fn from_value(v: Value) -> Self {
        Self {
            cell: unsafe { RawCell::from_value(v) },
            _phantom: PhantomData,
        }
    }

    pub fn get(&self) -> Tagged<T> {
        unsafe { Tagged::from_value_unchecked(self.cell.load()) }
    }

    pub fn inner(&self) -> Value {
        self.cell.load()
    }

    pub fn set(&self, heap: &impl LocalHeap, host: Value, value: impl Into<Tagged<T>>) {
        let v = value.into().erase();
        debug_assert!(!v.is_weak_ptr(), "weak value stored into a strong slot");
        if v.is_ptr() {
            heap.write_barrier(host, self.as_raw(), v);
        }
        self.cell.store_raw(v);
    }

    pub fn as_raw(&self) -> &RawCell {
        &self.cell
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

impl<T: HeapObject> GcSlot<T> {
    /// Reads the slot as a heap reference valid for the no-GC scope.
    pub fn heap_ref<'a>(&self, nogc: &'a NoGc<'a>) -> HeapRef<'a, T> {
        // Safe: `T` is this slot's declared type.
        unsafe { self.cell.heap_ref(nogc) }
    }
}

/// A slot that is either empty (`void`) or holds a strong reference to `T`.
///
/// Empty is encoded as the well-known `void` object, so the slot is always a
/// valid strong value that the GC can trace.
#[repr(transparent)]
pub struct OptionGcSlot<T = Value> {
    slot: GcSlot<T>,
}

impl<T> OptionGcSlot<T> {
    pub unsafe fn from_value(v: Value) -> Self {
        Self {
            slot: unsafe { GcSlot::from_value(v) },
        }
    }

    pub fn inner(&self) -> Value {
        self.slot.inner()
    }

    pub fn as_raw(&self) -> &RawCell {
        self.slot.as_raw()
    }

    pub fn set(&self, heap: &impl LocalHeap, host: Value, value: impl Into<Tagged<T>>) {
        self.slot.set(heap, host, value);
    }

    pub fn clear(&self, void: Value) {
        self.slot.cell.store_raw(void);
    }
}

impl<T: HeapObject> OptionGcSlot<T> {
    pub fn heap_ref<'a>(
        &self,
        nogc: &'a NoGc<'a>,
        heap: &impl LocalHeap,
    ) -> Option<HeapRef<'a, T>> {
        if self.inner() == heap.known().void.value() {
            return None;
        }
        Some(self.slot.heap_ref(nogc))
    }
}

#[repr(transparent)]
pub struct Register(RawCell);

impl Register {
    pub unsafe fn from_value(v: Value) -> Self {
        Self(unsafe { RawCell::from_value(v) })
    }

    pub fn inner(&self) -> Value {
        self.0.load()
    }

    pub fn store(&self, v: Value) {
        debug_assert!(!v.is_weak_ptr(), "weak value stored into a strong slot");
        self.0.store_raw(v);
    }

    pub fn heap_ref<'a, T: HeapObject>(&self, nogc: &'a NoGc<'a>) -> HeapRef<'a, T> {
        unsafe { self.0.heap_ref(nogc) }
    }

    pub fn as_raw(&self) -> &RawCell {
        &self.0
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

pub trait EdgeVisitable {
    fn visit_edges(&self, visitor: &mut impl Visitor);
}
