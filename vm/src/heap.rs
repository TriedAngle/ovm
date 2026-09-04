use crate::{
    Context, ContextInit, FixedArray, FixedByteArray, Global, Handle, HandleData, HandleScope,
    HandleSet, HeapObject, HeapPtr, InternedString, Map, MapInit, MapKind, Object, ObjectInit,
    ObjectSlotsInit, RootHandles, STRONG_PTR, SlotFlags, SlotName, Smi, Tagged, TransitionLock,
    Value, Word, string_content_hash,
};
use core::{
    alloc::Layout,
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    ops::FnOnce,
    ptr::NonNull,
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

#[derive(Clone, Copy)]
pub struct WellKnown {
    // vm internal machinery: sentinels, canonical empties, builtin maps
    pub map_map: Global<Map>,
    /// "hole"
    pub void: Global<Object>,
    /// Never user-visible: a call returning it signals a pending exception.
    pub exception: Global<Object>,
    /// exception map
    pub exception_map: Global<Map>,
    /// Canonical empty string (synced with intern)
    pub empty_string: Global<InternedString>,
    /// Shared empty backing for objects without data slots (never written in
    /// place; the first property store swaps in a fresh array).
    pub empty_fixed_array: Global<FixedArray>,
    /// TODO: Placeholder for now
    pub empty_context: Global<Context>,
    pub smi_map: Global<Map>,
    pub float_map: Global<Map>,
    pub array_map: Global<Map>,
    pub byte_array_map: Global<Map>,
    pub string_map: Global<Map>,
    pub symbol_map: Global<Map>,
    pub accessor_pair_map: Global<Map>,
    pub callable_map: Global<Map>,
    pub handler_table_map: Global<Map>,
    pub context_map: Global<Map>,
    // js userspace primitives and object base maps
    pub undefined: Global<Object>,
    pub undefined_map: Global<Map>,
    pub null: Global<Object>,
    pub null_map: Global<Map>,
    pub false_object: Global<Object>,
    pub true_object: Global<Object>,
    pub boolean_map: Global<Map>,
    /// Base map for ECMAScript array objects (elements + length, prototype
    /// %Array.prototype%).
    pub js_array_map: Global<Map>,
    /// Base map for ECMAScript error objects
    pub error_map: Global<Map>,
    // prototypes
    /// `%Object.prototype%`: root of the ordinary-object prototype hierarchy.
    pub object_prototype: Global<Object>,
    /// `%Array.prototype%`: an array object, parent of all array instance maps.
    pub array_prototype: Global<Object>,
    /// `%Error.prototype%`: parent of all error instance maps.
    pub error_prototype: Global<Object>,
    /// The realm global object: global variables are properties on it
    /// (top-level `var`/assignments; lexical script-context globals later).
    pub global_object: Global<Object>,
    /// Initial map of `%Object.prototype%`: every fresh `{}` gets it
    /// (extendable, parent = object_prototype). Shared with `global_object`.
    pub object_initial_map: Global<Map>,
    /// Function object map (the initial map of `%Function.prototype%`):
    /// slots[0] = shared CallableInfoObject, slots[1] = closure context.
    pub function_map: Global<Map>,
}

unsafe fn smi_handle<T>(roots: &RootHandles) -> Global<T> {
    roots.create_handle(unsafe { Tagged::from_value_unchecked(Smi::new(0).encode()) })
}

fn uninited_wellknown(roots: &RootHandles) -> WellKnown {
    let map = unsafe { smi_handle::<Map>(roots) };
    let obj = unsafe { smi_handle::<Object>(roots) };
    let string = unsafe { smi_handle::<InternedString>(roots) };
    let array = unsafe { smi_handle::<FixedArray>(roots) };
    let context = unsafe { smi_handle::<Context>(roots) };
    WellKnown {
        map_map: map,
        void: obj,
        exception: obj,
        exception_map: map,
        empty_string: string,
        empty_fixed_array: array,
        empty_context: context,
        smi_map: map,
        float_map: map,
        array_map: map,
        byte_array_map: map,
        string_map: map,
        symbol_map: map,
        accessor_pair_map: map,
        callable_map: map,
        handler_table_map: map,
        context_map: map,
        undefined: obj,
        undefined_map: map,
        null: obj,
        null_map: map,
        false_object: obj,
        true_object: obj,
        boolean_map: map,
        js_array_map: map,
        error_map: map,
        object_prototype: obj,
        array_prototype: obj,
        error_prototype: obj,
        global_object: obj,
        object_initial_map: map,
        function_map: map,
    }
}

fn alloc_map(heap: &mut Heap, roots: &RootHandles, kind: MapKind) -> Global<Map> {
    heap.allocate::<Map>(MapInit {
        kind,
        value_slot_count: 0,
        descriptors: &[],
    })
    .into_global(roots)
}

fn alloc_parent_map(
    heap: &mut Heap,
    roots: &RootHandles,
    kind: MapKind,
    parent: Global<Object>,
) -> Global<Map> {
    heap.allocate::<Map>(MapInit {
        kind,
        value_slot_count: 0,
        descriptors: &[(
            SlotName::from(Tagged::from_smi(Smi::new(1))),
            SlotFlags::CONST.union(SlotFlags::PARENT),
            parent.value(),
        )],
    })
    .into_global(roots)
}

fn alloc_object(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    roots: &RootHandles,
    map: Global<Map>,
) -> Global<Object> {
    heap.allocate_object(
        scope,
        ObjectSlotsInit {
            map,
            values: &[],
            elements: heap.known().empty_fixed_array.erase(),
            length: 0,
        },
    )
    .into_global(roots)
}

fn bootstrap_map_map_and_void(heap: &mut Heap, roots: &RootHandles) -> (Global<Map>, WellKnown) {
    let mut known = uninited_wellknown(roots);
    heap.set_known(known);

    let map_map = heap
        .allocate::<Map>(MapInit {
            kind: MapKind::MAP,
            value_slot_count: 0,
            descriptors: &[],
        })
        .into_global(roots);
    heap.no_gc(|nogc, heap| {
        map_map
            .heap_ref(nogc)
            .header
            .map
            .set(heap, map_map.value(), map_map.as_tagged());
    });
    known.map_map = map_map;

    let void_map = alloc_map(heap, roots, MapKind::OBJECT);
    let void = heap
        .allocate::<Object>(ObjectInit {
            map: void_map,
            slots: unsafe { smi_handle::<FixedArray>(roots) },
            elements: unsafe { smi_handle::<Value>(roots) },
            length: 0,
        })
        .into_global(roots);
    heap.no_gc(|nogc, _heap| {
        map_map.heap_ref(nogc).transitions.clear(void.value());
        void_map.heap_ref(nogc).transitions.clear(void.value());
    });
    known.void = void;
    heap.set_known(known);

    (void_map, known)
}

pub fn bootstrap_well_known(heap: &mut Heap, roots: &RootHandles) {
    let (_void_map, mut known) = bootstrap_map_map_and_void(heap, roots);
    let void = known.void;

    // All remaining maps: `Map::init` now reads the real map_map/void.
    let smi_map = alloc_map(heap, roots, MapKind::OBJECT);
    let float_map = alloc_map(heap, roots, MapKind::FLOAT);
    let array_map = alloc_map(heap, roots, MapKind::FIXED_ARRAY);
    let byte_array_map = alloc_map(heap, roots, MapKind::FIXED_BYTE_ARRAY);
    let string_map = alloc_map(heap, roots, MapKind::VM_STRING);
    let symbol_map = alloc_map(heap, roots, MapKind::SYMBOL);
    let accessor_pair_map = alloc_map(heap, roots, MapKind::ACCESSOR_PAIR);
    let callable_map = alloc_map(heap, roots, MapKind::CALLABLE_INFO);
    let handler_table_map = alloc_map(heap, roots, MapKind::HANDLER_TABLE);
    let context_map = alloc_map(heap, roots, MapKind::CONTEXT);
    // 2 slots: [0] callable info, [1] closure context
    let function_map = heap
        .allocate::<Map>(MapInit {
            kind: MapKind::OBJECT.union(MapKind::CALLABLE),
            value_slot_count: 2,
            descriptors: &[],
        })
        .into_global(roots);
    let object_prototype_map = alloc_map(heap, roots, MapKind::OBJECT.union(MapKind::EXTENDABLE));
    known.smi_map = smi_map;
    known.float_map = float_map;
    known.array_map = array_map;
    known.byte_array_map = byte_array_map;
    known.string_map = string_map;
    known.symbol_map = symbol_map;
    known.accessor_pair_map = accessor_pair_map;
    known.callable_map = callable_map;
    known.handler_table_map = handler_table_map;
    known.context_map = context_map;
    known.function_map = function_map;
    heap.set_known(known);

    let data = HandleData::new(void.value());
    let scope = unsafe { HandleScope::from_raw(NonNull::from(&data)) };

    let empty_slots = heap.allocate::<FixedArray>(&[]).into_global(roots);
    known.empty_fixed_array = heap.allocate::<FixedArray>(&[]).into_global(roots);
    heap.set_known(known);
    let object_prototype = alloc_object(heap, &scope, roots, object_prototype_map);

    let array_prototype_map = alloc_parent_map(
        heap,
        roots,
        MapKind::ARRAY.union(MapKind::EXTENDABLE),
        object_prototype,
    );
    let array_prototype = alloc_object(heap, &scope, roots, array_prototype_map);
    let js_array_map = alloc_parent_map(
        heap,
        roots,
        MapKind::ARRAY.union(MapKind::EXTENDABLE),
        array_prototype,
    );

    let error_prototype_map = alloc_parent_map(
        heap,
        roots,
        MapKind::OBJECT.union(MapKind::EXTENDABLE),
        object_prototype,
    );
    let error_prototype = alloc_object(heap, &scope, roots, error_prototype_map);

    let error_map = alloc_parent_map(
        heap,
        roots,
        MapKind::OBJECT.union(MapKind::EXTENDABLE),
        error_prototype,
    );

    let object_initial_map = alloc_parent_map(
        heap,
        roots,
        MapKind::OBJECT.union(MapKind::EXTENDABLE),
        object_prototype,
    );
    let global_object = alloc_object(heap, &scope, roots, object_initial_map);

    let undefined_map = alloc_parent_map(heap, roots, MapKind::OBJECT, object_prototype);
    let boolean_map = alloc_parent_map(heap, roots, MapKind::OBJECT, object_prototype);
    let null_map = alloc_map(heap, roots, MapKind::OBJECT);

    let undefined = alloc_object(heap, &scope, roots, undefined_map);
    let null = alloc_object(heap, &scope, roots, null_map);
    let false_object = alloc_object(heap, &scope, roots, boolean_map);
    let true_object = alloc_object(heap, &scope, roots, boolean_map);

    let exception_map = alloc_parent_map(heap, roots, MapKind::OBJECT, object_prototype);
    let exception = alloc_object(heap, &scope, roots, exception_map);

    let empty_string = {
        let backing = heap.allocate::<FixedByteArray>(&[]).into_handle(&scope);
        heap.allocate::<InternedString>((backing, string_content_hash(b"")))
            .into_global(roots)
    };

    let empty_context = heap
        .allocate::<Context>(ContextInit {
            outer: None,
            slots: empty_slots,
        })
        .into_global(roots);

    known.undefined = undefined;
    known.undefined_map = undefined_map;
    known.null = null;
    known.null_map = null_map;
    known.false_object = false_object;
    known.true_object = true_object;
    known.boolean_map = boolean_map;
    known.exception = exception;
    known.empty_string = empty_string;
    known.object_prototype = object_prototype;
    known.array_prototype = array_prototype;
    known.error_prototype = error_prototype;
    known.error_map = error_map;
    known.exception_map = exception_map;
    known.js_array_map = js_array_map;
    known.empty_context = empty_context;
    known.global_object = global_object;
    known.object_initial_map = object_initial_map;
    heap.set_known(known);
}
pub struct NoGc<'a> {
    _phantom: PhantomData<&'a mut &'a ()>,
}

impl NoGc<'_> {
    pub(crate) fn new() -> Self {
        Self {
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
    pub(crate) fn new(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _phantom: PhantomData,
        }
    }

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

pub struct AllocToken<'heap> {
    heap: &'heap mut Heap,
    next: Cell<*mut u8>,
    end: *mut u8,
}

impl<'heap> AllocToken<'heap> {
    pub(crate) fn new(heap: &'heap mut Heap, raw: NonNull<u8>, total: Layout) -> Self {
        Self {
            heap,
            next: Cell::new(raw.as_ptr()),
            end: unsafe { raw.as_ptr().add(total.size()) },
        }
    }

    pub fn allocate<T: HeapObject>(&self, config: T::Init<'_>) -> Fresh<'heap, T> {
        let mut ptr = self.bump(T::layout_for(&config)).cast::<T>();
        unsafe { ptr.as_mut() }.init(self.heap, &config);
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

    pub fn enter_no_gc<R>(&self, f: impl for<'a> FnOnce(&'a mut NoGc<'a>, &'a Heap) -> R) -> R {
        let mut guard = NoGc {
            _phantom: PhantomData,
        };
        f(&mut guard, &*self.heap)
    }

    pub fn remaining(&self) -> usize {
        self.end as usize - self.next.get() as usize
    }

    pub fn heap(&self) -> &Heap {
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

impl Drop for AllocToken<'_> {
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

    pub fn set(&self, heap: &Heap, host: Value, value: impl Into<Tagged<T>>) {
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

    pub fn set(&self, heap: &Heap, host: Value, value: impl Into<Tagged<T>>) {
        self.slot.set(heap, host, value);
    }

    pub fn clear(&self, void: Value) {
        self.slot.cell.store_raw(void);
    }
}

impl<T: HeapObject> OptionGcSlot<T> {
    pub fn heap_ref<'a>(&self, nogc: &'a NoGc<'a>, heap: &Heap) -> Option<HeapRef<'a, T>> {
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

/// Statistics reported by a [`GlobalHeap`] for introspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeapStats {
    pub used: usize,
    pub capacity: usize,
}

/// Function table of per-thread (local) heap operations.
pub struct HeapVtable {
    pub allocate_raw: fn(local: *mut (), layout: Layout) -> Result<NonNull<u8>, AllocError>,
    pub known: fn(local: *const ()) -> &'static WellKnown,
    pub set_known: fn(shared: *const (), known: WellKnown),
    pub transition_lock: fn(local: *const ()) -> TransitionLock,
    pub write_barrier: fn(local: *const (), host: Value, slot: &RawCell, value: Value),
    pub collection_requested: fn(local: *const ()) -> bool,
    pub park_for_collection: fn(local: *const ()),
    pub gc_in_progress: fn(local: *const ()) -> bool,
    pub drop_local: fn(local: *mut ()),
}

/// Function table of shared (global) heap operations.
pub struct GlobalVtable {
    /// Vtable used for [`Heap`]s created by [`GlobalHeap::new_local`].
    pub local_vtable: &'static HeapVtable,
    pub new_local: fn(shared: *const ()) -> *mut (),
    pub known: fn(shared: *const ()) -> &'static WellKnown,
    pub iterate_roots: fn(shared: *const (), roots: &mut dyn RootVisitor),
    pub collect: fn(shared: *const ()),
    pub should_collect: fn(shared: *const ()) -> bool,
    pub gc_in_progress: fn(shared: *const ()) -> bool,
    pub contains: fn(shared: *const (), addr: Word) -> bool,
    pub is_young: fn(shared: *const (), value: Value) -> bool,
    pub stats: fn(shared: *const ()) -> HeapStats,
    pub drop_shared: fn(shared: *mut ()),
}

/// Type-erased per-thread heap.
pub struct Heap {
    shared: *const (),
    local: *mut (),
    vtable: &'static HeapVtable,
}

unsafe impl Send for Heap {}
unsafe impl Sync for Heap {}

impl Heap {
    pub fn allocate_raw(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        (self.vtable.allocate_raw)(self.local, layout)
    }

    pub fn known(&self) -> &'static WellKnown {
        (self.vtable.known)(self.local)
    }

    fn set_known(&self, known: WellKnown) {
        (self.vtable.set_known)(self.shared, known)
    }

    pub fn transition_lock(&self) -> TransitionLock {
        (self.vtable.transition_lock)(self.local)
    }

    pub fn write_barrier(&self, host: Value, slot: &RawCell, value: Value) {
        (self.vtable.write_barrier)(self.local, host, slot, value)
    }

    pub fn collection_requested(&self) -> bool {
        (self.vtable.collection_requested)(self.local)
    }

    pub fn park_for_collection(&self) {
        (self.vtable.park_for_collection)(self.local)
    }

    pub fn gc_in_progress(&self) -> bool {
        (self.vtable.gc_in_progress)(self.local)
    }

    pub fn safepoint_poll(&mut self) {
        if self.collection_requested() {
            self.park_for_collection();
        }
    }

    pub fn allocate<T: HeapObject>(&mut self, config: T::Init<'_>) -> Fresh<'_, T> {
        let raw = self.allocate_raw(T::layout_for(&config));
        debug_assert!(raw.is_ok(), "allocation must not fail");
        let raw = unsafe { raw.unwrap_unchecked() };
        let mut ptr = raw.cast::<T>();
        unsafe { ptr.as_mut() }.init(self, &config);
        Fresh::new(ptr)
    }

    pub fn allocate_handle<'s, T: HeapObject>(
        &mut self,
        config: T::Init<'_>,
        scope: &'s HandleScope<'_>,
    ) -> Handle<'s, T> {
        self.allocate(config).into_handle(scope)
    }

    // TODO: potentially remove this in favor of a better allocate function
    pub fn allocate_object<'a>(
        &mut self,
        handles: &'a impl HandleSet,
        config: ObjectSlotsInit<'a, '_>,
    ) -> Fresh<'_, Object> {
        let slots = if config.values.is_empty() {
            handles.create_handle(self.known().empty_fixed_array.as_tagged())
        } else {
            handles.create_handle(self.allocate::<FixedArray>(config.values).into_tagged())
        };
        self.allocate::<Object>(ObjectInit {
            map: config.map,
            slots,
            elements: config.elements,
            length: config.length,
        })
    }

    pub fn allocate_enter_nogc<T: HeapObject, R>(
        &mut self,
        config: T::Init<'_>,
        f: impl for<'a> FnOnce(HeapRef<'a, T>, &'a mut NoGc<'a>, &'a Self) -> R,
    ) -> R {
        let ptr = self.allocate(config).into_ptr();
        self.no_gc(move |nogc, heap| {
            let r = unsafe { HeapRef::from_ptr(ptr) };
            f(r, nogc, heap)
        })
    }

    pub fn allocate_token(&mut self, total: Layout) -> AllocToken<'_> {
        let raw = self.allocate_raw(total);
        debug_assert!(raw.is_ok(), "allocation must not fail");
        let raw = unsafe { raw.unwrap_unchecked() };
        AllocToken::new(self, raw, total)
    }

    pub fn allocate_token_enter_nogc<R>(
        &mut self,
        total: Layout,
        f: impl for<'a> FnOnce(&AllocToken<'_>, &'a mut NoGc<'a>, &Self) -> R,
    ) -> R {
        let token = self.allocate_token(total);
        token.enter_no_gc(|nogc, heap| f(&token, nogc, heap))
    }

    pub fn no_gc<R>(&mut self, f: impl for<'a> FnOnce(&'a mut NoGc<'a>, &'a Self) -> R) -> R {
        let mut guard = NoGc::new();
        f(&mut guard, self)
    }
}

impl Drop for Heap {
    fn drop(&mut self) {
        (self.vtable.drop_local)(self.local);
    }
}

impl core::fmt::Debug for Heap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Heap")
            .field("local", &self.local)
            .field("shared", &self.shared)
            .finish_non_exhaustive()
    }
}

/// Type-erased shared heap. The single owner of the shared heap state.
pub struct GlobalHeap {
    state: *mut (),
    vtable: &'static GlobalVtable,
}

// SAFETY: the backend contract is that the pointed-to state is Send + Sync.
unsafe impl Send for GlobalHeap {}
unsafe impl Sync for GlobalHeap {}

impl GlobalHeap {
    pub fn new(state: *mut (), vtable: &'static GlobalVtable) -> Self {
        Self { state, vtable }
    }

    /// Create a per-thread local heap.
    pub fn new_local(&self) -> Heap {
        let local = (self.vtable.new_local)(self.state);
        Heap {
            shared: self.state,
            local,
            vtable: self.vtable.local_vtable,
        }
    }

    pub fn known(&self) -> &'static WellKnown {
        (self.vtable.known)(self.state)
    }

    pub fn iterate_roots(&self, roots: &mut impl RootVisitor) {
        (self.vtable.iterate_roots)(self.state, roots)
    }

    pub fn collect(&self) {
        (self.vtable.collect)(self.state)
    }

    pub fn should_collect(&self) -> bool {
        (self.vtable.should_collect)(self.state)
    }

    pub fn gc_in_progress(&self) -> bool {
        (self.vtable.gc_in_progress)(self.state)
    }

    pub fn contains(&self, addr: Word) -> bool {
        (self.vtable.contains)(self.state, addr)
    }

    pub fn is_young(&self, value: Value) -> bool {
        (self.vtable.is_young)(self.state, value)
    }

    pub fn stats(&self) -> HeapStats {
        (self.vtable.stats)(self.state)
    }
}

impl Drop for GlobalHeap {
    fn drop(&mut self) {
        (self.vtable.drop_shared)(self.state);
    }
}

impl core::fmt::Debug for GlobalHeap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GlobalHeap")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

/// Construction trait for concrete heap backends.
pub trait HeapBackend: Sized + Send + Sync {
    type Config;

    fn new(config: Self::Config) -> Result<Self, AllocError>;

    fn into_global(self) -> GlobalHeap;
}
