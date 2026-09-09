use crate::{
    AllocError, FixedArray, Float, Global, GlobalVtable, Handle, HandleScope, HandleSet,
    HeapBackend, HeapObject, HeapPtr, HeapStats, HeapVtable, Map, Object, ObjectInit,
    ObjectSlotsInit, RawCell, RootHandles, RootVisitor, STRONG_PTR, Smi, TAG_MASK, Tagged,
    TransitionLock, Value, Visitor, Word,
};

use crate::bootstrap::{KnownCell, WellKnown};

use core::{
    alloc::Layout, cell::Cell, marker::PhantomData, ops::FnOnce, ptr::NonNull,
};
pub struct NoGc<'a> {
    heap: &'a Heap,
    _phantom: PhantomData<&'a mut &'a ()>,
}

impl<'a> NoGc<'a> {
    pub(crate) fn new(heap: &'a Heap) -> Self {
        Self {
            heap,
            _phantom: PhantomData,
        }
    }

    /// The heap this guard protects, for read access and slot writes.
    pub fn heap(&self) -> &'a Heap {
        self.heap
    }
}

/// Non-allocating heap calls go through the guard: while it is alive the
/// heap is borrowed, so no collection can happen.
impl core::ops::Deref for NoGc<'_> {
    type Target = Heap;

    fn deref(&self) -> &Heap {
        self.heap
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

    pub fn into_handle<'s>(self, scope: &'s impl HandleSet) -> Handle<'s, T> {
        scope.create_handle(self.into_tagged())
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

    pub fn into_handle<'s>(self, scope: &'s impl HandleSet) -> Handle<'s, T> {
        scope.create_handle(self.into_tagged())
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
        // the token's reservation already proves no GC can happen here
        let nogc = NoGc::new(&*self.heap);
        unsafe { ptr.as_mut() }.init(&nogc, &config);
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

    pub fn enter_no_gc<R>(&self, f: impl for<'a> FnOnce(&'a mut NoGc<'a>) -> R) -> R {
        let mut guard = NoGc::new(&*self.heap);
        f(&mut guard)
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
            cell: unsafe {
                RawCell::from_word(Tagged::from_ptr(ptr).make_weak().erase().to_bits())
            },
            _phantom: PhantomData,
        }
    }

    /// A "maybe-weak" cell holding a strong reference for now: the GC treats
    /// it as reachable until we decide to weaken selected entries.
    pub fn new_strong(ptr: HeapPtr<T>) -> Self {
        Self {
            cell: unsafe { RawCell::from_word(Tagged::from_ptr(ptr).erase().to_bits()) },
            _phantom: PhantomData,
        }
    }

    pub fn as_raw(&self) -> &RawCell {
        &self.cell
    }

    pub fn is_cleared(&self) -> bool {
        Value::from_bits(self.cell.load()).is_cleared()
    }

    // TODO: this maybe doens't make much sense
    // if a WeakGcCell is always weak, then upgrading it doesn't actually upgrade but only pretend
    pub fn upgrade<'a>(&self, _nogc: &'a NoGc<'a>) -> Option<HeapRef<'a, T>> {
        let word = self.cell.load();
        if word == crate::WEAK_PTR {
            return None;
        }
        let strong = Value::from_bits(word & !TAG_MASK | STRONG_PTR);
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
            cell: unsafe { RawCell::from_word(v.to_bits()) },
            _phantom: PhantomData,
        }
    }

    pub fn get(&self) -> Tagged<T> {
        unsafe { Tagged::from_value_unchecked(self.inner()) }
    }

    pub fn inner(&self) -> Value {
        Value::from_bits(self.cell.load())
    }

    pub fn set(&self, nogc: &NoGc<'_>, host: impl Into<Value>, value: impl Into<Tagged<T>>) {
        let host = host.into();
        let v = value.into().erase();
        debug_assert!(!v.is_weak_ptr(), "weak value stored into a strong slot");
        if v.is_ptr() {
            nogc.write_barrier(host, self.as_raw(), v);
        }
        self.cell.store_raw(v.to_bits());
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
    pub fn heap_ref<'a>(&self, _nogc: &'a NoGc<'a>) -> HeapRef<'a, T> {
        // Safe: `T` is this slot's declared type.
        unsafe { HeapRef::from_ptr(Tagged::from_value_unchecked(self.inner()).into()) }
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

    pub fn set(&self, nogc: &NoGc<'_>, host: impl Into<Value>, value: impl Into<Tagged<T>>) {
        self.slot.set(nogc, host, value);
    }

    pub fn clear(&self, void: Value) {
        self.slot.cell.store_raw(void.to_bits());
    }
}

impl<T: HeapObject> OptionGcSlot<T> {
    pub fn heap_ref<'a>(&self, nogc: &'a NoGc<'a>) -> Option<HeapRef<'a, T>> {
        if self.inner() == nogc.known().void.value() {
            return None;
        }
        Some(self.slot.heap_ref(nogc))
    }
}

#[repr(transparent)]
pub struct Register(RawCell);

impl Register {
    pub unsafe fn from_value(v: Value) -> Self {
        Self(unsafe { RawCell::from_word(v.to_bits()) })
    }

    pub fn inner(&self) -> Value {
        Value::from_bits(self.0.load())
    }

    pub fn store(&self, v: Value) {
        debug_assert!(!v.is_weak_ptr(), "weak value stored into a strong slot");
        self.0.store_raw(v.to_bits());
    }

    pub fn heap_ref<'a, T: HeapObject>(&self, _nogc: &'a NoGc<'a>) -> HeapRef<'a, T> {
        unsafe { HeapRef::from_ptr(Tagged::from_value_unchecked(self.inner()).into()) }
    }

    pub fn as_raw(&self) -> &RawCell {
        &self.0
    }
}

pub trait EdgeVisitable {
    fn visit_edges(&self, visitor: &mut impl Visitor);
}


/// Type-erased per-thread heap.
pub struct Heap {
    shared: *const (),
    local: *mut (),
    vtable: &'static HeapVtable,
    transition_lock: TransitionLock,
    known: *const KnownCell,
}

unsafe impl Send for Heap {}
unsafe impl Sync for Heap {}

impl Heap {
    pub fn allocate_raw(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        (self.vtable.allocate_raw)(self.local, layout)
    }

    pub fn known(&self) -> &'static WellKnown {
        unsafe { (*self.known).get() }
    }

    pub fn set_known(&self, known: WellKnown) {
        unsafe { (*self.known).set(known) }
    }

    pub fn transition_lock(&self) -> TransitionLock {
        self.transition_lock.clone()
    }

    pub fn write_barrier(&self, host: Value, slot: &RawCell, value: Value) {
        (self.vtable.write_barrier)(self.local, host.to_bits(), slot, value.to_bits())
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
        let raw = self
            .allocate_raw(T::layout_for(&config))
            .expect("heap allocation failed (out of memory)");
        let mut ptr = raw.cast::<T>();
        let nogc = NoGc::new(self);
        unsafe { ptr.as_mut() }.init(&nogc, &config);
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
            self.known().empty_fixed_array
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

    /// Fresh ordinary object: `map`, optional inline values, empty elements,
    /// length 0.
    pub fn new_object<'a>(
        &mut self,
        handles: &'a impl HandleSet,
        map: Handle<'a, Map>,
        values: &'a [Value],
    ) -> Fresh<'_, Object> {
        self.allocate_object(
            handles,
            ObjectSlotsInit {
                map,
                values,
                elements: self.known().empty_fixed_array.erase(),
                length: 0,
            },
        )
    }

    /// A number value: a Smi when the double is an in-range integer, a
    /// freshly boxed Float otherwise. `-0.0` always boxes (it must not
    /// collapse into `+0`).
    pub fn new_number(&mut self, scope: &HandleScope<'_>, f: f64) -> Value {
        let r = f as i64; // saturating cast; the round-trip check rejects out-of-range values
        if f.is_finite()
            && f.fract() == 0.0
            && Smi::in_range(r)
            && (r as f64) == f
            && !(f == 0.0 && f.is_sign_negative())
        {
            return Smi::new(r).encode();
        }
        self.allocate_handle::<Float>(f, scope).value()
    }

    pub fn allocate_enter_nogc<T: HeapObject, R>(
        &mut self,
        config: T::Init<'_>,
        f: impl for<'a> FnOnce(HeapRef<'a, T>, &'a mut NoGc<'a>) -> R,
    ) -> R {
        let ptr = self.allocate(config).into_ptr();
        self.no_gc(move |nogc| {
            let r = unsafe { HeapRef::from_ptr(ptr) };
            f(r, nogc)
        })
    }

    pub fn allocate_token(&mut self, total: Layout) -> AllocToken<'_> {
        let raw = self
            .allocate_raw(total)
            .expect("heap allocation failed (out of memory)");
        AllocToken::new(self, raw, total)
    }

    pub fn allocate_token_enter_nogc<R>(
        &mut self,
        total: Layout,
        f: impl for<'a> FnOnce(&AllocToken<'_>, &'a mut NoGc<'a>) -> R,
    ) -> R {
        let token = self.allocate_token(total);
        token.enter_no_gc(|nogc| f(&token, nogc))
    }

    pub fn guard(&mut self) -> NoGc<'_> {
        NoGc::new(self)
    }

    pub fn no_gc<R>(&mut self, f: impl for<'a> FnOnce(&'a mut NoGc<'a>) -> R) -> R {
        let mut guard = NoGc::new(self);
        f(&mut guard)
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
pub struct GlobalHeap {
    state: *mut (),
    vtable: &'static GlobalVtable,
    transition_lock: TransitionLock,
}

unsafe impl Send for GlobalHeap {}
unsafe impl Sync for GlobalHeap {}

impl GlobalHeap {
    pub fn new(state: *mut (), vtable: &'static GlobalVtable) -> Self {
        Self {
            state,
            vtable,
            transition_lock: TransitionLock::new(),
        }
    }

    /// Build the shared heap from any backend configuration.
    pub fn from_backend<B: HeapBackend>(config: B::Config) -> Result<Self, AllocError> {
        let (state, vtable) = B::new(config)?.into_global();
        Ok(Self::new(state, vtable))
    }

    pub fn new_local(&self, known: &KnownCell) -> Heap {
        let local = (self.vtable.new_local)(self.state);
        Heap {
            shared: self.state,
            local,
            vtable: self.vtable.local_vtable,
            transition_lock: self.transition_lock.clone(),
            known: known as *const KnownCell,
        }
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
        (self.vtable.is_young)(self.state, value.to_bits())
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
