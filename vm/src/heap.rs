use std::sync::Arc;

use crate::{
    AllocError, FixedArray, Float, GcHost, Global, Handle, HandleScope, HandleSet, HeapBackend,
    HeapObject, HeapPtr, HeapStats, LocalHeap, Map, Object, ObjectInit, ObjectSlotsInit, RawCell,
    RootHandles, STRONG_PTR, SharedHeap, Smi, TAG_MASK, Tagged, TransitionLock, Value, Visitor,
    Word,
};

use crate::bootstrap::{KnownCell, WellKnown};

use core::{alloc::Layout, cell::Cell, marker::PhantomData, ops::FnOnce, ptr::NonNull};
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

    /// Total layout for parts carved in order; matches the 16-byte minimum
    /// alignment of [`AllocToken::allocate`].
    pub fn total_for(parts: &[Layout]) -> Layout {
        let mut end = 0usize;
        for part in parts {
            end = end.next_multiple_of(part.align().max(16)) + part.size();
        }
        Layout::from_size_align(end, 16).expect("token layout")
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
        let aligned = (self.next.get() as usize).next_multiple_of(layout.align().max(16));
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

/// A slot that is either empty (the hole) or holds a strong reference to `T`.
///
/// Empty is encoded as the well-known hole object, so the slot is always a
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

    pub fn clear(&self, heap: &Heap) {
        self.slot
            .cell
            .store_raw(heap.known().the_hole.value().to_bits());
    }
}

impl<T: HeapObject> OptionGcSlot<T> {
    pub fn heap_ref<'a>(&self, nogc: &'a NoGc<'a>) -> Option<HeapRef<'a, T>> {
        if self.inner() == nogc.known().the_hole.value() {
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
    fn visit_edges(&self, visitor: &mut dyn Visitor);
}

/// Type-erased per-thread heap.
pub struct Heap {
    local: Box<dyn LocalHeap>,
    transition_lock: TransitionLock,
    known: *const KnownCell,
}

unsafe impl Send for Heap {}

impl Heap {
    pub fn allocate_raw(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.local.allocate_raw(layout)
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
        self.local
            .write_barrier(host.to_bits(), slot, value.to_bits())
    }

    pub fn collection_requested(&self) -> bool {
        self.local.collection_requested()
    }

    pub fn park_for_collection(&self) {
        self.local.park_for_collection()
    }

    pub fn gc_in_progress(&self) -> bool {
        self.local.gc_in_progress()
    }

    pub fn safepoint_poll(&mut self) {
        if self.collection_requested() {
            self.park_for_collection();
        }
    }

    pub fn collect(&mut self) {
        self.local.force_collect();
    }

    pub fn collect_minor(&mut self) {
        self.local.collect_minor();
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
        let layout =
            Layout::from_size_align(total.size(), total.align().max(16)).expect("token layout");
        let raw = self
            .allocate_raw(layout)
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

impl core::fmt::Debug for Heap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Heap").finish_non_exhaustive()
    }
}

pub struct GlobalHeap {
    shared: Arc<dyn SharedHeap>,
    transition_lock: TransitionLock,
}

impl GlobalHeap {
    pub fn new(shared: Arc<dyn SharedHeap>) -> Self {
        Self {
            shared,
            transition_lock: TransitionLock::new(),
        }
    }

    /// Build the shared heap from any backend configuration.
    pub fn from_backend<B: HeapBackend>(config: B::Config) -> Result<Self, AllocError> {
        Ok(Self::new(B::new(config)?.into_shared()))
    }

    pub fn new_local(&self, known: &KnownCell) -> Heap {
        Heap {
            local: self.shared.new_local(),
            transition_lock: self.transition_lock.clone(),
            known: known as *const KnownCell,
        }
    }

    pub fn iterate_roots(&self, roots: &mut dyn Visitor) {
        self.shared.iterate_roots(roots)
    }

    pub fn set_host(&self, host: GcHost) {
        self.shared.set_host(host)
    }

    pub fn should_collect(&self) -> bool {
        self.shared.should_collect()
    }

    pub fn gc_in_progress(&self) -> bool {
        self.shared.gc_in_progress()
    }

    /// Run one full collection cycle synchronously. Must not be called from
    /// a thread that owns a local heap.
    pub fn collect(&self) {
        self.shared.force_collect()
    }

    pub fn contains(&self, addr: Word) -> bool {
        self.shared.contains(addr)
    }

    pub fn is_young(&self, value: Value) -> bool {
        self.shared.is_young(value.to_bits())
    }

    pub fn stats(&self) -> HeapStats {
        self.shared.stats()
    }
}

impl core::fmt::Debug for GlobalHeap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GlobalHeap").finish_non_exhaustive()
    }
}
