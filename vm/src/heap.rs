use std::sync::Arc;

use crate::{
    AllocError, FixedArray, Float, GcHost, Handle, HandleScope, HandleSet, HandleSlice,
    HeapBackend, HeapObject, HeapPtr, HeapStats, LocalHeap, Map, MaybeWeak, Object, ObjectInit,
    ObjectSlotsInit, RawCell, STRONG_PTR, SharedHeap, Smi, Tagged, TransitionLock, Value, Visitor,
    Word,
};

use crate::bootstrap::{KnownCell, WellKnown};

use crate::WEAK_PTR;
use core::{alloc::Layout, cell::Cell, marker::PhantomData, ops::FnOnce, ptr::NonNull};

pub struct AllocToken<'heap> {
    heap: &'heap mut Heap,
    next: Cell<*mut u8>,
    end: *mut u8,
}

impl<'heap> AllocToken<'heap> {
    /// # Safety
    /// `raw` must point to at least `total.size()` bytes reserved from
    /// `heap` for the token's whole lifetime, aligned to `total.align()`.
    /// `total` must match the summed layouts of every
    /// [`AllocToken::allocate`] call made on the returned token.
    pub unsafe fn new(heap: &'heap mut Heap, raw: NonNull<u8>, total: Layout) -> Self {
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

    pub fn allocate<T: HeapObject>(&self, config: T::Init<'_>) -> Tagged<'heap, T> {
        let mut ptr = self.bump(T::layout_for(&config)).cast::<T>();
        // the token's reservation already proves no GC can happen here
        unsafe { ptr.as_mut() }.init(self.heap(), &config);
        // Safety: fresh strong pointer; the token's heap borrow is the
        // anchor and no GC can run while it is outstanding.
        unsafe { Tagged::from_value_unchecked(HeapPtr::<T>::new(ptr.as_ptr()).encode_strong()) }
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

pub trait WordType: 'static {
    const IS_HEAP: bool;
}

impl WordType for Value {
    const IS_HEAP: bool = true;
}

impl WordType for Smi {
    const IS_HEAP: bool = false;
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

    /// Re-read the slot under a heap borrow: the word is current (the GC
    /// updates slots in place) and valid for `'a` because no collection
    /// can run while the borrow lives.
    pub fn get<'a>(&self, _heap: &'a Heap) -> Tagged<'a, T> {
        // Safety: see method docs.
        unsafe { Tagged::from_value_unchecked(Value::from_bits(self.cell.load())) }
    }

    pub fn inner(&self) -> Value {
        Value::from_bits(self.cell.load())
    }

    pub fn set<'h, 'x>(&self, heap: &Heap, host: Tagged<'h, Value>, value: impl Into<Tagged<'x, T>>)
    where
        T: 'x,
    {
        let value = value.into();
        let v = value.raw();
        debug_assert!(!v.is_weak_ptr(), "weak value stored into a strong slot");
        if v.is_ptr() {
            heap.write_barrier(host, self.as_raw(), value.erase());
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

#[repr(transparent)]
pub struct MaybeWeakGcSlot<T = Value> {
    cell: RawCell,
    _phantom: PhantomData<MaybeWeak<T>>,
}

unsafe impl<T> Send for MaybeWeakGcSlot<T> {}
unsafe impl<T> Sync for MaybeWeakGcSlot<T> {}

impl<T: HeapObject> MaybeWeakGcSlot<T> {
    /// A standalone cell holding a strong reference: the GC treats it as
    /// reachable until the entry is weakened.
    pub fn new_strong(ptr: HeapPtr<T>) -> Self {
        Self {
            // Safety: constructing the storage word of a fresh cell.
            cell: unsafe { RawCell::from_word(ptr.encode_strong().to_bits()) },
            _phantom: PhantomData,
        }
    }
}

impl<T> MaybeWeakGcSlot<T> {
    pub unsafe fn from_value(v: Value) -> Self {
        Self {
            cell: unsafe { RawCell::from_word(v.to_bits()) },
            _phantom: PhantomData,
        }
    }

    pub fn get<'a>(&self, _heap: &'a Heap) -> Tagged<'a, MaybeWeak<T>> {
        unsafe { Tagged::from_maybe_weak_unchecked(Value::from_bits(self.cell.load())) }
    }

    pub fn inner(&self) -> Value {
        Value::from_bits(self.cell.load())
    }

    pub fn is_cleared(&self) -> bool {
        self.inner().is_cleared()
    }

    pub fn as_raw(&self) -> &RawCell {
        &self.cell
    }
}

impl<T> MaybeWeakGcSlot<T> {
    pub fn set_strong<'h, 'x>(
        &self,
        heap: &Heap,
        host: Tagged<'h, Value>,
        value: impl Into<Tagged<'x, T>>,
    ) where
        T: 'x,
    {
        let value = value.into();
        let strong = value.raw();
        debug_assert!(
            !strong.is_weak_ptr(),
            "strong store of a weak/cleared word into a weak slot"
        );
        if strong.is_ptr() {
            heap.write_barrier(host, self.as_raw(), value.erase());
        }
        self.cell.store_raw(strong.to_bits());
    }

    pub fn set_weak<'h, 'x>(
        &self,
        heap: &Heap,
        host: Tagged<'h, Value>,
        value: impl Into<Tagged<'x, T>>,
    ) where
        T: 'x,
    {
        let value = value.into();
        let strong = value.raw();
        // the weak reference still participates in the generational
        // barrier: an old slot holding a young target must be remembered
        // so the minor collection can forward or clear it
        if strong.is_ptr() {
            heap.write_barrier(host, self.as_raw(), value.erase());
        }
        let weak = Value::from_bits(strong.to_bits() | WEAK_PTR);
        self.cell.store_raw(weak.to_bits());
    }

    /// Store an already-encoded maybe-weak word verbatim, taking the barrier
    /// when it points at a live heap object.
    pub fn set<'h, 'x>(
        &self,
        heap: &Heap,
        host: Tagged<'h, Value>,
        value: Tagged<'x, MaybeWeak<T>>,
    ) {
        if let Some(live) = value.upgrade() {
            heap.write_barrier(host, self.as_raw(), live.erase());
        }
        self.cell.store_raw(value.raw().to_bits());
    }

    pub fn upgrade<'a>(&self, _heap: &'a Heap) -> Option<Tagged<'a, T>> {
        let word = self.inner();
        if !word.is_ptr() || word.is_cleared() {
            return None;
        }
        let strong = Value::from_bits(word.raw_addr() | STRONG_PTR);
        Some(unsafe { Tagged::from_value_unchecked(strong) })
    }

    pub fn strengthen<'a>(&self, heap: &'a Heap) -> Option<Tagged<'a, T>> {
        self.get(heap).strengthen()
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

    pub fn get<'a>(&self, heap: &'a Heap) -> Option<Tagged<'a, T>> {
        // Safety: fresh root-slot read for a word comparison.
        if self.inner() == unsafe { heap.known().the_hole.read_unchecked() } {
            return None;
        }
        Some(self.slot.get(heap))
    }

    pub fn as_raw(&self) -> &RawCell {
        self.slot.as_raw()
    }

    pub fn set<'h, 'x>(&self, heap: &Heap, host: Tagged<'h, Value>, value: impl Into<Tagged<'x, T>>)
    where
        T: 'x,
    {
        self.slot.set(heap, host, value);
    }

    pub fn clear(&self, heap: &Heap) {
        // Safety: fresh root-slot read for a word store.
        self.slot
            .cell
            .store_raw(unsafe { heap.known().the_hole.read_unchecked() }.to_bits());
    }
}

#[repr(transparent)]
pub struct Register(RawCell);

impl Register {
    pub unsafe fn from_value(v: Value) -> Self {
        Self(unsafe { RawCell::from_word(v.to_bits()) })
    }

    /// Re-read the register under a heap borrow. Registers live in rooted
    /// memory that the GC updates in place, so the word is current; the
    /// anchor proves no GC runs before its use.
    pub fn read<'a>(&self, _heap: &'a Heap) -> Tagged<'a, Value> {
        // Safety: see method docs.
        unsafe { Tagged::from_value_unchecked(Value::from_bits(self.0.load())) }
    }

    /// Registers holding Smis (frame headers) can be read without an
    /// anchor: Smis never dangle.
    pub fn read_smi(&self) -> Smi {
        Smi::decode(Value::from_bits(self.0.load())).expect("register holds a Smi")
    }

    pub fn inner(&self) -> Value {
        Value::from_bits(self.0.load())
    }

    pub fn store<'x, T: 'x>(&self, v: Tagged<'x, T>) {
        let raw = v.raw();
        debug_assert!(!raw.is_weak_ptr(), "weak value stored into a strong slot");
        self.0.store_raw(raw.to_bits());
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
    #[cfg(feature = "stress-minor-gc")]
    stress_armed: std::sync::Arc<std::sync::atomic::AtomicBool>,
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

    pub fn write_barrier(&self, host: Tagged<'_, Value>, slot: &RawCell, value: Tagged<'_, Value>) {
        self.local
            .write_barrier(host.raw().to_bits(), slot, value.raw().to_bits())
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

    /// Allocate a fresh `T`. The returned `Tagged` is anchored at this
    /// borrow: any further allocation (or anything else requiring
    /// `&mut Heap`) requires rooting it first.
    pub fn allocate<'a, T: HeapObject>(&'a mut self, config: T::Init<'_>) -> Tagged<'a, T> {
        #[cfg(feature = "stress-minor-gc")]
        if self.stress_armed.load(std::sync::atomic::Ordering::Acquire) {
            self.collect_minor();
        }
        let raw = self
            .allocate_raw(T::layout_for(&config))
            .expect("heap allocation failed (out of memory)");
        let mut ptr = raw.cast::<T>();
        // Safety: raw memory just reserved; no GC can run inside init
        // (the &mut borrow is still outstanding).
        unsafe { ptr.as_mut() }.init(self, &config);
        // Safety: fresh strong pointer, anchored at this borrow.
        unsafe { Tagged::from_value_unchecked(HeapPtr::new(ptr.as_ptr()).encode_strong()) }
    }

    pub fn allocate_handle<'s, T: HeapObject>(
        &mut self,
        config: T::Init<'_>,
        scope: &'s HandleScope<'_>,
    ) -> Handle<'s, T> {
        self.allocate::<T>(config).into_handle(scope)
    }

    // TODO: potentially remove this in favor of a better allocate function
    pub fn allocate_object<'a>(
        &mut self,
        handles: &'a impl HandleSet,
        config: ObjectSlotsInit<'a, '_>,
    ) -> Tagged<'_, Object> {
        let slots: Handle<'a, FixedArray> = if config.values.is_empty() {
            self.known().empty_fixed_array
        } else {
            handles.create_handle(self.allocate::<FixedArray>(config.values))
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
        values: HandleSlice<'a>,
    ) -> Tagged<'_, Object> {
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
    pub fn new_number<'a>(&'a mut self, f: f64) -> Tagged<'a, Value> {
        let r = f as i64; // saturating cast; the round-trip check rejects out-of-range values
        if f.is_finite()
            && f.fract() == 0.0
            && Smi::in_range(r)
            && (r as f64) == f
            && !(f == 0.0 && f.is_sign_negative())
        {
            return Smi::new(r).into_tagged();
        }
        self.allocate::<Float>(f).erase()
    }

    /// CreateArrayFromList (ES 7.3.17): a fresh dense array holding
    /// `values` (a GC-visited slice: stage raw values first).
    pub fn new_array(
        &mut self,
        scope: &HandleScope<'_>,
        values: HandleSlice<'_>,
    ) -> Tagged<'_, Object> {
        let elements = self.allocate_handle::<FixedArray>(values, scope);
        self.allocate_object(
            scope,
            ObjectSlotsInit {
                map: self.known().js_array_map,
                values: HandleSlice::EMPTY,
                elements: elements.erase(),
                length: values.len(),
            },
        )
    }

    pub fn allocate_token(&mut self, total: Layout) -> AllocToken<'_> {
        let layout =
            Layout::from_size_align(total.size(), total.align().max(16)).expect("token layout");
        let raw = self
            .allocate_raw(layout)
            .expect("heap allocation failed (out of memory)");
        // Safety: `raw` is the freshly reserved region of exactly `layout`.
        unsafe { AllocToken::new(self, raw, total) }
    }

    pub fn allocate_token_enter_heap<R>(
        &mut self,
        total: Layout,
        f: impl for<'a> FnOnce(&AllocToken<'_>, &'a Heap) -> R,
    ) -> R {
        let token = self.allocate_token(total);
        f(&token, token.heap())
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
    #[cfg(feature = "stress-minor-gc")]
    stress_armed: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl GlobalHeap {
    pub fn new(shared: Arc<dyn SharedHeap>) -> Self {
        Self {
            shared,
            transition_lock: TransitionLock::new(),
            #[cfg(feature = "stress-minor-gc")]
            stress_armed: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Build the shared heap from any backend configuration.
    pub fn from_backend<B: HeapBackend>(config: B::Config) -> Result<Self, AllocError> {
        Ok(Self::new(B::new(config)?.into_shared()))
    }

    /// Arm the `stress-minor-gc` knob: bootstrap runs un-stressed; from
    /// here on every allocation triggers a minor collection first.
    #[cfg(feature = "stress-minor-gc")]
    pub fn arm_gc_stress(&self) {
        self.stress_armed
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub fn new_local(&self, known: &KnownCell) -> Heap {
        Heap {
            local: self.shared.new_local(),
            transition_lock: self.transition_lock.clone(),
            known: known as *const KnownCell,
            #[cfg(feature = "stress-minor-gc")]
            stress_armed: std::sync::Arc::clone(&self.stress_armed),
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
