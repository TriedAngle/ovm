use std::sync::Arc;

use crate::{
    AllocError, FixedArray, Float, GcHost, Handle, HandleScope, HandleSet, HandleSlice,
    HeapBackend, HeapObject, HeapStats, LocalHeap, Map, MaybeWeak, Object, ObjectInit,
    ObjectSlotsInit, PrototypeRegistry, RawCell, STRONG_PTR, SharedHeap, Smi, Tagged, Tlab, Value,
    Visitor, WeakFixedArray, Word,
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
        unsafe { Tagged::from_raw_ptr(ptr) }
    }

    pub fn remaining(&self) -> usize {
        self.end as usize - self.next.get() as usize
    }

    /// Abandon the rest of the reservation: the drop check then sees zero
    /// remaining. For optimistic paths that reserve up front and bail
    /// before carving everything — e.g. a publication that lost a race.
    pub fn discard_remaining(&self) {
        self.next.set(self.end);
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

    /// The current word without an anchor. It may be moved by a later
    /// collection; only safe to use as an opaque `Value`.
    pub fn raw(&self) -> Value {
        Value::from_bits(self.cell.load())
    }

    /// The Smi payload if the slot holds one; `None` for pointers. The
    /// heap-free read for slots whose value may be a Smi.
    pub fn try_smi(&self) -> Option<i64> {
        self.raw().to_i64()
    }

    /// The anchored referent, valid for the heap borrow.
    pub fn as_ref<'a>(&self, heap: &'a Heap) -> &'a T
    where
        T: HeapObject,
    {
        self.get(heap).as_ref()
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

    /// Initialize a slot without the generational barrier. The caller
    /// guarantees no old→young edge can form: either the host is a fresh
    /// (young) allocation, or `value` is a known old object (e.g. a
    /// bootstrap map). Hot object constructors use this for the header map.
    #[inline]
    pub fn init<'x, U: 'x>(&self, value: impl Into<Tagged<'x, U>>) {
        let v = value.into().raw();
        debug_assert!(!v.is_weak_ptr(), "weak value stored into a strong slot");
        self.cell.store_raw(v.to_bits());
    }

    pub const fn raw_get(this: *const Self) -> *mut T {
        this as *const T as *mut T
    }
}

impl GcSlot<Smi> {
    #[inline]
    pub fn to_smi(&self) -> Smi {
        Smi::decode(self.raw()).expect("GcSlot invariant violated")
    }

    /// The slot's value without the tag check: a `GcSlot<Smi>` is only ever
    /// written an encoded Smi (debug-asserted), so hot readers can skip
    /// `to_smi`'s branch and panic path.
    #[inline(always)]
    pub fn to_smi_unchecked(&self) -> Smi {
        let raw = self.raw();
        debug_assert!(raw.is_smi(), "GcSlot<Smi> invariant violated");
        Smi::new((raw.to_bits() as i64) >> 1)
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
    pub fn new_strong(value: Tagged<'_, T>) -> Self {
        debug_assert!(
            !value.raw().is_weak_ptr(),
            "weak value stored into a strong cell"
        );
        Self {
            // Safety: constructing the storage word of a fresh cell.
            cell: unsafe { RawCell::from_word(value.raw().to_bits()) },
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

    /// The current word without an anchor. It may be moved by a later
    /// collection; only safe to use as an opaque `Value`.
    pub fn raw(&self) -> Value {
        Value::from_bits(self.cell.load())
    }

    pub fn is_cleared(&self) -> bool {
        self.raw().is_cleared()
    }

    /// The anchored referent if the slot holds a live reference (strong or
    /// weak); `None` once it is cleared.
    pub fn as_ref<'a>(&self, heap: &'a Heap) -> Option<&'a T>
    where
        T: HeapObject,
    {
        self.get(heap).as_strong().map(|t| t.as_ref())
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
        if let Some(live) = value.as_strong() {
            heap.write_barrier(host, self.as_raw(), live.erase());
        }
        self.cell.store_raw(value.raw().to_bits());
    }

    /// Resolve a live reference (strong or weak) to its strong view;
    /// `None` once cleared.
    pub fn get_strong<'a>(&self, _heap: &'a Heap) -> Option<Tagged<'a, T>> {
        let word = self.raw();
        if !word.is_ptr() || word.is_cleared() {
            return None;
        }
        let strong = Value::from_bits(word.raw_addr() | STRONG_PTR);
        Some(unsafe { Tagged::from_value_unchecked(strong) })
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

    pub fn raw(&self) -> Value {
        self.slot.raw()
    }

    pub fn get<'a>(&self, heap: &'a Heap) -> Option<Tagged<'a, T>> {
        // Safety: fresh root-slot read for a word comparison.
        if self.raw() == heap.known().the_hole.raw() {
            return None;
        }
        Some(self.slot.get(heap))
    }

    /// The anchored referent if the slot is not the hole.
    pub fn as_ref<'a>(&self, heap: &'a Heap) -> Option<&'a T>
    where
        T: HeapObject,
    {
        self.get(heap).map(|t| t.as_ref())
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
            .store_raw(heap.known().the_hole.raw().to_bits());
    }
}

/// A strongly-held GC slot whose word is read and written atomically:
/// shared mutable map state (slack tracking) that several mutators may
/// touch. Smi payloads need no barrier; pointer stores go through the
/// generational barrier.
#[repr(transparent)]
pub struct AtomicGcSlot<T = Value> {
    slot: GcSlot<T>,
}

unsafe impl<T> Send for AtomicGcSlot<T> {}
unsafe impl<T> Sync for AtomicGcSlot<T> {}

impl<T> AtomicGcSlot<T> {
    pub fn load_word(&self, _heap: &Heap) -> Word {
        self.slot
            .as_raw()
            .load_atomic(core::sync::atomic::Ordering::Acquire)
    }

    /// Atomic store via a CAS loop (`RawCell` has no atomic store).
    pub fn store_word(&self, word: Word) {
        let raw = self.slot.as_raw();
        let mut current = raw.load_atomic(core::sync::atomic::Ordering::Relaxed);
        loop {
            match raw.compare_exchange(
                current,
                word,
                core::sync::atomic::Ordering::Release,
                core::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    pub fn compare_exchange_word(&self, current: Word, new: Word) -> Result<Word, Word> {
        self.slot.as_raw().compare_exchange(
            current,
            new,
            core::sync::atomic::Ordering::AcqRel,
            core::sync::atomic::Ordering::Acquire,
        )
    }

    pub fn load<'a>(&self, heap: &'a Heap) -> Tagged<'a, T> {
        // Safety: the GC keeps slots current; anchored by the borrow.
        unsafe { Tagged::from_value_unchecked(Value::from_bits(self.load_word(heap))) }
    }

    pub fn store<'h, 'x>(
        &self,
        heap: &Heap,
        host: Tagged<'h, Value>,
        value: impl Into<Tagged<'x, T>>,
    ) where
        T: 'x,
    {
        let value = value.into();
        let v = value.raw();
        debug_assert!(!v.is_weak_ptr(), "weak value stored into a strong slot");
        if v.is_ptr() {
            heap.write_barrier(host, self.slot.as_raw(), value.erase());
        }
        self.store_word(v.to_bits());
    }
}

impl AtomicGcSlot<Smi> {
    /// The packed Smi payload (used for bit-field counters).
    pub fn load_smi(&self, heap: &Heap) -> Smi {
        Smi::decode(Value::from_bits(self.load_word(heap)))
            .expect("AtomicGcSlot<Smi> invariant violated")
    }

    pub fn store_smi(&self, value: Smi) {
        self.store_word(value.into_tagged().raw().to_bits());
    }

    pub fn compare_exchange_smi(&self, current: Smi, new: Smi) -> Result<Smi, Smi> {
        let decode = |word: Word| {
            Smi::decode(Value::from_bits(word)).expect("AtomicGcSlot<Smi> invariant violated")
        };
        self.compare_exchange_word(
            current.into_tagged().raw().to_bits(),
            new.into_tagged().raw().to_bits(),
        )
        .map(decode)
        .map_err(decode)
    }
}

#[repr(transparent)]
pub struct AtomicOptionGcSlot<T = Value> {
    slot: OptionGcSlot<T>,
}

unsafe impl<T> Send for AtomicOptionGcSlot<T> {}
unsafe impl<T> Sync for AtomicOptionGcSlot<T> {}

impl<T> AtomicOptionGcSlot<T> {
    pub fn clear(&self, heap: &Heap) {
        self.slot.clear(heap);
    }

    pub fn load_word(&self, _heap: &Heap) -> Word {
        self.slot
            .slot
            .as_raw()
            .load_atomic(core::sync::atomic::Ordering::Acquire)
    }

    pub fn load<'a>(&self, heap: &'a Heap) -> Option<Tagged<'a, T>> {
        self.decode(heap, self.load_word(heap))
    }

    pub fn decode<'a>(&self, heap: &'a Heap, word: Word) -> Option<Tagged<'a, T>> {
        if Value::from_bits(word) == heap.known().the_hole.raw() {
            return None;
        }
        Some(unsafe { Tagged::from_value_unchecked(Value::from_bits(word)) })
    }

    pub fn publish<'h, 'x>(
        &self,
        heap: &Heap,
        host: Tagged<'h, Value>,
        expected: Word,
        value: Tagged<'x, T>,
    ) -> Result<(), Word>
    where
        T: 'x,
    {
        let v = value.raw();
        debug_assert!(!v.is_weak_ptr(), "weak value stored into a strong slot");
        match self.slot.slot.as_raw().compare_exchange(
            expected,
            v.to_bits(),
            core::sync::atomic::Ordering::AcqRel,
            core::sync::atomic::Ordering::Acquire,
        ) {
            Ok(_) => {
                if v.is_ptr() {
                    heap.write_barrier(host, self.slot.as_raw(), value.erase());
                }
                Ok(())
            }
            Err(current) => Err(current),
        }
    }

    pub fn as_raw(&self) -> &RawCell {
        self.slot.as_raw()
    }
}

#[repr(transparent)]
pub struct Register(RawCell);

impl Register {
    pub unsafe fn from_value(v: Tagged<'_, Value>) -> Self {
        Self(unsafe { RawCell::from_word(v.raw().to_bits()) })
    }

    pub fn get<'a>(&self, _heap: &'a Heap) -> Tagged<'a, Value> {
        unsafe { Tagged::from_value_unchecked(Value::from_bits(self.0.load())) }
    }

    #[inline(always)]
    pub fn read_smi_unchecked(&self) -> Smi {
        let raw = self.0.load() as i64;
        debug_assert!(raw & 1 == 0, "register does not hold a Smi");
        Smi::new(raw >> 1)
    }

    #[inline]
    pub fn read_smi(&self) -> Smi {
        Smi::decode(Value::from_bits(self.0.load())).expect("register holds a Smi")
    }

    pub fn raw(&self) -> Value {
        Value::from_bits(self.0.load())
    }

    pub fn as_ref<'a>(&self, _heap: &'a Heap) -> &'a Value {
        unsafe { &*self.0.as_ptr().cast::<Value>() }
    }

    pub fn store<'x, T: 'x>(&self, v: Tagged<'x, T>) {
        debug_assert!(!v.is_weak_ptr(), "weak value stored into a strong slot");
        self.0.store_raw(v.raw().to_bits());
    }

    pub fn as_raw(&self) -> &RawCell {
        &self.0
    }
}

pub trait EdgeVisitable {
    fn visit_edges(&self, visitor: &mut dyn Visitor);
}

impl EdgeVisitable for () {
    fn visit_edges(&self, _visitor: &mut dyn Visitor) {}
}

/// Type-erased per-thread heap.
pub struct Heap {
    local: Box<dyn LocalHeap>,
    tlab: Tlab,
    known: *const KnownCell,
    prototype_registry: *const PrototypeRegistry,
    #[cfg(feature = "stress-minor-gc")]
    stress_armed: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

unsafe impl Send for Heap {}

impl Heap {
    #[inline]
    pub fn allocate_raw(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        debug_assert!(
            layout.align() <= Tlab::ALIGN,
            "unsupported allocation alignment {}",
            layout.align()
        );
        let size = layout.size().next_multiple_of(Tlab::ALIGN);
        if let Some(ptr) = self.tlab.try_alloc(size) {
            return Ok(ptr);
        }
        self.allocate_slow(size)
    }

    #[cold]
    #[inline(never)]
    fn allocate_slow(&mut self, size: usize) -> Result<NonNull<u8>, AllocError> {
        // The collector may park or collect below, reclaiming the current
        // window: drop it before crossing the boundary.
        self.tlab.invalidate();
        let mut tlab = self.local.allocate_slow(size)?;
        let ptr = tlab
            .try_alloc(size)
            .expect("collector must supply a TLAB of at least min_size");
        self.tlab = tlab;
        Ok(ptr)
    }

    pub fn known(&self) -> &'static WellKnown {
        unsafe { (*self.known).get() }
    }

    pub fn prototype_registry(&self) -> &'static PrototypeRegistry {
        unsafe { &*self.prototype_registry }
    }

    pub fn set_known(&self, known: WellKnown) {
        unsafe { (*self.known).set(known) }
    }

    pub fn write_barrier(&self, host: Tagged<'_, Value>, slot: &RawCell, value: Tagged<'_, Value>) {
        self.local
            .write_barrier(host.raw().to_bits(), slot, value.raw().to_bits())
    }

    #[inline]
    pub fn collection_requested(&self) -> bool {
        self.local.collection_requested()
    }

    #[inline]
    pub fn park_for_collection(&self) -> bool {
        self.local.park_for_collection()
    }

    #[inline]
    pub fn take_cancel(&self) -> bool {
        self.local.take_cancel()
    }

    pub fn cancel_executions(&mut self, protocol: &dyn Fn()) {
        self.tlab.invalidate();
        self.local.cancel_executions(protocol)
    }

    pub fn gc_in_progress(&self) -> bool {
        self.local.gc_in_progress()
    }

    #[inline]
    pub fn safepoint_poll(&mut self) -> bool {
        if self.collection_requested() {
            self.tlab.invalidate();
            self.park_for_collection() && self.take_cancel()
        } else {
            false
        }
    }

    pub fn collect(&mut self) {
        self.tlab.invalidate();
        self.local.force_collect();
    }

    pub fn collect_minor(&mut self) {
        self.tlab.invalidate();
        self.local.collect_minor();
    }

    /// Allocate a fresh `T`. The returned `Tagged` is anchored at this
    /// borrow: any further allocation (or anything else requiring
    /// `&mut Heap`) requires rooting it first.
    #[inline]
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
        unsafe { Tagged::from_raw_ptr(ptr) }
    }

    #[inline]
    pub fn allocate_handle<'s, T: HeapObject>(
        &mut self,
        config: T::Init<'_>,
        handles: &'s impl HandleSet,
    ) -> Handle<'s, T> {
        self.allocate::<T>(config).as_handle(handles)
    }

    /// Allocate a fresh `FixedArray` of `len` slots, each initialized to
    /// the hole. A specialized single-allocation constructor for array
    /// growth: the caller overwrites the prefix with the old elements, so
    /// no staging slice is ever materialized.
    pub fn allocate_hole_array(&mut self, len: usize) -> Tagged<'_, FixedArray> {
        #[cfg(feature = "stress-minor-gc")]
        if self.stress_armed.load(std::sync::atomic::Ordering::Acquire) {
            self.collect_minor();
        }
        let raw = self
            .allocate_raw(FixedArray::<Value>::layout_for(len))
            .expect("heap allocation failed (out of memory)");
        // Resolve the fill after the reservation: a TLAB refill above can
        // run a minor collection (the hole is not immortal), and no GC can
        // run between this read and the slot writes below. Resolving earlier
        // left a stale hole word in the fresh array.
        let fill = self.known().the_hole.as_tagged(self).raw();
        let mut ptr = raw.cast::<FixedArray>();
        // Safety: raw memory just reserved; no GC can run inside init
        // (the &mut borrow is still outstanding).
        let obj = unsafe { ptr.as_mut() };
        let host = obj.tagged(self);
        obj.header.map.init(self.known().array_map.as_tagged(self));
        obj.size.set(self, host, Smi::new(len as i64));
        for i in 0..len {
            // the host is fresh, so no old→young barrier is needed
            obj.element_slot(i).as_raw().store_raw(fill.to_bits());
        }
        // Safety: fresh strong pointer, anchored at this borrow.
        unsafe { Tagged::from_raw_ptr(ptr) }
    }

    /// [`Heap::allocate_hole_array`] for a weak array: every entry is a
    /// weak reference to the hole (the "empty" sentinel). The fill word is
    /// resolved after the reservation so a TLAB-refill collection cannot
    /// leave stale holes behind.
    pub fn allocate_hole_weak_array(&mut self, len: usize) -> Tagged<'_, WeakFixedArray> {
        #[cfg(feature = "stress-minor-gc")]
        if self.stress_armed.load(std::sync::atomic::Ordering::Acquire) {
            self.collect_minor();
        }
        let raw = self
            .allocate_raw(WeakFixedArray::<Value>::layout_for(len))
            .expect("heap allocation failed (out of memory)");
        let fill = self
            .known()
            .the_hole
            .as_tagged(self)
            .as_maybe_weak()
            .raw()
            .to_bits();
        let mut ptr = raw.cast::<WeakFixedArray>();
        // Safety: raw memory just reserved; no GC can run inside init
        // (the &mut borrow is still outstanding).
        let obj = unsafe { ptr.as_mut() };
        let host = obj.tagged(self);
        obj.header.map.init(self.known().array_map.as_tagged(self));
        obj.size.set(self, host, Smi::new(len as i64));
        for i in 0..len {
            obj.element_slot(i).as_raw().store_raw(fill);
        }
        // Safety: fresh strong pointer, anchored at this borrow.
        unsafe { Tagged::from_raw_ptr(ptr) }
    }

    // TODO: potentially remove this in favor of a better allocate function
    #[inline]
    pub fn allocate_object<'a>(
        &mut self,
        handles: &'a impl HandleSet,
        config: ObjectSlotsInit<'a, '_>,
    ) -> Tagged<'_, Object> {
        let slots: Handle<'a, FixedArray> = if config.values.is_empty() {
            self.known().empty_fixed_array
        } else {
            self.allocate_handle::<FixedArray>(config.values, handles)
        };
        self.allocate::<Object>(ObjectInit {
            map: config.map,
            slots,
            elements: config.elements,
            length: config.length,
        })
    }

    #[inline]
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
                elements: self.known().empty_fixed_array,
                length: 0,
            },
        )
    }

    /// Like [`Heap::new_object`] with an empty slot list, but reserving
    /// `capacity` hole-filled slots so constructor property stores append
    /// in place instead of reallocating the backing array per property.
    #[inline]
    pub fn new_object_prealloc<'a>(
        &mut self,
        handles: &'a impl HandleSet,
        map: Handle<'a, Map>,
        capacity: usize,
    ) -> Tagged<'_, Object> {
        let capacity = capacity.max(map.as_tagged(self).as_ref().value_slot_count());
        let slots: Handle<'a, FixedArray> = if capacity == 0 {
            self.known().empty_fixed_array
        } else {
            self.allocate_hole_array(capacity).as_handle(handles)
        };
        self.allocate::<Object>(ObjectInit {
            map,
            slots,
            elements: self.known().empty_fixed_array,
            length: 0,
        })
    }

    /// Bump-only float boxing: reserves and initializes a fresh `Float`
    /// from the current TLAB. Never parks, collects or refills — `None`
    /// when the window is exhausted, leaving the caller free to take a
    /// slow path that snapshots any raw code pointer first (no GC can run
    /// on the `Some` path, so an `ip`/`code_ptr` pair captured before the
    /// call stays valid).
    #[inline]
    pub fn try_new_float<'a>(&'a mut self, f: f64) -> Option<Tagged<'a, Value>> {
        // With GC stress every allocation must collect first: decline the
        // fast path so the caller falls back to the boxing slow handler.
        #[cfg(feature = "stress-minor-gc")]
        if self.stress_armed.load(std::sync::atomic::Ordering::Acquire) {
            return None;
        }
        let size = Float::layout_for(&f).size().next_multiple_of(Tlab::ALIGN);
        let raw = self.tlab.try_alloc(size)?;
        let mut ptr = raw.cast::<Float>();
        // Safety: raw memory just reserved; no GC can run here.
        unsafe { ptr.as_mut() }.init(self, &f);
        // Safety: fresh strong pointer, anchored at this borrow.
        Some(unsafe { Tagged::<Float>::from_raw_ptr(ptr).erase() })
    }

    /// A number value: a Smi when the double is an in-range integer, a
    /// freshly boxed Float otherwise. `-0.0` always boxes (it must not
    /// collapse into `+0`). Takes the collector slow path (park/collect/
    /// refill) when the TLAB is exhausted.
    #[inline]
    pub fn new_number<'a>(&'a mut self, f: f64) -> Tagged<'a, Value> {
        if let Some(s) = Smi::from_f64(f) {
            return s.into_tagged();
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
                elements,
                length: values.len(),
            },
        )
    }

    /// A fresh packed array whose elements backing store is pre-sized to
    /// `capacity` (hole filler) with `length` 0: literal element stores
    /// append within headroom instead of reallocating.
    pub fn new_array_with_capacity(
        &mut self,
        scope: &HandleScope<'_>,
        capacity: usize,
    ) -> Tagged<'_, Object> {
        let elements = self.allocate_hole_array(capacity).as_handle(scope);
        self.allocate_object(
            scope,
            ObjectSlotsInit {
                map: self.known().js_array_map,
                values: HandleSlice::EMPTY,
                elements,
                length: 0,
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
    #[cfg(feature = "stress-minor-gc")]
    stress_armed: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl GlobalHeap {
    pub fn new(shared: Arc<dyn SharedHeap>) -> Self {
        Self {
            shared,
            #[cfg(feature = "stress-minor-gc")]
            stress_armed: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Build the shared heap from any backend configuration.
    pub fn from_backend<B: HeapBackend>(config: B::Config) -> Result<Self, AllocError> {
        Ok(Self::new(B::new(config)?.into_shared()))
    }

    /// Arm the `stress-minor-gc` knob: bootstrap runs un-stressed; from
    /// here on every allocation triggers a minor collection first. No-op
    /// unless compiled with the `stress-minor-gc` feature.
    pub fn arm_gc_stress(&self) {
        #[cfg(feature = "stress-minor-gc")]
        self.stress_armed
            .store(true, std::sync::atomic::Ordering::Release);
        #[cfg(not(feature = "stress-minor-gc"))]
        {}
    }

    pub fn new_local(&self, known: &KnownCell, prototype_registry: &PrototypeRegistry) -> Heap {
        Heap {
            local: self.shared.new_local(),
            tlab: Tlab::empty(),
            known: known as *const KnownCell,
            prototype_registry: prototype_registry as *const PrototypeRegistry,
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
