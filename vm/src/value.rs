use core::{marker::PhantomData, ptr::NonNull};

use crate::{Header, Heap, HeapObject, Map, Object, VmError};

// imports flattened
use crate::Handle;
use crate::HandleSet;
use crate::SlotName;

// The word/tag representation is shared with the heap ABI crate; the VM
// layers the typed Value/Tagged/HeapPtr wrappers on top of it.
pub use heap_api::{PTR_BIT, STRONG_PTR, TAG_MASK, TAG_SMI, WEAK_BIT, WEAK_PTR, Word};

/// Generic Value
/// Either SMI or Pointer
///
/// The type-erased word: it may be stored in GC-visited slots, compared
/// and inspected, but there is no safe way to promote it back into a
/// [`Tagged`] — a pointer word is only guaranteed valid directly after a
/// load under a `&Heap` borrow, which is exactly the lifetime a
/// `Tagged<'a, _>` carries.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Value(Word);

impl Value {
    pub const fn from_bits(bits: Word) -> Self {
        Self(bits)
    }

    pub const fn to_bits(self) -> Word {
        self.0
    }

    pub const fn raw_addr(self) -> Word {
        self.0 & !TAG_MASK
    }

    pub const fn is_smi(self) -> bool {
        self.0 & PTR_BIT == TAG_SMI
    }

    pub const fn is_ptr(self) -> bool {
        self.0 & PTR_BIT != TAG_SMI
    }

    pub const fn is_strong_ptr(self) -> bool {
        self.0 & TAG_MASK == STRONG_PTR
    }

    pub const fn is_weak_ptr(self) -> bool {
        self.0 & TAG_MASK == WEAK_PTR
    }

    pub const CLEARED: Value = Value(WEAK_PTR);

    pub const fn is_cleared(self) -> bool {
        self.0 == WEAK_PTR
    }

    /// The Smi payload as an `i64`; `None` for pointers (incl. weak).
    pub fn to_i64(self) -> Option<i64> {
        Smi::decode(self).map(|s| s.value())
    }

    /// # Safety
    /// The word must have been loaded under a borrow of the heap that is
    /// still alive for `'a` (rooted memory may be re-read at any time —
    /// the GC updates it in place — but a snapshot taken earlier may be
    /// stale), and no collection may have run since the load.
    pub unsafe fn assume_valid<'a>(self, _heap: &'a Heap) -> Tagged<'a, Value> {
        unsafe { Tagged::from_value_unchecked(self) }
    }
}

impl core::fmt::Debug for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_smi() {
            write!(f, "Value(Smi({}))", self.to_i64().unwrap())
        } else if self.is_weak_ptr() {
            write!(f, "Value(Weak({:#x}))", self.raw_addr())
        } else {
            write!(f, "Value(Strong({:#x}))", self.raw_addr())
        }
    }
}

/// SMI 63bit signed integer
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Smi(i64);

impl Smi {
    pub const MAX: i64 = i64::MAX >> 1;
    pub const MIN: i64 = !Self::MAX;

    pub const fn in_range(val: i64) -> bool {
        Self::MIN <= val && val <= Self::MAX
    }

    pub const fn new(val: i64) -> Self {
        debug_assert!(Self::in_range(val));
        Self(val)
    }

    pub const fn value(self) -> i64 {
        self.0
    }

    pub const fn encode(self) -> Value {
        Value((self.0 as Word) << 1)
    }

    // TODO: have unsafe veresion of this with debug check
    pub const fn decode(v: Value) -> Option<Smi> {
        if v.is_smi() {
            Some(Self((v.to_bits() as i64) >> 1))
        } else {
            None
        }
    }
}

/// Range-check a computed i64 and encode it as a Smi.
#[inline]
pub fn encode_smi(r: i64) -> Result<Value, VmError> {
    if !Smi::in_range(r) {
        return Err(VmError::Overflow);
    }
    Ok(Smi::new(r).encode())
}

/// Pointer to the Heap
/// because the GC may be moving this is NOT safe to dereference acroess GC safepoints.
pub struct HeapPtr<T>(NonNull<T>);

unsafe impl<T: HeapObject> Send for HeapPtr<T> {}
unsafe impl<T: HeapObject> Sync for HeapPtr<T> {}

impl<T> Clone for HeapPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for HeapPtr<T> {}

impl<T> core::fmt::Debug for HeapPtr<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "HeapPtr({:#x})", self.0.as_ptr() as Word)
    }
}

// TODO: consider swapping the interface safety
// creation should be safe and encoding to value unsafe?
impl<T> HeapPtr<T> {
    pub unsafe fn new(ptr: *mut T) -> Self {
        unsafe { Self(NonNull::new_unchecked(ptr)) }
    }

    pub const fn as_ptr(self) -> *mut T {
        self.0.as_ptr()
    }

    pub unsafe fn cast<U>(self) -> HeapPtr<U> {
        unsafe { HeapPtr::new(self.as_ptr() as *mut U) }
    }
}

impl HeapPtr<Value> {
    pub fn decode_strong(v: Value) -> Option<Self> {
        if v.is_strong_ptr() {
            Some(unsafe { Self::new(v.raw_addr() as *mut Value) })
        } else {
            None
        }
    }

    pub fn decode(v: Value) -> Option<Self> {
        if v.is_ptr() {
            Some(unsafe { Self::new(v.raw_addr() as *mut Value) })
        } else {
            None
        }
    }
}

impl<T: HeapObject> HeapPtr<T> {
    pub unsafe fn as_ref<'a>(self) -> &'a T {
        unsafe { &*self.0.as_ptr() }
    }

    pub unsafe fn as_mut<'a>(mut self) -> &'a mut T {
        unsafe { self.0.as_mut() }
    }

    pub fn encode_strong(self) -> Value {
        Value(self.as_ptr() as Word | STRONG_PTR)
    }
}

pub struct MaybeWeak<T>(PhantomData<fn() -> T>);

/// Tagged is a typed Value that is valid for the lifetime `'a` of the
/// heap borrow it was loaded (or allocated) under: a *flow type*. While
/// any `Tagged<'a, _>` exists, the borrow checker keeps the `'a` borrow
/// of the heap alive, so no allocation (and therefore no GC) can happen
/// through it. To carry a value across a GC safepoint it must be rooted
/// first: `Tagged<'a, T>` -> `Handle<'scope, T>` (via a handle scope).
///
/// Safe construction is only possible from Smis, fresh allocation, or
/// anchored reads (handle slots, GC slots, registers) — never from a
/// raw `Value`.
#[repr(transparent)]
pub struct Tagged<'a, T: 'a = Value> {
    raw: Value,
    _phantom: PhantomData<(&'a (), T)>,
}
impl<'a, T> Clone for Tagged<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Copy for Tagged<'a, T> {}

impl<'a, T> core::fmt::Debug for Tagged<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Tagged")
            .field(&core::any::type_name::<T>())
            .field(&self.raw)
            .finish()
    }
}

impl Smi {
    /// Smis are pointer-free and valid at any lifetime.
    pub fn into_tagged(self) -> Tagged<'static, Value> {
        Tagged::from(self)
    }
}

impl<'a> From<Smi> for Tagged<'a, Value> {
    fn from(smi: Smi) -> Self {
        Tagged {
            raw: smi.encode(),
            _phantom: PhantomData,
        }
    }
}

impl<'a> From<Smi> for Tagged<'a, Smi> {
    fn from(smi: Smi) -> Self {
        Self::from_smi(smi)
    }
}

impl<'a, T> Tagged<'a, T> {
    /// # Safety
    /// The value must be a strong (non-weak) word that was loaded (or
    /// allocated) under a borrow of the heap that is still alive for
    /// `'a`, and no GC may have run since.
    pub unsafe fn from_value_unchecked(value: Value) -> Self {
        debug_assert!(
            !value.is_weak_ptr(),
            "weak value in a strong Tagged: use Tagged<MaybeWeak<T>>"
        );
        Self {
            raw: value,
            _phantom: PhantomData,
        }
    }

    /// Try to reinterpret an erased word as a Smi: the only safe
    /// promotion from `Value`, since Smis cannot dangle.
    pub fn try_smi(value: Value) -> Option<Tagged<'static, Value>> {
        if value.is_smi() {
            Some(Tagged {
                raw: value,
                _phantom: PhantomData,
            })
        } else {
            None
        }
    }

    pub fn raw(self) -> Value {
        self.raw
    }

    /// Erase the phantom type only: the anchor is unchanged. This is
    /// purely type-level (any heap object is a `Value`).
    pub fn erase(self) -> Tagged<'a, Value> {
        Tagged {
            raw: self.raw,
            _phantom: PhantomData,
        }
    }

    pub unsafe fn cast<U>(self) -> Tagged<'a, U> {
        unsafe { Tagged::from_value_unchecked(self.raw) }
    }

    pub fn is_smi(self) -> bool {
        self.raw.is_smi()
    }

    pub fn is_ptr(self) -> bool {
        self.raw.is_ptr()
    }

    pub fn is_strong_ptr(self) -> bool {
        self.raw.is_strong_ptr()
    }

    pub fn is_weak_ptr(self) -> bool {
        self.raw.is_weak_ptr()
    }

    pub fn ptr_eq<U>(self, other: Tagged<'a, U>) -> bool {
        self.raw.to_bits() == other.raw.to_bits()
    }
}

/// Identity equality across anchors and phantom types: two `Tagged`s are
/// equal iff they carry the same word (the same convention as `Value`).
impl<'a, 'b, T, U> PartialEq<Tagged<'b, U>> for Tagged<'a, T> {
    fn eq(&self, other: &Tagged<'b, U>) -> bool {
        self.raw == other.raw
    }
}

impl<'a, T> Eq for Tagged<'a, T> {}

/// A raw word and an anchored word compare by identity too, so `*acc` and
/// a well-known singleton can be compared without erasing either side.
impl<'a, T> PartialEq<Value> for Tagged<'a, T> {
    fn eq(&self, other: &Value) -> bool {
        self.raw == *other
    }
}

impl<'a, T> PartialEq<Tagged<'a, T>> for Value {
    fn eq(&self, other: &Tagged<'a, T>) -> bool {
        *self == other.raw
    }
}

impl<'a> Tagged<'a, Value> {
    /// The Smi payload as an `i64`; `None` for pointers (incl. weak).
    pub fn to_i64(self) -> Option<i64> {
        self.raw.to_i64()
    }

    pub fn get_as<T: HeapObject>(self) -> Option<Tagged<'a, T>> {
        let ptr = HeapPtr::decode_strong(self.raw)?;
        // Safety: strong pointer; reads only the header's map slot.
        let map = unsafe { &*(ptr.as_ptr() as *const Header) }.map.inner();
        // Safety: raw header read under the anchor.
        let map_ref = unsafe { HeapPtr::<Map>::new(map.raw_addr() as *mut Map).as_ref() };
        let kind = map_ref.kind().kind();
        if !T::matches_kind(kind) {
            return None;
        }
        // Safety: the map-kind check above is the type witness; the
        // anchor `'a` proves no GC ran since the load.
        Some(unsafe { Tagged::from_value_unchecked(self.raw) })
    }

    /// Narrow an anchored value word to a property-name tag (type-level
    /// only: a name is an interned string, a symbol or a Smi, compared
    /// by word identity).
    pub fn as_name(self) -> Tagged<'a, SlotName> {
        Tagged {
            raw: self.raw,
            _phantom: PhantomData,
        }
    }

    pub fn as_heap_object(self) -> Option<Tagged<'a, Object>> {
        if self.raw.is_strong_ptr() {
            // Safety: strong pointer; the anchor `'a` proves no GC ran
            // since the load. No kind check: `Object` is the widest
            // header-prefixed view, callers narrow further.
            Some(unsafe { self.cast() })
        } else {
            None
        }
    }
}

impl<'a> Tagged<'a, Smi> {
    pub fn from_smi(smi: Smi) -> Self {
        unsafe { Self::from_value_unchecked(smi.encode()) }
    }

    pub fn smi(v: i64) -> Option<Self> {
        if Smi::in_range(v) {
            Some(Self::from_smi(Smi::new(v)))
        } else {
            None
        }
    }

    pub fn to_smi(self) -> Option<Smi> {
        Smi::decode(self.raw)
    }
}

impl<'a, T: HeapObject> Tagged<'a, T> {
    pub fn from_ptr(_heap: &'a Heap, ptr: HeapPtr<T>) -> Self {
        // Safety: `ptr` is a strong heap pointer re-anchored at a live
        // borrow of the heap — no GC can have run since it was obtained.
        unsafe { Self::from_value_unchecked(ptr.encode_strong()) }
    }

    pub fn as_ptr(self) -> Option<HeapPtr<T>> {
        if self.raw.is_strong_ptr() {
            Some(unsafe { HeapPtr::new(self.raw.raw_addr() as *mut T) })
        } else {
            None
        }
    }

    /// The anchored referent. The `'a` heap borrow proves the pointer is
    /// live and cannot move for the whole borrow, so no GC can invalidate
    /// it. (Unlike [`HeapPtr`], which is only usable until the next
    /// safepoint.)
    pub fn as_ref(self) -> &'a T {
        // Safety: a `Tagged<'a, T: HeapObject>` is always a strong heap
        // pointer valid for `'a` (Smi/weak words have no `HeapObject` type).
        unsafe { self.as_ptr().expect("strong pointer").as_ref() }
    }

    /// A rooted copy: the value may now cross GC safepoints.
    pub fn into_handle<'s>(self, scope: &'s impl HandleSet) -> Handle<'s, T>
    where
        T: 's,
    {
        scope.create_handle(self)
    }
}

impl<'a, T: HeapObject> core::ops::Deref for Tagged<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // Safety: `Tagged<'a, T>` holds a strong pointer valid for `'a`,
        // so the (shorter) borrow of `self` is valid too.
        unsafe { self.as_ptr().expect("strong pointer").as_ref() }
    }
}

impl<'a, T: HeapObject> From<Tagged<'a, T>> for HeapPtr<T> {
    fn from(v: Tagged<'a, T>) -> Self {
        unsafe { HeapPtr::new(v.raw().raw_addr() as *mut T) }
    }
}

impl<'a, T> From<Tagged<'a, T>> for Value {
    fn from(tagged: Tagged<'a, T>) -> Self {
        tagged.raw()
    }
}

impl<'a, T: HeapObject> Tagged<'a, T> {
    pub fn make_weak(self) -> Tagged<'a, MaybeWeak<T>> {
        Tagged {
            raw: Value(self.raw.to_bits() | WEAK_PTR),
            _phantom: PhantomData,
        }
    }
}

impl<'a, T> Tagged<'a, MaybeWeak<T>> {
    pub unsafe fn from_maybe_weak_unchecked(value: Value) -> Self {
        Tagged {
            raw: value,
            _phantom: PhantomData,
        }
    }

    pub fn from_strong(strong: Tagged<'a, T>) -> Self {
        Tagged {
            raw: strong.raw,
            _phantom: PhantomData,
        }
    }

    pub fn strengthen(self) -> Option<Tagged<'a, T>> {
        if self.raw.is_weak_ptr() {
            None
        } else {
            // SAFETY: the weak bit is clear.
            Some(unsafe { Tagged::from_value_unchecked(self.raw) })
        }
    }

    pub fn is_cleared(self) -> bool {
        self.raw.is_cleared()
    }
}

impl<'a> Tagged<'a, Value> {
    pub fn as_maybe_weak(self) -> Tagged<'a, MaybeWeak<Value>> {
        unsafe { Tagged::from_maybe_weak_unchecked(self.raw) }
    }

    pub fn as_weak(self) -> Tagged<'a, MaybeWeak<Value>> {
        let weak = Value::from_bits(self.raw.to_bits() | WEAK_PTR);
        unsafe { Tagged::from_maybe_weak_unchecked(weak) }
    }
}
