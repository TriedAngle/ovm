use core::{marker::PhantomData, ptr::NonNull};

use crate::{Heap, HeapObject, MapKind, Object, VmError};

// imports flattened
use crate::Handle;
use crate::HandleSet;
use crate::SlotName;

// The word/tag representation is shared with the heap ABI crate; the VM
// layers the typed Value/Tagged wrappers on top of it.
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

    /// The canonical comparison word: for pointers the weak tag bit is
    /// cleared, so a strong and a weak reference to the same object match;
    /// Smis keep their encoded value (bit 1 is data, not a tag).
    pub const fn raw_address_word(self) -> Word {
        if self.is_ptr() {
            self.0 & !WEAK_BIT
        } else {
            self.0
        }
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
    #[inline(always)]
    pub const fn new(val: i64) -> Self {
        debug_assert!(Self::in_range(val));
        Self(val)
    }
    #[inline(always)]
    pub const fn value(self) -> i64 {
        self.0
    }

    pub const fn encode(self) -> Value {
        Value((self.0 as Word) << 1)
    }

    // TODO: have unsafe veresion of this with debug check
    #[inline(always)]
    pub const fn decode(v: Value) -> Option<Smi> {
        if v.is_smi() {
            Some(Self((v.to_bits() as i64) >> 1))
        } else {
            None
        }
    }

    /// The `Smi` for an exactly-representable integer `f`, if any: `-0.0`,
    /// non-integral, out-of-range, NaN and infinite values are `None`.
    /// The saturating `as i64` plus the round-trip check makes this
    /// branch-only — no libm `trunc`/`fract`.
    #[inline(always)]
    pub fn from_f64(f: f64) -> Option<Smi> {
        let r = f as i64;
        if (r as f64) == f && Self::in_range(r) && !(f == 0.0 && f.is_sign_negative()) {
            Some(Self(r))
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
    ///
    /// Must not be used on a weak word: `Tagged<Value>` means "strong".
    /// Use [`Tagged::erase_weak`] for [`Tagged<MaybeWeak>`].
    pub fn erase(self) -> Tagged<'a, Value> {
        debug_assert!(
            !self.raw.is_weak_ptr(),
            "erase() on a weak Tagged promotes it to strong; use erase_weak()"
        );
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

    /// Pointer equality: compares the canonical address word
    /// ([`Value::raw_address_word`]), so strong and weak references to the
    /// same object are equal and Smis compare by their encoded value.
    pub fn ptr_eq<U>(self, other: Tagged<'a, U>) -> bool {
        self.raw.raw_address_word() == other.raw.raw_address_word()
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

    /// The *encoded* Smi word (`value << 1`, tag bit clear), or `None` for
    /// a heap pointer. Operations that are linear in the value — add, sub,
    /// negate, ordered compare — can run directly on this word: the shift
    /// is a factor common to both sides, so it factors out of the result.
    #[inline(always)]
    pub fn smi_bits(self) -> Option<i64> {
        if self.raw.is_smi() {
            Some(self.raw.to_bits() as i64)
        } else {
            None
        }
    }

    /// A Smi from an already-encoded word. Any even word is a valid Smi
    /// (the payload is the upper 63 bits), so this cannot fail and cannot
    /// dangle; the `'a` anchor is therefore unconstrained.
    #[inline(always)]
    pub fn from_smi_bits(bits: i64) -> Tagged<'a, Value> {
        debug_assert_eq!(bits & 1, 0, "Smi encoding must keep the tag bit clear");
        Tagged {
            raw: Value::from_bits(bits as Word),
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn get_as<T: HeapObject>(self, heap: &Heap) -> Option<Tagged<'a, T>> {
        // strong-pointer witness; every heap object is header-prefixed,
        // so the map slot is readable under the anchor
        let obj = self.as_heap_object()?;
        let kind = obj.as_ref().header.map.get(heap).as_ref().kind();
        // exact-kind fast path: one mask + compare, no decode match
        if kind.bits() & MapKind::KIND_MASK == T::KIND as u64 || T::matches_kind(kind.kind()) {
            // Safety: the map-kind check above is the type witness.
            return Some(unsafe { obj.cast::<T>() });
        }
        None
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
    /// A strong `Tagged` for a raw heap pointer, without going through a
    /// `Value` word.
    ///
    /// # Safety
    /// `ptr` must point at a live `T`, and no GC may run until the end of
    /// `'a` (the borrow the result is anchored at).
    pub unsafe fn from_raw_ptr(ptr: NonNull<T>) -> Self {
        Self {
            raw: Value::from_bits(ptr.as_ptr() as Word | STRONG_PTR),
            _phantom: PhantomData,
        }
    }

    pub fn from_ptr(_heap: &'a Heap, ptr: NonNull<T>) -> Self {
        // Safety: `ptr` is a strong heap pointer re-anchored at a live
        // borrow of the heap — no GC can have run since it was obtained.
        unsafe { Self::from_raw_ptr(ptr) }
    }

    /// The anchored referent. The `'a` heap borrow proves the pointer is
    /// live and cannot move for the whole borrow, so no GC can invalidate
    /// it: unlike a raw word, the value may be dereferenced freely until
    /// the borrow ends.
    pub fn as_ref(self) -> &'a T {
        // Safety: a `Tagged<'a, T: HeapObject>` is always a strong heap
        // pointer valid for `'a` (Smi/weak words have no `HeapObject` type).
        unsafe { self.as_ref_unchecked() }
    }

    /// [`as_ref`] without the strong-pointer witness. Hot handlers that
    /// already proved the anchor is a strong pointer (an unchecked cast from
    /// a verified slot) use this to skip the redundant tag check.
    ///
    /// # Safety
    /// `self` must hold a strong heap pointer (`raw.is_strong_ptr()`).
    #[inline(always)]
    pub unsafe fn as_ref_unchecked(self) -> &'a T {
        unsafe { &*(self.raw.raw_addr() as *const T) }
    }

    /// A rooted copy: the value may now cross GC safepoints.
    pub fn as_handle<'s>(self, scope: &'s impl HandleSet) -> Handle<'s, T>
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
        unsafe { self.as_ref_unchecked() }
    }
}

impl<'a, T> Tagged<'a, T> {
    /// The maybe-weak view of this word, without changing it.
    pub fn as_maybe_weak(self) -> Tagged<'a, MaybeWeak<T>> {
        Tagged {
            raw: self.raw,
            _phantom: PhantomData,
        }
    }

    /// The weak view of this word: sets the weak tag bit. Callers are
    /// responsible for the word being a heap pointer.
    pub fn as_weak(self) -> Tagged<'a, MaybeWeak<T>> {
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

    /// Erase the phantom type to the erased maybe-weak form, keeping the
    /// word (and its weak tag) exactly as stored. The counterpart of
    /// [`Tagged::erase`] for weak words.
    pub fn erase_weak(self) -> Tagged<'a, MaybeWeak<Value>> {
        Tagged {
            raw: self.raw,
            _phantom: PhantomData,
        }
    }

    /// The strong view of a live reference; `None` for cleared or
    /// non-pointer words.
    pub fn as_strong(self) -> Option<Tagged<'a, T>> {
        if !self.raw.is_ptr() || self.raw.is_cleared() {
            return None;
        }
        // SAFETY: live pointer; the weak bit is cleared.
        Some(unsafe {
            Tagged::from_value_unchecked(Value::from_bits(self.raw.raw_addr() | STRONG_PTR))
        })
    }

    pub fn is_cleared(self) -> bool {
        self.raw.is_cleared()
    }
}
