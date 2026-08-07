use core::{marker::PhantomData, ptr::NonNull};

use crate::HeapObject;

/// Word Size inside the heap
/// if we add compressed pointers we may need to duplicate this
/// maybe putting it to u32 is enough
pub type Word = u64;

pub const TAG_MASK: Word = 0b11;
pub const PTR_BIT: Word = 0b01;
pub const WEAK_BIT: Word = 0b10;

pub const TAG_SMI: Word = 0b0;
pub const STRONG_PTR: Word = 0b01;
pub const WEAK_PTR: Word = 0b11;

/// Generic Value
/// Either SMI or Pointer
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

    pub fn try_as<T: FromValue>(self) -> Option<T> {
        T::from_value(self)
    }
}

impl core::fmt::Debug for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_smi() {
            write!(f, "Value(Smi({}))", Smi::decode(*self).unwrap().value())
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

    pub const fn new(val: i64) -> Option<Self> {
        if Self::MIN <= val && val <= Self::MAX {
            Some(Self(val))
        } else {
            None
        }
    }

    pub const fn new_unchecked(val: i64) -> Self {
        debug_assert!(Self::MIN <= val && val <= Self::MAX);
        Self(val)
    }

    pub const fn value(self) -> i64 {
        self.0
    }

    pub const fn encode(self) -> Value {
        Value::from_bits((self.0 as Word) << 1)
    }

    pub const fn decode(v: Value) -> Option<Smi> {
        if v.is_smi() {
            Some(Self((v.to_bits() as i64) >> 1))
        } else {
            None
        }
    }
}

/// Pointer to the Heap
/// because the GC may be moving this is NOT safe to dereference acroess GC safepoints.
pub struct HeapPtr<T>(NonNull<T>);

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

impl<T: HeapObject> HeapPtr<T> {
    pub unsafe fn new_unchecked(ptr: *mut T) -> Self {
        debug_assert!(
            ptr as Word & TAG_MASK == 0,
            "heap objects must be >=4-aligned"
        );
        unsafe { Self(NonNull::new_unchecked(ptr)) }
    }

    pub const fn as_ptr(self) -> *mut T {
        self.0.as_ptr()
    }

    pub unsafe fn as_ref<'a>(self) -> &'a T {
        unsafe { &*self.0.as_ptr() }
    }

    pub unsafe fn as_mut<'a>(mut self) -> &'a mut T {
        unsafe { self.0.as_mut() }
    }

    pub fn encode_strong(self) -> Value {
        Value::from_bits(self.as_ptr() as Word | STRONG_PTR)
    }

    pub fn encode_weak(self) -> Value {
        Value::from_bits(self.as_ptr() as Word | WEAK_PTR)
    }

    pub fn decode_strong(v: Value) -> Option<Self> {
        if v.is_strong_ptr() {
            Some(unsafe { Self::new_unchecked(v.raw_addr() as *mut T) })
        } else {
            None
        }
    }

    pub fn decode(v: Value) -> Option<Self> {
        if v.is_ptr() {
            Some(unsafe { Self::new_unchecked(v.raw_addr() as *mut T) })
        } else {
            None
        }
    }
}

pub struct Strong;
pub struct Weak;

/// Indirect Pointer to the Heap
/// Will be updated when the GC is moving
#[repr(transparent)]
pub struct Handle<'scope, T, R = Strong> {
    slot: NonNull<Value>,
    _phantom: PhantomData<(&'scope (), T, R)>,
}

impl<'s, T, R> Clone for Handle<'s, T, R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'s, T, R> Copy for Handle<'s, T, R> {}

impl<'s, T, R> core::fmt::Debug for Handle<'s, T, R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Handle")
            .field("kind", &core::any::type_name::<R>())
            .field("value", &self.value())
            .finish()
    }
}

impl<'s, T, R> Handle<'s, T, R> {
    pub fn value(self) -> Value {
        unsafe { *self.slot.as_ptr() }
    }

    pub fn erase(self) -> Handle<'s, Value, R> {
        Handle {
            slot: self.slot,
            _phantom: PhantomData,
        }
    }
}

impl<'s, T: HeapObject> Handle<'s, T, Strong> {
    pub fn get(self) -> HeapPtr<T> {
        HeapPtr::decode_strong(self.value()).expect("strong local slot must contain strong pointer")
    }
}

impl<'s, T: HeapObject> Handle<'s, T, Weak> {}

pub trait FromValue: Sized {
    fn from_value(v: Value) -> Option<Self>;
}

pub trait IntoValue {
    fn into_value(self) -> Value;
}

impl FromValue for Value {
    fn from_value(v: Value) -> Option<Self> {
        Some(v)
    }
}

impl IntoValue for Value {
    fn into_value(self) -> Value {
        self
    }
}

impl FromValue for Smi {
    fn from_value(v: Value) -> Option<Self> {
        Smi::decode(v)
    }
}

impl IntoValue for Smi {
    fn into_value(self) -> Value {
        self.encode()
    }
}

impl<T: HeapObject> FromValue for HeapPtr<T> {
    fn from_value(v: Value) -> Option<Self> {
        HeapPtr::decode_strong(v)
    }
}

impl<T: HeapObject> IntoValue for Tagged<T> {
    fn into_value(self) -> Value {
        self.erase()
    }
}

impl<T: HeapObject> FromValue for Tagged<T> {
    fn from_value(v: Value) -> Option<Self> {
        if v.is_ptr() {
            Some(unsafe { Tagged::from_value_unchecked(v) })
        } else {
            None
        }
    }
}

/// Tagged is a typed Value
#[repr(transparent)]
pub struct Tagged<T> {
    raw: Value,
    _phantom: PhantomData<T>,
}
impl<T> Clone for Tagged<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Tagged<T> {}

impl<T> core::fmt::Debug for Tagged<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Tagged")
            .field(&core::any::type_name::<T>())
            .field(&self.raw)
            .finish()
    }
}

impl<T> Tagged<T> {
    pub unsafe fn from_value_unchecked(value: Value) -> Self {
        Self {
            raw: value,
            _phantom: PhantomData,
        }
    }

    pub fn erase(self) -> Value {
        self.raw
    }

    pub fn erase_tagged(self) -> Tagged<Value> {
        unsafe { Tagged::from_value_unchecked(self.raw) }
    }

    pub unsafe fn cast_unchecked<U>(self) -> Tagged<U> {
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

    pub fn ptr_eq<U>(self, other: Tagged<U>) -> bool {
        self.raw.to_bits() == other.raw.to_bits()
    }
}

impl Tagged<Smi> {
    pub fn from_smi(smi: Smi) -> Self {
        unsafe { Self::from_value_unchecked(smi.encode()) }
    }

    pub fn smi(v: i64) -> Option<Self> {
        Smi::new(v).map(Self::from_smi)
    }

    pub fn to_smi(self) -> Option<Smi> {
        Smi::decode(self.raw)
    }
}

impl<T: HeapObject> Tagged<T> {
    pub fn from_ptr(ptr: HeapPtr<T>) -> Self {
        unsafe { Self::from_value_unchecked(ptr.encode_strong()) }
    }

    pub fn as_ptr(self) -> Option<HeapPtr<T>> {
        HeapPtr::decode_strong(self.raw)
    }
}
