use core::{marker::PhantomData,  ptr::NonNull};

use crate::{HeapObject, HeapPtr, Strong, Tagged, Value, Weak, value::PointerStrength};

pub struct Handle<'scope, T, R: PointerStrength = Strong> {
    slot: NonNull<Value>,
    _phantom: PhantomData<(&'scope (), T, R)>,
}

impl<'s, T, R: PointerStrength> Clone for Handle<'s, T, R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'s, T, R: PointerStrength> Copy for Handle<'s, T, R> {}

impl<'s, T, R: PointerStrength> core::fmt::Debug for Handle<'s, T, R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Handle")
            .field("kind", &core::any::type_name::<R>())
            .field("value", &self.value())
            .finish()
    }
}

impl<'s, T, R: PointerStrength> Handle<'s, T, R> {
    pub fn from_slot(slot: NonNull<Value>) -> Self {
        Handle {
            slot,
            _phantom: PhantomData,
        }
    }

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

/// Auto-tagging: the handle's slot already holds the fully encoded
/// value, including the strong/weak tag.
impl<'s, T, R: PointerStrength> From<Handle<'s, T, R>> for Tagged<T> {
    fn from(h: Handle<'s, T, R>) -> Self {
        unsafe { Self::from_value_unchecked(h.value()) }
    }
}
