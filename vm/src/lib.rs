pub mod heap;
pub mod object;
pub mod value;

pub use heap::{
    AllocError, EdgeVisitable, GcSlot, LocalHeap, NoGc, RootVisitor, SharedHeap, WordType,
};
pub use object::{Array, Header, HeapObject, Map, ObjectKind};
pub use value::{FromValue, Handle, HeapPtr, IntoValue, Smi, Strong, Tagged, Value, Weak, Word};

pub type Local<'scope, T> = Handle<'scope, T, Strong>;
// pseudo-static
pub type Global<T> = Handle<'static, T, Strong>;

impl<'scope, T: HeapObject> Handle<'scope, T, Strong> {
    pub fn read<'a>(self, _guard: &'a NoGc<'a>) -> &'a T {
        unsafe { self.get().as_ref() }
    }
}

impl<'a> NoGc<'a> {
    pub fn read<T: HeapObject>(&'a self, h: Handle<'_, T, Strong>) -> &'a T {
        h.read(self)
    }
}
