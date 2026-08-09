pub mod heap;
pub mod object;
pub mod value;
pub mod handle;

pub use heap::{
    AllocError, EdgeVisitable, Fresh, GcSlot, HeapRef, LocalHeap, NoGc, RootVisitor, SharedHeap,
    WordType,
};
pub use object::{Array, Header, HeapObject, Map, ObjectKind, SlotName, VMString, InternedString};
pub use handle::{Handle};
pub use value::{HeapPtr, Smi, Strong, Tagged, Value, Weak, Word};

pub type Local<'scope, T> = Handle<'scope, T, Strong>;
// pseudo-static
pub type Global<T> = Handle<'static, T, Strong>;

const _: () = {
    use core::mem::size_of;
    assert!(size_of::<Value>() == size_of::<Word>());
    assert!(size_of::<Tagged<Value>>() == size_of::<Word>());
    assert!(size_of::<Tagged<VMString>>() == size_of::<Word>());
    assert!(size_of::<GcSlot>() == size_of::<Word>());
    assert!(size_of::<GcSlot<Smi>>() == size_of::<Word>());
    assert!(size_of::<GcSlot<VMString>>() == size_of::<Word>());
    assert!(size_of::<SlotName>() == size_of::<Word>());
};
