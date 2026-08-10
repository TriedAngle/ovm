pub mod heap;
pub mod object;
pub mod value;
pub mod handle;

pub use heap::{
    AllocError, EdgeVisitable, Fresh, GcSlot, HeapRef, LocalHeap, NoGc, RootVisitor, Heap,
    WordType,
};
pub use object::{Array, Header, HeapObject, Map, ObjectKind, SlotName, VMString, InternedString};
pub use handle::{EscapableHandleScope, Handle, HandleData, HandleScope};
pub use value::{HeapPtr, PointerStrength, Smi, Strong, Tagged, Value, Weak, Word};

pub type Local<'scope, T> = Handle<'scope, T, Strong>;
// pseudo-static
pub type Global<T> = Handle<'static, T, Strong>;

/// Number of slots per handle block.
pub const HANDLE_BLOCK_SIZE: usize = 1024;

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
