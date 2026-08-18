pub mod handle;
pub mod heap;
pub mod lookup;
pub mod object;
pub mod value;

pub use handle::{EscapableHandleScope, Handle, HandleData, HandleScope, RootHandles};
pub use heap::{
    AllocError, AllocToken, EdgeVisitable, Fresh, GcSlot, Heap, HeapRef, LocalHeap, NoGc, Register,
    RootVisitor, Visitor, WeakGcCell, WellKnown, WordType,
};
pub use lookup::Lookup;
pub use object::{
    AccessorPair, CallableInfoInit, CallableInfoObject, FixedArray, FixedByteArray, Float, Header,
    HeapObject, InternedString, Map, MapInit, SlotDescriptor, SlotFlags, SlotKind, SlotName,
    SlotsObject, SlotsObjectInit, Symbol, VMString,
};
pub use value::{HeapPtr, PointerStrength, Smi, Strong, Tagged, Value, ValueRef, Weak, Word};

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
    assert!(size_of::<Register>() == size_of::<Word>());
    assert!(size_of::<SlotName>() == size_of::<Word>());
};
