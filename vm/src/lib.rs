pub mod error;
pub mod handle;
pub mod heap;
pub mod lookup;
pub mod object;
pub mod transition;
pub mod value;

pub use error::VmError;
pub use handle::{EscapableHandleScope, Handle, HandleData, HandleScope, HandleSet, RootHandles};
pub use heap::{
    AllocError, AllocToken, EdgeVisitable, Fresh, GcSlot, Heap, HeapRef, LocalHeap, NoGc,
    OptionGcSlot, RawCell, Register, RootVisitor, Visitor, WeakGcCell, WellKnown, WordType,
};
pub use lookup::Lookup;
pub use object::{
    AccessorPair, CallableInfoInit, CallableInfoObject, Context, ContextInit, FixedArray,
    FixedByteArray, Float, HandlerEntry, HandlerEntryInit, HandlerTable, HandlerTableInit, Header,
    HeapObject, InternedString, Map, MapInit, MapKind, Object, ObjectInit, ObjectKind,
    ObjectSlotsInit, SlotDescriptor, SlotFlags, SlotKind, SlotName, Symbol, VMString,
    string_content_hash,
};
pub use transition::{StoreOutcome, StoreSemantics, TransitionGuard, TransitionLock};
pub use value::{
    HeapPtr, PointerStrength, STRONG_PTR, Smi, Strong, Tagged, Value, ValueRef, WEAK_PTR, Weak,
    Word,
};

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
    assert!(size_of::<OptionGcSlot<FixedArray>>() == size_of::<Word>());
    assert!(size_of::<GcSlot<Smi>>() == size_of::<Word>());
    assert!(size_of::<GcSlot<VMString>>() == size_of::<Word>());
    assert!(size_of::<Register>() == size_of::<Word>());
    assert!(size_of::<SlotName>() == size_of::<Word>());
};
