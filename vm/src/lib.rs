pub mod compare;
pub mod convert;
pub mod error;
pub mod handle;
pub mod heap;
pub mod interner;
pub mod lookup;
pub mod object;
pub mod transition;
pub mod value;

pub use compare::Compare;
pub use convert::Convert;
pub use error::VmError;
pub use handle::{
    EscapableHandleScope, GcSlice, Handle, HandleData, HandleScope, HandleSet, RootHandles,
};
pub use heap::{
    AllocError, AllocToken, EdgeVisitable, Fresh, GcSlot, GlobalHeap, GlobalVtable, Heap,
    HeapBackend, HeapRef, HeapStats, HeapVtable, NoGc, OptionGcSlot, RawCell, Register,
    RootVisitor, Visitor, WellKnown, WordType, bootstrap_basics, bootstrap_well_known,
    intern_well_known_strings,
};
pub use interner::StringInterner;
pub use lookup::{Key, LoadOutcome, Lookup, classify_key, element_value, load_outcome};
pub use object::{
    AccessorPair, CallTarget, CallableInfoInit, CallableInfoObject, Context, ContextInit,
    FixedArray, FixedByteArray, Float, HandlerEntry, HandlerEntryInit, HandlerTable,
    HandlerTableInit, Header, HeapObject, InternedString, Map, MapInit, MapKind, Object,
    ObjectInit, ObjectKind, ObjectSlotsInit, ScopeInfo, ScopeInfoInit, SlotDescriptor, SlotFlags,
    SlotName, Symbol, VMString, call_target, store_array_element, string_content_hash,
};
pub use transition::{
    Change, PropertyDescriptor, StoreOutcome, StoreSemantics, Transition, TransitionGuard,
    TransitionLock,
};
pub use value::{
    HeapPtr, PointerStrength, STRONG_PTR, Smi, Strong, Tagged, Value, ValueRef, WEAK_PTR, Weak,
    Word, encode_smi,
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
