pub mod array;
pub mod byte_array;
pub mod callable;
pub mod context;
pub mod float;
pub mod map;
pub mod object;
pub mod proxy;
pub mod string;
pub mod symbol;

pub use array::{FixedArray, WeakFixedArray, WeakFixedArrayInit};
pub use byte_array::FixedByteArray;
pub use callable::{CallableInfoInit, CallableInfoObject, FunctionKind};
pub use context::{
    Context, ContextInit, HandlerEntry, HandlerEntryInit, HandlerTable, HandlerTableInit,
    ScopeInfo, ScopeInfoInit,
};
pub use float::Float;
pub use map::{AccessorPair, Map, MapInit, MapKind, SlotDescriptor, SlotFlags, SlotName};
pub use object::{CallTarget, Object, ObjectInit, ObjectSlotsInit};
pub use proxy::{ProxyInit, ProxyObject};
pub use string::{DenseString, Encoding, StringData, decode_wtf8, string_content_hash};
pub use symbol::Symbol;

use core::{alloc::Layout, ptr::NonNull};

use crate::{
    EdgeVisitable, GcSlot, Heap, HeapPtr, STRONG_PTR, Tagged, Value, Visitor, WEAK_PTR, Word,
};

pub trait HeapObject: 'static {
    type Init<'a>;

    const KIND: ObjectKind;

    fn matches_kind(kind: ObjectKind) -> bool {
        kind == Self::KIND
    }

    fn layout_for(config: &Self::Init<'_>) -> Layout;

    /// Initialize the fresh object. `heap` is borrowed for the duration
    /// of the call only: no allocation is possible inside.
    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>);

    fn header(&self) -> &Header;

    fn layout(&self) -> Layout;

    fn erase(&self) -> Value
    where
        Self: Sized,
    {
        let addr = self as *const Self as *const Word as Word;
        Value::from_bits(addr | STRONG_PTR)
    }

    fn erase_weak(&self) -> Value
    where
        Self: Sized,
    {
        let addr = self as *const Self as *const Word as Word;
        Value::from_bits(addr | WEAK_PTR)
    }
}

#[repr(C)]
pub struct Header {
    pub map: GcSlot<Map>,
}

impl Header {
    pub fn map<'a>(&self, heap: &'a Heap) -> Tagged<'a, Map> {
        self.map.get(heap)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u64)]
pub enum ObjectKind {
    BuiltinStart = 0,
    Map = 1,
    FixedArray = 2,
    FixedByteArray = 3,
    DenseString = 4,
    AccessorPair = 5,
    CallableInfo = 6,
    Float = 7,
    Symbol = 8,
    HandlerTable = 9,
    Context = 10,
    ScopeInfo = 11,
    BuiltinEnd = 12,

    /// `elements` points to the well-known `empty_fixed_array`, `len` is 0
    Object = 13,
    /// `elements` points to a `FixedArray`.
    Array = 14,
    /// `elements` points to a `FixedByteArray`.
    ByteArray = 15,
    /// `elements` points to a `DenseString`.
    String = 16,
    /// A Proxy exotic object (`ProxyObject`): no own properties, all
    /// internal methods dispatch through handler traps.
    Proxy = 17,
    /// Sentinel and odd heap values
    Oddball = 18,
}

impl ObjectKind {
    pub const BUILTIN_START: u64 = ObjectKind::BuiltinStart as u64;
    pub const BUILTIN_END: u64 = ObjectKind::BuiltinEnd as u64;

    /// Whether values of this kind are ECMAScript receivers (JSReceiver)
    pub const fn is_js_receiver(self) -> bool {
        matches!(
            self,
            ObjectKind::Object
                | ObjectKind::Array
                | ObjectKind::ByteArray
                | ObjectKind::String
                | ObjectKind::Proxy
        )
    }
}

pub unsafe fn object_kind(addr: NonNull<()>) -> ObjectKind {
    let header = unsafe { &*addr.cast::<Header>().as_ptr() };
    // Safety: GC-callback context; raw header read.
    let map = header.map.inner();
    let map_ref = unsafe { HeapPtr::<Map>::new(map.raw_addr() as *mut Map).as_ref() };
    map_ref.kind().kind()
}

pub unsafe fn object_layout(addr: NonNull<()>) -> Layout {
    let kind = unsafe { object_kind(addr) };
    unsafe {
        match kind {
            ObjectKind::Map => (*addr.cast::<Map>().as_ptr()).layout(),
            ObjectKind::FixedArray => (*addr.cast::<FixedArray>().as_ptr()).layout(),
            ObjectKind::FixedByteArray => (*addr.cast::<FixedByteArray>().as_ptr()).layout(),
            ObjectKind::DenseString => (*addr.cast::<DenseString>().as_ptr()).layout(),
            ObjectKind::AccessorPair => (*addr.cast::<AccessorPair>().as_ptr()).layout(),
            ObjectKind::CallableInfo => (*addr.cast::<CallableInfoObject>().as_ptr()).layout(),
            ObjectKind::Float => (*addr.cast::<Float>().as_ptr()).layout(),
            ObjectKind::Symbol => (*addr.cast::<Symbol>().as_ptr()).layout(),
            ObjectKind::HandlerTable => (*addr.cast::<HandlerTable>().as_ptr()).layout(),
            ObjectKind::Context => (*addr.cast::<Context>().as_ptr()).layout(),
            ObjectKind::ScopeInfo => (*addr.cast::<ScopeInfo>().as_ptr()).layout(),
            ObjectKind::Object
            | ObjectKind::Array
            | ObjectKind::ByteArray
            | ObjectKind::String
            | ObjectKind::Oddball => (*addr.cast::<Object>().as_ptr()).layout(),
            ObjectKind::Proxy => (*addr.cast::<ProxyObject>().as_ptr()).layout(),
            ObjectKind::BuiltinStart | ObjectKind::BuiltinEnd => {
                unreachable!("sentinel kind in object header")
            }
        }
    }
}

pub unsafe fn visit_object(addr: NonNull<()>, visitor: &mut dyn Visitor) {
    let kind = unsafe { object_kind(addr) };
    unsafe {
        match kind {
            ObjectKind::Map => (*addr.cast::<Map>().as_ptr()).visit_edges(visitor),
            ObjectKind::FixedArray => (*addr.cast::<FixedArray>().as_ptr()).visit_edges(visitor),
            ObjectKind::FixedByteArray => {
                (*addr.cast::<FixedByteArray>().as_ptr()).visit_edges(visitor)
            }
            ObjectKind::DenseString => (*addr.cast::<DenseString>().as_ptr()).visit_edges(visitor),
            ObjectKind::AccessorPair => {
                (*addr.cast::<AccessorPair>().as_ptr()).visit_edges(visitor)
            }
            ObjectKind::CallableInfo => {
                (*addr.cast::<CallableInfoObject>().as_ptr()).visit_edges(visitor)
            }
            ObjectKind::Float => (*addr.cast::<Float>().as_ptr()).visit_edges(visitor),
            ObjectKind::Symbol => (*addr.cast::<Symbol>().as_ptr()).visit_edges(visitor),
            ObjectKind::HandlerTable => {
                (*addr.cast::<HandlerTable>().as_ptr()).visit_edges(visitor)
            }
            ObjectKind::Context => (*addr.cast::<Context>().as_ptr()).visit_edges(visitor),
            ObjectKind::ScopeInfo => (*addr.cast::<ScopeInfo>().as_ptr()).visit_edges(visitor),
            ObjectKind::Object
            | ObjectKind::Array
            | ObjectKind::ByteArray
            | ObjectKind::String
            | ObjectKind::Oddball => (*addr.cast::<Object>().as_ptr()).visit_edges(visitor),
            ObjectKind::Proxy => (*addr.cast::<ProxyObject>().as_ptr()).visit_edges(visitor),
            ObjectKind::BuiltinStart | ObjectKind::BuiltinEnd => {
                unreachable!("sentinel kind in object header")
            }
        }
    }
}
