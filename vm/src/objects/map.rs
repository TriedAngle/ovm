use core::alloc::Layout;

use crate::{
    AtomicOptionGcSlot, Compare, DenseString, EdgeVisitable, GcSlot, Handle, Header, Heap,
    HeapObject, ObjectKind, OptionGcSlot, Smi, Symbol, Tagged, Value, Visitor, WeakFixedArray,
};

#[repr(C)]
pub struct Map {
    pub header: Header,
    pub value_slot_count: GcSlot<Smi>,
    pub descriptor_count: GcSlot<Smi>,
    /// Object kind tag (low byte) and capability flags (second byte).
    pub kind: GcSlot<Smi>,
    /// The prototype(s) for property lookup:
    /// - an object: single parent (JS `[[Prototype]]`)
    /// - a `FixedArray` of objects: multiple parents in priority order (Self-style `parent*`)
    /// - the hole: no parents (null-proto root)
    pub prototype: GcSlot,
    pub pred: OptionGcSlot<Map>,
    /// Shared transition-tree edges: `[name, target]` pairs. Published
    /// RCU-style (see [`AtomicOptionGcSlot`]): arrays are immutable after
    /// publication, inserts race-publish a grown copy with a CAS.
    pub transitions: AtomicOptionGcSlot<WeakFixedArray>,
    pub descriptors: [SlotDescriptor; 0],
}

impl Map {
    pub fn layout_for(descriptor_count: usize) -> Layout {
        let descriptors_layout =
            Layout::array::<SlotDescriptor>(descriptor_count).expect("descriptors layout");
        Layout::new::<Self>()
            .extend(descriptors_layout)
            .expect("map layout")
            .0
    }

    pub fn value_slot_count(&self) -> usize {
        self.value_slot_count.to_smi().value() as usize
    }

    pub fn descriptor_count(&self) -> usize {
        self.descriptor_count.to_smi().value() as usize
    }

    pub fn pred<'a>(&self, heap: &'a Heap) -> Option<Tagged<'a, Map>> {
        self.pred.get(heap)
    }

    pub fn root_map<'a>(&'a self, heap: &'a Heap) -> Tagged<'a, Map> {
        let mut current: Tagged<'a, Map> = unsafe { Tagged::from_value_unchecked(self.erase()) };
        while let Some(pred) = current.pred(heap) {
            current = pred;
        }
        current
    }

    pub fn kind(&self) -> MapKind {
        MapKind::new(self.kind.to_smi().value() as u64)
    }

    fn data_ptr(&self) -> *mut SlotDescriptor {
        self.descriptors.as_ptr() as *mut SlotDescriptor
    }

    pub fn descriptors(&self) -> &[SlotDescriptor] {
        unsafe { core::slice::from_raw_parts(self.data_ptr(), self.descriptor_count()) }
    }

    pub fn descriptor(&self, i: usize) -> &SlotDescriptor {
        debug_assert!(i < self.descriptor_count());
        unsafe { &*self.data_ptr().add(i) }
    }

    pub fn find_transition<'a>(
        &self,
        heap: &'a Heap,
        name: Tagged<'a, SlotName>,
        flags: SlotFlags,
        pair: Option<(Tagged<'a, Value>, Tagged<'a, Value>)>,
    ) -> Option<Tagged<'a, Map>> {
        let array = self.transitions.load(heap)?;
        let pairs = array.as_slice();
        debug_assert!(
            pairs.len() % 2 == 0,
            "transition pairs are flat [name, map]"
        );
        for entry in pairs.as_chunks::<2>().0 {
            if !entry[0].get(heap).ptr_eq(name.erase()) {
                continue;
            }
            let Some(target) = entry[1].get_strong(heap) else {
                continue;
            };
            let Some(target) = target.get_as::<Map>() else {
                continue;
            };

            let Some(row) = target
                .descriptors()
                .iter()
                .find(|d| d.name(heap).ptr_eq(name) && d.flags() == flags)
            else {
                continue;
            };

            if let Some((get, set)) = pair {
                let matches = row
                    .value
                    .get(heap)
                    .get_as::<AccessorPair>()
                    .is_some_and(|p| {
                        Compare::same_value(heap, get, p.get.get(heap))
                            && Compare::same_value(heap, set, p.set.get(heap))
                    });
                if !matches {
                    continue;
                }
            }
            return Some(target);
        }
        None
    }

    pub fn find_prototype_transition<'a>(
        &self,
        heap: &'a Heap,
        proto: Tagged<'a, Value>,
    ) -> Option<Tagged<'a, Map>> {
        let sentinel = heap.known().prototype_transition_symbol.as_tagged(heap);
        let array = self.transitions.load(heap)?;
        let pairs = array.as_slice();
        for entry in pairs.as_chunks::<2>().0 {
            if !entry[0].get(heap).ptr_eq(sentinel.erase()) {
                continue;
            }
            let Some(target) = entry[1].get_strong(heap) else {
                continue;
            };
            let Some(target) = target.get_as::<Map>() else {
                continue;
            };
            if target.prototype.get(heap).ptr_eq(proto) {
                return Some(target);
            }
        }
        None
    }

    pub fn find_remove_transition<'a>(
        &self,
        heap: &'a Heap,
        name: Tagged<'a, SlotName>,
    ) -> Option<Tagged<'a, Map>> {
        let array = self.transitions.load(heap)?;
        let pairs = array.as_slice();
        debug_assert!(
            pairs.len() % 2 == 0,
            "transition pairs are flat [name, map]"
        );
        for entry in pairs.as_chunks::<2>().0 {
            if !entry[0].get(heap).ptr_eq(name.erase()) {
                continue;
            }
            let Some(target) = entry[1].get_strong(heap) else {
                continue;
            };
            let Some(target) = target.get_as::<Map>() else {
                continue;
            };
            if target.descriptor_count() + 1 == self.descriptor_count()
                && !target
                    .descriptors()
                    .iter()
                    .any(|d| d.name(heap).ptr_eq(name))
            {
                return Some(target);
            }
        }
        None
    }
}

pub struct MapInit<'a> {
    pub kind: MapKind,
    pub value_slot_count: usize,
    pub descriptors: &'a [(Handle<'a, SlotName>, SlotFlags, Handle<'a, Value>)],
    /// the hole = no prototype (null-proto for JS maps).
    /// Handled because `Map` allocation may move the prototype.
    pub prototype: Handle<'a, Value>,
}

impl HeapObject for Map {
    const KIND: ObjectKind = ObjectKind::Map;
    type Init<'a> = MapInit<'a>;

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.descriptors.len())
    }

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        let host = self.tagged(heap);
        self.header
            .map
            .set(heap, host, heap.known().map_map.as_tagged(heap));
        self.value_slot_count
            .set(heap, host, Smi::new(config.value_slot_count as i64));
        self.descriptor_count
            .set(heap, host, Smi::new(config.descriptors.len() as i64));
        self.kind
            .set(heap, host, Smi::new(config.kind.bits() as i64));
        self.prototype
            .set(heap, host, config.prototype.as_tagged(heap));
        self.pred.clear(heap);
        self.transitions.clear(heap);
        for (i, (name, flags, value)) in config.descriptors.iter().enumerate() {
            let d = self.descriptor(i);
            d.name.set(heap, host, name.as_tagged(heap));
            d.flags.set(heap, host, Smi::new(flags.bits() as i64));
            d.value.set(heap, host, value.as_tagged(heap));
        }
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.descriptor_count())
    }
}

impl EdgeVisitable for Map {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.prototype.as_raw());
        visitor.visit(self.pred.as_raw());
        visitor.visit(self.transitions.as_raw());
        for d in self.descriptors() {
            visitor.visit(d.name.as_raw());
            visitor.visit(d.value.as_raw());
        }
    }
}

/// Low byte: the `ObjectKind`. Higher bytes: capability flags
/// (extendable, callable, constructor, runtime) and representation flags
/// (Latin1 string payloads). Constructor implies callable.
/// RUNTIME is only valid together with CALLABLE and means slots[0] of the
/// object is a Smi runtime registry index instead of a `CallableInfoObject`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MapKind(u64);

impl MapKind {
    const KIND_MASK: u64 = 0xff;

    pub const EXTENDABLE: MapKind = MapKind(1 << 8);
    pub const CALLABLE: MapKind = MapKind(1 << 9);
    pub const CONSTRUCTOR: MapKind = MapKind(1 << 10);
    pub const RUNTIME: MapKind = MapKind(1 << 11);
    pub const PRIMITIVE_WRAPPER: MapKind = MapKind(1 << 12);
    pub const CLASS_CONSTRUCTOR: MapKind = MapKind(1 << 13);
    /// Dense-string payload encoding: set = one Latin-1 byte per code
    /// unit, clear = one UTF-16 code unit. Only meaningful (and always
    /// accurate) on `DENSE_STRING` maps — future string representations
    /// get their own kinds/flags.
    pub const LATIN1: MapKind = MapKind(1 << 14);

    pub const MAP: MapKind = MapKind(ObjectKind::Map as u64);
    pub const FIXED_ARRAY: MapKind = MapKind(ObjectKind::FixedArray as u64);
    pub const FIXED_BYTE_ARRAY: MapKind = MapKind(ObjectKind::FixedByteArray as u64);
    pub const DENSE_STRING: MapKind = MapKind(ObjectKind::DenseString as u64);
    pub const ACCESSOR_PAIR: MapKind = MapKind(ObjectKind::AccessorPair as u64);
    pub const CALLABLE_INFO: MapKind = MapKind(ObjectKind::CallableInfo as u64);
    pub const FLOAT: MapKind = MapKind(ObjectKind::Float as u64);
    pub const SYMBOL: MapKind = MapKind(ObjectKind::Symbol as u64);
    pub const HANDLER_TABLE: MapKind = MapKind(ObjectKind::HandlerTable as u64);
    pub const CONTEXT: MapKind = MapKind(ObjectKind::Context as u64);
    pub const SCOPE_INFO: MapKind = MapKind(ObjectKind::ScopeInfo as u64);
    pub const FEEDBACK_VECTOR: MapKind = MapKind(ObjectKind::FeedbackVector as u64);
    pub const OBJECT: MapKind = MapKind(ObjectKind::Object as u64);
    pub const ARRAY: MapKind = MapKind(ObjectKind::Array as u64);
    pub const BYTE_ARRAY: MapKind = MapKind(ObjectKind::ByteArray as u64);
    pub const STRING: MapKind = MapKind(ObjectKind::String as u64);
    pub const PROXY: MapKind = MapKind(ObjectKind::Proxy as u64);
    pub const ODDBALL: MapKind = MapKind(ObjectKind::Oddball as u64);

    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, flags: Self) -> bool {
        self.0 & flags.0 == flags.0
    }

    // TODO: consider transmute with debug assert
    pub const fn kind(self) -> ObjectKind {
        match Self(self.0 & Self::KIND_MASK) {
            Self::MAP => ObjectKind::Map,
            Self::FIXED_ARRAY => ObjectKind::FixedArray,
            Self::FIXED_BYTE_ARRAY => ObjectKind::FixedByteArray,
            Self::DENSE_STRING => ObjectKind::DenseString,
            Self::ACCESSOR_PAIR => ObjectKind::AccessorPair,
            Self::CALLABLE_INFO => ObjectKind::CallableInfo,
            Self::FLOAT => ObjectKind::Float,
            Self::SYMBOL => ObjectKind::Symbol,
            Self::HANDLER_TABLE => ObjectKind::HandlerTable,
            Self::CONTEXT => ObjectKind::Context,
            Self::SCOPE_INFO => ObjectKind::ScopeInfo,
            Self::FEEDBACK_VECTOR => ObjectKind::FeedbackVector,
            Self::OBJECT => ObjectKind::Object,
            Self::ARRAY => ObjectKind::Array,
            Self::BYTE_ARRAY => ObjectKind::ByteArray,
            Self::STRING => ObjectKind::String,
            Self::PROXY => ObjectKind::Proxy,
            Self::ODDBALL => ObjectKind::Oddball,
            _ => panic!("invalid object kind"),
        }
    }

    pub const fn is_builtin(self) -> bool {
        let kind = self.0 & Self::KIND_MASK;
        kind > ObjectKind::BUILTIN_START && kind < ObjectKind::BUILTIN_END
    }

    pub const fn is_extendable(self) -> bool {
        self.0 & Self::EXTENDABLE.0 != 0
    }

    pub const fn is_callable(self) -> bool {
        self.0 & Self::CALLABLE.0 != 0
    }

    pub const fn is_runtime(self) -> bool {
        self.0 & Self::RUNTIME.0 != 0
    }

    pub const fn is_constructor(self) -> bool {
        self.0 & Self::CONSTRUCTOR.0 != 0
    }

    pub const fn is_class_constructor(self) -> bool {
        self.0 & Self::CLASS_CONSTRUCTOR.0 != 0
    }
}

/// Descriptor flags. Data values live in the object's slots (the descriptor
/// holds a Smi offset); accessors embed the `AccessorPair` in the descriptor.
/// Writability is the WRITABLE attribute bit — there is no separate const kind.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SlotFlags(u64);

impl SlotFlags {
    pub const ACCESSOR: SlotFlags = SlotFlags(1 << 0);
    pub const WRITABLE: SlotFlags = SlotFlags(1 << 1);
    pub const CONFIGURABLE: SlotFlags = SlotFlags(1 << 2);
    pub const ENUMERABLE: SlotFlags = SlotFlags(1 << 3);

    /// The plain data slot: no flags set.
    pub const VALUE: SlotFlags = SlotFlags(0);

    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn is_accessor(self) -> bool {
        self.0 & Self::ACCESSOR.0 != 0
    }

    pub const fn is_writable(self) -> bool {
        self.0 & Self::WRITABLE.0 != 0
    }

    pub const fn is_configurable(self) -> bool {
        self.0 & Self::CONFIGURABLE.0 != 0
    }

    pub const fn is_enumerable(self) -> bool {
        self.0 & Self::ENUMERABLE.0 != 0
    }
}

#[repr(C)]
pub struct SlotDescriptor {
    pub name: GcSlot<SlotName>,
    pub flags: GcSlot<Smi>,
    pub value: GcSlot,
}

impl SlotDescriptor {
    pub fn name<'a>(&self, heap: &'a Heap) -> Tagged<'a, SlotName> {
        self.name.get(heap)
    }

    pub fn flags(&self) -> SlotFlags {
        SlotFlags::new(self.flags.to_smi().value() as u64)
    }

    pub fn offset(&self) -> usize {
        Smi::decode(self.value.raw()).expect("slot offset").value() as usize
    }
}

/// A property name: a string (the interner's canonical instance, by
/// convention — names compare by pointer), a symbol, or a smi index.
#[repr(transparent)]
pub struct SlotName(Value);

impl SlotName {
    /// The erased name word.
    pub fn into_value(self) -> Value {
        self.0
    }
}

impl<'a> From<Tagged<'a, DenseString>> for Tagged<'a, SlotName> {
    fn from(string: Tagged<'a, DenseString>) -> Self {
        // Safety: type-level narrowing
        unsafe { string.cast() }
    }
}

impl<'a> From<Tagged<'a, Symbol>> for Tagged<'a, SlotName> {
    fn from(symbol: Tagged<'a, Symbol>) -> Self {
        // Safety: type-level narrowing
        unsafe { symbol.cast() }
    }
}

impl<'a> From<Smi> for Tagged<'a, SlotName> {
    fn from(smi: Smi) -> Self {
        // Safety: Smi words are pointer-free
        unsafe { Tagged::from_value_unchecked(smi.encode()) }
    }
}

#[repr(C)]
pub struct AccessorPair {
    pub header: Header,
    pub get: GcSlot,
    pub set: GcSlot,
}

impl HeapObject for AccessorPair {
    const KIND: ObjectKind = ObjectKind::AccessorPair;
    type Init<'a> = (Handle<'a, Value>, Handle<'a, Value>);

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        let host = self.tagged(heap);
        self.header
            .map
            .set(heap, host, heap.known().accessor_pair_map.as_tagged(heap));
        // the config's handles are roots; no allocation runs inside init
        self.get.set(heap, host, config.0.as_tagged(heap));
        self.set.set(heap, host, config.1.as_tagged(heap));
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for AccessorPair {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.get.as_raw());
        visitor.visit(self.set.as_raw());
    }
}
