use core::{
    alloc::Layout,
    cell::{Cell, UnsafeCell},
};

use crate::{
    EdgeVisitable, GcSlot, Handle, HeapRef, LocalHeap, NoGc, OptionGcSlot, PointerStrength,
    STRONG_PTR, Smi, Tagged, TransitionGuard, Value, Visitor, WEAK_PTR, Word,
};

pub trait HeapObject: 'static {
    type Init<'a>;

    fn layout_for(config: &Self::Init<'_>) -> Layout;

    fn init(&mut self, heap: &impl LocalHeap, config: &Self::Init<'_>);

    fn header(&self) -> &Header;

    fn layout(&self) -> Layout;

    fn erase(&self) -> Value
    where
        Self: Sized,
    {
        let addr = self as *const Self as *const Word as Word;
        unsafe { Value::from_bits(addr | STRONG_PTR) }
    }

    fn erase_weak(&self) -> Value
    where
        Self: Sized,
    {
        let addr = self as *const Self as *const Word as Word;
        unsafe { Value::from_bits(addr | WEAK_PTR) }
    }
}

#[repr(C)]
pub struct Header {
    pub map: GcSlot<Map>,
}

impl Header {
    pub fn map(&self) -> Tagged<Map> {
        self.map.get()
    }
}

#[repr(C)]
pub struct Map {
    pub header: Header,
    pub value_slot_count: GcSlot<Smi>,
    pub descriptor_count: GcSlot<Smi>,
    /// Object kind tag (low byte) and capability flags (second byte).
    pub kind: GcSlot<Smi>,
    /// Empty, or a `FixedArray` of flat `[name, target_map]` transition pairs.
    // TODO: make transition targets weak (V8 does this so unused shape subtrees die)
    pub transitions: OptionGcSlot<FixedArray>,
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
        nogc: &'a NoGc<'a>,
        heap: &impl LocalHeap,
        name: SlotName,
        flags: SlotFlags,
    ) -> Option<HeapRef<'a, Map>> {
        let lock = heap.transition_lock();
        let guard = lock.acquire();
        self.find_transition_locked(nogc, heap, name, flags, &guard)
    }

    pub fn find_transition_locked<'a>(
        &self,
        nogc: &'a NoGc<'a>,
        heap: &impl LocalHeap,
        name: SlotName,
        flags: SlotFlags,
        _guard: &TransitionGuard<'_>,
    ) -> Option<HeapRef<'a, Map>> {
        let array = self.transitions.heap_ref(nogc, heap)?;
        let pairs = array.as_slice();
        debug_assert!(
            pairs.len() % 2 == 0,
            "transition pairs are flat [name, map]"
        );
        for pair in pairs.chunks_exact(2) {
            if pair[0].inner() != name.value() {
                continue;
            }

            let target = pair[1]
                .inner()
                .get_as::<Map>(nogc, heap.known().map_map)
                .expect("transition target must be a map");

            let count = target.descriptor_count();
            if count > 0 && target.descriptor(count - 1).flags() == flags {
                return Some(target);
            }
        }
        None
    }
}

pub struct MapInit<'a> {
    pub kind: MapKind,
    pub value_slot_count: usize,
    pub descriptors: &'a [(SlotName, SlotFlags, Value)],
}

impl HeapObject for Map {
    type Init<'a> = MapInit<'a>;

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.descriptors.len())
    }

    fn init(&mut self, heap: &impl LocalHeap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(heap, host, heap.known().map_map.as_tagged());
        self.value_slot_count
            .set(heap, host, Smi::new(config.value_slot_count as i64));
        self.descriptor_count
            .set(heap, host, Smi::new(config.descriptors.len() as i64));
        self.kind
            .set(heap, host, Smi::new(config.kind.bits() as i64));
        self.transitions.clear(heap.known().void.value());
        for (i, (name, flags, value)) in config.descriptors.iter().enumerate() {
            let d = self.descriptor(i);
            d.name.set(heap, host, name.tagged());
            d.flags.set(heap, host, Smi::new(flags.bits() as i64));
            d.value.set(heap, host, Tagged::from_value(*value));
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
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.transitions.as_raw());
        for d in self.descriptors() {
            visitor.visit(d.name.as_raw());
            visitor.visit(d.value.as_raw());
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u64)]
pub enum ObjectKind {
    BuiltinStart = 0,
    Map = 1,
    FixedArray = 2,
    FixedByteArray = 3,
    VMString = 4,
    AccessorPair = 5,
    CallableInfo = 6,
    Float = 7,
    Symbol = 8,
    BuiltinEnd = 9,

    /// elements and len empty
    Object = 10,
    /// `elements` points to a `FixedArray`.
    Array = 11,
    /// `elements` points to a `FixedByteArray`.
    ByteArray = 12,
    /// `elements` points to a `VMString`.
    String = 13,
}

impl ObjectKind {
    pub const BUILTIN_START: u64 = ObjectKind::BuiltinStart as u64;
    pub const BUILTIN_END: u64 = ObjectKind::BuiltinEnd as u64;
}

/// Low byte: the `ObjectKind`. Second byte: capability flags
/// (extendable, callable, constructor, native). Constructor implies callable.
/// NATIVE is only valid together with CALLABLE and means slots[0] of the
/// object is a Smi native registry index instead of a `CallableInfoObject`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MapKind(u64);

impl MapKind {
    const KIND_MASK: u64 = 0xff;

    pub const EXTENDABLE: MapKind = MapKind(1 << 8);
    pub const CALLABLE: MapKind = MapKind(1 << 9);
    pub const CONSTRUCTOR: MapKind = MapKind(1 << 10);
    pub const NATIVE: MapKind = MapKind(1 << 11);

    pub const MAP: MapKind = MapKind(ObjectKind::Map as u64);
    pub const FIXED_ARRAY: MapKind = MapKind(ObjectKind::FixedArray as u64);
    pub const FIXED_BYTE_ARRAY: MapKind = MapKind(ObjectKind::FixedByteArray as u64);
    pub const VM_STRING: MapKind = MapKind(ObjectKind::VMString as u64);
    pub const ACCESSOR_PAIR: MapKind = MapKind(ObjectKind::AccessorPair as u64);
    pub const CALLABLE_INFO: MapKind = MapKind(ObjectKind::CallableInfo as u64);
    pub const FLOAT: MapKind = MapKind(ObjectKind::Float as u64);
    pub const SYMBOL: MapKind = MapKind(ObjectKind::Symbol as u64);
    pub const OBJECT: MapKind = MapKind(ObjectKind::Object as u64);
    pub const ARRAY: MapKind = MapKind(ObjectKind::Array as u64);
    pub const BYTE_ARRAY: MapKind = MapKind(ObjectKind::ByteArray as u64);
    pub const STRING: MapKind = MapKind(ObjectKind::String as u64);

    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    // TODO: consider transmute with debug assert
    pub const fn kind(self) -> ObjectKind {
        match Self(self.0 & Self::KIND_MASK) {
            Self::MAP => ObjectKind::Map,
            Self::FIXED_ARRAY => ObjectKind::FixedArray,
            Self::FIXED_BYTE_ARRAY => ObjectKind::FixedByteArray,
            Self::VM_STRING => ObjectKind::VMString,
            Self::ACCESSOR_PAIR => ObjectKind::AccessorPair,
            Self::CALLABLE_INFO => ObjectKind::CallableInfo,
            Self::FLOAT => ObjectKind::Float,
            Self::SYMBOL => ObjectKind::Symbol,
            Self::OBJECT => ObjectKind::Object,
            Self::ARRAY => ObjectKind::Array,
            Self::BYTE_ARRAY => ObjectKind::ByteArray,
            Self::STRING => ObjectKind::String,
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

    pub const fn is_native(self) -> bool {
        self.0 & Self::NATIVE.0 != 0
    }

    pub const fn is_constructor(self) -> bool {
        self.0 & Self::CONSTRUCTOR.0 != 0
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SlotKind {
    Value,
    Const,
    Accessor,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SlotFlags(u64);

impl SlotFlags {
    const KIND_MASK: u64 = 0b11;
    const KIND_VALUE: u64 = 0b00;
    const KIND_CONST: u64 = 0b01;
    const KIND_ACCESSOR: u64 = 0b10;

    pub const PARENT: SlotFlags = SlotFlags(1 << 2);
    pub const WRITABLE: SlotFlags = SlotFlags(1 << 3);
    pub const CONFIGURABLE: SlotFlags = SlotFlags(1 << 4);
    pub const ENUMERABLE: SlotFlags = SlotFlags(1 << 5);

    pub const VALUE: SlotFlags = SlotFlags(Self::KIND_VALUE);
    pub const CONST: SlotFlags = SlotFlags(Self::KIND_CONST);
    pub const ACCESSOR: SlotFlags = SlotFlags(Self::KIND_ACCESSOR);

    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn kind(self) -> SlotKind {
        match self.0 & Self::KIND_MASK {
            Self::KIND_VALUE => SlotKind::Value,
            Self::KIND_CONST => SlotKind::Const,
            Self::KIND_ACCESSOR => SlotKind::Accessor,
            _ => panic!("invalid slot kind"),
        }
    }

    pub const fn is_parent(self) -> bool {
        self.0 & Self::PARENT.0 != 0
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
    pub fn name(&self) -> SlotName {
        SlotName(self.name.inner())
    }

    pub fn flags(&self) -> SlotFlags {
        SlotFlags::new(self.flags.to_smi().value() as u64)
    }

    pub fn offset(&self) -> usize {
        Smi::decode(self.value.inner())
            .expect("slot offset")
            .value() as usize
    }
}

#[repr(C)]
pub struct Object {
    pub header: Header,
    pub slots: GcSlot<FixedArray>,
    pub elements: GcSlot,
    pub length: GcSlot<Smi>,
}

impl Object {
    pub fn layout_for() -> Layout {
        Layout::new::<Self>()
    }

    pub fn callable_info<'a>(
        &'a self,
        guard: &'a NoGc<'a>,
        heap: &impl LocalHeap,
    ) -> Option<HeapRef<'a, CallableInfoObject>> {
        if !self.header.map.heap_ref(guard).kind().is_callable() {
            return None;
        }
        let info = self.slots.heap_ref(guard).at(0);
        info.get_as(guard, heap.known().callable_map)
    }

    pub fn native_index<'a>(&'a self, guard: &'a NoGc<'a>) -> Option<usize> {
        if !self.header.map.heap_ref(guard).kind().is_native() {
            return None;
        }
        let idx = Smi::decode(self.slots.heap_ref(guard).at(0))?.value();
        usize::try_from(idx).ok()
    }
}

pub struct ObjectInit<'a> {
    pub map: Handle<'a, Map>,
    pub slots: Handle<'a, FixedArray>,
    pub elements: Value,
    pub length: usize,
}

pub struct ObjectSlotsInit<'m, 'v> {
    pub map: Handle<'m, Map>,
    pub values: &'v [Value],
    pub elements: Value,
    pub length: usize,
}

impl HeapObject for Object {
    type Init<'a> = ObjectInit<'a>;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Self::layout_for()
    }

    fn init(&mut self, heap: &impl LocalHeap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header.map.set(heap, host, config.map.as_tagged());
        self.slots.set(heap, host, config.slots.as_tagged());
        self.elements
            .set(heap, host, Tagged::from_value(config.elements));
        self.length.set(heap, host, Smi::new(config.length as i64));
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for()
    }
}

impl EdgeVisitable for Object {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.slots.as_raw());
        visitor.visit(self.elements.as_raw());
    }
}

#[repr(C)]
pub struct FixedArray {
    pub header: Header,
    pub size: GcSlot<Smi>,
    pub values: [GcSlot; 0],
}

impl FixedArray {
    pub fn layout_for(len: usize) -> Layout {
        let values_layout = Layout::array::<GcSlot>(len).expect("values layout");
        Layout::new::<Self>()
            .extend(values_layout)
            .expect("array layout")
            .0
    }

    pub fn len(&self) -> usize {
        self.size.to_smi().value() as usize
    }

    pub fn data_ptr(&self) -> *mut GcSlot {
        self.values.as_ptr() as *mut GcSlot
    }

    pub fn as_slice(&self) -> &[GcSlot] {
        unsafe { core::slice::from_raw_parts(self.data_ptr(), self.len()) }
    }

    pub fn element_slot(&self, i: usize) -> &GcSlot {
        debug_assert!(i < self.len());
        unsafe { &*self.values.as_ptr().add(i) }
    }

    pub fn at(&self, i: usize) -> Value {
        self.element_slot(i).get().erase()
    }

    pub fn set(&self, heap: &impl LocalHeap, i: usize, v: Value) {
        self.element_slot(i)
            .set(heap, self.erase(), Tagged::from_value(v));
    }
}

impl HeapObject for FixedArray {
    type Init<'a> = &'a [Value];

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.len())
    }

    fn init(&mut self, heap: &impl LocalHeap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(heap, host, heap.known().array_map.as_tagged());
        self.size.set(heap, host, Smi::new(config.len() as i64));
        for (i, v) in config.iter().enumerate() {
            self.element_slot(i).set(heap, host, Tagged::from_value(*v));
        }
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.len())
    }
}

impl EdgeVisitable for FixedArray {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        visitor.visit(self.header.map.as_raw());
        let size = self.size.to_smi().value() as usize;
        for i in 0..size {
            visitor.visit(self.element_slot(i).as_raw());
        }
    }
}

#[repr(C)]
pub struct FixedByteArray {
    pub header: Header,
    pub size: GcSlot<Smi>,
    pub values: [UnsafeCell<u8>; 0],
}

impl FixedByteArray {
    pub fn layout_for(len: usize) -> Layout {
        let values_layout = Layout::array::<u8>(len).expect("values layout");
        Layout::new::<Self>()
            .extend(values_layout)
            .expect("bytearray layout")
            .0
    }

    pub fn len(&self) -> usize {
        self.size.to_smi().value() as usize
    }

    fn data_ptr(&self) -> *mut u8 {
        UnsafeCell::raw_get(self.values.as_ptr())
    }

    pub fn get(&self, i: usize) -> u8 {
        debug_assert!(i < self.len());
        unsafe { *self.data_ptr().add(i) }
    }
    pub fn set(&self, i: usize, b: u8) {
        debug_assert!(i < self.len());
        unsafe { *self.data_ptr().add(i) = b }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.data_ptr(), self.len()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.data_ptr(), self.len()) }
    }
}

impl HeapObject for FixedByteArray {
    type Init<'a> = &'a [u8];

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.len())
    }

    fn init(&mut self, heap: &impl LocalHeap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(heap, host, heap.known().byte_array_map.as_tagged());
        self.size.set(heap, host, Smi::new(config.len() as i64));
        for (i, b) in config.iter().enumerate() {
            self.set(i, *b);
        }
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.len())
    }
}

impl EdgeVisitable for FixedByteArray {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        visitor.visit(self.header.map.as_raw());
    }
}

#[repr(C)]
pub struct VMString {
    pub header: Header,
    pub backing: GcSlot<FixedByteArray>,
    pub hash: GcSlot<Smi>,
}

/// Content hash for strings (FNV-1a, masked into smi range).
/// TODO: decide on a hash algorithm
pub fn string_content_hash(bytes: &[u8]) -> i64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h & ((1 << 62) - 1)) as i64
}

impl VMString {
    pub fn backing<'a>(&self, nogc: &'a NoGc<'a>) -> HeapRef<'a, FixedByteArray> {
        self.backing.heap_ref(nogc)
    }

    pub fn hash(&self) -> i64 {
        self.hash.to_smi().value()
    }

    pub fn len<'a>(&self, nogc: &'a NoGc<'a>) -> usize {
        self.backing(nogc).len()
    }

    pub fn as_slice<'a>(&self, nogc: &'a NoGc<'a>) -> &'a [u8] {
        self.backing(nogc).as_ref().as_slice()
    }

    pub fn as_str<'a>(&self, nogc: &'a NoGc<'a>) -> Option<&'a str> {
        core::str::from_utf8(self.as_slice(nogc)).ok()
    }
}

impl HeapObject for VMString {
    type Init<'a> = (Handle<'a, FixedByteArray>, i64);

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, heap: &impl LocalHeap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(heap, host, heap.known().string_map.as_tagged());
        self.backing.set(heap, host, config.0.as_tagged());
        self.hash.set(heap, host, Smi::new(config.1));
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for VMString {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.backing.as_raw());
    }
}

#[repr(C)]
pub struct InternedString(pub VMString);

impl InternedString {
    pub fn string(&self) -> &VMString {
        &self.0
    }
}

impl HeapObject for InternedString {
    type Init<'a> = (Handle<'a, FixedByteArray>, i64);

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, heap: &impl LocalHeap, config: &Self::Init<'_>) {
        self.0.init(heap, config);
    }

    fn header(&self) -> &Header {
        self.0.header()
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for InternedString {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        self.0.visit_edges(visitor);
    }
}

#[repr(C)]
pub struct Symbol {
    pub header: Header,
    pub backing: GcSlot<FixedByteArray>,
}

impl Symbol {
    pub fn backing<'a>(&self, nogc: &'a NoGc<'a>) -> HeapRef<'a, FixedByteArray> {
        self.backing.heap_ref(nogc)
    }

    pub fn len<'a>(&self, nogc: &'a NoGc<'a>) -> usize {
        self.backing(nogc).len()
    }

    pub fn as_slice<'a>(&self, nogc: &'a NoGc<'a>) -> &'a [u8] {
        self.backing(nogc).as_ref().as_slice()
    }

    pub fn as_str<'a>(&self, nogc: &'a NoGc<'a>) -> Option<&'a str> {
        core::str::from_utf8(self.as_slice(nogc)).ok()
    }
}

impl HeapObject for Symbol {
    type Init<'a> = Handle<'a, FixedByteArray>;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, heap: &impl LocalHeap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(heap, host, heap.known().symbol_map.as_tagged());
        self.backing.set(heap, host, config.as_tagged());
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for Symbol {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.backing.as_raw());
    }
}

/// A property name: an interned string, a symbol, or a smi index.
#[repr(transparent)]
#[derive(Debug, Copy, Clone)]
pub struct SlotName(Value);

impl SlotName {
    pub fn value(self) -> Value {
        self.0
    }

    pub fn from_value(value: Value) -> Self {
        Self(value)
    }

    pub fn tagged(self) -> Tagged<SlotName> {
        unsafe { Tagged::from_value_unchecked(self.0) }
    }
}

impl From<Tagged<VMString>> for SlotName {
    fn from(string: Tagged<VMString>) -> Self {
        Self(string.erase())
    }
}

impl From<Tagged<Symbol>> for SlotName {
    fn from(symbol: Tagged<Symbol>) -> Self {
        Self(symbol.erase())
    }
}

impl From<Tagged<InternedString>> for SlotName {
    fn from(string: Tagged<InternedString>) -> Self {
        Self(string.erase())
    }
}

impl From<Tagged<Smi>> for SlotName {
    fn from(smi: Tagged<Smi>) -> Self {
        Self(smi.erase())
    }
}

impl<R: PointerStrength> From<Handle<'_, SlotName, R>> for SlotName {
    fn from(name: Handle<'_, SlotName, R>) -> Self {
        Self::from_value(name.value())
    }
}

impl PartialEq for SlotName {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for SlotName {}

#[repr(C)]
pub struct AccessorPair {
    pub header: Header,
    pub get: GcSlot,
    pub set: GcSlot,
}

impl HeapObject for AccessorPair {
    type Init<'a> = (Value, Value);

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, heap: &impl LocalHeap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(heap, host, heap.known().accessor_pair_map.as_tagged());
        self.get.set(heap, host, Tagged::from_value(config.0));
        self.set.set(heap, host, Tagged::from_value(config.1));
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for AccessorPair {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.get.as_raw());
        visitor.visit(self.set.as_raw());
    }
}

#[repr(C)]
pub struct CallableInfoObject {
    pub header: Header,
    pub bytecode: GcSlot<FixedByteArray>,
    pub constants: GcSlot<FixedArray>,
    pub register_count: GcSlot<Smi>,
    pub context: GcSlot,
}

pub struct CallableInfoInit<'a> {
    pub bytecode: Handle<'a, FixedByteArray>,
    pub constants: Handle<'a, FixedArray>,
    pub register_count: usize,
    pub context: Value,
}

impl HeapObject for CallableInfoObject {
    type Init<'a> = CallableInfoInit<'a>;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, heap: &impl LocalHeap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(heap, host, heap.known().callable_map.as_tagged());
        self.bytecode.set(heap, host, config.bytecode.as_tagged());
        self.constants.set(heap, host, config.constants.as_tagged());
        self.register_count
            .set(heap, host, Smi::new(config.register_count as i64));
        self.context
            .set(heap, host, Tagged::from_value(config.context));
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for CallableInfoObject {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.bytecode.as_raw());
        visitor.visit(self.constants.as_raw());
        visitor.visit(self.context.as_raw());
    }
}

#[repr(C)]
pub struct Float {
    pub header: Header,
    pub value: Cell<f64>,
}

impl HeapObject for Float {
    type Init<'a> = f64;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, heap: &impl LocalHeap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(heap, host, heap.known().float_map.as_tagged());
        self.value.set(*config);
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for Float {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        visitor.visit(self.header.map.as_raw());
    }
}
