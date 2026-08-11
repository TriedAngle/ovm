use core::{alloc::Layout, cell::UnsafeCell};

use crate::{
    EdgeVisitable, GcSlot, HeapRef, LocalHeap, NoGc, RootVisitor, Smi, Tagged, Value, Word,
    value::{STRONG_PTR, WEAK_PTR}
};

pub trait HeapObject: 'static {
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
    pub fn map(&self) -> Tagged<Map> {
        self.map.get()
    }
}

#[repr(u16)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ObjectKind {
    Map,
    Array,
    ByteArray,
    String,
    Symbol,
    SlotsObject,
    CallableObject,
    AccessorPair,
}

impl ObjectKind {
    pub fn from_smi(smi: Smi) -> Option<Self> {
        Some(match smi.value() as u16 {
            0 => Self::Map,
            1 => Self::Array,
            2 => Self::ByteArray,
            3 => Self::String,
            4 => Self::Symbol,
            5 => Self::SlotsObject,
            6 => Self::CallableObject,
            7 => Self::AccessorPair,
            _ => return None,
        })
    }

    pub fn to_smi(self) -> Smi {
        Smi::new_unchecked(self as i64)
    }
}

#[repr(C)]
pub struct Map {
    pub header: Header,
    /// ObjectKind
    pub kind: GcSlot<Smi>,           
    pub value_slot_count: GcSlot<Smi>,
    pub descriptor_count: GcSlot<Smi>,
    pub descriptors: [SlotDescriptor; 0],
}

impl Map {
    pub fn layout_for(descriptor_count: usize) -> Layout {
        let descriptors_layout = Layout::array::<SlotDescriptor>(descriptor_count)
            .expect("descriptors layout");
        Layout::new::<Self>()
            .extend(descriptors_layout)
            .expect("map layout")
            .0
    }

    pub fn object_kind(&self) -> ObjectKind {
        ObjectKind::from_smi(self.kind.to_smi()).expect("invalid object kind")
    }

    pub fn value_slot_count(&self) -> usize {
        self.value_slot_count.to_smi().value() as usize
    }

    pub fn descriptor_count(&self) -> usize {
        self.descriptor_count.to_smi().value() as usize
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

    pub fn lookup<'a>(&'a self, guard: &'a NoGc<'a>, obj: HeapRef<'a, SlotsObject>, name: SlotName) -> Lookup<'a> {
        for (index, d) in self.descriptors().iter().enumerate() {
            if !d.flags().is_parent() && d.name() == name {
                return match d.flags().kind() {
                    SlotKind::Value => Lookup::Data {
                        holder: obj,
                        map_index: index,
                        holder_index: d.offset(),
                        value: d.read(obj.as_ref()),
                    },
                    SlotKind::Const => Lookup::Const { holder: obj, map_index: index, value: d.read(obj.as_ref()) },
                    SlotKind::Accessor => Lookup::Accessor {
                        holder: obj,
                        map_index: index,
                        pair: unsafe { guard.get_unchecked(d.value.get().cast_unchecked()) },
                    },
                };
            }
        }

        for d in self.descriptors() {
            if d.flags().is_parent() {
                let parent = unsafe { guard.get_unchecked::<SlotsObject>(d.slot(obj.as_ref()).get().cast_unchecked()) };
                let result = parent.as_ref().lookup(guard, name);
                if !matches!(result, Lookup::NotFound) {
                    return result;
                }
            }
        }

        Lookup::NotFound
    }

    pub fn lookup_parent<'a>(&'a self, guard: &'a NoGc<'a>, obj: HeapRef<'a, SlotsObject>, name: SlotName, parent: SlotName) -> Lookup<'a> {
        match self.find_parent(parent) {
            Some(d) => {
                let parent = unsafe { guard.get_unchecked::<SlotsObject>(d.slot(obj.as_ref()).get().cast_unchecked()) };
                parent.as_ref().lookup(guard, name)
            }
            None => Lookup::NotFound,
        }
    }

    pub fn find_parent(&self, name: SlotName) -> Option<&SlotDescriptor> {
        self.descriptors()
            .iter()
            .find(|d| d.flags().is_parent() && d.name() == name)
    }
}

impl HeapObject for Map {
    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.descriptor_count())
    }
}

impl EdgeVisitable for Map {
    fn visit_edges(&self, visitor: &mut impl RootVisitor) {
        visitor.visit_slot(self.header.map.ereased());
        for d in self.descriptors() {
            visitor.visit_slot(d.name.ereased());
            visitor.visit_slot(d.value.ereased());
        }
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
        Smi::decode(self.value.inner()).expect("slot offset").value() as usize
    }

    pub fn slot<'s>(&'s self, obj: &'s SlotsObject) -> &'s GcSlot {
        match self.flags().kind() {
            SlotKind::Const | SlotKind::Accessor => &self.value,
            SlotKind::Value => obj.slot(self.offset()),
        }
    }

    pub fn read(&self, obj: &SlotsObject) -> Value {
        self.slot(obj).get().erase()
    }
}

#[repr(C)]
pub struct Array {
    pub header: Header,
    pub size: GcSlot<Smi>,
    pub values: [GcSlot; 0],
}

impl Array {
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
        self.element_slot(i).set(heap, self.erase(), Tagged::from_value(v));
    }
}

impl HeapObject for Array {
    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.len())
    }
}

impl EdgeVisitable for Array {
    fn visit_edges(&self, visitor: &mut impl RootVisitor) {
        visitor.visit_slot(self.header.map.ereased());
        let size = self.size.to_smi().value() as usize;
        for i in 0..size {
            visitor.visit_slot(self.element_slot(i));
        }
    }
}

#[repr(C)]
pub struct ByteArray {
    pub header: Header,
    pub size: GcSlot<Smi>,
    pub values: [UnsafeCell<u8>; 0],
}

impl ByteArray {
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

    /// Initialize a freshly allocated bytearray: set its length and copy
    /// `bytes` into it. Must be allocated with `layout_for(bytes.len())`.
    pub fn init(&self, heap: &impl LocalHeap, bytes: &[u8]) {
        self.size.set(heap, self.erase(), Smi::new_unchecked(bytes.len() as i64));
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data_ptr(), bytes.len()) };
    }
}

impl HeapObject for ByteArray {
    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.len())
    }
}

impl EdgeVisitable for ByteArray {
    fn visit_edges(&self, visitor: &mut impl RootVisitor) {
        visitor.visit_slot(self.header.map.ereased());
    }
}

#[repr(C)]
pub struct VMString { 
    pub header: Header,
    pub backing: GcSlot<ByteArray>,
    pub hash: GcSlot<Smi>
}

impl VMString {
    pub fn backing(&self) -> &ByteArray {
        let ptr = self.backing.get().as_ptr().expect("string backing");
        unsafe { ptr.as_ref() }
    }

    pub fn hash(&self) -> i64 {
        self.hash.to_smi().value()
    }

    pub fn len(&self) -> usize {
        self.backing().len()
    }

    pub fn as_slice(&self) -> &[u8] {
        self.backing().as_slice()
    }

    pub fn as_str(&self) -> Option<&str> {
        core::str::from_utf8(self.as_slice()).ok()
    }
}

impl HeapObject for VMString {
    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for VMString {
    fn visit_edges(&self, visitor: &mut impl RootVisitor) {
        visitor.visit_slot(self.header.map.ereased());
        visitor.visit_slot(self.backing.ereased());
    }
}

#[repr(C)]
pub struct InternedString(VMString);

impl InternedString {
    pub fn string(&self) -> &VMString {
        &self.0
    }
}

impl HeapObject for InternedString {
    fn header(&self) -> &Header {
        self.0.header()
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for InternedString {
    fn visit_edges(&self, visitor: &mut impl RootVisitor) {
        self.0.visit_edges(visitor);
    }
}

#[repr(C)]
pub struct Symbol {
    pub header: Header,
    pub backing: GcSlot<ByteArray>,
}

impl Symbol {
    pub fn backing(&self) -> &ByteArray {
        let ptr = self.backing.get().as_ptr().expect("symbol backing");
        unsafe { ptr.as_ref() }
    }

    pub fn len(&self) -> usize {
        self.backing().len()
    }

    pub fn as_slice(&self) -> &[u8] {
        self.backing().as_slice()
    }

    pub fn as_str(&self) -> Option<&str> {
        core::str::from_utf8(self.as_slice()).ok()
    }
}

impl HeapObject for Symbol {
    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for Symbol {
    fn visit_edges(&self, visitor: &mut impl RootVisitor) {
        visitor.visit_slot(self.header.map.ereased());
        visitor.visit_slot(self.backing.ereased());
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

impl From<Tagged<Smi>> for SlotName {
    fn from(smi: Tagged<Smi>) -> Self {
        Self(smi.erase())
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
    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for AccessorPair {
    fn visit_edges(&self, visitor: &mut impl RootVisitor) {
        visitor.visit_slot(self.header.map.ereased());
        visitor.visit_slot(self.get.ereased());
        visitor.visit_slot(self.set.ereased());
    }
}

#[repr(C)]
pub struct SlotsObject {
    pub header: Header,
    pub size: GcSlot<Smi>,
    pub slots: [GcSlot; 0],
}

impl SlotsObject {
    pub fn layout_for(len: usize) -> Layout {
        let slots_layout = Layout::array::<GcSlot>(len).expect("slots layout");
        Layout::new::<Self>()
            .extend(slots_layout)
            .expect("slots object layout")
            .0
    }

    pub fn len(&self) -> usize {
        self.size.to_smi().value() as usize
    }

    pub fn slot(&self, i: usize) -> &GcSlot {
        debug_assert!(i < self.len());
        unsafe { &*self.slots.as_ptr().add(i) }
    }

    pub fn lookup<'a>(&'a self, guard: &'a NoGc<'a>, name: SlotName) -> Lookup<'a> {
        guard.get(&self.header.map).as_ref().lookup(guard, HeapRef::from_ref(self), name)
    }

    pub fn lookup_parent<'a>(&'a self, guard: &'a NoGc<'a>, name: SlotName, parent: SlotName) -> Lookup<'a> {
        guard.get(&self.header.map).as_ref().lookup_parent(guard, HeapRef::from_ref(self), name, parent)
    }
}

pub enum Lookup<'a> {
    Data { holder: HeapRef<'a, SlotsObject>, map_index: usize, holder_index: usize, value: Value },
    Const { holder: HeapRef<'a, SlotsObject>, map_index: usize, value: Value },
    Accessor { holder: HeapRef<'a, SlotsObject>, map_index: usize, pair: HeapRef<'a, AccessorPair> },
    NotFound,
}

impl HeapObject for SlotsObject {
    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.len())
    }
}

impl EdgeVisitable for SlotsObject {
    fn visit_edges(&self, visitor: &mut impl RootVisitor) {
        visitor.visit_slot(self.header.map.ereased());
        for i in 0..self.len() {
            visitor.visit_slot(self.slot(i));
        }
    }
}

#[repr(C)]
pub struct CallableObject {
    pub header: Header,
    pub callable_info: GcSlot,
    pub context: GcSlot,
    pub size: GcSlot<Smi>,
    pub slots: [GcSlot; 0],
}

impl CallableObject {
    pub fn layout_for(len: usize) -> Layout {
        let slots_layout = Layout::array::<GcSlot>(len).expect("slots layout");
        Layout::new::<Self>()
            .extend(slots_layout)
            .expect("callable object layout")
            .0
    }

    pub fn len(&self) -> usize {
        self.size.to_smi().value() as usize
    }

    pub fn slot(&self, i: usize) -> &GcSlot {
        debug_assert!(i < self.len());
        unsafe { &*self.slots.as_ptr().add(i) }
    }
}

impl HeapObject for CallableObject {
    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.len())
    }
}

impl EdgeVisitable for CallableObject {
    fn visit_edges(&self, visitor: &mut impl RootVisitor) {
        visitor.visit_slot(self.header.map.ereased());
        visitor.visit_slot(self.callable_info.ereased());
        visitor.visit_slot(self.context.ereased());
        for i in 0..self.len() {
            visitor.visit_slot(self.slot(i));
        }
    }
}

impl AsRef<str> for VMString {
    fn as_ref(&self) -> &str {
        self.as_str().expect("string must be valid utf8")
    }
}

impl AsRef<str> for InternedString {
    fn as_ref(&self) -> &str {
        self.string().as_ref()
    }
}

impl AsRef<str> for Symbol {
    fn as_ref(&self) -> &str {
        self.as_str().expect("symbol must be valid utf8")
    }
}
