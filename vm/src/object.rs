use core::{alloc::Layout, cell::UnsafeCell};

use crate::{
    EdgeVisitable, GcSlot, LocalHeap, RootVisitor, Smi, Tagged, Value, Word, value::{STRONG_PTR, WEAK_PTR}
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
    map: GcSlot,
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
}

#[repr(C)]
pub struct Map {
    pub header: Header,
    pub kind: ObjectKind,
}
// TODO impl map and traits for map

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
        self.size.get().to_smi().expect("size").value() as usize
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
        visitor.visit_slot(&self.header.map);
        let size = self.size.get().to_smi().expect("size").value() as usize;
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
        self.size.get().to_smi().expect("size").value() as usize
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
        visitor.visit_slot(&self.header.map);
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
        self.hash.get().to_smi().expect("hash").value()
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
        visitor.visit_slot(&self.header.map);
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
        visitor.visit_slot(&self.header.map);
        visitor.visit_slot(self.backing.ereased());
    }
}

#[repr(C)]
pub union SlotName { 
    pub string: Tagged<VMString>,
    pub symbol: Tagged<Symbol>,
    pub smi: Tagged<Smi>,
}

impl From<Tagged<VMString>> for SlotName {
    fn from(string: Tagged<VMString>) -> Self {
        Self { string }
    }
}

impl From<Tagged<Symbol>> for SlotName {
    fn from(symbol: Tagged<Symbol>) -> Self {
        Self { symbol }
    }
}

impl From<Tagged<Smi>> for SlotName {
    fn from(smi: Tagged<Smi>) -> Self {
        Self { smi }
    }
}

impl PartialEq for SlotName {
    fn eq(&self, other: &Self) -> bool {
        // all variants are one word; compare the raw tagged bits
        unsafe { self.smi.erase() == other.smi.erase() }
    }
}

impl Eq for SlotName {}
