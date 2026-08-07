use core::{alloc::Layout, cell::UnsafeCell};

use crate::{
    EdgeVisitable, GcSlot, LocalHeap, RootVisitor, Smi, Value, Word,
    value::{STRONG_PTR, WEAK_PTR},
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

pub enum ObjectKind {
    Map,
    Array,
    Slots,
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
        self.size.get().value() as usize
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
        self.element_slot(i).get()
    }

    pub fn set(&self, heap: &impl LocalHeap, i: usize, v: Value) {
        self.element_slot(i).set(heap, self.erase(), v);
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
        let size = self.size.get().value() as usize;
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
        self.size.get().value() as usize
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
