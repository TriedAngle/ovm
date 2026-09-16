use core::{alloc::Layout, cell::UnsafeCell};

use crate::{EdgeVisitable, GcSlot, Header, HeapObject, NoGc, ObjectKind, Smi, Visitor};

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
    const KIND: ObjectKind = ObjectKind::FixedByteArray;
    type Init<'a> = &'a [u8];

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.len())
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().byte_array_map.as_tagged());
        self.size.set(nogc, host, Smi::new(config.len() as i64));
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
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
    }
}
