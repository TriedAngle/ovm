use core::alloc::Layout;

use crate::{EdgeVisitable, GcSlot, GcSlice, Header, HeapObject, NoGc, ObjectKind, Smi, Value, Visitor};

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

    pub fn set(&self, nogc: &NoGc<'_>, i: usize, v: Value) {
        self.element_slot(i).set(nogc, self.erase(), v);
    }
}

impl HeapObject for FixedArray {
    const KIND: ObjectKind = ObjectKind::FixedArray;
    type Init<'a> = GcSlice<'a>;

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.len())
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().array_map.as_tagged());
        self.size.set(nogc, host, Smi::new(config.len() as i64));
        for (i, v) in config.iter().enumerate() {
            self.element_slot(i).set(nogc, host, *v);
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
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        let size = self.size.to_smi().value() as usize;
        for i in 0..size {
            visitor.visit(self.element_slot(i).as_raw());
        }
    }
}
