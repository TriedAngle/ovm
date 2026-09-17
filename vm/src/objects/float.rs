use core::{alloc::Layout, cell::Cell};

use crate::{EdgeVisitable, Header, Heap, HeapObject, ObjectKind, Visitor};

#[repr(C)]
pub struct Float {
    pub header: Header,
    pub value: Cell<f64>,
}

impl HeapObject for Float {
    const KIND: ObjectKind = ObjectKind::Float;
    type Init<'a> = f64;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(heap, host, heap.known().float_map.as_tagged(heap));
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
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
    }
}
