use core::alloc::Layout;

use crate::{
    EdgeVisitable, Header, Heap, HeapObject, MaybeWeak, MaybeWeakGcSlot, ObjectKind, Tagged, Value,
    Visitor,
};

#[repr(C)]
pub struct Cell {
    pub header: Header,
    pub maybe_value: MaybeWeakGcSlot,
}

pub struct CellInit {
    pub value: Value,
}

impl Cell {
    pub fn is_valid(&self, _heap: &Heap) -> bool {
        !self.maybe_value.is_cleared()
    }

    pub fn is_cleared(&self) -> bool {
        self.maybe_value.is_cleared()
    }

    pub fn clear(&self) {
        self.maybe_value
            .as_raw()
            .store_raw(Value::CLEARED.to_bits());
    }

    pub fn payload<'a>(&self, heap: &'a Heap) -> Tagged<'a, MaybeWeak<Value>> {
        self.maybe_value.get(heap)
    }
}

impl HeapObject for Cell {
    const KIND: ObjectKind = ObjectKind::Cell;
    type Init<'a> = CellInit;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        let host = self.tagged(heap);
        self.header.map.init(heap.known().cell_map.as_tagged(heap));
        let word = unsafe { Tagged::from_maybe_weak_unchecked(config.value) };
        self.maybe_value.set(heap, host, word);
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for Cell {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.maybe_value.as_raw());
    }
}
