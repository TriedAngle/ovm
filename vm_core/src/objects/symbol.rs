use core::alloc::Layout;

use crate::{
    EdgeVisitable, FixedByteArray, GcSlot, Handle, HandleScope, Header, Heap, HeapObject,
    ObjectKind, Visitor,
};

#[repr(C)]
pub struct Symbol {
    pub header: Header,
    pub backing: GcSlot<FixedByteArray>,
}

impl Symbol {
    pub fn new<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        description: &[u8],
    ) -> Handle<'s, Symbol> {
        let backing = heap.allocate_handle::<FixedByteArray>(description, scope);
        heap.allocate_handle::<Symbol>(backing, scope)
    }
}

impl HeapObject for Symbol {
    const KIND: ObjectKind = ObjectKind::Symbol;
    type Init<'a> = Handle<'a, FixedByteArray>;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        let host = self.tagged(heap);
        self.header
            .map
            .set(heap, host, heap.known().symbol_map.as_tagged(heap));
        self.backing.set(heap, host, config.as_tagged(heap));
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for Symbol {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.backing.as_raw());
    }
}
