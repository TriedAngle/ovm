use core::alloc::Layout;

use crate::{
    EdgeVisitable, FixedByteArray, GcSlot, Handle, HandleScope, Header, Heap, HeapObject, NoGc,
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

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().symbol_map.as_tagged());
        self.backing.set(nogc, host, *config);
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
