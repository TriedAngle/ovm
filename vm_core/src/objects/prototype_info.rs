use core::alloc::Layout;

use crate::{
    EdgeVisitable, GcSlot, Header, Heap, HeapObject, ObjectKind, OptionGcSlot, Smi, Tagged, Value,
    Visitor, WeakFixedArray,
};

#[repr(C)]
pub struct PrototypeInfo {
    pub header: Header,
    pub prototype_users: OptionGcSlot<WeakFixedArray>,
    pub registry_slot: GcSlot<Smi>,
}

pub struct PrototypeInfoInit {
    pub registry_slot: i64,
}

impl Default for PrototypeInfoInit {
    fn default() -> Self {
        Self {
            registry_slot: PrototypeInfo::UNREGISTERED,
        }
    }
}

impl PrototypeInfo {
    pub const UNREGISTERED: i64 = -1;

    pub fn prototype_users<'a>(&self, heap: &'a Heap) -> Option<Tagged<'a, WeakFixedArray>> {
        self.prototype_users.get(heap)
    }

    pub fn registry_slot(&self) -> i64 {
        self.registry_slot.to_smi_unchecked().value()
    }

    pub fn set_registry_slot(&self, heap: &Heap, host: Tagged<'_, Value>, slot: i64) {
        self.registry_slot.set(heap, host, Smi::new(slot));
    }
}

impl HeapObject for PrototypeInfo {
    const KIND: ObjectKind = ObjectKind::PrototypeInfo;
    type Init<'a> = PrototypeInfoInit;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        let host = self.tagged(heap);
        self.header
            .map
            .init(heap.known().prototype_info_map.as_tagged(heap));
        self.prototype_users.clear(heap);
        self.registry_slot
            .set(heap, host, Smi::new(config.registry_slot));
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for PrototypeInfo {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.prototype_users.as_raw());
        visitor.visit(self.registry_slot.as_raw());
    }
}
