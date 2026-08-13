use crate::{
    AccessorPair, GcSlot, HeapPtr, HeapRef, LocalHeap, Map, NoGc, SlotDescriptor, SlotKind,
    SlotName, SlotsObject, Smi, ValueRef,
};

pub enum Lookup<'a> {
    Data {
        holder: ValueRef<'a>,
        map_index: usize,
        holder_index: usize,
        slot: &'a GcSlot,
    },
    Const {
        holder: ValueRef<'a>,
        map_index: usize,
        slot: &'a GcSlot,
    },
    Accessor {
        holder: ValueRef<'a>,
        map_index: usize,
        pair: HeapRef<'a, AccessorPair>,
    },
    NotFound,
}

impl GcSlot {
    pub fn value_ref<'a>(&self, _guard: &'a NoGc<'a>) -> ValueRef<'a> {
        let v = self.inner();
        if let Some(smi) = Smi::decode(v) {
            return ValueRef::Smi(smi);
        }
        let ptr = HeapPtr::<SlotsObject>::decode_strong(v)
            .expect("value-flow slots hold only smi or strong values");
        ValueRef::Object(unsafe { HeapRef::from_ptr(ptr) })
    }

    pub fn lookup<'a>(
        &self,
        guard: &'a NoGc<'a>,
        heap: &impl LocalHeap,
        name: SlotName,
    ) -> Lookup<'a> {
        let receiver = self.value_ref(guard);
        let map = match &receiver {
            ValueRef::Smi(_) => heap.known().smi_map.heap_ref(guard),
            ValueRef::Object(obj) => guard.get(&obj.as_ref().header.map),
        };
        map.as_ref().lookup(guard, heap, receiver, name)
    }
}

impl Map {
    pub fn lookup<'a>(
        &'a self,
        guard: &'a NoGc<'a>,
        heap: &impl LocalHeap,
        receiver: ValueRef<'a>,
        name: SlotName,
    ) -> Lookup<'a> {
        for (index, d) in self.descriptors().iter().enumerate() {
            if !d.flags().is_parent() && d.name() == name {
                return match d.flags().kind() {
                    SlotKind::Value => {
                        // smis have no slots
                        // not sure if we actually need to protect from this?
                        let ValueRef::Object(obj) = &receiver else {
                            panic!("value slot on the smi map")
                        };
                        Lookup::Data {
                            map_index: index,
                            holder_index: d.offset(),
                            slot: obj.as_ref().slot(d.offset()),
                            holder: receiver,
                        }
                    }
                    SlotKind::Const => Lookup::Const {
                        map_index: index,
                        slot: &d.value,
                        holder: receiver,
                    },
                    SlotKind::Accessor => Lookup::Accessor {
                        holder: receiver,
                        map_index: index,
                        pair: unsafe { guard.get_unchecked(d.value.get().cast_unchecked()) },
                    },
                };
            }
        }

        for d in self.descriptors() {
            if d.flags().is_parent() {
                let result = d.value.lookup(guard, heap, name);
                if !matches!(result, Lookup::NotFound) {
                    return result;
                }
            }
        }

        Lookup::NotFound
    }

    pub fn lookup_parent<'a>(
        &'a self,
        guard: &'a NoGc<'a>,
        heap: &impl LocalHeap,
        name: SlotName,
        parent: SlotName,
    ) -> Lookup<'a> {
        match self.find_parent(parent) {
            Some(d) => d.value.lookup(guard, heap, name),
            None => Lookup::NotFound,
        }
    }

    pub fn find_parent(&self, name: SlotName) -> Option<&SlotDescriptor> {
        self.descriptors()
            .iter()
            .find(|d| d.flags().is_parent() && d.name() == name)
    }
}

impl SlotsObject {
    pub fn lookup<'a>(
        &'a self,
        guard: &'a NoGc<'a>,
        heap: &impl LocalHeap,
        name: SlotName,
    ) -> Lookup<'a> {
        guard.get(&self.header.map).as_ref().lookup(
            guard,
            heap,
            ValueRef::Object(HeapRef::from_ref(self)),
            name,
        )
    }

    pub fn lookup_parent<'a>(
        &'a self,
        guard: &'a NoGc<'a>,
        heap: &impl LocalHeap,
        name: SlotName,
        parent: SlotName,
    ) -> Lookup<'a> {
        guard
            .get(&self.header.map)
            .as_ref()
            .lookup_parent(guard, heap, name, parent)
    }
}
