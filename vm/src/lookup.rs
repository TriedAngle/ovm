use crate::{
    AccessorPair, GcSlot, Heap, HeapPtr, HeapRef, InternedString, Map, NoGc, Object,
    SlotDescriptor, SlotFlags, SlotKind, SlotName, Smi, Symbol, Tagged, Value, ValueRef, VmError,
};

pub enum Lookup<'a> {
    Data {
        holder: ValueRef<'a>,
        map_index: usize,
        holder_index: usize,
        slot: &'a GcSlot,
        flags: SlotFlags,
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

/// A runtime property key: a smi element index or a name (interned string / symbol).
pub enum Key {
    Element(usize),
    Name(SlotName),
}

pub fn classify_key<'a>(nogc: &'a NoGc<'a>, heap: &'a Heap, key: Value) -> Result<Key, VmError> {
    if let Some(smi) = Smi::decode(key) {
        if smi.value() >= 0 {
            return usize::try_from(smi.value())
                .map(Key::Element)
                .map_err(|_| VmError::OutOfBounds);
        }
        // negative indices are ordinary (numeric) property names
        return Ok(Key::Name(SlotName::from(Tagged::from_smi(smi))));
    }
    if let Some(s) = key.get_as::<InternedString>(nogc, heap.known().string_map) {
        return Ok(Key::Name(SlotName::from(s.into_tagged())));
    }
    if let Some(s) = key.get_as::<Symbol>(nogc, heap.known().symbol_map) {
        return Ok(Key::Name(SlotName::from(s.into_tagged())));
    }
    Err(VmError::Type)
}

/// Fast element read for array objects: `None` if the receiver is not an
/// array, the index is past the end, or the slot is a hole — the caller must
/// fall back to a named property lookup.
pub fn element_value<'a>(
    nogc: &'a NoGc<'a>,
    heap: &'a Heap,
    receiver: Value,
    i: usize,
) -> Option<Value> {
    let ValueRef::Object(obj) = receiver.value_ref(nogc) else {
        return None;
    };
    let obj = obj.as_ref();
    if !obj.is_array(nogc) || i >= obj.length() {
        return None;
    }
    let elements = obj.elements_array(nogc, heap)?;
    if i >= elements.len() {
        // `length` can exceed the backing store: those indices are holes
        return None;
    }
    let v = elements.at(i);
    if v == heap.known().void.value() {
        return None;
    }
    Some(v)
}

/// The result of a property load: a plain value or a getter that must be invoked.
pub enum LoadOutcome {
    Value(Value),
    Getter(Value),
}

pub fn load_outcome<'a>(
    nogc: &'a NoGc<'a>,
    heap: &'a Heap,
    receiver: Value,
    name: SlotName,
) -> Result<LoadOutcome, VmError> {
    let known = heap.known();
    // null/undefined have no [[Prototype]]: property access throws
    if receiver == known.null.value() || receiver == known.undefined.value() {
        return Err(VmError::Type);
    }
    match receiver.lookup(nogc, heap, name) {
        Lookup::Data { slot, .. } | Lookup::Const { slot, .. } => {
            Ok(LoadOutcome::Value(slot.inner()))
        }
        Lookup::Accessor { pair, .. } => {
            let getter = pair.get.inner();
            // no getter (void sentinel): the load yields undefined
            if getter == known.void.value() {
                Ok(LoadOutcome::Value(known.undefined.value()))
            } else {
                Ok(LoadOutcome::Getter(getter))
            }
        }
        Lookup::NotFound => Ok(LoadOutcome::Value(known.undefined.value())),
    }
}

impl Value {
    pub fn value_ref<'a>(&self, guard: &'a NoGc<'a>) -> ValueRef<'a> {
        value_ref(*self, guard)
    }

    pub fn lookup<'a>(&self, guard: &'a NoGc<'a>, heap: &Heap, name: SlotName) -> Lookup<'a> {
        lookup_value(*self, guard, heap, name)
    }
}

fn value_ref<'a>(v: Value, _guard: &'a NoGc<'a>) -> ValueRef<'a> {
    if let Some(smi) = Smi::decode(v) {
        return ValueRef::Smi(smi);
    }
    let ptr = HeapPtr::decode_strong(v).expect("slots hold only smi or strong values");
    ValueRef::Object(unsafe { HeapRef::from_ptr(ptr.cast()) })
}

fn lookup_value<'a>(
    receiver: Value,
    guard: &'a NoGc<'a>,
    heap: &Heap,
    name: SlotName,
) -> Lookup<'a> {
    let receiver = value_ref(receiver, guard);
    let map = match &receiver {
        ValueRef::Smi(_) => heap.known().smi_map.heap_ref(guard),
        ValueRef::Object(obj) => obj.as_ref().header.map.heap_ref(guard),
    };
    map.as_ref().lookup(guard, heap, receiver, name)
}

impl Map {
    pub fn lookup<'a>(
        &'a self,
        guard: &'a NoGc<'a>,
        heap: &Heap,
        receiver: ValueRef<'a>,
        name: SlotName,
    ) -> Lookup<'a> {
        for (index, d) in self.descriptors().iter().enumerate() {
            if !d.flags().is_parent() && d.name() == name {
                return match d.flags().kind() {
                    SlotKind::Value => {
                        // TODO: smis have no value slots not sure if we need to protect from this?
                        // the same should be true for floats too !
                        // find case where this happens or remove this
                        let ValueRef::Object(obj) = &receiver else {
                            panic!("value slot on the smi map")
                        };
                        Lookup::Data {
                            map_index: index,
                            holder_index: d.offset(),
                            slot: obj
                                .as_ref()
                                .slots
                                .heap_ref(guard)
                                .as_ref()
                                .element_slot(d.offset()),
                            holder: receiver,
                            flags: d.flags(),
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
                        pair: unsafe { HeapRef::from_ptr(d.value.get().cast().into()) },
                    },
                };
            }
        }

        for d in self.descriptors() {
            if d.flags().is_parent() {
                let result = d.value.inner().lookup(guard, heap, name);
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
        heap: &Heap,
        name: SlotName,
        parent: SlotName,
    ) -> Lookup<'a> {
        match self.find_parent(parent) {
            Some(d) => d.value.inner().lookup(guard, heap, name),
            None => Lookup::NotFound,
        }
    }

    pub fn find_parent(&self, name: SlotName) -> Option<&SlotDescriptor> {
        self.descriptors()
            .iter()
            .find(|d| d.flags().is_parent() && d.name() == name)
    }
}

impl Object {
    pub fn lookup<'a>(&'a self, guard: &'a NoGc<'a>, heap: &Heap, name: SlotName) -> Lookup<'a> {
        self.header.map.heap_ref(guard).as_ref().lookup(
            guard,
            heap,
            ValueRef::Object(HeapRef::from_ref(self)),
            name,
        )
    }

    pub fn lookup_parent<'a>(
        &'a self,
        guard: &'a NoGc<'a>,
        heap: &Heap,
        name: SlotName,
        parent: SlotName,
    ) -> Lookup<'a> {
        self.header
            .map
            .heap_ref(guard)
            .as_ref()
            .lookup_parent(guard, heap, name, parent)
    }
}
