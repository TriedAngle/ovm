use crate::{
    AccessorPair, FixedArray, GcSlot, HeapRef, InternedString, Map, NoGc, Object, SlotFlags,
    SlotName, Smi, Symbol, Tagged, VMString, Value, VmError,
};

pub enum Lookup<'a> {
    Data {
        holder: HeapRef<'a, Object>,
        map_index: usize,
        holder_index: usize,
        slot: &'a GcSlot,
        flags: SlotFlags,
    },
    Accessor {
        holder: HeapRef<'a, Object>,
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

pub fn classify_key<'a>(nogc: &'a NoGc<'a>, key: Value) -> Result<Key, VmError> {
    if let Some(smi) = Smi::decode(key) {
        // ES 6.1.7: an array index is 0 ≤ i < 2^32−1; anything else (incl.
        // 4294967295 itself) is an ordinary named property
        if smi.value() >= 0 && smi.value() < u32::MAX as i64 {
            return usize::try_from(smi.value())
                .map(Key::Element)
                .map_err(|_| VmError::OutOfBounds);
        }
        return Ok(Key::Name(SlotName::from(Tagged::from_smi(smi))));
    }
    if let Some(s) = key.get_as::<InternedString>(nogc) {
        return Ok(Key::Name(SlotName::from(s.into_tagged())));
    }
    if let Some(s) = key.get_as::<Symbol>(nogc) {
        return Ok(Key::Name(SlotName::from(s.into_tagged())));
    }
    Err(VmError::Type)
}

/// Fast element read for array objects: `None` if the receiver is not an
/// array, the index is past the end, or the slot is a hole — the caller must
/// fall back to a named property lookup.
pub fn element_value<'a>(nogc: &'a NoGc<'a>, receiver: Value, i: usize) -> Option<Value> {
    let Some(obj) = receiver.as_heap_object(nogc) else {
        return None;
    };
    let obj = obj.as_ref();
    if !obj.is_array(nogc) || i >= obj.length() {
        return None;
    }
    let elements = obj.elements_array(nogc)?;
    if i >= elements.len() {
        // `length` can exceed the backing store: those indices are holes
        return None;
    }
    let v = elements.at(i);
    if v == nogc.known().void.value() {
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
    receiver: Value,
    name: SlotName,
) -> Result<LoadOutcome, VmError> {
    let known = nogc.known();
    if receiver == known.null.value() || receiver == known.undefined.value() {
        return Err(VmError::Type);
    }
    // JSArray `length` is an internal slot, not a map descriptor
    if let Some(obj) = receiver.as_heap_object(nogc) {
        let obj = obj.as_ref();
        if obj.is_array(nogc)
            && let Some(s) = name.value().get_as::<VMString>(nogc)
            && s.as_slice(nogc) == b"length"
        {
            return Ok(LoadOutcome::Value(obj.length.inner()));
        }
    }
    match receiver.lookup(nogc, name) {
        Lookup::Data { slot, .. } => Ok(LoadOutcome::Value(slot.inner())),
        Lookup::Accessor { pair, .. } => {
            let getter = pair.get.inner();
            if getter == known.undefined.value() {
                Ok(LoadOutcome::Value(known.undefined.value()))
            } else {
                Ok(LoadOutcome::Getter(getter))
            }
        }
        Lookup::NotFound => Ok(LoadOutcome::Value(known.undefined.value())),
    }
}

impl Value {
    /// Property lookup on any value: non-objects (Smis) find nothing. The
    /// smi map has no descriptors and a null prototype, so this matches the
    /// old smi-map walk; primitives with named properties need boxing first.
    pub fn lookup<'a>(&self, guard: &'a NoGc<'a>, name: SlotName) -> Lookup<'a> {
        let Some(obj) = self.as_heap_object(guard) else {
            return Lookup::NotFound;
        };
        obj.as_ref().lookup(guard, name)
    }
}

impl Map {
    pub fn lookup<'a>(
        &'a self,
        guard: &'a NoGc<'a>,
        receiver: HeapRef<'a, Object>,
        name: SlotName,
    ) -> Lookup<'a> {
        for (index, d) in self.descriptors().iter().enumerate() {
            if d.name() == name {
                if d.flags().is_accessor() {
                    return Lookup::Accessor {
                        holder: receiver,
                        map_index: index,
                        pair: unsafe { HeapRef::from_ptr(d.value.get().cast().into()) },
                    };
                }
                return Lookup::Data {
                    map_index: index,
                    holder_index: d.offset(),
                    slot: receiver.as_ref().slot(guard, d.offset()),
                    holder: receiver,
                    flags: d.flags(),
                };
            }
        }

        // Prototype walk:
        // - null: no parents (null-proto root)
        // - object: single parent (JS [[Prototype]])
        // - FixedArray: multiple parents in priority order (Self parent*)
        let proto = self.prototype.inner();
        if proto == guard.known().null.value() {
            return Lookup::NotFound;
        }
        if let Some(parents) = proto.get_as::<FixedArray>(guard) {
            for i in 0..parents.len() {
                let result = parents.at(i).lookup(guard, name);
                if !matches!(result, Lookup::NotFound) {
                    return result;
                }
            }
            return Lookup::NotFound;
        }
        proto.lookup(guard, name)
    }
}

impl Object {
    pub fn lookup<'a>(&'a self, guard: &'a NoGc<'a>, name: SlotName) -> Lookup<'a> {
        self.header
            .map
            .heap_ref(guard)
            .as_ref()
            .lookup(guard, HeapRef::from_ref(self), name)
    }
}
