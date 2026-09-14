use crate::{
    AccessorPair, FixedArray, FrameMeta, GcSlot, HeapRef, InternedString, Map, NoGc, Object,
    SlotFlags, SlotName, Smi, Stack, Symbol, Tagged, Value, VmError,
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
        // Canonical index strings ("0", "1", … up to 2^32−2) name the same
        // property as their numeric form (ES 6.1.7: ToString(i) is the
        // canonical key); non-canonical spellings ("01", "-0", "1e2") stay
        // ordinary names
        if let Some(i) = canonical_index(s.string().as_slice(nogc))
            && i <= u32::MAX as usize - 1
        {
            return Ok(Key::Element(i));
        }
        return Ok(Key::Name(SlotName::from(s.into_tagged())));
    }
    if let Some(s) = key.get_as::<Symbol>(nogc) {
        return Ok(Key::Name(SlotName::from(s.into_tagged())));
    }
    Err(VmError::Type)
}

/// Canonical array-index strings ("0", "1", "42"): digits only, no
/// leading zeros (ES 6.1.7). "01", "-0", "1e2" and any non-digit
/// spelling ("a", "+1") are not indices. The result is NOT
/// range-checked against 2^32−1 — callers decide whether a value that
/// large is still an array index.
pub fn canonical_index(bytes: &[u8]) -> Option<usize> {
    if bytes.is_empty() || bytes.len() > 10 {
        return None;
    }
    if bytes[0] == b'0' {
        return (bytes.len() == 1).then_some(0);
    }
    let mut n: usize = 0;
    for &b in bytes {
        if !b.is_ascii_digit() {
            return None;
        }
        n = n.checked_mul(10)?.checked_add((b - b'0') as usize)?;
    }
    Some(n)
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
    if let Some(obj) = receiver.as_heap_object(nogc)
        && let Some(v) = obj.as_ref().array_length(nogc, name)
    {
        return Ok(LoadOutcome::Value(v));
    }
    // string primitives expose `length` (UTF-16 code units) as an own
    // property without boxing (ES 5.4.3.1); index loads need a fresh
    // one-character string and stay unsupported here
    if let Some(s) = receiver.get_as::<crate::VMString>(nogc)
        && name
            .value()
            .get_as::<InternedString>(nogc)
            .is_some_and(|n| n.string().as_slice(nogc) == b"length")
    {
        let len = crate::natives::utf16_length(s.as_slice(nogc)) as i64;
        return Ok(LoadOutcome::Value(Smi::new(len).encode()));
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

/// ES 7.3.11 HasProperty (the `in` operator): walks the prototype chain
/// without invoking anything. Element indices consult the array elements
/// (including their backing-store holes); canonical index strings are
/// classified first so `"2" in o` and `2 in o` agree.
pub fn has_property<'a>(nogc: &'a NoGc<'a>, receiver: Value, name: SlotName) -> bool {
    let name = match classify_key(nogc, name.value()) {
        Ok(Key::Element(i)) => {
            // non-array receivers keep index keys as Smi-named
            // descriptors; canonicalize so the named walk finds them
            let smi_name = SlotName::from(Tagged::from_smi(Smi::new(i as i64)));
            if let Some(obj) = receiver.as_heap_object(nogc)
                && obj.as_ref().element_value(nogc, i).is_some()
            {
                return true;
            }
            smi_name
        }
        _ => name,
    };
    !matches!(receiver.lookup(nogc, name), Lookup::NotFound)
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

        lookup_in_parents(guard, self.prototype.inner(), name)
    }
}

pub fn lookup_in_parents<'a>(nogc: &'a NoGc<'a>, proto: Value, name: SlotName) -> Lookup<'a> {
    if proto == nogc.known().null.value() {
        return Lookup::NotFound;
    }
    if let Some(parents) = proto.get_as::<FixedArray>(nogc) {
        for i in 0..parents.len() {
            let result = parents.at(i).lookup(nogc, name);
            if !matches!(result, Lookup::NotFound) {
                return result;
            }
        }
        return Lookup::NotFound;
    }
    proto.lookup(nogc, name)
}

enum SuperStart<'a> {
    End,
    Object(Value),
    Parents(HeapRef<'a, FixedArray>),
}

/// The home object's [[Prototype]] slot, raw (null / object / FixedArray
/// of parents / unset). Extracted before ToPropertyKey so key coercion
/// cannot observe a different chain than the lookup uses (ES 15.4.2:
/// GetSuperBase happens first).
pub fn home_proto<'a>(nogc: &'a NoGc<'a>, value: Value) -> Option<Value> {
    let obj = value.as_heap_object(nogc)?;
    Some(obj.as_ref().header.map.heap_ref(nogc).prototype.inner())
}

fn super_start_from_proto<'a>(nogc: &'a NoGc<'a>, proto: Option<Value>) -> SuperStart<'a> {
    let Some(proto) = proto else {
        return SuperStart::End;
    };
    if proto == nogc.known().null.value() || !proto.is_strong_ptr() {
        return SuperStart::End;
    }
    if let Some(parents) = proto.get_as::<FixedArray>(nogc) {
        return SuperStart::Parents(parents);
    }
    SuperStart::Object(proto)
}

pub fn super_lookup<'a>(
    nogc: &'a NoGc<'a>,
    value: Value,
    name: SlotName,
) -> Result<LoadOutcome, VmError> {
    let proto = home_proto(nogc, value);
    super_lookup_from_proto(nogc, proto, name)
}

/// Super lookup with a pre-resolved prototype link (ES 15.4.2): the
/// prototype is read before the key is coerced, so user `toString` cannot
/// change which chain is searched.
pub fn super_lookup_from_proto<'a>(
    nogc: &'a NoGc<'a>,
    proto: Option<Value>,
    name: SlotName,
) -> Result<LoadOutcome, VmError> {
    match super_start_from_proto(nogc, proto) {
        // single parent: full load semantics
        SuperStart::Object(start) => load_outcome(nogc, start, name),
        SuperStart::Parents(parents) => {
            for i in 0..parents.len() {
                let parent = parents.at(i);
                if let Some(obj) = parent.as_heap_object(nogc)
                    && let Some(v) = obj.as_ref().array_length(nogc, name)
                {
                    return Ok(LoadOutcome::Value(v));
                }
                match parent.lookup(nogc, name) {
                    Lookup::Data { slot, .. } => {
                        return Ok(LoadOutcome::Value(slot.inner()));
                    }
                    Lookup::Accessor { pair, .. } => {
                        return Ok(LoadOutcome::Getter(pair.get.inner()));
                    }
                    Lookup::NotFound => continue,
                }
            }
            Ok(LoadOutcome::Value(nogc.known().undefined.value()))
        }
        SuperStart::End => Ok(LoadOutcome::Value(nogc.known().undefined.value())),
    }
}

/// The super constructor of the frame's running function: its own
/// [[Prototype]] (ES 10.2.2.2 GetSuperConstructor). `None` when the
/// prototype is absent or not a constructor. (Multiple prototypes are not
/// supported here: construction is not a property lookup.)
pub fn super_constructor<'a>(nogc: &'a NoGc<'a>, stack: &Stack, meta: &FrameMeta) -> Option<Value> {
    let callable = stack.callable_slot(meta).inner();
    let obj = callable.as_heap_object(nogc)?;
    let proto = obj.as_ref().header.map.heap_ref(nogc).prototype.inner();
    // must be a real constructor
    let proto_obj = proto.as_heap_object(nogc)?;
    if !proto_obj
        .as_ref()
        .header
        .map
        .heap_ref(nogc)
        .kind()
        .is_constructor()
    {
        return None;
    }
    Some(proto)
}

/// ES 7.3.26 PrivateElementFind restricted to fields: an own data
/// descriptor matching the private Symbol key (no prototype walk — private
/// elements live only on the instance itself).
pub fn private_find<'a>(nogc: &'a NoGc<'a>, obj: Value, key: Value) -> Option<&'a GcSlot> {
    let o = obj.as_heap_object(nogc)?;
    let map = o.as_ref().header.map.heap_ref(nogc);
    for d in map.descriptors() {
        if d.name() == SlotName::from_value(key) && !d.flags().is_accessor() {
            return Some(o.as_ref().slot(nogc, d.offset()));
        }
    }
    None
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
