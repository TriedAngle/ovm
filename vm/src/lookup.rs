use crate::proxy::Proxy;
use crate::{
    AccessorPair, Coercion, ContextState, Convert, DenseString, FixedArray, FrameMeta, GcSlot,
    Handle, HandleScope, Heap, HeapObject, Map, Object, PartialDescriptor, PropertyDescriptor,
    RuntimeContext, SlotFlags, SlotName, Smi, Stack, StringData, Symbol, Tagged, VM, Value,
    VmError,
};

pub enum Lookup<'a> {
    Data {
        holder: Tagged<'a, Object>,
        map_index: usize,
        holder_index: usize,
        slot: &'a GcSlot,
        flags: SlotFlags,
    },
    Accessor {
        holder: Tagged<'a, Object>,
        map_index: usize,
        pair: Tagged<'a, AccessorPair>,
    },
    NotFound,
}

/// A runtime property key: a smi element index or a name (interned string / symbol).
pub enum Key<'a> {
    Element(usize),
    Name(Tagged<'a, SlotName>),
}

impl Lookup<'_> {
    /// Classify a runtime property key: a canonical array index or a name.
    pub fn classify_key<'a>(heap: &'a Heap, key: Tagged<'a, Value>) -> Result<Key<'a>, VmError> {
        if let Some(smi) = Smi::decode(key.raw()) {
            // ES 6.1.7: an array index is 0 ≤ i < 2^32−1; anything else (incl.
            // 4294967295 itself) is an ordinary named property
            if smi.value() >= 0 && smi.value() < u32::MAX as i64 {
                return usize::try_from(smi.value())
                    .map(Key::Element)
                    .map_err(|_| VmError::OutOfBounds);
            }
            return Ok(Key::Name(Tagged::from(smi)));
        }
        if let Some(s) = key.get_as::<DenseString>() {
            // Canonical index strings ("0", "1", … up to 2^32−2) name the same
            // property as their numeric form (ES 6.1.7: ToString(i) is the
            // canonical key); non-canonical spellings ("01", "-0", "1e2") stay
            // ordinary names. String keys are interned upstream
            // (ToPropertyKey) — pointer identity identifies the name.
            if let Some(i) = canonical_index(s.as_ref().data(heap))
                && i < u32::MAX as usize
            {
                return Ok(Key::Element(i));
            }
            return Ok(Key::Name(s.into()));
        }
        if let Some(s) = key.get_as::<Symbol>() {
            return Ok(Key::Name(s.into()));
        }
        Err(VmError::Type)
    }
}

/// Canonical array-index strings ("0", "1", "42"): digits only, no
/// leading zeros (ES 6.1.7). "01", "-0", "1e2" and any non-digit
/// spelling ("a", "+1") are not indices. The result is NOT
/// range-checked against 2^32−1 — callers decide whether a value that
/// large is still an array index.
pub fn canonical_index(data: StringData<'_>) -> Option<usize> {
    if data.is_empty() || data.len() > 10 {
        return None;
    }
    if data.code_unit(0) == b'0' as u16 {
        return (data.len() == 1).then_some(0);
    }
    let mut n: usize = 0;
    for i in 0..data.len() {
        let c = data.code_unit(i);
        if !(b'0' as u16..=b'9' as u16).contains(&c) {
            return None;
        }
        n = n.checked_mul(10)?.checked_add((c - b'0' as u16) as usize)?;
    }
    Some(n)
}

/// The result of a property load: a plain value or a getter that must be invoked.
pub enum LoadOutcome<'a> {
    Value(Tagged<'a, Value>),
    Getter(Tagged<'a, Value>),
}

/// [[Get]] lookup starting at `holder`. Getters found here must be
/// invoked by the *caller* with the intended `this` — `o.x` passes
/// `holder = o`; proxy forwarding passes the target as the holder and
/// the proxy as the getter receiver.
pub fn load_outcome_on<'a>(
    heap: &'a Heap,
    holder: Tagged<'a, Value>,
    name: Tagged<'a, SlotName>,
) -> Result<LoadOutcome<'a>, VmError> {
    let known = heap.known();
    if holder == known.null.as_tagged(heap) || holder == known.undefined.as_tagged(heap) {
        return Err(VmError::Type);
    }
    if let Some(obj) = holder.as_heap_object()
        && let Some(v) = obj.as_ref().array_length(heap, name)
    {
        return Ok(LoadOutcome::Value(v));
    }
    // string primitives expose `length` (UTF-16 code units) as an own
    // property without boxing (ES 5.4.3.1); index loads need a fresh
    // one-character string and stay unsupported here
    if let Some(s) = holder.get_as::<DenseString>()
        && name
            .erase()
            .get_as::<DenseString>()
            .is_some_and(|n| n.as_ref().data(heap).matches_ascii(b"length"))
    {
        let len = s.len() as i64;
        return Ok(LoadOutcome::Value(Smi::new(len).into_tagged()));
    }
    match holder.lookup(heap, name) {
        Lookup::Data { slot, .. } => Ok(LoadOutcome::Value(slot.get(heap))),
        Lookup::Accessor { pair, .. } => {
            let getter = pair.get.get(heap);
            if getter == known.undefined.as_tagged(heap) {
                Ok(LoadOutcome::Value(known.undefined.as_tagged(heap).erase()))
            } else {
                Ok(LoadOutcome::Getter(getter))
            }
        }
        Lookup::NotFound => Ok(LoadOutcome::Value(known.undefined.as_tagged(heap).erase())),
    }
}

impl Lookup<'_> {
    pub fn load_outcome<'a>(
        heap: &'a Heap,
        receiver: Tagged<'a, Value>,
        name: Tagged<'a, SlotName>,
    ) -> Result<LoadOutcome<'a>, VmError> {
        load_outcome_on(heap, receiver, name)
    }

    /// [[Get]] for a keyed load: an element key consults dense array
    /// elements first (falling back to its canonical Smi name), a name
    /// takes the ordinary path. Getters are returned for the caller to
    /// invoke.
    pub fn load_outcome_keyed<'a>(
        heap: &'a Heap,
        receiver: Tagged<'a, Value>,
        key: Tagged<'a, SlotName>,
    ) -> Result<LoadOutcome<'a>, VmError> {
        match Lookup::classify_key(heap, key.erase())? {
            Key::Element(i) => match receiver
                .as_heap_object()
                .and_then(|obj| obj.as_ref().element_value(heap, i))
            {
                Some(v) => Ok(LoadOutcome::Value(v)),
                // past the end, a hole, or a non-array receiver: ordinary lookup
                None => Lookup::load_outcome(heap, receiver, Tagged::from(Smi::new(i as i64))),
            },
            Key::Name(name) => Lookup::load_outcome(heap, receiver, name),
        }
    }
}

impl DenseString {
    /// The one-unit string an index load on a string primitive yields
    /// (ES 5.4.3.1): `"ab"[1]` is "b". `None` for out-of-range keys and
    /// non-string receivers, which fall through to the ordinary property
    /// path.
    pub fn index_element(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<'_, Value>,
        key: Handle<'_, SlotName>,
    ) -> Option<Value> {
        let Ok(Key::Element(i)) = Lookup::classify_key(heap, key.as_tagged(heap).erase()) else {
            return None;
        };
        // Safety: fresh rooted-slot word.
        let receiver = receiver.as_tagged(heap).raw();
        DenseString::char_at(heap, scope, receiver, i).map(|s| s.as_tagged(heap).raw())
    }
}

/// Ordinary [[GetOwnProperty]] as a full descriptor (ES 10.1.5): dense
/// array elements, the JSArray `length` slot, and map descriptor rows.
/// The single reader both `Object.getOwnPropertyDescriptor` and the
/// proxy invariant checks build on. The returned descriptor is rooted in
/// `scope`, so it survives GC safepoints.
pub fn ordinary_own_descriptor<'a, 's>(
    heap: &'a Heap,
    scope: &'s HandleScope<'_>,
    obj: Tagged<'a, Value>,
    key: Tagged<'a, Value>,
) -> Option<PropertyDescriptor<'s>> {
    if let Ok(Key::Element(i)) = Lookup::classify_key(heap, key)
        && let Some(o) = obj.as_heap_object()
        && let Some(v) = o.as_ref().element_value(heap, i)
    {
        return Some(PropertyDescriptor::Data {
            value: scope.handle(v),
            writable: true,
            enumerable: true,
            configurable: true,
        });
    }
    let name = key.as_name();
    let o = obj.as_heap_object()?;
    if let Some(v) = o.as_ref().array_length(heap, name) {
        return Some(PropertyDescriptor::Data {
            value: scope.handle(v),
            writable: true,
            enumerable: false,
            configurable: false,
        });
    }
    let map = o.as_ref().map_ref(heap);
    for d in map.descriptors() {
        if !d.name(heap).ptr_eq(name) {
            continue;
        }
        if d.flags().is_accessor() {
            let pair = d
                .value
                .get(heap)
                .get_as::<AccessorPair>()
                .expect("accessor descriptor holds a pair");
            return Some(PropertyDescriptor::Accessor {
                get: scope.handle(pair.get.get(heap)),
                set: scope.handle(pair.set.get(heap)),
                enumerable: d.flags().is_enumerable(),
                configurable: d.flags().is_configurable(),
            });
        }
        let slot = o.as_ref().slot(heap, d.offset());
        return Some(PropertyDescriptor::Data {
            value: scope.handle(slot.get(heap)),
            writable: d.flags().is_writable(),
            enumerable: d.flags().is_enumerable(),
            configurable: d.flags().is_configurable(),
        });
    }
    None
}

impl<'a, T> Tagged<'a, T> {
    pub fn lookup(self, heap: &'a Heap, name: Tagged<'a, SlotName>) -> Lookup<'a> {
        let Some(obj) = self.erase().as_heap_object() else {
            return Lookup::NotFound;
        };
        obj.as_ref().lookup(heap, name)
    }
}

impl<'s, T> Handle<'s, T> {
    pub fn lookup<'a>(self, heap: &'a Heap, name: Tagged<'a, SlotName>) -> Lookup<'a>
    where
        T: 'a,
    {
        self.as_tagged(heap).lookup(heap, name)
    }
}

/// ES 7.3.11 HasProperty (the `in` operator): walks the prototype chain
/// without invoking anything. Element indices consult the array elements
/// (including their backing-store holes); canonical index strings are
/// classified first so `"2" in o` and `2 in o` agree. Arrays anywhere in
/// the chain own `"length"` through their internal slot (ES 10.4.2.1),
/// which the descriptor walk cannot see.
pub fn has_property<'a>(
    heap: &'a Heap,
    receiver: Tagged<'a, Value>,
    name: Tagged<'a, SlotName>,
) -> bool {
    let name = match Lookup::classify_key(heap, name.erase()) {
        Ok(Key::Element(i)) => {
            // non-array receivers keep index keys as Smi-named
            // descriptors; canonicalize so the named walk finds them
            let smi_name: Tagged<'a, SlotName> = Tagged::from(Smi::new(i as i64));
            if let Some(obj) = receiver.as_heap_object()
                && obj.as_ref().element_value(heap, i).is_some()
            {
                return true;
            }
            smi_name
        }
        _ => name,
    };
    if !matches!(receiver.lookup(heap, name), Lookup::NotFound) {
        return true;
    }
    // "length" may live in an array's internal slot at any chain level
    if name
        .erase()
        .get_as::<DenseString>()
        .is_some_and(|n| n.as_ref().data(heap).matches_ascii(b"length"))
    {
        return array_length_in_chain(heap, receiver);
    }
    false
}

/// Whether any object in `receiver`'s prototype chain (receiver
/// included) is an array — its `length` is an own non-configurable
/// property invisible to the descriptor walk.
fn array_length_in_chain<'a>(heap: &'a Heap, receiver: Tagged<'a, Value>) -> bool {
    let mut current = receiver;
    loop {
        let Some(obj) = current.as_heap_object() else {
            return false;
        };
        if obj.as_ref().is_array(heap) {
            return true;
        }
        let proto = obj.as_ref().header.map.get(heap).prototype.get(heap);
        if proto == heap.known().null.as_tagged(heap) || !proto.is_strong_ptr() {
            return false;
        }
        if let Some(parents) = proto.get_as::<FixedArray>() {
            for i in 0..parents.len() {
                if array_length_in_chain(heap, parents.at(heap, i)) {
                    return true;
                }
            }
            return false;
        }
        current = proto;
    }
}

impl Map {
    pub fn lookup<'a>(
        &'a self,
        heap: &'a Heap,
        receiver: Tagged<'a, Object>,
        name: Tagged<'a, SlotName>,
    ) -> Lookup<'a> {
        for (index, d) in self.descriptors().iter().enumerate() {
            if d.name(heap).ptr_eq(name) {
                if d.flags().is_accessor() {
                    return Lookup::Accessor {
                        holder: receiver,
                        map_index: index,
                        pair: d
                            .value
                            .get(heap)
                            .get_as::<AccessorPair>()
                            .expect("accessor descriptor holds an AccessorPair"),
                    };
                }
                return Lookup::Data {
                    map_index: index,
                    holder_index: d.offset(),
                    slot: receiver.as_ref().slot(heap, d.offset()),
                    holder: receiver,
                    flags: d.flags(),
                };
            }
        }

        lookup_in_parents(heap, self.prototype.get(heap), name)
    }
}

pub fn lookup_in_parents<'a>(
    heap: &'a Heap,
    proto: Tagged<'a, Value>,
    name: Tagged<'a, SlotName>,
) -> Lookup<'a> {
    if proto == heap.known().null.as_tagged(heap) {
        return Lookup::NotFound;
    }
    if let Some(parents) = proto.get_as::<FixedArray>() {
        for i in 0..parents.len() {
            let result = parents.at(heap, i).lookup(heap, name);
            if !matches!(result, Lookup::NotFound) {
                return result;
            }
        }
        return Lookup::NotFound;
    }
    proto.lookup(heap, name)
}

enum SuperStart<'a> {
    End,
    Object(Tagged<'a, Value>),
    Parents(Tagged<'a, FixedArray>),
}

/// The home object's [[Prototype]] slot, raw (null / object / FixedArray
/// of parents / unset). Extracted before ToPropertyKey so key coercion
/// cannot observe a different chain than the lookup uses (ES 15.4.2:
/// GetSuperBase happens first).
pub fn home_proto<'a>(heap: &'a Heap, value: Tagged<'a, Value>) -> Option<Tagged<'a, Value>> {
    let obj = value.as_heap_object()?;
    Some(obj.as_ref().header.map.get(heap).prototype.get(heap))
}

fn super_start_from_proto<'a>(heap: &'a Heap, proto: Option<Tagged<'a, Value>>) -> SuperStart<'a> {
    let Some(proto) = proto else {
        return SuperStart::End;
    };
    if proto == heap.known().null.as_tagged(heap) || !proto.is_strong_ptr() {
        return SuperStart::End;
    }
    if let Some(parents) = proto.get_as::<FixedArray>() {
        return SuperStart::Parents(parents);
    }
    SuperStart::Object(proto)
}

pub fn super_lookup<'a>(
    heap: &'a Heap,
    value: Tagged<'a, Value>,
    name: Tagged<'a, SlotName>,
) -> Result<LoadOutcome<'a>, VmError> {
    let proto = home_proto(heap, value);
    super_lookup_from_proto(heap, proto, name)
}

/// Super lookup with a pre-resolved prototype link (ES 15.4.2): the
/// prototype is read before the key is coerced, so user `toString` cannot
/// change which chain is searched.
pub fn super_lookup_from_proto<'a>(
    heap: &'a Heap,
    proto: Option<Tagged<'a, Value>>,
    name: Tagged<'a, SlotName>,
) -> Result<LoadOutcome<'a>, VmError> {
    match super_start_from_proto(heap, proto) {
        // single parent: full load semantics
        SuperStart::Object(start) => Lookup::load_outcome(heap, start, name),
        SuperStart::Parents(parents) => {
            for i in 0..parents.len() {
                let parent = parents.at(heap, i);
                if let Some(obj) = parent.as_heap_object()
                    && let Some(v) = obj.as_ref().array_length(heap, name)
                {
                    return Ok(LoadOutcome::Value(v));
                }
                match parent.lookup(heap, name) {
                    Lookup::Data { slot, .. } => {
                        return Ok(LoadOutcome::Value(slot.get(heap)));
                    }
                    Lookup::Accessor { pair, .. } => {
                        return Ok(LoadOutcome::Getter(pair.get.get(heap)));
                    }
                    Lookup::NotFound => continue,
                }
            }
            Ok(LoadOutcome::Value(
                heap.known().undefined.as_tagged(heap).erase(),
            ))
        }
        SuperStart::End => Ok(LoadOutcome::Value(
            heap.known().undefined.as_tagged(heap).erase(),
        )),
    }
}

/// The super constructor of the frame's running function: its own
/// [[Prototype]] (ES 10.2.2.2 GetSuperConstructor). `None` when the
/// prototype is absent or not a constructor. (Multiple prototypes are not
/// supported here: construction is not a property lookup.)
pub fn super_constructor<'a>(
    heap: &'a Heap,
    stack: &Stack,
    meta: &FrameMeta,
) -> Option<Tagged<'a, Value>> {
    let callable = stack.callable_slot(meta).read(heap);
    let obj = callable.as_heap_object()?;
    let proto = obj.as_ref().header.map.get(heap).prototype.get(heap);
    // must be a real constructor
    let proto_obj = proto.as_heap_object()?;
    if !proto_obj
        .as_ref()
        .header
        .map
        .get(heap)
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
pub fn private_find<'a>(
    heap: &'a Heap,
    obj: Tagged<'a, Value>,
    key: Tagged<'a, Value>,
) -> Option<&'a GcSlot> {
    let o = obj.as_heap_object()?;
    let map = o.as_ref().header.map.get(heap);
    for d in map.descriptors() {
        if d.name(heap).ptr_eq(key.as_name()) && !d.flags().is_accessor() {
            return Some(o.as_ref().slot(heap, d.offset()));
        }
    }
    None
}

impl Object {
    pub fn lookup<'a>(&'a self, heap: &'a Heap, name: Tagged<'a, SlotName>) -> Lookup<'a> {
        // Safety: `self` is borrowed for `'a` and the live heap borrow
        // proves no collection can run, so the object word is anchored.
        let receiver: Tagged<'a, Object> = unsafe { Tagged::from_value_unchecked(self.erase()) };
        self.header
            .map
            .get(heap)
            .as_ref()
            .lookup(heap, receiver, name)
    }
}

impl Lookup<'_> {
    /// Same, with the lookup start (`holder`) split from the getter
    /// receiver — the proxy forward shape: lookup on the target,
    /// `this` = the proxy.
    pub fn get_property_on<'a>(
        vm: &'a VM,
        heap: &'a mut Heap,
        state: &'a ContextState,
        holder: Handle<'_, Value>,
        receiver: Handle<'_, Value>,
        name: Handle<'_, Value>,
    ) -> Result<Coercion<'a>, VmError> {
        let cond_3 = Proxy::is_proxy(heap, holder.as_tagged(heap));
        if cond_3 {
            return Proxy::get(vm, heap, state, holder, receiver, name);
        }
        RuntimeContext::new(vm, heap, state).handle_scope(
            |vm, heap, state, scope| -> Result<Coercion<'a>, VmError> {
                let exception = heap.known().exception.as_tagged(heap).raw();
                let loaded = {
                    let heap_ref: &Heap = heap;
                    load_outcome_on(
                        heap_ref,
                        holder.as_tagged(heap_ref),
                        name.as_tagged(heap_ref).as_name(),
                    )?
                };
                match loaded {
                    LoadOutcome::Value(v) => {
                        let v = scope.handle(v);
                        Ok(Coercion::Value(v.as_tagged(heap)))
                    }
                    LoadOutcome::Getter(getter) => {
                        let getter = scope.handle(getter);
                        let args = scope.stage(&[receiver.as_tagged(&*heap).erase()]);
                        let result = scope.handle(RuntimeContext::call(
                            vm, &mut *heap, state, getter, args, None,
                        )?);
                        if result.as_tagged(heap).raw() == exception {
                            Ok(Coercion::Threw)
                        } else {
                            Ok(Coercion::Value(result.as_tagged(heap)))
                        }
                    }
                }
            },
        )
    }

    pub fn to_property_descriptor<'s>(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        scope: &'s HandleScope<'_>,
        attrs: Handle<'_, Value>,
    ) -> Result<Option<PartialDescriptor<'s>>, VmError> {
        let cond_4 = Convert::is_primitive(heap, attrs.as_tagged(heap));
        if cond_4 {
            return Err(VmError::Type);
        }
        let attrs = scope.handle(attrs.as_tagged(heap));
        let names: [Handle<'_, Value>; 6] = {
            let s = heap.known().strings;
            [
                s.value,
                s.get,
                s.set,
                s.writable,
                s.enumerable,
                s.configurable,
            ]
            .map(|n| scope.handle(n.as_tagged(heap).erase()))
        };
        let mut reads: Vec<Handle<'_, Value>> = Vec::new();
        for name in names {
            match Lookup::get_property_on(vm, heap, state, attrs, attrs, name)? {
                Coercion::Threw => return Ok(None),
                // Safety: fresh word from the call, no allocation since.
                Coercion::Value(v) => reads.push(scope.handle(v)),
            }
        }
        let (present, truthy) = {
            let undef = heap.known().undefined.as_tagged(heap).erase();
            let present = [
                !reads[0].as_tagged(heap).ptr_eq(undef),
                !reads[1].as_tagged(heap).ptr_eq(undef),
                !reads[2].as_tagged(heap).ptr_eq(undef),
                !reads[3].as_tagged(heap).ptr_eq(undef),
                !reads[4].as_tagged(heap).ptr_eq(undef),
                !reads[5].as_tagged(heap).ptr_eq(undef),
            ];
            let truthy = [
                Convert::is_truthy(heap, reads[3].as_tagged(heap)),
                Convert::is_truthy(heap, reads[4].as_tagged(heap)),
                Convert::is_truthy(heap, reads[5].as_tagged(heap)),
            ];
            (present, truthy)
        };
        let value = present[0].then_some(reads[0]);
        let get = present[1].then_some(reads[1]);
        let set = present[2].then_some(reads[2]);
        // accessor halves must be callable or undefined
        if get.is_some() || set.is_some() {
            for half in [get, set] {
                if let Some(h) = half
                    && !Object::is_callable(heap, h.as_tagged(heap))
                {
                    return Err(VmError::Type);
                }
            }
        }
        Ok(Some(PartialDescriptor {
            value,
            get,
            set,
            writable: present[3].then_some(truthy[0]),
            enumerable: present[4].then_some(truthy[1]),
            configurable: present[5].then_some(truthy[2]),
        }))
    }
}
