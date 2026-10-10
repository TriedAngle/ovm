use crate::proxy::Proxy;
use crate::reanchor;
use crate::{
    AccessorPair, Coercion, DETAILS_ACCESSOR, DETAILS_CONFIGURABLE, DETAILS_ENUMERABLE,
    DETAILS_WRITABLE, DenseString, FixedArray, Float, GcSlot, Handle, HandleScope, Heap, HostCtx,
    Map, NumberDictionary, Object, PartialDescriptor, PropertyDescriptor, SlotFlags, SlotName, Smi,
    StringData, StringOwn, Symbol, Tagged, ThreadState, VM, Value, VmError,
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

/// The result of a property load: a plain value or a getter that must be invoked.
pub enum LoadOutcome<'a> {
    Value(Tagged<'a, Value>),
    Getter(Tagged<'a, Value>),
}

impl<'a> Tagged<'a, Value> {
    pub fn classify_key(self, heap: &'a Heap) -> Result<Key<'a>, VmError> {
        if let Some(v) = self.to_i64() {
            if (0..u32::MAX as i64).contains(&v) {
                return usize::try_from(v)
                    .map(Key::Element)
                    .map_err(|_| VmError::OutOfBounds);
            }
            return Ok(Key::Name(Tagged::from(Smi::new(v))));
        }
        if let Some(s) = self.get_as::<DenseString>(heap) {
            if let Some(i) = s.as_ref().data(heap).canonical_index()
                && i < u32::MAX as usize
            {
                return Ok(Key::Element(i));
            }
            return Ok(Key::Name(s.into()));
        }
        if let Some(s) = self.get_as::<Symbol>(heap) {
            return Ok(Key::Name(s.into()));
        }
        Err(VmError::Type)
    }

    pub fn load_outcome_on(
        self,
        heap: &'a Heap,
        name: Tagged<'a, SlotName>,
    ) -> Result<LoadOutcome<'a>, VmError> {
        let known = heap.known();
        if self == known.null.as_tagged(heap) || self == known.undefined.as_tagged(heap) {
            return Err(VmError::Type);
        }
        if let Some(obj) = self.as_heap_object()
            && let Some(v) = obj.as_ref().array_length(heap, name)
        {
            return Ok(LoadOutcome::Value(v));
        }
        if let Some(s) = DenseString::from_receiver(heap, self)
            && let Some(StringOwn::Length(len)) = s.as_ref().own_key(heap, name)
        {
            return Ok(LoadOutcome::Value(Smi::new(len as i64).into_tagged()));
        }
        let holder = if self.to_i64().is_some() || self.get_as::<Float>(heap).is_some() {
            known.number_prototype.as_tagged(heap).erase()
        } else if self.get_as::<DenseString>(heap).is_some() {
            known.string_prototype.as_tagged(heap).erase()
        } else if self == known.true_object.as_tagged(heap)
            || self == known.false_object.as_tagged(heap)
        {
            known.boolean_prototype.as_tagged(heap).erase()
        } else {
            self
        };
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

    pub fn load_outcome(
        self,
        heap: &'a Heap,
        name: Tagged<'a, SlotName>,
    ) -> Result<LoadOutcome<'a>, VmError> {
        self.load_outcome_on(heap, name)
    }

    pub fn chain_holds_index_name(self, heap: &Heap, i: usize) -> bool {
        self.chain_holds_name(heap, ChainQuery::Index(i))
    }

    pub fn chain_holds_any_index_name(self, heap: &Heap) -> bool {
        self.chain_holds_name(heap, ChainQuery::AnyIndex)
    }

    fn chain_holds_name(self, heap: &Heap, query: ChainQuery) -> bool {
        let matches = |name: Tagged<'_, SlotName>| match query {
            ChainQuery::Index(i) => {
                let smi_name: Tagged<'_, SlotName> = Tagged::from(Smi::new(i as i64));
                name.ptr_eq(smi_name)
            }
            ChainQuery::AnyIndex => name.to_i64().is_some(),
        };
        let mut hop = self;
        while let Some(obj) = hop.as_heap_object() {
            let map = obj.as_ref().map(heap);
            let proto = map.as_ref().prototype.get(heap);
            if proto.get_as::<FixedArray>(heap).is_some() {
                return true;
            }
            let known = heap.known();
            if !proto.is_strong_ptr()
                || proto == known.null.as_tagged(heap).erase()
                || proto == known.the_hole.as_tagged(heap).erase()
                || proto == known.undefined.as_tagged(heap).erase()
            {
                return false;
            }
            let Some(next) = proto.get_as::<Object>(heap) else {
                return false;
            };
            if next.as_ref().map(heap).kind().is_proxy() {
                return true;
            }
            let next_map = next.as_ref().map(heap);
            if next_map
                .as_ref()
                .descriptors()
                .iter()
                .any(|d| matches(d.name(heap)))
            {
                return true;
            }
            // dictionary hops: accessor or read-only entries intercept
            // stores at their indices
            if let Some(dict) = next.as_ref().element_dictionary(heap) {
                let intercepts = |details: i64| {
                    details & (DETAILS_ACCESSOR | DETAILS_WRITABLE) != DETAILS_WRITABLE
                };
                let dirty = match query {
                    ChainQuery::Index(i) => dict
                        .as_ref()
                        .find(heap, i)
                        .map(|e| intercepts(dict.as_ref().details_at(heap, e)))
                        .unwrap_or(false),
                    ChainQuery::AnyIndex => {
                        let mut any = false;
                        dict.as_ref().for_each_entry(heap, |_, _, details| {
                            if intercepts(details) {
                                any = true;
                            }
                        });
                        any
                    }
                };
                if dirty {
                    return true;
                }
            }
            hop = next.erase();
        }
        false
    }

    pub fn load_outcome_keyed(
        self,
        heap: &'a Heap,
        key: Tagged<'a, SlotName>,
    ) -> Result<LoadOutcome<'a>, VmError> {
        match key.erase().classify_key(heap)? {
            Key::Element(i) => {
                let smi_name: Tagged<'a, SlotName> = Tagged::from(Smi::new(i as i64));
                if let Some(v) = self
                    .as_heap_object()
                    .and_then(|obj| obj.as_ref().element_value(heap, i))
                {
                    return Ok(LoadOutcome::Value(v));
                }
                // accessor entries in a sparse self are own getters
                if let Some(outcome) = self
                    .as_heap_object()
                    .and_then(|obj| obj.as_ref().element_dictionary(heap))
                    .and_then(|dict| dictionary_accessor_outcome(heap, dict, i))
                {
                    return Ok(outcome);
                }
                let mut hop = self;
                loop {
                    let Some(obj) = hop.as_heap_object() else {
                        break;
                    };
                    let map = obj.as_ref().map(heap);
                    let proto = map.as_ref().prototype.get(heap);
                    // Self-style multi-parent chains and exotic hops take
                    // the ordinary named walk
                    if proto.get_as::<FixedArray>(heap).is_some() {
                        break;
                    }
                    let known = heap.known();
                    if !proto.is_strong_ptr()
                        || proto == known.null.as_tagged(heap).erase()
                        || proto == known.the_hole.as_tagged(heap).erase()
                        || proto == known.undefined.as_tagged(heap).erase()
                    {
                        break;
                    }
                    let Some(next) = proto.get_as::<Object>(heap) else {
                        break;
                    };
                    if next.as_ref().map(heap).kind().is_proxy() {
                        break;
                    }
                    if let Some(v) = next.as_ref().element_value(heap, i) {
                        return Ok(LoadOutcome::Value(v));
                    }
                    if let Some(outcome) = next
                        .as_ref()
                        .element_dictionary(heap)
                        .and_then(|dict| dictionary_accessor_outcome(heap, dict, i))
                    {
                        return Ok(outcome);
                    }
                    let next_map = next.as_ref().map(heap);
                    if next_map
                        .as_ref()
                        .descriptors()
                        .iter()
                        .any(|d| d.name(heap).ptr_eq(smi_name))
                    {
                        return next.erase().load_outcome_on(heap, smi_name);
                    }
                    hop = next.erase();
                }
                self.load_outcome(heap, smi_name)
            }
            Key::Name(name) => self.load_outcome(heap, name),
        }
    }

    pub fn ordinary_own_descriptor<'s>(
        self,
        heap: &'a Heap,
        scope: &'s HandleScope<'_>,
        key: Tagged<'a, Value>,
    ) -> Option<PropertyDescriptor<'s>> {
        if let Ok(Key::Element(i)) = key.classify_key(heap)
            && let Some(o) = self.as_heap_object()
            && let Some(dict) = o.as_ref().element_dictionary(heap)
            && let Some(entry) = dict.as_ref().find(heap, i)
        {
            let details = dict.as_ref().details_at(heap, entry);
            if details & DETAILS_ACCESSOR != 0 {
                let pair = dict
                    .as_ref()
                    .value_at(heap, entry)
                    .get_as::<AccessorPair>(heap)?;
                return Some(PropertyDescriptor::Accessor {
                    get: scope.handle(pair.as_ref().get.get(heap)),
                    set: scope.handle(pair.as_ref().set.get(heap)),
                    enumerable: details & DETAILS_ENUMERABLE != 0,
                    configurable: details & DETAILS_CONFIGURABLE != 0,
                });
            }
            return Some(PropertyDescriptor::Data {
                value: scope.handle(dict.as_ref().value_at(heap, entry)),
                writable: details & DETAILS_WRITABLE != 0,
                enumerable: details & DETAILS_ENUMERABLE != 0,
                configurable: details & DETAILS_CONFIGURABLE != 0,
            });
        }
        if let Ok(Key::Element(i)) = key.classify_key(heap)
            && let Some(o) = self.as_heap_object()
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
        let o = self.as_heap_object()?;
        if let Some(v) = o.as_ref().array_length(heap, name) {
            return Some(PropertyDescriptor::Data {
                value: scope.handle(v),
                writable: true,
                enumerable: false,
                configurable: false,
            });
        }
        let map = o.as_ref().map(heap);
        for d in map.descriptors() {
            if !d.name(heap).ptr_eq(name) {
                continue;
            }
            if d.flags().is_accessor() {
                let pair = d
                    .value
                    .get(heap)
                    .get_as::<AccessorPair>(heap)
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

    pub fn has_property(self, heap: &'a Heap, name: Tagged<'a, SlotName>) -> bool {
        // String exotics own `length` and in-range indices
        if let Some(s) = DenseString::from_receiver(heap, self)
            && s.as_ref().own_key(heap, name).is_some()
        {
            return true;
        }
        let name = match name.erase().classify_key(heap) {
            Ok(Key::Element(i)) => {
                // non-array receivers keep index keys as Smi-named
                // descriptors; canonicalize so the named walk finds them
                let smi_name: Tagged<'a, SlotName> = Tagged::from(Smi::new(i as i64));
                if let Some(obj) = self.as_heap_object()
                    && (obj.as_ref().element_value(heap, i).is_some()
                        || obj
                            .as_ref()
                            .element_dictionary(heap)
                            .is_some_and(|d| d.as_ref().find(heap, i).is_some()))
                {
                    return true;
                }
                smi_name
            }
            _ => name,
        };
        !matches!(self.lookup(heap, name), Lookup::NotFound)
    }

    pub fn lookup_in_parents(self, heap: &'a Heap, name: Tagged<'a, SlotName>) -> Lookup<'a> {
        if self == heap.known().null.as_tagged(heap) {
            return Lookup::NotFound;
        }
        if let Some(pairs) = self.get_as::<FixedArray>(heap) {
            // look *inside* the parents: the pair names are slots of the child
            let mut i = 1;
            while i < pairs.len() {
                let result = pairs.at(heap, i).lookup(heap, name);
                if !matches!(result, Lookup::NotFound) {
                    return result;
                }
                i += 2;
            }
            return Lookup::NotFound;
        }
        self.lookup(heap, name)
    }

    /// The home object's [[Prototype]] slot, raw (null / object / FixedArray
    /// of parents / unset). Extracted before ToPropertyKey so key coercion
    /// cannot observe a different chain than the lookup uses (ES 15.4.2:
    /// GetSuperBase happens first).
    pub fn home_proto(self, heap: &'a Heap) -> Option<Tagged<'a, Value>> {
        let obj = self.as_heap_object()?;
        Some(obj.as_ref().prototype(heap))
    }

    pub fn super_lookup(
        self,
        heap: &'a Heap,
        name: Tagged<'a, SlotName>,
    ) -> Result<LoadOutcome<'a>, VmError> {
        let proto = self.home_proto(heap);
        Lookup::super_lookup_from_proto(heap, proto, name)
    }
}

impl Lookup<'_> {
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
            SuperStart::Object(start) => start.load_outcome(heap, name),
            SuperStart::Parents(parents) => {
                let mut i = 1;
                while i < parents.len() {
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
                        Lookup::NotFound => {
                            i += 2;
                        }
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
}

impl StringData<'_> {
    /// Canonical array-index form: decimal digits, no leading zero, no
    /// overflow past `u32::MAX`-worthy lengths.
    pub fn canonical_index(&self) -> Option<usize> {
        if self.is_empty() || self.len() > 10 {
            return None;
        }
        if self.code_unit(0) == b'0' as u16 {
            return (self.len() == 1).then_some(0);
        }
        let mut n: usize = 0;
        for i in 0..self.len() {
            let c = self.code_unit(i);
            if !(b'0' as u16..=b'9' as u16).contains(&c) {
                return None;
            }
            n = n.checked_mul(10)?.checked_add((c - b'0' as u16) as usize)?;
        }
        Some(n)
    }
}

impl Lookup<'_> {
    /// Same, with the lookup start (`holder`) split from the getter
    /// receiver — the proxy forward shape: lookup on the target,
    /// `this` = the proxy.
    pub fn get_property_on<'a>(
        vm: &'a VM,
        heap: &'a mut Heap,
        state: &'a ThreadState,
        holder: Handle<'_, Value>,
        receiver: Handle<'_, Value>,
        name: Handle<'_, Value>,
    ) -> Result<Coercion<'a>, VmError> {
        if Proxy::is_proxy(heap, holder.as_tagged(heap)) {
            return Proxy::get(vm, heap, state, holder, receiver, name);
        }
        state.handle_scope(|scope| -> Result<Coercion<'a>, VmError> {
            let loaded = {
                let heap_ref: &Heap = heap;
                holder
                    .as_tagged(heap_ref)
                    .load_outcome_on(heap_ref, name.as_tagged(heap_ref).as_name())?
            };
            match loaded {
                LoadOutcome::Value(v) => {
                    let v = scope.handle(v);
                    Ok(Coercion::Value(v.as_tagged(heap)))
                }
                LoadOutcome::Getter(getter) => {
                    let getter = scope.handle(getter);
                    let args = scope.stage(&[receiver.as_tagged(heap).erase()]);
                    let result = HostCtx::enter(vm, heap, state, getter, args, None)?;
                    reanchor!(
                        result,
                        heap,
                        |heap_ref: &Heap, tagged: Tagged<'a, Value>| {
                            if tagged == heap_ref.known().exception.as_tagged(heap_ref) {
                                Ok(Coercion::Threw)
                            } else {
                                Ok(Coercion::Value(tagged))
                            }
                        }
                    );
                }
            }
        })
    }

    pub fn to_property_descriptor<'s>(
        vm: &VM,
        heap: &mut Heap,
        state: &ThreadState,
        scope: &'s HandleScope<'_>,
        attrs: Handle<'_, Value>,
    ) -> Result<Option<PartialDescriptor<'s>>, VmError> {
        let cond_4 = attrs.as_tagged(heap).is_primitive(heap);
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
                reads[3].as_tagged(heap).is_truthy(heap),
                reads[4].as_tagged(heap).is_truthy(heap),
                reads[5].as_tagged(heap).is_truthy(heap),
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
                    && !h.as_tagged(heap).is_callable(heap)
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

enum SuperStart<'a> {
    End,
    Object(Tagged<'a, Value>),
    Parents(Tagged<'a, FixedArray>),
}

fn super_start_from_proto<'a>(heap: &'a Heap, proto: Option<Tagged<'a, Value>>) -> SuperStart<'a> {
    let Some(proto) = proto else {
        return SuperStart::End;
    };
    if proto == heap.known().null.as_tagged(heap) || !proto.is_strong_ptr() {
        return SuperStart::End;
    }
    if let Some(parents) = proto.get_as::<FixedArray>(heap) {
        return SuperStart::Parents(parents);
    }
    SuperStart::Object(proto)
}

enum ChainQuery {
    Index(usize),
    AnyIndex,
}

/// The getter outcome for an accessor dictionary entry at `i`.
fn dictionary_accessor_outcome<'a>(
    heap: &'a Heap,
    dict: Tagged<'a, NumberDictionary>,
    i: usize,
) -> Option<LoadOutcome<'a>> {
    let entry = dict.as_ref().find(heap, i)?;
    if !dict.as_ref().is_accessor_at(heap, entry) {
        return None;
    }
    let getter = dict
        .as_ref()
        .value_at(heap, entry)
        .get_as::<AccessorPair>(heap)?
        .as_ref()
        .get
        .get(heap);
    if getter == heap.known().undefined.as_tagged(heap) {
        return Some(LoadOutcome::Value(
            heap.known().undefined.as_tagged(heap).erase(),
        ));
    }
    Some(LoadOutcome::Getter(getter))
}

impl<'a, T> Tagged<'a, T> {
    #[inline(always)]
    pub fn lookup(self, heap: &'a Heap, name: Tagged<'a, SlotName>) -> Lookup<'a> {
        let Some(obj) = self.erase().as_heap_object() else {
            return Lookup::NotFound;
        };
        obj.as_ref().lookup(heap, name)
    }
}

impl<'s, T> Handle<'s, T> {
    #[inline(always)]
    pub fn lookup<'a>(self, heap: &'a Heap, name: Tagged<'a, SlotName>) -> Lookup<'a>
    where
        T: 'a,
    {
        self.as_tagged(heap).lookup(heap, name)
    }
}

impl Map {
    pub fn lookup<'a>(
        &self,
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
                            .get_as::<AccessorPair>(heap)
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

        let proto = self.prototype.get(heap);
        if let Some(pairs) = proto.get_as::<FixedArray>(heap) {
            let name_word = name.erase();
            let mut i = 0;
            while i < pairs.len() {
                if pairs.at(heap, i).ptr_eq(name_word) {
                    return Lookup::Data {
                        holder: receiver,
                        map_index: i / 2,
                        holder_index: i + 1,
                        slot: pairs.as_ref().element_slot(i + 1),
                        flags: SlotFlags::VALUE,
                    };
                }
                i += 2;
            }
        }
        proto.lookup_in_parents(heap, name)
    }
}

impl Object {
    pub fn lookup<'a>(&self, heap: &'a Heap, name: Tagged<'a, SlotName>) -> Lookup<'a> {
        let receiver = Tagged::<Object>::anchored(heap, self);
        // accessor entries in a sparse receiver act like accessor
        // descriptors for the named walk
        if let Some(pair) = self.dictionary_accessor(heap, name.as_index()) {
            return Lookup::Accessor {
                holder: receiver,
                map_index: usize::MAX,
                pair,
            };
        }
        self.header
            .map
            .get(heap)
            .as_ref()
            .lookup(heap, receiver, name)
    }

    /// The `AccessorPair` of a sparse accessor entry at `index`, when
    /// this object is a dictionary-mode array holding one.
    pub fn dictionary_accessor<'a>(
        &self,
        heap: &'a Heap,
        index: Option<usize>,
    ) -> Option<Tagged<'a, AccessorPair>> {
        let index = index?;
        let dict = self.element_dictionary(heap)?;
        let entry = dict.as_ref().find(heap, index)?;
        if !dict.as_ref().is_accessor_at(heap, entry) {
            return None;
        }
        dict.as_ref()
            .value_at(heap, entry)
            .get_as::<AccessorPair>(heap)
    }
}
