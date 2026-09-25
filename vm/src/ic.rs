use crate::{
    Acc, AccessorPair, FeedbackVector, FixedArray, Handle, HandleScope, Heap, Map, MaybeWeak,
    Object, ObjectKind, SlotName, Smi, Tagged, Value, WeakFixedArray, WeakFixedArrayInit,
};

/// Beyond this many live (map, handler) pairs a site goes megamorphic.
pub const MAX_POLYMORPHIC_ENTRIES: usize = 4;

/// Smi kinds for bare (chain-free) handlers.
const KIND_FIELD: i64 = 0;
const KIND_SLOW: i64 = 1;

/// Smi kinds stored at index 0 of a chain-handler array.
const CHAIN_FIELD: i64 = 0;
const CHAIN_ACCESSOR: i64 = 1;
const CHAIN_SETTER: i64 = 4;
const CHAIN_NON_EXISTENT: i64 = 3;
const CHAIN_PARENT_NAME: i64 = 5;

fn kind_smi(kind: i64, payload: i64) -> Smi {
    Smi::new(kind | (payload << 8))
}

/// Decode the `(kind, payload)` Smi view of a handler word.
fn decode_smi(word: Tagged<'_, MaybeWeak<Value>>) -> Option<(i64, i64)> {
    let v = Smi::decode(word.raw())?.value();
    Some((v & 0xff, v >> 8))
}

/// Result of a load hit.
pub enum Hit<'a> {
    Value(Tagged<'a, Value>),
    /// invoke with the receiver as `this`
    Getter(Tagged<'a, Value>),
    NotFound,
}

/// Result of a store hit.
pub enum StoreHit<'a> {
    /// store completed
    Done,
    /// invoke with (receiver, value)
    Setter(Tagged<'a, Value>),
}

/// What the slow-path store decided.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StoreOutcomeKind {
    /// own writable data written (or a silently ignored failure)
    Done,
    /// property added: the receiver changed maps
    Transition,
    /// a setter was (or will be) invoked
    CallSetter,
}

/// `Some` only for ordinary JSReceiver objects (not proxies): the only
/// receivers whose lookups the map walk fully describes.
fn ic_receiver<'a>(receiver: Tagged<'a, Value>, heap: &'a Heap) -> Option<Tagged<'a, Object>> {
    let obj = receiver.as_heap_object()?;
    let kind = obj.map_ref(heap).kind().kind();
    if !kind.is_js_receiver() || kind == ObjectKind::Proxy {
        return None;
    }
    Some(obj)
}

/// Handler word for `map`'s monomorphic or polymorphic entry.
fn probe<'a>(
    heap: &'a Heap,
    vector: Tagged<'a, FeedbackVector>,
    slot: usize,
    map: Tagged<'a, Map>,
) -> Option<Tagged<'a, MaybeWeak<Value>>> {
    let state = vector.as_ref().slot(slot).get(heap);
    if state.raw().is_weak_ptr() {
        if !state.ptr_eq(map) {
            return None;
        }
        return Some(vector.as_ref().slot(slot + 1).get(heap));
    }
    let pairs = state.as_strong()?.get_as::<WeakFixedArray>()?;
    let len = pairs.as_ref().len();
    let mut i = 0;
    while i + 1 < len {
        if let Some(m) = pairs.get(heap, i).as_strong()
            && m.ptr_eq(map)
        {
            return Some(pairs.as_ref().get(heap, i + 1));
        }
        i += 2;
    }
    None
}

/// A built handler, fully rooted.
enum Handler<'s> {
    /// `Field { offset }` or `Slow`
    Smi(Smi),
    /// a chain handler array (field through the chain, accessor/setter,
    /// parent name, non-existent)
    Chain(Handle<'s, WeakFixedArray>),
    /// a store transition: migrate the receiver to this map
    WeakMap(Handle<'s, Map>),
}

impl Handler<'_> {
    fn word<'a>(&self, heap: &'a Heap) -> Tagged<'a, MaybeWeak<Value>> {
        match self {
            Self::Smi(s) => s.into_tagged().as_maybe_weak(),
            Self::Chain(arr) => arr.as_tagged(heap).erase().as_maybe_weak(),
            Self::WeakMap(m) => m.as_tagged(heap).erase().as_weak(),
        }
    }
}

/// Handler word into a pair-array cell.
fn set_word_at(
    arr: Tagged<'_, WeakFixedArray>,
    heap: &Heap,
    i: usize,
    word: Tagged<'_, MaybeWeak<Value>>,
) {
    arr.as_ref().set(heap, i, word);
}

struct ChainEntry<'s> {
    /// next object's parent-pair element index; -1 = prototype slot
    hop: i64,
    /// owning entry index; -1 = receiver
    owner: i64,
    map: Handle<'s, Map>,
}

/// What a walk step found.
enum Found<'s> {
    Data {
        offset: usize,
    },
    Accessor {
        pair: Handle<'s, AccessorPair>,
    },
    ParentName {
        index: i64,
    },
    /// a prototype shape the chain encoding cannot describe
    Uncacheable,
}

/// Mirror of `Map::lookup`: own descriptors, parent names, then parents
/// in priority order, recursively. Entries are never popped: siblings
/// explored before the holder must still lack the property, so their maps
/// are verified on hit too.
fn walk<'s>(
    heap: &Heap,
    scope: &'s HandleScope<'_>,
    obj: Tagged<'_, Object>,
    name: Tagged<'_, SlotName>,
    my_index: i64,
    entries: &mut Vec<ChainEntry<'s>>,
) -> Option<Found<'s>> {
    let map = obj.map_ref(heap);
    for d in map.descriptors() {
        if !d.name(heap).ptr_eq(name) {
            continue;
        }
        if d.flags().is_accessor() {
            let pair = d
                .value
                .get(heap)
                .get_as::<AccessorPair>()
                .expect("accessor descriptor holds an AccessorPair");
            return Some(Found::Accessor {
                pair: scope.handle(pair),
            });
        }
        return Some(Found::Data { offset: d.offset() });
    }
    let proto = map.prototype.get(heap);
    if proto == heap.known().null.as_tagged(heap) {
        return None;
    }
    if let Some(pairs) = proto.get_as::<FixedArray>() {
        let mut i = 0;
        while i < pairs.len() {
            if pairs.at(heap, i).ptr_eq(name) {
                return Some(Found::ParentName {
                    index: (i + 1) as i64,
                });
            }
            i += 2;
        }
        let mut i = 1;
        while i < pairs.len() {
            let Some(parent) = pairs.at(heap, i).as_heap_object() else {
                i += 2;
                continue;
            };
            let index = entries.len() as i64;
            entries.push(ChainEntry {
                hop: i as i64,
                owner: my_index,
                map: scope.handle(parent.map_ref(heap)),
            });
            if let Some(found) = walk(heap, scope, parent, name, index, entries) {
                return Some(found);
            }
            i += 2;
        }
        return None;
    }
    let Some(proto_obj) = proto.as_heap_object() else {
        return Some(Found::Uncacheable);
    };
    if !proto_obj.map_ref(heap).kind().kind().is_js_receiver() {
        return Some(Found::Uncacheable);
    }
    let index = entries.len() as i64;
    entries.push(ChainEntry {
        hop: -1,
        owner: my_index,
        map: scope.handle(proto_obj.map_ref(heap)),
    });
    walk(heap, scope, proto_obj, name, index, entries)
}

/// Resolve a chain handler's holder; `None` on any mismatch or dead
/// link.
fn verify_chain<'a>(
    heap: &'a Heap,
    receiver: Tagged<'a, Object>,
    chain: Tagged<'a, WeakFixedArray>,
) -> Option<Tagged<'a, Object>> {
    let chain_ref = chain.as_ref();
    if chain_ref.len() < 2 || (chain_ref.len() - 2) % 3 != 0 {
        return None;
    }
    let count = (chain_ref.len() - 2) / 3;
    let mut resolved: Vec<Tagged<'a, Object>> = Vec::with_capacity(count);
    for e in 0..count {
        let base = 2 + e * 3;
        let hop = Smi::decode(chain_ref.get(heap, base).raw())?.value();
        let owner_idx = Smi::decode(chain_ref.get(heap, base + 1).raw())?.value();
        let expected = chain_ref.get(heap, base + 2).as_strong()?;
        let owner: Tagged<'a, Object> = if owner_idx < 0 {
            receiver
        } else {
            *resolved.get(owner_idx as usize)?
        };
        let proto = owner.as_ref().map_ref(heap).prototype.get(heap);
        let next = if hop < 0 {
            proto.as_heap_object()?
        } else {
            proto
                .get_as::<FixedArray>()?
                .at(heap, hop as usize)
                .as_heap_object()?
        };
        if !next.map_ref(heap).ptr_eq(expected) {
            return None;
        }
        resolved.push(next);
    }
    resolved.last().copied().or(Some(receiver))
}

/// Allocate `[kind, payload, (hop, owner, map) × n]`.
fn build_chain_array<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    kind: i64,
    payload: i64,
    pair: Option<&Handle<'_, AccessorPair>>,
    entries: &[ChainEntry<'_>],
) -> Handle<'s, WeakFixedArray> {
    let len = 2 + 3 * entries.len();
    let total = WeakFixedArray::<Value>::layout_for(len);
    heap.allocate_token_enter_heap(total, |token, heap| {
        let mut words: Vec<Tagged<'_, MaybeWeak<Value>>> = Vec::with_capacity(len);
        words.push(kind_smi(kind, payload).into_tagged().as_maybe_weak());
        words.push(match pair {
            Some(pair) => pair.as_tagged(heap).erase().as_weak(),
            None => Smi::new(0).into_tagged().as_maybe_weak(),
        });
        for e in entries {
            words.push(Smi::new(e.hop).into_tagged().as_maybe_weak());
            words.push(Smi::new(e.owner).into_tagged().as_maybe_weak());
            words.push(e.map.as_tagged(heap).erase().as_weak());
        }
        let arr = token.allocate::<WeakFixedArray>(WeakFixedArrayInit { values: &words });
        scope.handle(arr)
    })
}

/// What the load analysis found, rooted so it survives handler
/// construction.
struct Plan<'s> {
    kind: PlanKind,
    /// field offset (data) or parent-pair element index (parent name)
    offset: usize,
    /// visited prototype chain in walk order
    entries: Vec<ChainEntry<'s>>,
    pair: Option<Handle<'s, AccessorPair>>,
}

enum PlanKind {
    Field,
    Slow,
    Accessor,
    ParentName,
    NonExistent,
}

impl Plan<'_> {
    fn non_existent(&self) -> bool {
        matches!(self.kind, PlanKind::NonExistent)
    }

    fn into_handler<'s>(self, heap: &mut Heap, scope: &'s HandleScope<'_>) -> Handler<'s> {
        match self.kind {
            PlanKind::Field if self.entries.is_empty() => {
                Handler::Smi(kind_smi(KIND_FIELD, self.offset as i64))
            }
            PlanKind::Field => Handler::Chain(build_chain_array(
                heap,
                scope,
                CHAIN_FIELD,
                self.offset as i64,
                None,
                &self.entries,
            )),
            PlanKind::Slow => Handler::Smi(kind_smi(KIND_SLOW, 0)),
            PlanKind::Accessor => Handler::Chain(build_chain_array(
                heap,
                scope,
                CHAIN_ACCESSOR,
                0,
                self.pair.as_ref(),
                &self.entries,
            )),
            PlanKind::ParentName => Handler::Chain(build_chain_array(
                heap,
                scope,
                CHAIN_PARENT_NAME,
                self.offset as i64,
                None,
                &self.entries,
            )),
            PlanKind::NonExistent => Handler::Chain(build_chain_array(
                heap,
                scope,
                CHAIN_NON_EXISTENT,
                0,
                None,
                &self.entries,
            )),
        }
    }
}

/// Analyze a load into a cacheable plan.
fn analyze_load<'s>(
    heap: &Heap,
    scope: &'s HandleScope<'_>,
    receiver: Tagged<'_, Object>,
    name: Tagged<'_, SlotName>,
) -> Plan<'s> {
    let slow = || Plan {
        kind: PlanKind::Slow,
        offset: 0,
        entries: Vec::new(),
        pair: None,
    };
    if receiver.as_ref().array_length(heap, name).is_some() {
        return slow();
    }
    let mut entries: Vec<ChainEntry<'s>> = Vec::new();
    match walk(heap, scope, receiver, name, -1, &mut entries) {
        Some(Found::Data { offset }) => Plan {
            kind: PlanKind::Field,
            offset,
            entries,
            pair: None,
        },
        Some(Found::Accessor { pair }) => Plan {
            kind: PlanKind::Accessor,
            offset: 0,
            entries,
            pair: Some(pair),
        },
        Some(Found::ParentName { index }) => Plan {
            kind: PlanKind::ParentName,
            offset: index as usize,
            entries,
            pair: None,
        },
        None => Plan {
            kind: PlanKind::NonExistent,
            offset: 0,
            entries,
            pair: None,
        },
        Some(Found::Uncacheable) => slow(),
    }
}

/// Interpreter-facing entry points.
pub struct InlineCache;

/// Execute one load handler word; dead payloads are a miss.
fn apply_load_handler<'a>(
    heap: &'a Heap,
    receiver: Tagged<'a, Object>,
    handler: Tagged<'a, MaybeWeak<Value>>,
) -> Option<Hit<'a>> {
    if !handler.raw().is_ptr() {
        let (kind, payload) = decode_smi(handler)?;
        return match kind {
            KIND_FIELD => Some(Hit::Value(receiver.slot(heap, payload as usize).get(heap))),
            _ => None,
        };
    }
    let chain = handler.as_strong()?.get_as::<WeakFixedArray>()?;
    let chain_ref = chain.as_ref();
    let head = decode_smi(chain_ref.get(heap, 0))?;
    let holder = verify_chain(heap, receiver, chain)?;
    match head.0 {
        CHAIN_FIELD => Some(Hit::Value(holder.slot(heap, head.1 as usize).get(heap))),
        CHAIN_ACCESSOR => {
            let pair = chain_ref
                .get(heap, 1)
                .as_strong()?
                .get_as::<AccessorPair>()?;
            let getter = pair.get.get(heap);
            Some(if getter == heap.known().undefined.as_tagged(heap) {
                Hit::Value(heap.known().undefined.as_tagged(heap).erase())
            } else {
                Hit::Getter(getter)
            })
        }
        CHAIN_PARENT_NAME => {
            let pairs = holder
                .map_ref(heap)
                .prototype
                .get(heap)
                .get_as::<FixedArray>()?;
            Some(Hit::Value(pairs.at(heap, head.1 as usize)))
        }
        CHAIN_NON_EXISTENT => Some(Hit::NotFound),
        _ => None,
    }
}

struct StorePlan<'s> {
    field: Option<usize>,
    setter: Option<Handle<'s, AccessorPair>>,
    entries: Vec<ChainEntry<'s>>,
}

/// Store semantics over the shared walk: own writable data → field,
/// accessor anywhere → setter, everything else → Slow.
fn analyze_store<'s>(
    heap: &Heap,
    scope: &'s HandleScope<'_>,
    receiver: Tagged<'_, Object>,
    name: Tagged<'_, SlotName>,
) -> StorePlan<'s> {
    let mut plan = StorePlan {
        field: None,
        setter: None,
        entries: Vec::new(),
    };
    if receiver.as_ref().array_length(heap, name).is_some() {
        return plan;
    }
    let mut entries: Vec<ChainEntry<'s>> = Vec::new();
    match walk(heap, scope, receiver, name, -1, &mut entries) {
        Some(Found::Data { offset }) if entries.is_empty() => plan.field = Some(offset),
        Some(Found::Accessor { pair }) => plan.setter = Some(pair),
        _ => {}
    }
    plan.entries = entries;
    plan
}

fn store_handler<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    transition_target: Option<Handle<'s, Map>>,
    plan: StorePlan<'s>,
) -> Handler<'s> {
    if let Some(target) = transition_target {
        return Handler::WeakMap(target);
    }
    if let Some(offset) = plan.field {
        return Handler::Smi(kind_smi(KIND_FIELD, offset as i64));
    }
    match plan.setter {
        Some(pair) => Handler::Chain(build_chain_array(
            heap,
            scope,
            CHAIN_SETTER,
            0,
            Some(&pair),
            &plan.entries,
        )),
        None => Handler::Smi(kind_smi(KIND_SLOW, 0)),
    }
}

/// A decoded store handler, rooted for the mutating phase.
enum StoreAction<'s> {
    Field(usize),
    Transition(Handle<'s, Map>),
    Setter(Handle<'s, Value>),
    /// setter-less accessor: writes are silently ignored
    Noop,
}

impl InlineCache {
    pub fn try_load<'a>(
        heap: &'a Heap,
        vector: Option<Tagged<'a, FeedbackVector>>,
        slot: usize,
        receiver: Tagged<'a, Value>,
    ) -> Option<Hit<'a>> {
        let vector = vector?;
        let obj = ic_receiver(receiver, heap)?;
        vector.as_ref().site(slot)?;
        let map = obj.map_ref(heap);
        let handler = probe(heap, vector, slot, map)?;
        apply_load_handler(heap, obj, handler)
    }

    pub fn update_load(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        vector: Option<Handle<'_, FeedbackVector>>,
        slot: usize,
        receiver: Option<Handle<'_, Object>>,
        name: Handle<'_, SlotName>,
        cache_non_existent: bool,
    ) {
        let Some(vector) = vector else { return };
        let Some(receiver) = receiver else { return };
        if vector.as_tagged(heap).site(slot).is_none() {
            return;
        }
        let recv = receiver.as_tagged(heap);
        if ic_receiver(recv.erase(), heap).is_none() {
            return;
        }
        let plan = analyze_load(heap, scope, recv, name.as_tagged(heap));
        if plan.non_existent() && !cache_non_existent {
            return;
        }
        let map = scope.handle(recv.map_ref(heap));
        let handler = plan.into_handler(heap, scope);
        update_site(heap, scope, vector, slot, &map, &handler);
    }

    pub fn try_store<'a>(
        heap: &'a mut Heap,
        scope: &HandleScope<'_>,
        vector: Option<Handle<'_, FeedbackVector>>,
        slot: usize,
        receiver: Handle<'_, Value>,
        name: Handle<'_, SlotName>,
        acc: &Acc<'_>,
    ) -> Option<StoreHit<'a>> {
        let vector = vector?;
        let recv = ic_receiver(receiver.as_tagged(heap).erase(), heap)?;
        vector.as_tagged(heap).site(slot)?;
        let map = scope.handle(recv.map_ref(heap));
        let action = {
            let handler = probe(heap, vector.as_tagged(heap), slot, map.as_tagged(heap))?;
            if !handler.raw().is_ptr() {
                let (kind, payload) = decode_smi(handler)?;
                match kind {
                    KIND_FIELD => StoreAction::Field(payload as usize),
                    _ => return None,
                }
            } else if handler.raw().is_weak_ptr() {
                let target = handler.as_strong()?.get_as::<Map>()?;
                StoreAction::Transition(scope.handle(target))
            } else {
                let chain = handler.as_strong()?.get_as::<WeakFixedArray>()?;
                let chain_ref = chain.as_ref();
                if decode_smi(chain_ref.get(heap, 0))?.0 != CHAIN_SETTER {
                    return None;
                }
                verify_chain(heap, recv, chain)?;
                let pair = chain_ref
                    .get(heap, 1)
                    .as_strong()?
                    .get_as::<AccessorPair>()?;
                let setter = pair.set.get(heap);
                if setter == heap.known().undefined.as_tagged(heap) {
                    StoreAction::Noop
                } else {
                    StoreAction::Setter(scope.handle(setter))
                }
            }
        };
        match action {
            StoreAction::Field(offset) => {
                let recv = receiver.as_tagged(heap).as_heap_object()?;
                let host = recv.erase();
                recv.as_ref()
                    .slot(heap, offset)
                    .set(heap, host, acc.get(heap));
                Some(StoreHit::Done)
            }
            StoreAction::Transition(target) => {
                apply_transition(heap, scope, receiver, name, acc, &target)
                    .then_some(StoreHit::Done)
            }
            StoreAction::Setter(setter) => Some(StoreHit::Setter(setter.as_tagged(heap))),
            StoreAction::Noop => Some(StoreHit::Done),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_store(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        vector: Option<Handle<'_, FeedbackVector>>,
        slot: usize,
        receiver: Handle<'_, Value>,
        name: Handle<'_, SlotName>,
        prev_map: Handle<'_, Map>,
        outcome: StoreOutcomeKind,
    ) {
        let Some(vector) = vector else { return };
        if vector.as_tagged(heap).site(slot).is_none() {
            return;
        }
        let Some(recv) = ic_receiver(receiver.as_tagged(heap).erase(), heap) else {
            return;
        };
        let target =
            (outcome == StoreOutcomeKind::Transition).then(|| scope.handle(recv.map_ref(heap)));
        let plan = analyze_store(heap, scope, recv, name.as_tagged(heap));
        let handler = store_handler(heap, scope, target, plan);
        update_site(heap, scope, vector, slot, &prev_map, &handler);
    }
}

/// Migrate `receiver` to the cached transition target: verify the target
/// row, grow slots when appending, store, swap the map.
fn apply_transition(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    receiver: Handle<'_, Value>,
    name: Handle<'_, SlotName>,
    acc: &Acc<'_>,
    target: &Handle<'_, Map>,
) -> bool {
    let recv = receiver
        .as_tagged(heap)
        .as_heap_object()
        .expect("gated receiver");
    let target_ref = target.as_tagged(heap);
    let Some(row) = target_ref
        .descriptors()
        .iter()
        .find(|d| d.name(heap).ptr_eq(name.as_tagged(heap)))
    else {
        return false;
    };
    if row.flags().is_accessor() {
        return false;
    }
    let offset = row.offset();
    let old_len = recv.slots.get(heap).as_slice().len();
    let new_len = target_ref.value_slot_count();

    if new_len == old_len && offset < old_len {
        let host = recv.erase();
        recv.slot(heap, offset).set(heap, host, acc.get(heap));
        recv.as_ref()
            .header
            .map
            .set(heap, host, target.as_tagged(heap));
        return true;
    }
    if new_len != old_len + 1 || offset != old_len {
        return false;
    }

    heap.allocate_token_enter_heap(FixedArray::<Value>::layout_for(new_len), |token, heap| {
        let recv = receiver
            .as_tagged(heap)
            .as_heap_object()
            .expect("gated receiver");
        let mut values: Vec<Tagged<'_, Value>> = recv
            .slots
            .get(heap)
            .as_slice()
            .iter()
            .map(|slot| slot.get(heap))
            .collect();
        values.push(acc.get(heap));
        let slots = token.allocate::<FixedArray>(scope.stage(&values));
        let host = recv.erase();
        recv.slots.set(heap, host, slots);
        recv.header.map.set(heap, host, target.as_tagged(heap));
    });
    true
}

#[cfg(feature = "ic-stats")]
static UPDATE_LOADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn update_site(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    vector: Handle<'_, FeedbackVector>,
    slot: usize,
    map: &Handle<'_, Map>,
    handler: &Handler<'_>,
) {
    #[cfg(feature = "ic-stats")]
    {
        static ICSTAT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ICSTAT.get_or_init(|| std::env::var_os("OVM_ICSTAT").is_some()) {
            let n = UPDATE_LOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n % 100_000 == 0 {
                eprintln!("update_site calls: {n}");
            }
        }
    }
    let vec_t = vector.as_tagged(heap);
    let Some((state, _)) = vec_t.site(slot) else {
        return;
    };
    let state = state.get(heap);

    if !state.raw().is_ptr() {
        vec_t.set_mono(heap, slot, map.as_tagged(heap), handler.word(heap));
        return;
    }
    if state.raw().is_weak_ptr() {
        if let Some(current) = state.as_strong() {
            if current.ptr_eq(map.as_tagged(heap)) {
                // Same map but the handler missed: a chain changed behind
                // an identical receiver map, or the payload died. A Slow
                // handler is final.
                let old = vec_t.site(slot).expect("checked").1.get(heap);
                if decode_smi(old).is_some_and(|(k, _)| k == KIND_SLOW) {
                    return;
                }
                vec_t.set_handler(heap, slot, handler.word(heap));
                return;
            }
            promote_from_mono(heap, scope, vector, slot, map, handler);
            return;
        }
    } else if let Some(strong) = state.as_strong() {
        if strong.ptr_eq(heap.known().megamorphic_symbol.as_tagged(heap).erase()) {
            return;
        }
        #[cfg(feature = "ic-stats")]
        {
            static ICSTAT2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *ICSTAT2.get_or_init(|| std::env::var_os("OVM_ICSTAT").is_some()) {
                eprintln!("non-mono state at slot {slot}: {:?}", state.raw());
            }
        }
        if let Some(pairs) = strong.get_as::<WeakFixedArray>() {
            let pairs = scope.handle(pairs);
            update_poly(heap, scope, vector, slot, pairs, map, handler);
            return;
        }
    }
    vec_t.set_mono(heap, slot, map.as_tagged(heap), handler.word(heap));
}

/// Replace the monomorphic entry with a pair array of two entries.
fn promote_from_mono(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    vector: Handle<'_, FeedbackVector>,
    slot: usize,
    map: &Handle<'_, Map>,
    handler: &Handler<'_>,
) {
    let total = WeakFixedArray::<Value>::layout_for(4);
    let arr = heap.allocate_token_enter_heap(total, |token, heap| {
        let vec_t = vector.as_tagged(heap);
        let old_map = vec_t
            .slot(slot)
            .get(heap)
            .as_strong()
            .expect("caller checked live");
        let old_handler = vec_t.as_ref().slot(slot + 1).get(heap);
        let words = [
            old_map.as_weak(),
            old_handler,
            map.as_tagged(heap).erase().as_weak(),
            handler.word(heap),
        ];
        let arr = token.allocate::<WeakFixedArray>(WeakFixedArrayInit { values: &words });
        scope.handle(arr)
    });
    vector
        .as_tagged(heap)
        .set_poly(heap, slot, arr.as_tagged(heap));
}

/// Recompute a known map's handler in place, append a pair, or go
/// megamorphic when full.
fn update_poly(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    vector: Handle<'_, FeedbackVector>,
    slot: usize,
    pairs: Handle<'_, WeakFixedArray>,
    map: &Handle<'_, Map>,
    handler: &Handler<'_>,
) {
    let pairs_t = pairs.as_tagged(heap);
    let len = pairs_t.as_ref().len();
    let mut found_at = None;
    let mut live = 0usize;
    let mut i = 0;
    while i + 1 < len {
        if let Some(m) = pairs_t.get(heap, i).as_strong() {
            if m.ptr_eq(map.as_tagged(heap)) {
                found_at = Some(i + 1);
                break;
            }
            live += 1;
        }
        i += 2;
    }
    if let Some(index) = found_at {
        set_word_at(pairs_t, heap, index, handler.word(heap));
        return;
    }
    if live >= MAX_POLYMORPHIC_ENTRIES {
        vector.as_tagged(heap).set_megamorphic(heap, slot);
        return;
    }
    let new_len = 2 * (live + 1);
    let total = WeakFixedArray::<Value>::layout_for(new_len);
    let arr = heap.allocate_token_enter_heap(total, |token, heap| {
        let pairs_t = pairs.as_tagged(heap);
        let len = pairs_t.as_ref().len();
        let mut words: Vec<Tagged<'_, MaybeWeak<Value>>> = Vec::with_capacity(new_len);
        let mut i = 0;
        while i + 1 < len {
            if let Some(m) = pairs_t.get(heap, i).as_strong() {
                words.push(m.as_weak());
                words.push(pairs_t.as_ref().get(heap, i + 1));
            }
            i += 2;
        }
        words.push(map.as_tagged(heap).erase().as_weak());
        words.push(handler.word(heap));
        let arr = token.allocate::<WeakFixedArray>(WeakFixedArrayInit { values: &words });
        scope.handle(arr)
    });
    vector
        .as_tagged(heap)
        .set_poly(heap, slot, arr.as_tagged(heap));
}
