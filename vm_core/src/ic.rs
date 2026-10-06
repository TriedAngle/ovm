use crate::{
    AccessorPair, CallTarget, CallableInfoObject, Cell, Context, DataHandler, DataHandlerInit,
    DenseString, FeedbackVector, FixedArray, FunctionKind, Handle, HandleScope, Heap, Map,
    MaybeWeak, Object, Prototype, RuntimeIndex, SlotName, Smi, Tagged, Value, WeakFixedArray,
    WeakFixedArrayInit,
};

/// Beyond this many live (map, handler) pairs a site goes megamorphic.
pub const MAX_POLYMORPHIC_ENTRIES: usize = 4;

const KIND_FIELD: i64 = 0;
const KIND_SLOW: i64 = 1;
/// Dense array element load; payload = flags + (when packed) the holey
/// epoch the handler was recorded under.
const KIND_ELEMENT: i64 = 2;
/// Dense array element store; payload = flags.
const KIND_ELEMENT_STORE: i64 = 3;
/// `DenseString` index load; payload = flags.
const KIND_INDEXED_STRING: i64 = 4;
/// Array `length`: read the receiver's length slot directly (the map check
/// in the probe already pinned it to an array map). Payload unused.
const KIND_ARRAY_LENGTH: i64 = 5;
/// Prototype getter: `data[0]` is the weak `AccessorPair`; the receiver's
/// chain is guarded by the handler's validity cell.
const KIND_ACCESSOR: i64 = 6;
/// Prototype setter: `data[0]` is the weak `AccessorPair`; the receiver's
/// chain is guarded by the handler's validity cell.
const KIND_SETTER: i64 = 7;
/// Cached miss: the guarded chain has no such property.
const KIND_NON_EXISTENT: i64 = 8;
/// Self-style parent-name slot: `data[0]` is the weak holder object; the
/// payload is the pair element index holding the value.
const KIND_PARENT: i64 = 9;

/// Element-load payload flags.
const ELEMENT_HOLEY: i64 = 1 << 0;
const ELEMENT_ALLOW_OOB: i64 = 1 << 1;
const ELEMENT_EPOCH_SHIFT: i64 = 2;

/// Element-store payload flags.
const STORE_HOLEY: i64 = 1 << 0;
const STORE_GROW: i64 = 1 << 1;

/// Result of the inlined monomorphic probe (see [`InlineCache::probe_mono`]).
pub enum MonoProbe<'a> {
    /// a hot-kind handler (bare field / one-hop prototype field) loaded
    /// inline
    Value(Tagged<'a, Value>),
    /// mono (or first-pair poly) match whose handler needs the out-of-line
    /// application (accessors, chains, slow)
    Handler(Tagged<'a, Object>, Tagged<'a, MaybeWeak<Value>>),
    /// a polymorphic site whose first pair was tried inline: continue the
    /// search from `start`
    Poly {
        obj: Tagged<'a, Object>,
        map: Tagged<'a, Map>,
        pairs: Tagged<'a, WeakFixedArray>,
        start: usize,
    },
    /// megamorphic / weak mismatch / unusable state: go cold
    Miss,
    /// not a heap receiver (or no feedback): the cold path decides
    NotReceiver,
}

/// Result of a load hit.
pub enum Hit<'a> {
    Value(Tagged<'a, Value>),
    /// invoke with the receiver as `this`
    Getter(Tagged<'a, Value>),
    NotFound,
}

/// Result of a keyed element load hit.
pub enum ElementHit<'a> {
    Value(Tagged<'a, Value>),
    /// a single-code-unit string load (`"ab"[1]`); the caller allocates
    Char(u16),
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

/// A built handler, fully rooted.
enum Handler<'s> {
    /// A bare `(kind, payload)` directive: an own field, array length, a
    /// cached miss, `Slow`, or an element handler.
    Smi(Smi),
    /// A guarded handler (prototype field/accessor/setter/parent-name/miss).
    Data(Handle<'s, DataHandler>),
    /// An own accessor, stored as a whole weak pair.
    WeakPair(Handle<'s, AccessorPair>),
    /// A store transition: migrate the receiver to this map.
    WeakMap(Handle<'s, Map>),
}

/// What a walk step found.
enum Found<'s> {
    Data {
        offset: usize,
        holder: Handle<'s, Object>,
    },
    Accessor {
        pair: Handle<'s, AccessorPair>,
        holder: Handle<'s, Object>,
    },
    ParentName {
        index: i64,
        holder: Handle<'s, Object>,
    },
    /// a prototype shape the encoding cannot describe
    Uncacheable,
}

/// What the load analysis found, rooted so it survives handler
/// construction.
#[derive(Clone, Copy)]
struct Plan<'s> {
    kind: PlanKind,
    /// field offset (data) or parent-pair element index (parent name)
    offset: usize,
    receiver_map: Handle<'s, Map>,
    pair: Option<Handle<'s, AccessorPair>>,
    holder: Option<Handle<'s, Object>>,
    /// the property is an own property of the receiver (no guarded chain)
    own: bool,
}

#[derive(Clone, Copy)]
enum PlanKind {
    Field,
    ArrayLength,
    Slow,
    Accessor,
    ParentName,
    NonExistent,
}

struct StorePlan<'s> {
    field: Option<usize>,
    setter: Option<Handle<'s, AccessorPair>>,
    /// the setter is an own accessor (no guarded chain)
    setter_own: bool,
    receiver_map: Handle<'s, Map>,
}

enum StoreAction<'s> {
    Field(usize),
    Transition(Handle<'s, Map>),
    Setter(Handle<'s, Value>),
    /// setter-less accessor: writes are silently ignored
    Noop,
}

/// A monomorphic bytecode-callee hit: everything a frame push needs,
/// pre-decoded.
pub struct CallHit<'a> {
    pub target: Tagged<'a, Object>,
    pub info: Tagged<'a, CallableInfoObject>,
    pub context: Tagged<'a, Context>,
    pub kind: FunctionKind,
}

pub enum CallProbe<'a> {
    Bytecode(CallHit<'a>),
    Runtime(RuntimeIndex),
    Miss,
}

/// What the generic path resolved a call site's callee to.
pub enum CallRecord<'a> {
    Bytecode(Tagged<'a, CallableInfoObject>),
    Runtime(RuntimeIndex),
}

pub struct InlineCache;

fn kind_smi(kind: i64, payload: i64) -> Smi {
    Smi::new(kind | (payload << 8))
}

/// Decode the `(kind, payload)` view of a handler word.
#[inline]
fn decode_handler_smi<T>(word: Tagged<'_, T>) -> Option<(i64, i64)> {
    let v = word.to_i64()?;
    Some((v & 0xff, v >> 8))
}

/// `Some` only for ordinary JSReceiver objects (not proxies): the only
/// receivers whose lookups the map walk fully describes.
#[inline(always)]
fn ic_receiver<'a>(receiver: Tagged<'a, Value>, heap: &'a Heap) -> Option<Tagged<'a, Object>> {
    let obj = receiver.as_heap_object()?;
    let kind = obj.map_ref(heap).kind();
    if !kind.is_js_receiver() || kind.is_proxy() {
        return None;
    }
    Some(obj)
}

/// Handler word for `map`'s monomorphic or polymorphic entry.
#[inline(always)]
fn probe<'a>(
    heap: &'a Heap,
    vector: Tagged<'a, FeedbackVector>,
    slot: usize,
    map: Tagged<'a, Map>,
) -> Option<Tagged<'a, MaybeWeak<Value>>> {
    let state = vector.slot(slot).get(heap);
    if state.is_weak_ptr() {
        if !state.ptr_eq(map) {
            return None;
        }
        return Some(vector.slot(slot + 1).get(heap));
    }
    let pairs = state.as_strong()?.get_as::<WeakFixedArray>(heap)?;
    let len = pairs.len();
    let mut i = 0;
    while i + 1 < len {
        if let Some(m) = pairs.get(heap, i).as_strong()
            && m.ptr_eq(map)
        {
            return Some(pairs.get(heap, i + 1));
        }
        i += 2;
    }
    None
}

#[inline(always)]
fn probe_element_smi<'a>(
    heap: &'a Heap,
    vector: Option<Tagged<'a, FeedbackVector>>,
    slot: usize,
    receiver: Tagged<'a, Value>,
) -> Option<(Tagged<'a, Object>, i64, i64)> {
    let vector = vector?;
    let obj = receiver.as_heap_object()?;
    vector.site(slot)?;
    let handler = probe(heap, vector, slot, obj.map_ref(heap))?;
    if handler.is_ptr() {
        return None;
    }
    let (kind, payload) = decode_handler_smi(handler)?;
    Some((obj, kind, payload))
}

impl Handler<'_> {
    fn word<'a>(&self, heap: &'a Heap) -> Tagged<'a, MaybeWeak<Value>> {
        match self {
            Self::Smi(s) => s.into_tagged().as_maybe_weak(),
            Self::Data(dh) => dh.as_tagged(heap).erase().as_maybe_weak(),
            Self::WeakPair(pair) => pair.as_tagged(heap).erase().as_weak(),
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
    arr.set(heap, i, word);
}

fn walk<'s>(
    heap: &Heap,
    scope: &'s HandleScope<'_>,
    obj: Tagged<'_, Object>,
    name: Tagged<'_, SlotName>,
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
                .get_as::<AccessorPair>(heap)
                .expect("accessor descriptor holds an AccessorPair");
            return Some(Found::Accessor {
                pair: scope.handle(pair),
                holder: scope.handle(obj),
            });
        }
        return Some(Found::Data {
            offset: d.offset(),
            holder: scope.handle(obj),
        });
    }
    let proto = map.prototype.get(heap);
    if proto == heap.known().null.as_tagged(heap) {
        return None;
    }
    if let Some(pairs) = proto.get_as::<FixedArray>(heap) {
        let mut i = 0;
        while i < pairs.len() {
            if pairs.at(heap, i).ptr_eq(name) {
                return Some(Found::ParentName {
                    index: (i + 1) as i64,
                    holder: scope.handle(obj),
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
            if let Some(found) = walk(heap, scope, parent, name) {
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
    walk(heap, scope, proto_obj, name)
}

/// Build a guarded handler. `None` when the receiver's prototype closure
/// cannot be tracked (a proxy or non-receiver prototype).
fn build_data_handler<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    receiver_map: &Handle<'s, Map>,
    smi_handler: Smi,
    data: &[Handle<'s, Value>],
) -> Option<Handle<'s, DataHandler>> {
    let cell = Prototype::get_or_create_prototype_chain_validity_cell(heap, scope, *receiver_map)?;
    let count = data.len();
    Some(
        heap.allocate_token_enter_heap(DataHandler::layout_for(count), |token, heap| {
            let validity_cell = cell.as_tagged(heap).erase().as_maybe_weak();
            let words: Vec<Tagged<'_, MaybeWeak<Value>>> = data
                .iter()
                .map(|word| word.as_tagged(heap).erase().as_weak())
                .collect();
            token
                .allocate::<DataHandler>(DataHandlerInit {
                    smi_handler,
                    validity_cell,
                    data: &words,
                })
                .as_handle(scope)
        }),
    )
}

/// Build a guarded handler, falling back to `Slow` when the closure cannot
/// be tracked.
fn guarded<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    receiver_map: &Handle<'s, Map>,
    smi_handler: Smi,
    data: &[Handle<'s, Value>],
) -> Handler<'s> {
    match build_data_handler(heap, scope, receiver_map, smi_handler, data) {
        Some(dh) => Handler::Data(dh),
        None => Handler::Smi(kind_smi(KIND_SLOW, 0)),
    }
}

/// Whether `map` has any prototype edge at all (object parent or a
/// Self-style parent list).
fn has_parents(heap: &Heap, map: Tagged<'_, Map>) -> bool {
    let proto = map.prototype.get(heap);
    if !proto.is_strong_ptr() {
        return false;
    }
    let known = heap.known();
    !(proto.ptr_eq(known.null.as_tagged(heap).erase())
        || proto.ptr_eq(known.undefined.as_tagged(heap).erase())
        || proto.ptr_eq(known.the_hole.as_tagged(heap).erase()))
}

impl<'s> Plan<'s> {
    fn non_existent(&self) -> bool {
        matches!(self.kind, PlanKind::NonExistent)
    }

    fn into_handler(self, heap: &mut Heap, scope: &'s HandleScope<'_>) -> Handler<'s> {
        let receiver_map = &self.receiver_map;
        match self.kind {
            // Own data field: a bare, chain-free Smi handler.
            PlanKind::Field if self.own => Handler::Smi(kind_smi(KIND_FIELD, self.offset as i64)),
            PlanKind::Field => guarded(
                heap,
                scope,
                receiver_map,
                kind_smi(KIND_FIELD, self.offset as i64),
                &[self.holder.expect("prototype field has a holder").erase()],
            ),
            PlanKind::ArrayLength => Handler::Smi(kind_smi(KIND_ARRAY_LENGTH, 0)),
            PlanKind::Slow => Handler::Smi(kind_smi(KIND_SLOW, 0)),
            // Own accessor: the whole weak pair.
            PlanKind::Accessor if self.own => {
                let pair = self.pair.expect("accessor has a pair");
                Handler::WeakPair(pair.as_tagged(heap).as_handle(scope))
            }
            PlanKind::Accessor => guarded(
                heap,
                scope,
                receiver_map,
                kind_smi(KIND_ACCESSOR, 0),
                &[self.pair.expect("accessor has a pair").erase()],
            ),
            PlanKind::ParentName => guarded(
                heap,
                scope,
                receiver_map,
                kind_smi(KIND_PARENT, self.offset as i64),
                &[self.holder.expect("parent name has a holder").erase()],
            ),
            PlanKind::NonExistent if self.own => Handler::Smi(kind_smi(KIND_NON_EXISTENT, 0)),
            PlanKind::NonExistent => guarded(
                heap,
                scope,
                receiver_map,
                kind_smi(KIND_NON_EXISTENT, 0),
                &[],
            ),
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
    let receiver_map = scope.handle(receiver.map_ref(heap));
    let mk = |kind, offset, pair, holder, own| Plan {
        kind,
        offset,
        receiver_map,
        pair,
        holder,
        own,
    };
    if receiver.array_length(heap, name).is_some() {
        return mk(PlanKind::ArrayLength, 0, None, None, true);
    }
    match walk(heap, scope, receiver, name) {
        Some(Found::Data { offset, holder }) => {
            let own = holder.as_tagged(heap).ptr_eq(receiver);
            mk(PlanKind::Field, offset, None, Some(holder), own)
        }
        Some(Found::Accessor { pair, holder }) => {
            let own = holder.as_tagged(heap).ptr_eq(receiver);
            mk(PlanKind::Accessor, 0, Some(pair), Some(holder), own)
        }
        Some(Found::ParentName { index, holder }) => mk(
            PlanKind::ParentName,
            index as usize,
            None,
            Some(holder),
            false,
        ),
        None => {
            let own = !has_parents(heap, receiver_map.as_tagged(heap));
            mk(PlanKind::NonExistent, 0, None, None, own)
        }
        Some(Found::Uncacheable) => mk(PlanKind::Slow, 0, None, None, false),
    }
}

#[inline(always)]
fn check_prototype_validity_cell(heap: &Heap, handler: Tagged<'_, DataHandler>) -> bool {
    match handler.validity_cell(heap).as_strong() {
        None => true,
        Some(cell) => cell
            .get_as::<Cell>(heap)
            .is_some_and(|cell| cell.is_valid(heap)),
    }
}

fn getter_hit<'a>(heap: &'a Heap, pair: Tagged<'a, AccessorPair>) -> Hit<'a> {
    let getter = pair.get.get(heap);
    if getter == heap.known().undefined.as_tagged(heap) {
        Hit::Value(heap.known().undefined.as_tagged(heap).erase())
    } else {
        Hit::Getter(getter)
    }
}

#[inline(always)]
fn handler_holder<'a>(
    heap: &'a Heap,
    handler: Tagged<'a, DataHandler>,
) -> Option<Tagged<'a, Object>> {
    handler.data(heap, 0).as_strong()?.get_as::<Object>(heap)
}

#[cold]
#[inline(never)]
fn apply_parent_handler<'a>(
    heap: &'a Heap,
    handler: Tagged<'a, DataHandler>,
    payload: i64,
) -> Option<Hit<'a>> {
    let holder = handler_holder(heap, handler)?;
    let pairs = holder
        .map_ref(heap)
        .prototype
        .get(heap)
        .get_as::<FixedArray>(heap)?;
    Some(Hit::Value(pairs.at(heap, payload as usize)))
}

#[inline(always)]
fn apply_data_handler<'a>(heap: &'a Heap, handler: Tagged<'a, DataHandler>) -> Option<Hit<'a>> {
    if !check_prototype_validity_cell(heap, handler) {
        return None;
    }
    let (kind, payload) = decode_handler_smi(handler.smi_handler(heap))?;
    match kind {
        KIND_FIELD => {
            let holder = handler_holder(heap, handler)?;
            Some(Hit::Value(holder.slot(heap, payload as usize).get(heap)))
        }
        KIND_ACCESSOR => {
            let pair = handler
                .data(heap, 0)
                .as_strong()?
                .get_as::<AccessorPair>(heap)?;
            Some(getter_hit(heap, pair))
        }
        KIND_PARENT => apply_parent_handler(heap, handler, payload),
        KIND_NON_EXISTENT => Some(Hit::NotFound),
        _ => None,
    }
}

#[inline(always)]
fn apply_fast<'a>(
    heap: &'a Heap,
    receiver: Tagged<'a, Object>,
    handler: Tagged<'a, MaybeWeak<Value>>,
) -> Option<Hit<'a>> {
    if !handler.is_ptr() {
        let (kind, payload) = decode_handler_smi(handler)?;
        return match kind {
            KIND_FIELD => Some(Hit::Value(receiver.slot(heap, payload as usize).get(heap))),
            KIND_ARRAY_LENGTH => Some(Hit::Value(receiver.length.get(heap).erase())),
            KIND_NON_EXISTENT => Some(Hit::NotFound),
            _ => None,
        };
    }
    if handler.is_weak_ptr() {
        // own accessor as a whole weak pair, cleared weak maps, ...
        return None;
    }
    apply_data_handler(heap, handler.as_strong()?.get_as::<DataHandler>(heap)?)
}

/// Finish an inline mono probe: the value when the interpreter can complete
/// the load itself (an own field or a guarded prototype field), otherwise
/// the handler for the out-of-line path.
#[inline(always)]
fn mono_result<'a>(
    heap: &'a Heap,
    obj: Tagged<'a, Object>,
    handler: Tagged<'a, MaybeWeak<Value>>,
) -> MonoProbe<'a> {
    match apply_fast(heap, obj, handler) {
        Some(Hit::Value(v)) => MonoProbe::Value(v),
        _ => MonoProbe::Handler(obj, handler),
    }
}

#[inline(always)]
fn apply_load_handler<'a>(
    heap: &'a Heap,
    receiver: Tagged<'a, Object>,
    handler: Tagged<'a, MaybeWeak<Value>>,
) -> Option<Hit<'a>> {
    // A weak word is an own accessor stored as a whole pair; everything else
    // (bare Smi or strong DataHandler) is handled by the fast path.
    if handler.is_weak_ptr() {
        let pair = handler.as_strong()?.get_as::<AccessorPair>(heap)?;
        return Some(getter_hit(heap, pair));
    }
    apply_fast(heap, receiver, handler)
}

/// Execute a dense-array element load handler.
fn apply_element_load<'a>(
    heap: &'a Heap,
    obj: Tagged<'a, Object>,
    index: usize,
    payload: i64,
) -> Option<ElementHit<'a>> {
    let holey = payload & ELEMENT_HOLEY != 0;
    let allow_oob = payload & ELEMENT_ALLOW_OOB != 0;
    let oob = |heap: &'a Heap| {
        (allow_oob && heap.indexed_props_valid())
            .then(|| ElementHit::Value(heap.known().undefined.as_tagged(heap).erase()))
    };
    let len = obj.length();
    if index >= len {
        return oob(heap);
    }
    let elements = obj.elements.get(heap);
    if !elements.is_strong_ptr() {
        return None;
    }
    if index >= elements.len() {
        // `length` past the backing store: the index is a hole
        return oob(heap);
    }
    let v = elements.at(heap, index);
    if holey {
        if v == heap.known().the_hole.as_tagged(heap) {
            return oob(heap);
        }
        return Some(ElementHit::Value(v));
    }
    // packed promise: only valid while no array promoted its map since
    // this handler was recorded
    let epoch = (payload >> ELEMENT_EPOCH_SHIFT) as u32;
    if epoch != heap.holey_epoch() {
        return None;
    }
    Some(ElementHit::Value(v))
}

/// Execute an indexed-string load handler.
fn apply_indexed_string<'a>(
    heap: &'a Heap,
    receiver: Tagged<'a, Value>,
    index: usize,
    payload: i64,
) -> Option<ElementHit<'a>> {
    let s = receiver.get_as::<DenseString>(heap)?;
    if index >= s.len() {
        if payload & ELEMENT_ALLOW_OOB != 0 && heap.indexed_props_valid() {
            return Some(ElementHit::Value(
                heap.known().undefined.as_tagged(heap).erase(),
            ));
        }
        return None;
    }
    Some(ElementHit::Char(s.code_unit(heap, index)))
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
        setter_own: false,
        receiver_map: scope.handle(receiver.map_ref(heap)),
    };
    match walk(heap, scope, receiver, name) {
        Some(Found::Data { offset, holder }) if holder.as_tagged(heap).ptr_eq(receiver) => {
            plan.field = Some(offset)
        }
        Some(Found::Accessor { pair, holder }) => {
            plan.setter_own = holder.as_tagged(heap).ptr_eq(receiver);
            plan.setter = Some(pair);
        }
        _ => {}
    }
    plan
}

fn store_handler<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    transition_target: Option<Handle<'s, Map>>,
    plan: StorePlan<'s>,
) -> Handler<'s> {
    if let Some(target) = transition_target {
        if !target.as_tagged(heap).is_prototype() {
            Prototype::ensure_store_transition_validity_cell(heap, scope, target);
            return Handler::WeakMap(target);
        }
        return Handler::Smi(kind_smi(KIND_SLOW, 0));
    }
    if let Some(offset) = plan.field {
        return Handler::Smi(kind_smi(KIND_FIELD, offset as i64));
    }
    let Some(pair) = plan.setter else {
        return Handler::Smi(kind_smi(KIND_SLOW, 0));
    };
    if plan.setter_own {
        return Handler::WeakPair(pair);
    }
    guarded(
        heap,
        scope,
        &plan.receiver_map,
        kind_smi(KIND_SETTER, 0),
        &[pair.erase()],
    )
}

/// A setter found on a prototype or owned by the receiver: run it, or treat
/// a setter-less accessor as a silently ignored write.
fn setter_action<'s>(
    heap: &Heap,
    scope: &'s HandleScope<'_>,
    pair: Tagged<'_, AccessorPair>,
) -> StoreAction<'s> {
    let setter = pair.set.get(heap);
    if setter == heap.known().undefined.as_tagged(heap) {
        StoreAction::Noop
    } else {
        StoreAction::Setter(scope.handle(setter))
    }
}

impl InlineCache {
    #[inline(always)]
    pub fn probe_mono<'a>(
        heap: &'a Heap,
        vector: Option<Tagged<'a, FeedbackVector>>,
        slot: usize,
        receiver: Tagged<'a, Value>,
    ) -> MonoProbe<'a> {
        let Some(vector) = vector else {
            return MonoProbe::NotReceiver;
        };
        let Some(obj) = receiver.as_heap_object() else {
            return MonoProbe::NotReceiver;
        };
        let map = obj.map_ref(heap);
        let (state_slot, handler_slot) = vector.site_unchecked(slot);
        let state = state_slot.get(heap);

        if state.is_weak_ref_to(map) {
            return mono_result(heap, obj, handler_slot.get(heap));
        }
        if state.is_weak_ptr() {
            return MonoProbe::Miss;
        }

        let Some(pairs) = state
            .as_strong()
            .and_then(|s| s.get_as::<WeakFixedArray>(heap))
        else {
            return MonoProbe::Miss;
        };
        let len = pairs.len();
        if len >= 2
            && let Some(m) = pairs.get(heap, 0).as_strong()
            && m.ptr_eq(map)
        {
            return mono_result(heap, obj, pairs.get(heap, 1));
        }
        if len >= 4
            && let Some(m) = pairs.get(heap, 2).as_strong()
            && m.ptr_eq(map)
        {
            return mono_result(heap, obj, pairs.get(heap, 3));
        }
        MonoProbe::Poly {
            obj,
            map,
            pairs,
            start: if len >= 4 { 4 } else { 2 },
        }
    }

    #[inline(never)]
    pub fn apply_mono<'a>(
        heap: &'a Heap,
        obj: Tagged<'a, Object>,
        handler: Tagged<'a, MaybeWeak<Value>>,
    ) -> Option<Hit<'a>> {
        apply_load_handler(heap, obj, handler)
    }

    #[inline(never)]
    pub fn try_load_resume<'a>(
        heap: &'a Heap,
        obj: Tagged<'a, Object>,
        map: Tagged<'a, Map>,
        pairs: Tagged<'a, WeakFixedArray>,
        start: usize,
    ) -> Option<Hit<'a>> {
        let len = pairs.len();
        let mut i = start;
        while i + 1 < len {
            if let Some(m) = pairs.get(heap, i).as_strong()
                && m.ptr_eq(map)
            {
                return apply_load_handler(heap, obj, pairs.get(heap, i + 1));
            }
            i += 2;
        }
        None
    }

    pub fn try_load<'a>(
        heap: &'a Heap,
        vector: Option<Tagged<'a, FeedbackVector>>,
        slot: usize,
        receiver: Tagged<'a, Value>,
    ) -> Option<Hit<'a>> {
        let vector = vector?;
        let obj = ic_receiver(receiver, heap)?;
        vector.site(slot)?;
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

    #[inline]
    pub fn try_load_element<'a>(
        heap: &'a Heap,
        vector: Option<Tagged<'a, FeedbackVector>>,
        slot: usize,
        receiver: Tagged<'a, Value>,
        index: usize,
    ) -> Option<ElementHit<'a>> {
        let (obj, kind, payload) = probe_element_smi(heap, vector, slot, receiver)?;
        match kind {
            KIND_ELEMENT => apply_element_load(heap, obj, index, payload),
            KIND_INDEXED_STRING => apply_indexed_string(heap, receiver, index, payload),
            _ => None,
        }
    }

    #[inline]
    pub fn try_store_element<'a>(
        heap: &'a Heap,
        vector: Option<Tagged<'a, FeedbackVector>>,
        slot: usize,
        receiver: Tagged<'a, Value>,
        index: usize,
        value: Tagged<'a, Value>,
    ) -> Option<Tagged<'a, Value>> {
        let (obj, kind, payload) = probe_element_smi(heap, vector, slot, receiver)?;
        if kind != KIND_ELEMENT_STORE {
            return None;
        }
        let len = obj.length();
        if index > len || (index == len && payload & STORE_GROW == 0) {
            return None;
        }
        let elements = obj.elements.get(heap);
        if !elements.is_strong_ptr() || index >= elements.len() {
            return None;
        }
        if index < len
            && payload & STORE_HOLEY != 0
            && elements.at(heap, index) == heap.known().the_hole.as_tagged(heap)
        {
            return None;
        }
        elements.set(heap, index, value);
        if index == len {
            obj.length
                .set(heap, obj.erase(), Smi::new((index + 1) as i64));
        }
        Some(value)
    }

    /// Re-record the keyed-load site's element handler from a slow-path
    /// observation.
    pub fn update_load_element(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        vector: Option<Handle<'_, FeedbackVector>>,
        slot: usize,
        receiver: Option<Handle<'_, Value>>,
        index: usize,
    ) {
        let Some(vector) = vector else { return };
        let Some(receiver) = receiver else { return };
        if vector.as_tagged(heap).site(slot).is_none() {
            return;
        }
        let recv = receiver.as_tagged(heap);
        let Some(obj) = recv.as_heap_object() else {
            return;
        };
        let map = scope.handle(obj.map_ref(heap));
        let handler = if let Some(s) = recv.get_as::<DenseString>(heap) {
            if index < s.len() && s.code_unit(heap, index) > 0xFF {
                return;
            }
            if index >= s.len() && !heap.indexed_props_valid() {
                return;
            }
            let mut payload = 0;
            if heap.indexed_props_valid() {
                payload |= ELEMENT_ALLOW_OOB;
            }
            Handler::Smi(kind_smi(KIND_INDEXED_STRING, payload))
        } else if obj.is_array(heap) {
            let mut payload = 0;
            if map.as_tagged(heap).kind().is_holey() {
                payload |= ELEMENT_HOLEY;
            }
            if heap.indexed_props_valid() {
                payload |= ELEMENT_ALLOW_OOB;
            }
            if payload & ELEMENT_HOLEY == 0 {
                payload |= (heap.holey_epoch() as i64) << ELEMENT_EPOCH_SHIFT;
            }
            Handler::Smi(kind_smi(KIND_ELEMENT, payload))
        } else {
            return;
        };
        update_site(heap, scope, vector, slot, &map, &handler);
    }

    pub fn update_store_element(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        vector: Option<Handle<'_, FeedbackVector>>,
        slot: usize,
        receiver: Option<Handle<'_, Value>>,
        grew: bool,
    ) {
        let Some(vector) = vector else { return };
        let Some(receiver) = receiver else { return };
        if vector.as_tagged(heap).site(slot).is_none() {
            return;
        }
        let recv = receiver.as_tagged(heap);
        let Some(obj) = recv.as_heap_object() else {
            return;
        };
        if !obj.is_array(heap) {
            return;
        }
        let map = scope.handle(obj.map_ref(heap));
        let mut payload = 0;
        if map.as_tagged(heap).kind().is_holey() {
            payload |= STORE_HOLEY;
        }
        if grew {
            payload |= STORE_GROW;
        }
        let handler = Handler::Smi(kind_smi(KIND_ELEMENT_STORE, payload));
        update_site(heap, scope, vector, slot, &map, &handler);
    }

    #[inline]
    pub fn try_store_fast(
        heap: &mut Heap,
        vector: Option<Tagged<'_, FeedbackVector>>,
        slot: usize,
        receiver: Tagged<'_, Value>,
        name: Tagged<'_, SlotName>,
        value: Tagged<'_, Value>,
    ) -> bool {
        let Some(vector) = vector else {
            return false;
        };
        let Some(recv) = ic_receiver(receiver, heap) else {
            return false;
        };
        if vector.site(slot).is_none() {
            return false;
        }
        let map = recv.map_ref(heap);
        let Some(handler) = probe(heap, vector, slot, map) else {
            return false;
        };
        if !handler.is_ptr() {
            let Some((KIND_FIELD, payload)) = decode_handler_smi(handler) else {
                return false;
            };
            recv.slot(heap, payload as usize)
                .set(heap, recv.erase(), value);
            return true;
        }
        if handler.is_weak_ptr() {
            // Cached own-property add (StoreHandler::StoreTransition):
            // swap to the target map and write the pre-reserved slot. A
            // missing reservation falls back to the cold grow path.
            let Some(strong) = handler.as_strong() else {
                return false;
            };
            let Some(target) = strong.get_as::<Map>(heap) else {
                return false;
            };
            if !target.is_prototype_validity_cell_valid(heap) {
                return false;
            }
            let Some(row) = target
                .descriptors()
                .iter()
                .find(|d| d.name(heap).ptr_eq(name.erase()) && !d.flags().is_accessor())
            else {
                return false;
            };
            let offset = row.offset();
            let old_len = recv.slots.get(heap).as_slice().len();
            if offset >= old_len {
                return false;
            }
            let host = recv.erase();
            recv.slot(heap, offset).set(heap, host, value);
            Prototype::shape_changed(heap, recv.map_ref(heap));
            recv.header.map.set(heap, host, target);
            return true;
        }
        false
    }

    pub fn try_store<'a>(
        heap: &'a mut Heap,
        scope: &HandleScope<'_>,
        vector: Option<Handle<'_, FeedbackVector>>,
        slot: usize,
        receiver: Handle<'_, Value>,
        name: Handle<'_, SlotName>,
        value: Handle<'_, Value>,
    ) -> Option<StoreHit<'a>> {
        let vector = vector?;
        let recv = ic_receiver(receiver.as_tagged(heap).erase(), heap)?;
        vector.as_tagged(heap).site(slot)?;
        let map = scope.handle(recv.map_ref(heap));
        let action = {
            let handler = probe(heap, vector.as_tagged(heap), slot, map.as_tagged(heap))?;
            if !handler.is_ptr() {
                let (kind, payload) = decode_handler_smi(handler)?;
                match kind {
                    KIND_FIELD => StoreAction::Field(payload as usize),
                    _ => return None,
                }
            } else if handler.is_weak_ptr() {
                let strong = handler.as_strong()?;
                if let Some(target) = strong.get_as::<Map>(heap) {
                    if !target.is_prototype_validity_cell_valid(heap) {
                        return None;
                    }
                    StoreAction::Transition(scope.handle(target))
                } else {
                    // own setter as a whole weak pair
                    setter_action(heap, scope, strong.get_as::<AccessorPair>(heap)?)
                }
            } else {
                let data_handler = handler.as_strong()?.get_as::<DataHandler>(heap)?;
                if !check_prototype_validity_cell(heap, data_handler) {
                    return None;
                }
                let (kind, _) = decode_handler_smi(data_handler.smi_handler(heap))?;
                match kind {
                    KIND_SETTER => {
                        let pair = data_handler
                            .data(heap, 0)
                            .as_strong()?
                            .get_as::<AccessorPair>(heap)?;
                        setter_action(heap, scope, pair)
                    }
                    _ => return None,
                }
            }
        };
        match action {
            StoreAction::Field(offset) => {
                let recv = receiver.as_tagged(heap).as_heap_object()?;
                let host = recv.erase();
                recv.slot(heap, offset)
                    .set(heap, host, value.as_tagged(heap));
                Some(StoreHit::Done)
            }
            StoreAction::Transition(target) => {
                apply_transition(heap, scope, receiver, name, value, &target)
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
    value: Handle<'_, Value>,
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

    if offset < old_len {
        // The receiver's array already reserves the slot: an own-field
        // write or an append inside the slack capacity.
        let host = recv.erase();
        recv.slot(heap, offset)
            .set(heap, host, value.as_tagged(heap));
        Prototype::shape_changed(heap, recv.map_ref(heap));
        recv.header.map.set(heap, host, target.as_tagged(heap));
        return true;
    }
    if new_len != old_len + 1 || offset != old_len {
        return false;
    }

    // Reservation exhausted: grow with fresh headroom so the next appends
    // land in place.
    let capacity = new_len + Map::SLACK_MARGIN;
    let slots = heap.allocate_hole_array(capacity).as_handle(scope);
    {
        let recv = receiver
            .as_tagged(heap)
            .as_heap_object()
            .expect("gated receiver");
        let old = recv.slots.get(heap);
        let new = slots.as_tagged(heap);
        for k in 0..old_len {
            new.set(heap, k, old.at(heap, k));
        }
        new.set(heap, offset, value.as_tagged(heap));
        let host = recv.erase();
        recv.slots.set(heap, host, new);
        Prototype::shape_changed(heap, recv.map_ref(heap));
        recv.header.map.set(heap, host, target.as_tagged(heap));
    }
    true
}

fn update_site(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    vector: Handle<'_, FeedbackVector>,
    slot: usize,
    map: &Handle<'_, Map>,
    handler: &Handler<'_>,
) {
    let vec_t = vector.as_tagged(heap);
    let Some((state, _)) = vec_t.site(slot) else {
        return;
    };
    let state = state.get(heap);

    if !state.is_ptr() {
        vec_t.set_mono(heap, slot, map.as_tagged(heap), handler.word(heap));
        return;
    }
    if state.is_weak_ptr() {
        if let Some(current) = state.as_strong() {
            if current.ptr_eq(map.as_tagged(heap)) {
                // Same map but the handler missed: a chain changed behind
                // an identical receiver map, or the payload died. A Slow
                // handler is final.
                let old = vec_t.site(slot).expect("checked").1.get(heap);
                if decode_handler_smi(old).is_some_and(|(k, _)| k == KIND_SLOW) {
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
        if let Some(pairs) = strong.get_as::<WeakFixedArray>(heap) {
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
        let old_handler = vec_t.slot(slot + 1).get(heap);
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
    let len = pairs_t.len();
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
        let len = pairs_t.len();
        let mut words: Vec<Tagged<'_, MaybeWeak<Value>>> = Vec::with_capacity(new_len);
        let mut i = 0;
        while i + 1 < len {
            if let Some(m) = pairs_t.get(heap, i).as_strong() {
                words.push(m.as_weak());
                words.push(pairs_t.get(heap, i + 1));
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

pub const CALL_TAG_RUNTIME: i64 = 1;

#[inline]
fn encode_runtime_payload(index: RuntimeIndex) -> i64 {
    CALL_TAG_RUNTIME | ((index.0 as i64) << 1)
}

#[inline]
fn decode_runtime_payload(raw: i64) -> Option<RuntimeIndex> {
    (raw & CALL_TAG_RUNTIME != 0).then_some(RuntimeIndex((raw >> 1) as usize))
}

/// Decode the packed callable descriptor: `register_count | formal_min<<16
/// | kind<<32` (mirrors `Object::call_target`).
#[inline(always)]
fn decode_descriptor(desc: i64) -> (usize, usize, FunctionKind) {
    let desc = desc as u64;
    (
        (desc & 0xffff) as usize,
        ((desc >> 16) & 0xffff) as usize,
        FunctionKind::decode(((desc >> 32) & 0xf) as i64),
    )
}

impl InlineCache {
    /// Probe a call site: a weak-callee match returns the pre-decoded hit
    /// (or the packed runtime callee); anything else is a miss for the
    /// generic call path to handle and record.
    #[inline(always)]
    pub fn call_probe<'a>(
        heap: &'a Heap,
        vector: Option<Tagged<'a, FeedbackVector>>,
        fb: usize,
        callee: Tagged<'a, Value>,
    ) -> CallProbe<'a> {
        let Some(vector) = vector else {
            return CallProbe::Miss;
        };
        let state = vector.slot(fb).get(heap);
        if !state.is_weak_ptr() {
            if state.is_strong_ptr() && !state.ptr_eq(heap.known().the_hole.as_tagged(heap).erase())
            {
                return match Object::call_target(heap, callee) {
                    Some(CallTarget::Runtime(rt)) => CallProbe::Runtime(rt),
                    Some(CallTarget::Bytecode {
                        target,
                        info,
                        context,
                        kind,
                    }) => CallProbe::Bytecode(CallHit {
                        target,
                        info,
                        context,
                        kind,
                    }),
                    _ => CallProbe::Miss,
                };
            }
            return CallProbe::Miss;
        }
        if !state.ptr_eq(callee) {
            return CallProbe::Miss;
        }
        let payload = vector.slot(fb + 1).get(heap);
        if let Some(tag) = payload.to_i64() {
            // runtime callee: index and shape come straight from the
            // payload — no object decode needed
            return match decode_runtime_payload(tag) {
                Some(rt) => CallProbe::Runtime(rt),
                None => CallProbe::Miss,
            };
        }

        let Some(info) = payload
            .as_strong()
            .and_then(|p| p.get_as::<CallableInfoObject>(heap))
        else {
            return CallProbe::Miss;
        };
        let (_, _, kind) = decode_descriptor(info.descriptor.to_smi_unchecked().value());
        let Some(target) = callee.as_heap_object() else {
            return CallProbe::Miss;
        };
        let Some(context) = target.slot(heap, 1).get(heap).get_as::<Context>(heap) else {
            return CallProbe::Miss;
        };
        CallProbe::Bytecode(CallHit {
            target,
            info,
            context,
            kind,
        })
    }
}

impl InlineCache {
    /// Record a call site's callee after the generic path resolved it:
    /// re-arm a cleared weak entry, flip the site megamorphic on an
    /// unrelated second callee, and keep it monomorphic across closures
    /// sharing the same `CallableInfoObject`.
    pub fn call_update(
        heap: &Heap,
        vector: Option<Tagged<FeedbackVector>>,
        fb: usize,
        callee: Tagged<'_, Value>,
        record: CallRecord<'_>,
    ) {
        let Some(vector) = vector else {
            return;
        };
        let Some((state_slot, tag_slot)) = vector.site(fb) else {
            return;
        };
        let host = vector.erase();
        if state_slot.is_cleared() {
            // a collected weak entry leaves the site free to become
            // monomorphic again
        } else {
            let state = state_slot.get(heap);
            if state.is_strong_ptr() {
                let hole = heap.known().the_hole.as_tagged(heap).erase();
                if !state.ptr_eq(hole) {
                    // megamorphic: leave it alone
                    return;
                }
            } else if !state.ptr_eq(callee) {
                let same_code = match (&record, state.as_strong().and_then(|c| c.as_heap_object()))
                {
                    (CallRecord::Bytecode(info), Some(old)) => old
                        .callable_info(heap)
                        .is_some_and(|old_info| old_info.ptr_eq(*info)),
                    _ => false,
                };
                if !same_code {
                    vector.set_megamorphic(heap, fb);
                    return;
                }
            }
        }
        state_slot.set_weak(heap, host, callee);
        match record {
            // bytecode: the payload IS the resolved info
            CallRecord::Bytecode(info) => tag_slot.set_strong(heap, host, info.erase()),
            // runtime: index and argument shape packed into the payload
            CallRecord::Runtime(rt) => tag_slot.set(
                heap,
                host,
                Smi::new(encode_runtime_payload(rt))
                    .into_tagged()
                    .as_maybe_weak(),
            ),
        }
    }
}
