use core::alloc::Layout;

use crate::proxy::Proxy;
use crate::{
    AccessorPair, AllocToken, Compare, FixedArray, Handle, HandleScope, Heap, HeapObject, Key,
    Lookup, Map, MapInit, MapKind, MaybeWeak, Object, Prototype, SlotFlags, SlotName, Smi, Symbol,
    Tagged, Value, VmError, WeakFixedArray, WeakFixedArrayInit,
};

/// Store semantics:
/// - Self-style writes through to an inherited writable slot
/// - JS-style shadows it with a new own property on the receiver.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StoreSemantics {
    WriteThrough,
    Shadow,
}

#[derive(Debug, Copy, Clone)]
pub enum StoreOutcome<'s> {
    Done,
    Transition {
        receiver: Handle<'s, Object>,
        name: Handle<'s, SlotName>,
    },
    CallSetter {
        setter: Handle<'s, Value>,
    },
}

pub enum SiblingChange<'s> {
    Prototype(Handle<'s, Value>),
    Holey,
}

impl<'s> SiblingChange<'s> {
    fn kind(&self, kind: MapKind) -> MapKind {
        match self {
            Self::Prototype(_) => kind,
            Self::Holey => kind.union(MapKind::HOLEY),
        }
    }

    fn prototype(
        &self,
        heap: &Heap,
        scope: &'s HandleScope<'_>,
        parent: Tagged<'_, Map>,
    ) -> Handle<'s, Value> {
        match self {
            Self::Prototype(proto) => *proto,
            Self::Holey => scope.handle(parent.prototype.get(heap)),
        }
    }

    fn find<'a>(&self, heap: &'a Heap, parent: Tagged<'a, Map>) -> Option<Tagged<'a, Map>> {
        match self {
            Self::Prototype(proto) => parent.find_prototype_transition(heap, proto.as_tagged(heap)),
            Self::Holey => parent.find_holey_transition(heap),
        }
    }

    fn sentinel<'a>(&self, heap: &'a Heap) -> Tagged<'a, Symbol> {
        match self {
            Self::Prototype(_) => heap.known().prototype_transition_symbol.as_tagged(heap),
            Self::Holey => heap.known().holey_transition_symbol.as_tagged(heap),
        }
    }
}

impl<'a> Tagged<'a, Value> {
    pub fn store_lookup<'s>(
        self,
        heap: &'a Heap,
        scope: &'s HandleScope<'_>,
        name: Tagged<'a, SlotName>,
        value: Tagged<'a, Value>,
        semantics: StoreSemantics,
    ) -> Result<StoreOutcome<'s>, VmError> {
        // TODO(strict-mode): take the active function's language mode and
        // distinguish throwing strict failures from ignored sloppy failures.
        // null/undefined have no [[Prototype]]: property access throws
        if self.ptr_eq(heap.known().null.as_tagged(heap).erase())
            || self.ptr_eq(heap.known().undefined.as_tagged(heap).erase())
        {
            return Err(VmError::Type);
        }
        // non-receiver heap values (VMStrings, Symbols, Floats, ...) and
        // smis are not property stores' targets: sloppy-mode stores onto
        // primitives are silently ignored (strict throws — deferred with
        // the other language-mode TODOs)
        let Some(receiver) = scope.cast::<Object>(heap, self) else {
            return Ok(StoreOutcome::Done);
        };
        match self.lookup(heap, name) {
            Lookup::Data {
                slot,
                holder,
                flags,
                ..
            } => {
                if !flags.is_writable() {
                    return Err(VmError::Type);
                }
                let host = holder.erase();
                if semantics == StoreSemantics::Shadow && host != self {
                    // inherited writable data property: JS creates an own
                    // property on the receiver
                    return Ok(StoreOutcome::Transition {
                        receiver,
                        name: scope.handle(name),
                    });
                }
                slot.set(heap, host, value);
                Ok(StoreOutcome::Done)
            }
            Lookup::NotFound => Ok(StoreOutcome::Transition {
                receiver,
                name: scope.handle(name),
            }),
            Lookup::Accessor { pair, .. } => {
                let setter = pair.set.get(heap);
                // no setter (undefined sentinel): sloppy-mode writes to a
                // setter-less accessor are silently ignored
                if setter.ptr_eq(heap.known().undefined.as_tagged(heap).erase()) {
                    return Ok(StoreOutcome::Done);
                }
                Ok(StoreOutcome::CallSetter {
                    setter: scope.handle(setter),
                })
            }
        }
    }
}

impl<'s, T> Handle<'s, T> {
    pub fn store_lookup<'a, 'd>(
        self,
        heap: &'a Heap,
        scope: &'d HandleScope<'_>,
        name: Tagged<'a, SlotName>,
        value: Tagged<'a, Value>,
        semantics: StoreSemantics,
    ) -> Result<StoreOutcome<'d>, VmError> {
        self.as_tagged(heap)
            .erase()
            .store_lookup(heap, scope, name, value, semantics)
    }
}

impl<'a> Tagged<'a, Value> {
    pub fn store_lookup_existing(
        self,
        heap: &'a Heap,
        name: Tagged<'a, SlotName>,
        value: Tagged<'a, Value>,
        semantics: StoreSemantics,
    ) -> Result<bool, VmError> {
        if self.ptr_eq(heap.known().null.as_tagged(heap).erase())
            || self.ptr_eq(heap.known().undefined.as_tagged(heap).erase())
        {
            return Err(VmError::Type);
        }
        debug_assert!(
            !Proxy::is_proxy(heap, self.erase()),
            "fast store on a proxy receiver"
        );
        if !self
            .as_heap_object()
            .is_some_and(|obj| Object::matches_kind(obj.map_ref(heap).kind().kind()))
        {
            return Ok(true);
        }
        match self.lookup(heap, name) {
            Lookup::Data {
                slot,
                holder,
                flags,
                ..
            } => {
                if !flags.is_writable() {
                    return Err(VmError::Type);
                }
                let host = holder.erase();
                if semantics == StoreSemantics::Shadow && host != self {
                    return Ok(false);
                }
                slot.set(heap, host, value);
                Ok(true)
            }
            Lookup::NotFound => Ok(false),
            Lookup::Accessor { .. } => {
                debug_assert!(false, "fast store on an accessor property");
                Ok(true)
            }
        }
    }
}

fn super_store_on_receiver<'a, 's>(
    heap: &'a Heap,
    scope: &'s HandleScope<'_>,
    recv: Tagged<'a, Value>,
    name: Tagged<'a, SlotName>,
    value: Tagged<'a, Value>,
) -> Result<StoreOutcome<'s>, VmError> {
    let Some(receiver) = scope.cast::<Object>(heap, recv) else {
        return Err(VmError::Type);
    };
    match recv.lookup(heap, name) {
        // already owned (the nearest hit is the receiver itself, not an
        // inherited one): overwrite the slot instead of re-adding it
        Lookup::Data {
            holder,
            slot,
            flags,
            ..
        } if holder.as_ref().erase() == recv => {
            if !flags.is_writable() {
                return Err(VmError::Type);
            }
            slot.set(heap, recv, value);
            Ok(StoreOutcome::Done)
        }
        // own accessor: Receiver.[[DefineOwnProperty]]({value}) on an
        // accessor is an incompatible change (ES 9.1.9.2 step 3.d.i)
        Lookup::Accessor { holder, .. } if holder.as_ref().erase() == recv => Err(VmError::Type),
        // not owned by the receiver: define a fresh own property
        _ => Ok(StoreOutcome::Transition {
            receiver,
            name: scope.handle(name),
        }),
    }
}
pub struct Transition;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Change {
    /// Append a new last descriptor.
    Append,
    /// Replace the descriptor at `index`, keeping its position.
    Replace { index: usize },
}

impl Transition {
    pub fn target<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        parent: impl for<'a> Fn(&'a Heap) -> Tagged<'a, Map>,
        name: Handle<SlotName>,
        flags: SlotFlags,
        pair: Option<(Handle<'_, Value>, Handle<'_, Value>)>,
        change: Change,
    ) -> Handle<'s, Map> {
        debug_assert_eq!(flags.is_accessor(), pair.is_some());
        loop {
            let pair_values = pair.map(|(get, set)| (get.as_tagged(heap), set.as_tagged(heap)));
            if let Some(target) =
                parent(heap).find_transition(heap, name.as_tagged(heap), flags, pair_values)
            {
                return target.as_handle(scope);
            }

            let parent_ref = parent(heap);
            let old_row = match change {
                Change::Append => None,
                Change::Replace { index } => {
                    let d = &parent_ref.descriptors()[index];
                    let offset = (!d.flags().is_accessor())
                        .then(|| Smi::decode(d.value.get(heap).raw()).expect("data row offset"));
                    Some((d.flags(), offset))
                }
            };
            let kind = parent_ref.kind();
            let descriptor_count = parent_ref.descriptor_count();
            let value_slot_count = parent_ref.value_slot_count();
            let prototype = scope.handle(parent_ref.prototype.get(heap));
            let pairs_len = parent_ref.transitions.load(heap).map_or(0, |a| a.len());

            let grow = !flags.is_accessor()
                && match change {
                    Change::Append => true,
                    Change::Replace { .. } => old_row.expect("replace row").0.is_accessor(),
                };
            let row_offset = match change {
                Change::Replace { .. } if !grow && !flags.is_accessor() => old_row
                    .expect("replace row")
                    .1
                    .expect("data row offset")
                    .value()
                    as usize,
                _ => value_slot_count,
            };
            let appends = usize::from(matches!(change, Change::Append));

            let map_layout = Map::layout_for(descriptor_count + appends);
            let pairs_layout = WeakFixedArray::<Value>::layout_for(pairs_len + 2);
            let total = match pair.is_some() {
                true => AllocToken::total_for(&[
                    Layout::new::<AccessorPair>(),
                    map_layout,
                    pairs_layout,
                ]),
                false => AllocToken::total_for(&[map_layout, pairs_layout]),
            };

            let attempt = heap.allocate_token_enter_heap(total, |token, heap| {
                let parent_ref = parent(heap);
                let word = parent_ref.transitions.load_word(heap);
                let old = parent_ref.transitions.decode(heap, word);
                if old.map_or(0, |a| a.len()) != pairs_len {
                    token.discard_remaining();
                    return None;
                }

                let name_word = name.as_tagged(heap);
                let row_value = match pair {
                    Some((get, set)) => {
                        scope.handle(token.allocate::<AccessorPair>((get, set)).erase())
                    }
                    None => scope.handle(Smi::new(row_offset as i64)),
                };

                let mut descriptors: Vec<(Handle<'_, SlotName>, SlotFlags, Handle<'_, Value>)> =
                    parent_ref
                        .descriptors()
                        .iter()
                        .map(|d| {
                            (
                                scope.handle(d.name(heap)),
                                d.flags(),
                                scope.handle(d.value.get(heap)),
                            )
                        })
                        .collect();
                let name_handle = scope.handle(name_word);
                match change {
                    Change::Append => descriptors.push((name_handle, flags, row_value)),
                    Change::Replace { index } => {
                        descriptors[index] = (name_handle, flags, row_value)
                    }
                }
                let child = token.allocate::<Map>(MapInit {
                    kind,
                    value_slot_count: value_slot_count + usize::from(grow),
                    descriptors: &descriptors,
                    prototype,
                });
                child.pred.set(heap, child.erase(), parent_ref);

                let mut pairs: Vec<Tagged<'_, MaybeWeak<Value>>> =
                    Vec::with_capacity(pairs_len + 2);
                if let Some(old) = old {
                    for entry in old.as_slice().as_chunks::<2>().0 {
                        pairs.push(entry[0].get(heap));
                        pairs.push(entry[1].get(heap));
                    }
                }
                pairs.push(name_word.erase().as_maybe_weak());
                pairs.push(child.erase().as_weak());
                let pairs = token.allocate::<WeakFixedArray>(WeakFixedArrayInit { values: &pairs });

                match parent_ref
                    .transitions
                    .publish(heap, parent_ref.erase(), word, pairs)
                {
                    Ok(()) => Some(child.as_handle(scope)),
                    Err(_) => None,
                }
            });
            if let Some(child) = attempt {
                return child;
            }
        }
    }

    pub fn sibling_target<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        parent: impl for<'a> Fn(&'a Heap) -> Tagged<'a, Map>,
        change: SiblingChange<'s>,
    ) -> Handle<'s, Map> {
        loop {
            if let Some(target) = change.find(heap, parent(heap)) {
                return target.as_handle(scope);
            }

            let parent_ref = parent(heap);
            let kind = change.kind(parent_ref.kind());
            let descriptor_count = parent_ref.descriptor_count();
            let value_slot_count = parent_ref.value_slot_count();
            let prototype = change.prototype(heap, scope, parent_ref);
            let pairs_len = parent_ref.transitions.load(heap).map_or(0, |a| a.len());

            let map_layout = Map::layout_for(descriptor_count);
            let pairs_layout = WeakFixedArray::<Value>::layout_for(pairs_len + 2);
            let total = AllocToken::total_for(&[map_layout, pairs_layout]);

            let attempt = heap.allocate_token_enter_heap(total, |token, heap| {
                let parent_ref = parent(heap);
                let word = parent_ref.transitions.load_word(heap);
                let old = parent_ref.transitions.decode(heap, word);
                if old.map_or(0, |a| a.len()) != pairs_len {
                    token.discard_remaining();
                    return None;
                }

                let descriptors: Vec<(Handle<'_, SlotName>, SlotFlags, Handle<'_, Value>)> =
                    parent_ref
                        .descriptors()
                        .iter()
                        .map(|d| {
                            (
                                scope.handle(d.name(heap)),
                                d.flags(),
                                scope.handle(d.value.get(heap)),
                            )
                        })
                        .collect();
                let child = token.allocate::<Map>(MapInit {
                    kind,
                    value_slot_count,
                    descriptors: &descriptors,
                    prototype,
                });
                child.pred.set(heap, child.erase(), parent_ref);

                let sentinel = change.sentinel(heap);
                let mut pairs: Vec<Tagged<'_, MaybeWeak<Value>>> =
                    Vec::with_capacity(pairs_len + 2);
                if let Some(old) = old {
                    for entry in old.as_slice().as_chunks::<2>().0 {
                        pairs.push(entry[0].get(heap));
                        pairs.push(entry[1].get(heap));
                    }
                }
                pairs.push(sentinel.erase().as_maybe_weak());
                pairs.push(child.erase().as_weak());
                let pairs = token.allocate::<WeakFixedArray>(WeakFixedArrayInit { values: &pairs });

                match parent_ref
                    .transitions
                    .publish(heap, parent_ref.erase(), word, pairs)
                {
                    Ok(()) => Some(child.as_handle(scope)),
                    Err(_) => None,
                }
            });
            if let Some(child) = attempt {
                return child;
            }
            // another thread published first: retry the lookup
        }
    }

    fn grow_slots_and_swap(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        flags: SlotFlags,
        value: Handle<Value>,
    ) {
        // The append's target row and offset, plus the parent's used count
        // and the array's current reservation: a preallocated array can
        // absorb the write in place.
        let (offset, parent_count, old_len) = {
            let receiver_ref = receiver.as_tagged(heap);
            let parent = receiver_ref.map_ref(heap);
            let target = parent
                .find_transition(heap, name.as_tagged(heap), flags, None)
                .expect("transition recorded above");
            let index = target
                .as_ref()
                .descriptors()
                .iter()
                .position(|d| d.name(heap).ptr_eq(name.as_tagged(heap)) && d.flags() == flags)
                .expect("transition target row");
            (
                target.as_ref().descriptor(index).offset(),
                parent.value_slot_count(),
                receiver_ref.slots.get(heap).as_slice().len(),
            )
        };

        if offset < old_len {
            let receiver_ref = receiver.as_tagged(heap);
            let target = receiver_ref
                .map_ref(heap)
                .find_transition(heap, name.as_tagged(heap), flags, None)
                .expect("transition recorded above");
            let host = receiver_ref.erase();
            receiver_ref
                .slot(heap, offset)
                .set(heap, host, value.as_tagged(heap));
            Prototype::shape_changed(heap, receiver_ref.map_ref(heap));
            receiver_ref.header.map.set(heap, host, target);
            return;
        }

        // Reservation exhausted: grow with fresh headroom so the next
        // appends land in place.
        debug_assert_eq!(offset, old_len, "append out of order");
        debug_assert_eq!(offset, parent_count, "slot count desynced from map");
        let capacity = parent_count + 1 + Map::SLACK_MARGIN;
        let slots = heap.allocate_hole_array(capacity).as_handle(scope);
        let receiver_ref = receiver.as_tagged(heap);
        let target = receiver_ref
            .map_ref(heap)
            .find_transition(heap, name.as_tagged(heap), flags, None)
            .expect("transition recorded above");
        {
            let old = receiver_ref.slots.get(heap);
            let new = slots.as_tagged(heap);
            for k in 0..old_len {
                new.as_ref().set(heap, k, old.at(heap, k));
            }
            new.as_ref().set(heap, offset, value.as_tagged(heap));
        }
        let host = receiver_ref.erase();
        receiver_ref.slots.set(heap, host, slots.as_tagged(heap));
        Prototype::shape_changed(heap, receiver_ref.map_ref(heap));
        receiver_ref.header.map.set(heap, host, target);
    }

    fn swap_map(
        heap: &mut Heap,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        flags: SlotFlags,
        pair: Option<(Handle<'_, Value>, Handle<'_, Value>)>,
    ) {
        let receiver_ref = receiver.as_tagged(heap);
        let pair_values = pair.map(|(get, set)| (get.as_tagged(heap), set.as_tagged(heap)));
        let target = receiver_ref
            .map_ref(heap)
            .find_transition(heap, name.as_tagged(heap), flags, pair_values)
            .expect("transition recorded above");
        Prototype::shape_changed(heap, receiver_ref.map_ref(heap));
        receiver_ref
            .header
            .map
            .set(heap, receiver.as_tagged(heap).erase(), target);
    }

    fn write_slot(
        heap: &mut Heap,
        receiver: Handle<Object>,
        index: usize,
        value: Handle<'_, Value>,
    ) {
        let offset = receiver.as_tagged(heap).map_ref(heap).descriptors()[index].offset();
        receiver.as_tagged(heap).slot(heap, offset).set(
            heap,
            receiver.as_tagged(heap).erase(),
            value.as_tagged(heap),
        );
    }

    pub fn remove_property(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
    ) {
        let (kind, prototype, surviving, values) = {
            let name_word = name.as_tagged(heap);
            let obj = receiver.as_tagged(heap);
            let parent = obj.map_ref(heap);
            let descriptors = parent.descriptors();
            let index = descriptors
                .iter()
                .position(|d| d.name(heap).ptr_eq(name_word))
                .expect("caller verified the descriptor exists");
            // the structural region is everything below the first
            // descriptor-owned slot (function info/context; empty for
            // ordinary objects)
            let base = descriptors
                .iter()
                .filter(|d| !d.flags().is_accessor())
                .map(|d| d.offset())
                .min()
                .unwrap_or(parent.value_slot_count());
            // names are rooted here and re-anchored in the allocating
            // closures: a Tagged cannot escape this non-allocating region
            let mut surviving: Vec<(Handle<'_, SlotName>, SlotFlags, Handle<'_, Value>)> =
                Vec::with_capacity(descriptors.len() - 1);
            let mut values: Vec<Handle<'_, Value>> = obj.slots.get(heap).as_slice()[..base]
                .iter()
                .map(|slot| scope.handle(slot.get(heap)))
                .collect();
            for (i, d) in descriptors.iter().enumerate() {
                if i == index {
                    continue;
                }
                if d.flags().is_accessor() {
                    // accessors embed their pair in the descriptor row
                    surviving.push((
                        scope.handle(d.name(heap)),
                        d.flags(),
                        scope.handle(d.value.get(heap)),
                    ));
                } else {
                    // data rows re-dense their offsets; the value rides
                    // along in slot order
                    values.push(scope.handle(obj.slot(heap, d.offset()).get(heap)));
                    surviving.push((
                        scope.handle(d.name(heap)),
                        d.flags(),
                        scope.handle(Smi::new(values.len() as i64 - 1)),
                    ));
                }
            }
            (
                parent.kind(),
                scope.handle(parent.prototype.get(heap)),
                surviving,
                values,
            )
        };

        // the parent map is immutable except its transitions slot, so the
        // descriptor/value snapshot stays valid across every retry below
        loop {
            // shared child map: reuse it instead of growing the tree (the
            // initial check, and the loser's convergence point after a
            // lost publication race)
            let existing = receiver
                .as_tagged(heap)
                .map_ref(heap)
                .find_remove_transition(heap, name.as_tagged(heap))
                .map(|m| m.as_handle(scope));
            if let Some(existing) = existing {
                // only this receiver's slots need compacting
                let values = {
                    let anchored: Vec<Tagged<'_, Value>> =
                        values.iter().map(|h| h.as_tagged(heap)).collect();
                    scope.stage(&anchored)
                };
                heap.allocate_token_enter_heap(
                    FixedArray::<Value>::layout_for(values.len()),
                    |token, heap| {
                        let obj = receiver.as_tagged(heap);
                        let slots = token.allocate::<FixedArray>(values);
                        let host = receiver.as_tagged(heap).erase();
                        obj.slots.set(heap, host, slots);
                        Prototype::shape_changed(heap, obj.map_ref(heap));
                        obj.header.map.set(heap, host, existing.as_tagged(heap));
                    },
                );
                return;
            }

            let pairs_len = receiver
                .as_tagged(heap)
                .map_ref(heap)
                .transitions
                .load(heap)
                .map_or(0, |a| a.len());

            let map_layout = Map::layout_for(surviving.len());
            let pairs_layout = WeakFixedArray::<Value>::layout_for(pairs_len + 2);
            let slots_layout = FixedArray::<Value>::layout_for(values.len());
            let total = AllocToken::total_for(&[map_layout, pairs_layout, slots_layout]);
            let values = {
                let anchored: Vec<Tagged<'_, Value>> =
                    values.iter().map(|h| h.as_tagged(heap)).collect();
                scope.stage(&anchored)
            };
            let published = heap.allocate_token_enter_heap(total, |token, heap| {
                let obj = receiver.as_tagged(heap);
                let parent = obj.map_ref(heap);
                // re-derive the publication after the reservation; see
                // Transition::target for the same-length invariant
                let word = parent.transitions.load_word(heap);
                let old = parent.transitions.decode(heap, word);
                if old.map_or(0, |a| a.len()) != pairs_len {
                    token.discard_remaining();
                    return None;
                }
                let child = token.allocate::<Map>(MapInit {
                    kind,
                    value_slot_count: values.len(),
                    descriptors: &surviving,
                    prototype,
                });
                child.pred.set(heap, child.erase(), parent);
                let name_word = name.as_tagged(heap);
                let mut pairs: Vec<Tagged<'_, MaybeWeak<Value>>> =
                    Vec::with_capacity(pairs_len + 2);
                if let Some(old) = old {
                    for entry in old.as_slice().as_chunks::<2>().0 {
                        pairs.push(entry[0].get(heap));
                        pairs.push(entry[1].get(heap));
                    }
                }
                pairs.push(name_word.erase().as_maybe_weak());
                pairs.push(child.erase().as_weak());
                let pairs = token.allocate::<WeakFixedArray>(WeakFixedArrayInit { values: &pairs });
                let slots = token.allocate::<FixedArray>(values);
                match parent
                    .transitions
                    .publish(heap, parent.erase(), word, pairs)
                {
                    Ok(()) => {
                        let host = receiver.as_tagged(heap).erase();
                        obj.slots.set(heap, host, slots);
                        Prototype::shape_changed(heap, obj.map_ref(heap));
                        obj.header.map.set(heap, host, child);
                        Some(())
                    }
                    Err(_) => None,
                }
            });
            if published.is_some() {
                return;
            }
            // lost a race: retry — the find above converges on the
            // winner's shared child map, or re-extends the longer array
        }
    }

    fn define(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        desc: PropertyDescriptor<'_>,
        change: Change,
    ) {
        let flags = desc.flags();
        match desc {
            PropertyDescriptor::Data { value, .. } => {
                let grow = match change {
                    Change::Append => true,
                    Change::Replace { index } => {
                        receiver.as_tagged(heap).map_ref(heap).descriptors()[index]
                            .flags()
                            .is_accessor()
                    }
                };
                Self::target(
                    heap,
                    scope,
                    |heap| receiver.as_tagged(heap).map_ref(heap),
                    name,
                    flags,
                    None,
                    change,
                );
                if grow {
                    Self::grow_slots_and_swap(heap, scope, receiver, name, flags, value);
                } else {
                    let Change::Replace { index } = change else {
                        unreachable!("appends always grow")
                    };
                    Self::write_slot(heap, receiver, index, value);
                    Self::swap_map(heap, receiver, name, flags, None);
                }
            }
            PropertyDescriptor::Accessor { get, set, .. } => {
                Self::target(
                    heap,
                    scope,
                    |heap| receiver.as_tagged(heap).map_ref(heap),
                    name,
                    flags,
                    Some((get, set)),
                    change,
                );
                Self::swap_map(heap, receiver, name, flags, Some((get, set)));
            }
        }
    }

    /// `super.x = v` (ES 15.4.4 PutValue on a super reference): the store
    /// walks the chain starting at the pre-resolved parent link (read
    /// before ToPropertyKey; any of the three prototype shapes) but the
    /// receiver is `this`:
    /// - `Shadow` (JS): inherited writable data properties create an own
    ///   property on the receiver (OrdinarySet's receiver != O path),
    ///   setters run with the receiver, misses define on the receiver
    /// - `WriteThrough` (Self-style): inherited writable data properties
    ///   are written at the holder instead of shadowing
    pub fn super_store_lookup<'a, 's>(
        heap: &'a Heap,
        scope: &'s HandleScope<'_>,
        proto: Option<Tagged<'a, Value>>,
        recv: Tagged<'a, Value>,
        name: Tagged<'a, SlotName>,
        value: Tagged<'a, Value>,
        semantics: StoreSemantics,
    ) -> Result<StoreOutcome<'s>, VmError> {
        if recv.ptr_eq(heap.known().null.as_tagged(heap).erase())
            || recv.ptr_eq(heap.known().undefined.as_tagged(heap).erase())
        {
            return Err(VmError::Type);
        }
        let Some(proto) = proto else {
            // non-object home: no parent chain, define on the receiver
            return super_store_on_receiver(heap, scope, recv, name, value);
        };
        match Lookup::lookup_in_parents(heap, proto, name) {
            Lookup::Data {
                slot,
                holder,
                flags,
                ..
            } => {
                if !flags.is_writable() {
                    return Err(VmError::Type);
                }
                match semantics {
                    StoreSemantics::WriteThrough => {
                        let host = holder.erase();
                        slot.set(heap, host, value);
                        Ok(StoreOutcome::Done)
                    }
                    // receiver (`this`) differs from the holder by
                    // construction: OrdinarySet creates an own property on
                    // the receiver
                    StoreSemantics::Shadow => {
                        super_store_on_receiver(heap, scope, recv, name, value)
                    }
                }
            }
            Lookup::NotFound => super_store_on_receiver(heap, scope, recv, name, value),
            Lookup::Accessor { pair, .. } => {
                let setter = pair.set.get(heap);
                if setter.ptr_eq(heap.known().undefined.as_tagged(heap).erase()) {
                    return Ok(StoreOutcome::Done);
                }
                Ok(StoreOutcome::CallSetter {
                    setter: scope.handle(setter),
                })
            }
        }
    }

    pub fn is_compatible_property_descriptor(
        heap: &Heap,
        extensible: bool,
        desc: &PartialDescriptor<'_>,
        current: Option<&PartialDescriptor<'_>>,
    ) -> bool {
        let Some(current) = current else {
            return extensible;
        };

        if current.configurable != Some(false) {
            return true;
        }
        if desc.configurable == Some(true) {
            return false;
        }
        if let Some(e) = desc.enumerable
            && e != current.enumerable.unwrap_or(false)
        {
            return false;
        }
        // a non-configurable property cannot change kind
        let cur_is_data = current.is_data_descriptor();
        if !desc.is_generic_descriptor() && desc.is_data_descriptor() != cur_is_data {
            return false;
        }
        let undefined = heap.known().undefined.as_tagged(heap).erase();
        if cur_is_data && desc.is_data_descriptor() {
            if current.writable != Some(true) {
                if desc.writable == Some(true) {
                    return false;
                }
                if let Some(v) = desc.value
                    && !Compare::same_value(
                        heap,
                        v.as_tagged(heap),
                        current.value.map_or(undefined, |w| w.as_tagged(heap)),
                    )
                {
                    return false;
                }
            }
        } else if !cur_is_data && desc.is_accessor_descriptor() {
            let is_absent = |h: Option<Handle<'_, Value>>| {
                h.is_none_or(|h| h.as_tagged(heap).ptr_eq(undefined))
            };
            if is_absent(current.get)
                && let Some(g) = desc.get
                && !g.as_tagged(heap).ptr_eq(undefined)
            {
                return false;
            }
            if is_absent(current.set)
                && let Some(s) = desc.set
                && !s.as_tagged(heap).ptr_eq(undefined)
            {
                return false;
            }
        }
        true
    }
}

#[derive(Debug, Copy, Clone)]
pub enum PropertyDescriptor<'s> {
    Data {
        value: Handle<'s, Value>,
        writable: bool,
        enumerable: bool,
        configurable: bool,
    },
    Accessor {
        get: Handle<'s, Value>,
        set: Handle<'s, Value>,
        enumerable: bool,
        configurable: bool,
    },
}

impl<'s> PropertyDescriptor<'s> {
    pub const fn data(value: Handle<'s, Value>) -> Self {
        Self::Data {
            value,
            writable: true,
            enumerable: true,
            configurable: true,
        }
    }

    pub const fn method(value: Handle<'s, Value>) -> Self {
        Self::Data {
            value,
            writable: true,
            enumerable: false,
            configurable: true,
        }
    }

    pub const fn non_enumerable(value: Handle<'s, Value>) -> Self {
        Self::Data {
            value,
            writable: false,
            enumerable: false,
            configurable: true,
        }
    }

    pub const fn flags(self) -> SlotFlags {
        match self {
            Self::Data {
                writable,
                enumerable,
                configurable,
                ..
            } => {
                let mut flags = SlotFlags::VALUE;
                if writable {
                    flags = flags.union(SlotFlags::WRITABLE);
                }
                if enumerable {
                    flags = flags.union(SlotFlags::ENUMERABLE);
                }
                if configurable {
                    flags = flags.union(SlotFlags::CONFIGURABLE);
                }
                flags
            }
            Self::Accessor {
                enumerable,
                configurable,
                ..
            } => {
                let mut flags = SlotFlags::ACCESSOR;
                if enumerable {
                    flags = flags.union(SlotFlags::ENUMERABLE);
                }
                if configurable {
                    flags = flags.union(SlotFlags::CONFIGURABLE);
                }
                flags
            }
        }
    }
}

enum DefineAction<'s> {
    /// Valid, nothing to change.
    Nothing,
    /// data→data with unchanged attributes: write the existing slot.
    WriteDataSlot { value: Handle<'s, Value> },
    /// Replace the descriptor with the new one (data or accessor).
    Redefine { desc: PropertyDescriptor<'s> },
}

impl Object {
    pub fn add_own_property(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        desc: PropertyDescriptor<'_>,
    ) -> Result<bool, VmError> {
        debug_assert!(
            !receiver
                .as_tagged(heap)
                .map_ref(heap)
                .descriptors()
                .iter()
                .any(|d| d.name(heap).ptr_eq(name.as_tagged(heap))),
            "add_own_property requires the name to be absent from the receiver's own map"
        );
        let cond_7 = receiver.as_tagged(heap).is_extendable(heap);
        if !cond_7 {
            return Ok(false);
        }
        // an integer-named own property can shadow an element hole
        if Smi::decode(name.as_tagged(heap).raw()).is_some()
            && receiver.as_tagged(heap).as_ref().is_array(heap)
        {
            Object::promote_holey(heap, scope, &receiver);
        }
        Transition::define(heap, scope, receiver, name, desc, Change::Append);
        Ok(true)
    }

    pub fn add_parent(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        value: Handle<'_, Value>,
    ) -> Result<(), VmError> {
        if !receiver.as_tagged(heap).is_extendable(heap) {
            return Err(VmError::NotExtensible);
        }
        let mut pairs: Vec<Handle<'_, Value>> = Vec::new();
        {
            let base = receiver.as_tagged(heap).map_ref(heap).prototype.get(heap);
            if let Some(existing) = base.get_as::<FixedArray>(heap) {
                for i in 0..existing.len() {
                    pairs.push(scope.handle(existing.at(heap, i)));
                }
            }
        }
        pairs.push(name.erase());
        pairs.push(value);
        let anchored: Vec<Tagged<'_, Value>> = pairs.iter().map(|h| h.as_tagged(heap)).collect();
        let pairs = heap.allocate_handle::<FixedArray>(scope.stage(&anchored), scope);
        Self::set_prototype(heap, scope, receiver, pairs.erase())
    }

    pub fn define_own_property(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        desc: PropertyDescriptor<'_>,
    ) -> Result<bool, VmError> {
        let current = receiver
            .as_tagged(heap)
            .map_ref(heap)
            .descriptors()
            .iter()
            .enumerate()
            .find(|(_, d)| d.name(heap).ptr_eq(name.as_tagged(heap)))
            .map(|(index, d)| (index, d.flags(), scope.handle(d.value.get(heap))));
        let Some((index, cur_flags, cur_desc_value)) = current else {
            return Self::add_own_property(heap, scope, receiver, name, desc);
        };

        let Some(action) = ({
            validate_define(
                heap,
                receiver.as_tagged(heap),
                cur_flags,
                cur_desc_value,
                desc,
            )
        }) else {
            return Ok(false);
        };
        // an integer-named own property can shadow an element hole
        if Smi::decode(name.as_tagged(heap).raw()).is_some()
            && receiver.as_tagged(heap).as_ref().is_array(heap)
        {
            Object::promote_holey(heap, scope, &receiver);
        }
        Self::apply_define(heap, scope, receiver, name, index, action);
        Ok(true)
    }

    pub fn delete_own_property(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        key: Handle<'_, Value>,
    ) -> Result<bool, VmError> {
        let name = 'name: {
            let key = match Lookup::classify_key(heap, key.as_tagged(heap)) {
                Ok(key) => key,
                Err(err) => return Err(err),
            };
            match key {
                Key::Element(i) => {
                    // array elements live in the elements backing store,
                    // outside the descriptors: delete punches a hole
                    let obj = receiver.as_tagged(heap);
                    if obj.as_ref().is_array(heap) {
                        // indices at/past `length` were never own properties
                        if i < obj.as_ref().length()
                            && let Some(elements) = obj.as_ref().elements_array(heap)
                            && i < elements.len()
                        {
                            elements.set(heap, i, heap.known().the_hole.as_tagged(heap).erase());
                            Object::promote_holey(heap, scope, &receiver);
                        }
                        break 'name None;
                    }
                    // other receivers hold numeric keys as named descriptors
                    let smi_name: Tagged<'_, SlotName> = Tagged::from(Smi::new(i as i64));
                    Some(scope.handle(smi_name))
                }
                Key::Name(name) => Some(scope.handle(name)),
            }
        };
        let Some(name) = name else {
            return Ok(true);
        };
        Self::delete_named_property(heap, scope, receiver, name)
    }

    fn delete_named_property(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
    ) -> Result<bool, VmError> {
        // array `length` lives in a dedicated slot outside the
        // descriptors and is non-configurable (ES 10.4.2)
        {
            let cond_8 = receiver
                .as_tagged(heap)
                .as_ref()
                .array_length(heap, name.as_tagged(heap))
                .is_some();
            if cond_8 {
                return Ok(false);
            }
        }
        // OrdinaryDelete: absent → true, non-configurable → false,
        // configurable → remove
        let configurable = receiver
            .as_tagged(heap)
            .map_ref(heap)
            .descriptors()
            .iter()
            .find(|d| d.name(heap).ptr_eq(name.as_tagged(heap)))
            .map(|d| d.flags().is_configurable());
        if configurable != Some(true) {
            return Ok(configurable.is_none());
        }
        Transition::remove_property(heap, scope, receiver, name);
        Ok(true)
    }

    fn apply_define(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        index: usize,
        action: DefineAction,
    ) {
        match action {
            DefineAction::Nothing => {}
            DefineAction::WriteDataSlot { value } => {
                // the in-place data write needs no map change
                Transition::write_slot(heap, receiver, index, value);
            }
            DefineAction::Redefine { desc } => {
                Transition::define(heap, scope, receiver, name, desc, Change::Replace { index })
            }
        }
    }

    /// - `proto` must be:
    /// - arbitrary normal `Object` for JS semantics
    /// - `FixedArray` of objects (Self-style multiple parents),
    /// - the hole sentinel; other values are
    ///   silently ignored (sloppy `__proto__` semantics)
    /// - cycles and non-extensible receivers throw (TODO: make this optional for Self semantics)
    pub fn set_prototype(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        proto: Handle<Value>,
    ) -> Result<(), VmError> {
        let is_null = proto
            .as_tagged(heap)
            .ptr_eq(heap.known().null.as_tagged(heap).erase());
        if !is_null && !{ proto.as_tagged(heap).is_strong_ptr() } {
            // silently ignore non-object prototypes (sloppy-mode semantics)
            return Ok(());
        }

        {
            let cond_9 = receiver
                .as_tagged(heap)
                .map_ref(heap)
                .prototype
                .get(heap)
                .ptr_eq(proto.as_tagged(heap));
            if cond_9 {
                return Ok(());
            }
        }
        {
            let cond_10 = receiver
                .as_tagged(heap)
                .map_ref(heap)
                .kind()
                .is_extendable();
            if !cond_10 {
                return Err(VmError::NotExtensible);
            }
        }

        {
            let null = heap.known().null.as_tagged(heap).erase();
            let this = receiver.as_tagged(heap).erase();
            let start = proto.as_tagged(heap);
            fn walk<'b>(
                heap: &'b Heap,
                null: Tagged<'b, Value>,
                this: Tagged<'b, Value>,
                mut p: Tagged<'b, Value>,
            ) -> Result<(), VmError> {
                while p.is_strong_ptr() && !p.ptr_eq(null) {
                    if p.ptr_eq(this) {
                        return Err(VmError::Type);
                    }
                    let Some(o) = p.as_heap_object() else {
                        break;
                    };
                    p = o.as_ref().map_ref(heap).prototype.get(heap);
                }
                Ok(())
            }
            if let Some(pairs) = start.get_as::<FixedArray>(heap) {
                let mut i = 1;
                while i < pairs.len() {
                    walk(heap, null, this, pairs.at(heap, i))?;
                    i += 2;
                }
            } else {
                walk(heap, null, this, start)?;
            }
            Ok(())
        }?;

        let target = Transition::sibling_target(
            heap,
            scope,
            |heap| receiver.as_tagged(heap).map_ref(heap),
            SiblingChange::Prototype(proto),
        );
        let host = receiver.as_tagged(heap).erase();
        Prototype::shape_changed(heap, receiver.as_tagged(heap).map_ref(heap));
        receiver
            .as_tagged(heap)
            .header
            .map
            .set(heap, host, target.as_tagged(heap));
        Ok(())
    }
}

fn validate_define<'a, 's>(
    heap: &'a Heap,
    receiver: Tagged<'a, Object>,
    cur_flags: SlotFlags,
    cur_desc_value: Handle<'s, Value>,
    desc: PropertyDescriptor<'s>,
) -> Option<DefineAction<'s>> {
    let cur_configurable = cur_flags.is_configurable();

    if cur_flags.is_accessor() {
        if cur_configurable {
            return Some(DefineAction::Redefine { desc });
        }
        let PropertyDescriptor::Accessor {
            get,
            set,
            enumerable,
            configurable,
        } = desc
        else {
            return None;
        };
        if configurable || enumerable != cur_flags.is_enumerable() {
            return None;
        }
        let cur_pair = cur_desc_value
            .as_tagged(heap)
            .get_as::<AccessorPair>(heap)
            .expect("accessor descriptor must hold a pair");
        if !Compare::same_value(heap, get.as_tagged(heap), cur_pair.get.get(heap))
            || !Compare::same_value(heap, set.as_tagged(heap), cur_pair.set.get(heap))
        {
            return None;
        }
        return Some(DefineAction::Nothing);
    }

    if let PropertyDescriptor::Accessor { .. } = desc {
        if !cur_configurable {
            return None;
        }
        return Some(DefineAction::Redefine { desc });
    }

    let PropertyDescriptor::Data {
        value,
        writable,
        enumerable,
        configurable,
    } = desc
    else {
        unreachable!("descriptor kind checked above")
    };
    if !cur_configurable {
        if configurable || enumerable != cur_flags.is_enumerable() {
            return None;
        }
        if !cur_flags.is_writable() {
            if writable {
                return None;
            }

            let offset = Smi::decode(cur_desc_value.as_tagged(heap).raw())
                .expect("data row offset")
                .value() as usize;
            if !Compare::same_value(
                heap,
                value.as_tagged(heap),
                receiver.slot(heap, offset).get(heap),
            ) {
                return None;
            }
            return Some(DefineAction::Nothing);
        }
    }
    if desc.flags() == cur_flags {
        return Some(DefineAction::WriteDataSlot { value });
    }
    Some(DefineAction::Redefine { desc })
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PartialDescriptor<'s> {
    pub value: Option<Handle<'s, Value>>,
    pub get: Option<Handle<'s, Value>>,
    pub set: Option<Handle<'s, Value>>,
    pub writable: Option<bool>,
    pub enumerable: Option<bool>,
    pub configurable: Option<bool>,
}

impl<'s> PartialDescriptor<'s> {
    pub fn value(value: Handle<'s, Value>) -> Self {
        Self {
            value: Some(value),
            ..Self::default()
        }
    }

    /// ES 6.2.6.2 IsDataDescriptor.
    pub fn is_data_descriptor(&self) -> bool {
        self.value.is_some() || self.writable.is_some()
    }

    /// ES 6.2.6.1 IsAccessorDescriptor.
    pub fn is_accessor_descriptor(&self) -> bool {
        self.get.is_some() || self.set.is_some()
    }

    /// ES 6.2.6.3 IsGenericDescriptor.
    pub fn is_generic_descriptor(&self) -> bool {
        !self.is_data_descriptor() && !self.is_accessor_descriptor()
    }

    pub fn complete_against(
        &self,
        undefined: Handle<'s, Value>,
        current: Option<&PropertyDescriptor<'s>>,
    ) -> PropertyDescriptor<'s> {
        // accessor if either side (desc first, then current) says so
        let accessor = self.is_accessor_descriptor()
            || !self.is_data_descriptor()
                && current.is_some_and(|c| matches!(c, PropertyDescriptor::Accessor { .. }));

        if accessor {
            let cur = match current {
                Some(PropertyDescriptor::Accessor { get, set, .. }) => (Some(*get), Some(*set)),
                _ => (None, None),
            };
            PropertyDescriptor::Accessor {
                get: self.get.or(cur.0).unwrap_or(undefined),
                set: self.set.or(cur.1).unwrap_or(undefined),
                enumerable: self
                    .enumerable
                    .or(current.map(c_enumumerable))
                    .unwrap_or(false),
                configurable: self
                    .configurable
                    .or(current.map(c_configurable))
                    .unwrap_or(false),
            }
        } else {
            let cur_value = match current {
                Some(PropertyDescriptor::Data { value, .. }) => Some(*value),
                _ => None,
            };
            PropertyDescriptor::Data {
                value: self.value.or(cur_value).unwrap_or(undefined),
                writable: self
                    .writable
                    .or(current.and_then(c_writable))
                    .unwrap_or(false),
                enumerable: self
                    .enumerable
                    .or(current.map(c_enumumerable))
                    .unwrap_or(false),
                configurable: self
                    .configurable
                    .or(current.map(c_configurable))
                    .unwrap_or(false),
            }
        }
    }
}

impl<'s> From<&PropertyDescriptor<'s>> for PartialDescriptor<'s> {
    fn from(d: &PropertyDescriptor<'s>) -> Self {
        match *d {
            PropertyDescriptor::Data {
                value,
                writable,
                enumerable,
                configurable,
            } => Self {
                value: Some(value),
                get: None,
                set: None,
                writable: Some(writable),
                enumerable: Some(enumerable),
                configurable: Some(configurable),
            },
            PropertyDescriptor::Accessor {
                get,
                set,
                enumerable,
                configurable,
            } => Self {
                value: None,
                get: Some(get),
                set: Some(set),
                writable: None,
                enumerable: Some(enumerable),
                configurable: Some(configurable),
            },
        }
    }
}

fn c_enumumerable(d: &PropertyDescriptor<'_>) -> bool {
    match d {
        PropertyDescriptor::Data { enumerable, .. }
        | PropertyDescriptor::Accessor { enumerable, .. } => *enumerable,
    }
}

fn c_configurable(d: &PropertyDescriptor<'_>) -> bool {
    match d {
        PropertyDescriptor::Data { configurable, .. }
        | PropertyDescriptor::Accessor { configurable, .. } => *configurable,
    }
}

fn c_writable(d: &PropertyDescriptor<'_>) -> Option<bool> {
    match d {
        PropertyDescriptor::Data { writable, .. } => Some(*writable),
        PropertyDescriptor::Accessor { .. } => None,
    }
}
