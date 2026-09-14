use std::sync::{Arc, Mutex, MutexGuard};

use core::alloc::Layout;

use crate::{
    AccessorPair, AllocToken, Compare, FixedArray, Handle, HandleScope, Heap, HeapObject, HeapRef,
    Key, Lookup, Map, MapInit, NoGc, Object, SlotFlags, SlotName, Smi, Tagged, Value, VmError,
    classify_key, lookup_in_parents,
};

/// Serializes map-transition tree mutations across threads. VM-internal:
/// to the heap, transition arrays are ordinary traced objects.
#[derive(Clone)]
pub struct TransitionLock(Arc<Mutex<()>>);

impl TransitionLock {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(())))
    }

    pub fn acquire(&self) -> TransitionGuard<'_> {
        TransitionGuard {
            _guard: self.0.lock().unwrap(),
        }
    }
}

/// Proof that the heap's transition lock is held.
pub struct TransitionGuard<'a> {
    _guard: MutexGuard<'a, ()>,
}

/// Store semantics:
/// - Self-style writes through to an inherited writable slot
/// - JS-style shadows it with a new own property on the receiver.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StoreSemantics {
    WriteThrough,
    Shadow,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StoreOutcome {
    Done,
    Transition { receiver: Value, name: SlotName },
    CallSetter { setter: Value },
}

impl Value {
    pub fn store_lookup<'a>(
        &self,
        nogc: &'a NoGc<'a>,
        name: SlotName,
        value: Value,
        semantics: StoreSemantics,
    ) -> Result<StoreOutcome, VmError> {
        // TODO(strict-mode): take the active function's language mode and
        // distinguish throwing strict failures from ignored sloppy failures.
        let receiver = *self;
        // null/undefined have no [[Prototype]]: property access throws
        if receiver == nogc.known().null.value() || receiver == nogc.known().undefined.value() {
            return Err(VmError::Type);
        }
        // non-receiver heap values (VMStrings, Symbols, Floats, ...) and
        // smis are not property stores' targets: sloppy-mode stores onto
        // primitives are silently ignored (strict throws — deferred with
        // the other language-mode TODOs)
        if let Some(obj) = receiver.as_heap_object(nogc) {
            let kind = obj.as_ref().header.map.heap_ref(nogc).kind().kind();
            if !Object::matches_kind(kind) {
                return Ok(StoreOutcome::Done);
            }
        } else {
            return Ok(StoreOutcome::Done);
        }
        match receiver.lookup(nogc, name) {
            Lookup::Data {
                slot,
                holder,
                flags,
                ..
            } => {
                if !flags.is_writable() {
                    return Err(VmError::Type);
                }
                let host = holder.as_ref().erase();
                if semantics == StoreSemantics::Shadow && host != receiver {
                    // inherited writable data property: JS creates an own
                    // property on the receiver
                    if !receiver.is_strong_ptr() {
                        return Err(VmError::Type);
                    }
                    return Ok(StoreOutcome::Transition { receiver, name });
                }
                slot.set(nogc, host, value);
                Ok(StoreOutcome::Done)
            }
            Lookup::NotFound => {
                // adding a property requires a heap receiver
                if !receiver.is_strong_ptr() {
                    return Err(VmError::Type);
                }
                Ok(StoreOutcome::Transition { receiver, name })
            }
            Lookup::Accessor { pair, .. } => {
                let setter = pair.set.inner();
                // no setter (undefined sentinel): sloppy-mode writes to a
                // setter-less accessor are silently ignored
                if setter == nogc.known().undefined.value() {
                    return Ok(StoreOutcome::Done);
                }
                Ok(StoreOutcome::CallSetter { setter })
            }
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
pub fn super_store_lookup<'a>(
    nogc: &'a NoGc<'a>,
    proto: Option<Value>,
    recv: Value,
    name: SlotName,
    value: Value,
    semantics: StoreSemantics,
) -> Result<StoreOutcome, VmError> {
    if recv == nogc.known().null.value() || recv == nogc.known().undefined.value() {
        return Err(VmError::Type);
    }
    let Some(proto) = proto else {
        // non-object home: no parent chain, define on the receiver
        return super_store_on_receiver(nogc, recv, name, value);
    };
    match lookup_in_parents(nogc, proto, name) {
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
                    let host = holder.as_ref().erase();
                    slot.set(nogc, host, value);
                    Ok(StoreOutcome::Done)
                }
                // receiver (`this`) differs from the holder by
                // construction: OrdinarySet creates an own property on
                // the receiver
                StoreSemantics::Shadow => super_store_on_receiver(nogc, recv, name, value),
            }
        }
        Lookup::NotFound => super_store_on_receiver(nogc, recv, name, value),
        Lookup::Accessor { pair, .. } => {
            let setter = pair.set.inner();
            if setter == nogc.known().undefined.value() {
                return Ok(StoreOutcome::Done);
            }
            Ok(StoreOutcome::CallSetter { setter })
        }
    }
}

/// OrdinarySet's final receiver step (ES 9.1.9.2 step 3): the parent walk
/// resolved to a writable data property (or exhausted the chain, which
/// implies the default writable descriptor), so the write lands on the
/// receiver — a writable data property the receiver already owns is
/// overwritten in place, and only a true miss defines a fresh own
/// property. An own accessor (or non-writable own data property)
/// rejects the `{value}` define: a TypeError at these strict sites.
fn super_store_on_receiver<'a>(
    nogc: &'a NoGc<'a>,
    recv: Value,
    name: SlotName,
    value: Value,
) -> Result<StoreOutcome, VmError> {
    if !recv.is_strong_ptr() {
        return Err(VmError::Type);
    }
    match recv.lookup(nogc, name) {
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
            slot.set(nogc, recv, value);
            Ok(StoreOutcome::Done)
        }
        // own accessor: Receiver.[[DefineOwnProperty]]({value}) on an
        // accessor is an incompatible change (ES 9.1.9.2 step 3.d.i)
        Lookup::Accessor { holder, .. } if holder.as_ref().erase() == recv => Err(VmError::Type),
        // not owned by the receiver: define a fresh own property
        _ => Ok(StoreOutcome::Transition {
            receiver: recv,
            name,
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
    pub fn target(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        parent: impl for<'a> Fn(&'a NoGc<'a>) -> HeapRef<'a, Map>,
        name: Handle<SlotName>,
        flags: SlotFlags,
        pair: Option<(Handle<'_, Value>, Handle<'_, Value>)>,
        change: Change,
    ) -> Tagged<Map> {
        debug_assert_eq!(flags.is_accessor(), pair.is_some());
        let lock = heap.transition_lock();
        let guard = lock.acquire();
        let pair_values = pair.map(|(get, set)| (get.value(), set.value()));

        let (kind, descriptor_count, value_slot_count, pairs_len, prototype, old_row) = {
            let nogc = heap.guard();
            let parent_ref = parent(&nogc);
            if let Some(target) =
                parent_ref.find_transition_locked(&nogc, name.into(), flags, pair_values, &guard)
            {
                return target.into_tagged();
            }
            let old_row = match change {
                Change::Append => None,
                Change::Replace { index } => {
                    let d = &parent_ref.descriptors()[index];
                    Some((d.flags(), d.value.inner()))
                }
            };
            (
                parent_ref.kind(),
                parent_ref.descriptor_count(),
                parent_ref.value_slot_count(),
                parent_ref
                    .transitions
                    .heap_ref(&nogc)
                    .map_or(0, |a| a.len()),
                parent_ref.prototype.inner(),
                old_row,
            )
        };
        let prototype = scope.handle(prototype);

        let grow = !flags.is_accessor()
            && match change {
                Change::Append => true,
                Change::Replace { .. } => old_row.expect("replace row").0.is_accessor(),
            };
        let row_offset = match change {
            Change::Replace { .. } if !grow && !flags.is_accessor() => {
                Smi::decode(old_row.expect("replace row").1)
                    .expect("data row offset")
                    .value() as usize
            }
            _ => value_slot_count,
        };
        let appends = usize::from(matches!(change, Change::Append));

        let map_layout = Map::layout_for(descriptor_count + appends);
        let pairs_layout = FixedArray::layout_for(pairs_len + 2);
        let total = match pair.is_some() {
            true => {
                AllocToken::total_for(&[Layout::new::<AccessorPair>(), map_layout, pairs_layout])
            }
            false => AllocToken::total_for(&[map_layout, pairs_layout]),
        };

        heap.allocate_token_enter_nogc(total, |token, nogc| {
            let parent_ref = parent(nogc);
            let row_value = match pair {
                Some((get, set)) => token
                    .allocate::<AccessorPair>((get.value(), set.value()))
                    .erase(),
                None => Smi::new(row_offset as i64).encode(),
            };

            let mut descriptors: Vec<(SlotName, SlotFlags, Value)> = parent_ref
                .descriptors()
                .iter()
                .map(|d| (d.name(), d.flags(), d.value.inner()))
                .collect();
            match change {
                Change::Append => descriptors.push((name.into(), flags, row_value)),
                Change::Replace { index } => descriptors[index] = (name.into(), flags, row_value),
            }
            let child = token.allocate::<Map>(MapInit {
                kind,
                value_slot_count: value_slot_count + usize::from(grow),
                descriptors: &descriptors,
                prototype,
            });

            let mut pairs: Vec<Value> = Vec::with_capacity(pairs_len + 2);
            if let Some(old) = parent_ref.transitions.heap_ref(nogc) {
                pairs.extend(old.as_slice().iter().map(|slot| slot.inner()));
            }
            pairs.push(name.value());
            pairs.push(child.erase());
            let pairs = token.allocate::<FixedArray>(&pairs);
            parent_ref
                .transitions
                .set(nogc, parent_ref.erase(), pairs.into_tagged());

            child.into_tagged()
        })
    }

    fn grow_slots_and_swap(
        heap: &mut Heap,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        flags: SlotFlags,
        value: Handle<Value>,
    ) {
        let slot_count = {
            let nogc = heap.guard();
            receiver.heap_ref(&nogc).map_ref(&nogc).value_slot_count() + 1
        };
        heap.allocate_token_enter_nogc(FixedArray::layout_for(slot_count), |token, nogc| {
            let receiver_ref = receiver.heap_ref(nogc);
            let target = receiver_ref
                .map_ref(nogc)
                .find_transition(nogc, name.into(), flags, None)
                .expect("transition recorded above");
            let mut values: Vec<Value> = Vec::with_capacity(slot_count);
            values.extend(
                receiver_ref
                    .slots
                    .heap_ref(nogc)
                    .as_slice()
                    .iter()
                    .map(|slot| slot.inner()),
            );
            values.push(value.value());
            debug_assert_eq!(values.len(), slot_count, "slot count desynced from map");
            let slots = token.allocate::<FixedArray>(&values);
            let host = receiver.value();
            receiver_ref.slots.set(nogc, host, slots.into_tagged());
            receiver_ref
                .header
                .map
                .set(nogc, host, target.into_tagged());
        });
    }

    fn swap_map(
        heap: &mut Heap,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        flags: SlotFlags,
        pair: Option<(Value, Value)>,
    ) {
        let nogc = heap.guard();
        let receiver_ref = receiver.heap_ref(&nogc);
        let target = receiver_ref
            .map_ref(&nogc)
            .find_transition(&nogc, name.into(), flags, pair)
            .expect("transition recorded above");
        receiver_ref
            .header
            .map
            .set(&nogc, receiver.value(), target.into_tagged());
    }

    fn write_slot(heap: &mut Heap, receiver: Handle<Object>, index: usize, value: Value) {
        let nogc = heap.guard();
        let offset = receiver.heap_ref(&nogc).map_ref(&nogc).descriptors()[index].offset();
        receiver
            .heap_ref(&nogc)
            .slot(&nogc, offset)
            .set(&nogc, receiver.value(), value);
    }

    /// Remove the own configurable property `name` (OrdinaryDelete
    /// step 4, ES 10.1.10.1): `receiver` migrates to a child map that
    /// lacks the descriptor and its slots compact. The child is shared
    /// through the transition tree like adds and redefines, so
    /// same-shaped deletions converge on one map.
    ///
    /// Slots below the descriptors' region survive untouched (they are
    /// structural: function info and context); slots orphaned by
    /// data→accessor redefines — no descriptor references them anymore —
    /// are dropped.
    ///
    /// The caller has verified the descriptor exists and is configurable.
    pub fn remove_property(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
    ) {
        let lock = heap.transition_lock();
        let guard = lock.acquire();

        let (existing, kind, prototype, surviving, values, pairs_len) = {
            let nogc = heap.guard();
            let obj = receiver.heap_ref(&nogc);
            let parent = obj.map_ref(&nogc);
            let descriptors = parent.descriptors();
            let index = descriptors
                .iter()
                .position(|d| d.name() == name.into())
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
            let existing = parent
                .find_remove_transition_locked(&nogc, name.into(), &guard)
                .map(|m| m.into_tagged());
            let mut surviving: Vec<(SlotName, SlotFlags, Value)> =
                Vec::with_capacity(descriptors.len() - 1);
            let mut values: Vec<Value> = obj.slots.heap_ref(&nogc).as_slice()[..base]
                .iter()
                .map(|slot| slot.inner())
                .collect();
            for (i, d) in descriptors.iter().enumerate() {
                if i == index {
                    continue;
                }
                if d.flags().is_accessor() {
                    // accessors embed their pair in the descriptor row
                    surviving.push((d.name(), d.flags(), d.value.inner()));
                } else {
                    // data rows re-dense their offsets; the value rides
                    // along in slot order
                    values.push(obj.slot(&nogc, d.offset()).inner());
                    surviving.push((
                        d.name(),
                        d.flags(),
                        Smi::new(values.len() as i64 - 1).encode(),
                    ));
                }
            }
            let pairs_len = parent.transitions.heap_ref(&nogc).map_or(0, |a| a.len());
            (
                existing,
                parent.kind(),
                parent.prototype.inner(),
                surviving,
                values,
                pairs_len,
            )
        };
        let prototype = scope.handle(prototype);

        if let Some(existing) = existing {
            // shared child map: only this receiver's slots need compacting
            heap.allocate_token_enter_nogc(FixedArray::layout_for(values.len()), |token, nogc| {
                let obj = receiver.heap_ref(nogc);
                let slots = token.allocate::<FixedArray>(&values);
                let host = receiver.value();
                obj.slots.set(nogc, host, slots.into_tagged());
                obj.header.map.set(nogc, host, existing);
            });
            return;
        }

        let map_layout = Map::layout_for(surviving.len());
        let pairs_layout = FixedArray::layout_for(pairs_len + 2);
        let slots_layout = FixedArray::layout_for(values.len());
        let total = AllocToken::total_for(&[map_layout, pairs_layout, slots_layout]);
        heap.allocate_token_enter_nogc(total, |token, nogc| {
            let obj = receiver.heap_ref(nogc);
            let parent = obj.map_ref(nogc);
            let child = token.allocate::<Map>(MapInit {
                kind,
                value_slot_count: values.len(),
                descriptors: &surviving,
                prototype,
            });
            let mut pairs: Vec<Value> = Vec::with_capacity(pairs_len + 2);
            if let Some(old) = parent.transitions.heap_ref(nogc) {
                pairs.extend(old.as_slice().iter().map(|slot| slot.inner()));
            }
            pairs.push(name.value());
            pairs.push(child.erase());
            let pairs = token.allocate::<FixedArray>(&pairs);
            parent
                .transitions
                .set(nogc, parent.erase(), pairs.into_tagged());
            let slots = token.allocate::<FixedArray>(&values);
            let host = receiver.value();
            obj.slots.set(nogc, host, slots.into_tagged());
            obj.header.map.set(nogc, host, child.into_tagged());
        });
    }

    fn define(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        desc: PropertyDescriptor,
        change: Change,
    ) {
        let flags = desc.flags();
        match desc {
            PropertyDescriptor::Data { value, .. } => {
                let value = scope.handle(value);

                let grow = match change {
                    Change::Append => true,
                    Change::Replace { index } => {
                        let nogc = heap.guard();
                        receiver.heap_ref(&nogc).map_ref(&nogc).descriptors()[index]
                            .flags()
                            .is_accessor()
                    }
                };
                Self::target(
                    heap,
                    scope,
                    |nogc| receiver.heap_ref(nogc).map_ref(nogc),
                    name,
                    flags,
                    None,
                    change,
                );
                if grow {
                    Self::grow_slots_and_swap(heap, receiver, name, flags, value);
                } else {
                    let Change::Replace { index } = change else {
                        unreachable!("appends always grow")
                    };
                    Self::write_slot(heap, receiver, index, value.value());
                    Self::swap_map(heap, receiver, name, flags, None);
                }
            }
            PropertyDescriptor::Accessor { get, set, .. } => {
                let get = scope.handle(get);
                let set = scope.handle(set);
                Self::target(
                    heap,
                    scope,
                    |nogc| receiver.heap_ref(nogc).map_ref(nogc),
                    name,
                    flags,
                    Some((get, set)),
                    change,
                );
                Self::swap_map(
                    heap,
                    receiver,
                    name,
                    flags,
                    Some((get.value(), set.value())),
                );
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PropertyDescriptor {
    Data {
        value: Value,
        writable: bool,
        enumerable: bool,
        configurable: bool,
    },
    Accessor {
        get: Value,
        set: Value,
        enumerable: bool,
        configurable: bool,
    },
}

impl PropertyDescriptor {
    pub const fn data(value: Value) -> Self {
        Self::Data {
            value,
            writable: true,
            enumerable: true,
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

enum DefineAction {
    /// Valid, nothing to change.
    Nothing,
    /// data→data with unchanged attributes: write the existing slot.
    WriteDataSlot { value: Value },
    /// Replace the descriptor with the new one (data or accessor).
    Redefine { desc: PropertyDescriptor },
}

impl Object {
    pub fn add_own_property(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        desc: PropertyDescriptor,
    ) -> Result<bool, VmError> {
        {
            let nogc = heap.guard();
            debug_assert!(
                !receiver
                    .heap_ref(&nogc)
                    .map_ref(&nogc)
                    .descriptors()
                    .iter()
                    .any(|d| d.name() == name.into()),
                "add_own_property requires the name to be absent from the receiver's own map"
            );
            if !receiver.heap_ref(&nogc).is_extendable(&nogc) {
                return Ok(false);
            }
        }
        Transition::define(heap, scope, receiver, name, desc, Change::Append);
        Ok(true)
    }

    pub fn define_own_property(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        desc: PropertyDescriptor,
    ) -> Result<bool, VmError> {
        let nogc = heap.guard();
        let current = receiver
            .heap_ref(&nogc)
            .map_ref(&nogc)
            .descriptors()
            .iter()
            .enumerate()
            .find(|(_, d)| d.name() == name.into())
            .map(|(index, d)| (index, d.flags(), d.value.inner()));
        let Some((index, cur_flags, cur_desc_value)) = current else {
            return Self::add_own_property(heap, scope, receiver, name, desc);
        };

        let Some(action) = validate_define(
            &nogc,
            receiver.heap_ref(&nogc),
            cur_flags,
            cur_desc_value,
            desc,
        ) else {
            return Ok(false);
        };
        Self::apply_define(heap, scope, receiver, name, index, action);
        Ok(true)
    }

    pub fn add_own_property_values(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Value,
        name: SlotName,
        desc: PropertyDescriptor,
    ) -> Result<bool, VmError> {
        let (receiver, name) = root_define_inputs(scope, receiver, name);
        Self::add_own_property(heap, scope, receiver, name, desc)
    }

    pub fn define_own_property_values(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Value,
        name: SlotName,
        desc: PropertyDescriptor,
    ) -> Result<bool, VmError> {
        let (receiver, name) = root_define_inputs(scope, receiver, name);
        Self::define_own_property(heap, scope, receiver, name, desc)
    }

    /// `[[Delete]]` for ordinary and array-exotic objects (ES 10.1.10.1
    /// OrdinaryDelete): absent properties and punched holes delete as
    /// `true`, non-configurable ones as `false`, configurable ones are
    /// removed. The caller has performed ToObject/ToPropertyKey.
    pub fn delete_own_property(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        key: Value,
    ) -> Result<bool, VmError> {
        match classify_key(&heap.guard(), key)? {
            Key::Element(i) => {
                // array elements live in the elements backing store,
                // outside the descriptors: delete punches a hole
                let is_array = heap.no_gc(|nogc| receiver.heap_ref(nogc).as_ref().is_array(nogc));
                if is_array {
                    heap.no_gc(|nogc| {
                        let obj = receiver.heap_ref(nogc);
                        // indices at/past `length` were never own properties
                        if i < obj.as_ref().length()
                            && let Some(elements) = obj.as_ref().elements_array(nogc)
                            && i < elements.len()
                        {
                            elements.set(nogc, i, nogc.known().the_hole.value());
                        }
                        Ok(())
                    })?;
                    return Ok(true);
                }
                // other receivers hold numeric keys as named descriptors
                let name = SlotName::from(Tagged::from_smi(Smi::new(i as i64)));
                Self::delete_named_property(heap, scope, receiver, name)
            }
            Key::Name(name) => Self::delete_named_property(heap, scope, receiver, name),
        }
    }

    fn delete_named_property(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: SlotName,
    ) -> Result<bool, VmError> {
        // array `length` lives in a dedicated slot outside the
        // descriptors and is non-configurable (ES 10.4.2)
        {
            let nogc = heap.guard();
            if receiver
                .heap_ref(&nogc)
                .as_ref()
                .array_length(&nogc, name)
                .is_some()
            {
                return Ok(false);
            }
        }
        // OrdinaryDelete: absent → true, non-configurable → false,
        // configurable → remove
        let configurable = heap.no_gc(|nogc| {
            receiver
                .heap_ref(nogc)
                .map_ref(nogc)
                .descriptors()
                .iter()
                .find(|d| d.name() == name)
                .map(|d| d.flags().is_configurable())
        });
        if configurable != Some(true) {
            return Ok(configurable.is_none());
        }
        let name = scope.handle(name.tagged());
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
        receiver: Value,
        proto: Value,
    ) -> Result<(), VmError> {
        let receiver_handle = scope.cast::<Object>(receiver).ok_or(VmError::Type)?;

        let is_null = proto == heap.known().null.value();
        if !is_null && !proto.is_strong_ptr() {
            // silently ignore non-object prototypes (sloppy-mode semantics)
            return Ok(());
        }

        {
            let nogc = heap.guard();
            let map = receiver_handle.heap_ref(&nogc).map_ref(&nogc);
            if map.prototype.inner() == proto {
                return Ok(());
            }
            if !map.kind().is_extendable() {
                return Err(VmError::NotExtensible);
            }
        }

        // cycle check: the receiver must not appear in any proposed chain
        // (FixedArray prototypes contribute one chain per element)
        {
            let nogc = heap.guard();
            let walk = |p: Value| -> Result<(), VmError> {
                let mut p = p;
                while p.is_strong_ptr() && p != nogc.known().null.value() {
                    if p == receiver {
                        return Err(VmError::Type);
                    }
                    let Some(o) = p.as_heap_object(&nogc) else {
                        break;
                    };
                    p = o.as_ref().map_ref(&nogc).prototype.inner();
                }
                Ok(())
            };
            if let Some(parents) = proto.get_as::<FixedArray>(&nogc) {
                for i in 0..parents.len() {
                    walk(parents.at(i))?;
                }
            } else {
                walk(proto)?;
            }
        }

        let proto_handle = scope.handle(proto);

        let descriptor_count = {
            let nogc = heap.guard();
            receiver_handle
                .heap_ref(&nogc)
                .map_ref(&nogc)
                .descriptor_count()
        };

        heap.allocate_token_enter_nogc(Map::layout_for(descriptor_count), |token, nogc| {
            let obj = receiver_handle.heap_ref(nogc);
            let map = obj.map_ref(nogc);
            let kind = map.kind();
            let value_slot_count = map.value_slot_count();
            let descriptors: Vec<(SlotName, SlotFlags, Value)> = map
                .descriptors()
                .iter()
                .map(|d| (d.name(), d.flags(), d.value.inner()))
                .collect();
            let new_map = token.allocate::<Map>(MapInit {
                kind,
                value_slot_count,
                descriptors: &descriptors,
                prototype: proto_handle,
            });
            let host = receiver_handle.value();
            obj.header.map.set(nogc, host, new_map.into_tagged());
            Ok(())
        })
    }
}

fn root_define_inputs<'s>(
    scope: &'s HandleScope<'_>,
    receiver: Value,
    name: SlotName,
) -> (Handle<'s, Object>, Handle<'s, SlotName>) {
    // store paths only reach here for object receivers (lookup dispatch)
    let receiver = scope
        .cast::<Object>(receiver)
        .expect("store receiver is an object");
    let name = scope.handle(name.tagged());
    (receiver, name)
}

fn validate_define<'a>(
    nogc: &'a NoGc<'a>,
    receiver: HeapRef<'a, Object>,
    cur_flags: SlotFlags,
    cur_desc_value: Value,
    desc: PropertyDescriptor,
) -> Option<DefineAction> {
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
            .get_as::<AccessorPair>(nogc)
            .expect("accessor descriptor must hold a pair");
        if !Compare::same_value(nogc, get, cur_pair.get.inner())
            || !Compare::same_value(nogc, set, cur_pair.set.inner())
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

            let offset = cur_desc_value.to_i64().unwrap() as usize;
            if !Compare::same_value(nogc, value, receiver.slot(nogc, offset).inner()) {
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
