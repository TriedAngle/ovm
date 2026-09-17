use std::sync::{Arc, Mutex, MutexGuard};

use core::alloc::Layout;

use crate::{
    AccessorPair, AllocToken, Compare, FixedArray, Handle, HandleScope, Heap, HeapObject, HeapRef,
    Key, Lookup, Map, MapInit, Object, SlotFlags, SlotName, Smi, Tagged, Value, VmError,
    classify_key, lookup_in_parents,
};

/// Serializes map-transition tree mutations across threads. VM-internal:
/// to the heap, transition arrays are ordinary traced objects.
#[derive(Clone)]
pub struct TransitionLock(Arc<Mutex<()>>);

impl Default for TransitionLock {
    fn default() -> Self {
        Self::new()
    }
}

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

impl<'a> Tagged<'a, Value> {
    pub fn store_lookup(
        self,
        heap: &'a Heap,
        name: SlotName,
        value: Tagged<'a, Value>,
        semantics: StoreSemantics,
    ) -> Result<StoreOutcome, VmError> {
        // TODO(strict-mode): take the active function's language mode and
        // distinguish throwing strict failures from ignored sloppy failures.
        let receiver = self.erase();
        // null/undefined have no [[Prototype]]: property access throws
        if receiver == heap.known().null.as_tagged(heap).erase()
            || receiver == heap.known().undefined.as_tagged(heap).erase()
        {
            return Err(VmError::Type);
        }
        // non-receiver heap values (VMStrings, Symbols, Floats, ...) and
        // smis are not property stores' targets: sloppy-mode stores onto
        // primitives are silently ignored (strict throws — deferred with
        // the other language-mode TODOs)
        if let Some(obj) = self.as_heap_object() {
            let kind = obj.as_ref().header.map.heap_ref(heap).kind().kind();
            if !Object::matches_kind(kind) {
                return Ok(StoreOutcome::Done);
            }
        } else {
            return Ok(StoreOutcome::Done);
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
                let host = holder.as_ref().erase();
                if semantics == StoreSemantics::Shadow && host != receiver {
                    // inherited writable data property: JS creates an own
                    // property on the receiver
                    if !receiver.is_strong_ptr() {
                        return Err(VmError::Type);
                    }
                    return Ok(StoreOutcome::Transition { receiver, name });
                }
                slot.set(heap, host, value);
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
                let setter = pair.set.get(heap).erase();
                // no setter (undefined sentinel): sloppy-mode writes to a
                // setter-less accessor are silently ignored
                if setter == heap.known().undefined.as_tagged(heap).erase() {
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
    heap: &'a Heap,
    proto: Option<Tagged<'a, Value>>,
    recv: Tagged<'a, Value>,
    name: SlotName,
    value: Tagged<'a, Value>,
    semantics: StoreSemantics,
) -> Result<StoreOutcome, VmError> {
    if recv.erase() == heap.known().null.as_tagged(heap).erase()
        || recv.erase() == heap.known().undefined.as_tagged(heap).erase()
    {
        return Err(VmError::Type);
    }
    let Some(proto) = proto else {
        // non-object home: no parent chain, define on the receiver
        return super_store_on_receiver(heap, recv, name, value);
    };
    match lookup_in_parents(heap, proto, name) {
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
                    slot.set(heap, host, value);
                    Ok(StoreOutcome::Done)
                }
                // receiver (`this`) differs from the holder by
                // construction: OrdinarySet creates an own property on
                // the receiver
                StoreSemantics::Shadow => super_store_on_receiver(heap, recv, name, value),
            }
        }
        Lookup::NotFound => super_store_on_receiver(heap, recv, name, value),
        Lookup::Accessor { pair, .. } => {
            let setter = pair.set.get(heap).erase();
            if setter == heap.known().undefined.as_tagged(heap).erase() {
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
    heap: &'a Heap,
    recv: Tagged<'a, Value>,
    name: SlotName,
    value: Tagged<'a, Value>,
) -> Result<StoreOutcome, VmError> {
    if !recv.is_strong_ptr() {
        return Err(VmError::Type);
    }
    match recv.lookup(heap, name) {
        // already owned (the nearest hit is the receiver itself, not an
        // inherited one): overwrite the slot instead of re-adding it
        Lookup::Data {
            holder,
            slot,
            flags,
            ..
        } if holder.as_ref().erase() == recv.erase() => {
            if !flags.is_writable() {
                return Err(VmError::Type);
            }
            slot.set(heap, recv.erase(), value);
            Ok(StoreOutcome::Done)
        }
        // own accessor: Receiver.[[DefineOwnProperty]]({value}) on an
        // accessor is an incompatible change (ES 9.1.9.2 step 3.d.i)
        Lookup::Accessor { holder, .. } if holder.as_ref().erase() == recv.erase() => {
            Err(VmError::Type)
        }
        // not owned by the receiver: define a fresh own property
        _ => Ok(StoreOutcome::Transition {
            receiver: recv.erase(),
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
    pub fn target<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        parent: impl for<'a> Fn(&'a Heap) -> HeapRef<'a, Map>,
        name: Handle<SlotName>,
        flags: SlotFlags,
        pair: Option<(Handle<'_, Value>, Handle<'_, Value>)>,
        change: Change,
    ) -> Handle<'s, Map> {
        debug_assert_eq!(flags.is_accessor(), pair.is_some());
        let lock = heap.transition_lock();
        let guard = lock.acquire();

        // shared child map: reuse it instead of growing the tree
        if let Some(target) = heap.no_gc(|heap| {
            let pair_values =
                pair.map(|(get, set)| (get.as_tagged(heap).erase(), set.as_tagged(heap).erase()));
            parent(heap)
                .find_transition_locked(
                    heap,
                    SlotName::from_value(unsafe { name.read_unchecked() }),
                    flags,
                    pair_values,
                    &guard,
                )
                .map(|m| m.into_handle(scope))
        }) {
            return target;
        }

        let (kind, descriptor_count, value_slot_count, pairs_len, prototype, old_row) =
            heap.no_gc(|heap| {
                let parent_ref = parent(heap);
                let old_row = match change {
                    Change::Append => None,
                    Change::Replace { index } => {
                        let d = &parent_ref.descriptors()[index];
                        Some((d.flags(), d.value.get(heap).erase()))
                    }
                };
                (
                    parent_ref.kind(),
                    parent_ref.descriptor_count(),
                    parent_ref.value_slot_count(),
                    parent_ref.transitions.heap_ref(heap).map_or(0, |a| a.len()),
                    scope.handle(parent_ref.prototype.get(heap)),
                    old_row,
                )
            });

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

        heap.allocate_token_enter_heap(total, |token, heap| {
            let parent_ref = parent(heap);
            let name_word = SlotName::from_value(unsafe { name.read_unchecked() });
            let row_value = match pair {
                Some((get, set)) => {
                    scope.handle(token.allocate::<AccessorPair>((get, set)).erase_type())
                }
                None => scope.handle(Smi::new(row_offset as i64)),
            };

            let mut descriptors: Vec<(SlotName, SlotFlags, Handle<'_, Value>)> = parent_ref
                .descriptors()
                .iter()
                .map(|d| (d.name(), d.flags(), scope.handle(d.value.get(heap))))
                .collect();
            match change {
                Change::Append => descriptors.push((name_word, flags, row_value)),
                Change::Replace { index } => descriptors[index] = (name_word, flags, row_value),
            }
            let child = token.allocate::<Map>(MapInit {
                kind,
                value_slot_count: value_slot_count + usize::from(grow),
                descriptors: &descriptors,
                prototype,
            });

            let mut pairs: Vec<Value> = Vec::with_capacity(pairs_len + 2);
            if let Some(old) = parent_ref.transitions.heap_ref(heap) {
                pairs.extend(old.as_slice().iter().map(|slot| slot.inner()));
            }
            pairs.push(name_word.value());
            pairs.push(child.erase());
            let pairs = token.allocate::<FixedArray>(scope.stage_words(&pairs));
            parent_ref.transitions.set(heap, parent_ref.erase(), pairs);

            child.into_handle(scope)
        })
    }

    fn grow_slots_and_swap(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        flags: SlotFlags,
        value: Handle<Value>,
    ) {
        let slot_count =
            heap.no_gc(|heap| receiver.heap_ref(heap).map_ref(heap).value_slot_count() + 1);
        heap.allocate_token_enter_heap(FixedArray::layout_for(slot_count), |token, heap| {
            let receiver_ref = receiver.heap_ref(heap);
            let target = receiver_ref
                .map_ref(heap)
                .find_transition(
                    heap,
                    SlotName::from_value(unsafe { name.read_unchecked() }),
                    flags,
                    None,
                )
                .expect("transition recorded above");
            let mut values: Vec<Value> = Vec::with_capacity(slot_count);
            values.extend(
                receiver_ref
                    .slots
                    .heap_ref(heap)
                    .as_slice()
                    .iter()
                    .map(|slot| slot.inner()),
            );
            values.push(value.as_tagged(heap).erase());
            debug_assert_eq!(values.len(), slot_count, "slot count desynced from map");
            let slots = token.allocate::<FixedArray>(scope.stage_words(&values));
            let host = receiver.as_tagged(heap).erase();
            receiver_ref.slots.set(heap, host, slots);
            receiver_ref
                .header
                .map
                .set(heap, host, target.into_tagged());
        });
    }

    fn swap_map(
        heap: &mut Heap,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        flags: SlotFlags,
        pair: Option<(Handle<'_, Value>, Handle<'_, Value>)>,
    ) {
        heap.no_gc(|heap| {
            let receiver_ref = receiver.heap_ref(heap);
            let pair_words =
                pair.map(|(get, set)| (get.as_tagged(heap).erase(), set.as_tagged(heap).erase()));
            let target = receiver_ref
                .map_ref(heap)
                .find_transition(
                    heap,
                    SlotName::from_value(unsafe { name.read_unchecked() }),
                    flags,
                    pair_words,
                )
                .expect("transition recorded above");
            receiver_ref.header.map.set(
                heap,
                receiver.as_tagged(heap).erase(),
                target.into_tagged(),
            );
        });
    }

    fn write_slot(heap: &mut Heap, receiver: Handle<Object>, index: usize, value: Value) {
        heap.no_gc(|heap| {
            // Safety: the word came from a rooted read with no allocation
            // in between (define paths never allocate before this write).
            let value = unsafe { value.assume_valid(heap) };
            let offset = receiver.heap_ref(heap).map_ref(heap).descriptors()[index].offset();
            receiver.heap_ref(heap).slot(heap, offset).set(
                heap,
                receiver.as_tagged(heap).erase(),
                value,
            );
        });
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

        let (existing, kind, prototype, surviving, values, pairs_len) = heap.no_gc(|heap| {
            let name_word = SlotName::from_value(unsafe { name.read_unchecked() });
            let obj = receiver.heap_ref(heap);
            let parent = obj.map_ref(heap);
            let descriptors = parent.descriptors();
            let index = descriptors
                .iter()
                .position(|d| d.name() == name_word)
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
                .find_remove_transition_locked(heap, name_word, &guard)
                .map(|m| m.into_handle(scope));
            let mut surviving: Vec<(SlotName, SlotFlags, Handle<'_, Value>)> =
                Vec::with_capacity(descriptors.len() - 1);
            let mut values: Vec<Value> = obj.slots.heap_ref(heap).as_slice()[..base]
                .iter()
                .map(|slot| slot.inner())
                .collect();
            for (i, d) in descriptors.iter().enumerate() {
                if i == index {
                    continue;
                }
                if d.flags().is_accessor() {
                    // accessors embed their pair in the descriptor row
                    surviving.push((d.name(), d.flags(), scope.handle(d.value.get(heap))));
                } else {
                    // data rows re-dense their offsets; the value rides
                    // along in slot order
                    values.push(obj.slot(heap, d.offset()).inner());
                    surviving.push((
                        d.name(),
                        d.flags(),
                        scope.handle(Smi::new(values.len() as i64 - 1)),
                    ));
                }
            }
            let pairs_len = parent.transitions.heap_ref(heap).map_or(0, |a| a.len());
            (
                existing,
                parent.kind(),
                scope.handle(parent.prototype.get(heap)),
                surviving,
                values,
                pairs_len,
            )
        });

        if let Some(existing) = existing {
            // shared child map: only this receiver's slots need compacting
            let values = scope.stage_words(&values);
            heap.allocate_token_enter_heap(FixedArray::layout_for(values.len()), |token, heap| {
                let obj = receiver.heap_ref(heap);
                let slots = token.allocate::<FixedArray>(values);
                let host = receiver.as_tagged(heap).erase();
                obj.slots.set(heap, host, slots);
                obj.header.map.set(heap, host, existing.as_tagged(heap));
            });
            return;
        }

        let map_layout = Map::layout_for(surviving.len());
        let pairs_layout = FixedArray::layout_for(pairs_len + 2);
        let slots_layout = FixedArray::layout_for(values.len());
        let total = AllocToken::total_for(&[map_layout, pairs_layout, slots_layout]);
        let values = scope.stage_words(&values);
        heap.allocate_token_enter_heap(total, |token, heap| {
            let obj = receiver.heap_ref(heap);
            let parent = obj.map_ref(heap);
            let child = token.allocate::<Map>(MapInit {
                kind,
                value_slot_count: values.len(),
                descriptors: &surviving,
                prototype,
            });
            let name_word = SlotName::from_value(unsafe { name.read_unchecked() });
            let mut pairs: Vec<Value> = Vec::with_capacity(pairs_len + 2);
            if let Some(old) = parent.transitions.heap_ref(heap) {
                pairs.extend(old.as_slice().iter().map(|slot| slot.inner()));
            }
            pairs.push(name_word.value());
            pairs.push(child.erase());
            let pairs = token.allocate::<FixedArray>(scope.stage_words(&pairs));
            parent.transitions.set(heap, parent.erase(), pairs);
            let slots = token.allocate::<FixedArray>(values);
            let host = receiver.as_tagged(heap).erase();
            obj.slots.set(heap, host, slots);
            obj.header.map.set(heap, host, child);
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
                // Safety: descriptor words are fresh at entry (no
                // allocation happens before this rooting).
                let value = scope.handle(unsafe { value.assume_valid(&*heap) });

                let grow = match change {
                    Change::Append => true,
                    Change::Replace { index } => heap.no_gc(|heap| {
                        receiver.heap_ref(heap).map_ref(heap).descriptors()[index]
                            .flags()
                            .is_accessor()
                    }),
                };
                Self::target(
                    heap,
                    scope,
                    |heap| receiver.heap_ref(heap).map_ref(heap),
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
                    Self::write_slot(heap, receiver, index, value.as_tagged(heap).erase());
                    Self::swap_map(heap, receiver, name, flags, None);
                }
            }
            PropertyDescriptor::Accessor { get, set, .. } => {
                // Safety: as above.
                let get = scope.handle(unsafe { get.assume_valid(&*heap) });
                let set = scope.handle(unsafe { set.assume_valid(&*heap) });
                Self::target(
                    heap,
                    scope,
                    |heap| receiver.heap_ref(heap).map_ref(heap),
                    name,
                    flags,
                    Some((get, set)),
                    change,
                );
                Self::swap_map(heap, receiver, name, flags, Some((get, set)));
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
        heap.no_gc(|heap| {
            debug_assert!(
                !receiver
                    .heap_ref(heap)
                    .map_ref(heap)
                    .descriptors()
                    .iter()
                    .any(|d| d.name() == SlotName::from_value(unsafe { name.read_unchecked() })),
                "add_own_property requires the name to be absent from the receiver's own map"
            );
        });
        if !heap.no_gc(|heap| receiver.heap_ref(heap).is_extendable(heap)) {
            return Ok(false);
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
        let current = heap.no_gc(|heap| {
            receiver
                .heap_ref(heap)
                .map_ref(heap)
                .descriptors()
                .iter()
                .enumerate()
                .find(|(_, d)| d.name() == SlotName::from_value(unsafe { name.read_unchecked() }))
                .map(|(index, d)| (index, d.flags(), d.value.get(heap).erase()))
        });
        let Some((index, cur_flags, cur_desc_value)) = current else {
            return Self::add_own_property(heap, scope, receiver, name, desc);
        };

        let Some(action) = heap.no_gc(|heap| {
            validate_define(
                heap,
                receiver.heap_ref(heap),
                cur_flags,
                cur_desc_value,
                desc,
            )
        }) else {
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
        let (receiver, name) = root_define_inputs(scope, heap, receiver, name);
        Self::add_own_property(heap, scope, receiver, name, desc)
    }

    pub fn define_own_property_values(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Value,
        name: SlotName,
        desc: PropertyDescriptor,
    ) -> Result<bool, VmError> {
        let (receiver, name) = root_define_inputs(scope, heap, receiver, name);
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
        match heap.no_gc(|heap| classify_key(heap, unsafe { key.assume_valid(heap) }))? {
            Key::Element(i) => {
                // array elements live in the elements backing store,
                // outside the descriptors: delete punches a hole
                let is_array = heap.no_gc(|heap| receiver.heap_ref(heap).as_ref().is_array(heap));
                if is_array {
                    heap.no_gc(|heap| {
                        let obj = receiver.heap_ref(heap);
                        // indices at/past `length` were never own properties
                        if i < obj.as_ref().length()
                            && let Some(elements) = obj.as_ref().elements_array(heap)
                            && i < elements.len()
                        {
                            elements.set(
                                heap,
                                i,
                                heap.known().the_hole.as_tagged(heap).erase_type(),
                            );
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
        if heap.no_gc(|heap| {
            receiver
                .heap_ref(heap)
                .as_ref()
                .array_length(heap, name)
                .is_some()
        }) {
            return Ok(false);
        }
        // OrdinaryDelete: absent → true, non-configurable → false,
        // configurable → remove
        let configurable = heap.no_gc(|heap| {
            receiver
                .heap_ref(heap)
                .map_ref(heap)
                .descriptors()
                .iter()
                .find(|d| d.name() == name)
                .map(|d| d.flags().is_configurable())
        });
        if configurable != Some(true) {
            return Ok(configurable.is_none());
        }
        let name = scope.handle(unsafe { name.tagged(&*heap) });
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
        // Safety: caller-supplied word, fresh at entry.
        let receiver_handle = scope
            .cast::<Object>(unsafe { receiver.assume_valid(&*heap) })
            .ok_or(VmError::Type)?;

        let is_null = proto == heap.known().null.as_tagged(&*heap).erase();
        if !is_null && !proto.is_strong_ptr() {
            // silently ignore non-object prototypes (sloppy-mode semantics)
            return Ok(());
        }

        if heap.no_gc(|heap| {
            receiver_handle
                .heap_ref(heap)
                .map_ref(heap)
                .prototype
                .get(heap)
                .erase()
                == proto
        }) {
            return Ok(());
        }
        if !heap.no_gc(|heap| {
            receiver_handle
                .heap_ref(heap)
                .map_ref(heap)
                .kind()
                .is_extendable()
        }) {
            return Err(VmError::NotExtensible);
        }

        // cycle check: the receiver must not appear in any proposed chain
        // (FixedArray prototypes contribute one chain per element)
        heap.no_gc(|heap| -> Result<(), VmError> {
            let null = heap.known().null.as_tagged(heap).erase();
            let this = receiver_handle.as_tagged(heap).erase();
            // Safety: `proto` was read by the caller with no allocation
            // since entry.
            let start = unsafe { proto.assume_valid(heap) };
            fn walk<'b>(
                heap: &'b Heap,
                null: Value,
                this: Value,
                mut p: Tagged<'b, Value>,
            ) -> Result<(), VmError> {
                while p.is_strong_ptr() && p.erase() != null {
                    if p.erase() == this {
                        return Err(VmError::Type);
                    }
                    let Some(o) = p.as_heap_object() else {
                        break;
                    };
                    p = o.as_ref().map_ref(heap).prototype.get(heap);
                }
                Ok(())
            }
            if let Some(parents) = start.get_as::<FixedArray>() {
                for i in 0..parents.len() {
                    walk(heap, null, this, parents.at(heap, i))?;
                }
            } else {
                walk(heap, null, this, start)?;
            }
            Ok(())
        })?;

        // Safety: as above.
        let proto_handle = scope.handle(unsafe { proto.assume_valid(&*heap) });

        let descriptor_count = heap.no_gc(|heap| {
            receiver_handle
                .heap_ref(heap)
                .map_ref(heap)
                .descriptor_count()
        });

        heap.allocate_token_enter_heap(Map::layout_for(descriptor_count), |token, heap| {
            let obj = receiver_handle.heap_ref(heap);
            let map = obj.map_ref(heap);
            let kind = map.kind();
            let value_slot_count = map.value_slot_count();
            let descriptors: Vec<(SlotName, SlotFlags, Handle<'_, Value>)> = map
                .descriptors()
                .iter()
                .map(|d| (d.name(), d.flags(), scope.handle(d.value.get(heap))))
                .collect();
            let new_map = token.allocate::<Map>(MapInit {
                kind,
                value_slot_count,
                descriptors: &descriptors,
                prototype: proto_handle,
            });
            let host = receiver_handle.as_tagged(heap).erase();
            obj.header.map.set(heap, host, new_map);
            Ok(())
        })
    }
}

fn root_define_inputs<'s>(
    scope: &'s HandleScope<'_>,
    heap: &Heap,
    receiver: Value,
    name: SlotName,
) -> (Handle<'s, Object>, Handle<'s, SlotName>) {
    // store paths only reach here for object receivers (lookup dispatch)
    // Safety: both words are fresh caller reads (no allocation since).
    let receiver = scope
        .cast::<Object>(unsafe { receiver.assume_valid(heap) })
        .expect("store receiver is an object");
    let name = scope.handle(unsafe { name.tagged(heap) });
    (receiver, name)
}

fn validate_define<'a>(
    heap: &'a Heap,
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
        // Safety: descriptor words and the descriptor-row slot are fresh
        // reads under the anchor `heap`.
        let cur_pair = unsafe { cur_desc_value.assume_valid(heap) }
            .get_as::<AccessorPair>()
            .expect("accessor descriptor must hold a pair");
        if !Compare::same_value(
            heap,
            unsafe { get.assume_valid(heap) },
            cur_pair.get.get(heap),
        ) || !Compare::same_value(
            heap,
            unsafe { set.assume_valid(heap) },
            cur_pair.set.get(heap),
        ) {
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
            // Safety: as above.
            if !Compare::same_value(
                heap,
                unsafe { value.assume_valid(heap) },
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
pub struct PartialDescriptor {
    pub value: Option<Value>,
    pub get: Option<Value>,
    pub set: Option<Value>,
    pub writable: Option<bool>,
    pub enumerable: Option<bool>,
    pub configurable: Option<bool>,
}

impl PartialDescriptor {
    pub fn value(value: Value) -> Self {
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
        undefined: Value,
        current: Option<&PropertyDescriptor>,
    ) -> PropertyDescriptor {
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

impl From<&PropertyDescriptor> for PartialDescriptor {
    fn from(d: &PropertyDescriptor) -> Self {
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

fn c_enumumerable(d: &PropertyDescriptor) -> bool {
    match d {
        PropertyDescriptor::Data { enumerable, .. }
        | PropertyDescriptor::Accessor { enumerable, .. } => *enumerable,
    }
}

fn c_configurable(d: &PropertyDescriptor) -> bool {
    match d {
        PropertyDescriptor::Data { configurable, .. }
        | PropertyDescriptor::Accessor { configurable, .. } => *configurable,
    }
}

fn c_writable(d: &PropertyDescriptor) -> Option<bool> {
    match d {
        PropertyDescriptor::Data { writable, .. } => Some(*writable),
        PropertyDescriptor::Accessor { .. } => None,
    }
}

pub fn is_compatible_property_descriptor(
    heap: &Heap,
    extensible: bool,
    desc: &PartialDescriptor,
    current: Option<&PartialDescriptor>,
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
    let undefined_t = heap.known().undefined.as_tagged(heap);
    let undefined = undefined_t.erase();
    if cur_is_data && desc.is_data_descriptor() {
        if current.writable != Some(true) {
            if desc.writable == Some(true) {
                return false;
            }
            if let Some(v) = desc.value
                && !Compare::same_value(
                    heap,
                    // Safety: descriptor words are read fresh under the
                    // anchor by the callers of this check.
                    unsafe { v.assume_valid(heap) },
                    current.value.map_or(undefined_t.erase_type(), |w| unsafe {
                        w.assume_valid(heap)
                    }),
                )
            {
                return false;
            }
        }
    } else if !cur_is_data && desc.is_accessor_descriptor() {
        if current.get.unwrap_or(undefined) == undefined
            && let Some(g) = desc.get
            && g != undefined
        {
            return false;
        }
        if current.set.unwrap_or(undefined) == undefined
            && let Some(s) = desc.set
            && s != undefined
        {
            return false;
        }
    }
    true
}
