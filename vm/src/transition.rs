use std::sync::{Arc, Mutex, MutexGuard};

use core::alloc::Layout;

use crate::{
    AccessorPair, Compare, FixedArray, Handle, HandleScope, Heap, HeapObject, HeapRef, Lookup, Map,
    MapInit, NoGc, Object, SlotFlags, SlotName, Smi, Tagged, Value, VmError,
};

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
        let mut total = map_layout
            .extend(pairs_layout)
            .expect("transition layout")
            .0;
        if pair.is_some() {
            total = Layout::new::<AccessorPair>()
                .extend(total)
                .expect("transition layout")
                .0;
        }

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
    /// - the `void` sentinel; other values are
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
