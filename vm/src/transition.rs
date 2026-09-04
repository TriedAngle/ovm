use std::sync::{Arc, Mutex, MutexGuard};

use core::alloc::Layout;

use crate::{
    AccessorPair, FixedArray, Handle, HandleScope, Heap, HeapObject, HeapRef, Lookup, Map, MapInit,
    NoGc, Object, SlotFlags, SlotName, Smi, Tagged, Value, ValueRef, VmError,
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

/// Default attributes for assignment-created properties
const DATA_PROPERTY_FLAGS: SlotFlags = SlotFlags::VALUE
    .union(SlotFlags::WRITABLE)
    .union(SlotFlags::ENUMERABLE)
    .union(SlotFlags::CONFIGURABLE);

/// Default attributes for define-created accessor properties
const ACCESSOR_PROPERTY_FLAGS: SlotFlags = SlotFlags::ACCESSOR
    .union(SlotFlags::ENUMERABLE)
    .union(SlotFlags::CONFIGURABLE);

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
        heap: &'a Heap,
        name: SlotName,
        value: Value,
        semantics: StoreSemantics,
    ) -> Result<StoreOutcome, VmError> {
        let receiver = *self;
        // null/undefined have no [[Prototype]]: property access throws
        if receiver == heap.known().null.value() || receiver == heap.known().undefined.value() {
            return Err(VmError::Type);
        }
        match receiver.lookup(nogc, heap, name) {
            Lookup::Data {
                slot,
                holder,
                flags,
                ..
            } => {
                if !flags.is_writable() {
                    return Err(VmError::Type);
                }
                let ValueRef::Object(host) = holder else {
                    return Err(VmError::Type);
                };
                let host = host.as_ref().erase();
                if semantics == StoreSemantics::Shadow && host != receiver {
                    // inherited writable data property: JS creates an own
                    // property on the receiver
                    if !receiver.is_strong_ptr() {
                        return Err(VmError::Type);
                    }
                    return Ok(StoreOutcome::Transition { receiver, name });
                }
                slot.set(heap, host, Tagged::from_value(value));
                Ok(StoreOutcome::Done)
            }
            Lookup::NotFound => {
                // adding a property requires a heap receiver
                if !receiver.is_strong_ptr() {
                    return Err(VmError::Type);
                }
                Ok(StoreOutcome::Transition { receiver, name })
            }
            // const is never writable
            Lookup::Const { .. } => Err(VmError::Type),
            Lookup::Accessor { pair, .. } => {
                let setter = pair.set.inner();
                // no setter (undefined sentinel): sloppy-mode writes to a
                // setter-less accessor are silently ignored
                if setter == heap.known().undefined.value() {
                    return Ok(StoreOutcome::Done);
                }
                Ok(StoreOutcome::CallSetter { setter })
            }
        }
    }
}

fn transition_target(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    parent: impl for<'a> Fn(&'a NoGc<'a>) -> HeapRef<'a, Map>,
    name: Handle<SlotName>,
    flags: SlotFlags,
) -> Tagged<Map> {
    let lock = heap.transition_lock();
    let guard = lock.acquire();

    let existing = heap.no_gc(|nogc, heap| {
        parent(nogc)
            .find_transition_locked(nogc, heap, name.into(), flags, &guard)
            .map(|target| target.into_tagged())
    });
    if let Some(target) = existing {
        return target;
    }

    let (descriptor_count, value_slot_count, kind, pairs_len, prototype) =
        heap.no_gc(|nogc, heap| {
            let parent = parent(nogc);
            let pairs_len = parent
                .transitions
                .heap_ref(nogc, heap)
                .map_or(0, |a| a.len());
            (
                parent.descriptor_count(),
                parent.value_slot_count(),
                parent.kind(),
                pairs_len,
                parent.prototype.inner(),
            )
        });
    let prototype = scope
        .create_handle(Tagged::from_value(prototype))
        .expect("map prototype is a strong pointer");

    let map_layout = Map::layout_for(descriptor_count + 1);
    let pairs_layout = FixedArray::layout_for(pairs_len + 2);
    let total = map_layout
        .extend(pairs_layout)
        .expect("transition layout")
        .0;

    heap.allocate_token_enter_nogc(total, |token, nogc, heap| {
        let parent_ref = parent(nogc);

        let mut descriptors: Vec<(SlotName, SlotFlags, Value)> = parent_ref
            .descriptors()
            .iter()
            .map(|d| (d.name(), d.flags(), d.value.inner()))
            .collect();
        descriptors.push((
            name.into(),
            flags,
            Smi::new(value_slot_count as i64).encode(),
        ));
        let child = token.allocate::<Map>(MapInit {
            kind,
            value_slot_count: value_slot_count + 1,
            descriptors: &descriptors,
            prototype,
        });

        let mut pairs: Vec<Value> = Vec::with_capacity(pairs_len + 2);
        if let Some(old) = parent_ref.transitions.heap_ref(nogc, heap) {
            pairs.extend(old.as_slice().iter().map(|slot| slot.inner()));
        }
        pairs.push(name.value());
        pairs.push(child.erase());
        let pairs = token.allocate::<FixedArray>(&pairs);
        parent_ref
            .transitions
            .set(heap, parent_ref.erase(), pairs.into_tagged());

        child.into_tagged()
    })
}

impl Map {
    pub fn transition_target(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        parent: Handle<Map>,
        name: Handle<SlotName>,
        flags: SlotFlags,
    ) -> Tagged<Map> {
        transition_target(heap, scope, |nogc| parent.heap_ref(nogc), name, flags)
    }
}

impl Object {
    pub fn store_new_data_property(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        value: Handle<Value>,
    ) -> Result<(), VmError> {
        let (extendable, slot_count) = heap.no_gc(|nogc, _| {
            let map = receiver.heap_ref(nogc).header.map.heap_ref(nogc);
            (map.kind().is_extendable(), map.value_slot_count() + 1)
        });
        if !extendable {
            return Err(VmError::NotExtensible);
        }

        transition_target(
            heap,
            scope,
            |nogc| receiver.heap_ref(nogc).header.map.heap_ref(nogc),
            name,
            DATA_PROPERTY_FLAGS,
        );

        heap.allocate_token_enter_nogc(FixedArray::layout_for(slot_count), |token, nogc, heap| {
            let receiver_ref = receiver.heap_ref(nogc);

            let target = receiver_ref
                .header
                .map
                .heap_ref(nogc)
                .find_transition(nogc, heap, name.into(), DATA_PROPERTY_FLAGS)
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
            receiver_ref.slots.set(heap, host, slots.into_tagged());
            receiver_ref
                .header
                .map
                .set(heap, host, target.into_tagged());
        });
        Ok(())
    }

    /// Define a new own accessor property backed by a fresh `AccessorPair`
    ///
    /// Unlike data properties this never goes through the transition cache:
    /// the descriptor value is the pair identity, which is not shareable
    /// between objects, so a fresh child map is created every time.
    pub fn store_new_accessor_property(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        get: Handle<Value>,
        set: Handle<Value>,
    ) -> Result<(), VmError> {
        let (extendable, kind, descriptor_count, value_slot_count, prototype) =
            heap.no_gc(|nogc, _| {
                let map = receiver.heap_ref(nogc).header.map.heap_ref(nogc);
                (
                    map.kind().is_extendable(),
                    map.kind(),
                    map.descriptor_count(),
                    map.value_slot_count(),
                    map.prototype.inner(),
                )
            });
        if !extendable {
            return Err(VmError::NotExtensible);
        }
        let prototype = scope
            .create_handle(Tagged::from_value(prototype))
            .expect("map prototype is a strong pointer");

        let layout = Layout::new::<AccessorPair>()
            .extend(Map::layout_for(descriptor_count + 1))
            .expect("accessor layout")
            .0;

        heap.allocate_token_enter_nogc(layout, |token, nogc, heap| {
            let pair = token.allocate::<AccessorPair>((get.value(), set.value()));

            let receiver_ref = receiver.heap_ref(nogc);
            let mut descriptors: Vec<(SlotName, SlotFlags, Value)> = receiver_ref
                .header
                .map
                .heap_ref(nogc)
                .descriptors()
                .iter()
                .map(|d| (d.name(), d.flags(), d.value.inner()))
                .collect();
            descriptors.push((name.into(), ACCESSOR_PROPERTY_FLAGS, pair.erase()));

            let map = token.allocate::<Map>(MapInit {
                kind,
                value_slot_count,
                descriptors: &descriptors,
                prototype,
            });
            let host = receiver.value();
            receiver_ref.header.map.set(heap, host, map.into_tagged());
        });
        Ok(())
    }
}

/// `Object::store_new_data_property` for unrooted inputs: roots
/// receiver/name/value in `scope`, then defines the data property.
pub fn store_new_data_property_values(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    receiver: Value,
    name: SlotName,
    value: Value,
) -> Result<(), VmError> {
    let receiver = scope
        .create_handle(unsafe { Tagged::<Object>::from_value_unchecked(receiver) })
        .expect("receiver must be strong");
    let name = scope
        .create_handle(name.tagged())
        .expect("name must be strong");
    let value = scope
        .create_handle(Tagged::from_value(value))
        .expect("value must be strong");
    Object::store_new_data_property(heap, scope, receiver, name, value)
}

/// - `proto` must be:
/// - arbitrary normal `Object` for JS smenatics
/// - `FixedArray` of objects (Self-style multiple parents),
/// - the`void` sentinel; other values are
///   silently ignored (sloppy semantics)
/// - cycles and non-extensible receivers throw (TODO: make this optional for Self semantics)
pub fn set_prototype(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    receiver: Value,
    proto: Value,
) -> Result<(), VmError> {
    let receiver_handle = scope
        .create_handle(unsafe { Tagged::<Object>::from_value_unchecked(receiver) })
        .expect("receiver must be strong");

    let is_null = proto == heap.known().null.value();
    if !is_null && !proto.is_strong_ptr() {
        // silently ignore non-object prototypes (sloppy-mode semantics)
        return Ok(());
    }

    let (extendable, unchanged) = heap.no_gc(|nogc, _heap| {
        let map = receiver_handle.heap_ref(nogc).header.map.heap_ref(nogc);
        (map.kind().is_extendable(), map.prototype.inner() == proto)
    });
    if unchanged {
        return Ok(());
    }
    if !extendable {
        return Err(VmError::NotExtensible);
    }

    // cycle check: the receiver must not appear in any proposed chain
    // (FixedArray prototypes contribute one chain per element)
    heap.no_gc(|nogc, heap| {
        let walk = |p: Value| -> Result<(), VmError> {
            let mut p = p;
            while p.is_strong_ptr() && p != heap.known().null.value() {
                if p == receiver {
                    return Err(VmError::Type);
                }
                let ValueRef::Object(o) = p.value_ref(nogc) else {
                    break;
                };
                p = o.as_ref().header.map.heap_ref(nogc).prototype.inner();
            }
            Ok(())
        };
        if let Some(parents) = proto.get_as::<FixedArray>(nogc, heap.known().array_map) {
            for i in 0..parents.len() {
                walk(parents.at(i))?;
            }
        } else {
            walk(proto)?;
        }
        Ok(())
    })?;

    let proto_handle = scope
        .create_handle(Tagged::from_value(proto))
        .expect("prototype must be a strong pointer");

    let descriptor_count = heap.no_gc(|nogc, _| {
        receiver_handle
            .heap_ref(nogc)
            .header
            .map
            .heap_ref(nogc)
            .descriptor_count()
    });

    heap.allocate_token_enter_nogc(Map::layout_for(descriptor_count), |token, nogc, heap| {
        let obj = receiver_handle.heap_ref(nogc);
        let map = obj.header.map.heap_ref(nogc);
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
        let host = receiver;
        obj.header.map.set(heap, host, new_map.into_tagged());
        Ok(())
    })
}
