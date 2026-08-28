use std::sync::{Arc, Mutex, MutexGuard};

use crate::{
    FixedArray, Handle, HeapObject, HeapRef, LocalHeap, Lookup, Map, MapInit, NoGc, Object,
    SlotFlags, SlotName, Smi, Tagged, Value, ValueRef, VmError,
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
}

impl Value {
    pub fn store_lookup<'a>(
        &self,
        nogc: &'a NoGc<'a>,
        heap: &'a impl LocalHeap,
        name: SlotName,
        value: Value,
        semantics: StoreSemantics,
    ) -> Result<StoreOutcome, VmError> {
        let receiver = *self;
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
            Lookup::Accessor { .. } => unimplemented!("TODO: implement accessors"),
        }
    }
}

fn transition_target(
    heap: &mut impl LocalHeap,
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

    let (descriptor_count, value_slot_count, kind, pairs_len) = heap.no_gc(|nogc, heap| {
        let parent = parent(nogc);
        let pairs_len = parent.transitions.heap_ref(nogc, heap).map_or(0, |a| a.len());
        (
            parent.descriptor_count(),
            parent.value_slot_count(),
            parent.kind(),
            pairs_len,
        )
    });

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
        heap: &mut impl LocalHeap,
        parent: Handle<Map>,
        name: Handle<SlotName>,
        flags: SlotFlags,
    ) -> Tagged<Map> {
        transition_target(heap, |nogc| parent.heap_ref(nogc), name, flags)
    }
}

impl Object {
    pub fn store_new_data_property(
        heap: &mut impl LocalHeap,
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
                .find_transition(
                    nogc,
                    heap,
                    name.into(),
                    DATA_PROPERTY_FLAGS,
                )
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
            receiver_ref.header.map.set(heap, host, target.into_tagged());
        });
        Ok(())
    }
}
