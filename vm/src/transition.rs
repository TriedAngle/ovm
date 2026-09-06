use std::sync::{Arc, Mutex, MutexGuard};

use core::alloc::Layout;

use crate::{
    AccessorPair, Compare, FixedArray, Handle, HandleScope, Heap, HeapObject, HeapRef, Lookup, Map,
    MapInit, NoGc, Object, SlotFlags, SlotKind, SlotName, Smi, Tagged, Value, ValueRef, VmError,
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

/// Cached map transition for an ES 9.1.6.3 redefinition: change the
/// descriptor at `index` to `new_flags`. `grow` appends a fresh slot
/// (accessor → data); otherwise the existing offset is kept. Accessor
/// redefines skip this path: their rows embed the per-object pair.
fn redefine_target(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    receiver: Handle<Object>,
    name: Handle<SlotName>,
    index: usize,
    new_flags: SlotFlags,
    grow: bool,
) {
    let lock = heap.transition_lock();
    let guard = lock.acquire();

    let existing = heap.no_gc(|nogc, heap| {
        receiver
            .heap_ref(nogc)
            .header
            .map
            .heap_ref(nogc)
            .find_transition_locked(nogc, heap, name.into(), new_flags, &guard)
            .map(|target| target.into_tagged())
    });
    if existing.is_some() {
        return;
    }

    let (kind, descriptor_count, value_slot_count, pairs_len, prototype) =
        heap.no_gc(|nogc, heap| {
            let parent = receiver.heap_ref(nogc).header.map.heap_ref(nogc);
            let pairs_len = parent
                .transitions
                .heap_ref(nogc, heap)
                .map_or(0, |a| a.len());
            (
                parent.kind(),
                parent.descriptor_count(),
                parent.value_slot_count(),
                pairs_len,
                parent.prototype.inner(),
            )
        });
    let prototype = scope
        .create_handle(Tagged::from_value(prototype))
        .expect("map prototype is a strong pointer");

    let map_layout = Map::layout_for(descriptor_count);
    let pairs_layout = FixedArray::layout_for(pairs_len + 2);
    let total = map_layout
        .extend(pairs_layout)
        .expect("redefine transition layout")
        .0;

    heap.allocate_token_enter_nogc(total, |token, nogc, heap| {
        let parent_ref = receiver.heap_ref(nogc).header.map.heap_ref(nogc);

        let mut descriptors: Vec<(SlotName, SlotFlags, Value)> = parent_ref
            .descriptors()
            .iter()
            .map(|d| (d.name(), d.flags(), d.value.inner()))
            .collect();
        let offset = if grow {
            Smi::new(value_slot_count as i64).encode()
        } else {
            descriptors[index].2
        };
        descriptors[index] = (name.into(), new_flags, offset);
        let child = token.allocate::<Map>(MapInit {
            kind,
            value_slot_count: if grow {
                value_slot_count + 1
            } else {
                value_slot_count
            },
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
    });
}

/// A property descriptor for `[[DefineOwnProperty]]` (ES 9.1.6) in its
/// complete form: every field is explicitly present. The JS-level
/// `Object.defineProperty` field defaulting is the native's job; internal
/// definitional stores (literals, [[Set]] shadowing) pass full descriptors.
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
    /// The assignment/literal descriptor: writable, enumerable, configurable.
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

/// What a valid `[[DefineOwnProperty]]` must actually do.
enum DefineAction {
    /// Valid, nothing to change.
    Nothing,
    /// data→data with unchanged attributes: write the existing slot.
    WriteDataSlot { value: Value },
    /// Replace the descriptor in place; `reuse_slot` keeps the current data
    /// slot (attributes-only change), otherwise a fresh slot is appended.
    RedefineData {
        value: Value,
        flags: SlotFlags,
        reuse_slot: bool,
    },
    /// Replace the descriptor with an accessor (fresh pair).
    RedefineAccessor {
        get: Value,
        set: Value,
        flags: SlotFlags,
    },
}

impl Object {
    /// Define a property known to be ABSENT from the receiver's own map:
    /// the extensible-check + transition/define half of `[[DefineOwnProperty]]`,
    /// called directly when the lookup phase already proved `NOT_FOUND` (the
    /// `[[Set]]` shadowing path, literal fast paths) — no second scan.
    pub fn add_own_property(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        desc: PropertyDescriptor,
    ) -> Result<bool, VmError> {
        debug_assert!(
            heap.no_gc(|nogc, _| !receiver
                .heap_ref(nogc)
                .header
                .map
                .heap_ref(nogc)
                .descriptors()
                .iter()
                .any(|d| d.name() == name.into())),
            "add_own_property requires the name to be absent from the receiver's own map"
        );
        let extensible = heap.no_gc(|nogc, _| {
            receiver
                .heap_ref(nogc)
                .header
                .map
                .heap_ref(nogc)
                .kind()
                .is_extendable()
        });
        if !extensible {
            return Ok(false);
        }
        define_absent(heap, scope, receiver, name, desc)?;
        Ok(true)
    }

    /// ES 9.1.6 OrdinaryDefineOwnProperty: define or redefine an own property
    /// with full descriptor semantics. The single definitional primitive the
    /// other property paths build on. Returns `false` when the receiver is
    /// non-extensible and the property is absent, or when a non-configurable
    /// property forbids the change.
    pub fn define_own_property(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: Handle<Object>,
        name: Handle<SlotName>,
        desc: PropertyDescriptor,
    ) -> Result<bool, VmError> {
        let current = heap.no_gc(|nogc, _| {
            let map = receiver.heap_ref(nogc).header.map.heap_ref(nogc);
            map.descriptors()
                .iter()
                .enumerate()
                .find(|(_, d)| d.name() == name.into())
                .map(|(index, d)| (index, d.flags(), d.value.inner()))
        });
        let Some((index, cur_flags, cur_desc_value)) = current else {
            return Self::add_own_property(heap, scope, receiver, name, desc);
        };

        // 3. ValidateAndApplyPropertyDescriptor
        let action = heap.no_gc(|nogc, heap| {
            validate_define(
                nogc,
                heap,
                receiver.heap_ref(nogc),
                cur_flags,
                cur_desc_value,
                desc,
            )
        });
        match action {
            Some(DefineAction::Nothing) => Ok(true),
            Some(action) => {
                apply_define(heap, scope, receiver, name, index, action)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

/// ValidateAndApplyPropertyDescriptor (ES 10.1.6.3) for the case where the
/// property already exists. Returns the action to perform, or `None` when
/// the change is forbidden.
fn validate_define<'a>(
    nogc: &'a NoGc<'a>,
    heap: &Heap,
    receiver: HeapRef<'a, Object>,
    cur_flags: SlotFlags,
    cur_desc_value: Value,
    desc: PropertyDescriptor,
) -> Option<DefineAction> {
    let cur_kind = cur_flags.kind();
    let cur_configurable = cur_flags.is_configurable();
    let cur_enumerable = cur_flags.is_enumerable();
    let known = heap.known();

    // The current value: from the object's slot (Value kind) or the
    // descriptor itself (Const).
    let current_value = |nogc: &'a NoGc<'a>, offset: Value| -> Value {
        if cur_kind == SlotKind::Value {
            let offset = Smi::decode(offset).unwrap().value() as usize;
            receiver
                .as_ref()
                .slots
                .heap_ref(nogc)
                .element_slot(offset)
                .inner()
        } else {
            offset
        }
    };

    match (cur_kind, desc) {
        // both data: attributes may change only within the non-configurable
        // constraints; the value is writable or must be SameValue
        (
            SlotKind::Value | SlotKind::Const,
            PropertyDescriptor::Data {
                value,
                writable,
                enumerable,
                configurable,
            },
        ) => {
            let cur_writable = cur_flags.is_writable();
            if !cur_configurable {
                if configurable || enumerable != cur_enumerable {
                    return None;
                }
                if !cur_writable {
                    if writable {
                        return None;
                    }
                    if !Compare::same_value(nogc, heap, value, current_value(nogc, cur_desc_value))
                    {
                        return None;
                    }
                    return Some(DefineAction::Nothing);
                }
            }
            if desc.flags() == cur_flags {
                // plain value update: no shape change
                return Some(DefineAction::WriteDataSlot { value });
            }
            Some(DefineAction::RedefineData {
                value,
                flags: desc.flags(),
                // Const descriptors carry their value in the descriptor and
                // have no slot to reuse
                reuse_slot: cur_kind == SlotKind::Value,
            })
        }
        // data → accessor: only configurable properties can convert
        (SlotKind::Value | SlotKind::Const, PropertyDescriptor::Accessor { get, set, .. }) => {
            if !cur_configurable {
                return None;
            }
            Some(DefineAction::RedefineAccessor {
                get,
                set,
                flags: desc.flags(),
            })
        }
        // accessor → accessor: pair replacement (configurable) or exact
        // identity (non-configurable)
        (
            SlotKind::Accessor,
            PropertyDescriptor::Accessor {
                get,
                set,
                enumerable,
                configurable,
            },
        ) => {
            if !cur_configurable {
                if configurable || enumerable != cur_enumerable {
                    return None;
                }
                let cur_pair = cur_desc_value
                    .get_as::<AccessorPair>(nogc, known.accessor_pair_map)
                    .expect("accessor descriptor must hold a pair");
                if !Compare::same_value(nogc, heap, get, cur_pair.get.inner())
                    || !Compare::same_value(nogc, heap, set, cur_pair.set.inner())
                {
                    return None;
                }
                return Some(DefineAction::Nothing);
            }
            Some(DefineAction::RedefineAccessor {
                get,
                set,
                flags: desc.flags(),
            })
        }
        // accessor → data: only configurable properties can convert; the
        // new value gets a fresh slot
        (SlotKind::Accessor, PropertyDescriptor::Data { value, .. }) => {
            if !cur_configurable {
                return None;
            }
            Some(DefineAction::RedefineData {
                value,
                flags: desc.flags(),
                reuse_slot: false,
            })
        }
    }
}

/// Absent property on an extensible receiver: transition to a new map and
/// (for data) a new slot, or a fresh map + pair for accessors.
fn define_absent(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    receiver: Handle<Object>,
    name: Handle<SlotName>,
    desc: PropertyDescriptor,
) -> Result<(), VmError> {
    match desc {
        PropertyDescriptor::Data { value, .. } => {
            let flags = desc.flags();
            // root the value before the transition allocation (GC may move it)
            let value = scope
                .create_handle(Tagged::from_value(value))
                .expect("value must be strong");
            let slot_count = heap.no_gc(|nogc, _| {
                receiver
                    .heap_ref(nogc)
                    .header
                    .map
                    .heap_ref(nogc)
                    .value_slot_count()
                    + 1
            });
            transition_target(
                heap,
                scope,
                |nogc| receiver.heap_ref(nogc).header.map.heap_ref(nogc),
                name,
                flags,
            );
            heap.allocate_token_enter_nogc(
                FixedArray::layout_for(slot_count),
                |token, nogc, heap| {
                    let receiver_ref = receiver.heap_ref(nogc);
                    let target = receiver_ref
                        .header
                        .map
                        .heap_ref(nogc)
                        .find_transition(nogc, heap, name.into(), flags)
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
                },
            );
            Ok(())
        }
        PropertyDescriptor::Accessor { get, set, .. } => {
            // root the pair halves before the token entry allocates
            let get = scope
                .create_handle(Tagged::from_value(get))
                .expect("get must be strong");
            let set = scope
                .create_handle(Tagged::from_value(set))
                .expect("set must be strong");
            let (kind, descriptor_count, value_slot_count, prototype) = heap.no_gc(|nogc, _| {
                let map = receiver.heap_ref(nogc).header.map.heap_ref(nogc);
                (
                    map.kind(),
                    map.descriptor_count(),
                    map.value_slot_count(),
                    map.prototype.inner(),
                )
            });
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
                descriptors.push((name.into(), desc.flags(), pair.erase()));
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
}

/// Perform the redefinition: replace the descriptor at `index`. Data
/// redefines go through the shared transition tree (cached by name+flags,
/// like adds); accessor redefines allocate a fresh map each time because
/// the descriptor row embeds the per-object pair. Values are written into
/// the slot in place (attributes-only) or appended (accessor → data).
fn apply_define(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    receiver: Handle<Object>,
    name: Handle<SlotName>,
    index: usize,
    action: DefineAction,
) -> Result<(), VmError> {
    // the in-place data write needs no map change
    if let DefineAction::WriteDataSlot { value } = action {
        heap.no_gc(|nogc, heap| {
            let receiver_ref = receiver.heap_ref(nogc);
            let map = receiver_ref.header.map.heap_ref(nogc);
            let descriptor = &map.descriptors()[index];
            let offset = Smi::decode(descriptor.value.inner()).unwrap().value() as usize;
            receiver_ref.slots.heap_ref(nogc).element_slot(offset).set(
                heap,
                receiver.value(),
                Tagged::from_value(value),
            );
        });
        return Ok(());
    }

    let (kind, descriptor_count, value_slot_count, prototype) = heap.no_gc(|nogc, _| {
        let map = receiver.heap_ref(nogc).header.map.heap_ref(nogc);
        (
            map.kind(),
            map.descriptor_count(),
            map.value_slot_count(),
            map.prototype.inner(),
        )
    });
    let prototype = scope
        .create_handle(Tagged::from_value(prototype))
        .expect("map prototype is a strong pointer");

    match action {
        DefineAction::RedefineData {
            value,
            flags,
            reuse_slot,
        } => {
            // root the value before the transition allocation (GC may move it)
            let value = scope
                .create_handle(Tagged::from_value(value))
                .expect("value must be strong");
            // shared transition tree: objects with the same map converging
            // on the same redefinition share the target map (keyed by name,
            // verified by flags — same tree as adds)
            redefine_target(heap, scope, receiver, name, index, flags, !reuse_slot);
            if reuse_slot {
                // attributes-only: swap the map, keep the slot offset
                heap.no_gc(|nogc, heap| {
                    let receiver_ref = receiver.heap_ref(nogc);
                    let target = receiver_ref
                        .header
                        .map
                        .heap_ref(nogc)
                        .find_transition(nogc, heap, name.into(), flags)
                        .expect("transition recorded above");
                    let offset = Smi::decode(target.descriptors()[index].value.inner())
                        .unwrap()
                        .value() as usize;
                    let host = receiver.value();
                    receiver_ref.slots.heap_ref(nogc).element_slot(offset).set(
                        heap,
                        host,
                        Tagged::from_value(value.value()),
                    );
                    receiver_ref
                        .header
                        .map
                        .set(heap, host, target.into_tagged());
                });
            } else {
                // accessor → data: the target map appends a fresh slot
                let slot_count = value_slot_count + 1;
                heap.allocate_token_enter_nogc(
                    FixedArray::layout_for(slot_count),
                    |token, nogc, heap| {
                        let receiver_ref = receiver.heap_ref(nogc);
                        let target = receiver_ref
                            .header
                            .map
                            .heap_ref(nogc)
                            .find_transition(nogc, heap, name.into(), flags)
                            .expect("transition recorded above")
                            .into_tagged();
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
                        receiver_ref.header.map.set(heap, host, target);
                    },
                );
            }
        }
        DefineAction::RedefineAccessor { get, set, flags } => {
            // root the pair halves before the token entry allocates
            let get = scope
                .create_handle(Tagged::from_value(get))
                .expect("get must be strong");
            let set = scope
                .create_handle(Tagged::from_value(set))
                .expect("set must be strong");
            // fresh map: the descriptor row embeds the per-object pair,
            // so accessor maps cannot be shared
            let layout = Map::layout_for(descriptor_count)
                .extend(Layout::new::<AccessorPair>())
                .expect("redefine layout")
                .0;
            heap.allocate_token_enter_nogc(layout, |token, nogc, heap| {
                let receiver_ref = receiver.heap_ref(nogc);
                let map = receiver_ref.header.map.heap_ref(nogc);
                let mut descriptors: Vec<(SlotName, SlotFlags, Value)> = map
                    .descriptors()
                    .iter()
                    .map(|d| (d.name(), d.flags(), d.value.inner()))
                    .collect();
                let host = receiver.value();
                let pair = token.allocate::<AccessorPair>((get.value(), set.value()));
                descriptors[index] = (name.into(), flags, pair.erase());
                let new_map = token.allocate::<Map>(MapInit {
                    kind,
                    value_slot_count,
                    descriptors: &descriptors,
                    prototype,
                });
                receiver_ref
                    .header
                    .map
                    .set(heap, host, new_map.into_tagged());
            });
        }
        DefineAction::WriteDataSlot { .. } => unreachable!("handled above"),
        DefineAction::Nothing => unreachable!("Nothing is handled by the caller"),
    }
    Ok(())
}

/// `Object::add_own_property` for unrooted inputs (the `[[Set]]` shadowing
/// path: the lookup phase already proved the name is absent).
pub fn add_own_property_values(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    receiver: Value,
    name: SlotName,
    desc: PropertyDescriptor,
) -> Result<bool, VmError> {
    let receiver = scope
        .create_handle(unsafe { Tagged::<Object>::from_value_unchecked(receiver) })
        .expect("receiver must be strong");
    let name = scope
        .create_handle(name.tagged())
        .expect("name must be strong");
    Object::add_own_property(heap, scope, receiver, name, desc)
}

/// `Object::define_own_property` for unrooted inputs: roots
/// receiver/name in `scope`, then defines the property.
pub fn define_own_property_values(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    receiver: Value,
    name: SlotName,
    desc: PropertyDescriptor,
) -> Result<bool, VmError> {
    let receiver = scope
        .create_handle(unsafe { Tagged::<Object>::from_value_unchecked(receiver) })
        .expect("receiver must be strong");
    let name = scope
        .create_handle(name.tagged())
        .expect("name must be strong");
    Object::define_own_property(heap, scope, receiver, name, desc)
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
        // the raw `receiver` param may be stale after the token entry
        // allocated: re-read through the handle for the write barrier
        let host = receiver_handle.value();
        obj.header.map.set(heap, host, new_map.into_tagged());
        Ok(())
    })
}
