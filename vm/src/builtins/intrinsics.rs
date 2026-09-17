//! The CallRuntime natives: one implementation per `bytecode::RuntimeFn`,
//! plus the `runtime_fn` table the native registry seeds its fixed
//! 0..COUNT range with. These are compiled-language semantics (each cites
//! its ES section), not JS-visible library functions.

use crate::{
    AccessorPair, Context, Convert, DenseString, FixedArray, GcSlice,
    Handle, Heap, HeapRef, Key, LoadOutcome, Lookup, NoGc, Object, ObjectSlotsInit, PropertyDescriptor,
    SlotName, Smi, StoreOutcome, StoreSemantics, StringData, Symbol, Tagged, Value, VmError,
    classify_key, function_kind_of, home_proto, load_outcome, private_find,
    runtime::Coercion, super_constructor, super_lookup_from_proto, super_store_lookup,
};

use crate::natives::{NativeContext, NativeFn};
use crate::{ContextState, VM};

/// The fixed runtime-helper table: one implementation per
/// `bytecode::RuntimeFn`. The exhaustive match is the compile-time link
/// between the ABI ids and their implementations — adding a variant
/// without an entry here is a compile error, and `NativeRegistry::new`
/// registers them in `RuntimeFn::ALL` order so registry indices equal
/// discriminants.
pub(crate) fn runtime_fn(id: bytecode::RuntimeFn) -> NativeFn {
    match id {
        bytecode::RuntimeFn::GetIterator => get_iterator,
        bytecode::RuntimeFn::IteratorNext => iterator_next,
        bytecode::RuntimeFn::IteratorDone => iterator_done,
        bytecode::RuntimeFn::IteratorValue => iterator_value,
        bytecode::RuntimeFn::HasProperty => has_property,
        bytecode::RuntimeFn::CopyDataProperties => copy_data_properties,
        bytecode::RuntimeFn::CreatePrivateName => create_private_name,
        bytecode::RuntimeFn::PrivateGet => private_get,
        bytecode::RuntimeFn::PrivateSet => private_set,
        bytecode::RuntimeFn::PrivateIn => private_in,
        bytecode::RuntimeFn::SetClassFields => set_class_fields,
        bytecode::RuntimeFn::InitInstanceFields => init_instance_fields,
        bytecode::RuntimeFn::RequireObjectCoercible => require_object_coercible,
        bytecode::RuntimeFn::DeletePropertySloppy => delete_property_sloppy,
        bytecode::RuntimeFn::DeletePropertyStrict => delete_property_strict,
        bytecode::RuntimeFn::DeleteIdentifierSloppy => delete_identifier_sloppy,
        bytecode::RuntimeFn::DeleteSuperProperty => delete_super_property,
        bytecode::RuntimeFn::ForInEnumerate => for_in_enumerate,
        bytecode::RuntimeFn::ForInNext => for_in_next,
        bytecode::RuntimeFn::SetFunctionName => set_function_name,
        bytecode::RuntimeFn::InstallAccessor => install_accessor,
        bytecode::RuntimeFn::DefineOwnProperty => define_own_property,
        bytecode::RuntimeFn::SetPrototype => set_prototype,
        bytecode::RuntimeFn::ThrowIfNotConstructorOrNull => throw_if_not_constructor_or_null,
        bytecode::RuntimeFn::ThrowIfNotObjectOrNull => throw_if_not_object_or_null,
        bytecode::RuntimeFn::ThrowSuperNotCalledIfHole => throw_super_not_called_if_hole,
        bytecode::RuntimeFn::ThrowSuperAlreadyCalledIfNotHole => {
            throw_super_already_called_if_not_hole
        }
        bytecode::RuntimeFn::ConstructSuper => construct_super,
        bytecode::RuntimeFn::ConstructSuperAllArgs => construct_super_all_args,
        bytecode::RuntimeFn::ConstructSuperVia => construct_super_via,
        bytecode::RuntimeFn::LoadDynamicName => load_dynamic_name,
        bytecode::RuntimeFn::StoreDynamicName => store_dynamic_name,
        bytecode::RuntimeFn::CreateRestParameter => create_rest_parameter,
        bytecode::RuntimeFn::SuperGetProperty => super_get_property,
        bytecode::RuntimeFn::SuperSetProperty => super_set_property,
    }
}

// ---- runtime helpers ------------------------------------------------------

/// RequireObjectCoercible (ES 7.2.2): (value) -> value, TypeError on
/// null/undefined (object destructuring sources).
fn require_object_coercible(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    // internal-native convention: the register list IS the argument list
    // (no receiver slot)
    let arg = args.get(0).ok_or(VmError::Arity)?;
    let nullish = nctx
        .heap()
        .no_gc(|nogc| arg == nogc.known().null.value() || arg == nogc.known().undefined.value());
    if nullish {
        return Err(VmError::Type);
    }
    Ok(arg)
}

// ---- delete (ES 13.5.1) ----------------------------------------------------

/// `delete obj.key` in sloppy code: (obj, key) -> bool.
fn delete_property_sloppy(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    delete_property(nctx, args, false)
}

/// `delete obj.key` in strict code: (obj, key) -> bool, TypeError when
/// the delete fails (ES 13.5.1.2 step 4.h).
fn delete_property_strict(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    delete_property(nctx, args, true)
}

fn delete_property(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
    strict: bool,
) -> Result<Value, VmError> {
    let raw_target = args.get(0).ok_or(VmError::Arity)?;
    let raw_key = args.get(1).ok_or(VmError::Arity)?;
    // the reference's key is coerced before the base is touched (ES
    // 13.15.5 EvaluatePropertyAccess: user toString/valueOf of a
    // computed key runs even when the delete afterwards throws);
    // the coercion allocates, so the base must stay rooted across it
    nctx.handle_scope(|nctx, scope| {
        let target = scope.handle(raw_target);
        let target = target.value();
        let (vm, heap, state) = nctx.split();
        let Some(key) = crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(nctx.heap().known().exception.value());
        };
        // proxies run their `deleteProperty` trap (ES 20.2.5.4); the
        // returned boolean flows through the strict handling below
        let ok = if nctx
            .heap()
            .no_gc(|nogc| crate::proxy::is_proxy(nogc, target))
        {
            let (vm, heap, state) = nctx.split();
            match crate::proxy::delete(vm, heap, state, target, key)? {
                Coercion::Threw => return Ok(heap.known().exception.value()),
                Coercion::Value(v) => heap.no_gc(|nogc| Convert::is_truthy(nogc, v)),
            }
        } else {
            delete_property_core(nctx, target, key)?
        };
        if strict && !ok {
            return Err(VmError::Type);
        }
        Ok(Convert::boolean(nctx.heap(), ok))
    })
}

fn delete_property_core(
    nctx: &mut NativeContext<'_>,
    target: Value,
    key: Value,
) -> Result<bool, VmError> {
    // ToObject (ES 7.2.3): a null/undefined base throws
    let nullish = nctx.heap().no_gc(|nogc| {
        target == nogc.known().null.value() || target == nogc.known().undefined.value()
    });
    if nullish {
        return Err(VmError::Type);
    }
    // primitives: ToObject creates a fresh wrapper whose only own
    // properties are a string's non-configurable length/indices
    if nctx
        .heap()
        .no_gc(|nogc| Convert::is_primitive(nogc, target))
    {
        let owned = nctx
            .heap()
            .no_gc(|nogc| string_exotic_own(nogc, target, key));
        return Ok(!owned);
    }
    nctx.handle_scope(|nctx, scope| {
        let receiver = scope
            .cast::<Object>(target)
            .expect("non-primitive receivers are objects");
        Object::delete_own_property(nctx.heap(), &scope, receiver, key)
    })
}

/// Whether a ToObject'd primitive owns `key` non-configurably: only
/// String wrappers own anything — "length" and their indices (ES
/// 10.4.3.3/4 StringGetOwnProperty). Deleting those yields false; every
/// other primitive property deletes as absent (true).
fn string_exotic_own<'a>(nogc: &'a NoGc<'a>, target: Value, key: Value) -> bool {
    let Some(s) = target.get_as::<DenseString>(nogc) else {
        return false;
    };
    if let Some(idx) = Smi::decode(key) {
        let i = idx.value();
        return i >= 0 && (i as u64) < s.len() as u64;
    }
    let Some(name) = key.get_as::<DenseString>(nogc) else {
        return false; // symbols own nothing on primitives
    };
    let data = name.as_ref().data(nogc);
    data.matches_ascii(b"length")
        || crate::lookup::canonical_index(data).is_some_and(|i| i < s.len())
}

/// The one-code-unit string at index `i` of a string value, freshly
/// allocated (string comparisons are by content, so identity never
/// shows). `None` when the receiver is not a string or `i` is out of
/// range (ES 6.1.4: string indices are code units).
pub(crate) fn string_char_at(
    heap: &mut Heap,
    scope: &crate::HandleScope<'_>,
    receiver: Value,
    i: usize,
) -> Option<Value> {
    let units = heap.no_gc(|nogc| {
        let s = receiver.get_as::<DenseString>(nogc)?;
        (i < s.len()).then(|| [s.code_unit(nogc, i)])
    })?;
    Some(DenseString::from_units(heap, scope, &units).value())
}

/// Sloppy `delete x` on an unresolved name (ES 13.5.1.2 step 5 →
/// GlobalEnvironmentRecord.DeleteBinding): (name) -> bool. Declared
/// bindings resolve statically and compile to `false`; only global-object
/// properties reach here, and sloppy references never throw on failure.
fn delete_identifier_sloppy(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let name = args.get(0).ok_or(VmError::Arity)?;
    let global = nctx.heap().known().global_object.value();
    let ok = delete_property_core(nctx, global, name)?;
    Ok(Convert::boolean(nctx.heap(), ok))
}

/// `delete super.x` (ES 13.5.1.2 step 4.c): ReferenceError in both
/// language modes. The reference has already been evaluated (including
/// the uninitialized-`this` check and the key expression); the key is
/// never coerced — delete-super fails before any ToPropertyKey.
fn delete_super_property(
    _nctx: &mut NativeContext<'_>,
    _args: GcSlice<'_>,
) -> Result<Value, VmError> {
    Err(VmError::Reference)
}

// ---- for-in (ES 14.7.5) ----------------------------------------------------

/// Enumerator slot layout (a hidden object of `for_in_enumerator_map`):
/// [0] the current prototype-chain level (an object, or a string
///     primitive for level 0 of string subjects),
/// [1] the level's own-string-key snapshot (a FixedArray of interned
///     strings, taken when the level is reached),
/// [2] the snapshot cursor (Smi),
/// [3] keys already registered (a FixedArray; yielded keys and
///     non-enumerable shadowing keys both enter it, ES 14.7.5.9).
const FOR_IN_LEVEL: usize = 0;
const FOR_IN_KEYS: usize = 1;
const FOR_IN_INDEX: usize = 2;
const FOR_IN_VISITED: usize = 3;

/// for-in head (ES 14.7.5.6 ForIn/OfHeadEvaluation, enumerate):
/// (subject) -> enumerator | undefined. null/undefined subjects run
/// zero iterations; objects and strings snapshot level 0 of the lazy
/// chain walk. Other primitives' prototypes are not walked yet (their
/// own properties are none, so they enumerate empty).
fn for_in_enumerate(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let subject = args.get(0).ok_or(VmError::Arity)?;
    let nullish = nctx.heap().no_gc(|nogc| {
        subject == nogc.known().null.value() || subject == nogc.known().undefined.value()
    });
    if nullish {
        return Ok(nctx.heap().known().undefined.value());
    }
    let level = nctx
        .heap()
        .no_gc(|nogc| for_in_initial_level(nogc, subject));
    let Some(level) = level else {
        return Ok(nctx.heap().known().undefined.value());
    };
    nctx.handle_scope(|nctx, scope| {
        // the level must survive the key collection and FixedArray
        // allocation below (both allocate)
        let level = scope.handle(level);
        let (vm, heap, _) = nctx.split();
        let keys = for_in_level_keys(vm, heap, &scope, level.value())?;

        let keys = heap.allocate_handle::<FixedArray>(scope.stage(&keys), &scope);
        let empty = heap.known().empty_fixed_array;
        let map = heap.known().for_in_enumerator_map;
        let enumerator = heap.new_object(
            &scope,
            map,
            scope.stage(&[level.value(), keys.value(), Smi::new(0).encode(), empty.value()]),
        );
        Ok(enumerator.into_tagged().erase())
    })
}

/// Level 0 of the chain for a subject: objects are their own level 0;
/// string primitives enumerate their indices (a fresh wrapper would be
/// unobservable otherwise). Other primitives have no own properties —
/// their level 0 is the constructor's prototype, so additions to
/// `Number.prototype` etc. are observable (ES 14.7.5.9: the walk starts
/// at ToObject(subject)). `None` when the prototype is unreachable.
fn for_in_initial_level<'a>(nogc: &'a NoGc<'a>, subject: Value) -> Option<Value> {
    if subject.get_as::<DenseString>(nogc).is_some() {
        return Some(subject);
    }
    if !Convert::is_primitive(nogc, subject) {
        return Some(subject);
    }
    let ctor_name = if Smi::decode(subject).is_some() {
        "Number"
    } else if subject.get_as::<crate::Float>(nogc).is_some() {
        "Number"
    } else if subject == nogc.known().true_object.value()
        || subject == nogc.known().false_object.value()
    {
        "Boolean"
    } else if subject.get_as::<Symbol>(nogc).is_some() {
        "Symbol"
    } else {
        return None;
    };
    let global = nogc.known().global_object.value();
    let strings = nogc.known().strings;
    let name = SlotName::from_value(match ctor_name {
        "Number" => strings.number_ctor.value(),
        "Boolean" => strings.boolean_ctor.value(),
        _ => strings.symbol_ctor.value(),
    });
    let ctor = match crate::lookup::load_outcome(nogc, global, name).ok()? {
        crate::LoadOutcome::Value(v) if v.is_strong_ptr() => v,
        _ => return None,
    };
    let proto_name = SlotName::from_value(nogc.known().strings.prototype.value());
    match crate::lookup::load_outcome(nogc, ctor, proto_name).ok()? {
        crate::LoadOutcome::Value(p) if p.is_strong_ptr() => Some(p),
        _ => None,
    }
}

/// The own string keys of a level in [[OwnPropertyKeys]] order (ES
/// 10.1.11: array indices ascending, then strings in insertion order;
/// symbols excluded). Index keys are interned to their canonical string
/// form, the value a for-in binding receives. Enumerability is NOT
/// filtered here: EnumerateObjectProperties checks it lazily per key,
/// and non-enumerable own keys must still register as visited.
fn for_in_level_keys(
    vm: &VM,
    heap: &mut Heap,
    scope: &crate::HandleScope<'_>,
    level: Value,
) -> Result<Vec<Value>, VmError> {
    // raw pass: Smi index keys (to be interned) and ready name keys
    let (mut indices, names) = heap.no_gc(|nogc| {
        let mut indices: Vec<i64> = Vec::new();
        let mut names: Vec<Value> = Vec::new();
        if let Some(s) = level.get_as::<DenseString>(nogc) {
            // string exotic: the only own string keys are the indices
            // ("length" is non-enumerable; the wrapper's own "length"
            // shadowing String.prototype additions is not modeled)
            indices.extend(0..s.len() as i64);
            return (indices, names);
        }
        let Some(obj) = level.as_heap_object(nogc) else {
            return (indices, names);
        };
        let obj = obj.as_ref();
        // array elements: non-hole indices ascending
        if obj.is_array(nogc) {
            let len = obj
                .length()
                .min(obj.elements_array(nogc).map(|e| e.len()).unwrap_or(0));
            for i in 0..len {
                if obj.element_value(nogc, i).is_some() {
                    indices.push(i as i64);
                }
            }
        }

        for d in obj.map_ref(nogc).descriptors() {
            let name = d.name();

            if let Some(smi) = Smi::decode(name.value()) {
                let v = smi.value();
                // array-index-range Smi names are index keys; anything
                // else (negative, ≥ 2^32−1) keeps insertion order
                if (0..u32::MAX as i64).contains(&v) {
                    indices.push(v);
                } else {
                    names.push(name.value());
                }
                continue;
            }
            if name.value().get_as::<Symbol>(nogc).is_some() {
                continue; // symbols are never yielded
            }
            // canonical index strings classify as index keys (a store
            // through them creates a Smi-named descriptor, but object
            // literals and defines can still reach here)
            let index = name
                .value()
                .get_as::<DenseString>(nogc)
                .and_then(|s| crate::lookup::canonical_index(s.as_ref().data(nogc)))
                .filter(|i| *i <= u32::MAX as usize - 1);
            match index {
                Some(i) => indices.push(i as i64),
                None => names.push(name.value()),
            }
        }
        (indices, names)
    });

    indices.sort_unstable();
    indices.dedup();
    // root every key before the next allocates: interning index keys
    // promotes earlier results, and raw copies would dangle
    // root the name keys too: the interning loop below allocates
    let names: Vec<Handle<'_, Value>> = names.iter().map(|v| scope.handle(*v)).collect();
    let mut keys: Vec<Handle<'_, Value>> = Vec::with_capacity(indices.len() + names.len());
    for i in indices {
        let s = vm.interner().intern_str(heap, scope, &i.to_string());
        keys.push(scope.handle(s.value()));
    }
    keys.extend(names);
    let keys: Vec<Value> = keys.iter().map(|h| h.value()).collect();
    Ok(keys)
}

/// for-in iteration step (ES 14.7.5.9 EnumerateObjectProperties):
/// (enumerator) -> next key string | undefined. Per candidate key, the
/// own descriptor is checked lazily against the key's own level —
/// deleted-since-snapshot keys are skipped unvisited; keys shadowed by
/// an earlier level (yielded or non-enumerable) are skipped; enumerable
/// survivors are yielded at most once. When a level's snapshot runs
/// dry, the walk advances to the live prototype and snapshots it.
fn for_in_next(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let enumerator = args.get(0).ok_or(VmError::Arity)?;
    if enumerator == nctx.heap().known().undefined.value() {
        // nullish subject: the head produced no enumerator
        return Ok(nctx.heap().known().undefined.value());
    }
    nctx.handle_scope(|nctx, scope| {
        // the enumerator must survive the allocations below (key
        // interning, visited-array growth): read it through the handle
        // at every use, never a raw snapshot
        let enumerator = scope.handle(enumerator);
        let (vm, heap, _) = nctx.split();
        loop {
            // one candidate per turn: the cursor advances before the
            // key is examined, so skipped keys are never revisited
            let candidate = heap.no_gc(|nogc| -> Result<Option<Value>, VmError> {
                let Some(obj) = enumerator.value().as_heap_object(nogc) else {
                    return Err(VmError::Type);
                };
                let slots = obj.as_ref().slots.heap_ref(nogc);
                let keys = slots
                    .at(FOR_IN_KEYS)
                    .get_as::<FixedArray>(nogc)
                    .ok_or(VmError::Type)?;
                let index = Smi::decode(slots.at(FOR_IN_INDEX))
                    .ok_or(VmError::Type)?
                    .value() as usize;
                let Some(key) = (index < keys.len()).then(|| keys.at(index)) else {
                    return Ok(None);
                };
                slots.set(nogc, FOR_IN_INDEX, Smi::new(index as i64 + 1).encode());
                Ok(Some(key))
            })?;
            let Some(key) = candidate else {
                // snapshot exhausted: advance to the live prototype
                let level = heap.no_gc(|nogc| -> Result<Value, VmError> {
                    let Some(obj) = enumerator.value().as_heap_object(nogc) else {
                        return Err(VmError::Type);
                    };
                    Ok(obj.as_ref().slots.heap_ref(nogc).at(FOR_IN_LEVEL))
                })?;
                let Some(proto) = for_in_next_level(vm, heap, level)? else {
                    return Ok(heap.known().undefined.value());
                };
                // for_in_level_keys allocates (interning): keep the new
                // level rooted across it
                let proto = scope.handle(proto);
                let keys = for_in_level_keys(vm, heap, &scope, proto.value())?;
                let keys = heap.allocate_handle::<FixedArray>(scope.stage(&keys), &scope);
                heap.no_gc(|nogc| -> Result<(), VmError> {
                    let Some(obj) = enumerator.value().as_heap_object(nogc) else {
                        return Err(VmError::Type);
                    };
                    let slots = obj.as_ref().slots.heap_ref(nogc);
                    slots.set(nogc, FOR_IN_LEVEL, proto.value());
                    slots.set(nogc, FOR_IN_KEYS, keys.value());
                    slots.set(nogc, FOR_IN_INDEX, Smi::new(0).encode());
                    Ok(())
                })?;
                continue;
            };
            // the candidate must survive the visited-array growth below
            let key = scope.handle(key);
            // lazy [[GetOwnProperty]] on the key's own level: a key
            // deleted since the snapshot is skipped without registering
            let level = heap.no_gc(|nogc| -> Result<Value, VmError> {
                let Some(obj) = enumerator.value().as_heap_object(nogc) else {
                    return Err(VmError::Type);
                };
                Ok(obj.as_ref().slots.heap_ref(nogc).at(FOR_IN_LEVEL))
            })?;
            let own = heap.no_gc(|nogc| for_in_own_state(nogc, level, key.value()));
            let Some(enumerable) = own else {
                continue;
            };
            // already registered (yielded earlier, or shadowing
            // non-enumerable on a closer level): skip
            let seen = heap.no_gc(|nogc| -> Result<bool, VmError> {
                let Some(obj) = enumerator.value().as_heap_object(nogc) else {
                    return Err(VmError::Type);
                };
                let visited = obj
                    .as_ref()
                    .slots
                    .heap_ref(nogc)
                    .at(FOR_IN_VISITED)
                    .get_as::<FixedArray>(nogc)
                    .ok_or(VmError::Type)?;
                Ok(visited.as_slice().iter().any(|s| s.inner() == key.value()))
            })?;
            if seen {
                continue;
            }
            // register the key — yielded or shadowing, both at most once
            {
                let visited = heap.no_gc(|nogc| -> Result<Vec<Value>, VmError> {
                    let Some(obj) = enumerator.value().as_heap_object(nogc) else {
                        return Err(VmError::Type);
                    };
                    Ok(obj
                        .as_ref()
                        .slots
                        .heap_ref(nogc)
                        .at(FOR_IN_VISITED)
                        .get_as::<FixedArray>(nogc)
                        .ok_or(VmError::Type)?
                        .as_slice()
                        .iter()
                        .map(|s| s.inner())
                        .collect())
                })?;
                let mut visited = visited;
                visited.push(key.value());
                let visited = heap.allocate_handle::<FixedArray>(scope.stage(&visited), &scope);
                heap.no_gc(|nogc| -> Result<(), VmError> {
                    let Some(obj) = enumerator.value().as_heap_object(nogc) else {
                        return Err(VmError::Type);
                    };
                    obj.as_ref()
                        .slots
                        .heap_ref(nogc)
                        .set(nogc, FOR_IN_VISITED, visited.value());
                    Ok(())
                })?;
            }
            if !enumerable {
                continue;
            }
            return Ok(key.value());
        }
    })
}

/// The next level of the prototype chain: an object's live [[Prototype]]
/// (read at advance time, so mutations between iterations are visible),
/// or `String.prototype` for a string primitive level. Multi-parent
/// (Self-style) and null prototypes end the walk.
fn for_in_next_level(_vm: &VM, heap: &mut Heap, level: Value) -> Result<Option<Value>, VmError> {
    heap.no_gc(|nogc| {
        if level.get_as::<DenseString>(nogc).is_some() {
            // String.prototype via the global object (both plain data
            // lookups; no user code can run)
            let global = nogc.known().global_object.value();
            let string_name = SlotName::from_value(nogc.known().strings.string.value());
            let Some(string_ctor) = crate::lookup::load_outcome(nogc, global, string_name)
                .ok()
                .and_then(|o| match o {
                    crate::LoadOutcome::Value(v) => Some(v),
                    crate::LoadOutcome::Getter(_) => None,
                })
            else {
                return Ok(None);
            };
            let proto_name = SlotName::from_value(nogc.known().strings.prototype.value());
            let proto = crate::lookup::load_outcome(nogc, string_ctor, proto_name)
                .ok()
                .and_then(|o| match o {
                    crate::LoadOutcome::Value(v) => Some(v),
                    crate::LoadOutcome::Getter(_) => None,
                });
            return Ok(proto.filter(|p| p.is_strong_ptr()));
        }
        let Some(obj) = level.as_heap_object(nogc) else {
            return Ok(None);
        };
        let proto = obj.as_ref().map_ref(nogc).prototype.inner();
        if proto == nogc.known().the_hole.value() || proto == nogc.known().null.value() {
            return Ok(None);
        }
        // a FixedArray prototype is the Self-style multi-parent form;
        // the chain walk does not model it (ends the enumeration)
        Ok(proto
            .get_as::<FixedArray>(nogc)
            .map_or(Some(proto), |_| None))
    })
}

/// The lazy own-property state of `key` on its own level: `None` when
/// the property is gone (deleted since the snapshot), else its
/// [[Enumerable]]. Own-only — the shadow check against other levels is
/// the visited set's job.
fn for_in_own_state<'a>(nogc: &'a NoGc<'a>, level: Value, key: Value) -> Option<bool> {
    match crate::lookup::classify_key(nogc, key).ok()? {
        crate::Key::Element(i) => {
            if let Some(s) = level.get_as::<DenseString>(nogc) {
                // string indices are enumerable own properties
                return Some((i as u64) < s.len() as u64);
            }
            let obj = level.as_heap_object(nogc)?;
            let obj = obj.as_ref();
            if obj.is_array(nogc) {
                return obj.element_value(nogc, i).is_some().then_some(true);
            }
            // plain objects keep index keys as Smi-named descriptors
            let name = SlotName::from(Tagged::from_smi(Smi::new(i as i64)));
            obj.map_ref(nogc)
                .descriptors()
                .iter()
                .find(|d| d.name() == name)
                .map(|d| d.flags().is_enumerable())
        }
        crate::Key::Name(name) => {
            let obj = level.as_heap_object(nogc)?;
            let obj = obj.as_ref();

            // arrays hold "length" outside the descriptors (never a
            // snapshot key) — any other name lives in them
            obj.map_ref(nogc)
                .descriptors()
                .iter()
                .find(|d| d.name() == name)
                .map(|d| d.flags().is_enumerable())
        }
    }
}

/// GetIterator (ES 8.5.4): (obj) -> iterator.
fn get_iterator(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let obj = args.get(0).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let symbol = heap.known().iterator_symbol.value();
    let method = crate::runtime::Runtime::get_property(vm, heap, state, obj, symbol)?;
    let method = match method {
        Coercion::Threw => return Ok(nctx.heap().known().exception.value()),
        Coercion::Value(v) => v,
    };
    if method == nctx.heap().known().undefined.value()
        || method == nctx.heap().known().null.value()
        || !crate::runtime::Runtime::is_callable(nctx.heap(), method)
    {
        return Err(VmError::Type); // "obj is not iterable"
    }
    nctx.handle_scope(|nctx, scope| nctx.call(method, scope.stage(&[obj])))
}

/// IteratorNext (ES 8.5.6): (iterator) -> result object.
fn iterator_next(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let iter = args.get(0).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let next_name = heap.known().strings.next.value();
    let next = crate::runtime::Runtime::get_property(vm, heap, state, iter, next_name)?;
    let next = match next {
        Coercion::Threw => return Ok(nctx.heap().known().exception.value()),
        Coercion::Value(v) => v,
    };
    let result = nctx.handle_scope(|nctx, scope| nctx.call(next, scope.stage(&[iter])))?;
    if result == nctx.heap().known().exception.value() {
        return Ok(result);
    }
    if nctx
        .heap()
        .no_gc(|nogc| Convert::is_primitive(nogc, result))
    {
        return Err(VmError::Type); // IteratorNext result must be an Object
    }
    Ok(result)
}

/// IteratorComplete (ES 8.5.7): (result) -> bool.
fn iterator_done(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let result = args.get(0).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let done_name = heap.known().strings.done.value();
    let done = crate::runtime::Runtime::get_property(vm, heap, state, result, done_name)?;
    match done {
        Coercion::Threw => Ok(nctx.heap().known().exception.value()),
        Coercion::Value(v) => {
            let truthy = nctx.heap().no_gc(|nogc| Convert::is_truthy(nogc, v));
            Ok(Convert::boolean(nctx.heap(), truthy))
        }
    }
}

/// IteratorValue (ES 8.5.8): (result) -> value.
fn iterator_value(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let result = args.get(0).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let value_name = heap.known().strings.value.value();
    match crate::runtime::Runtime::get_property(vm, heap, state, result, value_name)? {
        Coercion::Threw => Ok(nctx.heap().known().exception.value()),
        Coercion::Value(v) => Ok(v),
    }
}

/// The `in` operator (ES 14.11.2): (key, obj) -> bool. Proxy receivers
/// run their `has` trap (ES 20.2.5.9).
fn has_property(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let raw_key = args.get(0).ok_or(VmError::Arity)?;
    let raw_obj = args.get(1).ok_or(VmError::Arity)?;
    // the key coercion allocates: root the receiver across it
    nctx.handle_scope(|nctx, scope| {
        let obj = scope.handle(raw_obj);
        let obj = obj.value();
        let (vm, heap, state) = nctx.split();
        let Some(key) = crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(nctx.heap().known().exception.value());
        };
        if heap.no_gc(|nogc| crate::proxy::is_proxy(nogc, obj)) {
            return match crate::proxy::has(vm, heap, state, obj, key)? {
                Coercion::Threw => Ok(heap.known().exception.value()),
                Coercion::Value(v) => Ok(v),
            };
        }
        // lookup::has_property covers array `length` slots along the chain
        let has = heap.no_gc(|nogc| crate::lookup::has_property(nogc, obj, SlotName::from_value(key)));
        Ok(Convert::boolean(nctx.heap(), has))
    })
}

/// CopyDataProperties (ES 8.5.1) with an exclusion list (object rest):
/// (excluded..., target, source); `excluded` has count−2 entries.
fn copy_data_properties(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let n = args.len();
    if n < 2 {
        return Err(VmError::Arity);
    }
    let target = args.as_slice()[n - 2];
    let source = args.as_slice()[n - 1];
    let excluded: &[Value] = &args.as_slice()[..n - 2];
    let nullish = nctx.heap().no_gc(|nogc| {
        source == nogc.known().null.value() || source == nogc.known().undefined.value()
    });
    if nullish {
        return Ok(target);
    }
    // only heap objects contribute (string sources need boxing)
    if nctx
        .heap()
        .no_gc(|nogc| Convert::is_primitive(nogc, source))
    {
        return Ok(target);
    }
    let (threw, target_out) = nctx.handle_scope(|nctx, scope| -> Result<(bool, Value), VmError> {
        // target and source survive getter calls and property adds below:
        // root them once, not per iteration from raw copies
        let target_handle = scope.handle(target);
        let source_handle = scope.handle(source);
        // canonicalize the excluded keys (interning strings) so a plain
        // bits comparison suffices against the source's descriptor names
        let excluded: Vec<Value> = {
            let (vm, heap, state) = nctx.split();
            let mut out = Vec::with_capacity(excluded.len());
            for &k in excluded {
                match crate::runtime::Runtime::to_property_key(vm, heap, state, k)? {
                    Some(k) => out.push(k),
                    None => return Ok((true, target_handle.value())),
                }
            }
            out
        };
        // enumerate own enumerable keys: element indices ascending, then
        // named descriptors in insertion order; collected AFTER the
        // exclusion canonicalization so no allocation can stale them
        let mut keys: Vec<Value> = Vec::new();
        nctx.heap().no_gc(|nogc| {
            let Some(obj) = source_handle.value().as_heap_object(nogc) else {
                return;
            };
            if obj.as_ref().is_array(nogc) {
                let len = obj.as_ref().length().min(
                    obj.as_ref()
                        .elements_array(nogc)
                        .map(|e| e.len())
                        .unwrap_or(0),
                );
                for i in 0..len {
                    if obj.as_ref().element_value(nogc, i).is_some() {
                        keys.push(Smi::new(i as i64).encode());
                    }
                }
            }
            for d in obj.as_ref().header.map.heap_ref(nogc).descriptors() {
                if d.flags().is_enumerable() {
                    keys.push(d.name().value());
                }
            }
        });
        let (vm, heap, state) = nctx.split();
        for key in keys {
            if excluded.contains(&key) {
                continue;
            }
            let key = scope.handle(key);
            // full [[Get]] (getters may run)
            let value = match crate::runtime::Runtime::get_property(
                vm,
                heap,
                state,
                source_handle.value(),
                key.value(),
            )? {
                Coercion::Threw => return Ok((true, target_handle.value())),
                Coercion::Value(v) => v,
            };
            // CreateDataProperty: skipped when already present
            let exists = heap.no_gc(|nogc| {
                !matches!(
                    target_handle.value().lookup(nogc, SlotName::from_value(key.value())),
                    Lookup::NotFound
                )
            });
            if exists {
                continue;
            }
            let value = scope.handle(value);
            Object::add_own_property_values(
                heap,
                &scope,
                target_handle.value(),
                SlotName::from_value(key.value()),
                PropertyDescriptor::data(value.value()),
            )?;
        }
        Ok((false, target_handle.value()))
    })?;
    if threw {
        return Ok(nctx.heap().known().exception.value());
    }
    // re-read through the handle: the copy loop allocated (getters,
    // property adds) and may have moved the target
    Ok(target_out)
}

/// A fresh private name: (description) -> Symbol.
fn create_private_name(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let desc = args.get(0);
    let text = nctx.heap().no_gc(|nogc| {
        desc.and_then(|d| d.get_as::<DenseString>(nogc))
            .map(|s| s.to_rust_string(nogc))
    });
    nctx.handle_scope(|nctx, scope| {
        let desc = text.unwrap_or_default();
        Ok(Symbol::new(nctx.heap(), &scope, desc.as_bytes())
            .as_tagged()
            .erase())
    })
}

/// PrivateGet (ES 7.3.30): (obj, key) -> value, TypeError when absent.
fn private_get(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let obj = args.get(0).ok_or(VmError::Arity)?;
    let key = args.get(1).ok_or(VmError::Arity)?;
    match nctx
        .heap()
        .no_gc(|nogc| private_find(nogc, obj, key).map(|s| s.inner()))
    {
        Some(v) => Ok(v),
        None => Err(VmError::Type),
    }
}

/// PrivateSet (ES 7.3.31): (obj, key, value), TypeError when absent.
fn private_set(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let obj = args.get(0).ok_or(VmError::Arity)?;
    let key = args.get(1).ok_or(VmError::Arity)?;
    let value = args.get(2).ok_or(VmError::Arity)?;
    let stored = nctx
        .heap()
        .no_gc(|nogc| match private_find(nogc, obj, key) {
            Some(slot) => {
                slot.set(nogc, obj, value);
                true
            }
            None => false,
        });
    if !stored {
        return Err(VmError::Type);
    }
    Ok(value)
}

/// `#x in obj`: (key, obj) -> bool (own private presence only).
fn private_in(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let key = args.get(0).ok_or(VmError::Arity)?;
    let obj = args.get(1).ok_or(VmError::Arity)?;
    let has = nctx
        .heap()
        .no_gc(|nogc| private_find(nogc, obj, key).is_some());
    Ok(Convert::boolean(nctx.heap(), has))
}

/// Attach the instance-field array to the class constructor:
/// (ctor, fields).
fn set_class_fields(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let ctor = args.get(0).ok_or(VmError::Arity)?;
    let fields = args.get(1).ok_or(VmError::Arity)?;
    let ok = nctx.heap().no_gc(|nogc| {
        let Some(obj) = ctor.as_heap_object(nogc) else {
            return false;
        };
        if !obj
            .as_ref()
            .header
            .map
            .heap_ref(nogc)
            .kind()
            .is_class_constructor()
        {
            return false;
        }
        let slots = obj.as_ref().slots.heap_ref(nogc);
        if slots.len() < 3 {
            return false;
        }
        slots.as_ref().element_slot(2).set(nogc, ctor, fields);
        true
    });
    if !ok {
        return Err(VmError::Type);
    }
    Ok(ctor)
}

/// InitializeInstanceElements (ES 7.3.33): (ctor, instance) -> instance.
/// Runs each field initializer with the instance as receiver and defines
/// the result onto it ({w+, e+, c+}).
fn init_instance_fields(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let ctor = args.get(0).ok_or(VmError::Arity)?;
    let instance = args.get(1).ok_or(VmError::Arity)?;
    let fields = nctx.heap().no_gc(|nogc| {
        let Some(obj) = ctor.as_heap_object(nogc) else {
            return None;
        };
        let slots = obj.as_ref().slots.heap_ref(nogc);
        (slots.len() >= 3).then(|| slots.at(2))
    });
    let Some(fields) = fields else {
        return Err(VmError::Type);
    };
    if fields == nctx.heap().known().undefined.value() {
        return Ok(instance);
    }
    let count = nctx.heap().no_gc(|nogc| {
        fields
            .as_heap_object(nogc)
            .map(|o| o.as_ref().length())
            .unwrap_or(0)
    });
    let threw_or_failed = nctx.handle_scope(|nctx, scope| -> Result<bool, VmError> {
        let instance = scope.handle(instance);
        let fields = scope.handle(fields);
        let exception = nctx.heap().known().exception.value();
        let mut i = 0;
        while i + 1 < count {
            let raw_key = nctx.heap().no_gc(|nogc| {
                fields
                    .value()
                    .as_heap_object(nogc)
                    .and_then(|o| o.as_ref().element_value(nogc, i))
                    .unwrap_or_else(|| nogc.known().undefined.value())
            });
            let init = nctx.heap().no_gc(|nogc| {
                fields
                    .value()
                    .as_heap_object(nogc)
                    .and_then(|o| o.as_ref().element_value(nogc, i + 1))
                    .unwrap_or_else(|| nogc.known().undefined.value())
            });
            // computed keys need ToPropertyKey canonicalization
            let key = {
                let (vm, heap, state) = nctx.split();
                match crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? {
                    Some(k) => k,
                    None => return Ok(true),
                }
            };
            // the initializer call allocates (user code): the key must
            // stay rooted across it
            let key = scope.handle(key);
            let value = nctx.call(init, scope.stage(&[instance.value()]))?;
            if value == exception {
                return Ok(true);
            }
            let value = scope.handle(value);
            let defined = {
                let heap = nctx.heap();
                Object::define_own_property_values(
                    heap,
                    &scope,
                    instance.value(),
                    SlotName::from_value(key.value()),
                    PropertyDescriptor::Data {
                        value: value.value(),
                        writable: true,
                        enumerable: true,
                        configurable: true,
                    },
                )?
            };
            if !defined {
                return Err(VmError::Type);
            }
            i += 2;
        }
        Ok(false)
    })?;
    if threw_or_failed {
        return Ok(nctx.heap().known().exception.value());
    }
    Ok(instance)
}

// ---- frame access -----------------------------------------------------------

/// The current (calling) frame's context: `CallRuntime` runs in place, so
/// the interpreter's cache still holds the frame executing the call.
fn frame_context_value(state: &ContextState) -> Result<Value, VmError> {
    if !state.cache.is_active() {
        return Err(VmError::Type);
    }
    Ok(state.stack.context_slot(&state.cache.frame_meta()).inner())
}

/// Find the slot named `name` in `context`'s chain (direct eval). Returns
/// the slot cell, or Reference when no context in the chain has the name.
fn dynamic_slot<'a>(
    nogc: &'a NoGc<'a>,
    context: &mut HeapRef<'a, Context>,
    name: Value,
) -> Result<&'a crate::GcSlot, VmError> {
    // both sides are interned (constant pool / ScopeInfo names), so
    // pointer identity decides — no content comparison in lookup
    name.get_as::<DenseString>(nogc).ok_or(VmError::Type)?;
    loop {
        let ctx = context.as_ref();
        let names = ctx.scope_info.heap_ref(nogc).as_ref().names.heap_ref(nogc);
        for i in 0..names.len() {
            if names.at(i) == name {
                return Ok(ctx.slots.heap_ref(nogc).as_ref().element_slot(i));
            }
        }
        match ctx.outer.heap_ref(nogc) {
            Some(outer) => *context = outer,
            None => return Err(VmError::Reference),
        }
    }
}

/// Walk the current frame's context chain looking for a slot named
/// `name` (direct eval): Some(slot value) found (possibly the hole),
/// None when the whole chain lacks the name.
fn dynamic_lookup_frame(
    heap: &mut Heap,
    state: &ContextState,
    name: Value,
) -> Result<Option<Value>, VmError> {
    let context = frame_context_value(state)?;
    heap.no_gc(|nogc| {
        let mut context = context.get_as::<Context>(nogc).ok_or(VmError::Type)?;
        match dynamic_slot(nogc, &mut context, name) {
            Ok(slot) => Ok(Some(slot.inner())),
            Err(VmError::Reference) => Ok(None),
            Err(e) => Err(e),
        }
    })
}

/// The current frame's super constructor and new.target (direct
/// super() calls, ES 15.4.3): the running closure's [[Prototype]] must
/// be a constructor.
fn frame_super_parts(heap: &mut Heap, state: &ContextState) -> Result<(Value, Value), VmError> {
    if !state.cache.is_active() {
        return Err(VmError::Type);
    }
    let meta = state.cache.frame_meta();
    heap.no_gc(|nogc| {
        let Some(callee) = super_constructor(nogc, &state.stack, &meta) else {
            return Err(VmError::Type);
        };
        Ok((callee, state.stack.new_target_slot(&meta).inner()))
    })
}

// ---- store outcomes ---------------------------------------------------------

/// Apply a store outcome: transitions add the property on the receiver,
/// setters are invoked with (receiver, value). Returns `true` when a
/// setter threw (the pending exception is set; the caller propagates the
/// exception sentinel).
fn apply_store_outcome(
    nctx: &mut NativeContext<'_>,
    receiver: Value,
    outcome: StoreOutcome,
    value: Value,
) -> Result<bool, VmError> {
    match outcome {
        StoreOutcome::Transition { receiver, name } => {
            nctx.handle_scope(|nctx, scope| {
                Object::add_own_property_values(
                    nctx.heap(),
                    &scope,
                    receiver,
                    name,
                    PropertyDescriptor::data(value),
                )
                // TODO(strict-mode): a false result must throw in strict code;
                // the current store path preserves its existing sloppy result.
                .map(|_| false)
            })
        }
        StoreOutcome::CallSetter { setter } => {
            let exception = nctx.heap().known().exception.value();
            let result = nctx
                .handle_scope(|nctx, scope| nctx.call(setter, scope.stage(&[receiver, value])))?;
            Ok(result == exception)
        }
        StoreOutcome::Done => Ok(false),
    }
}

/// A full [[Get]] that treats non-callable getters (an absent half of an
/// accessor pair) as undefined instead of throwing.
fn get_property_lenient(
    nctx: &mut NativeContext<'_>,
    receiver: Value,
    name: Value,
) -> Result<Value, VmError> {
    let outcome = nctx
        .heap()
        .no_gc(|nogc| load_outcome(nogc, receiver, SlotName::from_value(name)))?;
    match outcome {
        LoadOutcome::Value(v) => Ok(v),
        LoadOutcome::Getter(getter) => {
            let undefined = nctx.heap().known().undefined.value();
            if getter == undefined || !crate::runtime::Runtime::is_callable(nctx.heap(), getter) {
                return Ok(undefined);
            }
            nctx.handle_scope(|nctx, scope| nctx.call(getter, scope.stage(&[receiver])))
        }
    }
}

// ---- class definition helpers -----------------------------------------------

/// SetFunctionName (ES 8.4.4): (fn, key, prefix) -> fn. Redefines `name`
/// on the closure ({w−, e−, c+}); the prefix discriminant (a Smi) is
/// 0 none, 1 "get ", 2 "set ". Class members named `name` define over the
/// constructor after ClassDefinitionEvaluation set its name — an already
/// explicitly defined `name` wins (ES 15.7.14: SetFunctionName happens
/// before element installation).
fn set_function_name(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let fn_value = args.get(0).ok_or(VmError::Arity)?;
    let raw_key = args.get(1).ok_or(VmError::Arity)?;
    let prefix = Smi::decode(args.get(2).ok_or(VmError::Arity)?)
        .map(|s| s.value())
        .unwrap_or(0);
    // name construction allocates (interning): the closure must stay
    // rooted across it
    nctx.handle_scope(|nctx, scope| {
        let fn_value = scope.handle(fn_value);
        let (vm, heap, state) = nctx.split();
        let Some(key) = crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(nctx.heap().known().exception.value());
        };
        let units = state.handle_scope(|scope| -> Result<Vec<u16>, VmError> {
            let text = Convert::to_string(heap, &scope, key)?;
            Ok(heap.no_gc(|nogc| {
                let s = text
                    .get_as::<DenseString>(nogc)
                    .expect("ToString yields a string")
                    .as_ref();
                let mut full: Vec<u16> = match prefix {
                    1 => b"get ".iter().map(|&b| b as u16).collect(),
                    2 => b"set ".iter().map(|&b| b as u16).collect(),
                    _ => Vec::new(),
                };
                s.data(nogc).write_units(&mut full);
                full
            }))
        })?;
        let name = state.handle_scope(|scope| {
            vm.interner()
                .intern(heap, &scope, StringData::Utf16(&units))
                .value()
        });
        let defined = {
            let Some(fn_obj) = scope.cast::<Object>(fn_value.value()) else {
                return Err(VmError::Type);
            };
            let name_key = heap.known().strings.name;
            // the closure's own placeholder is never writable nor an accessor,
            // so only explicit member defines match here
            let explicit = heap.no_gc(|nogc| {
                let plain = SlotName::from_value(name_key.value());
                match fn_obj.heap_ref(nogc).as_ref().lookup(nogc, plain) {
                    Lookup::Data { flags, .. } => flags.is_writable(),
                    Lookup::Accessor { .. } => true,
                    Lookup::NotFound => false,
                }
            });
            let defined = if explicit {
                true
            } else {
                Object::define_own_property(
                    heap,
                    &scope,
                    fn_obj,
                    name_key,
                    PropertyDescriptor::Data {
                        value: name,
                        writable: false,
                        enumerable: false,
                        configurable: true,
                    },
                )?
            };
            defined
        };
        if !defined {
            return Err(VmError::Type);
        }
        Ok(fn_value.value())
    })
}

/// Accessor member installation (ES 14.3.10): (target, key, closure,
/// flags). Defines one accessor half, merging with an existing pair under
/// the same key; flags bit 0 marks the getter half, PropertyFlags bits
/// carry enumerability.
fn install_accessor(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let raw_target = args.get(0).ok_or(VmError::Arity)?;
    let raw_key = args.get(1).ok_or(VmError::Arity)?;
    let raw_closure = args.get(2).ok_or(VmError::Arity)?;
    let flags = Smi::decode(args.get(3).ok_or(VmError::Arity)?)
        .map(|s| s.value() as u32)
        .unwrap_or(0);
    // the key coercion allocates: root the target and closure across it
    nctx.handle_scope(|nctx, scope| {
        let target = scope.handle(raw_target);
        let closure = scope.handle(raw_closure);
        let target = target.value();
        let closure = closure.value();
        let (vm, heap, state) = nctx.split();
        let Some(key) = crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(nctx.heap().known().exception.value());
        };
        let is_getter = flags & 1 != 0;
        let enumerable = flags & bytecode::PropertyFlags::DontEnum.bits() == 0;
        let (name, desc) = heap.no_gc(|nogc| -> Result<_, VmError> {
            if target.as_heap_object(nogc).is_none() {
                return Err(VmError::Type);
            }
            let name = match classify_key(nogc, key)? {
                Key::Element(i) => SlotName::from(Tagged::from_smi(Smi::new(i as i64))),
                Key::Name(name) => name,
            };
            // existing own accessor half, if any (own descriptors only)
            let undefined = nogc.known().undefined.value();
            let mut get = undefined;
            let mut set = undefined;
            if let Some(obj) = target.as_heap_object(nogc) {
                for d in obj.as_ref().header.map.heap_ref(nogc).descriptors() {
                    if d.name() == name && d.flags().is_accessor() {
                        let pair = d
                            .value
                            .inner()
                            .get_as::<AccessorPair>(nogc)
                            .expect("accessor descriptor holds a pair");
                        let pair = pair.as_ref();
                        get = pair.get.inner();
                        set = pair.set.inner();
                        break;
                    }
                }
            }
            if is_getter {
                get = closure;
            } else {
                set = closure;
            }
            Ok((
                name,
                PropertyDescriptor::Accessor {
                    get,
                    set,
                    enumerable,
                    configurable: true,
                },
            ))
        })?;
        let defined = Object::define_own_property_values(heap, &scope, target, name, desc)?;
        if !defined {
            return Err(VmError::Type);
        }
        Ok(closure)
    })
}

/// [[DefineOwnProperty]] with exact attributes (class member
/// installation): (obj, key, value, flags) -> obj. Define sites are
/// strict-mode code: a rejected define throws a TypeError. flags are
/// PropertyFlags bits (the Accessor bit: the value is an AccessorPair).
fn define_own_property(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let raw_receiver = args.get(0).ok_or(VmError::Arity)?;
    let raw_key = args.get(1).ok_or(VmError::Arity)?;
    let raw_value = args.get(2).ok_or(VmError::Arity)?;
    let flags = Smi::decode(args.get(3).ok_or(VmError::Arity)?)
        .map(|s| s.value() as u32)
        .unwrap_or(0);
    // the key coercion allocates: root the receiver and value across it
    nctx.handle_scope(|nctx, scope| {
        let receiver = scope.handle(raw_receiver);
        let value = scope.handle(raw_value);
        let receiver = receiver.value();
        let value = value.value();
        let (vm, heap, state) = nctx.split();
        let Some(key) = crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(nctx.heap().known().exception.value());
        };
        // proxies run their `defineProperty` trap (ES 20.2.5.6); define
        // sites are strict-mode: a rejected define throws
        if heap.no_gc(|nogc| crate::proxy::is_proxy(nogc, receiver)) {
            let partial = heap.no_gc(|nogc| -> Result<crate::PartialDescriptor, VmError> {
                let enumerable = flags & bytecode::PropertyFlags::DontEnum.bits() == 0;
                let configurable = flags & bytecode::PropertyFlags::DontDelete.bits() == 0;
                if flags & bytecode::PropertyFlags::Accessor.bits() != 0 {
                    let pair = value.get_as::<AccessorPair>(nogc).ok_or(VmError::Type)?;
                    let pair = pair.as_ref();
                    Ok(crate::PartialDescriptor {
                        value: None,
                        get: Some(pair.get.inner()),
                        set: Some(pair.set.inner()),
                        writable: None,
                        enumerable: Some(enumerable),
                        configurable: Some(configurable),
                    })
                } else {
                    Ok(crate::PartialDescriptor {
                        value: Some(value),
                        get: None,
                        set: None,
                        writable: Some(flags & bytecode::PropertyFlags::ReadOnly.bits() == 0),
                        enumerable: Some(enumerable),
                        configurable: Some(configurable),
                    })
                }
            })?;
            return match crate::proxy::define_internal(vm, heap, state, receiver, key, partial)? {
                crate::proxy::Flow::Threw => Ok(heap.known().exception.value()),
                crate::proxy::Flow::Value(false) => Err(VmError::Type),
                crate::proxy::Flow::Value(true) => Ok(receiver),
            };
        }
        let (name, desc) = heap.no_gc(|nogc| -> Result<_, VmError> {
            if receiver.as_heap_object(nogc).is_none() {
                return Err(VmError::Type);
            }
            let name = match classify_key(nogc, key)? {
                Key::Element(i) => SlotName::from(Tagged::from_smi(Smi::new(i as i64))),
                Key::Name(name) => name,
            };
            let enumerable = flags & bytecode::PropertyFlags::DontEnum.bits() == 0;
            let configurable = flags & bytecode::PropertyFlags::DontDelete.bits() == 0;
            let desc = if flags & bytecode::PropertyFlags::Accessor.bits() != 0 {
                let pair = value.get_as::<AccessorPair>(nogc).ok_or(VmError::Type)?;
                let pair = pair.as_ref();
                PropertyDescriptor::Accessor {
                    get: pair.get.inner(),
                    set: pair.set.inner(),
                    enumerable,
                    configurable,
                }
            } else {
                PropertyDescriptor::Data {
                    value,
                    writable: flags & bytecode::PropertyFlags::ReadOnly.bits() == 0,
                    enumerable,
                    configurable,
                }
            };
            Ok((name, desc))
        })?;
        let defined = Object::define_own_property_values(heap, &scope, receiver, name, desc)?;
        if !defined {
            return Err(VmError::Type);
        }
        Ok(receiver)
    })
}

/// [[SetPrototypeOf]] (class prototype wiring): (obj, proto) -> obj.
fn set_prototype(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let obj = args.get(0).ok_or(VmError::Arity)?;
    let proto = args.get(1).ok_or(VmError::Arity)?;
    let (_, heap, state) = nctx.split();
    state.handle_scope(|scope| Object::set_prototype(heap, &scope, obj, proto))?;
    Ok(obj)
}

/// Class extends validation (ES 15.7.14 step 15.e): (value) -> value,
/// TypeError unless the superclass is null or a constructor.
fn throw_if_not_constructor_or_null(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let v = args.get(0).ok_or(VmError::Arity)?;
    let ok = nctx.heap().no_gc(|nogc| {
        if v == nogc.known().null.value() {
            return true;
        }
        let Some(obj) = v.as_heap_object(nogc) else {
            return false;
        };
        obj.as_ref()
            .header
            .map
            .heap_ref(nogc)
            .kind()
            .is_constructor()
    });
    if !ok {
        return Err(VmError::Type);
    }
    Ok(v)
}

/// superCtor.prototype validation: (value) -> value, TypeError unless the
/// value is an Object or null.
fn throw_if_not_object_or_null(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let v = args.get(0).ok_or(VmError::Arity)?;
    let ok = nctx
        .heap()
        .no_gc(|nogc| v == nogc.known().null.value() || !Convert::is_primitive(nogc, v));
    if !ok {
        return Err(VmError::Type);
    }
    Ok(v)
}

/// [[ThisBindingStatus]] guard of derived constructors (ES 10.2.2):
/// (value) -> value, ReferenceError when `this` is still the hole.
fn throw_super_not_called_if_hole(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let v = args.get(0).ok_or(VmError::Arity)?;
    if v == nctx.heap().known().the_hole.value() {
        // "Must call super constructor before accessing 'this'"
        return Err(VmError::Reference);
    }
    Ok(v)
}

/// InitializeThisBinding guard (ES 10.2.2): (value) -> value,
/// ReferenceError unless `this` is still the hole (super() runs once).
fn throw_super_already_called_if_not_hole(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let v = args.get(0).ok_or(VmError::Arity)?;
    if v != nctx.heap().known().the_hole.value() {
        // "Super constructor may only be called once"
        return Err(VmError::Reference);
    }
    Ok(v)
}

// ---- super() construction (ES 15.4.3) ---------------------------------------

/// The shared ConstructSuper tail: construct `callee` with `new_target`,
/// giving derived parents the hole receiver. The instance lands in the
/// return value; the exception sentinel escapes when user code threw.
fn construct_super_construct(
    nctx: &mut NativeContext<'_>,
    callee_v: Value,
    new_target_v: Value,
    args: &[Value],
) -> Result<Value, VmError> {
    let undefined = nctx.heap().known().undefined.value();
    if new_target_v == undefined || new_target_v == nctx.heap().known().the_hole.value() {
        // not inside a [[Construct]]: reachable via an arrow that escaped
        // the constructor
        return Err(VmError::Type);
    }
    let derived = nctx.heap().no_gc(|nogc| {
        function_kind_of(nogc, callee_v).is_some_and(|k| k.is_derived_class_constructor())
    });
    nctx.handle_scope(|nctx, scope| {
        let Some(callee) = scope.cast::<Object>(callee_v) else {
            return Err(VmError::Type);
        };
        let Some(new_target) = scope.cast::<Object>(new_target_v) else {
            return Err(VmError::Type);
        };
        // root the forwarded arguments before anything allocates: they
        // are raw copies of caller stack slots and go stale when a GC
        // moves their targets (create_construct_receiver allocates)
        let args: Vec<Handle<'_, Value>> = args.iter().map(|v| scope.handle(*v)).collect();
        let (receiver, allocated) = if derived {
            (
                scope.handle(nctx.heap().known().the_hole.value()),
                false,
            )
        } else {
            let (vm, heap, state) = nctx.split();
            match crate::runtime::Runtime::create_construct_receiver(vm, heap, state, new_target) {
                // the receiver must stay rooted across the callee call:
                // a primitive return falls back to it after the call
                // allocated (and possibly moved it)
                Ok(Some(r)) => (scope.handle(r), true),
                Ok(None) => return Ok(nctx.heap().known().exception.value()),
                Err(err) => return Err(err),
            }
        };
        let mut args_v = Vec::with_capacity(args.len() + 1);
        args_v.push(receiver.value());
        args_v.extend(args.iter().map(|h| h.value()));
        let result = {
            let (vm, heap, state) = nctx.split();
            NativeContext::new(vm, heap, state).call_construct(
                callee.value(),
                new_target.value(),
                scope.stage(&args_v),
            )
        }?;
        if result == nctx.heap().known().exception.value() {
            return Ok(result);
        }
        Ok(if Convert::is_primitive(&nctx.heap().guard(), result) {
            if allocated {
                receiver.value()
            } else {
                // a derived constructor returned a primitive: only
                // reachable via `return <primitive>` (ES 9.2.2.1)
                return Err(VmError::Type);
            }
        } else {
            result
        })
    })
}

/// super(...): (args...) -> instance. Resolves the super constructor and
/// new.target from the executing frame.
fn construct_super(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let (callee, new_target) = {
        let (_, heap, state) = nctx.split();
        frame_super_parts(heap, state)?
    };
    construct_super_construct(nctx, callee, new_target, args.as_slice())
}

/// super() forwarding the frame's full argument list (synthesized default
/// derived constructors, ES 15.7.13): () -> instance.
fn construct_super_all_args(
    nctx: &mut NativeContext<'_>,
    _args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let (callee, new_target, args) = {
        let (_, heap, state) = nctx.split();
        let (callee, new_target) = frame_super_parts(heap, state)?;
        if !state.cache.is_active() {
            return Err(VmError::Type);
        }
        let meta = state.cache.frame_meta();
        let argc = state.stack.argc(&meta).saturating_sub(1);
        let args = state.stack.args(&meta, -2, argc).as_slice().to_vec();
        (callee, new_target, args)
    };
    construct_super_construct(nctx, callee, new_target, &args)
}

/// Arrow-delegated super(): (args..., closure, new_target) -> instance.
/// The constructor closure and its new.target ride the tail of the
/// argument window (threaded through .this_function).
fn construct_super_via(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let n = args.len();
    if n < 2 {
        return Err(VmError::Arity);
    }
    let closure = args.as_slice()[n - 2];
    let new_target = args.as_slice()[n - 1];
    let callee = nctx.heap().no_gc(|nogc| {
        let Some(obj) = closure.as_heap_object(nogc) else {
            return None;
        };
        let proto = obj.as_ref().header.map.heap_ref(nogc).prototype.inner();
        let proto_obj = proto.as_heap_object(nogc)?;
        if !proto_obj
            .as_ref()
            .header
            .map
            .heap_ref(nogc)
            .kind()
            .is_constructor()
        {
            return None;
        }
        Some(proto)
    });
    let Some(callee) = callee else {
        return Err(VmError::Type);
    };
    construct_super_construct(nctx, callee, new_target, &args.as_slice()[..n - 2])
}

// ---- dynamic names (direct eval) ---------------------------------------------

/// Direct-eval name load: (name) -> value. Walks the frame context chain
/// by name; unresolved names fall back to the global object.
fn load_dynamic_name(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let name = args.get(0).ok_or(VmError::Arity)?;
    let found = {
        let (_, heap, state) = nctx.split();
        dynamic_lookup_frame(heap, state, name)?
    };
    match found {
        Some(v) if v != nctx.heap().known().the_hole.value() => Ok(v),
        Some(_) => Err(VmError::Reference),
        None => {
            // unresolved: fall back to a global object property
            let global = nctx.heap().known().global_object.value();
            get_property_lenient(nctx, global, name)
        }
    }
}

/// Direct-eval name store: (value, name) -> value. Writes through to the
/// context-chain slot; unresolved names store on the global object.
fn store_dynamic_name(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let value = args.get(0).ok_or(VmError::Arity)?;
    let name = args.get(1).ok_or(VmError::Arity)?;
    let found = {
        let (_, heap, state) = nctx.split();
        dynamic_lookup_frame(heap, state, name)?
    };
    match found {
        Some(v) if v != nctx.heap().known().the_hole.value() => {
            // write through to the found slot
            let (_, heap, state) = nctx.split();
            let context = frame_context_value(state)?;
            heap.no_gc(|nogc| {
                let mut context = context.get_as::<Context>(nogc).ok_or(VmError::Type)?;
                let target = dynamic_slot(nogc, &mut context, name)?;
                let host = context.into_tagged().erase();
                target.set(nogc, host, value);
                Ok(())
            })?;
        }
        Some(_) => return Err(VmError::Reference),
        None => {
            let global = nctx.heap().known().global_object.value();
            let outcome = nctx.heap().no_gc(|nogc| {
                global.store_lookup(
                    nogc,
                    SlotName::from_value(name),
                    value,
                    StoreSemantics::WriteThrough,
                )
            })?;
            if apply_store_outcome(nctx, global, outcome, value)? {
                return Ok(nctx.heap().known().exception.value());
            }
        }
    }
    Ok(value)
}

// ---- rest parameters ---------------------------------------------------------

/// A fresh array of the frame's arguments from formal index `first`:
/// (first) -> array.
fn create_rest_parameter(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let first = Smi::decode(args.get(0).ok_or(VmError::Arity)?)
        .map(|s| s.value() as usize)
        .unwrap_or(0);
    let values: Vec<Value> = {
        let (_, heap, state) = nctx.split();
        if !state.cache.is_active() {
            return Err(VmError::Type);
        }
        let meta = state.cache.frame_meta();
        let argc = state.stack.argc(&meta); // receiver included
        let count = argc.saturating_sub(1).saturating_sub(first);
        let _ = heap;
        (0..count)
            .map(|i| state.stack.reg(&meta, -((first + i + 2) as i32)))
            .collect()
    };
    let arr = nctx.handle_scope(|nctx, scope| {
        let heap = nctx.heap();
        let elements = heap.allocate_handle::<FixedArray>(scope.stage(&values), &scope);
        heap.allocate_object(
            &scope,
            ObjectSlotsInit {
                map: heap.known().js_array_map,
                values: GcSlice::EMPTY,
                elements: elements.erase(),
                length: values.len(),
            },
        )
        .into_tagged()
        .erase()
    });
    Ok(arr)
}

// ---- super property access (ES 15.4.2 / 15.4.4) -------------------------------

/// super.x load: (home, recv, key) -> value. GetSuperBase of the home
/// object walked with the split receiver/lookup-start; key coercion runs
/// after the parent link is resolved (user toString must not change the
/// chain searched).
fn super_get_property(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let home = args.get(0).ok_or(VmError::Arity)?;
    let raw_recv = args.get(1).ok_or(VmError::Arity)?;
    let raw_key = args.get(2).ok_or(VmError::Arity)?;
    if raw_recv == nctx.heap().known().the_hole.value() {
        // super.x before super() in a derived constructor
        return Err(VmError::Reference);
    }
    // the key coercion allocates: root home/recv across it
    nctx.handle_scope(|nctx, scope| {
        let home = scope.handle(home);
        let recv = scope.handle(raw_recv);
        let home = home.value();
        let recv = recv.value();
        let proto = nctx.heap().no_gc(|nogc| home_proto(nogc, home));
        let (vm, heap, state) = nctx.split();
        let Some(key) = crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(heap.known().exception.value());
        };
        let outcome = nctx.heap().no_gc(|nogc| {
            let name = match classify_key(nogc, key)? {
                Key::Element(i) => SlotName::from(Tagged::from_smi(Smi::new(i as i64))),
                Key::Name(name) => name,
            };
            super_lookup_from_proto(nogc, proto, name)
        })?;
        match outcome {
            LoadOutcome::Value(v) => Ok(v),
            LoadOutcome::Getter(getter) => {
                let undefined = nctx.heap().known().undefined.value();
                if getter == undefined || !crate::runtime::Runtime::is_callable(nctx.heap(), getter) {
                    return Ok(undefined);
                }
                nctx.call(getter, scope.stage(&[recv]))
            }
        }
    })
}

/// super.x store: (home, recv, key, value, semantics) -> value. ES stores
/// shadow inherited data properties on `this` unless the write-through
/// semantics flag is set; the parent link is resolved before any user key
/// coercion runs.
fn super_set_property(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let home = args.get(0).ok_or(VmError::Arity)?;
    let raw_recv = args.get(1).ok_or(VmError::Arity)?;
    let raw_key = args.get(2).ok_or(VmError::Arity)?;
    let raw_value = args.get(3).ok_or(VmError::Arity)?;
    let semantics_flag = Smi::decode(args.get(4).ok_or(VmError::Arity)?)
        .map(|s| s.value() as u32)
        .unwrap_or(0);
    if raw_recv == nctx.heap().known().the_hole.value() {
        return Err(VmError::Reference);
    }
    // the key coercion allocates: root home/recv/value across it
    nctx.handle_scope(|nctx, scope| {
        let home = scope.handle(home);
        let recv = scope.handle(raw_recv);
        let value = scope.handle(raw_value);
        let home = home.value();
        let recv = recv.value();
        let value = value.value();
        let proto = nctx.heap().no_gc(|nogc| home_proto(nogc, home));
        let (vm, heap, state) = nctx.split();
        let Some(key) = crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(heap.known().exception.value());
        };
        let semantics = if semantics_flag & bytecode::SUPER_STORE_WRITE_THROUGH != 0 {
            StoreSemantics::WriteThrough
        } else {
            StoreSemantics::Shadow
        };
        let outcome = nctx.heap().no_gc(|nogc| {
            let name = match classify_key(nogc, key)? {
                Key::Element(i) => SlotName::from(Tagged::from_smi(Smi::new(i as i64))),
                Key::Name(name) => name,
            };
            super_store_lookup(nogc, proto, recv, name, value, semantics)
        })?;
        if apply_store_outcome(nctx, recv, outcome, value)? {
            return Ok(nctx.heap().known().exception.value());
        }
        Ok(value)
    })
}
