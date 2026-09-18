//! The CallRuntime runtimes: one implementation per `bytecode::RuntimeFn`,
//! plus the `runtime_fn` table the runtime registry seeds its fixed
//! 0..COUNT range with. These are compiled-language semantics (each cites
//! its ES section), not JS-visible library functions.

use crate::{
    AccessorPair, Context, Convert, DenseString, FixedArray, Handle, HandleSlice, Heap, HeapRef,
    Key, LoadOutcome, Lookup, Object, ObjectSlotsInit, PropertyDescriptor, SlotName, Smi,
    StoreOutcome, StoreSemantics, StringData, Symbol, Tagged, Value, VmError, home_proto,
    private_find, runtime::Coercion, super_constructor, super_lookup_from_proto,
    super_store_lookup,
};

use crate::Float;
use crate::GcSlot;
use crate::HandleScope;
use crate::PartialDescriptor;
use crate::lookup::canonical_index;
use crate::lookup::has_property as lookup_has_property;
use crate::proxy::Flow;
use crate::proxy::Proxy;
use crate::{ContextState, VM};
use crate::{RuntimeCall, RuntimeContext};

/// The fixed runtime-helper table: one implementation per
/// `bytecode::RuntimeFn`. The exhaustive match is the compile-time link
/// between the ABI ids and their implementations — adding a variant
/// without an entry here is a compile error, and `RuntimeRegistry::new`
/// registers them in `RuntimeFn::ALL` order so registry indices equal
/// discriminants.
pub fn runtime_fn(id: bytecode::RuntimeFn) -> RuntimeCall {
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
fn require_object_coercible<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    // internal-runtime convention: the register list IS the argument list
    // (no receiver slot)
    let RuntimeContext { heap, .. } = nctx;
    let arg = args
        .get(0)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?;
    let null = heap.known().null.as_tagged(heap);
    let undefined = heap.known().undefined.as_tagged(heap);
    if arg.ptr_eq(null) || arg.ptr_eq(undefined) {
        return Err(VmError::Type);
    }
    Ok(arg)
}

// ---- delete (ES 13.5.1) ----------------------------------------------------

/// `delete obj.key` in sloppy code: (obj, key) -> bool.
fn delete_property_sloppy<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    delete_property(nctx, args, false)
}

/// `delete obj.key` in strict code: (obj, key) -> bool, TypeError when
/// the delete fails (ES 13.5.1.2 step 4.h).
fn delete_property_strict<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    delete_property(nctx, args, true)
}

fn delete_property<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
    strict: bool,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    // Safety: fresh argument words; nothing allocates before they are
    // rooted / consumed below.
    let (raw_target, raw_key) = {
        let heap = &*heap;
        (
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
        )
    };
    // the reference's key is coerced before the base is touched (ES
    // 13.15.5 EvaluatePropertyAccess: user toString/valueOf of a
    // computed key runs even when the delete afterwards throws);
    // the coercion allocates, so the base must stay rooted across it
    state.handle_scope(|scope| {
        // Safety: fresh argument word, rooted below before any allocation.
        let target = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_target) });
        let target = target.as_tagged(&*heap).raw();
        let Some(key) = Object::to_property_key(
            vm,
            heap,
            state,
            // Safety: fresh argument word, fresh at entry.
            scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_key) }),
        )?
        else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        // proxies run their `deleteProperty` trap (ES 20.2.5.4); the
        // returned boolean flows through the strict handling below
        let ok = {
            let cond_23 = {
                let heap = &*heap;
                Proxy::is_proxy(heap, unsafe { target.assume_valid(heap) })
            };
            if cond_23 {
                // Safety: fresh rooted-slot word (re-read above) plus a fresh
                // coercion result, both consumed by the trap call.
                let key_word = unsafe { key.read_unchecked() };
                match Proxy::delete(
                    vm,
                    heap,
                    state,
                    unsafe { Tagged::<Value>::from_value_unchecked(target) },
                    unsafe { Tagged::<Value>::from_value_unchecked(key_word) },
                )? {
                    Coercion::Threw => {
                        return Ok(heap.known().exception.as_tagged(heap).erase());
                    }
                    Coercion::Value(v) => {
                        let v = scope.handle(v);
                        Convert::is_truthy(heap, v.as_tagged(heap))
                    }
                }
            } else {
                // Safety: fresh rooted name word, consumed by the delete.
                delete_property_core(heap, &scope, target, unsafe { key.read_unchecked() })?
            }
        };
        if strict && !ok {
            return Err(VmError::Type);
        }
        Ok(Convert::boolean(heap, ok))
    })
}

fn delete_property_core(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    target: Value,
    key: Value,
) -> Result<bool, VmError> {
    // ToObject (ES 7.2.3): a null/undefined base throws
    let nullish = {
        let heap = &*heap;
        let null = heap.known().null.as_tagged(heap).raw();
        let undefined = heap.known().undefined.as_tagged(heap).raw();
        target == null || target == undefined
    };
    if nullish {
        return Err(VmError::Type);
    }
    // primitives: ToObject creates a fresh wrapper whose only own
    // properties are a string's non-configurable length/indices
    {
        let cond_24 = {
            let heap = &*heap;
            Convert::is_primitive(heap, unsafe { target.assume_valid(heap) })
        };
        if cond_24 {
            let owned = {
                let heap = &*heap;
                string_exotic_own(
                    heap,
                    // Safety: caller-supplied words, fresh at entry.
                    unsafe { target.assume_valid(heap) },
                    unsafe { key.assume_valid(heap) },
                )
            };
            return Ok(!owned);
        }
    }
    let receiver = scope
        .cast::<Object>(
            // Safety: caller-supplied word, fresh at entry.
            unsafe { target.assume_valid(&*heap) },
        )
        .expect("non-primitive receivers are objects");
    // Safety: caller-supplied word, rooted before the delete.
    let key = scope.handle(unsafe { key.assume_valid(&*heap) });
    Object::delete_own_property(heap, scope, receiver, key)
}

/// Whether a ToObject'd primitive owns `key` non-configurably: only
/// String wrappers own anything — "length" and their indices (ES
/// 10.4.3.3/4 StringGetOwnProperty). Deleting those yields false; every
/// other primitive property deletes as absent (true).
fn string_exotic_own(heap: &Heap, target: Tagged<'_, Value>, key: Tagged<'_, Value>) -> bool {
    let Some(s) = target.get_as::<DenseString>() else {
        return false;
    };
    if let Some(idx) = Smi::decode(key.raw()) {
        let i = idx.value();
        return i >= 0 && (i as u64) < s.len() as u64;
    }
    let Some(name) = key.get_as::<DenseString>() else {
        return false; // symbols own nothing on primitives
    };
    let data = name.as_ref().data(heap);
    data.matches_ascii(b"length") || canonical_index(data).is_some_and(|i| i < s.len())
}

/// Sloppy `delete x` on an unresolved name (ES 13.5.1.2 step 5 →
/// GlobalEnvironmentRecord.DeleteBinding): (name) -> bool. Declared
/// bindings resolve statically and compile to `false`; only global-object
/// properties reach here, and sloppy references never throw on failure.
fn delete_identifier_sloppy<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, state, .. } = nctx;
    // Safety: fresh argument word, consumed below.
    let name = {
        let heap = &*heap;
        args.get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?
            .raw()
    };
    // Safety: fresh root-slot word, consumed by the delete.
    let global = unsafe { heap.known().global_object.read_unchecked() };
    let ok = state.handle_scope(|scope| delete_property_core(heap, &scope, global, name))?;
    Ok(Convert::boolean(heap, ok))
}

/// `delete super.x` (ES 13.5.1.2 step 4.c): ReferenceError in both
/// language modes. The reference has already been evaluated (including
/// the uninitialized-`this` check and the key expression); the key is
/// never coerced — delete-super fails before any ToPropertyKey.
fn delete_super_property<'a>(
    _nctx: RuntimeContext<'a>,
    _args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
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
fn for_in_enumerate<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    // Safety: fresh argument word; nothing below allocates before it is
    // rooted.
    let subject = {
        let heap = &*heap;
        args.get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?
            .raw()
    };
    let nullish = {
        let heap = &*heap;
        let null = heap.known().null.as_tagged(heap).raw();
        let undefined = heap.known().undefined.as_tagged(heap).raw();
        subject == null || subject == undefined
    };
    if nullish {
        return Ok(heap.known().undefined.as_tagged(heap).erase());
    }
    let level = {
        let heap = &*heap;
        for_in_initial_level(heap, unsafe { subject.assume_valid(heap) })
    };
    let Some(level) = level else {
        return Ok(heap.known().undefined.as_tagged(heap).erase());
    };
    state.handle_scope(|scope| {
        // the level must survive the key collection and FixedArray
        // allocation below (both allocate)
        // Safety: fresh word from the non-allocating level walk, rooted
        // below before any allocation.
        let level = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(level) });
        let level_word = level.as_tagged(&*heap).raw();
        let keys = for_in_level_keys(vm, heap, &scope, level_word)?;

        let keys = heap.allocate_handle::<FixedArray>(
            scope.stage(
                &keys
                    .iter()
                    .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                    .collect::<Vec<_>>(),
            ),
            &scope,
        );
        let empty = heap.known().empty_fixed_array;
        let map = heap.known().for_in_enumerator_map;
        // Safety: fresh rooted-slot words staged into the fresh object.
        let level_word = level.as_tagged(&*heap).raw();
        let keys_word = keys.as_tagged(&*heap).raw();
        let empty_word = empty.as_tagged(&*heap).raw();
        let enumerator = heap.new_object(
            &scope,
            map,
            scope.stage(&[
                unsafe { Tagged::<Value>::from_value_unchecked(level_word) },
                unsafe { Tagged::<Value>::from_value_unchecked(keys_word) },
                Smi::new(0).into_tagged(),
                unsafe { Tagged::<Value>::from_value_unchecked(empty_word) },
            ]),
        );
        Ok(enumerator.erase())
    })
}

/// Level 0 of the chain for a subject: objects are their own level 0;
/// string primitives enumerate their indices (a fresh wrapper would be
/// unobservable otherwise). Other primitives have no own properties —
/// their level 0 is the constructor's prototype, so additions to
/// `Number.prototype` etc. are observable (ES 14.7.5.9: the walk starts
/// at ToObject(subject)). `None` when the prototype is unreachable.
fn for_in_initial_level(heap: &Heap, subject: Tagged<'_, Value>) -> Option<Value> {
    if subject.get_as::<DenseString>().is_some() {
        return Some(subject.raw());
    }
    if !Convert::is_primitive(heap, subject) {
        return Some(subject.raw());
    }
    let ctor_name = if Smi::decode(subject.raw()).is_some() {
        "Number"
    } else if subject.get_as::<Float>().is_some() {
        "Number"
    } else if subject == heap.known().true_object.as_tagged(heap)
        || subject == heap.known().false_object.as_tagged(heap)
    {
        "Boolean"
    } else if subject.get_as::<Symbol>().is_some() {
        "Symbol"
    } else {
        return None;
    };
    let global = heap.known().global_object.as_tagged(heap).raw();
    let strings = heap.known().strings;
    // Safety: fresh root-slot words read for the lookups.
    let ctor_handle = match ctor_name {
        "Number" => strings.number_ctor,
        "Boolean" => strings.boolean_ctor,
        _ => strings.symbol_ctor,
    };
    let ctor = match Lookup::load_outcome(
        heap,
        unsafe { global.assume_valid(heap) },
        ctor_handle.as_tagged(heap),
    )
    .ok()?
    {
        LoadOutcome::Value(v) if v.is_strong_ptr() => v.raw(),
        _ => return None,
    };
    match Lookup::load_outcome(
        heap,
        unsafe { ctor.assume_valid(heap) },
        // Safety: fresh root-slot word read for the lookup.
        heap.known().strings.prototype.as_tagged(heap),
    )
    .ok()?
    {
        LoadOutcome::Value(p) if p.is_strong_ptr() => Some(p.raw()),
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
    scope: &HandleScope<'_>,
    level: Value,
) -> Result<Vec<Value>, VmError> {
    // raw pass: Smi index keys (to be interned) and ready name keys
    let (mut indices, names) = 'keys: {
        // Safety: caller-supplied word, fresh at entry.
        let level = unsafe { level.assume_valid(heap) };
        let mut indices: Vec<i64> = Vec::new();
        let mut names: Vec<Value> = Vec::new();
        if let Some(s) = level.get_as::<DenseString>() {
            // string exotic: the only own string keys are the indices
            // ("length" is non-enumerable; the wrapper's own "length"
            // shadowing String.prototype additions is not modeled)
            indices.extend(0..s.len() as i64);
            break 'keys (indices, names);
        }
        let Some(obj) = level.as_heap_object() else {
            break 'keys (indices, names);
        };
        let obj = obj.as_ref();
        // array elements: non-hole indices ascending
        if obj.is_array(heap) {
            let len = obj
                .length()
                .min(obj.elements_array(heap).map(|e| e.len()).unwrap_or(0));
            for i in 0..len {
                if obj.element_value(heap, i).is_some() {
                    indices.push(i as i64);
                }
            }
        }

        for d in obj.map_ref(heap).descriptors() {
            let name = d.name(heap);

            if let Some(smi) = Smi::decode(name.raw()) {
                let v = smi.value();
                // array-index-range Smi names are index keys; anything
                // else (negative, ≥ 2^32−1) keeps insertion order
                if (0..u32::MAX as i64).contains(&v) {
                    indices.push(v);
                } else {
                    names.push(name.raw());
                }
                continue;
            }
            if name.erase().get_as::<Symbol>().is_some() {
                continue; // symbols are never yielded
            }
            // canonical index strings classify as index keys (a store
            // through them creates a Smi-named descriptor, but object
            // literals and defines can still reach here)
            let index = name
                .erase()
                .get_as::<DenseString>()
                .and_then(|s| canonical_index(s.as_ref().data(heap)))
                .filter(|i| *i < u32::MAX as usize);
            match index {
                Some(i) => indices.push(i as i64),
                None => names.push(name.raw()),
            }
        }
        (indices, names)
    };

    indices.sort_unstable();
    indices.dedup();
    // root every key before the next allocates: interning index keys
    // promotes earlier results, and raw copies would dangle
    // root the name keys too: the interning loop below allocates
    // Safety: fresh words from the walk above (no allocation since).
    let names: Vec<Handle<'_, Value>> = names
        .iter()
        .map(|v| scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(*v) }))
        .collect();
    let mut keys: Vec<Handle<'_, Value>> = Vec::with_capacity(indices.len() + names.len());
    for i in indices {
        let s = vm.interner().intern_str(heap, scope, &i.to_string());
        keys.push(scope.handle(s.as_tagged(heap).erase()));
    }
    keys.extend(names);
    // fresh words out of the rooted slots, consumed by the caller's
    // immediate staging
    Ok(keys.iter().map(|h| unsafe { h.read_unchecked() }).collect())
}

/// for-in iteration step (ES 14.7.5.9 EnumerateObjectProperties):
/// (enumerator) -> next key string | undefined. Per candidate key, the
/// own descriptor is checked lazily against the key's own level —
/// deleted-since-snapshot keys are skipped unvisited; keys shadowed by
/// an earlier level (yielded or non-enumerable) are skipped; enumerable
/// survivors are yielded at most once. When a level's snapshot runs
/// dry, the walk advances to the live prototype and snapshots it.
fn for_in_next<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    // Safety: fresh argument word; nothing below allocates before it is
    // rooted.
    let enumerator_word = {
        let heap = &*heap;
        args.get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?
            .raw()
    };
    if enumerator_word == {
        let heap = &*heap;
        heap.known().undefined.as_tagged(heap).raw()
    } {
        // nullish subject: the head produced no enumerator
        return Ok(heap.known().undefined.as_tagged(heap).erase());
    }
    state.handle_scope(|scope| {
        // the enumerator must survive the allocations below (key
        // interning, visited-array growth): read it through the handle
        // at every use, never a raw snapshot
        // Safety: fresh argument word, rooted below before any allocation.
        let enumerator =
            scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(enumerator_word) });
        loop {
            // one candidate per turn: the cursor advances before the
            // key is examined, so skipped keys are never revisited
            let candidate = 'candidate: {
                let Some(obj) =
                    unsafe { enumerator.read_unchecked().assume_valid(heap) }.as_heap_object()
                else {
                    return Err(VmError::Type);
                };
                let slots = obj.as_ref().slots.heap_ref(heap);
                let keys = slots
                    .at(heap, FOR_IN_KEYS)
                    .get_as::<FixedArray>()
                    .ok_or(VmError::Type)?;
                let index = Smi::decode(slots.at(heap, FOR_IN_INDEX).raw())
                    .ok_or(VmError::Type)?
                    .value() as usize;
                let Some(key) = (index < keys.len()).then(|| keys.at(heap, index)) else {
                    break 'candidate None;
                };
                slots.set(heap, FOR_IN_INDEX, Smi::new(index as i64 + 1).into_tagged());
                Some(key.raw())
            };
            let Some(key) = candidate else {
                // snapshot exhausted: advance to the live prototype
                let level = {
                    let Some(obj) =
                        unsafe { enumerator.read_unchecked().assume_valid(heap) }.as_heap_object()
                    else {
                        return Err(VmError::Type);
                    };
                    obj.as_ref()
                        .slots
                        .heap_ref(heap)
                        .at(heap, FOR_IN_LEVEL)
                        .raw()
                };
                let Some(proto) = for_in_next_level(vm, heap, level)? else {
                    return Ok(heap.known().undefined.as_tagged(heap).erase());
                };
                // for_in_level_keys allocates (interning): keep the new
                // level rooted across it
                // Safety: fresh word from the walk, rooted below.
                let proto = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(proto) });
                let proto_word = proto.as_tagged(heap).raw();
                let keys = for_in_level_keys(vm, heap, &scope, proto_word)?;
                let keys = heap.allocate_handle::<FixedArray>(
                    scope.stage(
                        &keys
                            .iter()
                            .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                            .collect::<Vec<_>>(),
                    ),
                    &scope,
                );
                let Some(obj) =
                    unsafe { enumerator.read_unchecked().assume_valid(heap) }.as_heap_object()
                else {
                    return Err(VmError::Type);
                };
                let slots = obj.as_ref().slots.heap_ref(heap);
                // Safety: fresh rooted-slot words stored below.
                slots.set(heap, FOR_IN_LEVEL, proto.as_tagged(heap).erase());
                slots.set(heap, FOR_IN_KEYS, keys.as_tagged(heap).erase());
                slots.set(heap, FOR_IN_INDEX, Smi::new(0).into_tagged());
                continue;
            };
            // the candidate must survive the visited-array growth below
            // Safety: fresh walk word, rooted below before any allocation.
            let key = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(key) });
            // lazy [[GetOwnProperty]] on the key's own level: a key
            // deleted since the snapshot is skipped without registering
            let level = {
                let Some(obj) =
                    unsafe { enumerator.read_unchecked().assume_valid(heap) }.as_heap_object()
                else {
                    return Err(VmError::Type);
                };
                obj.as_ref()
                    .slots
                    .heap_ref(heap)
                    .at(heap, FOR_IN_LEVEL)
                    .raw()
            };
            let own = for_in_own_state(
                heap,
                // Safety: fresh rooted-slot words, re-read now.
                unsafe { level.assume_valid(heap) }.raw(),
                unsafe { key.read_unchecked().assume_valid(heap) }.raw(),
            );
            let Some(enumerable) = own else {
                continue;
            };
            // already registered (yielded earlier, or shadowing
            // non-enumerable on a closer level): skip
            let seen = {
                let Some(obj) =
                    unsafe { enumerator.read_unchecked().assume_valid(heap) }.as_heap_object()
                else {
                    return Err(VmError::Type);
                };
                let visited = obj
                    .as_ref()
                    .slots
                    .heap_ref(heap)
                    .at(heap, FOR_IN_VISITED)
                    .get_as::<FixedArray>()
                    .ok_or(VmError::Type)?;
                let key_word = unsafe { key.read_unchecked() };
                visited.as_slice().iter().any(|s| s.inner() == key_word)
            };
            if seen {
                continue;
            }
            // register the key — yielded or shadowing, both at most once
            {
                let visited = {
                    let Some(obj) =
                        unsafe { enumerator.read_unchecked().assume_valid(heap) }.as_heap_object()
                    else {
                        return Err(VmError::Type);
                    };
                    let key_word = unsafe { key.read_unchecked() };
                    obj.as_ref()
                        .slots
                        .heap_ref(heap)
                        .at(heap, FOR_IN_VISITED)
                        .get_as::<FixedArray>()
                        .ok_or(VmError::Type)?
                        .as_slice()
                        .iter()
                        .map(|s| s.inner())
                        .chain([key_word])
                        .collect::<Vec<_>>()
                };
                let visited = heap.allocate_handle::<FixedArray>(
                    scope.stage(
                        &visited
                            .iter()
                            .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                            .collect::<Vec<_>>(),
                    ),
                    &scope,
                );
                let Some(obj) =
                    unsafe { enumerator.read_unchecked().assume_valid(heap) }.as_heap_object()
                else {
                    return Err(VmError::Type);
                };
                obj.as_ref().slots.heap_ref(heap).set(
                    heap,
                    FOR_IN_VISITED,
                    visited.as_tagged(heap).erase(),
                );
            }
            if !enumerable {
                continue;
            }
            return Ok(key.as_tagged(heap));
        }
    })
}

/// The next level of the prototype chain: an object's live [[Prototype]]
/// (read at advance time, so mutations between iterations are visible),
/// or `String.prototype` for a string primitive level. Multi-parent
/// (Self-style) and null prototypes end the walk.
fn for_in_next_level(_vm: &VM, heap: &mut Heap, level: Value) -> Result<Option<Value>, VmError> {
    // Safety: caller-supplied word, fresh at entry.
    if unsafe { level.assume_valid(heap) }
        .get_as::<DenseString>()
        .is_some()
    {
        // String.prototype via the global object (both plain data
        // lookups; no user code can run)
        // Safety: fresh root-slot words read for the lookups.
        let global = heap.known().global_object.as_tagged(heap).raw();
        let Some(string_ctor) = Lookup::load_outcome(
            heap,
            // Safety: root-slot word, still fresh.
            unsafe { global.assume_valid(heap) },
            heap.known().strings.string.as_tagged(heap),
        )
        .ok()
        .and_then(|o| match o {
            LoadOutcome::Value(v) => Some(v.raw()),
            LoadOutcome::Getter(_) => None,
        }) else {
            return Ok(None);
        };
        let proto = Lookup::load_outcome(
            heap,
            // Safety: walk word, still fresh (no allocation since).
            unsafe { string_ctor.assume_valid(heap) },
            // Safety: fresh root-slot word read for the lookup.
            heap.known().strings.prototype.as_tagged(heap),
        )
        .ok()
        .and_then(|o| match o {
            LoadOutcome::Value(v) => Some(v.raw()),
            LoadOutcome::Getter(_) => None,
        });
        return Ok(proto.filter(|p| p.is_strong_ptr()));
    }
    // Safety: caller-supplied word, fresh at entry.
    let Some(obj) = unsafe { level.assume_valid(heap) }.as_heap_object() else {
        return Ok(None);
    };
    let proto = obj.as_ref().map_ref(heap).prototype.inner();
    let hole = heap.known().the_hole.as_tagged(heap).raw();
    let null = heap.known().null.as_tagged(heap).raw();
    if proto == hole || proto == null {
        return Ok(None);
    }
    // a FixedArray prototype is the Self-style multi-parent form;
    // the chain walk does not model it (ends the enumeration)
    Ok(unsafe { proto.assume_valid(heap) }
        .get_as::<FixedArray>()
        .map_or(Some(proto), |_| None))
}

/// The lazy own-property state of `key` on its own level: `None` when
/// the property is gone (deleted since the snapshot), else its
/// [[Enumerable]]. Own-only — the shadow check against other levels is
/// the visited set's job.
fn for_in_own_state(heap: &Heap, level: Value, key: Value) -> Option<bool> {
    // Safety: caller-supplied words, fresh at entry.
    match Lookup::classify_key(heap, unsafe { key.assume_valid(heap) }).ok()? {
        Key::Element(i) => {
            // Safety: caller-supplied word, fresh at entry.
            if let Some(s) = unsafe { level.assume_valid(heap) }.get_as::<DenseString>() {
                // string indices are enumerable own properties
                return Some((i as u64) < s.len() as u64);
            }
            // Safety: caller-supplied word, fresh at entry.
            let obj = unsafe { level.assume_valid(heap) }.as_heap_object()?;
            let obj = obj.as_ref();
            if obj.is_array(heap) {
                return obj.element_value(heap, i).is_some().then_some(true);
            }
            // plain objects keep index keys as Smi-named descriptors
            let name = Tagged::<SlotName>::from(Smi::new(i as i64));
            obj.map_ref(heap)
                .descriptors()
                .iter()
                .find(|d| d.name(heap).ptr_eq(name))
                .map(|d| d.flags().is_enumerable())
        }
        Key::Name(name) => {
            // Safety: caller-supplied word, fresh at entry.
            let obj = unsafe { level.assume_valid(heap) }.as_heap_object()?;
            let obj = obj.as_ref();

            // arrays hold "length" outside the descriptors (never a
            // snapshot key) — any other name lives in them
            obj.map_ref(heap)
                .descriptors()
                .iter()
                .find(|d| d.name(heap).ptr_eq(name))
                .map(|d| d.flags().is_enumerable())
        }
    }
}

/// GetIterator (ES 8.5.4): (obj) -> iterator.
fn get_iterator<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        // Safety: fresh argument word, rooted below before any allocation.
        let obj = scope.handle({
            let heap = &*heap;
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
        });
        let symbol = scope.handle(heap.known().iterator_symbol.as_tagged(heap).erase());
        let method = Lookup::get_property_on(vm, heap, state, obj, obj, symbol)?;
        let method = match method {
            Coercion::Threw => {
                return Ok(heap.known().exception.as_tagged(heap).erase());
            }
            Coercion::Value(v) => scope.handle(v),
        };
        let (undefined_or_null, callable) = {
            let heap = &*heap;
            let method = method.as_tagged(heap);
            (
                method.raw() == heap.known().undefined.as_tagged(heap).raw()
                    || method.raw() == heap.known().null.as_tagged(heap).raw(),
                Object::is_callable(heap, method),
            )
        };
        if undefined_or_null || !callable {
            return Err(VmError::Type); // "obj is not iterable"
        }
        let obj_word = obj.as_tagged(&*heap).raw();
        RuntimeContext::call(
            vm,
            heap,
            state,
            method,
            scope.stage(&[unsafe { Tagged::<Value>::from_value_unchecked(obj_word) }]),
            None,
        )
    })
}

/// IteratorNext (ES 8.5.6): (iterator) -> result object.
fn iterator_next<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        // Safety: fresh argument word, rooted below before any allocation.
        let iter = scope.handle({
            let heap = &*heap;
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
        });
        let next_name = scope.handle(heap.known().strings.next.as_tagged(heap).erase());
        let next = Lookup::get_property_on(vm, heap, state, iter, iter, next_name)?;
        let next = match next {
            Coercion::Threw => {
                return Ok(heap.known().exception.as_tagged(heap).erase());
            }
            Coercion::Value(v) => scope.handle(v),
        };
        let iter_word = iter.as_tagged(&*heap).raw();
        let result = scope.handle(RuntimeContext::call(
            vm,
            &mut *heap,
            state,
            next,
            scope.stage(&[unsafe { Tagged::<Value>::from_value_unchecked(iter_word) }]),
            None,
        )?);
        let cond_25 = {
            let heap = &*heap;
            result.as_tagged(heap).raw() == heap.known().exception.as_tagged(heap).raw()
        };
        if cond_25 {
            return Ok(result.as_tagged(heap));
        }
        {
            let heap = &*heap;
            let cond_26 = Convert::is_primitive(heap, result.as_tagged(heap));
            if cond_26 {
                return Err(VmError::Type); // IteratorNext result must be an Object
            }
        }
        Ok(result.as_tagged(heap))
    })
}

/// IteratorComplete (ES 8.5.7): (result) -> bool.
fn iterator_done<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let result = args.get(0).ok_or(VmError::Arity)?;
        let done_name = scope.handle(heap.known().strings.done.as_tagged(heap).erase());
        let v = match Lookup::get_property_on(vm, heap, state, result, result, done_name)? {
            Coercion::Threw => return Ok(heap.known().exception.as_tagged(heap).erase()),
            Coercion::Value(v) => scope.handle(v),
        };
        let truthy = Convert::is_truthy(heap, v.as_tagged(heap));
        Ok(Convert::boolean(heap, truthy))
    })
}

/// IteratorValue (ES 8.5.8): (result) -> value.
fn iterator_value<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let result = args.get(0).ok_or(VmError::Arity)?;
        let value_name = scope.handle(heap.known().strings.value.as_tagged(heap).erase());
        let v = match Lookup::get_property_on(vm, heap, state, result, result, value_name)? {
            Coercion::Threw => return Ok(heap.known().exception.as_tagged(heap).erase()),
            Coercion::Value(v) => scope.handle(v),
        };
        Ok(v.as_tagged(heap))
    })
}

/// The `in` operator (ES 14.11.2): (key, obj) -> bool. Proxy receivers
/// run their `has` trap (ES 20.2.5.9).
fn has_property<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    // Safety: fresh argument words; nothing allocates before they are
    // rooted / consumed below.
    let (raw_key, raw_obj) = {
        let heap = &*heap;
        (
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
        )
    };
    // the key coercion allocates (wrapper keys run toString/valueOf):
    // root the receiver and re-read it after the coercion — a raw
    // snapshot taken before would go stale
    state.handle_scope(|scope| {
        // Safety: fresh argument word, rooted below before any allocation.
        let obj = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_obj) });
        let Some(key) = Object::to_property_key(
            vm,
            heap,
            state,
            // Safety: fresh argument word, fresh at entry.
            scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_key) }),
        )?
        else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        let obj = obj.as_tagged(heap).raw();
        let cond_27 = Proxy::is_proxy(heap, unsafe { obj.assume_valid(heap) });
        if cond_27 {
            // Safety: fresh rooted-slot word plus a fresh coercion
            // result, both consumed by the trap call.
            let key_word = unsafe { key.read_unchecked() };
            let has = match Proxy::has(
                vm,
                heap,
                state,
                unsafe { Tagged::<Value>::from_value_unchecked(obj) },
                unsafe { Tagged::<Value>::from_value_unchecked(key_word) },
            )? {
                Coercion::Threw => return Ok(heap.known().exception.as_tagged(heap).erase()),
                Coercion::Value(v) => scope.handle(v),
            };
            return Ok(has.as_tagged(heap));
        }
        // lookup::has_property covers array `length` slots along the chain
        let has = lookup_has_property(
            heap,
            // Safety: fresh rooted-slot word re-read under the anchor.
            unsafe { obj.assume_valid(heap) },
            key.as_tagged(heap),
        );
        Ok(Convert::boolean(heap, has))
    })
}

/// CopyDataProperties (ES 8.5.1) with an exclusion list (object rest):
/// (excluded..., target, source); `excluded` has count−2 entries.
fn copy_data_properties<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let n = args.len();
    if n < 2 {
        return Err(VmError::Arity);
    }
    // Safety: fresh argument words, consumed below.
    let (target, source, excluded) = {
        let heap = &*heap;
        (
            args.get(n - 2).map(|h| h.as_tagged(heap)).map(|v| v.raw()),
            args.get(n - 1).map(|h| h.as_tagged(heap)).map(|v| v.raw()),
            (0..n - 2)
                .map(|i| args.get(i).map(|h| h.as_tagged(heap)).map(|v| v.raw()))
                .collect::<Option<Vec<_>>>(),
        )
    };
    let (Some(target), Some(source), Some(excluded)) = (target, source, excluded) else {
        return Err(VmError::Arity);
    };
    let nullish = {
        let heap = &*heap;
        let null = heap.known().null.as_tagged(heap).raw();
        let undefined = heap.known().undefined.as_tagged(heap).raw();
        source == null || source == undefined
    };
    if nullish {
        return Ok(unsafe { Tagged::<Value>::from_value_unchecked(target) });
    }
    // only heap objects contribute (string sources need boxing)
    {
        let cond_28 = {
            let heap = &*heap;
            Convert::is_primitive(heap, unsafe { source.assume_valid(heap) })
        };
        if cond_28 {
            return Ok(unsafe { Tagged::<Value>::from_value_unchecked(target) });
        }
    }
    let target_out = state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        // target and source survive getter calls and property adds below:
        // root them once, not per iteration from raw copies
        // Safety: fresh argument words, rooted below before any allocation.
        let target_handle = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(target) });
        let source_handle = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(source) });
        // canonicalize the excluded keys (interning strings) so a plain
        // bits comparison suffices against the source's descriptor names
        let excluded: Vec<Value> = {
            let mut out = Vec::with_capacity(excluded.len());
            for k in excluded {
                match Object::to_property_key(
                    vm,
                    heap,
                    state,
                    // Safety: fresh argument word, fresh at entry.
                    scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(k) }),
                )? {
                    Some(k) => out.push(k.raw()),
                    None => return Ok(heap.known().exception.as_tagged(heap).erase()),
                }
            }
            out
        };
        // enumerate own enumerable keys: element indices ascending, then
        // named descriptors in insertion order; collected AFTER the
        // exclusion canonicalization so no allocation can stale them
        let mut keys: Vec<Value> = Vec::new();
        'collect: {
            let heap = &*heap;
            let Some(obj) =
                unsafe { source_handle.read_unchecked().assume_valid(heap) }.as_heap_object()
            else {
                break 'collect;
            };
            if obj.as_ref().is_array(heap) {
                let len = obj.as_ref().length().min(
                    obj.as_ref()
                        .elements_array(heap)
                        .map(|e| e.len())
                        .unwrap_or(0),
                );
                for i in 0..len {
                    if obj.as_ref().element_value(heap, i).is_some() {
                        keys.push(Smi::new(i as i64).encode());
                    }
                }
            }
            for d in obj.as_ref().header.map.heap_ref(heap).descriptors() {
                if d.flags().is_enumerable() {
                    keys.push(d.name(heap).raw());
                }
            }
        }
        // root every key: the getter calls below allocate and raw copies
        // would dangle
        // Safety: fresh words from the walk above (no allocation since).
        let keys: Vec<Handle<'_, Value>> = keys
            .iter()
            .map(|k| scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(*k) }))
            .collect();
        for key in keys {
            let key_word = key.as_tagged(heap).raw();
            if excluded.contains(&key_word) {
                continue;
            }
            // full [[Get]] (getters may run)
            let value = match Lookup::get_property_on(
                vm,
                heap,
                state,
                source_handle,
                source_handle,
                key,
            )? {
                Coercion::Threw => return Ok(heap.known().exception.as_tagged(heap).erase()),
                Coercion::Value(v) => scope.handle(v),
            };
            // CreateDataProperty: skipped when already present
            let exists = !matches!(
                target_handle.lookup(heap, key.as_tagged(heap).as_name()),
                Lookup::NotFound
            );
            if exists {
                continue;
            }
            let target_obj = scope
                .cast::<Object>(target_handle.as_tagged(heap))
                .expect("copy target is an object");
            let key_name: Handle<'_, SlotName> = scope.handle(key.as_tagged(heap).as_name());
            Object::add_own_property(
                heap,
                &scope,
                target_obj,
                key_name,
                PropertyDescriptor::data(value),
            )?;
        }
        Ok(target_handle.as_tagged(heap))
    })?;
    // re-read through the handle: the copy loop allocated (getters,
    // property adds) and may have moved the target
    Ok(target_out)
}

/// A fresh private name: (description) -> Symbol.
fn create_private_name<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, state, .. } = nctx;
    let text = {
        let heap = &*heap;
        args.get(1)
            .map(|h| h.as_tagged(heap))
            .and_then(|d| d.get_as::<DenseString>())
            .map(|s| s.to_rust_string(heap))
    };
    state.handle_scope(|scope| {
        let desc = text.unwrap_or_default();
        let sym = Symbol::new(heap, &scope, desc.as_bytes());
        Ok(sym.as_tagged(heap).erase())
    })
}

/// PrivateGet (ES 7.3.30): (obj, key) -> value, TypeError when absent.
fn private_get<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let heap = &*heap;
    let obj = args
        .get(0)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?;
    let key = args
        .get(1)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?;
    match private_find(heap, obj, key) {
        Some(s) => Ok(s.get(heap)),
        None => Err(VmError::Type),
    }
}

/// PrivateSet (ES 7.3.31): (obj, key, value), TypeError when absent.
fn private_set<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let value = {
        let heap = &*heap;
        let obj = args
            .get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?;
        let key = args
            .get(1)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?;
        let value = args
            .get(2)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?;
        match private_find(heap, obj, key) {
            Some(slot) => {
                slot.set(heap, obj.raw(), value);
                Ok(value)
            }
            None => Err(VmError::Type),
        }
    }?;
    Ok(value)
}

/// `#x in obj`: (key, obj) -> bool (own private presence only).
fn private_in<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let has = {
        let heap = &*heap;
        let key = args
            .get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?;
        let obj = args
            .get(1)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?;
        private_find(heap, obj, key).is_some()
    };
    Ok(Convert::boolean(heap, has))
}

/// Attach the instance-field array to the class constructor:
/// (ctor, fields).
fn set_class_fields<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let heap = &*heap;
    let ctor = args
        .get(0)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?;
    let fields = args
        .get(1)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?;
    let mut ok = false;
    if let Some(obj) = ctor.as_heap_object() {
        let slots = obj.as_ref().slots.heap_ref(heap);
        if obj
            .as_ref()
            .header
            .map
            .heap_ref(heap)
            .kind()
            .is_class_constructor()
            && slots.len() >= 3
        {
            slots.as_ref().element_slot(2).set(heap, ctor.raw(), fields);
            ok = true;
        }
    }
    if !ok {
        return Err(VmError::Type);
    }
    Ok(ctor)
}

/// InitializeInstanceElements (ES 7.3.33): (ctor, instance) -> instance.
/// Runs each field initializer with the instance as receiver and defines
/// the result onto it ({w+, e+, c+}).
fn init_instance_fields<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    // Safety: fresh argument words; nothing below allocates before they
    // are rooted / consumed.
    let (ctor, instance) = {
        let heap = &*heap;
        (
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
        )
    };
    let fields = 'fields: {
        let heap = &*heap;
        let Some(obj) = unsafe { ctor.assume_valid(heap) }.as_heap_object() else {
            break 'fields None;
        };
        let slots = obj.as_ref().slots.heap_ref(heap);
        (slots.len() >= 3).then(|| slots.at(heap, 2).raw())
    };
    let Some(fields) = fields else {
        return Err(VmError::Type);
    };
    let cond_29 = {
        let heap = &*heap;
        fields == heap.known().undefined.as_tagged(heap).raw()
    };
    if cond_29 {
        return Ok(unsafe { Tagged::<Value>::from_value_unchecked(instance) });
    }
    let count = {
        let heap = &*heap;
        unsafe { fields.assume_valid(heap) }
            .as_heap_object()
            .map(|o| o.as_ref().length())
            .unwrap_or(0)
    };
    let instance_out = state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        // Safety: fresh argument words, rooted below before any allocation.
        let instance = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(instance) });
        // Safety: fresh walk word, rooted below before any allocation.
        let fields = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(fields) });
        // Safety: fresh root-slot word; singletons never move.
        let exception = heap.known().exception.as_tagged(heap).raw();
        let mut i = 0;
        while i + 1 < count {
            let raw_key = {
                let heap = &*heap;
                unsafe { fields.read_unchecked().assume_valid(heap) }
                    .as_heap_object()
                    .and_then(|o| o.as_ref().element_value(heap, i))
                    .map(|v| v.raw())
                    .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).raw())
            };
            // computed keys need ToPropertyKey canonicalization
            let key = {
                match Object::to_property_key(
                    vm,
                    heap,
                    state,
                    // Safety: fresh walk word, no GC since the read.
                    scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_key) }),
                )? {
                    Some(k) => scope.handle(k),
                    None => return Ok(heap.known().exception.as_tagged(heap).erase()),
                }
            };
            // the initializer call allocates (user code): the key stays
            // rooted in the scope across it
            // re-read the initializer after the coercion (it allocated)
            let init = {
                let heap = &*heap;
                unsafe { fields.read_unchecked().assume_valid(heap) }
                    .as_heap_object()
                    .and_then(|o| o.as_ref().element_value(heap, i + 1))
                    .map(|v| v.raw())
                    .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).raw())
            };
            // Safety: fresh walk word, consumed by the call.
            let init = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(init) });
            // Safety: fresh rooted-slot words staged for the call.
            let instance_word = instance.as_tagged(&*heap).raw();
            let value = scope.handle(RuntimeContext::call(
                vm,
                &mut *heap,
                state,
                init,
                scope.stage(&[unsafe { Tagged::<Value>::from_value_unchecked(instance_word) }]),
                None,
            )?);
            if value.as_tagged(heap).raw() == exception {
                return Ok(heap.known().exception.as_tagged(heap).erase());
            }
            let instance_obj = scope
                .cast::<Object>(instance.as_tagged(&*heap))
                .expect("class instance is an object");
            let defined = Object::define_own_property(
                heap,
                &scope,
                instance_obj,
                key,
                PropertyDescriptor::Data {
                    value,
                    writable: true,
                    enumerable: true,
                    configurable: true,
                },
            )?;
            if !defined {
                return Err(VmError::Type);
            }
            i += 2;
        }
        Ok(instance.as_tagged(heap))
    })?;
    Ok(instance_out)
}

// ---- frame access -----------------------------------------------------------

/// The current (calling) frame's context: `CallRuntime` runs in place, so
/// the interpreter's cache still holds the frame executing the call.
fn frame_context_value<'a>(
    state: &ContextState,
    heap: &'a Heap,
) -> Result<Tagged<'a, Value>, VmError> {
    if !state.cache.is_active() {
        return Err(VmError::Type);
    }
    Ok(state.stack.context(heap, &state.cache.frame_meta()))
}

/// Find the slot named `name` in `context`'s chain (direct eval). Returns
/// the slot cell, or Reference when no context in the chain has the name.
fn dynamic_slot<'a>(
    heap: &'a Heap,
    context: &mut HeapRef<'a, Context>,
    name: Value,
) -> Result<&'a GcSlot, VmError> {
    // both sides are interned (constant pool / ScopeInfo names), so
    // pointer identity decides — no content comparison in lookup
    // Safety: caller-supplied word, fresh at entry.
    unsafe { name.assume_valid(heap) }
        .get_as::<DenseString>()
        .ok_or(VmError::Type)?;
    loop {
        let ctx = context.as_ref();
        let names = ctx.scope_info.heap_ref(heap).as_ref().names.heap_ref(heap);
        for i in 0..names.len() {
            if names.at(heap, i) == name {
                return Ok(ctx.slots.heap_ref(heap).as_ref().element_slot(i));
            }
        }
        match ctx.outer.heap_ref(heap) {
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
    let context = frame_context_value(state, heap)?;
    let mut context = context.get_as::<Context>().ok_or(VmError::Type)?;
    match dynamic_slot(heap, &mut context, name) {
        Ok(slot) => Ok(Some(slot.inner())),
        Err(VmError::Reference) => Ok(None),
        Err(e) => Err(e),
    }
}

/// The current frame's super constructor and new.target (direct
/// super() calls, ES 15.4.3): the running closure's [[Prototype]] must
/// be a constructor.
fn frame_super_parts(heap: &mut Heap, state: &ContextState) -> Result<(Value, Value), VmError> {
    if !state.cache.is_active() {
        return Err(VmError::Type);
    }
    let meta = state.cache.frame_meta();
    let Some(callee) = super_constructor(heap, &state.stack, &meta) else {
        return Err(VmError::Type);
    };
    Ok((callee.raw(), state.stack.new_target_slot(&meta).inner()))
}

// ---- store outcomes ---------------------------------------------------------

/// Apply a store outcome: transitions add the property on the receiver,
/// setters are invoked with (receiver, value). Returns `true` when a
/// setter threw (the pending exception is set; the caller propagates the
/// exception sentinel).
fn apply_store_outcome(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    receiver: Value,
    outcome: StoreOutcome<'_>,
    value: Value,
) -> Result<bool, VmError> {
    match outcome {
        StoreOutcome::Transition {
            receiver: recv,
            name,
        } => {
            state.handle_scope(|scope| {
                // Safety: caller-supplied value word, rooted before the define.
                let value = scope.handle(unsafe { value.assume_valid(&*heap) });
                Object::add_own_property(heap, &scope, recv, name, PropertyDescriptor::data(value))
                    // TODO(strict-mode): a false result must throw in strict code;
                    // the current store path preserves its existing sloppy result.
                    .map(|_| false)
            })
        }
        StoreOutcome::CallSetter { setter } => {
            // Safety: fresh root-slot word read for the comparison below.
            let exception = heap.known().exception.as_tagged(heap).raw();
            let result = state.handle_scope(|scope| {
                RuntimeContext::call(
                    vm,
                    heap,
                    state,
                    setter,
                    // Safety: caller-supplied words, staged for the call.
                    scope.stage(&[
                        unsafe { Tagged::<Value>::from_value_unchecked(receiver) },
                        unsafe { Tagged::<Value>::from_value_unchecked(value) },
                    ]),
                    None,
                )
            })?;
            Ok(result == exception)
        }
        StoreOutcome::Done => Ok(false),
    }
}

/// A full [[Get]] that treats non-callable getters (an absent half of an
/// accessor pair) as undefined instead of throwing.
fn get_property_lenient<'a>(
    nctx: RuntimeContext<'a>,
    receiver: Value,
    name: Value,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    // (plain value, getter) — both raw fresh words
    // Safety: caller-supplied words, fresh at entry.
    let outcome = {
        let heap = &*heap;
        match Lookup::load_outcome(
            heap,
            unsafe { receiver.assume_valid(heap) },
            // Safety: caller-supplied name word, fresh at entry.
            unsafe { name.assume_valid(heap) }.as_name(),
        )? {
            LoadOutcome::Value(v) => Ok((Some(v.raw()), None)),
            LoadOutcome::Getter(g) => Ok((None, Some(g.raw()))),
        }
    }?;
    if let Some(v) = outcome.0 {
        return Ok(unsafe { Tagged::<Value>::from_value_unchecked(v) });
    }
    // Safety: fresh walk word from the lookup above.
    let getter = outcome.1.expect("one of the two arms is set");
    let undefined = {
        let heap = &*heap;
        heap.known().undefined.as_tagged(heap).raw()
    };
    if getter == undefined
        || !{
            let heap = &*heap;
            Object::is_callable(heap, unsafe {
                Tagged::<Value>::from_value_unchecked(getter)
            })
        }
    {
        return Ok(unsafe { Tagged::<Value>::from_value_unchecked(undefined) });
    }
    state.handle_scope(|scope| {
        RuntimeContext::call(
            vm,
            heap,
            state,
            scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(getter) }),
            // Safety: caller-supplied word, staged for the call.
            scope.stage(&[unsafe { Tagged::<Value>::from_value_unchecked(receiver) }]),
            None,
        )
    })
}

// ---- class definition helpers -----------------------------------------------

/// SetFunctionName (ES 8.4.4): (fn, key, prefix) -> fn. Redefines `name`
/// on the closure ({w−, e−, c+}); the prefix discriminant (a Smi) is
/// 0 none, 1 "get ", 2 "set ". Class members named `name` define over the
/// constructor after ClassDefinitionEvaluation set its name — an already
/// explicitly defined `name` wins (ES 15.7.14: SetFunctionName happens
/// before element installation).
fn set_function_name<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    // Safety: fresh argument words; nothing allocates before they are
    // rooted / consumed below.
    let (fn_value, raw_key, prefix) = {
        let heap = &*heap;
        let prefix = Smi::decode(
            args.get(2)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
        )
        .map(|s| s.value())
        .unwrap_or(0);
        (
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            prefix,
        )
    };
    // name construction allocates (interning): the closure must stay
    // rooted across it
    state.handle_scope(|scope| {
        // Safety: fresh argument word, rooted below before any allocation.
        let fn_value = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(fn_value) });
        let Some(key) = Object::to_property_key(
            vm,
            heap,
            state,
            // Safety: fresh argument word, fresh at entry.
            scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_key) }),
        )?
        else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key).erase();
        let units = state.handle_scope(|scope| -> Result<Vec<u16>, VmError> {
            let text = Convert::to_string(heap, &scope, key)?;
            let text = text.raw();
            Ok({
                // Safety: fresh word, no allocation since the read.
                let s = unsafe { text.assume_valid(heap) }
                    .get_as::<DenseString>()
                    .expect("ToString yields a string")
                    .as_ref();
                let mut full: Vec<u16> = match prefix {
                    1 => b"get ".iter().map(|&b| b as u16).collect(),
                    2 => b"set ".iter().map(|&b| b as u16).collect(),
                    _ => Vec::new(),
                };
                s.data(heap).write_units(&mut full);
                full
            })
        })?;
        let name = state.handle_scope(|scope| {
            // Safety: fresh rooted-slot word read for the immediate use.
            unsafe {
                vm.interner()
                    .intern(heap, &scope, StringData::Utf16(&units))
                    .read_unchecked()
            }
        });
        let defined = {
            let Some(fn_obj) = scope.cast::<Object>(fn_value.as_tagged(heap)) else {
                return Err(VmError::Type);
            };
            let name_key = heap.known().strings.name;
            // the closure's own placeholder is never writable nor an accessor,
            // so only explicit member defines match here
            let explicit = match fn_obj
                .heap_ref(heap)
                .as_ref()
                .lookup(heap, name_key.as_tagged(heap))
            {
                Lookup::Data { flags, .. } => flags.is_writable(),
                Lookup::Accessor { .. } => true,
                Lookup::NotFound => false,
            };

            if explicit {
                true
            } else {
                // Safety: interned word, rooted before the define.
                let name = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(name) });
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
            }
        };
        if !defined {
            return Err(VmError::Type);
        }
        Ok(fn_value.as_tagged(heap))
    })
}

/// Accessor member installation (ES 14.3.10): (target, key, closure,
/// flags). Defines one accessor half, merging with an existing pair under
/// the same key; flags bit 0 marks the getter half, PropertyFlags bits
/// carry enumerability.
fn install_accessor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    // Safety: fresh argument words; nothing allocates before they are
    // rooted / consumed below.
    let (raw_target, raw_key, raw_closure, flags) = {
        let heap = &*heap;
        let flags = Smi::decode(
            args.get(3)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
        )
        .map(|s| s.value() as u32)
        .unwrap_or(0);
        (
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(2)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            flags,
        )
    };
    // the key coercion allocates: root the target and closure across it
    state.handle_scope(|scope| {
        // Safety: fresh argument words, rooted below before any allocation.
        let target = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_target) });
        let closure = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_closure) });
        let Some(key) = Object::to_property_key(
            vm,
            heap,
            state,
            // Safety: fresh argument word, fresh at entry.
            scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_key) }),
        )?
        else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        let is_getter = flags & 1 != 0;
        let enumerable = flags & bytecode::PropertyFlags::DontEnum.bits() == 0;
        let (name, desc) = {
            // Safety: fresh rooted-slot words re-read under the anchor.
            if target.as_tagged(heap).as_heap_object().is_none() {
                return Err(VmError::Type);
            }
            let name = match Lookup::classify_key(heap, key.as_tagged(heap).erase())? {
                Key::Element(i) => Tagged::from(Smi::new(i as i64)),
                Key::Name(name) => name,
            };
            // existing own accessor half, if any (own descriptors only)
            let mut get = heap.known().undefined.as_tagged(heap).erase();
            let mut set = get;
            if let Some(obj) = target.as_tagged(heap).as_heap_object() {
                for d in obj.as_ref().header.map.heap_ref(heap).descriptors() {
                    if d.name(heap).ptr_eq(name) && d.flags().is_accessor() {
                        let pair = d
                            .value
                            .get(heap)
                            .get_as::<AccessorPair>()
                            .expect("accessor descriptor holds a pair");
                        get = pair.get.get(heap);
                        set = pair.set.get(heap);
                        break;
                    }
                }
            }
            // Safety: fresh rooted-slot word for the new half.
            let closure_word = closure.as_tagged(heap).erase();
            if is_getter {
                get = closure_word;
            } else {
                set = closure_word;
            }
            Ok((
                // the name must outlive this non-allocating region: root it
                scope.handle(name),
                PropertyDescriptor::Accessor {
                    get: scope.handle(get),
                    set: scope.handle(set),
                    enumerable,
                    configurable: true,
                },
            ))
        }?;
        let target_obj = scope
            .cast::<Object>(target.as_tagged(heap))
            .expect("checked object above");
        let defined = Object::define_own_property(heap, &scope, target_obj, name, desc)?;
        if !defined {
            return Err(VmError::Type);
        }
        Ok(closure.as_tagged(heap))
    })
}

/// [[DefineOwnProperty]] with exact attributes (class member
/// installation): (obj, key, value, flags) -> obj. Define sites are
/// strict-mode code: a rejected define throws a TypeError. flags are
/// PropertyFlags bits (the Accessor bit: the value is an AccessorPair).
fn define_own_property<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    // Safety: fresh argument words; nothing allocates before they are
    // rooted / consumed below.
    let (raw_receiver, raw_key, raw_value, flags) = {
        let heap = &*heap;
        let flags = Smi::decode(
            args.get(3)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
        )
        .map(|s| s.value() as u32)
        .unwrap_or(0);
        (
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(2)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            flags,
        )
    };
    // the key coercion allocates: root the receiver and value across it
    state.handle_scope(|scope| {
        // Safety: fresh argument words, rooted below before any allocation.
        let receiver = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_receiver) });
        let value = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_value) });
        let Some(key) = Object::to_property_key(
            vm,
            heap,
            state,
            // Safety: fresh argument word, fresh at entry.
            scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_key) }),
        )?
        else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        // proxies run their `defineProperty` trap (ES 20.2.5.6); define
        // sites are strict-mode: a rejected define throws
        {
            let cond_30 = Proxy::is_proxy(heap, receiver.as_tagged(heap));
            if cond_30 {
                let partial = {
                    let enumerable = flags & bytecode::PropertyFlags::DontEnum.bits() == 0;
                    let configurable = flags & bytecode::PropertyFlags::DontDelete.bits() == 0;
                    if flags & bytecode::PropertyFlags::Accessor.bits() != 0 {
                        let pair = value
                            .as_tagged(heap)
                            .get_as::<AccessorPair>()
                            .ok_or(VmError::Type)?;
                        let pair = pair.as_ref();
                        Ok(PartialDescriptor {
                            value: None,
                            get: Some(scope.handle(pair.get.get(heap))),
                            set: Some(scope.handle(pair.set.get(heap))),
                            writable: None,
                            enumerable: Some(enumerable),
                            configurable: Some(configurable),
                        })
                    } else {
                        Ok(PartialDescriptor {
                            value: Some(scope.handle(value.as_tagged(heap))),
                            get: None,
                            set: None,
                            writable: Some(flags & bytecode::PropertyFlags::ReadOnly.bits() == 0),
                            enumerable: Some(enumerable),
                            configurable: Some(configurable),
                        })
                    }
                }?;
                // Safety: rooted handle words, consumed by the trap call.
                let recv_word = unsafe { receiver.read_unchecked() };
                let key_word = unsafe { key.read_unchecked() };
                return match Proxy::define_internal(
                    vm,
                    heap,
                    state,
                    &scope,
                    unsafe { Tagged::<Value>::from_value_unchecked(recv_word) },
                    unsafe { Tagged::<Value>::from_value_unchecked(key_word) },
                    partial,
                )? {
                    Flow::Threw => Ok(heap.known().exception.as_tagged(heap).erase()),
                    Flow::Value(false) => Err(VmError::Type),
                    Flow::Value(true) => Ok(receiver.as_tagged(heap)),
                };
            }
        }
        let (name, desc) = {
            if receiver.as_tagged(heap).as_heap_object().is_none() {
                return Err(VmError::Type);
            }
            // Safety: fresh rooted name word re-read under the anchor.
            let name = match Lookup::classify_key(heap, key.as_tagged(heap).erase())? {
                Key::Element(i) => Tagged::from(Smi::new(i as i64)),
                Key::Name(name) => name,
            };
            let enumerable = flags & bytecode::PropertyFlags::DontEnum.bits() == 0;
            let configurable = flags & bytecode::PropertyFlags::DontDelete.bits() == 0;
            let desc = if flags & bytecode::PropertyFlags::Accessor.bits() != 0 {
                let pair = value
                    .as_tagged(heap)
                    .get_as::<AccessorPair>()
                    .ok_or(VmError::Type)?;
                let pair = pair.as_ref();
                PropertyDescriptor::Accessor {
                    get: scope.handle(pair.get.get(heap)),
                    set: scope.handle(pair.set.get(heap)),
                    enumerable,
                    configurable,
                }
            } else {
                PropertyDescriptor::Data {
                    value: scope.handle(value.as_tagged(heap)),
                    writable: flags & bytecode::PropertyFlags::ReadOnly.bits() == 0,
                    enumerable,
                    configurable,
                }
            };
            (scope.handle(name), desc)
        };
        let receiver_obj = scope
            .cast::<Object>(receiver.as_tagged(heap))
            .expect("checked object above");
        let defined = Object::define_own_property(heap, &scope, receiver_obj, name, desc)?;
        if !defined {
            return Err(VmError::Type);
        }
        Ok(receiver.as_tagged(heap))
    })
}

/// [[SetPrototypeOf]] (class prototype wiring): (obj, proto) -> obj.
fn set_prototype<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, state, .. } = nctx;
    state.handle_scope(|scope| {
        // Safety: fresh argument words, rooted below before any allocation.
        let obj = scope.handle({
            let heap = &*heap;
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
        });
        let proto = scope.handle({
            let heap = &*heap;
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
        });
        let obj_ref = scope
            .cast::<Object>(obj.as_tagged(&*heap))
            .expect("obj is an object");
        Object::set_prototype(heap, &scope, obj_ref, proto)?;
        Ok(obj.as_tagged(heap))
    })
}

/// Class extends validation (ES 15.7.14 step 15.e): (value) -> value,
/// TypeError unless the superclass is null or a constructor.
fn throw_if_not_constructor_or_null<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let heap = &*heap;
    let v = args
        .get(0)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?;
    let ok = if v == heap.known().null.as_tagged(heap) {
        true
    } else {
        v.as_heap_object().is_some_and(|obj| {
            obj.as_ref()
                .header
                .map
                .heap_ref(heap)
                .kind()
                .is_constructor()
        })
    };
    if !ok {
        return Err(VmError::Type);
    }
    Ok(v)
}

/// superCtor.prototype validation: (value) -> value, TypeError unless the
/// value is an Object or null.
fn throw_if_not_object_or_null<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let v = {
        let heap = &*heap;
        args.get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?
            .raw()
    };
    let ok = {
        let heap = &*heap;
        v == heap.known().null.as_tagged(heap).raw()
            || !Convert::is_primitive(heap, unsafe { v.assume_valid(heap) })
    };
    if !ok {
        return Err(VmError::Type);
    }
    Ok(unsafe { Tagged::<Value>::from_value_unchecked(v) })
}

/// [[ThisBindingStatus]] guard of derived constructors (ES 10.2.2):
/// (value) -> value, ReferenceError when `this` is still the hole.
fn throw_super_not_called_if_hole<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let v = {
        let heap = &*heap;
        args.get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?
            .raw()
    };
    let cond_31 = {
        let heap = &*heap;
        v == heap.known().the_hole.as_tagged(heap).raw()
    };
    if cond_31 {
        // "Must call super constructor before accessing 'this'"
        return Err(VmError::Reference);
    }
    Ok(unsafe { Tagged::<Value>::from_value_unchecked(v) })
}

/// InitializeThisBinding guard (ES 10.2.2): (value) -> value,
/// ReferenceError unless `this` is still the hole (super() runs once).
fn throw_super_already_called_if_not_hole<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let v = {
        let heap = &*heap;
        args.get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?
            .raw()
    };
    let cond_32 = {
        let heap = &*heap;
        v == heap.known().the_hole.as_tagged(heap).raw()
    };
    if !cond_32 {
        // "Super constructor may only be called once"
        return Err(VmError::Reference);
    }
    Ok(unsafe { Tagged::<Value>::from_value_unchecked(v) })
}

// ---- super() construction (ES 15.4.3) ---------------------------------------

/// The shared ConstructSuper tail: construct `callee` with `new_target`,
/// giving derived parents the hole receiver. The instance lands in the
/// return value; the exception sentinel escapes when user code threw.
fn construct_super_construct<'a>(
    nctx: RuntimeContext<'a>,
    callee_v: Value,
    new_target_v: Value,
    args: &[Value],
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        // Safety: fresh root-slot words for the oddball singletons.
        let undefined = heap.known().undefined.as_tagged(heap).raw();
        let the_hole = heap.known().the_hole.as_tagged(heap).raw();
        if new_target_v == undefined || new_target_v == the_hole {
            // not inside a [[Construct]]: reachable via an arrow that escaped
            // the constructor
            return Err(VmError::Type);
        }
        let derived = {
            let heap = &*heap;
            // Safety: caller-supplied word, fresh at entry.
            unsafe { callee_v.assume_valid(heap) }
                .as_heap_object()
                .and_then(|obj| obj.as_ref().callable_info(heap))
                .is_some_and(|info| info.function_kind().is_derived_class_constructor())
        };
        // Safety: caller-supplied words, fresh at entry, rooted below.
        let Some(callee) =
            scope.cast::<Object>(unsafe { Tagged::<Value>::from_value_unchecked(callee_v) })
        else {
            return Err(VmError::Type);
        };
        let Some(new_target) =
            scope.cast::<Object>(unsafe { Tagged::<Value>::from_value_unchecked(new_target_v) })
        else {
            return Err(VmError::Type);
        };
        // root the forwarded arguments before anything allocates: they
        // are raw copies of caller stack slots and go stale when a GC
        // moves their targets (create_construct_receiver allocates)
        // Safety: caller-supplied words, fresh at entry, rooted below.
        let args: Vec<Handle<'_, Value>> = args
            .iter()
            .map(|v| scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(*v) }))
            .collect();
        let (receiver, allocated) = if derived {
            // Safety: fresh root-slot word, rooted below.
            (
                scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(the_hole) }),
                false,
            )
        } else {
            match Object::create_construct_receiver_value(vm, heap, state, new_target.erase()) {
                // the receiver must stay rooted across the callee call:
                // a primitive return falls back to it after the call
                // allocated (and possibly moved it)
                Ok(Some(r)) => (scope.handle(r), true),
                Ok(None) => return Ok(heap.known().exception.as_tagged(heap).erase()),
                Err(err) => return Err(err),
            }
        };
        // Safety: fresh rooted-slot words staged for the call.
        let mut args_v = Vec::with_capacity(args.len() + 1);
        args_v.push(receiver.as_tagged(&*heap).raw());
        args_v.extend(args.iter().map(|h| unsafe { h.read_unchecked() }));
        let result = scope.handle(RuntimeContext::call(
            vm,
            &mut *heap,
            state,
            callee.erase(),
            scope.stage(
                &args_v
                    .iter()
                    .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                    .collect::<Vec<_>>(),
            ),
            Some(new_target.erase()),
        )?);
        let cond_33 = {
            let heap = &*heap;
            result.as_tagged(heap).raw() == heap.known().exception.as_tagged(heap).raw()
        };
        if cond_33 {
            return Ok(result.as_tagged(heap));
        }
        let cond_34 = {
            let heap = &*heap;
            Convert::is_primitive(heap, result.as_tagged(heap))
        };
        if cond_34 {
            if allocated {
                Ok(receiver.as_tagged(heap))
            } else {
                // a derived constructor returned a primitive: only
                // reachable via `return <primitive>` (ES 9.2.2.1)
                Err(VmError::Type)
            }
        } else {
            Ok(result.as_tagged(heap))
        }
    })
}

/// super(...): (args...) -> instance. Resolves the super constructor and
/// new.target from the executing frame.
fn construct_super<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let (callee, new_target) = frame_super_parts(heap, state)?;
    // Safety: fresh register words, consumed below.
    let arg_words = {
        let heap = &*heap;
        args.iter()
            .map(|h| h.as_tagged(heap))
            .map(|v| v.raw())
            .collect::<Vec<_>>()
    };
    construct_super_construct(
        RuntimeContext::new(vm, heap, state),
        callee,
        new_target,
        &arg_words,
    )
}

/// super() forwarding the frame's full argument list (synthesized default
/// derived constructors, ES 15.7.13): () -> instance.
fn construct_super_all_args<'a>(
    nctx: RuntimeContext<'a>,
    _args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let (callee, new_target, args) = {
        let (callee, new_target) = frame_super_parts(heap, state)?;
        if !state.cache.is_active() {
            return Err(VmError::Type);
        }
        let meta = state.cache.frame_meta();
        let argc = state.stack.argc(&meta).saturating_sub(1);
        let slice = state.stack.args(&meta, -2, argc);
        // Safety: fresh register words, consumed below.
        let args = slice
            .iter()
            .map(|h| h.as_tagged(&*heap))
            .map(|v| v.raw())
            .collect::<Vec<_>>();
        (callee, new_target, args)
    };
    construct_super_construct(
        RuntimeContext::new(vm, heap, state),
        callee,
        new_target,
        &args,
    )
}

/// Arrow-delegated super(): (args..., closure, new_target) -> instance.
/// The constructor closure and its new.target ride the tail of the
/// argument window (threaded through .this_function).
fn construct_super_via<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let n = args.len();
    if n < 2 {
        return Err(VmError::Arity);
    }
    // Safety: fresh argument words, consumed below.
    let (closure, new_target, arg_words) = {
        let heap = &*heap;
        (
            args.get(n - 2)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(n - 1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            (0..n - 2)
                .map(|i| args.get(i).map(|h| h.as_tagged(heap)).map(|v| v.raw()))
                .collect::<Option<Vec<_>>>()
                .ok_or(VmError::Arity)?,
        )
    };
    let callee = 'callee: {
        let heap = &*heap;
        let Some(obj) = unsafe { closure.assume_valid(heap) }.as_heap_object() else {
            break 'callee None;
        };
        let proto = obj.as_ref().header.map.heap_ref(heap).prototype.inner();
        let Some(proto_obj) = unsafe { proto.assume_valid(heap) }.as_heap_object() else {
            break 'callee None;
        };
        if !proto_obj
            .as_ref()
            .header
            .map
            .heap_ref(heap)
            .kind()
            .is_constructor()
        {
            break 'callee None;
        }
        Some(proto)
    };
    let Some(callee) = callee else {
        return Err(VmError::Type);
    };
    construct_super_construct(
        RuntimeContext::new(vm, heap, state),
        callee,
        new_target,
        &arg_words,
    )
}

// ---- dynamic names (direct eval) ---------------------------------------------

/// Direct-eval name load: (name) -> value. Walks the frame context chain
/// by name; unresolved names fall back to the global object.
fn load_dynamic_name<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    // Safety: fresh argument word, consumed below.
    let name = {
        let heap = &*heap;
        args.get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?
            .raw()
    };
    let found = dynamic_lookup_frame(heap, state, name)?;
    match found {
        Some(v) if !is_the_hole(heap, v) => Ok(unsafe { Tagged::<Value>::from_value_unchecked(v) }),
        Some(_) => Err(VmError::Reference),
        None => {
            // unresolved: fall back to a global object property
            // Safety: fresh root-slot word, consumed below.
            let global = unsafe { heap.known().global_object.read_unchecked() };
            get_property_lenient(RuntimeContext::new(vm, heap, state), global, name)
        }
    }
}

/// Direct-eval name store: (value, name) -> value. Writes through to the
/// context-chain slot; unresolved names store on the global object.
fn store_dynamic_name<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    // Safety: fresh argument words, consumed below.
    let (value, name) = {
        let heap = &*heap;
        (
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
        )
    };
    let found = dynamic_lookup_frame(heap, state, name)?;
    match found {
        Some(v) if !is_the_hole(heap, v) => {
            // write through to the found slot
            {
                let context = frame_context_value(state, heap)?;
                let mut context = context.get_as::<Context>().ok_or(VmError::Type)?;
                let target = dynamic_slot(heap, &mut context, name)?;
                // Safety: fresh anchored word, stored below.
                let host = context.into_tagged().raw();
                // Safety: caller-supplied word, fresh at entry, stored now.
                let v = unsafe { value.assume_valid(heap) };
                target.set(heap, host, v);
                Ok(())
            }?;
        }
        Some(_) => return Err(VmError::Reference),
        None => {
            // Safety: fresh root-slot word, consumed below.
            let global = unsafe { heap.known().global_object.read_unchecked() };
            let threw = state.handle_scope(|scope| -> Result<bool, VmError> {
                let outcome = {
                    let heap = &*heap;
                    // Safety: fresh root-slot word, consumed here.
                    unsafe { global.assume_valid(heap) }.store_lookup(
                        heap,
                        &scope,
                        // Safety: caller-supplied name word, fresh at entry.
                        unsafe { name.assume_valid(heap) }.as_name(),
                        // Safety: caller-supplied word, fresh at entry, stored here.
                        unsafe { value.assume_valid(heap) },
                        StoreSemantics::WriteThrough,
                    )
                }?;
                apply_store_outcome(vm, heap, state, global, outcome, value)
            })?;
            if threw {
                return Ok(heap.known().exception.as_tagged(heap).erase());
            }
        }
    }
    Ok(args
        .get(0)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?)
}

/// Fresh singleton-word compare against the hole sentinel.
fn is_the_hole(heap: &Heap, v: Value) -> bool {
    v == heap.known().the_hole.as_tagged(heap).raw()
}

// ---- rest parameters ---------------------------------------------------------

/// A fresh array of the frame's arguments from formal index `first`:
/// (first) -> array.
fn create_rest_parameter<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, state, .. } = nctx;
    let first = {
        let heap = &*heap;
        Smi::decode(
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
        )
        .map(|s| s.value() as usize)
        .unwrap_or(0)
    };
    let values: Vec<Value> = {
        if !state.cache.is_active() {
            return Err(VmError::Type);
        }
        let meta = state.cache.frame_meta();
        let argc = state.stack.argc(&meta); // receiver included
        let count = argc.saturating_sub(1).saturating_sub(first);
        // Safety: fresh register words, consumed below.
        (0..count)
            .map(|i| {
                state
                    .stack
                    .reg(heap, &meta, -((first + i + 2) as i32))
                    .raw()
            })
            .collect()
    };
    state.handle_scope(|scope| {
        let elements = heap.allocate_handle::<FixedArray>(
            scope.stage(
                &values
                    .iter()
                    .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                    .collect::<Vec<_>>(),
            ),
            &scope,
        );
        let map = heap.known().js_array_map;
        Ok(heap
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values: HandleSlice::EMPTY,
                    elements: elements.erase(),
                    length: values.len(),
                },
            )
            .erase())
    })
}

// ---- super property access (ES 15.4.2 / 15.4.4) -------------------------------

/// super.x load: (home, recv, key) -> value. GetSuperBase of the home
/// object walked with the split receiver/lookup-start; key coercion runs
/// after the parent link is resolved (user toString must not change the
/// chain searched).
fn super_get_property<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    // Safety: fresh argument words; nothing allocates before they are
    // rooted / consumed below.
    let (home, raw_recv, raw_key) = {
        let heap = &*heap;
        (
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(2)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
        )
    };
    let cond_35 = {
        let heap = &*heap;
        raw_recv == heap.known().the_hole.as_tagged(heap).raw()
    };
    if cond_35 {
        // super.x before super() in a derived constructor
        return Err(VmError::Reference);
    }
    // the key coercion allocates: root home/recv across it
    state.handle_scope(|scope| {
        // Safety: fresh argument words, rooted below before any allocation.
        let home = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(home) });
        let recv = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_recv) });
        let Some(key) = Object::to_property_key(
            vm,
            heap,
            state,
            // Safety: fresh argument word, fresh at entry.
            scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_key) }),
        )?
        else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        // re-read through the handles: the coercion above allocated
        let recv = recv.as_tagged(heap).raw();
        // (plain value, getter) — both raw fresh words
        // Safety: fresh anchored handle words plus a fresh coercion result.
        let outcome = {
            let heap = &*heap;
            let proto = home_proto(heap, home.as_tagged(heap));
            let name = match Lookup::classify_key(heap, key.as_tagged(heap).erase())? {
                Key::Element(i) => Tagged::from(Smi::new(i as i64)),
                Key::Name(name) => name,
            };
            match super_lookup_from_proto(heap, proto, name)? {
                LoadOutcome::Value(v) => (Some(v.raw()), None),
                LoadOutcome::Getter(g) => (None, Some(g.raw())),
            }
        };
        if let Some(v) = outcome.0 {
            return Ok(unsafe { Tagged::<Value>::from_value_unchecked(v) });
        }
        // Safety: fresh walk word from the lookup above.
        let getter = outcome.1.expect("one of the two arms is set");
        let undefined = {
            let heap = &*heap;
            heap.known().undefined.as_tagged(heap).raw()
        };
        if getter == undefined
            || !{
                let heap = &*heap;
                Object::is_callable(heap, unsafe {
                    Tagged::<Value>::from_value_unchecked(getter)
                })
            }
        {
            return Ok(unsafe { Tagged::<Value>::from_value_unchecked(undefined) });
        }
        RuntimeContext::call(
            vm,
            heap,
            state,
            scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(getter) }),
            // Safety: fresh rooted-slot word, staged for the call.
            scope.stage(&[unsafe { Tagged::<Value>::from_value_unchecked(recv) }]),
            None,
        )
    })
}

/// super.x store: (home, recv, key, value, semantics) -> value. ES stores
/// shadow inherited data properties on `this` unless the write-through
/// semantics flag is set; the parent link is resolved before any user key
/// coercion runs.
fn super_set_property<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    // Safety: fresh argument words; nothing allocates before they are
    // rooted / consumed below.
    let (home, raw_recv, raw_key, raw_value, semantics_flag) = {
        let heap = &*heap;
        let semantics_flag = Smi::decode(
            args.get(4)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
        )
        .map(|s| s.value() as u32)
        .unwrap_or(0);
        (
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(2)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            args.get(3)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw(),
            semantics_flag,
        )
    };
    let cond_36 = {
        let heap = &*heap;
        raw_recv == heap.known().the_hole.as_tagged(heap).raw()
    };
    if cond_36 {
        return Err(VmError::Reference);
    }
    // the key coercion allocates: root home/recv/value across it
    state.handle_scope(|scope| {
        // Safety: fresh argument words, rooted below before any allocation.
        let home = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(home) });
        let recv = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_recv) });
        let value = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_value) });
        let Some(key) = Object::to_property_key(
            vm,
            heap,
            state,
            // Safety: fresh argument word, fresh at entry.
            scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_key) }),
        )?
        else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        // Safety: rooted handle words, consumed below.
        let recv_word = unsafe { recv.read_unchecked() };
        let value_word = unsafe { value.read_unchecked() };
        let semantics = if semantics_flag & bytecode::SUPER_STORE_WRITE_THROUGH != 0 {
            StoreSemantics::WriteThrough
        } else {
            StoreSemantics::Shadow
        };
        let outcome = {
            let heap = &*heap;
            let proto = home_proto(heap, home.as_tagged(heap));
            let name = match Lookup::classify_key(heap, key.as_tagged(heap).erase())? {
                Key::Element(i) => Tagged::from(Smi::new(i as i64)),
                Key::Name(name) => name,
            };
            super_store_lookup(
                heap,
                &scope,
                proto,
                unsafe { Tagged::<Value>::from_value_unchecked(recv_word) },
                name,
                unsafe { Tagged::<Value>::from_value_unchecked(value_word) },
                semantics,
            )
        }?;
        if apply_store_outcome(vm, heap, state, recv_word, outcome, value_word)? {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        }
        Ok(value.as_tagged(heap))
    })
}
