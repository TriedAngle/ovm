//! The CallRuntime runtimes: one implementation per `bytecode::RuntimeFn`,
//! plus the `runtime_fn` table the runtime registry seeds its fixed
//! 0..COUNT range with. These are compiled-language semantics (each cites
//! its ES section), not JS-visible library functions.

use crate::{
    AccessorPair, Context, Convert, DenseString, FixedArray, Handle, HandleSlice, Heap, Key,
    LoadOutcome, Lookup, Object, ObjectSlotsInit, PropertyDescriptor, SlotName, Smi, StoreOutcome,
    StoreSemantics, StringData, Symbol, Tagged, Transition, Value, VmError, runtime::Coercion,
};

use crate::Float;
use crate::GcSlot;
use crate::HandleScope;
use crate::PartialDescriptor;
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
    let target = args.get(0).ok_or(VmError::Arity)?;
    let raw_key = args.get(1).ok_or(VmError::Arity)?;
    // the reference's key is coerced before the base is touched (ES
    // 13.15.5 EvaluatePropertyAccess: user toString/valueOf of a
    // computed key runs even when the delete afterwards throws);
    // the coercion allocates, so the base must stay rooted across it
    state.handle_scope(|scope| {
        let Some(key) = Object::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        // proxies run their `deleteProperty` trap (ES 20.2.5.4); the
        // returned boolean flows through the strict handling below
        let ok = if Proxy::is_proxy(heap, target.as_tagged(heap)) {
            match Proxy::delete(vm, heap, state, target, key.erase())? {
                Coercion::Threw => {
                    return Ok(heap.known().exception.as_tagged(heap).erase());
                }
                Coercion::Value(v) => {
                    let v = scope.handle(v);
                    Convert::is_truthy(heap, v.as_tagged(heap))
                }
            }
        } else {
            delete_property_core(heap, &scope, target, key.erase())?
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
    target: Handle<'_, Value>,
    key: Handle<'_, Value>,
) -> Result<bool, VmError> {
    // ToObject (ES 7.2.3): a null/undefined base throws
    let nullish = {
        let target = target.as_tagged(heap);
        let null = heap.known().null.as_tagged(heap);
        let undefined = heap.known().undefined.as_tagged(heap);
        target.ptr_eq(null) || target.ptr_eq(undefined)
    };
    if nullish {
        return Err(VmError::Type);
    }
    // primitives: ToObject creates a fresh wrapper whose only own
    // properties are a string's non-configurable length/indices
    if Convert::is_primitive(heap, target.as_tagged(heap)) {
        let owned = string_exotic_own(heap, target.as_tagged(heap), key.as_tagged(heap));
        return Ok(!owned);
    }
    let receiver = scope
        .cast::<Object>(target.as_tagged(heap))
        .expect("non-primitive receivers are objects");
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
    data.matches_ascii(b"length") || Lookup::canonical_index(data).is_some_and(|i| i < s.len())
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
    let name = args.get(0).ok_or(VmError::Arity)?;
    let ok = state.handle_scope(|scope| {
        let global = scope.handle(heap.known().global_object.as_tagged(heap).erase());
        delete_property_core(heap, &scope, global, name)
    })?;
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

/// Stage rooted handles into a fresh contiguous slice of `scope` slots:
/// the words are copied out of their (short-lived) anchors so they can be
/// consumed by an allocation or call.
fn stage_handles<'s>(
    heap: &Heap,
    scope: &'s HandleScope<'_>,
    handles: &[Handle<'_, Value>],
) -> HandleSlice<'s> {
    let anchored: Vec<Tagged<'_, Value>> =
        handles.iter().map(|h| h.as_tagged(heap).erase()).collect();
    scope.stage(&anchored)
}

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
    let subject = args.get(0).ok_or(VmError::Arity)?;
    state.handle_scope(|scope| {
        let subject = subject.as_tagged(heap);
        if subject.ptr_eq(heap.known().null.as_tagged(heap))
            || subject.ptr_eq(heap.known().undefined.as_tagged(heap))
        {
            return Ok(heap.known().undefined.as_tagged(heap).erase());
        }
        let Some(level) = for_in_initial_level(heap, subject) else {
            return Ok(heap.known().undefined.as_tagged(heap).erase());
        };
        // the level must survive the key collection and FixedArray
        // allocation below (both allocate)
        let level = scope.handle(level);
        let keys = for_in_level_keys(vm, heap, &scope, level)?;
        let staged = stage_handles(heap, &scope, &keys);
        let keys = heap.allocate_handle::<FixedArray>(staged, &scope);
        let empty = heap.known().empty_fixed_array;
        let map = heap.known().for_in_enumerator_map;
        let words = [
            level.as_tagged(heap).erase(),
            keys.as_tagged(heap).erase(),
            Smi::new(0).into_tagged(),
            empty.as_tagged(heap).erase(),
        ];
        let staged = scope.stage(&words);
        let enumerator = heap.new_object(&scope, map, staged);
        Ok(enumerator.erase())
    })
}

/// Level 0 of the chain for a subject: objects are their own level 0;
/// string primitives enumerate their indices (a fresh wrapper would be
/// unobservable otherwise). Other primitives have no own properties —
/// their level 0 is the constructor's prototype, so additions to
/// `Number.prototype` etc. are observable (ES 14.7.5.9: the walk starts
/// at ToObject(subject)). `None` when the prototype is unreachable.
fn for_in_initial_level<'a>(
    heap: &'a Heap,
    subject: Tagged<'a, Value>,
) -> Option<Tagged<'a, Value>> {
    if subject.get_as::<DenseString>().is_some() {
        return Some(subject);
    }
    if !Convert::is_primitive(heap, subject) {
        return Some(subject);
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
    let global = heap.known().global_object.as_tagged(heap);
    let strings = heap.known().strings;
    let ctor_handle = match ctor_name {
        "Number" => strings.number_ctor,
        "Boolean" => strings.boolean_ctor,
        _ => strings.symbol_ctor,
    };
    let ctor = match Lookup::load_outcome(heap, global.erase(), ctor_handle.as_tagged(heap)).ok()? {
        LoadOutcome::Value(v) if v.is_strong_ptr() => v,
        _ => return None,
    };
    match Lookup::load_outcome(heap, ctor, heap.known().strings.prototype.as_tagged(heap)).ok()? {
        LoadOutcome::Value(p) if p.is_strong_ptr() => Some(p),
        _ => None,
    }
}

/// The own string keys of a level in [[OwnPropertyKeys]] order (ES
/// 10.1.11: array indices ascending, then strings in insertion order;
/// symbols excluded). Index keys are interned to their canonical string
/// form, the value a for-in binding receives. Enumerability is NOT
/// filtered here: EnumerateObjectProperties checks it lazily per key,
/// and non-enumerable own keys must still register as visited.
fn for_in_level_keys<'s>(
    vm: &VM,
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    level: Handle<'_, Value>,
) -> Result<Vec<Handle<'s, Value>>, VmError> {
    // raw pass: Smi index keys (to be interned) and ready name keys
    let (mut indices, names) = 'keys: {
        let level = level.as_tagged(heap);
        let mut indices: Vec<i64> = Vec::new();
        let mut names: Vec<Handle<'s, Value>> = Vec::new();
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
                    names.push(scope.handle(name.erase()));
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
                .and_then(|s| Lookup::canonical_index(s.as_ref().data(heap)))
                .filter(|i| *i < u32::MAX as usize);
            match index {
                Some(i) => indices.push(i as i64),
                None => names.push(scope.handle(name.erase())),
            }
        }
        (indices, names)
    };

    indices.sort_unstable();
    indices.dedup();
    // root every key before the next allocates: interning index keys
    // promotes earlier results, and raw copies would dangle
    let mut keys: Vec<Handle<'s, Value>> = Vec::with_capacity(indices.len() + names.len());
    for i in indices {
        let s = vm.interner().intern_str(heap, scope, &i.to_string());
        keys.push(s.erase());
    }
    keys.extend(names);
    Ok(keys)
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
    let enumerator = args.get(0).ok_or(VmError::Arity)?;
    state.handle_scope(|scope| {
        if enumerator
            .as_tagged(heap)
            .ptr_eq(heap.known().undefined.as_tagged(heap))
        {
            // nullish subject: the head produced no enumerator
            return Ok(heap.known().undefined.as_tagged(heap).erase());
        }
        loop {
            // one candidate per turn: the cursor advances before the
            // key is examined, so skipped keys are never revisited
            let candidate = 'candidate: {
                let Some(obj) = enumerator.as_tagged(heap).as_heap_object() else {
                    return Err(VmError::Type);
                };
                let slots = obj.as_ref().slots.get(heap);
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
                Some(scope.handle(key))
            };
            let Some(key) = candidate else {
                // snapshot exhausted: advance to the live prototype
                let level = {
                    let Some(obj) = enumerator.as_tagged(heap).as_heap_object() else {
                        return Err(VmError::Type);
                    };
                    scope.handle(obj.as_ref().slots.get(heap).at(heap, FOR_IN_LEVEL))
                };
                let Some(proto) = for_in_next_level(heap, &scope, level)? else {
                    return Ok(heap.known().undefined.as_tagged(heap).erase());
                };
                let keys = for_in_level_keys(vm, heap, &scope, proto)?;
                let staged = stage_handles(heap, &scope, &keys);
                let keys = heap.allocate_handle::<FixedArray>(staged, &scope);
                let Some(obj) = enumerator.as_tagged(heap).as_heap_object() else {
                    return Err(VmError::Type);
                };
                let slots = obj.as_ref().slots.get(heap);
                slots.set(heap, FOR_IN_LEVEL, proto.as_tagged(heap).erase());
                slots.set(heap, FOR_IN_KEYS, keys.as_tagged(heap).erase());
                slots.set(heap, FOR_IN_INDEX, Smi::new(0).into_tagged());
                continue;
            };
            // lazy [[GetOwnProperty]] on the key's own level: a key
            // deleted since the snapshot is skipped without registering
            let level = {
                let Some(obj) = enumerator.as_tagged(heap).as_heap_object() else {
                    return Err(VmError::Type);
                };
                obj.as_ref().slots.get(heap).at(heap, FOR_IN_LEVEL)
            };
            let own = for_in_own_state(heap, level, key.as_tagged(heap));
            let Some(enumerable) = own else {
                continue;
            };
            // already registered (yielded earlier, or shadowing
            // non-enumerable on a closer level): skip
            let seen = {
                let Some(obj) = enumerator.as_tagged(heap).as_heap_object() else {
                    return Err(VmError::Type);
                };
                let visited = obj
                    .as_ref()
                    .slots
                    .get(heap)
                    .at(heap, FOR_IN_VISITED)
                    .get_as::<FixedArray>()
                    .ok_or(VmError::Type)?;
                let key_word = key.as_tagged(heap);
                visited
                    .as_slice()
                    .iter()
                    .any(|s| s.get(heap).ptr_eq(key_word))
            };
            if seen {
                continue;
            }
            // register the key — yielded or shadowing, both at most once
            {
                let visited = {
                    let Some(obj) = enumerator.as_tagged(heap).as_heap_object() else {
                        return Err(VmError::Type);
                    };
                    let visited = obj
                        .as_ref()
                        .slots
                        .get(heap)
                        .at(heap, FOR_IN_VISITED)
                        .get_as::<FixedArray>()
                        .ok_or(VmError::Type)?;
                    let mut words: Vec<Tagged<'_, Value>> =
                        visited.as_slice().iter().map(|s| s.get(heap)).collect();
                    words.push(key.as_tagged(heap));
                    words
                };
                let staged = scope.stage(&visited);
                let visited = heap.allocate_handle::<FixedArray>(staged, &scope);
                let Some(obj) = enumerator.as_tagged(heap).as_heap_object() else {
                    return Err(VmError::Type);
                };
                obj.as_ref().slots.get(heap).set(
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
fn for_in_next_level<'s>(
    heap: &Heap,
    scope: &'s HandleScope<'_>,
    level: Handle<'_, Value>,
) -> Result<Option<Handle<'s, Value>>, VmError> {
    let level = level.as_tagged(heap);
    if level.get_as::<DenseString>().is_some() {
        // String.prototype via the global object (both plain data
        // lookups; no user code can run)
        let global = heap.known().global_object.as_tagged(heap);
        let Some(string_ctor) = Lookup::load_outcome(
            heap,
            global.erase(),
            heap.known().strings.string.as_tagged(heap),
        )
        .ok()
        .and_then(|o| match o {
            LoadOutcome::Value(v) => Some(v),
            LoadOutcome::Getter(_) => None,
        }) else {
            return Ok(None);
        };
        let proto = Lookup::load_outcome(
            heap,
            string_ctor,
            heap.known().strings.prototype.as_tagged(heap),
        )
        .ok()
        .and_then(|o| match o {
            LoadOutcome::Value(v) => Some(v),
            LoadOutcome::Getter(_) => None,
        });
        return Ok(proto.filter(|p| p.is_strong_ptr()).map(|p| scope.handle(p)));
    }
    let Some(obj) = level.as_heap_object() else {
        return Ok(None);
    };
    let proto = obj.as_ref().map_ref(heap).prototype.get(heap);
    let hole = heap.known().the_hole.as_tagged(heap);
    if proto.ptr_eq(hole) || proto.ptr_eq(heap.known().null.as_tagged(heap)) {
        return Ok(None);
    }
    // a FixedArray prototype is the Self-style multi-parent form;
    // the chain walk does not model it (ends the enumeration)
    Ok(proto
        .get_as::<FixedArray>()
        .map_or(Some(proto), |_| None)
        .map(|p| scope.handle(p)))
}

/// The lazy own-property state of `key` on its own level: `None` when
/// the property is gone (deleted since the snapshot), else its
/// [[Enumerable]]. Own-only — the shadow check against other levels is
/// the visited set's job.
fn for_in_own_state(heap: &Heap, level: Tagged<'_, Value>, key: Tagged<'_, Value>) -> Option<bool> {
    match Lookup::classify_key(heap, key).ok()? {
        Key::Element(i) => {
            if let Some(s) = level.get_as::<DenseString>() {
                // string indices are enumerable own properties
                return Some((i as u64) < s.len() as u64);
            }
            let obj = level.as_heap_object()?;
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
            let obj = level.as_heap_object()?;
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
        let obj = args.get(0).ok_or(VmError::Arity)?;
        let symbol = scope.handle(heap.known().iterator_symbol.as_tagged(heap).erase());
        let method = Lookup::get_property_on(vm, heap, state, obj, obj, symbol)?;
        let method = match method {
            Coercion::Threw => {
                return Ok(heap.known().exception.as_tagged(heap).erase());
            }
            Coercion::Value(v) => scope.handle(v),
        };
        let method_tagged = method.as_tagged(heap);
        if method_tagged.ptr_eq(heap.known().undefined.as_tagged(heap))
            || method_tagged.ptr_eq(heap.known().null.as_tagged(heap))
            || !Object::is_callable(heap, method_tagged)
        {
            return Err(VmError::Type); // "obj is not iterable"
        }
        let call_args = stage_handles(heap, &scope, &[obj]);
        RuntimeContext::call(vm, heap, state, method, call_args, None)
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
        let iter = args.get(0).ok_or(VmError::Arity)?;
        let next_name = scope.handle(heap.known().strings.next.as_tagged(heap).erase());
        let next = Lookup::get_property_on(vm, heap, state, iter, iter, next_name)?;
        let next = match next {
            Coercion::Threw => {
                return Ok(heap.known().exception.as_tagged(heap).erase());
            }
            Coercion::Value(v) => scope.handle(v),
        };
        let call_args = stage_handles(heap, &scope, &[iter]);
        let result = scope.handle(RuntimeContext::call(
            vm, heap, state, next, call_args, None,
        )?);
        if result
            .as_tagged(heap)
            .ptr_eq(heap.known().exception.as_tagged(heap))
        {
            return Ok(result.as_tagged(heap));
        }
        if Convert::is_primitive(heap, result.as_tagged(heap)) {
            return Err(VmError::Type); // IteratorNext result must be an Object
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
    let key = args.get(0).ok_or(VmError::Arity)?;
    let obj = args.get(1).ok_or(VmError::Arity)?;
    // the key coercion allocates (wrapper keys run toString/valueOf):
    // the receiver stays rooted in its argument handle across it
    state.handle_scope(|scope| {
        let Some(key) = Object::to_property_key(vm, heap, state, key)? else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        if Proxy::is_proxy(heap, obj.as_tagged(heap)) {
            let has = match Proxy::has(vm, heap, state, obj, key.erase())? {
                Coercion::Threw => return Ok(heap.known().exception.as_tagged(heap).erase()),
                Coercion::Value(v) => scope.handle(v),
            };
            return Ok(has.as_tagged(heap));
        }
        // lookup::has_property covers array `length` slots along the chain
        let has = Lookup::has_property(heap, obj.as_tagged(heap), key.as_tagged(heap));
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
    let target = args.get(n - 2).ok_or(VmError::Arity)?;
    let source = args.get(n - 1).ok_or(VmError::Arity)?;
    let excluded: Vec<Handle<'_, Value>> =
        match (0..n - 2).map(|i| args.get(i)).collect::<Option<Vec<_>>>() {
            Some(excluded) => excluded,
            None => return Err(VmError::Arity),
        };
    {
        let source_tagged = source.as_tagged(heap);
        // null/undefined and other primitives contribute nothing (string
        // sources would need boxing)
        if source_tagged.ptr_eq(heap.known().null.as_tagged(heap))
            || source_tagged.ptr_eq(heap.known().undefined.as_tagged(heap))
            || Convert::is_primitive(heap, source_tagged)
        {
            return Ok(target.as_tagged(heap));
        }
    }
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        // canonicalize the excluded keys (interning strings) so a plain
        // bits comparison suffices against the source's descriptor names
        let excluded: Vec<Handle<'_, SlotName>> = {
            let mut out = Vec::with_capacity(excluded.len());
            for k in excluded {
                match Object::to_property_key(vm, heap, state, k)? {
                    Some(k) => out.push(scope.handle(k)),
                    None => return Ok(heap.known().exception.as_tagged(heap).erase()),
                }
            }
            out
        };
        // enumerate own enumerable keys: element indices ascending, then
        // named descriptors in insertion order; collected AFTER the
        // exclusion canonicalization so no allocation can stale them
        let keys: Vec<Handle<'_, Value>> = {
            let Some(obj) = source.as_tagged(heap).as_heap_object() else {
                return Ok(target.as_tagged(heap));
            };
            let obj = obj.as_ref();
            let mut keys = Vec::new();
            if obj.is_array(heap) {
                let len = obj
                    .length()
                    .min(obj.elements_array(heap).map(|e| e.len()).unwrap_or(0));
                for i in 0..len {
                    if obj.element_value(heap, i).is_some() {
                        keys.push(scope.handle(Smi::new(i as i64)));
                    }
                }
            }
            for d in obj.header.map.get(heap).descriptors() {
                if d.flags().is_enumerable() {
                    keys.push(scope.handle(d.name(heap).erase()));
                }
            }
            keys
        };
        for key in keys {
            let key_name = key.as_tagged(heap).as_name();
            if excluded.iter().any(|e| e.as_tagged(heap).ptr_eq(key_name)) {
                continue;
            }
            // full [[Get]] (getters may run)
            let value = match Lookup::get_property_on(vm, heap, state, source, source, key)? {
                Coercion::Threw => return Ok(heap.known().exception.as_tagged(heap).erase()),
                Coercion::Value(v) => scope.handle(v),
            };
            // CreateDataProperty: skipped when already present
            let exists = !matches!(
                target.lookup(heap, key.as_tagged(heap).as_name()),
                Lookup::NotFound
            );
            if exists {
                continue;
            }
            let target_obj = scope
                .cast::<Object>(target.as_tagged(heap))
                .expect("copy target is an object");
            let key_name = scope.handle(key.as_tagged(heap).as_name());
            Object::add_own_property(
                heap,
                &scope,
                target_obj,
                key_name,
                PropertyDescriptor::data(value),
            )?;
        }
        // re-read through the handle: the copy loop allocated (getters,
        // property adds) and may have moved the target
        Ok(target.as_tagged(heap))
    })
}

/// A fresh private name: (description) -> Symbol.
fn create_private_name<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, state, .. } = nctx;
    let text = args
        .get(1)
        .map(|h| h.as_tagged(heap))
        .and_then(|d| d.get_as::<DenseString>())
        .map(|s| s.to_rust_string(heap));
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
    let obj = args.get(0).ok_or(VmError::Arity)?;
    let key = args.get(1).ok_or(VmError::Arity)?;
    match Lookup::private_find(heap, obj.as_tagged(heap), key.as_tagged(heap)) {
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
    let obj = args.get(0).ok_or(VmError::Arity)?;
    let key = args.get(1).ok_or(VmError::Arity)?;
    let value = args.get(2).ok_or(VmError::Arity)?;
    let value = value.as_tagged(heap);
    match Lookup::private_find(heap, obj.as_tagged(heap), key.as_tagged(heap)) {
        Some(slot) => {
            slot.set(heap, obj.as_tagged(heap), value);
            Ok(value)
        }
        None => Err(VmError::Type),
    }
}

/// `#x in obj`: (key, obj) -> bool (own private presence only).
fn private_in<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let key = args.get(0).ok_or(VmError::Arity)?;
    let obj = args.get(1).ok_or(VmError::Arity)?;
    let has = Lookup::private_find(heap, obj.as_tagged(heap), key.as_tagged(heap)).is_some();
    Ok(Convert::boolean(heap, has))
}

/// Attach the instance-field array to the class constructor:
/// (ctor, fields).
fn set_class_fields<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let ctor = args.get(0).ok_or(VmError::Arity)?;
    let fields = args.get(1).ok_or(VmError::Arity)?;
    let mut ok = false;
    if let Some(obj) = ctor.as_tagged(heap).as_heap_object() {
        let slots = obj.as_ref().slots.get(heap);
        if obj
            .as_ref()
            .header
            .map
            .get(heap)
            .kind()
            .is_class_constructor()
            && slots.len() >= 3
        {
            slots.as_ref().element_slot(2).set(
                heap,
                ctor.as_tagged(heap).erase(),
                fields.as_tagged(heap).erase(),
            );
            ok = true;
        }
    }
    if !ok {
        return Err(VmError::Type);
    }
    Ok(ctor.as_tagged(heap))
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
    let ctor = args.get(0).ok_or(VmError::Arity)?;
    let instance = args.get(1).ok_or(VmError::Arity)?;
    state.handle_scope(|scope| -> Result<Tagged<'a, Value>, VmError> {
        let fields = {
            let Some(obj) = ctor.as_tagged(heap).as_heap_object() else {
                return Err(VmError::Type);
            };
            let slots = obj.as_ref().slots.get(heap);
            if slots.len() < 3 {
                return Err(VmError::Type);
            }
            slots.at(heap, 2)
        };
        if fields.ptr_eq(heap.known().undefined.as_tagged(heap)) {
            return Ok(instance.as_tagged(heap));
        }
        // root the field list across the initializer calls below
        let fields = scope.handle(fields);
        let count = fields
            .as_tagged(heap)
            .as_heap_object()
            .map(|o| o.as_ref().length())
            .unwrap_or(0);
        let mut i = 0;
        while i + 1 < count {
            let raw_key = fields
                .as_tagged(heap)
                .as_heap_object()
                .and_then(|o| o.as_ref().element_value(heap, i))
                .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
            // computed keys need ToPropertyKey canonicalization
            let Some(key) = Object::to_property_key(vm, heap, state, scope.handle(raw_key))? else {
                return Ok(heap.known().exception.as_tagged(heap).erase());
            };
            let key = scope.handle(key);
            // recompute the initializer after the coercion (it allocated)
            let init = fields
                .as_tagged(heap)
                .as_heap_object()
                .and_then(|o| o.as_ref().element_value(heap, i + 1))
                .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
            let call_args = {
                let inst = instance.as_tagged(heap);
                scope.stage(&[inst])
            };
            let value = scope.handle(RuntimeContext::call(
                vm,
                heap,
                state,
                scope.handle(init),
                call_args,
                None,
            )?);
            if value
                .as_tagged(heap)
                .ptr_eq(heap.known().exception.as_tagged(heap))
            {
                return Ok(heap.known().exception.as_tagged(heap).erase());
            }
            let instance_obj = scope
                .cast::<Object>(instance.as_tagged(heap))
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
    })
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
    context: &mut Tagged<'a, Context>,
    name: Tagged<'a, Value>,
) -> Result<&'a GcSlot, VmError> {
    // both sides are interned (constant pool / ScopeInfo names), so
    // pointer identity decides — no content comparison in lookup
    name.get_as::<DenseString>().ok_or(VmError::Type)?;
    loop {
        let ctx = context.as_ref();
        let names = ctx.scope_info.get(heap).as_ref().names.get(heap);
        for i in 0..names.len() {
            if names.at(heap, i) == name {
                return Ok(ctx.slots.get(heap).as_ref().element_slot(i));
            }
        }
        match ctx.outer.get(heap) {
            Some(outer) => *context = outer,
            None => return Err(VmError::Reference),
        }
    }
}

/// Walk the current frame's context chain looking for a slot named
/// `name` (direct eval): Some(slot value) found (possibly the hole),
/// None when the whole chain lacks the name. The result is rooted in
/// `scope` so callers may allocate before inspecting it.
fn dynamic_lookup_frame<'s>(
    heap: &Heap,
    state: &ContextState,
    scope: &'s HandleScope<'_>,
    name: Handle<'_, Value>,
) -> Result<Option<Handle<'s, Value>>, VmError> {
    let context = frame_context_value(state, heap)?;
    let mut context = context.get_as::<Context>().ok_or(VmError::Type)?;
    match dynamic_slot(heap, &mut context, name.as_tagged(heap)) {
        Ok(slot) => Ok(Some(scope.handle(slot.get(heap)))),
        Err(VmError::Reference) => Ok(None),
        Err(e) => Err(e),
    }
}

/// The current frame's super constructor and new.target (direct
/// super() calls, ES 15.4.3): the running closure's [[Prototype]] must
/// be a constructor.
fn frame_super_parts<'a>(
    heap: &'a Heap,
    state: &ContextState,
) -> Result<(Tagged<'a, Value>, Tagged<'a, Value>), VmError> {
    if !state.cache.is_active() {
        return Err(VmError::Type);
    }
    let meta = state.cache.frame_meta();
    let Some(callee) = Lookup::super_constructor(heap, &state.stack, &meta) else {
        return Err(VmError::Type);
    };
    Ok((callee, state.stack.new_target_slot(&meta).get(heap)))
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
    receiver: Handle<'_, Value>,
    outcome: StoreOutcome<'_>,
    value: Handle<'_, Value>,
) -> Result<bool, VmError> {
    match outcome {
        StoreOutcome::Transition {
            receiver: recv,
            name,
        } => {
            state.handle_scope(|scope| {
                Object::add_own_property(heap, &scope, recv, name, PropertyDescriptor::data(value))
                    // TODO(strict-mode): a false result must throw in strict code;
                    // the current store path preserves its existing sloppy result.
                    .map(|_| false)
            })
        }
        StoreOutcome::CallSetter { setter } => state.handle_scope(|scope| {
            let call_args = {
                let recv = receiver.as_tagged(heap);
                let val = value.as_tagged(heap);
                scope.stage(&[recv, val])
            };
            let result = RuntimeContext::call(vm, heap, state, setter, call_args, None)?;
            let word = result.raw();
            Ok(word == heap.known().exception.as_tagged(heap).erase().raw())
        }),
        StoreOutcome::Done => Ok(false),
    }
}

/// A full [[Get]] that treats non-callable getters (an absent half of an
/// accessor pair) as undefined instead of throwing.
fn get_property_lenient<'a>(
    nctx: RuntimeContext<'a>,
    receiver: Handle<'_, Value>,
    name: Handle<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let (value, getter) = match Lookup::load_outcome(
            heap,
            receiver.as_tagged(heap),
            name.as_tagged(heap).as_name(),
        )? {
            LoadOutcome::Value(v) => (Some(scope.handle(v)), None),
            LoadOutcome::Getter(g) => (None, Some(scope.handle(g))),
        };
        if let Some(v) = value {
            return Ok(v.as_tagged(heap));
        }
        let getter = getter.expect("one of the two arms is set");
        let skip = {
            let getter_tagged = getter.as_tagged(heap);
            getter_tagged.ptr_eq(heap.known().undefined.as_tagged(heap).erase())
                || !Object::is_callable(heap, getter_tagged)
        };
        if skip {
            return Ok(heap.known().undefined.as_tagged(heap).erase());
        }
        let call_args = {
            let recv = receiver.as_tagged(heap);
            scope.stage(&[recv])
        };
        RuntimeContext::call(vm, heap, state, getter, call_args, None)
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
    let fn_value = args.get(0).ok_or(VmError::Arity)?;
    let raw_key = args.get(1).ok_or(VmError::Arity)?;
    let prefix = args
        .get(2)
        .map(|h| h.as_tagged(heap))
        .and_then(|v| Smi::decode(v.raw()))
        .map(|s| s.value())
        .unwrap_or(0);
    // name construction allocates (interning): the closure stays rooted
    // in its argument handle across it
    state.handle_scope(|scope| {
        let Some(key) = Object::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key).erase();
        let text = scope.handle(Convert::to_string(heap, &scope, key)?);
        let units = {
            let s = text
                .as_tagged(heap)
                .get_as::<DenseString>()
                .expect("ToString yields a string");
            let mut full: Vec<u16> = match prefix {
                1 => b"get ".iter().map(|&b| b as u16).collect(),
                2 => b"set ".iter().map(|&b| b as u16).collect(),
                _ => Vec::new(),
            };
            s.as_ref().data(heap).write_units(&mut full);
            full
        };
        let name = vm
            .interner()
            .intern(heap, &scope, StringData::Utf16(&units))
            .erase();
        let Some(fn_obj) = scope.cast::<Object>(fn_value.as_tagged(heap)) else {
            return Err(VmError::Type);
        };
        let name_key = heap.known().strings.name;
        // the closure's own placeholder is never writable nor an accessor,
        // so only explicit member defines match here
        let explicit = match fn_obj
            .as_tagged(heap)
            .lookup(heap, name_key.as_tagged(heap))
        {
            Lookup::Data { flags, .. } => flags.is_writable(),
            Lookup::Accessor { .. } => true,
            Lookup::NotFound => false,
        };
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
    let target = args.get(0).ok_or(VmError::Arity)?;
    let raw_key = args.get(1).ok_or(VmError::Arity)?;
    let closure = args.get(2).ok_or(VmError::Arity)?;
    let flags = args
        .get(3)
        .map(|h| h.as_tagged(heap))
        .and_then(|v| Smi::decode(v.raw()))
        .map(|s| s.value() as u32)
        .unwrap_or(0);
    // the key coercion allocates: target and closure stay rooted in
    // their argument handles across it
    state.handle_scope(|scope| {
        let Some(key) = Object::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        let is_getter = flags & 1 != 0;
        let enumerable = flags & bytecode::PropertyFlags::DontEnum.bits() == 0;
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
            for d in obj.as_ref().header.map.get(heap).descriptors() {
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
        let closure_word = closure.as_tagged(heap).erase();
        if is_getter {
            get = closure_word;
        } else {
            set = closure_word;
        }
        // the name must outlive this non-allocating region: root it
        let name = scope.handle(name);
        let desc = PropertyDescriptor::Accessor {
            get: scope.handle(get),
            set: scope.handle(set),
            enumerable,
            configurable: true,
        };
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
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    let raw_key = args.get(1).ok_or(VmError::Arity)?;
    let value = args.get(2).ok_or(VmError::Arity)?;
    let flags = args
        .get(3)
        .map(|h| h.as_tagged(heap))
        .and_then(|v| Smi::decode(v.raw()))
        .map(|s| s.value() as u32)
        .unwrap_or(0);
    // the key coercion allocates: receiver and value stay rooted in
    // their argument handles across it
    state.handle_scope(|scope| {
        let Some(key) = Object::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        // proxies run their `defineProperty` trap (ES 20.2.5.6); define
        // sites are strict-mode: a rejected define throws
        if Proxy::is_proxy(heap, receiver.as_tagged(heap)) {
            let enumerable = flags & bytecode::PropertyFlags::DontEnum.bits() == 0;
            let configurable = flags & bytecode::PropertyFlags::DontDelete.bits() == 0;
            let partial = if flags & bytecode::PropertyFlags::Accessor.bits() != 0 {
                let pair = value
                    .as_tagged(heap)
                    .get_as::<AccessorPair>()
                    .ok_or(VmError::Type)?;
                let pair = pair.as_ref();
                PartialDescriptor {
                    value: None,
                    get: Some(scope.handle(pair.get.get(heap))),
                    set: Some(scope.handle(pair.set.get(heap))),
                    writable: None,
                    enumerable: Some(enumerable),
                    configurable: Some(configurable),
                }
            } else {
                PartialDescriptor {
                    value: Some(scope.handle(value.as_tagged(heap))),
                    get: None,
                    set: None,
                    writable: Some(flags & bytecode::PropertyFlags::ReadOnly.bits() == 0),
                    enumerable: Some(enumerable),
                    configurable: Some(configurable),
                }
            };
            return match Proxy::define_internal(
                vm,
                heap,
                state,
                &scope,
                receiver,
                key.erase(),
                partial,
            )? {
                Flow::Threw => Ok(heap.known().exception.as_tagged(heap).erase()),
                Flow::Value(false) => Err(VmError::Type),
                Flow::Value(true) => Ok(receiver.as_tagged(heap)),
            };
        }
        if receiver.as_tagged(heap).as_heap_object().is_none() {
            return Err(VmError::Type);
        }
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
        let name = scope.handle(name);
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
        let obj = args.get(0).ok_or(VmError::Arity)?;
        let proto = args.get(1).ok_or(VmError::Arity)?;
        let obj_ref = scope
            .cast::<Object>(obj.as_tagged(heap))
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
    let v = args.get(0).ok_or(VmError::Arity)?;
    let v_tagged = v.as_tagged(heap);
    let ok = v_tagged.ptr_eq(heap.known().null.as_tagged(heap).erase())
        || v_tagged
            .as_heap_object()
            .is_some_and(|obj| obj.as_ref().header.map.get(heap).kind().is_constructor());
    if !ok {
        return Err(VmError::Type);
    }
    Ok(v_tagged)
}

/// superCtor.prototype validation: (value) -> value, TypeError unless the
/// value is an Object or null.
fn throw_if_not_object_or_null<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let v = args.get(0).ok_or(VmError::Arity)?;
    let v_tagged = v.as_tagged(heap);
    let ok = v_tagged.ptr_eq(heap.known().null.as_tagged(heap).erase())
        || !Convert::is_primitive(heap, v_tagged);
    if !ok {
        return Err(VmError::Type);
    }
    Ok(v_tagged)
}

/// [[ThisBindingStatus]] guard of derived constructors (ES 10.2.2):
/// (value) -> value, ReferenceError when `this` is still the hole.
fn throw_super_not_called_if_hole<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let v = args.get(0).ok_or(VmError::Arity)?;
    let v_tagged = v.as_tagged(heap);
    if v_tagged.ptr_eq(heap.known().the_hole.as_tagged(heap).erase()) {
        // "Must call super constructor before accessing 'this'"
        return Err(VmError::Reference);
    }
    Ok(v_tagged)
}

/// InitializeThisBinding guard (ES 10.2.2): (value) -> value,
/// ReferenceError unless `this` is still the hole (super() runs once).
fn throw_super_already_called_if_not_hole<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let v = args.get(0).ok_or(VmError::Arity)?;
    let v_tagged = v.as_tagged(heap);
    if !v_tagged.ptr_eq(heap.known().the_hole.as_tagged(heap).erase()) {
        // "Super constructor may only be called once"
        return Err(VmError::Reference);
    }
    Ok(v_tagged)
}

// ---- super() construction (ES 15.4.3) ---------------------------------------

/// The shared ConstructSuper tail: construct `callee` with `new_target`,
/// giving derived parents the hole receiver. The instance lands in the
/// return value; the exception sentinel escapes when user code threw.
fn construct_super_construct<'a>(
    nctx: RuntimeContext<'a>,
    callee: Handle<'_, Value>,
    new_target: Handle<'_, Value>,
    args: &[Handle<'_, Value>],
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let new_target_tagged = new_target.as_tagged(heap);
        if new_target_tagged.ptr_eq(heap.known().undefined.as_tagged(heap))
            || new_target_tagged.ptr_eq(heap.known().the_hole.as_tagged(heap))
        {
            // not inside a [[Construct]]: reachable via an arrow that escaped
            // the constructor
            return Err(VmError::Type);
        }
        let derived = callee
            .as_tagged(heap)
            .as_heap_object()
            .and_then(|obj| obj.as_ref().callable_info(heap))
            .is_some_and(|info| info.function_kind().is_derived_class_constructor());
        let Some(callee) = scope.cast::<Object>(callee.as_tagged(heap)) else {
            return Err(VmError::Type);
        };
        let Some(new_target) = scope.cast::<Object>(new_target.as_tagged(heap)) else {
            return Err(VmError::Type);
        };
        let (receiver, allocated) = if derived {
            (
                scope.handle(heap.known().the_hole.as_tagged(heap).erase()),
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
        let mut args_v = Vec::with_capacity(args.len() + 1);
        args_v.push(receiver);
        args_v.extend(args.iter().copied());
        let call_args = stage_handles(heap, &scope, &args_v);
        let result = scope.handle(RuntimeContext::call(
            vm,
            heap,
            state,
            callee.erase(),
            call_args,
            Some(new_target.erase()),
        )?);
        if result
            .as_tagged(heap)
            .ptr_eq(heap.known().exception.as_tagged(heap))
        {
            return Ok(result.as_tagged(heap));
        }
        if Convert::is_primitive(heap, result.as_tagged(heap)) {
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
    state.handle_scope(|scope| {
        let (callee, new_target) = frame_super_parts(heap, state)?;
        // keep the resolved parts rooted across the construct below
        let callee = scope.handle(callee);
        let new_target = scope.handle(new_target);
        let arg_words: Vec<Handle<'_, Value>> = args
            .iter()
            .map(|h| scope.handle(h.as_tagged(heap)))
            .collect();
        construct_super_construct(
            RuntimeContext::new(vm, heap, state),
            callee,
            new_target,
            &arg_words,
        )
    })
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
    state.handle_scope(|scope| {
        let (callee, new_target) = frame_super_parts(heap, state)?;
        let callee = scope.handle(callee);
        let new_target = scope.handle(new_target);
        if !state.cache.is_active() {
            return Err(VmError::Type);
        }
        let meta = state.cache.frame_meta();
        let argc = state.stack.argc(&meta).saturating_sub(1);
        let slice = state.stack.args(&meta, -2, argc);
        let arg_words: Vec<Handle<'_, Value>> = slice
            .iter()
            .map(|h| scope.handle(h.as_tagged(heap)))
            .collect();
        construct_super_construct(
            RuntimeContext::new(vm, heap, state),
            callee,
            new_target,
            &arg_words,
        )
    })
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
    state.handle_scope(|scope| {
        let closure = args.get(n - 2).ok_or(VmError::Arity)?;
        let new_target = args.get(n - 1).ok_or(VmError::Arity)?;
        let callee = {
            let Some(obj) = closure.as_tagged(heap).as_heap_object() else {
                return Err(VmError::Type);
            };
            let proto = obj.as_ref().header.map.get(heap).prototype.get(heap);
            let Some(proto_obj) = proto.as_heap_object() else {
                return Err(VmError::Type);
            };
            if !proto_obj
                .as_ref()
                .header
                .map
                .get(heap)
                .kind()
                .is_constructor()
            {
                return Err(VmError::Type);
            }
            scope.handle(proto)
        };
        let arg_words: Vec<Handle<'_, Value>> = (0..n - 2)
            .map(|i| args.get(i).ok_or(VmError::Arity))
            .collect::<Result<_, _>>()?;
        construct_super_construct(
            RuntimeContext::new(vm, heap, state),
            callee,
            new_target,
            &arg_words,
        )
    })
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
    let name = args.get(0).ok_or(VmError::Arity)?;
    state.handle_scope(|scope| {
        let found = dynamic_lookup_frame(heap, state, &scope, name)?;
        match found {
            Some(v) if !is_the_hole(heap, v.as_tagged(heap)) => Ok(v.as_tagged(heap)),
            Some(_) => Err(VmError::Reference),
            None => {
                // unresolved: fall back to a global object property
                let global = heap.known().global_object.erase();
                get_property_lenient(RuntimeContext::new(vm, heap, state), global, name)
            }
        }
    })
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
    let value = args.get(0).ok_or(VmError::Arity)?;
    let name = args.get(1).ok_or(VmError::Arity)?;
    state.handle_scope(|scope| {
        let found = dynamic_lookup_frame(heap, state, &scope, name)?;
        match found {
            Some(v) if !is_the_hole(heap, v.as_tagged(heap)) => {
                // write through to the found slot
                let context = frame_context_value(state, heap)?;
                let mut context = context.get_as::<Context>().ok_or(VmError::Type)?;
                let target = dynamic_slot(heap, &mut context, name.as_tagged(heap))?;
                let host = context.erase();
                target.set(heap, host, value.as_tagged(heap));
            }
            Some(_) => return Err(VmError::Reference),
            None => {
                let global = heap.known().global_object.erase();
                let threw = state.handle_scope(|scope| -> Result<bool, VmError> {
                    let outcome = global.store_lookup(
                        heap,
                        &scope,
                        name.as_tagged(heap).as_name(),
                        value.as_tagged(heap),
                        StoreSemantics::WriteThrough,
                    )?;
                    apply_store_outcome(vm, heap, state, global, outcome, value)
                })?;
                if threw {
                    return Ok(heap.known().exception.as_tagged(heap).erase());
                }
            }
        }
        Ok(value.as_tagged(heap))
    })
}

/// Fresh singleton-word compare against the hole sentinel.
fn is_the_hole(heap: &Heap, v: Tagged<'_, Value>) -> bool {
    v.ptr_eq(heap.known().the_hole.as_tagged(heap))
}

// ---- rest parameters ---------------------------------------------------------

/// A fresh array of the frame's arguments from formal index `first`:
/// (first) -> array.
fn create_rest_parameter<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, state, .. } = nctx;
    let first = args
        .get(0)
        .map(|h| h.as_tagged(heap))
        .and_then(|v| Smi::decode(v.raw()))
        .map(|s| s.value() as usize)
        .unwrap_or(0);
    state.handle_scope(|scope| {
        if !state.cache.is_active() {
            return Err(VmError::Type);
        }
        let meta = state.cache.frame_meta();
        let argc = state.stack.argc(&meta); // receiver included
        let count = argc.saturating_sub(1).saturating_sub(first);
        let values: Vec<Handle<'_, Value>> = (0..count)
            .map(|i| scope.handle(state.stack.reg(heap, &meta, -((first + i + 2) as i32))))
            .collect();
        let elements =
            heap.allocate_handle::<FixedArray>(stage_handles(heap, &scope, &values), &scope);
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
    let home = args.get(0).ok_or(VmError::Arity)?;
    let recv = args.get(1).ok_or(VmError::Arity)?;
    let raw_key = args.get(2).ok_or(VmError::Arity)?;
    if recv
        .as_tagged(heap)
        .ptr_eq(heap.known().the_hole.as_tagged(heap).erase())
    {
        // super.x before super() in a derived constructor
        return Err(VmError::Reference);
    }
    // the key coercion allocates: home/recv stay rooted in their
    // argument handles across it
    state.handle_scope(|scope| {
        let Some(key) = Object::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        let outcome = {
            let proto = Lookup::home_proto(heap, home.as_tagged(heap));
            let name = match Lookup::classify_key(heap, key.as_tagged(heap).erase())? {
                Key::Element(i) => Tagged::from(Smi::new(i as i64)),
                Key::Name(name) => name,
            };
            match Lookup::super_lookup_from_proto(heap, proto, name)? {
                LoadOutcome::Value(v) => (Some(scope.handle(v)), None),
                LoadOutcome::Getter(g) => (None, Some(scope.handle(g))),
            }
        };
        if let Some(v) = outcome.0 {
            return Ok(v.as_tagged(heap));
        }
        let getter = outcome.1.expect("one of the two arms is set");
        let skip = {
            let getter_tagged = getter.as_tagged(heap);
            getter_tagged.ptr_eq(heap.known().undefined.as_tagged(heap).erase())
                || !Object::is_callable(heap, getter_tagged)
        };
        if skip {
            return Ok(heap.known().undefined.as_tagged(heap).erase());
        }
        let call_args = {
            let r = recv.as_tagged(heap);
            scope.stage(&[r])
        };
        RuntimeContext::call(vm, heap, state, getter, call_args, None)
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
    let home = args.get(0).ok_or(VmError::Arity)?;
    let recv = args.get(1).ok_or(VmError::Arity)?;
    let raw_key = args.get(2).ok_or(VmError::Arity)?;
    let value = args.get(3).ok_or(VmError::Arity)?;
    let semantics_flag = args
        .get(4)
        .map(|h| h.as_tagged(heap))
        .and_then(|v| Smi::decode(v.raw()))
        .map(|s| s.value() as u32)
        .unwrap_or(0);
    if recv
        .as_tagged(heap)
        .ptr_eq(heap.known().the_hole.as_tagged(heap).erase())
    {
        return Err(VmError::Reference);
    }
    // the key coercion allocates: home/recv/value stay rooted in their
    // argument handles across it
    state.handle_scope(|scope| {
        let Some(key) = Object::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        let semantics = if semantics_flag & bytecode::SUPER_STORE_WRITE_THROUGH != 0 {
            StoreSemantics::WriteThrough
        } else {
            StoreSemantics::Shadow
        };
        let outcome = {
            let proto = Lookup::home_proto(heap, home.as_tagged(heap));
            let name = match Lookup::classify_key(heap, key.as_tagged(heap).erase())? {
                Key::Element(i) => Tagged::from(Smi::new(i as i64)),
                Key::Name(name) => name,
            };
            Transition::super_store_lookup(
                heap,
                &scope,
                proto,
                recv.as_tagged(heap),
                name,
                value.as_tagged(heap),
                semantics,
            )
        }?;
        if apply_store_outcome(vm, heap, state, recv, outcome, value)? {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        }
        Ok(value.as_tagged(heap))
    })
}
