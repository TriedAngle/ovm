//! ES 20.1: the Object constructor, statics, and prototype methods.
use crate::Key;
use crate::Lookup;
use crate::RuntimeContext;
use crate::proxy::Flow;
use crate::proxy::Proxy;
use crate::runtime::Coercion;

use crate::{
    ContextState, Convert, Handle, HandleScope, HandleSlice, Heap, Object, PropertyDescriptor,
    SlotName, Smi, Tagged, VM, Value, VmError,
};

/// Stub: `Object.prototype.toString` returns "[object Object]".
pub fn object_to_string<'a>(
    nctx: RuntimeContext<'a>,
    _args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let s = vm.interner().intern_str(heap, &scope, "[object Object]");
        Ok(s.as_tagged(heap).erase())
    })
}

/// `Object(x)`: returns objects unchanged (boxing of primitives is not
/// implemented yet); `new Object()`: the interpreter prepends the fresh
/// receiver, so [[Construct]] just returns it.
pub fn object_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    if nctx.is_construct() {
        let RuntimeContext { heap, .. } = nctx;
        return args.get(0).map(|h| h.as_tagged(heap)).ok_or(VmError::Arity);
    }
    let RuntimeContext { heap, .. } = nctx;
    let arg = args
        .get(1)
        .map(|h| h.as_tagged(heap))
        .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
    let cond_37 = Convert::is_primitive(heap, arg);
    if cond_37 {
        // TODO: box primitives (String/Symbol wrappers)
        return Err(VmError::Type);
    }
    Ok(arg)
}

/// `Object.getPrototypeOf(o)`: the receiver's map prototype. Primitive
/// arguments are a TypeError until ToObject boxing exists (ES5 behavior;
/// ES2015+ boxes them).
pub fn object_get_prototype_of<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let arg = args
        .get(1)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?;
    let Some(obj) = arg.as_heap_object() else {
        return Err(VmError::Type);
    };
    Ok(obj.as_ref().header.map.get(heap).prototype.get(heap))
}

/// `Object.create(O [, Properties])` (ES 20.1.2.2): a fresh extensible
/// ordinary object with `O` as its [[Prototype]] and no own properties.
/// The `Properties` argument is accepted only as `undefined` (property
/// descriptors are not implemented for it yet).
pub fn object_create<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, state, .. } = nctx;
    state.handle_scope(|scope| {
        let proto = scope.handle(
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        );
        // If Type(O) is neither Object nor Null, throw a TypeError
        let null = heap.known().null.as_tagged(heap).raw();
        let proto_ok = proto.as_tagged(heap).raw() == null
            || !Convert::is_primitive(heap, proto.as_tagged(heap));
        if !proto_ok {
            return Err(VmError::Type);
        }
        if let Some(props) = args.get(2) {
            let undefined = heap.known().undefined.as_tagged(heap).raw();
            if props.as_tagged(heap).raw() != undefined {
                return Err(VmError::Type);
            }
        }
        let map = heap.known().object_initial_map;
        let empty: [Tagged<'_, Value>; 0] = [];
        let obj = scope.handle(heap.new_object(&scope, map, scope.stage(&empty)));
        let obj = scope
            .cast::<Object>(obj.as_tagged(heap).erase())
            .expect("fresh object");
        Object::set_prototype(heap, &scope, obj, proto)?;
        Ok(obj.as_tagged(heap).erase())
    })
}

/// `Object.setPrototypeOf(O, proto)` (ES 20.1.2.20): primitives return O
/// unchanged (after RequireObjectCoercible); proto must be an object or
/// null; the underlying [[SetPrototypeOf]] may reject (non-extensible
/// receiver, prototype cycles) with a TypeError.
/// Own enumerable-property keys in specification order: integer indices
/// ascending, then string keys in insertion order (ES 8.6.2, the
/// descriptors array is insertion-ordered).
pub fn own_property_keys(heap: &Heap, target: Tagged<'_, Value>) -> Vec<Value> {
    let mut keys = Vec::new();
    let Some(obj) = target.as_heap_object() else {
        return keys;
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
    for d in obj.as_ref().header.map.get(heap).descriptors() {
        keys.push(d.name(heap).raw());
    }
    keys
}

/// `Object.prototype.hasOwnProperty(key)` (ES 20.4.3.2, own properties
/// only).
pub fn object_has_own_property<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let receiver = scope.handle(
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        );
        let raw_key = scope.handle(
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        );
        // the key coercion allocates (float/wrapper keys intern or run
        // user code): the receiver must stay rooted across it
        let Some(key) = Object::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        let has = 'has: {
            let key = key.as_tagged(heap);
            if let Key::Element(i) =
                Lookup::classify_key(heap, key.erase()).unwrap_or(Key::Name(key))
                && let Some(obj) = receiver.as_tagged(heap).as_heap_object()
                && obj.as_ref().element_value(heap, i).is_some()
            {
                break 'has true;
            }
            match receiver.lookup(heap, key) {
                Lookup::NotFound => false,
                // the array `length` internal slot counts as an own property
                _ => true,
            }
        };
        Ok(Convert::boolean(heap, has))
    })
}

/// `Object.prototype.propertyIsEnumerable(key)` (ES 20.4.3.5).
pub fn object_property_is_enumerable<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let receiver = scope.handle(
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        );
        let raw_key = scope.handle(
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        );
        // the key coercion allocates (float/wrapper keys intern or run
        // user code): the receiver must stay rooted across it
        let Some(key) = Object::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        let enumerable = 'enumerable: {
            let key = key.as_tagged(heap);
            if let Key::Element(i) =
                Lookup::classify_key(heap, key.erase()).unwrap_or(Key::Name(key))
                && let Some(obj) = receiver.as_tagged(heap).as_heap_object()
                && obj.as_ref().element_value(heap, i).is_some()
            {
                break 'enumerable true; // array elements are enumerable
            }
            match receiver.lookup(heap, key) {
                Lookup::Data { flags, .. } => flags.is_enumerable(),
                Lookup::Accessor {
                    holder, map_index, ..
                } => holder
                    .as_ref()
                    .header
                    .map
                    .get(heap)
                    .descriptor(map_index)
                    .flags()
                    .is_enumerable(),
                Lookup::NotFound => false,
            }
        };
        Ok(Convert::boolean(heap, enumerable))
    })
}

/// `Object.getOwnPropertyNames(O)` (ES 20.1.2.7).
pub fn object_get_own_property_names<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, state, .. } = nctx;
    let target = args
        .get(1)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?;
    let names: Vec<Value> = {
        let mut keys = own_property_keys(heap, target);
        // arrays also list "length" (and it sorts with the strings)
        if let Some(obj) = target.as_heap_object()
            && obj.as_ref().is_array(heap)
        {
            keys.push(heap.known().strings.length.as_tagged(heap).raw());
        }
        keys
    };
    state.handle_scope(|scope| {
        let staged = scope.stage(
            &names
                .iter()
                .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                .collect::<Vec<_>>(),
        );
        Ok(heap.new_array(&scope, staged).erase())
    })
}

/// Build a plain `{ key: value, ... }` object from static field names.
pub fn plain_object<'a>(
    vm: &'a VM,
    heap: &'a mut Heap,
    state: &'a ContextState,
    fields: &[(&'static str, Handle<'_, Value>)],
) -> Result<Tagged<'a, Value>, VmError> {
    state.handle_scope(|scope| {
        let map = heap.known().object_initial_map;
        let obj = heap
            .new_object(&scope, map, HandleSlice::EMPTY)
            .as_handle(&scope);
        for (name, value) in fields {
            let name = vm.interner().intern_str(heap, &scope, name);
            let name = scope.handle(name.as_tagged(heap));
            Object::define_own_property(heap, &scope, obj, name, PropertyDescriptor::data(*value))?;
        }
        Ok(obj.as_tagged(heap).erase())
    })
}

/// `Object.getOwnPropertyDescriptor(O, P)` (ES 20.1.2.5): the shared
/// raw descriptor reader (`Lookup::ordinary_own_descriptor`) converted
/// to a descriptor object via FromPropertyDescriptor semantics.
pub fn object_get_own_property_descriptor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        // the key coercion allocates (Float keys intern a string): the
        // target must stay rooted across it
        let target = scope.handle(
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        );
        let raw_key = scope.handle(
            args.get(2)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        );
        let Some(key) = Object::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        let desc = Lookup::ordinary_own_descriptor(
            heap,
            &scope,
            target.as_tagged(heap),
            key.as_tagged(heap).erase(),
        );
        // root the oddball singletons once for the descriptor fields
        let true_v = scope.handle(heap.known().true_object.as_tagged(heap).erase());
        let false_v = scope.handle(heap.known().false_object.as_tagged(heap).erase());
        let bool_ = |b| if b { true_v } else { false_v };
        match desc {
            Some(PropertyDescriptor::Data {
                value,
                writable,
                enumerable,
                configurable,
            }) => plain_object(
                vm,
                heap,
                state,
                &[
                    ("value", value),
                    ("writable", bool_(writable)),
                    ("enumerable", bool_(enumerable)),
                    ("configurable", bool_(configurable)),
                ],
            ),
            Some(PropertyDescriptor::Accessor {
                get,
                set,
                enumerable,
                configurable,
            }) => plain_object(
                vm,
                heap,
                state,
                &[
                    ("get", get),
                    ("set", set),
                    ("enumerable", bool_(enumerable)),
                    ("configurable", bool_(configurable)),
                ],
            ),
            None => Ok(heap.known().undefined.as_tagged(heap).erase()),
        }
    })
}

/// `Object.defineProperty(O, P, Attributes)` (ES 20.1.2.4):
/// ToPropertyDescriptor + [[DefineOwnProperty]] (through the
/// `defineProperty` trap for proxy receivers, ES 20.2.5.6).
pub fn object_define_property<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        // root the target, key, and descriptor: the key coercion and the
        // descriptor conversion below allocate (user getters run), which
        // would leave raw copies stale
        let target = scope.handle(
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        );
        let attrs = scope.handle(
            args.get(3)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        );
        let raw_key = scope.handle(
            args.get(2)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        );
        let Some(key) = Object::to_property_key(vm, heap, state, raw_key)? else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        // shared ToPropertyDescriptor; proxies and ordinary targets both
        // complete/validate inside define_internal
        let partial = match Lookup::to_property_descriptor(vm, heap, state, &scope, attrs)? {
            Some(partial) => partial,
            None => return Ok(heap.known().exception.as_tagged(heap).erase()),
        };
        match Proxy::define_internal(vm, heap, state, &scope, target, key.erase(), partial)? {
            Flow::Threw => Ok(heap.known().exception.as_tagged(heap).erase()),
            Flow::Value(false) => Err(VmError::Type),
            Flow::Value(true) => Ok(target.as_tagged(heap).erase()),
        }
    })
}

pub fn object_set_prototype_of<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, state, .. } = nctx;
    state.handle_scope(|scope| {
        let target = scope.handle(
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        );
        let proto = scope.handle(
            args.get(2)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        );
        let (nullish, target_is_object, proto_ok) = {
            let null = heap.known().null.as_tagged(heap).raw();
            let undefined = heap.known().undefined.as_tagged(heap).raw();
            (
                target.as_tagged(heap).raw() == null || target.as_tagged(heap).raw() == undefined,
                !Convert::is_primitive(heap, target.as_tagged(heap)),
                proto.as_tagged(heap).raw() == null
                    || !Convert::is_primitive(heap, proto.as_tagged(heap)),
            )
        };
        // RequireObjectCoercible(O)
        if nullish {
            return Err(VmError::Type);
        }
        // primitives are returned unchanged
        if !target_is_object {
            return Ok(target.as_tagged(heap).erase());
        }
        if !proto_ok {
            return Err(VmError::Type);
        }
        let target_obj = scope
            .cast::<Object>(target.as_tagged(heap))
            .expect("target checked to be an object");
        Object::set_prototype(heap, &scope, target_obj, proto)?;
        Ok(target.as_tagged(heap).erase())
    })
}

/// `Object.preventExtensions(O)` (ES 20.1.2.16): through the
/// `preventExtensions` trap for proxies (ES 20.2.5.3).
pub fn object_prevent_extensions<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let target = scope.handle(
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?,
        );
        let nullish = {
            let null = heap.known().null.as_tagged(heap).raw();
            let undefined = heap.known().undefined.as_tagged(heap).raw();
            target.as_tagged(heap).raw() == null || target.as_tagged(heap).raw() == undefined
        };
        if nullish {
            return Err(VmError::Type);
        }
        // the (possibly proxy) receiver is returned after traps ran user
        // code: keep it rooted across the call
        if !Proxy::is_js_receiver(heap, target.as_tagged(heap)) {
            return Ok(target.as_tagged(heap).erase()); // primitives returned unchanged
        }
        let raw = match Proxy::prevent_extensions(
            vm,
            heap,
            state,
            // Safety: fresh rooted-slot word, consumed by the call.
            unsafe { Tagged::<Value>::from_value_unchecked(target.raw()) },
        )? {
            Coercion::Threw => None,
            Coercion::Value(v) => Some(v.raw()),
        };
        let Some(raw) = raw else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        let v = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw) });
        let cond_39 = Convert::is_truthy(heap, v.as_tagged(heap));
        if !cond_39 {
            Err(VmError::Message("object is not extensible"))
        } else {
            // re-read through the handle: the trap above ran user
            // code and may have moved the receiver
            Ok(target.as_tagged(heap).erase())
        }
    })
}

/// `Object.isExtensible(O)` (ES 20.1.2.14): primitives are `false`;
/// proxies run the `isExtensible` trap with its must-match invariant.
pub fn object_is_extensible<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let target = args
            .get(1)
            .map(|h| h.as_tagged(heap).raw())
            .ok_or(VmError::Arity)?;
        if !Proxy::is_js_receiver(heap, unsafe { target.assume_valid(heap) }) {
            return Ok(Convert::boolean(heap, false));
        }
        let raw = match Proxy::is_extensible(
            vm,
            heap,
            state,
            // Safety: fresh argument word, consumed by the call.
            unsafe { Tagged::<Value>::from_value_unchecked(target) },
        )? {
            Coercion::Threw => None,
            Coercion::Value(v) => Some(v.raw()),
        };
        let Some(raw) = raw else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        let v = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw) });
        Ok(v.as_tagged(heap).erase())
    })
}

/// SetIntegrityLevel (ES 7.3.15/16) for ordinary objects: clone the map
/// with `configurable` (and for freeze `writable`) cleared on every
/// descriptor and EXTENDABLE dropped. Dense array elements keep their
/// intrinsic attributes (TODO: element sealing with the elements
/// machinery).
pub fn set_integrity_flags(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    obj: Handle<'_, Object>,
    freeze: bool,
) {
    use crate::{Map, MapInit, MapKind, SlotFlags};
    let (kind, descriptor_count, already) = {
        let map = obj.as_tagged(heap).map_ref(heap);
        (
            map.kind(),
            map.descriptors().len(),
            !map.kind().is_extendable()
                && map.descriptors().iter().all(|d| {
                    let flags = d.flags();
                    !flags.is_configurable()
                        && (!freeze || flags.is_accessor() || !flags.is_writable())
                }),
        )
    };
    if already {
        return;
    }
    // clone the map with `configurable` (and for freeze `writable`)
    // cleared on every descriptor and EXTENDABLE dropped: names are
    // read under the enter-heap anchor so the descriptor rows stay
    // anchored across the (allocating) map build
    heap.allocate_token_enter_heap(Map::layout_for(descriptor_count), |token, heap| {
        let obj_ref = obj.as_tagged(heap);
        let map = obj_ref.map_ref(heap);
        // Safety: fresh map-slot word, rooted below before the allocation.
        let prototype = scope.handle(map.prototype.get(heap));
        let descriptors: Vec<(Handle<'_, SlotName>, SlotFlags, Handle<'_, Value>)> = map
            .descriptors()
            .iter()
            .map(|d| {
                let mut flags = d.flags();
                flags = SlotFlags::new(flags.bits() & !SlotFlags::CONFIGURABLE.bits());
                if freeze && !flags.is_accessor() {
                    flags = SlotFlags::new(flags.bits() & !SlotFlags::WRITABLE.bits());
                }
                (
                    scope.handle(d.name(heap)),
                    flags,
                    scope.handle(d.value.get(heap)),
                )
            })
            .collect();
        let new_map = token.allocate::<Map>(MapInit {
            kind: MapKind::new(kind.bits() & !MapKind::EXTENDABLE.bits()),
            value_slot_count: map.value_slot_count(),
            descriptors: &descriptors,
            prototype,
        });
        obj_ref
            .header
            .map
            .set(heap, obj.as_tagged(heap).erase(), new_map);
    });
}

/// `Object.seal(O)` (ES 20.1.2.17).
pub fn object_seal<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let target = args
        .get(1)
        .map(|h| h.as_tagged(heap).raw())
        .ok_or(VmError::Arity)?;
    let nullish = {
        let null = heap.known().null.as_tagged(heap).raw();
        let undefined = heap.known().undefined.as_tagged(heap).raw();
        target == null || target == undefined
    };
    if nullish {
        return Err(VmError::Type);
    }
    // traps run user code before the receiver is returned: keep it rooted
    state.handle_scope(|scope| {
        // Safety: fresh argument word, rooted before any allocation.
        let target_handle = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(target) });
        if !Proxy::is_js_receiver(heap, target_handle.as_tagged(heap)) {
            return Ok(target_handle.as_tagged(heap).erase());
        }
        // [[PreventExtensions]] first (traps included)
        let raw = match Proxy::prevent_extensions(
            vm,
            heap,
            state,
            // Safety: fresh rooted-slot word, consumed by the call.
            unsafe { Tagged::<Value>::from_value_unchecked(target_handle.raw()) },
        )? {
            Coercion::Threw => None,
            Coercion::Value(v) => Some(v.raw()),
        };
        let Some(raw) = raw else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        let v = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw) });
        let cond_42 = Convert::is_truthy(heap, v.as_tagged(heap));
        if !cond_42 {
            return Err(VmError::Message("object is not extensible"));
        }
        // TODO: per-key [[DefineOwnProperty]] through the defineProperty
        // trap once ownKeys lands (proxy targets); ordinary targets:
        // re-read through the handle: the trap may have moved the receiver
        if Proxy::is_proxy(heap, target_handle.as_tagged(heap)) {
            return Ok(target_handle.as_tagged(heap).erase());
        }
        let obj = scope
            .cast::<Object>(target_handle.as_tagged(heap))
            .expect("checked above");
        set_integrity_flags(heap, &scope, obj, false);
        // re-read through the handle: traps may have moved the receiver
        Ok(target_handle.as_tagged(heap).erase())
    })
}

/// `Object.freeze(O)` (ES 20.1.2.9).
pub fn object_freeze<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let target = args
        .get(1)
        .map(|h| h.as_tagged(heap).raw())
        .ok_or(VmError::Arity)?;
    let nullish = {
        let null = heap.known().null.as_tagged(heap).raw();
        let undefined = heap.known().undefined.as_tagged(heap).raw();
        target == null || target == undefined
    };
    if nullish {
        return Err(VmError::Type);
    }
    // traps run user code before the receiver is returned: keep it rooted
    state.handle_scope(|scope| {
        // Safety: fresh argument word, rooted before any allocation.
        let target_handle = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(target) });
        if !Proxy::is_js_receiver(heap, target_handle.as_tagged(heap)) {
            return Ok(target_handle.as_tagged(heap).erase());
        }
        let raw = match Proxy::prevent_extensions(
            vm,
            heap,
            state,
            // Safety: fresh rooted-slot word, consumed by the call.
            unsafe { Tagged::<Value>::from_value_unchecked(target_handle.raw()) },
        )? {
            Coercion::Threw => None,
            Coercion::Value(v) => Some(v.raw()),
        };
        let Some(raw) = raw else {
            return Ok(heap.known().exception.as_tagged(heap).erase());
        };
        let v = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw) });
        let cond_45 = Convert::is_truthy(heap, v.as_tagged(heap));
        if !cond_45 {
            return Err(VmError::Message("object is not extensible"));
        }
        // re-read through the handle: the trap may have moved the receiver
        if Proxy::is_proxy(heap, target_handle.as_tagged(heap)) {
            return Ok(target_handle.as_tagged(heap).erase());
        }
        let obj = scope
            .cast::<Object>(target_handle.as_tagged(heap))
            .expect("checked above");
        set_integrity_flags(heap, &scope, obj, true);
        // re-read through the handle: traps may have moved the receiver
        Ok(target_handle.as_tagged(heap).erase())
    })
}
