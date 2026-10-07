//! ES 20.1: the Object constructor, statics, and prototype methods.
use vm_core::Coercion;
use vm_core::HostCtx;
use vm_core::Key;
use vm_core::Lookup;
use vm_core::proxy::Flow;
use vm_core::proxy::Proxy;

use vm_core::{
    Args, Convert, Handle, HandleScope, HandleSlice, Heap, Object, PropertyDescriptor, SlotName,
    Smi, Tagged, ThreadState, VM, Value, VmError,
};
use vm_core::{raise_runtime, rt_try};

/// Stub: `Object.prototype.toString` returns "[object Object]".
pub fn object_to_string<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    _args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let s = vm.interner().intern_str(heap, &scope, "[object Object]");
        s.as_tagged(heap).erase()
    })
}

/// `Object(x)`: returns objects unchanged (boxing of primitives is not
/// implemented yet); `new Object()`: the interpreter prepends the fresh
/// receiver, so [[Construct]] just returns it.
pub fn object_constructor<'a>(
    nctx: HostCtx<'a>,
    new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let is_construct = new_target.is_some();
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    let arg = args.get_handle(heap, 1);
    if is_construct {
        if !Convert::is_primitive(heap, arg.as_tagged(heap)) {
            // `new Object(obj)` returns it unchanged (ES 20.1.1.1 step 3)
            return arg.as_tagged(heap);
        }
        // `new Object()` / primitive: allocate a fresh object
        return state.handle_scope(|scope| {
            heap.new_object(&scope, heap.known().object_initial_map, HandleSlice::EMPTY)
                .erase()
        });
    }
    if Convert::is_primitive(heap, arg.as_tagged(heap)) {
        // TODO: box primitives (String/Symbol wrappers)
        return raise_runtime(vm, heap, state, VmError::Type);
    }
    arg.as_tagged(heap)
}

/// `Object.getPrototypeOf(o)`: the receiver's map prototype. Primitive
/// arguments are a TypeError until ToObject boxing exists (ES5 behavior;
/// ES2015+ boxes them).
pub fn object_get_prototype_of<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    let arg = args.get(heap, 1);
    let arg = arg;
    let Some(obj) = arg.as_heap_object() else {
        return raise_runtime(vm, heap, state, VmError::Type);
    };
    obj.as_ref().header.map.get(heap).prototype.get(heap)
}

/// `Object.create(O [, Properties])` (ES 20.1.2.2): a fresh extensible
/// ordinary object with `O` as its [[Prototype]] and no own properties.
/// The `Properties` argument is accepted only as `undefined` (property
/// descriptors are not implemented for it yet).
pub fn object_create<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let proto_arg = args.get(heap, 1);
        let proto = scope.handle(proto_arg);
        // If Type(O) is neither Object nor Null, throw a TypeError
        let null = heap.known().null.as_tagged(heap).raw();
        let proto_ok = proto.as_tagged(heap).raw() == null
            || !Convert::is_primitive(heap, proto.as_tagged(heap));
        if !proto_ok {
            return raise_runtime(vm, heap, state, VmError::Type);
        }
        if args.len() > 2 {
            let props = args.get(heap, 2);
            let undefined = heap.known().undefined.as_tagged(heap).raw();
            if props.raw() != undefined {
                return raise_runtime(vm, heap, state, VmError::Type);
            }
        }
        let map = heap.known().object_initial_map;
        let empty: [Tagged<'_, Value>; 0] = [];
        let obj = scope.handle(heap.new_object(&scope, map, scope.stage(&empty)));
        let obj = scope
            .cast::<Object>(heap, obj.as_tagged(heap).erase())
            .expect("fresh object");
        rt_try!(
            vm,
            heap,
            state,
            Object::set_prototype(heap, &scope, obj, proto)
        );
        obj.as_tagged(heap).erase()
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
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let receiver = args.get(heap, 0);
        let receiver = scope.handle(receiver);
        let raw_key = args.get(heap, 1);
        let raw_key = scope.handle(raw_key);
        // the key coercion allocates (float/wrapper keys intern or run
        // user code): the receiver must stay rooted across it
        let Some(key) = rt_try!(
            vm,
            heap,
            state,
            Object::to_property_key(vm, heap, state, raw_key)
        ) else {
            return heap.known().exception.as_tagged(heap).erase();
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
        Convert::boolean(heap, has)
    })
}

/// `Object.prototype.propertyIsEnumerable(key)` (ES 20.4.3.5).
pub fn object_property_is_enumerable<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let receiver = args.get(heap, 0);
        let receiver = scope.handle(receiver);
        let raw_key = args.get(heap, 1);
        let raw_key = scope.handle(raw_key);
        // the key coercion allocates (float/wrapper keys intern or run
        // user code): the receiver must stay rooted across it
        let Some(key) = rt_try!(
            vm,
            heap,
            state,
            Object::to_property_key(vm, heap, state, raw_key)
        ) else {
            return heap.known().exception.as_tagged(heap).erase();
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
        Convert::boolean(heap, enumerable)
    })
}

/// `Object.getOwnPropertyNames(O)` (ES 20.1.2.7).
pub fn object_get_own_property_names<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm: _, heap, state, ..
    } = nctx;
    let target = args.get(heap, 1);
    let target = target;
    // array `length` is a real (accessor) descriptor on the array map, so
    // `own_property_keys` already lists it
    let names: Vec<Value> = own_property_keys(heap, target);
    state.handle_scope(|scope| {
        let staged = scope.stage(
            &names
                .iter()
                .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                .collect::<Vec<_>>(),
        );
        heap.new_array(&scope, staged).erase()
    })
}

/// Build a plain `{ key: value, ... }` object from static field names.
pub fn plain_object<'a>(
    vm: &'a VM,
    heap: &'a mut Heap,
    state: &'a ThreadState,
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
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        // the key coercion allocates (Float keys intern a string): the
        // target must stay rooted across it
        let target = args.get(heap, 1);
        let target = scope.handle(target);
        let raw_key = args.get(heap, 2);
        let raw_key = scope.handle(raw_key);
        let Some(key) = rt_try!(
            vm,
            heap,
            state,
            Object::to_property_key(vm, heap, state, raw_key)
        ) else {
            return heap.known().exception.as_tagged(heap).erase();
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
            }) => match plain_object(
                vm,
                heap,
                state,
                &[
                    ("value", value),
                    ("writable", bool_(writable)),
                    ("enumerable", bool_(enumerable)),
                    ("configurable", bool_(configurable)),
                ],
            )
            .map(|v| v.raw())
            {
                Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
                Err(err) => return raise_runtime(vm, heap, state, err),
            },
            Some(PropertyDescriptor::Accessor {
                get,
                set,
                enumerable,
                configurable,
            }) => match plain_object(
                vm,
                heap,
                state,
                &[
                    ("get", get),
                    ("set", set),
                    ("enumerable", bool_(enumerable)),
                    ("configurable", bool_(configurable)),
                ],
            )
            .map(|v| v.raw())
            {
                Ok(v) => unsafe { Tagged::<Value>::from_value_unchecked(v) },
                Err(err) => return raise_runtime(vm, heap, state, err),
            },
            None => heap.known().undefined.as_tagged(heap).erase(),
        }
    })
}

/// `Object.defineProperty(O, P, Attributes)` (ES 20.1.2.4):
/// ToPropertyDescriptor + [[DefineOwnProperty]] (through the
/// `defineProperty` trap for proxy receivers, ES 20.2.5.6).
pub fn object_define_property<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        // root the target, key, and descriptor: the key coercion and the
        // descriptor conversion below allocate (user getters run), which
        // would leave raw copies stale
        let target = args.get(heap, 1);
        let target = scope.handle(target);
        let attrs = args.get(heap, 3);
        let attrs = scope.handle(attrs);
        let raw_key = args.get(heap, 2);
        let raw_key = scope.handle(raw_key);
        let Some(key) = rt_try!(
            vm,
            heap,
            state,
            Object::to_property_key(vm, heap, state, raw_key)
        ) else {
            return heap.known().exception.as_tagged(heap).erase();
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        // shared ToPropertyDescriptor; proxies and ordinary targets both
        // complete/validate inside define_internal
        let partial = match rt_try!(
            vm,
            heap,
            state,
            Lookup::to_property_descriptor(vm, heap, state, &scope, attrs)
        ) {
            Some(partial) => partial,
            None => return heap.known().exception.as_tagged(heap).erase(),
        };
        match rt_try!(
            vm,
            heap,
            state,
            Proxy::define_internal(vm, heap, state, &scope, target, key.erase(), partial)
        ) {
            Flow::Threw => heap.known().exception.as_tagged(heap).erase(),
            Flow::Value(false) => raise_runtime(vm, heap, state, VmError::Type),
            Flow::Value(true) => target.as_tagged(heap).erase(),
        }
    })
}

pub fn object_set_prototype_of<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let target = args.get(heap, 1);
        let target = scope.handle(target);
        let proto = args.get(heap, 2);
        let proto = scope.handle(proto);
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
            return raise_runtime(vm, heap, state, VmError::Type);
        }
        // primitives are returned unchanged
        if !target_is_object {
            return target.as_tagged(heap).erase();
        }
        if !proto_ok {
            return raise_runtime(vm, heap, state, VmError::Type);
        }
        let target_obj = scope
            .cast::<Object>(heap, target.as_tagged(heap))
            .expect("target checked to be an object");
        rt_try!(
            vm,
            heap,
            state,
            Object::set_prototype(heap, &scope, target_obj, proto)
        );
        target.as_tagged(heap).erase()
    })
}

/// `Object.preventExtensions(O)` (ES 20.1.2.16): through the
/// `preventExtensions` trap for proxies (ES 20.2.5.3).
pub fn object_prevent_extensions<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let target = args.get(heap, 1);
        let target = scope.handle(target);
        let nullish = {
            let null = heap.known().null.as_tagged(heap).raw();
            let undefined = heap.known().undefined.as_tagged(heap).raw();
            target.as_tagged(heap).raw() == null || target.as_tagged(heap).raw() == undefined
        };
        if nullish {
            return raise_runtime(vm, heap, state, VmError::Type);
        }
        // the (possibly proxy) receiver is returned after traps ran user
        // code: keep it rooted across the call
        if !Proxy::is_js_receiver(heap, target.as_tagged(heap)) {
            return target.as_tagged(heap).erase(); // primitives returned unchanged
        }
        let raw = match rt_try!(
            vm,
            heap,
            state,
            Proxy::prevent_extensions(vm, heap, state, target)
        ) {
            Coercion::Threw => None,
            Coercion::Value(v) => Some(v.raw()),
        };
        let Some(raw) = raw else {
            return heap.known().exception.as_tagged(heap).erase();
        };
        let v = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw) });
        let cond_39 = Convert::is_truthy(heap, v.as_tagged(heap));
        if !cond_39 {
            raise_runtime(vm, heap, state, VmError::NotExtensible)
        } else {
            // re-read through the handle: the trap above ran user
            // code and may have moved the receiver
            target.as_tagged(heap).erase()
        }
    })
}

/// `Object.isExtensible(O)` (ES 20.1.2.14): primitives are `false`;
/// proxies run the `isExtensible` trap with its must-match invariant.
pub fn object_is_extensible<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let target = args.get_handle(heap, 1);
        if !Proxy::is_js_receiver(heap, target.as_tagged(heap)) {
            return Convert::boolean(heap, false);
        }
        // Safety: rooted argument word, consumed by the call.
        let target = unsafe { Tagged::<Value>::from_value_unchecked(target.raw()) };
        let extensible = match rt_try!(
            vm,
            heap,
            state,
            Proxy::is_extensible(vm, heap, state, target)
        ) {
            Coercion::Threw => None,
            Coercion::Value(v) => Some(scope.handle(v)),
        };
        let Some(v) = extensible else {
            return heap.known().exception.as_tagged(heap).erase();
        };
        v.as_tagged(heap).erase()
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
    use vm_core::{Map, MapInit, MapKind, SlotFlags};
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
        vm_core::Prototype::shape_changed(heap, obj_ref.map_ref(heap));
        obj_ref
            .header
            .map
            .set(heap, obj.as_tagged(heap).erase(), new_map);
    });
}

/// `Object.seal(O)` (ES 20.1.2.17).
pub fn object_seal<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    let target = args.get_handle(heap, 1);
    let nullish = {
        let target = target.as_tagged(heap);
        let null = heap.known().null.as_tagged(heap).raw();
        let undefined = heap.known().undefined.as_tagged(heap).raw();
        target.raw() == null || target.raw() == undefined
    };
    if nullish {
        return raise_runtime(vm, heap, state, VmError::Type);
    }
    // traps run user code before the receiver is returned: keep it rooted
    state.handle_scope(|scope| {
        // Safety: fresh argument word, rooted before any allocation.
        let target_handle = target;
        if !Proxy::is_js_receiver(heap, target_handle.as_tagged(heap)) {
            return target_handle.as_tagged(heap).erase();
        }
        // [[PreventExtensions]] first (traps included)
        let raw = match rt_try!(
            vm,
            heap,
            state,
            Proxy::prevent_extensions(vm, heap, state, target_handle)
        ) {
            Coercion::Threw => None,
            Coercion::Value(v) => Some(v.raw()),
        };
        let Some(raw) = raw else {
            return heap.known().exception.as_tagged(heap).erase();
        };
        let v = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw) });
        let cond_42 = Convert::is_truthy(heap, v.as_tagged(heap));
        if !cond_42 {
            return raise_runtime(vm, heap, state, VmError::NotExtensible);
        }
        // TODO: per-key [[DefineOwnProperty]] through the defineProperty
        // trap once ownKeys lands (proxy targets); ordinary targets:
        // re-read through the handle: the trap may have moved the receiver
        if Proxy::is_proxy(heap, target_handle.as_tagged(heap)) {
            return target_handle.as_tagged(heap).erase();
        }
        let obj = scope
            .cast::<Object>(heap, target_handle.as_tagged(heap))
            .expect("checked above");
        set_integrity_flags(heap, &scope, obj, false);
        // re-read through the handle: traps may have moved the receiver
        target_handle.as_tagged(heap).erase()
    })
}

/// `Object.freeze(O)` (ES 20.1.2.9).
pub fn object_freeze<'a>(
    nctx: HostCtx<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: Args,
) -> Tagged<'a, Value> {
    let HostCtx {
        vm, heap, state, ..
    } = nctx;
    let target = args.get_handle(heap, 1);
    let nullish = {
        let target = target.as_tagged(heap);
        let null = heap.known().null.as_tagged(heap).raw();
        let undefined = heap.known().undefined.as_tagged(heap).raw();
        target.raw() == null || target.raw() == undefined
    };
    if nullish {
        return raise_runtime(vm, heap, state, VmError::Type);
    }
    // traps run user code before the receiver is returned: keep it rooted
    state.handle_scope(|scope| {
        // Safety: fresh argument word, rooted before any allocation.
        let target_handle = target;
        if !Proxy::is_js_receiver(heap, target_handle.as_tagged(heap)) {
            return target_handle.as_tagged(heap).erase();
        }
        let raw = match rt_try!(
            vm,
            heap,
            state,
            Proxy::prevent_extensions(vm, heap, state, target_handle)
        ) {
            Coercion::Threw => None,
            Coercion::Value(v) => Some(v.raw()),
        };
        let Some(raw) = raw else {
            return heap.known().exception.as_tagged(heap).erase();
        };
        let v = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw) });
        let cond_45 = Convert::is_truthy(heap, v.as_tagged(heap));
        if !cond_45 {
            return raise_runtime(vm, heap, state, VmError::NotExtensible);
        }
        // re-read through the handle: the trap may have moved the receiver
        if Proxy::is_proxy(heap, target_handle.as_tagged(heap)) {
            return target_handle.as_tagged(heap).erase();
        }
        let obj = scope
            .cast::<Object>(heap, target_handle.as_tagged(heap))
            .expect("checked above");
        set_integrity_flags(heap, &scope, obj, true);
        // re-read through the handle: traps may have moved the receiver
        target_handle.as_tagged(heap).erase()
    })
}
