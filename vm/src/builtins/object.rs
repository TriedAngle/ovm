//! ES 20.1: the Object constructor, statics, and prototype methods.

use crate::{
    Convert, GcSlice, Handle, HandleScope, Heap, Object, PropertyDescriptor, SlotName,
    Smi, Value, VmError,
};

/// Stub: `Object.prototype.toString` returns "[object Object]".
pub(crate) fn object_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
    _args: GcSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| Ok(nctx.intern(&scope, "[object Object]").value()))
}

/// `Object(x)`: returns objects unchanged (boxing of primitives is not
/// implemented yet); `new Object()`: the interpreter prepends the fresh
/// receiver, so [[Construct]] just returns it.
pub(crate) fn object_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    if nctx.is_construct() {
        return args.get(0).ok_or(VmError::Arity);
    }
    let arg = args.get(1).unwrap_or(nctx.heap().known().undefined.value());
    if nctx
        .heap()
        .no_gc(|nogc| crate::Convert::is_primitive(nogc, arg))
    {
        // TODO: box primitives (String/Symbol wrappers)
        return Err(VmError::Type);
    }
    Ok(arg)
}

/// `Object.getPrototypeOf(o)`: the receiver's map prototype. Primitive
/// arguments are a TypeError until ToObject boxing exists (ES5 behavior;
/// ES2015+ boxes them).
pub(crate) fn object_get_prototype_of(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let arg = args.get(1).ok_or(VmError::Arity)?;
    nctx.heap().no_gc(|nogc| {
        let Some(obj) = arg.as_heap_object(nogc) else {
            return Err(VmError::Type);
        };
        Ok(obj.as_ref().header.map.heap_ref(nogc).prototype.inner())
    })
}

/// `Object.setPrototypeOf(O, proto)` (ES 20.1.2.20): primitives return O
/// unchanged (after RequireObjectCoercible); proto must be an object or
/// null; the underlying [[SetPrototypeOf]] may reject (non-extensible
/// receiver, prototype cycles) with a TypeError.
/// Own enumerable-property keys in specification order: integer indices
/// ascending, then string keys in insertion order (ES 8.6.2, the
/// descriptors array is insertion-ordered).
pub(crate) fn own_property_keys<'a>(nogc: &'a crate::NoGc<'a>, target: Value) -> Vec<Value> {
    let mut keys = Vec::new();
    let Some(obj) = target.as_heap_object(nogc) else {
        return keys;
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
        keys.push(d.name().value());
    }
    keys
}

/// `Object.prototype.hasOwnProperty(key)` (ES 20.4.3.2, own properties
/// only).
pub(crate) fn object_has_own_property(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    let raw_key = args.get(1).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let Some(key) = crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? else {
        return Ok(heap.known().exception.value());
    };
    let has = heap.no_gc(|nogc| {
        let key = crate::SlotName::from_value(key);
        if let crate::Key::Element(i) =
            crate::classify_key(nogc, key.value()).unwrap_or(crate::Key::Name(key))
        {
            if let Some(obj) = receiver.as_heap_object(nogc)
                && obj.as_ref().element_value(nogc, i).is_some()
            {
                return true;
            }
        }
        match receiver.lookup(nogc, key) {
            crate::Lookup::NotFound => false,
            // the array `length` internal slot counts as an own property
            _ => true,
        }
    });
    Ok(Convert::boolean(nctx.heap(), has))
}

/// `Object.prototype.propertyIsEnumerable(key)` (ES 20.4.3.5).
pub(crate) fn object_property_is_enumerable(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    let raw_key = args.get(1).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let Some(key) = crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? else {
        return Ok(heap.known().exception.value());
    };
    let enumerable = heap.no_gc(|nogc| {
        let key = crate::SlotName::from_value(key);
        if let crate::Key::Element(i) =
            crate::classify_key(nogc, key.value()).unwrap_or(crate::Key::Name(key))
        {
            if let Some(obj) = receiver.as_heap_object(nogc)
                && obj.as_ref().element_value(nogc, i).is_some()
            {
                return true; // array elements are enumerable
            }
        }
        match receiver.lookup(nogc, key) {
            crate::Lookup::Data { flags, .. } => flags.is_enumerable(),
            crate::Lookup::Accessor {
                holder, map_index, ..
            } => holder
                .as_ref()
                .header
                .map
                .heap_ref(nogc)
                .descriptor(map_index)
                .flags()
                .is_enumerable(),
            crate::Lookup::NotFound => false,
        }
    });
    Ok(Convert::boolean(nctx.heap(), enumerable))
}

/// `Object.getOwnPropertyNames(O)` (ES 20.1.2.7).
pub(crate) fn object_get_own_property_names(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let target = args.get(1).ok_or(VmError::Arity)?;
    let names: Vec<Value> = nctx.heap().no_gc(|nogc| {
        let mut keys = own_property_keys(nogc, target);
        // arrays also list "length" (and it sorts with the strings)
        if let Some(obj) = target.as_heap_object(nogc)
            && obj.as_ref().is_array(nogc)
        {
            keys.push(nogc.known().strings.length.value());
        }
        keys
    });
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let staged = scope.stage(&names);
        Ok(heap.new_array(&scope, staged).into_tagged().erase())
    })
}

/// Build a plain `{ key: value, ... }` object from static field names.
pub(crate) fn plain_object(
    nctx: &mut crate::natives::NativeContext<'_>,
    fields: &[(&'static str, Value)],
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let map = nctx.heap().known().object_initial_map;
        let obj = nctx.heap().new_object(&scope, map, GcSlice::EMPTY).into_handle(&scope);
        for (name, value) in fields {
            let name = nctx.intern(&scope, name);
            let name = scope.handle(SlotName::from(name.as_tagged()).tagged());
            Object::define_own_property(
                nctx.heap(),
                &scope,
                obj,
                name,
                PropertyDescriptor::data(*value),
            )?;
        }
        Ok(obj.value())
    })
}

/// `Object.getOwnPropertyDescriptor(O, P)` (ES 20.1.2.5): the shared
/// raw descriptor reader (`lookup::ordinary_own_descriptor`) converted
/// to a descriptor object via FromPropertyDescriptor semantics.
pub(crate) fn object_get_own_property_descriptor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let target = args.get(1).ok_or(VmError::Arity)?;
    let raw_key = args.get(2).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let Some(key) = crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? else {
        return Ok(heap.known().exception.value());
    };
    let desc = heap.no_gc(|nogc| crate::lookup::ordinary_own_descriptor(nogc, target, key));
    let undefined = nctx.heap().known().undefined.value();
    let true_v = nctx.heap().known().true_object.value();
    let false_v = nctx.heap().known().false_object.value();
    let bool_ = |b| if b { true_v } else { false_v };
    match desc {
        Some(crate::PropertyDescriptor::Data {
            value,
            writable,
            enumerable,
            configurable,
        }) => plain_object(
            nctx,
            &[
                ("value", value),
                ("writable", bool_(writable)),
                ("enumerable", bool_(enumerable)),
                ("configurable", bool_(configurable)),
            ],
        ),
        Some(crate::PropertyDescriptor::Accessor {
            get,
            set,
            enumerable,
            configurable,
        }) => plain_object(
            nctx,
            &[
                ("get", get),
                ("set", set),
                ("enumerable", bool_(enumerable)),
                ("configurable", bool_(configurable)),
            ],
        ),
        None => Ok(undefined),
    }
}

/// `Object.defineProperty(O, P, Attributes)` (ES 20.1.2.4):
/// ToPropertyDescriptor + [[DefineOwnProperty]] (through the
/// `defineProperty` trap for proxy receivers, ES 20.2.5.6).
pub(crate) fn object_define_property(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let target = args.get(1).ok_or(VmError::Arity)?;
    let raw_key = args.get(2).ok_or(VmError::Arity)?;
    let attrs = args.get(3).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let Some(key) = crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? else {
        return Ok(heap.known().exception.value());
    };
    // shared ToPropertyDescriptor; proxies and ordinary targets both
    // complete/validate inside define_internal
    let partial = match crate::runtime::Runtime::to_property_descriptor(vm, heap, state, attrs)? {
        Some(partial) => partial,
        None => return Ok(heap.known().exception.value()),
    };
    nctx.handle_scope(|nctx, scope| {
        let target = scope.handle(target);
        let key = scope.handle(key);
        let (vm, heap, state) = nctx.split();
        match crate::proxy::define_internal(vm, heap, state, target.value(), key.value(), partial)?
        {
            crate::proxy::Flow::Threw => Ok(heap.known().exception.value()),
            crate::proxy::Flow::Value(false) => Err(VmError::Type),
            crate::proxy::Flow::Value(true) => Ok(target.value()),
        }
    })
}

pub(crate) fn object_set_prototype_of(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let target = args.get(1).ok_or(VmError::Arity)?;
    let proto = args.get(2).ok_or(VmError::Arity)?;
    let (nullish, target_is_object, proto_ok) = nctx.heap().no_gc(|nogc| {
        (
            target == nogc.known().null.value() || target == nogc.known().undefined.value(),
            !crate::Convert::is_primitive(nogc, target),
            proto == nogc.known().null.value() || !crate::Convert::is_primitive(nogc, proto),
        )
    });
    // RequireObjectCoercible(O)
    if nullish {
        return Err(VmError::Type);
    }
    // primitives are returned unchanged
    if !target_is_object {
        return Ok(target);
    }
    if !proto_ok {
        return Err(VmError::Type);
    }
    nctx.handle_scope(|nctx, scope| {
        crate::Object::set_prototype(nctx.heap(), &scope, target, proto)?;
        Ok(target)
    })
}

/// `Object.preventExtensions(O)` (ES 20.1.2.16): through the
/// `preventExtensions` trap for proxies (ES 20.2.5.3).
pub(crate) fn object_prevent_extensions(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let raw_target = args.get(1).ok_or(VmError::Arity)?;
    let nullish = nctx.heap().no_gc(|nogc| {
        raw_target == nogc.known().null.value() || raw_target == nogc.known().undefined.value()
    });
    if nullish {
        return Err(VmError::Type);
    }
    // the (possibly proxy) receiver is returned after traps ran user
    // code: keep it rooted across the call
    nctx.handle_scope(|nctx, scope| {
        let target_handle = scope.handle(raw_target);
        let target = target_handle.value();
        if !nctx
            .heap()
            .no_gc(|nogc| crate::proxy::is_js_receiver(nogc, target))
        {
            return Ok(target); // primitives returned unchanged
        }
        let (vm, heap, state) = nctx.split();
        match crate::proxy::prevent_extensions(vm, heap, state, target)? {
            crate::runtime::Coercion::Threw => Ok(heap.known().exception.value()),
            crate::runtime::Coercion::Value(v) => {
                if !heap.no_gc(|nogc| Convert::is_truthy(nogc, v)) {
                    Err(VmError::Message("object is not extensible"))
                } else {
                    // re-read through the handle: the trap above ran user
                    // code and may have moved the receiver
                    Ok(target_handle.value())
                }
            }
        }
    })
}

/// `Object.isExtensible(O)` (ES 20.1.2.14): primitives are `false`;
/// proxies run the `isExtensible` trap with its must-match invariant.
pub(crate) fn object_is_extensible(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let target = args.get(1).ok_or(VmError::Arity)?;
    if !nctx
        .heap()
        .no_gc(|nogc| crate::proxy::is_js_receiver(nogc, target))
    {
        return Ok(Convert::boolean(nctx.heap(), false));
    }
    let (vm, heap, state) = nctx.split();
    match crate::proxy::is_extensible(vm, heap, state, target)? {
        crate::runtime::Coercion::Threw => Ok(heap.known().exception.value()),
        crate::runtime::Coercion::Value(v) => Ok(v),
    }
}

/// SetIntegrityLevel (ES 7.3.15/16) for ordinary objects: clone the map
/// with `configurable` (and for freeze `writable`) cleared on every
/// descriptor and EXTENDABLE dropped. Dense array elements keep their
/// intrinsic attributes (TODO: element sealing with the elements
/// machinery).
pub(crate) fn set_integrity_flags(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    obj: Handle<'_, Object>,
    freeze: bool,
) {
    use crate::{Map, MapInit, MapKind, SlotFlags};
    let (kind, prototype, descriptors) = heap.no_gc(|nogc| {
        let map = obj.heap_ref(nogc).map_ref(nogc);
        (
            map.kind(),
            map.prototype.inner(),
            map.descriptors()
                .iter()
                .map(|d| (d.name(), d.flags(), scope.handle(d.value.inner())))
                .collect::<Vec<_>>(),
        )
    });
    let already = !kind.is_extendable()
        && descriptors.iter().all(|(_, flags, _)| {
            !flags.is_configurable() && (!freeze || flags.is_accessor() || !flags.is_writable())
        });
    if already {
        return;
    }
    let prototype = scope.handle(prototype);
    let descriptors: Vec<_> = descriptors
        .into_iter()
        .map(|(name, flags, value)| {
            let mut flags = flags;
            flags = SlotFlags::new(flags.bits() & !SlotFlags::CONFIGURABLE.bits());
            if freeze && !flags.is_accessor() {
                flags = SlotFlags::new(flags.bits() & !SlotFlags::WRITABLE.bits());
            }
            (name, flags, value)
        })
        .collect();
    heap.allocate_token_enter_nogc(Map::layout_for(descriptors.len()), |token, nogc| {
        let obj_ref = obj.heap_ref(nogc);
        let new_map = token.allocate::<Map>(MapInit {
            kind: MapKind::new(kind.bits() & !MapKind::EXTENDABLE.bits()),
            value_slot_count: obj_ref.map_ref(nogc).value_slot_count(),
            descriptors: &descriptors,
            prototype,
        });
        obj_ref
            .header
            .map
            .set(nogc, obj.value(), new_map.into_tagged());
    });
}

/// `Object.seal(O)` (ES 20.1.2.17).
pub(crate) fn object_seal(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let raw_target = args.get(1).ok_or(VmError::Arity)?;
    let nullish = nctx.heap().no_gc(|nogc| {
        raw_target == nogc.known().null.value() || raw_target == nogc.known().undefined.value()
    });
    if nullish {
        return Err(VmError::Type);
    }
    // traps run user code before the receiver is returned: keep it rooted
    nctx.handle_scope(|nctx, scope| {
        let target_handle = scope.handle(raw_target);
        let target = target_handle.value();
        if !nctx
            .heap()
            .no_gc(|nogc| crate::proxy::is_js_receiver(nogc, target))
        {
            return Ok(target);
        }
        let (vm, heap, state) = nctx.split();
        // [[PreventExtensions]] first (traps included)
        match crate::proxy::prevent_extensions(vm, heap, state, target)? {
            crate::runtime::Coercion::Threw => return Ok(heap.known().exception.value()),
            crate::runtime::Coercion::Value(v) => {
                if !heap.no_gc(|nogc| Convert::is_truthy(nogc, v)) {
                    return Err(VmError::Message("object is not extensible"));
                }
            }
        }
        // TODO: per-key [[DefineOwnProperty]] through the defineProperty
        // trap once ownKeys lands (proxy targets); ordinary targets:
        let (_, heap, _) = nctx.split();
        // re-read through the handle: the trap may have moved the receiver
        let target = target_handle.value();
        if heap.no_gc(|nogc| crate::proxy::is_proxy(nogc, target)) {
            return Ok(target);
        }
        let obj = scope.cast::<Object>(target).expect("checked above");
        set_integrity_flags(nctx.heap(), &scope, obj, false);
        // re-read through the handle: traps may have moved the receiver
        Ok(target_handle.value())
    })
}

/// `Object.freeze(O)` (ES 20.1.2.9).
pub(crate) fn object_freeze(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let raw_target = args.get(1).ok_or(VmError::Arity)?;
    let nullish = nctx.heap().no_gc(|nogc| {
        raw_target == nogc.known().null.value() || raw_target == nogc.known().undefined.value()
    });
    if nullish {
        return Err(VmError::Type);
    }
    // traps run user code before the receiver is returned: keep it rooted
    nctx.handle_scope(|nctx, scope| {
        let target_handle = scope.handle(raw_target);
        let target = target_handle.value();
        if !nctx
            .heap()
            .no_gc(|nogc| crate::proxy::is_js_receiver(nogc, target))
        {
            return Ok(target);
        }
        let (vm, heap, state) = nctx.split();
        match crate::proxy::prevent_extensions(vm, heap, state, target)? {
            crate::runtime::Coercion::Threw => return Ok(heap.known().exception.value()),
            crate::runtime::Coercion::Value(v) => {
                if !heap.no_gc(|nogc| Convert::is_truthy(nogc, v)) {
                    return Err(VmError::Message("object is not extensible"));
                }
            }
        }
        let (_, heap, _) = nctx.split();
        // re-read through the handle: the trap may have moved the receiver
        let target = target_handle.value();
        if heap.no_gc(|nogc| crate::proxy::is_proxy(nogc, target)) {
            return Ok(target);
        }
        let obj = scope.cast::<Object>(target).expect("checked above");
        set_integrity_flags(nctx.heap(), &scope, obj, true);
        // re-read through the handle: traps may have moved the receiver
        Ok(target_handle.value())
    })
}
