//! ES 20.1: the Object constructor, statics, and prototype methods.
use crate::Key;
use crate::Lookup;
use crate::lookup::ordinary_own_descriptor;
use crate::natives::NativeContext;
use crate::proxy::Flow;
use crate::proxy::Proxy;
use crate::runtime::Coercion;
use crate::runtime::Runtime;

use crate::{
    Convert, Handle, HandleScope, HandleSlice, Heap, Object, PropertyDescriptor, SlotName, Smi,
    Tagged, Value, VmError,
};

/// Stub: `Object.prototype.toString` returns "[object Object]".
pub fn object_to_string(
    nctx: &mut NativeContext<'_>,
    _args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let s = nctx.intern(&scope, "[object Object]");
        // Safety: fresh rooted-slot word, returned without an
        // intervening allocation.
        Ok(unsafe { s.read_unchecked() })
    })
}

/// `Object(x)`: returns objects unchanged (boxing of primitives is not
/// implemented yet); `new Object()`: the interpreter prepends the fresh
/// receiver, so [[Construct]] just returns it.
pub fn object_constructor(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    if nctx.is_construct() {
        return {
            let heap = &*nctx.heap();
            Ok(args
                .get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
                .raw())
        };
    }
    let arg = {
        let heap = &*nctx.heap();
        args.get(1)
            .map(|h| h.as_tagged(heap))
            .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase())
            .raw()
    };
    let cond_37 = {
        let heap = &*nctx.heap();
        Convert::is_primitive(heap, unsafe { arg.assume_valid(heap) })
    };
    if cond_37 {
        // TODO: box primitives (String/Symbol wrappers)
        return Err(VmError::Type);
    }
    Ok(arg)
}

/// `Object.getPrototypeOf(o)`: the receiver's map prototype. Primitive
/// arguments are a TypeError until ToObject boxing exists (ES5 behavior;
/// ES2015+ boxes them).
pub fn object_get_prototype_of(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    let heap = &*nctx.heap();
    let arg = args
        .get(1)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?;
    let Some(obj) = arg.as_heap_object() else {
        return Err(VmError::Type);
    };
    Ok(obj.as_ref().header.map.heap_ref(heap).prototype.inner())
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
    for d in obj.as_ref().header.map.heap_ref(heap).descriptors() {
        keys.push(d.name(heap).raw());
    }
    keys
}

/// `Object.prototype.hasOwnProperty(key)` (ES 20.4.3.2, own properties
/// only).
pub fn object_has_own_property(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let receiver = scope.handle({
            let heap = &*nctx.heap();
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
        });
        let raw_key = scope.handle({
            let heap = &*nctx.heap();
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
        });
        // the key coercion allocates (float/wrapper keys intern or run
        // user code): the receiver must stay rooted across it
        let (vm, heap, state) = nctx.split();
        let Some(key) = Runtime::to_property_key(vm, heap, state, raw_key)? else {
            // Safety: fresh root-slot word read for the immediate return.
            return Ok(unsafe { heap.known().exception.read_unchecked() });
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        let has = 'has: {
            let key = key.as_tagged(heap);
            if let Key::Element(i) =
                crate::Lookup::classify_key(heap, key.erase()).unwrap_or(Key::Name(key))
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
        Ok(Convert::boolean(heap, has).raw())
    })
}

/// `Object.prototype.propertyIsEnumerable(key)` (ES 20.4.3.5).
pub fn object_property_is_enumerable(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let receiver = scope.handle({
            let heap = &*nctx.heap();
            args.get(0)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
        });
        let raw_key = scope.handle({
            let heap = &*nctx.heap();
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
        });
        // the key coercion allocates (float/wrapper keys intern or run
        // user code): the receiver must stay rooted across it
        let (vm, heap, state) = nctx.split();
        let Some(key) = Runtime::to_property_key(vm, heap, state, raw_key)? else {
            // Safety: fresh root-slot word read for the immediate return.
            return Ok(unsafe { heap.known().exception.read_unchecked() });
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        let enumerable = 'enumerable: {
            let key = key.as_tagged(heap);
            if let Key::Element(i) =
                crate::Lookup::classify_key(heap, key.erase()).unwrap_or(Key::Name(key))
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
                    .heap_ref(heap)
                    .descriptor(map_index)
                    .flags()
                    .is_enumerable(),
                Lookup::NotFound => false,
            }
        };
        Ok(Convert::boolean(heap, enumerable).raw())
    })
}

/// `Object.getOwnPropertyNames(O)` (ES 20.1.2.7).
pub fn object_get_own_property_names(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    // Safety: fresh argument word; nothing below allocates before the
    // walk re-reads it under heap-borrow anchors.
    let target = {
        let heap = &*nctx.heap();
        args.get(1)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?
            .raw()
    };
    let names: Vec<Value> = {
        let heap = &*nctx.heap();
        let mut keys = own_property_keys(heap, unsafe { target.assume_valid(heap) });
        // arrays also list "length" (and it sorts with the strings)
        if let Some(obj) = unsafe { target.assume_valid(heap) }.as_heap_object()
            && obj.as_ref().is_array(heap)
        {
            // Safety: fresh root-slot word read for the storage copy.
            keys.push(unsafe { heap.known().strings.length.read_unchecked() });
        }
        keys
    };
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let staged = scope.stage(
            &names
                .iter()
                .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                .collect::<Vec<_>>(),
        );
        Ok(heap.new_array(&scope, staged).erase().raw())
    })
}

/// Build a plain `{ key: value, ... }` object from static field names.
pub fn plain_object(
    nctx: &mut NativeContext<'_>,
    fields: &[(&'static str, Handle<'_, Value>)],
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let map = nctx.heap().known().object_initial_map;
        let obj = nctx
            .heap()
            .new_object(&scope, map, HandleSlice::EMPTY)
            .into_handle(&scope);
        for (name, value) in fields {
            let name = nctx.intern(&scope, name);
            let name = scope.handle(name.as_tagged(&*nctx.heap()));
            Object::define_own_property(
                nctx.heap(),
                &scope,
                obj,
                name,
                PropertyDescriptor::data(*value),
            )?;
        }
        // Safety: fresh rooted-slot word, returned without an
        // intervening allocation.
        Ok(unsafe { obj.read_unchecked() })
    })
}

/// `Object.getOwnPropertyDescriptor(O, P)` (ES 20.1.2.5): the shared
/// raw descriptor reader (`lookup::ordinary_own_descriptor`) converted
/// to a descriptor object via FromPropertyDescriptor semantics.
pub fn object_get_own_property_descriptor(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let (raw_target, raw_key) = {
            let heap = &*nctx.heap();
            (
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
        // the key coercion allocates (Float keys intern a string): the
        // target must stay rooted across it
        // Safety: fresh argument word, rooted below before any allocation.
        let target = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_target) });
        let (vm, heap, state) = nctx.split();
        let Some(key) = Runtime::to_property_key(
            vm,
            heap,
            state,
            // Safety: fresh argument word, fresh at entry.
            scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_key) }),
        )?
        else {
            // Safety: fresh root-slot word read for the immediate return.
            return Ok(unsafe { heap.known().exception.read_unchecked() });
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        let desc = ordinary_own_descriptor(
            heap,
            &scope,
            target.as_tagged(heap),
            // fresh rooted name word re-read under the anchor
            key.as_tagged(heap).erase(),
        );
        // root the oddball singletons once for the descriptor fields
        let undefined = unsafe { nctx.heap().known().undefined.read_unchecked() };
        let true_v = scope.handle(unsafe {
            Tagged::<Value>::from_value_unchecked(nctx.heap().known().true_object.read_unchecked())
        });
        let false_v = scope.handle(unsafe {
            Tagged::<Value>::from_value_unchecked(nctx.heap().known().false_object.read_unchecked())
        });
        let bool_ = |b| if b { true_v } else { false_v };
        match desc {
            Some(PropertyDescriptor::Data {
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
            Some(PropertyDescriptor::Accessor {
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
    })
}

/// `Object.defineProperty(O, P, Attributes)` (ES 20.1.2.4):
/// ToPropertyDescriptor + [[DefineOwnProperty]] (through the
/// `defineProperty` trap for proxy receivers, ES 20.2.5.6).
pub fn object_define_property(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let (raw_target, raw_key, raw_attrs) = {
            let heap = &*nctx.heap();
            (
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
            )
        };
        // root the target, key, and descriptor: the key coercion and the
        // descriptor conversion below allocate (user getters run), which
        // would leave raw copies stale
        // Safety: fresh argument words, rooted below before any allocation.
        let target = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_target) });
        let attrs = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_attrs) });
        let (vm, heap, state) = nctx.split();
        let Some(key) = Runtime::to_property_key(
            vm,
            heap,
            state,
            // Safety: fresh argument word, fresh at entry.
            scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(raw_key) }),
        )?
        else {
            // Safety: fresh root-slot word read for the immediate return.
            return Ok(unsafe { heap.known().exception.read_unchecked() });
        };
        // root the name: the tagged result anchors the `&mut` borrow
        let key = scope.handle(key);
        // shared ToPropertyDescriptor; proxies and ordinary targets both
        // complete/validate inside define_internal
        let partial = match Runtime::to_property_descriptor(vm, heap, state, &scope, attrs)? {
            Some(partial) => partial,
            // Safety: fresh root-slot word read for the immediate return.
            None => return Ok(unsafe { heap.known().exception.read_unchecked() }),
        };
        let (vm, heap, state) = nctx.split();
        // Safety: fresh rooted-slot words, consumed by the call.
        match Proxy::define_internal(
            vm,
            heap,
            state,
            &scope,
            unsafe { Tagged::<Value>::from_value_unchecked(target.read_unchecked()) },
            // Safety: fresh rooted name word, consumed by the trap call.
            unsafe { Tagged::<Value>::from_value_unchecked(key.read_unchecked()) },
            partial,
        )? {
            // Safety: fresh root-slot word read for the immediate return.
            Flow::Threw => Ok(unsafe { heap.known().exception.read_unchecked() }),
            Flow::Value(false) => Err(VmError::Type),
            Flow::Value(true) => Ok(unsafe { target.read_unchecked() }),
        }
    })
}

pub fn object_set_prototype_of(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    let (target, proto) = {
        let heap = &*nctx.heap();
        (
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
    let (nullish, target_is_object, proto_ok) = {
        let heap = &*nctx.heap();
        let null = heap.known().null.as_tagged(heap).raw();
        let undefined = heap.known().undefined.as_tagged(heap).raw();
        (
            target == null || target == undefined,
            !Convert::is_primitive(heap, unsafe { target.assume_valid(heap) }),
            proto == null || !Convert::is_primitive(heap, unsafe { proto.assume_valid(heap) }),
        )
    };
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
        let target_h = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(target) });
        let proto_h = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(proto) });
        let target_obj = scope
            .cast::<Object>(target_h.as_tagged(&*nctx.heap()))
            .expect("target checked to be an object");
        Object::set_prototype(nctx.heap(), &scope, target_obj, proto_h)?;
        Ok(target)
    })
}

/// `Object.preventExtensions(O)` (ES 20.1.2.16): through the
/// `preventExtensions` trap for proxies (ES 20.2.5.3).
pub fn object_prevent_extensions(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        // Safety: fresh argument word, rooted below before any allocation.
        let raw_target = scope.handle({
            let heap = &*nctx.heap();
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Arity)?
        });
        let target = raw_target.as_tagged(&*nctx.heap()).raw();
        let nullish = {
            let heap = &*nctx.heap();
            let null = heap.known().null.as_tagged(heap).raw();
            let undefined = heap.known().undefined.as_tagged(heap).raw();
            // Safety: fresh rooted-slot word re-read under the anchor.
            target == null || target == undefined
        };
        if nullish {
            return Err(VmError::Type);
        }
        // the (possibly proxy) receiver is returned after traps ran user
        // code: keep it rooted across the call
        {
            let cond_38 = {
                let heap = &*nctx.heap();
                Proxy::is_js_receiver(heap, unsafe { target.assume_valid(heap) })
            };
            if !cond_38 {
                return Ok(target); // primitives returned unchanged
            }
        }
        let (vm, heap, state) = nctx.split();
        // Safety: fresh rooted-slot word, consumed by the call.
        let target = unsafe { Tagged::<Value>::from_value_unchecked(target) };
        match Proxy::prevent_extensions(vm, heap, state, target)? {
            // Safety: fresh root-slot word read for the immediate return.
            Coercion::Threw => Ok(unsafe { heap.known().exception.read_unchecked() }),
            Coercion::Value(v) => {
                let v = scope.handle(v);
                let cond_39 = Convert::is_truthy(heap, v.as_tagged(heap));
                if !cond_39 {
                    Err(VmError::Message("object is not extensible"))
                } else {
                    // re-read through the handle: the trap above ran user
                    // code and may have moved the receiver
                    Ok(raw_target.as_tagged(heap).raw())
                }
            }
        }
    })
}

/// `Object.isExtensible(O)` (ES 20.1.2.14): primitives are `false`;
/// proxies run the `isExtensible` trap with its must-match invariant.
pub fn object_is_extensible(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    let target = {
        let heap = &*nctx.heap();
        args.get(1)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?
            .raw()
    };
    let cond_40 = {
        let heap = &*nctx.heap();
        Proxy::is_js_receiver(heap, unsafe { target.assume_valid(heap) })
    };
    if !cond_40 {
        return Ok({
            let heap = &*nctx.heap();
            Convert::boolean(heap, false).raw()
        });
    }
    let (vm, heap, state) = nctx.split();
    // Safety: fresh argument word, consumed by the call.
    let target = unsafe { Tagged::<Value>::from_value_unchecked(target) };
    match Proxy::is_extensible(vm, heap, state, target)? {
        // Safety: fresh root-slot word read for the immediate return.
        Coercion::Threw => Ok(unsafe { heap.known().exception.read_unchecked() }),
        Coercion::Value(v) => Ok(v.raw()),
    }
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
        let map = obj.heap_ref(heap).map_ref(heap);
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
        let obj_ref = obj.heap_ref(heap);
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
        // Safety: fresh handle word re-read under the anchor.
        obj_ref
            .header
            .map
            .set(heap, obj.as_tagged(heap).raw(), new_map);
    });
}

/// `Object.seal(O)` (ES 20.1.2.17).
pub fn object_seal(nctx: &mut NativeContext<'_>, args: HandleSlice<'_>) -> Result<Value, VmError> {
    // Safety: fresh argument word; nothing below allocates before its
    // re-reads under heap-borrow anchors.
    let target = {
        let heap = &*nctx.heap();
        args.get(1)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?
            .raw()
    };
    let nullish = {
        let heap = &*nctx.heap();
        let null = heap.known().null.as_tagged(heap).raw();
        let undefined = heap.known().undefined.as_tagged(heap).raw();
        target == null || target == undefined
    };
    if nullish {
        return Err(VmError::Type);
    }
    // traps run user code before the receiver is returned: keep it rooted
    nctx.handle_scope(|nctx, scope| {
        // Safety: fresh argument word, rooted below before any allocation.
        let target_handle = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(target) });
        let target = target_handle.as_tagged(&*nctx.heap()).raw();
        let cond_41 = {
            let heap = &*nctx.heap();
            Proxy::is_js_receiver(heap, unsafe { target.assume_valid(heap) })
        };
        if !cond_41 {
            return Ok(target);
        }
        let (vm, heap, state) = nctx.split();
        // Safety: fresh rooted-slot word, consumed by the call.
        let t = unsafe { Tagged::<Value>::from_value_unchecked(target) };
        // [[PreventExtensions]] first (traps included)
        match Proxy::prevent_extensions(vm, heap, state, t)? {
            // Safety: fresh root-slot word read for the immediate return.
            Coercion::Threw => {
                // Safety: fresh root-slot word read for the immediate return.
                return Ok(unsafe { heap.known().exception.read_unchecked() });
            }
            Coercion::Value(v) => {
                let v = scope.handle(v);
                let cond_42 = Convert::is_truthy(heap, v.as_tagged(heap));
                if !cond_42 {
                    return Err(VmError::Message("object is not extensible"));
                }
            }
        }
        // TODO: per-key [[DefineOwnProperty]] through the defineProperty
        // trap once ownKeys lands (proxy targets); ordinary targets:
        let (_, heap, _) = nctx.split();
        // re-read through the handle: the trap may have moved the receiver
        let target = target_handle.as_tagged(heap).raw();
        let cond_43 = Proxy::is_proxy(heap, unsafe { target.assume_valid(heap) });
        if cond_43 {
            return Ok(target);
        }
        let obj = scope
            .cast::<Object>(unsafe { target.assume_valid(heap) })
            .expect("checked above");
        set_integrity_flags(heap, &scope, obj, false);
        // re-read through the handle: traps may have moved the receiver
        Ok(target_handle.as_tagged(heap).raw())
    })
}

/// `Object.freeze(O)` (ES 20.1.2.9).
pub fn object_freeze(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    // Safety: fresh argument word; nothing below allocates before its
    // re-reads under heap-borrow anchors.
    let target = {
        let heap = &*nctx.heap();
        args.get(1)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?
            .raw()
    };
    let nullish = {
        let heap = &*nctx.heap();
        let null = heap.known().null.as_tagged(heap).raw();
        let undefined = heap.known().undefined.as_tagged(heap).raw();
        target == null || target == undefined
    };
    if nullish {
        return Err(VmError::Type);
    }
    // traps run user code before the receiver is returned: keep it rooted
    nctx.handle_scope(|nctx, scope| {
        // Safety: fresh argument word, rooted below before any allocation.
        let target_handle = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(target) });
        let target = target_handle.as_tagged(&*nctx.heap()).raw();
        let cond_44 = {
            let heap = &*nctx.heap();
            Proxy::is_js_receiver(heap, unsafe { target.assume_valid(heap) })
        };
        if !cond_44 {
            return Ok(target);
        }
        let (vm, heap, state) = nctx.split();
        // Safety: fresh rooted-slot word, consumed by the call.
        let t = unsafe { Tagged::<Value>::from_value_unchecked(target) };
        match Proxy::prevent_extensions(vm, heap, state, t)? {
            Coercion::Threw => {
                // Safety: fresh root-slot word read for the immediate return.
                return Ok(unsafe { heap.known().exception.read_unchecked() });
            }
            Coercion::Value(v) => {
                let v = scope.handle(v);
                let cond_45 = Convert::is_truthy(heap, v.as_tagged(heap));
                if !cond_45 {
                    return Err(VmError::Message("object is not extensible"));
                }
            }
        }
        let (_, heap, _) = nctx.split();
        // re-read through the handle: the trap may have moved the receiver
        let target = target_handle.as_tagged(heap).raw();
        let cond_46 = Proxy::is_proxy(heap, unsafe { target.assume_valid(heap) });
        if cond_46 {
            return Ok(target);
        }
        let obj = scope
            .cast::<Object>(unsafe { target.assume_valid(heap) })
            .expect("checked above");
        set_integrity_flags(heap, &scope, obj, true);
        // re-read through the handle: traps may have moved the receiver
        Ok(target_handle.as_tagged(heap).raw())
    })
}
