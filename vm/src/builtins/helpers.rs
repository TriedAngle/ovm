//! Install helpers: map/function-object allocation and property-definition
//! utilities shared by the builtin installation in `mod.rs`.

use crate::materialize::materialize_closure_vm;
use crate::natives::{NativeContext, NativeIndex};
use crate::{
    GcSlice, HandleScope, Heap, Map, MapInit, MapKind, Object, PropertyDescriptor, SlotName, Smi,
    Tagged, Value, VmError,
};

pub(crate) fn alloc_map(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    roots: &crate::RootHandles,
    kind: MapKind,
    prototype: crate::Global<Object>,
) -> Result<crate::Global<Map>, VmError> {
    alloc_map_with_slots(heap, scope, roots, kind, prototype, 0)
}

pub(crate) fn alloc_map_with_slots(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    roots: &crate::RootHandles,
    kind: MapKind,
    prototype: crate::Global<Object>,
    value_slot_count: usize,
) -> Result<crate::Global<Map>, VmError> {
    // Safety: fresh root-slot word, rooted below before any allocation.
    let proto =
        scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(prototype.read_unchecked()) });
    Ok(roots.create_handle(heap.allocate::<Map>(MapInit {
        kind,
        value_slot_count,
        descriptors: &[],
        prototype: proto,
    })))
}

/// A native function object: `CALLABLE | CONSTRUCTOR | NATIVE`, slots[0] =
/// native index, slots[1] = empty context, [[Prototype]] = Function.prototype.
pub(crate) fn make_native_function(
    thread: &mut crate::Thread,
    scope: &HandleScope<'_>,
    roots: &crate::RootHandles,
    index: NativeIndex,
) -> Result<crate::Global<Object>, VmError> {
    let heap = thread.heap();
    let kind = MapKind::OBJECT
        .union(MapKind::CALLABLE)
        .union(MapKind::CONSTRUCTOR)
        .union(MapKind::NATIVE)
        .union(MapKind::EXTENDABLE);
    let map = alloc_map_with_slots(heap, scope, roots, kind, heap.known().function_prototype, 2)?;
    let empty_context = heap.known().empty_context;
    let obj = heap.new_object(
        scope,
        map,
        scope.stage(&[
            Smi::new(index.0 as i64).into_tagged(),
            // Safety: fresh root-slot word staged into rooted slots.
            unsafe { Tagged::<Value>::from_value_unchecked(empty_context.read_unchecked()) },
        ]),
    );
    Ok(roots.create_handle(obj))
}

/// A non-constructor native function (`Proxy.revocable`-style statics).
pub(crate) fn make_native_plain_function(
    thread: &mut crate::Thread,
    scope: &HandleScope<'_>,
    roots: &crate::RootHandles,
    index: NativeIndex,
) -> Result<crate::Global<Object>, VmError> {
    let heap = thread.heap();
    let kind = MapKind::OBJECT
        .union(MapKind::CALLABLE)
        .union(MapKind::NATIVE)
        .union(MapKind::EXTENDABLE);
    let map = alloc_map_with_slots(heap, scope, roots, kind, heap.known().function_prototype, 2)?;
    let empty_context = heap.known().empty_context;
    let obj = heap.new_object(
        scope,
        map,
        scope.stage(&[
            Smi::new(index.0 as i64).into_tagged(),
            // Safety: fresh root-slot word staged into rooted slots.
            unsafe { Tagged::<Value>::from_value_unchecked(empty_context.read_unchecked()) },
        ]),
    );
    Ok(roots.create_handle(obj))
}

/// Compile and run a JS prelude once at install time (BIND_PRELUDE,
/// REVOKE_PRELUDE): its top-level assignments install hidden helpers.
pub(crate) fn run_prelude(
    thread: &mut crate::Thread,
    scope: &HandleScope<'_>,
    src: &str,
    name: &str,
) -> Result<(), VmError> {
    let (vm, heap, state) = thread.split();
    let empty = heap.known().empty_context;
    let closure = {
        let mut p = parser::Parser::new(parser::Utf8SliceStream::new(src));
        p.parse_script().map_err(|e| {
            eprintln!("{name} prelude parse error: {e}");
            VmError::Type
        })?;
        let ast = p.into_ast();
        let compiled = base_compiler::compile_script(&ast).map_err(|e| {
            eprintln!("{name} prelude compile error: {e}");
            VmError::Type
        })?;
        materialize_closure_vm(vm, heap, state, scope, &compiled, empty)?
    };
    let (vm, heap, state) = thread.split();
    let result = NativeContext::new(vm, heap, state).call(
        // Safety: fresh rooted-slot word, consumed by the call.
        unsafe { Tagged::<Value>::from_value_unchecked(closure.read_unchecked()) },
        GcSlice::EMPTY,
    )?;
    if result == unsafe { heap.known().exception.read_unchecked() } {
        if let Some(ex) = state.take_pending_exception() {
            eprintln!("{name} prelude threw: {ex:?}")
        }
        return Err(VmError::Type);
    }
    Ok(())
}

/// A constructor function + its prototype object (with `.constructor`),
/// the function installed on the global object under `name`.
pub(crate) fn install_constructor(
    thread: &mut crate::Thread,
    scope: &HandleScope<'_>,
    roots: &crate::RootHandles,
    index: NativeIndex,
    name: &str,
    proto_parent: crate::Global<Object>,
) -> Result<(crate::Global<Object>, crate::Global<Object>), VmError> {
    let name_str = thread.intern(scope, name);
    let fn_obj = make_native_function(thread, scope, roots, index)?;

    // prototype object: fresh extendable object chained to proto_parent
    let map = alloc_map(
        thread.heap(),
        scope,
        roots,
        MapKind::OBJECT.union(MapKind::EXTENDABLE),
        proto_parent,
    )?;
    let proto = roots.create_handle(thread.heap().new_object(scope, map, GcSlice::EMPTY));

    // proto.constructor = fn; fn.prototype = proto
    // (built-in methods/constructor properties are non-enumerable, ES 20+)
    let constructor_str = thread.intern(scope, "constructor");
    let prototype_str = thread.intern(scope, "prototype");
    define_method_prop(
        thread.heap(),
        scope,
        proto,
        SlotName::from_value(unsafe { constructor_str.read_unchecked() }),
        // Safety: fresh root-slot word; the define roots its inputs.
        unsafe { fn_obj.read_unchecked() },
    )?;
    define_method_prop(
        thread.heap(),
        scope,
        fn_obj,
        SlotName::from_value(unsafe { prototype_str.read_unchecked() }),
        // Safety: fresh root-slot word; the define roots its inputs.
        unsafe { proto.read_unchecked() },
    )?;

    // global.Name = fn
    let global = thread.heap().known().global_object;
    define_data(
        thread.heap(),
        scope,
        global,
        SlotName::from_value(unsafe { name_str.read_unchecked() }),
        // Safety: fresh root-slot word; the define roots its inputs.
        unsafe { fn_obj.read_unchecked() },
    )?;
    Ok((fn_obj, proto))
}

pub(crate) fn install_method(
    thread: &mut crate::Thread,
    scope: &HandleScope<'_>,
    roots: &crate::RootHandles,
    receiver: crate::Global<Object>,
    name: &str,
    index: NativeIndex,
) -> Result<(), VmError> {
    let method = make_native_function(thread, scope, roots, index)?;
    let name_str = thread.intern(scope, name);
    define_method_prop(
        thread.heap(),
        scope,
        receiver,
        SlotName::from_value(unsafe { name_str.read_unchecked() }),
        // Safety: fresh root-slot word; the define roots its inputs.
        unsafe { method.read_unchecked() },
    )?;
    Ok(())
}

/// A built-in method property: {writable: true, enumerable: false,
/// configurable: true} (ES 20.1.3-style attributes for prototype
/// methods). Non-enumerability keeps for-in/`Object.keys` clean.
#[allow(clippy::too_many_arguments)]
pub(crate) fn define_method_prop(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    object: crate::Global<Object>,
    name: SlotName,
    value: Value,
) -> Result<(), VmError> {
    // Safety: fresh root-slot word, rooted below before any allocation.
    let obj =
        scope.handle(unsafe { Tagged::<Object>::from_value_unchecked(object.read_unchecked()) });
    // Safety: pointer-keyed name word, rooted below before any allocation.
    let name = scope.handle(unsafe { name.tagged(heap) });
    Object::define_own_property(
        heap,
        scope,
        obj,
        name,
        PropertyDescriptor::Data {
            value,
            writable: true,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(())
}

pub(crate) fn define_data(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    object: crate::Global<Object>,
    name: impl Into<SlotName>,
    value: Value,
) -> Result<(), VmError> {
    // Safety: fresh root-slot word, rooted below before any allocation.
    let obj =
        scope.handle(unsafe { Tagged::<Object>::from_value_unchecked(object.read_unchecked()) });
    // Safety: pointer-keyed name word, rooted below before any allocation.
    let name = scope.handle(unsafe { name.into().tagged(heap) });
    Object::define_own_property(heap, scope, obj, name, PropertyDescriptor::data(value))?;
    Ok(())
}

/// {writable: false, enumerable: false, configurable: true} — the spec
/// attributes of builtin `length`/`name` properties.
pub(crate) fn define_non_enumerable(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    object: crate::Global<Object>,
    name: crate::Global<SlotName>,
    value: Value,
) -> Result<(), VmError> {
    // Safety: fresh root-slot word, rooted below before any allocation.
    let obj =
        scope.handle(unsafe { Tagged::<Object>::from_value_unchecked(object.read_unchecked()) });
    // Safety: fresh root-slot word, rooted below before any allocation.
    let name = scope.handle(unsafe {
        name.read_unchecked()
            .assume_valid(&*heap)
            .cast::<SlotName>()
    });
    Object::define_own_property(
        heap,
        scope,
        obj,
        name,
        PropertyDescriptor::Data {
            value,
            writable: false,
            enumerable: false,
            configurable: true,
        },
    )?;
    Ok(())
}

/// Read slots[0] of a `PRIMITIVE_WRAPPER` receiver.
pub(crate) fn wrapper_value(heap: &Heap, receiver: Tagged<'_, Value>) -> Result<Value, VmError> {
    let Some(obj) = receiver.as_heap_object() else {
        return Err(VmError::Type);
    };
    let map = obj.as_ref().header.map.heap_ref(heap);
    if !map.kind().contains(MapKind::PRIMITIVE_WRAPPER) {
        return Err(VmError::Type);
    }
    Ok(obj.as_ref().slots.heap_ref(heap).at(heap, 0).erase())
}
