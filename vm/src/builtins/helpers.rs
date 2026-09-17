//! Install helpers: map/function-object allocation and property-definition
//! utilities shared by the builtin installation in `mod.rs`.

use crate::materialize::materialize_closure_vm;
use crate::natives::{NativeContext, NativeIndex};
use crate::{
    GcSlice, HandleScope, Heap, Map, MapInit, MapKind, Object, PropertyDescriptor,
    SlotName, Smi, Value, VmError,
};

pub(crate) fn alloc_map<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    roots: &crate::RootHandles,
    kind: MapKind,
    prototype: crate::Global<Object>,
) -> Result<crate::Global<Map>, VmError> {
    alloc_map_with_slots(heap, scope, roots, kind, prototype, 0)
}

pub(crate) fn alloc_map_with_slots<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    roots: &crate::RootHandles,
    kind: MapKind,
    prototype: crate::Global<Object>,
    value_slot_count: usize,
) -> Result<crate::Global<Map>, VmError> {
    let proto = scope.handle(prototype.value());
    Ok(heap
        .allocate::<Map>(MapInit {
            kind,
            value_slot_count,
            descriptors: &[],
            prototype: proto,
        })
        .into_global(roots))
}

/// A native function object: `CALLABLE | CONSTRUCTOR | NATIVE`, slots[0] =
/// native index, slots[1] = empty context, [[Prototype]] = Function.prototype.
pub(crate) fn make_native_function<'s>(
    thread: &mut crate::Thread,
    scope: &'s HandleScope<'_>,
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
    let obj = heap
        .new_object(
            scope,
            map,
            scope.stage(&[Smi::new(index.0 as i64).encode(), empty_context.value()]),
        )
        .into_global(roots);
    Ok(obj)
}

/// A non-constructor native function (`Proxy.revocable`-style statics).
pub(crate) fn make_native_plain_function<'s>(
    thread: &mut crate::Thread,
    scope: &'s HandleScope<'_>,
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
    let obj = heap
        .new_object(
            scope,
            map,
            scope.stage(&[Smi::new(index.0 as i64).encode(), empty_context.value()]),
        )
        .into_global(roots);
    Ok(obj)
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
    let result = NativeContext::new(vm, heap, state).call(closure.value(), GcSlice::EMPTY)?;
    if result == heap.known().exception.value() {
        state
            .take_pending_exception()
            .map(|ex| eprintln!("{name} prelude threw: {ex:?}"));
        return Err(VmError::Type);
    }
    Ok(())
}

/// A constructor function + its prototype object (with `.constructor`),
/// the function installed on the global object under `name`.
pub(crate) fn install_constructor<'s>(
    thread: &mut crate::Thread,
    scope: &'s HandleScope<'_>,
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
    let proto = thread.heap().new_object(scope, map, GcSlice::EMPTY).into_global(roots);

    // proto.constructor = fn; fn.prototype = proto
    // (built-in methods/constructor properties are non-enumerable, ES 20+)
    let constructor_str = thread.intern(scope, "constructor");
    let prototype_str = thread.intern(scope, "prototype");
    define_method_prop(
        thread.heap(),
        scope,
        proto,
        SlotName::from(constructor_str.as_tagged()),
        fn_obj.value(),
    )?;
    define_method_prop(
        thread.heap(),
        scope,
        fn_obj,
        SlotName::from(prototype_str.as_tagged()),
        proto.value(),
    )?;

    // global.Name = fn
    let global = thread.heap().known().global_object;
    define_data(
        thread.heap(),
        scope,
        global,
        SlotName::from(name_str.as_tagged()),
        fn_obj.value(),
    )?;
    Ok((fn_obj, proto))
}

pub(crate) fn install_method<'s>(
    thread: &mut crate::Thread,
    scope: &'s HandleScope<'_>,
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
        SlotName::from(name_str.as_tagged()),
        method.value(),
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
    let obj = scope.handle(object.as_tagged());
    let name = scope.handle(name.tagged());
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
    let obj = scope.handle(object.as_tagged());
    let name = scope.handle(name.into().tagged());
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
    let obj = scope.handle(object.as_tagged());
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
pub(crate) fn wrapper_value(heap: &mut Heap, receiver: Value) -> Result<Value, VmError> {
    heap.no_gc(|nogc| {
        let Some(obj) = receiver.as_heap_object(nogc) else {
            return Err(VmError::Type);
        };
        let map = obj.as_ref().header.map.heap_ref(nogc);
        if !map.kind().contains(MapKind::PRIMITIVE_WRAPPER) {
            return Err(VmError::Type);
        }
        Ok(obj.as_ref().slots.heap_ref(nogc).at(0))
    })
}
