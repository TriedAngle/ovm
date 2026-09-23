use crate::Thread;
use crate::materialize::Materialize;
use crate::{
    Handle, HandleScope, HandleSlice, Heap, Map, MapInit, MapKind, Object, PropertyDescriptor,
    SlotName, Smi, Tagged, Value, VmError,
};
use crate::{RuntimeContext, RuntimeIndex};

/// A runtime function object: `CALLABLE | CONSTRUCTOR | RUNTIME`, slots[0] =
/// runtime index, slots[1] = empty context, [[Prototype]] = Function.prototype.
pub fn make_runtime_function<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    index: RuntimeIndex,
) -> Result<Handle<'s, Object>, VmError> {
    let heap = thread.heap();
    let kind = MapKind::OBJECT
        .union(MapKind::CALLABLE)
        .union(MapKind::CONSTRUCTOR)
        .union(MapKind::RUNTIME)
        .union(MapKind::EXTENDABLE);
    let map = heap.allocate_handle::<Map>(
        MapInit {
            kind,
            value_slot_count: 2,
            descriptors: &[],
            prototype: heap.known().function_prototype.erase(),
        },
        scope,
    );
    let empty_context = heap.known().empty_context;
    let obj = heap.new_object(
        scope,
        map,
        scope.stage(&[
            Smi::new(index.0 as i64).into_tagged(),
            empty_context.as_tagged(heap).erase(),
        ]),
    );
    Ok(scope.handle(obj))
}

/// A non-constructor runtime function (`Proxy.revocable`-style statics).
pub fn make_runtime_plain_function<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    index: RuntimeIndex,
) -> Result<Handle<'s, Object>, VmError> {
    let heap = thread.heap();
    let kind = MapKind::OBJECT
        .union(MapKind::CALLABLE)
        .union(MapKind::RUNTIME)
        .union(MapKind::EXTENDABLE);
    let map = heap.allocate_handle::<Map>(
        MapInit {
            kind,
            value_slot_count: 2,
            descriptors: &[],
            prototype: heap.known().function_prototype.erase(),
        },
        scope,
    );
    let empty_context = heap.known().empty_context;
    let obj = heap.new_object(
        scope,
        map,
        scope.stage(&[
            Smi::new(index.0 as i64).into_tagged(),
            empty_context.as_tagged(heap).erase(),
        ]),
    );
    Ok(scope.handle(obj))
}

/// Compile and run a JS prelude once at install time (BIND_PRELUDE,
/// REVOKE_PRELUDE): its top-level assignments install hidden helpers.
pub fn run_prelude(
    thread: &mut Thread,
    scope: &HandleScope<'_>,
    src: &str,
    name: &str,
) -> Result<(), VmError> {
    let (vm, heap, state) = thread.split();
    let empty = heap.known().empty_context;
    let closure = {
        let program = js_compiler::compile_js(src, bytecode::SourceMode::Script).map_err(|e| {
            eprintln!("{name} prelude compile error: {e}");
            VmError::Type
        })?;
        Materialize::closure_vm(vm, heap, state, scope, &program, empty)?
    };
    let (vm, heap, state) = thread.split();
    let exception = heap.known().exception.as_tagged(heap).raw();
    let result = RuntimeContext::call(
        vm,
        &mut *heap,
        state,
        closure.erase(),
        HandleSlice::EMPTY,
        None,
    )?;
    if result.raw() == exception {
        if let Some(ex) = state.take_pending_exception() {
            eprintln!("{name} prelude threw: {ex:?}")
        }
        return Err(VmError::Type);
    }
    Ok(())
}

/// A constructor function + its prototype object (with `.constructor`),
/// the function installed on the global object under `name`.
pub fn install_constructor<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    index: RuntimeIndex,
    name: &str,
    proto_parent: Handle<'_, Object>,
) -> Result<(Handle<'s, Object>, Handle<'s, Object>), VmError> {
    let name_str = thread.intern(scope, name);
    let fn_obj = make_runtime_function(thread, scope, index)?;

    // prototype object: fresh extendable object chained to proto_parent
    let map = thread.heap().allocate_handle::<Map>(
        MapInit {
            kind: MapKind::OBJECT.union(MapKind::EXTENDABLE),
            value_slot_count: 0,
            descriptors: &[],
            prototype: proto_parent.erase(),
        },
        scope,
    );
    let proto = scope.handle(thread.heap().new_object(scope, map, HandleSlice::EMPTY));

    // proto.constructor = fn; fn.prototype = proto
    // (built-in methods/constructor properties are non-enumerable, ES 20+)
    let constructor_str = thread.intern(scope, "constructor");
    // Safety: fresh interned word, rooted below before any allocation.
    let constructor_name = scope.handle(constructor_str.as_tagged(&*thread.heap()));
    define_method_prop(
        thread.heap(),
        scope,
        proto,
        constructor_name,
        fn_obj.erase(),
    )?;
    let prototype_str = thread.intern(scope, "prototype");
    // Safety: fresh interned word, rooted below before any allocation.
    let prototype_name = scope.handle(prototype_str.as_tagged(&*thread.heap()));
    define_method_prop(thread.heap(), scope, fn_obj, prototype_name, proto.erase())?;

    // global.Name = fn
    let global = thread.heap().known().global_object;
    // Safety: fresh interned word, rooted below before any allocation.
    let name = scope.handle(name_str.as_tagged(&*thread.heap()));
    define_data(thread.heap(), scope, global, name, fn_obj.erase())?;
    Ok((fn_obj, proto))
}

pub fn install_method(
    thread: &mut Thread,
    scope: &HandleScope<'_>,
    receiver: Handle<'_, Object>,
    name: &str,
    index: RuntimeIndex,
) -> Result<(), VmError> {
    let method = make_runtime_function(thread, scope, index)?;
    let name_str = thread.intern(scope, name);
    // Safety: fresh interned word, rooted below before any allocation.
    let method_name = scope.handle(name_str.as_tagged(&*thread.heap()));
    define_method_prop(thread.heap(), scope, receiver, method_name, method.erase())?;
    Ok(())
}

/// A built-in method property: {writable: true, enumerable: false,
/// configurable: true} (ES 20.1.3-style attributes for prototype
/// methods). Non-enumerability keeps for-in/`Object.keys` clean.
#[allow(clippy::too_many_arguments)]
pub fn define_method_prop(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    object: Handle<'_, Object>,
    name: Handle<'_, SlotName>,
    value: Handle<'_, Value>,
) -> Result<(), VmError> {
    Object::define_own_property(
        heap,
        scope,
        object,
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

pub fn define_data(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    object: Handle<'_, Object>,
    name: Handle<'_, SlotName>,
    value: Handle<'_, Value>,
) -> Result<(), VmError> {
    Object::define_own_property(heap, scope, object, name, PropertyDescriptor::data(value))?;
    Ok(())
}

/// {writable: false, enumerable: false, configurable: true} — the spec
/// attributes of builtin `length`/`name` properties.
pub fn define_non_enumerable(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    object: Handle<'_, Object>,
    name: Handle<'_, SlotName>,
    value: Handle<'_, Value>,
) -> Result<(), VmError> {
    Object::define_own_property(
        heap,
        scope,
        object,
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
pub fn wrapper_value<'a>(
    heap: &'a Heap,
    receiver: Tagged<'_, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    let Some(obj) = receiver.as_heap_object() else {
        return Err(VmError::Type);
    };
    let map = obj.as_ref().header.map.get(heap);
    if !map.kind().contains(MapKind::PRIMITIVE_WRAPPER) {
        return Err(VmError::Type);
    }
    Ok(obj.as_ref().slots.get(heap).at(heap, 0))
}
