use crate::Thread;
use crate::materialize::Materialize;
use crate::{
    DenseString, Float, Handle, HandleScope, HandleSlice, Heap, Map, MapInit, MapKind, Object,
    PropertyDescriptor, Smi, Tagged, Value, VmError,
};
use crate::{RuntimeContext, RuntimeIndex};

/// A runtime function object: `CALLABLE | CONSTRUCTOR | RUNTIME`, slots[0] =
/// runtime index, slots[1] = empty context, [[Prototype]] = Function.prototype.
pub fn make_runtime_function<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    index: RuntimeIndex,
) -> Result<Handle<'s, Object>, VmError> {
    make_runtime_function_in(thread.heap(), scope, index)
}

pub fn make_runtime_function_in<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    index: RuntimeIndex,
) -> Result<Handle<'s, Object>, VmError> {
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
    make_runtime_plain_function_in(thread.heap(), scope, index)
}

pub fn make_runtime_plain_function_in<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    index: RuntimeIndex,
) -> Result<Handle<'s, Object>, VmError> {
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

/// Compile and run a prelude once at install time (BIND_PRELUDE,
/// REVOKE_PRELUDE): its top-level assignments install hidden helpers.
pub fn run_prelude(
    thread: &mut Thread,
    scope: &HandleScope<'_>,
    src: &str,
    name: &str,
    compile: bytecode::CompileFn,
) -> Result<(), VmError> {
    let (vm, heap, state) = thread.split();
    let empty = heap.known().empty_context;
    let closure = {
        let program = compile(src, bytecode::SourceMode::Script).map_err(|e| {
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
    Object::define_own_property(
        thread.heap(),
        scope,
        proto,
        constructor_name,
        PropertyDescriptor::method(fn_obj.erase()),
    )?;
    let prototype_str = thread.intern(scope, "prototype");
    // Safety: fresh interned word, rooted below before any allocation.
    let prototype_name = scope.handle(prototype_str.as_tagged(&*thread.heap()));
    Object::define_own_property(
        thread.heap(),
        scope,
        fn_obj,
        prototype_name,
        PropertyDescriptor::method(proto.erase()),
    )?;

    // global.Name = fn
    let global = thread.heap().known().global_object;
    // Safety: fresh interned word, rooted below before any allocation.
    let name = scope.handle(name_str.as_tagged(&*thread.heap()));
    Object::define_own_property(
        thread.heap(),
        scope,
        global,
        name,
        PropertyDescriptor::data(fn_obj.erase()),
    )?;
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
    Object::define_own_property(
        thread.heap(),
        scope,
        receiver,
        method_name,
        PropertyDescriptor::method(method.erase()),
    )?;
    Ok(())
}

/// Read slots[0] of a `PRIMITIVE_WRAPPER` receiver — or the receiver
/// itself when it is an unboxed primitive: builtin `this`-values are
/// never auto-boxed (ES 5.2.3), so `Number.prototype.toString` and
/// friends must accept raw Smi/Float/string/boolean receivers.
pub fn wrapper_value<'a>(
    heap: &'a Heap,
    receiver: Tagged<'a, Value>,
) -> Result<Tagged<'a, Value>, VmError> {
    {
        let known = heap.known();
        let is_primitive = receiver.as_heap_object().is_none()
            || receiver.get_as::<Float>().is_some()
            || receiver.get_as::<DenseString>().is_some()
            || receiver == known.true_object.as_tagged(heap)
            || receiver == known.false_object.as_tagged(heap);
        if is_primitive {
            return Ok(receiver);
        }
    }
    let Some(obj) = receiver.as_heap_object() else {
        return Err(VmError::Type);
    };
    let map = obj.as_ref().header.map.get(heap);
    if !map.kind().contains(MapKind::PRIMITIVE_WRAPPER) {
        return Err(VmError::Type);
    }
    Ok(obj.as_ref().slots.get(heap).at(heap, 0))
}
