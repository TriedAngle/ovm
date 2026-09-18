//! Install helpers: map/function-object allocation and property-definition
//! utilities shared by the builtin installation in `mod.rs`.

use crate::Global;
use crate::RootHandles;
use crate::Thread;
use crate::materialize::materialize_closure_vm;
use crate::{
    Handle, HandleScope, HandleSlice, Heap, Map, MapInit, MapKind, Object, PropertyDescriptor,
    SlotName, Smi, Tagged, Value, VmError,
};
use crate::{RuntimeContext, RuntimeIndex};

pub fn alloc_map(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    roots: &RootHandles,
    kind: MapKind,
    prototype: Global<Object>,
) -> Result<Global<Map>, VmError> {
    alloc_map_with_slots(heap, scope, roots, kind, prototype, 0)
}

pub fn alloc_map_with_slots(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    roots: &RootHandles,
    kind: MapKind,
    prototype: Global<Object>,
    value_slot_count: usize,
) -> Result<Global<Map>, VmError> {
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

/// A runtime function object: `CALLABLE | CONSTRUCTOR | RUNTIME`, slots[0] =
/// runtime index, slots[1] = empty context, [[Prototype]] = Function.prototype.
pub fn make_runtime_function(
    thread: &mut Thread,
    scope: &HandleScope<'_>,
    roots: &RootHandles,
    index: RuntimeIndex,
) -> Result<Global<Object>, VmError> {
    let heap = thread.heap();
    let kind = MapKind::OBJECT
        .union(MapKind::CALLABLE)
        .union(MapKind::CONSTRUCTOR)
        .union(MapKind::RUNTIME)
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

/// A non-constructor runtime function (`Proxy.revocable`-style statics).
pub fn make_runtime_plain_function(
    thread: &mut Thread,
    scope: &HandleScope<'_>,
    roots: &RootHandles,
    index: RuntimeIndex,
) -> Result<Global<Object>, VmError> {
    let heap = thread.heap();
    let kind = MapKind::OBJECT
        .union(MapKind::CALLABLE)
        .union(MapKind::RUNTIME)
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
pub fn run_prelude(
    thread: &mut Thread,
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
    let result = RuntimeContext::new(vm, heap, state).call(
        // Safety: fresh rooted-slot word, consumed by the call.
        unsafe { Tagged::<Value>::from_value_unchecked(closure.read_unchecked()) },
        HandleSlice::EMPTY,
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
pub fn install_constructor(
    thread: &mut Thread,
    scope: &HandleScope<'_>,
    roots: &RootHandles,
    index: RuntimeIndex,
    name: &str,
    proto_parent: Global<Object>,
) -> Result<(Global<Object>, Global<Object>), VmError> {
    let name_str = thread.intern(scope, name);
    let fn_obj = make_runtime_function(thread, scope, roots, index)?;

    // prototype object: fresh extendable object chained to proto_parent
    let map = alloc_map(
        thread.heap(),
        scope,
        roots,
        MapKind::OBJECT.union(MapKind::EXTENDABLE),
        proto_parent,
    )?;
    let proto = roots.create_handle(thread.heap().new_object(scope, map, HandleSlice::EMPTY));

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
    roots: &RootHandles,
    receiver: Global<Object>,
    name: &str,
    index: RuntimeIndex,
) -> Result<(), VmError> {
    let method = make_runtime_function(thread, scope, roots, index)?;
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
    object: Global<Object>,
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
    object: Global<Object>,
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
    object: Global<Object>,
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
pub fn wrapper_value(heap: &Heap, receiver: Tagged<'_, Value>) -> Result<Value, VmError> {
    let Some(obj) = receiver.as_heap_object() else {
        return Err(VmError::Type);
    };
    let map = obj.as_ref().header.map.heap_ref(heap);
    if !map.kind().contains(MapKind::PRIMITIVE_WRAPPER) {
        return Err(VmError::Type);
    }
    Ok(obj.as_ref().slots.heap_ref(heap).at(heap, 0).raw())
}
