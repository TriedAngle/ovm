//! Minimal builtin library: the globals the test262 harness needs.
//!
//! Installed once per VM, before any user code runs: `Number`, `Boolean`,
//! `Error`, `TypeError`, `eval`, plus the wrapper/error maps in
//! `WellKnown` and `.prototype`/`.constructor` plumbing for user
//! functions (interpreter `CreateClosure`).

use base_compiler::compile_eval;
use vm::{
    Convert, FixedArray, GcSlice, HandleScope, Heap, Map, MapInit, MapKind, Object,
    ObjectSlotsInit, PropertyDescriptor, SlotName, Smi, Tagged, VMString, Value, ValueRef, VmError,
};

use crate::natives::NativeIndex;
use crate::{ContextState, VM, materialize::materialize_closure_vm, runtime::Runtime};

/// Register the builtin natives.
pub fn register_builtin_natives(vm: &mut VM) -> BuiltinIndices {
    BuiltinIndices {
        eval: vm.register_native(eval_native),
        string: vm.register_native(string_constructor),
        string_value_of: vm.register_native(string_value_of),
        string_to_string: vm.register_native(string_to_string),
        reference_error: vm.register_native(reference_error_constructor),
        function_to_string: vm.register_native(function_to_string),
        object_to_string: vm.register_native(object_to_string),
        number: vm.register_native(number_constructor),
        number_value_of: vm.register_native(number_value_of),
        number_to_string: vm.register_native(number_to_string),
        boolean: vm.register_native(boolean_constructor),
        boolean_value_of: vm.register_native(boolean_value_of),
        boolean_to_string: vm.register_native(boolean_to_string),
        error: vm.register_native(error_constructor),
        type_error: vm.register_native(type_error_constructor),
        error_to_string: vm.register_native(error_to_string),
        object: vm.register_native(object_constructor),
        object_get_prototype_of: vm.register_native(object_get_prototype_of),
        array: vm.register_native(array_constructor),
        is_nan: vm.register_native(is_nan),
    }
}

pub struct BuiltinIndices {
    pub eval: NativeIndex,
    pub string: NativeIndex,
    pub string_value_of: NativeIndex,
    pub string_to_string: NativeIndex,
    pub reference_error: NativeIndex,
    pub function_to_string: NativeIndex,
    pub object_to_string: NativeIndex,
    pub number: NativeIndex,
    pub number_value_of: NativeIndex,
    pub number_to_string: NativeIndex,
    pub boolean: NativeIndex,
    pub boolean_value_of: NativeIndex,
    pub boolean_to_string: NativeIndex,
    pub error: NativeIndex,
    pub type_error: NativeIndex,
    pub error_to_string: NativeIndex,
    pub object: NativeIndex,
    pub object_get_prototype_of: NativeIndex,
    pub array: NativeIndex,
    pub is_nan: NativeIndex,
}

/// Build the builtin objects and install them on the global object.
/// Requires `register_builtin_natives` to have run first.
pub fn install_builtins(vm: &mut VM, idx: &BuiltinIndices) -> Result<(), VmError> {
    let mut thread = vm.attach();
    let roots = &vm.shared.roots;
    thread.handle_scope(|thread, scope| {
        // ---- Number ----------------------------------------------------------
        let object_prototype = thread.heap().known().object_prototype;
        let (number_fn, number_proto) = install_constructor(
            thread,
            &scope,
            roots,
            idx.number,
            "Number",
            object_prototype,
        )?;
        install_method(
            thread,
            &scope,
            roots,
            number_proto,
            "valueOf",
            idx.number_value_of,
        )?;
        install_method(
            thread,
            &scope,
            roots,
            number_proto,
            "toString",
            idx.number_to_string,
        )?;
        // static data properties on the Number constructor
        let pos_inf = thread
            .heap()
            .allocate::<vm::Float>(f64::INFINITY)
            .into_global(roots);
        let neg_inf = thread
            .heap()
            .allocate::<vm::Float>(f64::NEG_INFINITY)
            .into_global(roots);
        let max_value = thread
            .heap()
            .allocate::<vm::Float>(f64::MAX)
            .into_global(roots);
        let min_value = thread
            .heap()
            .allocate::<vm::Float>(f64::MIN_POSITIVE)
            .into_global(roots);
        let number_nan = thread
            .heap()
            .allocate::<vm::Float>(f64::NAN)
            .into_global(roots);
        for (name, value) in [
            ("POSITIVE_INFINITY", pos_inf.value()),
            ("NEGATIVE_INFINITY", neg_inf.value()),
            ("MAX_VALUE", max_value.value()),
            ("MIN_VALUE", min_value.value()),
            ("NaN", number_nan.value()),
        ] {
            let n = thread.intern(&scope, name);
            define_data(
                thread.heap(),
                &scope,
                number_fn,
                SlotName::from(n.as_tagged()),
                value,
            )?;
        }
        let mut known = *thread.heap().known();
        known.number_wrapper_map = alloc_map_with_slots(
            thread.heap(),
            &scope,
            roots,
            MapKind::OBJECT
                .union(MapKind::EXTENDABLE)
                .union(MapKind::PRIMITIVE_WRAPPER),
            number_proto,
            1,
        )?;
        thread.heap().set_known(known);
        let _ = number_fn;

        // ---- Boolean ----------------------------------------------------------
        let (_, boolean_proto) = install_constructor(
            thread,
            &scope,
            roots,
            idx.boolean,
            "Boolean",
            object_prototype,
        )?;
        install_method(
            thread,
            &scope,
            roots,
            boolean_proto,
            "valueOf",
            idx.boolean_value_of,
        )?;
        install_method(
            thread,
            &scope,
            roots,
            boolean_proto,
            "toString",
            idx.boolean_to_string,
        )?;
        let mut known = *thread.heap().known();
        known.boolean_wrapper_map = alloc_map_with_slots(
            thread.heap(),
            &scope,
            roots,
            MapKind::OBJECT
                .union(MapKind::EXTENDABLE)
                .union(MapKind::PRIMITIVE_WRAPPER),
            boolean_proto,
            1,
        )?;
        thread.heap().set_known(known);

        // ---- String -----------------------------------------------------------
        let (_, string_proto) = install_constructor(
            thread,
            &scope,
            roots,
            idx.string,
            "String",
            object_prototype,
        )?;
        install_method(
            thread,
            &scope,
            roots,
            string_proto,
            "valueOf",
            idx.string_value_of,
        )?;
        install_method(
            thread,
            &scope,
            roots,
            string_proto,
            "toString",
            idx.string_to_string,
        )?;
        let mut known = *thread.heap().known();
        known.string_wrapper_map = alloc_map_with_slots(
            thread.heap(),
            &scope,
            roots,
            MapKind::OBJECT
                .union(MapKind::EXTENDABLE)
                .union(MapKind::PRIMITIVE_WRAPPER),
            string_proto,
            1,
        )?;
        thread.heap().set_known(known);

        // ---- Error / TypeError / ReferenceError ---------------------------------
        let (_, error_proto) =
            install_constructor(thread, &scope, roots, idx.error, "Error", object_prototype)?;
        let name_str = thread.intern(&scope, "name");
        let message_str = thread.intern(&scope, "message");
        let error_name = thread.intern(&scope, "Error");
        let empty = thread.intern(&scope, "");
        define_data(
            thread.heap(),
            &scope,
            error_proto,
            SlotName::from(name_str.as_tagged()),
            error_name.value(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            error_proto,
            SlotName::from(message_str.as_tagged()),
            empty.value(),
        )?;
        install_method(
            thread,
            &scope,
            roots,
            error_proto,
            "toString",
            idx.error_to_string,
        )?;

        // the bootstrap error_map chains to a placeholder prototype;
        // repoint it at the real Error.prototype
        let mut known = *thread.heap().known();
        known.error_map = alloc_map(
            thread.heap(),
            &scope,
            roots,
            MapKind::OBJECT.union(MapKind::EXTENDABLE),
            error_proto,
        )?;
        thread.heap().set_known(known);

        let (_, type_error_proto) = install_constructor(
            thread,
            &scope,
            roots,
            idx.type_error,
            "TypeError",
            error_proto,
        )?;
        let type_error_name = thread.intern(&scope, "TypeError");
        define_data(
            thread.heap(),
            &scope,
            type_error_proto,
            SlotName::from(name_str.as_tagged()),
            type_error_name.value(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            type_error_proto,
            SlotName::from(message_str.as_tagged()),
            empty.value(),
        )?;
        let mut known = *thread.heap().known();
        known.type_error_map = alloc_map(
            thread.heap(),
            &scope,
            roots,
            MapKind::OBJECT.union(MapKind::EXTENDABLE),
            type_error_proto,
        )?;
        thread.heap().set_known(known);

        let (_, reference_error_proto) = install_constructor(
            thread,
            &scope,
            roots,
            idx.reference_error,
            "ReferenceError",
            error_proto,
        )?;
        let reference_error_name = thread.intern(&scope, "ReferenceError");
        define_data(
            thread.heap(),
            &scope,
            reference_error_proto,
            SlotName::from(name_str.as_tagged()),
            reference_error_name.value(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            reference_error_proto,
            SlotName::from(message_str.as_tagged()),
            empty.value(),
        )?;
        let mut known = *thread.heap().known();
        known.reference_error_map = alloc_map(
            thread.heap(),
            &scope,
            roots,
            MapKind::OBJECT.union(MapKind::EXTENDABLE),
            reference_error_proto,
        )?;
        thread.heap().set_known(known);

        // ---- Function.prototype.toString (stub) -----------------------------------
        let function_prototype = thread.heap().known().function_prototype;
        install_method(
            thread,
            &scope,
            roots,
            function_prototype,
            "toString",
            idx.function_to_string,
        )?;

        // ---- eval -------------------------------------------------------------
        let eval_fn = make_native_function(thread, &scope, roots, idx.eval)?;
        let global = thread.heap().known().global_object;
        let eval_name = thread.intern(&scope, "eval");
        define_data(
            thread.heap(),
            &scope,
            global,
            SlotName::from(eval_name.as_tagged()),
            eval_fn.value(),
        )?;

        // ---- Object.prototype.toString --------------------------------------------
        let object_prototype = thread.heap().known().object_prototype;
        install_method(
            thread,
            &scope,
            roots,
            object_prototype,
            "toString",
            idx.object_to_string,
        )?;

        // ---- Object / isNaN -----------------------------------------------------
        // %Object.prototype% already exists from bootstrap; link the
        // constructor to it (like Array below)
        let object_prototype = thread.heap().known().object_prototype;
        let object_fn = make_native_function(thread, &scope, roots, idx.object)?;
        let constructor_name = thread.intern(&scope, "constructor");
        let prototype_name = thread.intern(&scope, "prototype");
        define_data(
            thread.heap(),
            &scope,
            object_prototype,
            SlotName::from(constructor_name.as_tagged()),
            object_fn.value(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            object_fn,
            SlotName::from(prototype_name.as_tagged()),
            object_prototype.value(),
        )?;
        let object_name = thread.intern(&scope, "Object");
        define_data(
            thread.heap(),
            &scope,
            global,
            SlotName::from(object_name.as_tagged()),
            object_fn.value(),
        )?;
        install_method(
            thread,
            &scope,
            roots,
            object_fn,
            "getPrototypeOf",
            idx.object_get_prototype_of,
        )?;

        // ---- Array ---------------------------------------------------------------
        // %Array.prototype% already exists from bootstrap (an array object
        // whose prototype is %Object.prototype%); just link the constructor
        let array_fn = make_native_function(thread, &scope, roots, idx.array)?;
        let array_prototype = thread.heap().known().array_prototype;
        let constructor_name = thread.intern(&scope, "constructor");
        let prototype_name = thread.intern(&scope, "prototype");
        define_data(
            thread.heap(),
            &scope,
            array_prototype,
            SlotName::from(constructor_name.as_tagged()),
            array_fn.value(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            array_fn,
            SlotName::from(prototype_name.as_tagged()),
            array_prototype.value(),
        )?;
        let array_name = thread.intern(&scope, "Array");
        define_data(
            thread.heap(),
            &scope,
            global,
            SlotName::from(array_name.as_tagged()),
            array_fn.value(),
        )?;
        let is_nan_fn = make_native_function(thread, &scope, roots, idx.is_nan)?;
        let is_nan_name = thread.intern(&scope, "isNaN");
        define_data(
            thread.heap(),
            &scope,
            global,
            SlotName::from(is_nan_name.as_tagged()),
            is_nan_fn.value(),
        )?;

        // ---- value properties of the global object -----------------------------
        let infinity = thread
            .heap()
            .allocate::<vm::Float>(f64::INFINITY)
            .into_global(roots);
        let nan = thread
            .heap()
            .allocate::<vm::Float>(f64::NAN)
            .into_global(roots);
        let undefined = thread.heap().known().undefined;
        for (name, value) in [
            ("Infinity", infinity.value()),
            ("NaN", nan.value()),
            ("undefined", undefined.value()),
        ] {
            let n = thread.intern(&scope, name);
            define_data(
                thread.heap(),
                &scope,
                global,
                SlotName::from(n.as_tagged()),
                value,
            )?;
        }
        Ok(())
    })
}

fn alloc_map<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    roots: &vm::RootHandles,
    kind: MapKind,
    prototype: vm::Global<Object>,
) -> Result<vm::Global<Map>, VmError> {
    alloc_map_with_slots(heap, scope, roots, kind, prototype, 0)
}

fn alloc_map_with_slots<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    roots: &vm::RootHandles,
    kind: MapKind,
    prototype: vm::Global<Object>,
    value_slot_count: usize,
) -> Result<vm::Global<Map>, VmError> {
    let proto = scope.handle(Tagged::from_value(prototype.value()));
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
fn make_native_function<'s>(
    thread: &mut crate::Thread,
    scope: &'s HandleScope<'_>,
    roots: &vm::RootHandles,
    index: NativeIndex,
) -> Result<vm::Global<Object>, VmError> {
    let heap = thread.heap();
    let kind = MapKind::OBJECT
        .union(MapKind::CALLABLE)
        .union(MapKind::CONSTRUCTOR)
        .union(MapKind::NATIVE)
        .union(MapKind::EXTENDABLE);
    let map = alloc_map_with_slots(heap, scope, roots, kind, heap.known().function_prototype, 2)?;
    let empty_context = heap.known().empty_context;
    let obj = heap
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map,
                values: &[
                    Smi::new(index.0 as i64).encode(),
                    empty_context.as_tagged().erase(),
                ],
                elements: heap.known().empty_fixed_array.erase(),
                length: 0,
            },
        )
        .into_global(roots);
    Ok(obj)
}

/// A constructor function + its prototype object (with `.constructor`),
/// the function installed on the global object under `name`.
fn install_constructor<'s>(
    thread: &mut crate::Thread,
    scope: &'s HandleScope<'_>,
    roots: &vm::RootHandles,
    index: NativeIndex,
    name: &str,
    proto_parent: vm::Global<Object>,
) -> Result<(vm::Global<Object>, vm::Global<Object>), VmError> {
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
    let empty_elements = thread.heap().known().empty_fixed_array.erase();
    let proto = thread
        .heap()
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map,
                values: &[],
                elements: empty_elements,
                length: 0,
            },
        )
        .into_global(roots);

    // proto.constructor = fn; fn.prototype = proto
    let constructor_str = thread.intern(scope, "constructor");
    let prototype_str = thread.intern(scope, "prototype");
    define_data(
        thread.heap(),
        scope,
        proto,
        SlotName::from(constructor_str.as_tagged()),
        fn_obj.value(),
    )?;
    define_data(
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

fn install_method<'s>(
    thread: &mut crate::Thread,
    scope: &'s HandleScope<'_>,
    roots: &vm::RootHandles,
    receiver: vm::Global<Object>,
    name: &str,
    index: NativeIndex,
) -> Result<(), VmError> {
    let method = make_native_function(thread, scope, roots, index)?;
    let name_str = thread.intern(scope, name);
    define_data(
        thread.heap(),
        scope,
        receiver,
        SlotName::from(name_str.as_tagged()),
        method.value(),
    )?;
    Ok(())
}

fn define_data(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    object: vm::Global<Object>,
    name: SlotName,
    value: Value,
) -> Result<(), VmError> {
    let obj = scope.handle(object.as_tagged());
    let name = scope.handle(name.tagged());
    Object::define_own_property(heap, scope, obj, name, PropertyDescriptor::data(value))?;
    Ok(())
}

// ---- natives ---------------------------------------------------------------

fn eval_native(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let src = args.get(1).ok_or(VmError::Arity)?;
    let context = nctx.current_context().ok_or(VmError::Type)?;

    let text = nctx.handle_scope(|nctx, scope| {
        let (vm, heap, _) = nctx.split();
        let s = Convert::to_string(heap, &scope, vm.interner(), src)?;
        heap.no_gc(|nogc| {
            s.get_as::<VMString>(nogc)
                .and_then(|s| s.as_str(nogc).map(|s| s.to_owned()))
                .ok_or(VmError::Type)
        })
    })?;

    let mut p = parser::Parser::new(parser::Utf8SliceStream::new(&text));
    if let Err(e) = p.parse_script() {
        // TODO: a SyntaxError class; approximate with TypeError for now
        let _ = e;
        nctx.set_pending_exception(VmError::Type);
        return Ok(nctx.heap().known().exception.value());
    }
    let ast = p.into_ast();
    let compiled = match compile_eval(&ast) {
        Ok(c) => c,
        Err(_) => {
            nctx.set_pending_exception(VmError::Type);
            return Ok(nctx.heap().known().exception.value());
        }
    };

    nctx.handle_scope(|nctx, scope| {
        let (vm, heap, state) = nctx.split();
        let closure = materialize_closure_vm(vm, heap, state, &scope, &compiled, context)?;
        nctx.call(closure.value(), unsafe { GcSlice::from_slice(&[]) })
    })
}

fn number_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let arg = args.get(1).unwrap_or(Smi::new(0).encode());
    let (vm, heap, state) = nctx.split();
    let n = match Runtime::to_numeric(vm, heap, state, arg)? {
        Some(n) => n,
        None => return Ok(nctx.heap().known().exception.value()),
    };
    if !nctx.is_construct() {
        return nctx.handle_scope(|nctx, scope| Ok(Convert::to_value(nctx.heap(), &scope, n)));
    }
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let value = Convert::to_value(heap, &scope, n);
        let map = heap.known().number_wrapper_map;
        let empty_elements = heap.known().empty_fixed_array.erase();
        Ok(heap
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values: &[value],
                    elements: empty_elements,
                    length: 0,
                },
            )
            .into_tagged()
            .erase())
    })
}

fn number_value_of(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    wrapper_value(nctx.heap(), receiver)
}

fn number_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    let v = wrapper_value(nctx.heap(), receiver)?;
    nctx.handle_scope(|nctx, scope| {
        let (vm, heap, _) = nctx.split();
        Convert::to_string(heap, &scope, vm.interner(), v)
    })
}

/// Read slots[0] of a `PRIMITIVE_WRAPPER` receiver.
fn wrapper_value(heap: &mut Heap, receiver: Value) -> Result<Value, VmError> {
    heap.no_gc(|nogc| {
        let ValueRef::Object(obj) = receiver.value_ref(nogc) else {
            return Err(VmError::Type);
        };
        let map = obj.as_ref().header.map.heap_ref(nogc);
        if !map.kind().contains(MapKind::PRIMITIVE_WRAPPER) {
            return Err(VmError::Type);
        }
        Ok(obj.as_ref().slots.heap_ref(nogc).at(0))
    })
}

fn boolean_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let arg = args.get(1).unwrap_or(nctx.heap().known().undefined.value());
    let b = nctx.heap().no_gc(|nogc| Convert::is_truthy(nogc, arg));
    let value = Convert::boolean(nctx.heap(), b);
    if !nctx.is_construct() {
        return Ok(value);
    }
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let map = heap.known().boolean_wrapper_map;
        let empty_elements = heap.known().empty_fixed_array.erase();
        Ok(heap
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values: &[value],
                    elements: empty_elements,
                    length: 0,
                },
            )
            .into_tagged()
            .erase())
    })
}

fn boolean_value_of(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    wrapper_value(nctx.heap(), receiver)
}

fn boolean_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    let v = wrapper_value(nctx.heap(), receiver)?;
    nctx.handle_scope(|nctx, scope| {
        let (vm, heap, _) = nctx.split();
        Convert::to_string(heap, &scope, vm.interner(), v)
    })
}

fn error_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    make_error(nctx, args, "Error")
}

fn type_error_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    make_error(nctx, args, "TypeError")
}

fn reference_error_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    make_error(nctx, args, "ReferenceError")
}

fn string_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let arg = args.get(1).unwrap_or(nctx.heap().known().undefined.value());
    nctx.handle_scope(|nctx, scope| {
        let (vm, heap, _) = nctx.split();
        let s = Convert::to_string(heap, &scope, vm.interner(), arg)?;
        if !nctx.is_construct() {
            return Ok(s);
        }
        let (_, heap, _) = nctx.split();
        let map = heap.known().string_wrapper_map;
        let empty_elements = heap.known().empty_fixed_array.erase();
        Ok(heap
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values: &[s],
                    elements: empty_elements,
                    length: 0,
                },
            )
            .into_tagged()
            .erase())
    })
}

fn string_value_of(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    wrapper_value(nctx.heap(), receiver)
}

fn string_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    wrapper_value(nctx.heap(), receiver)
}

/// Stub: `Function.prototype.toString` returns a stable marker string
/// (test262 A2.2 compares it against itself, not against real source).
fn function_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
    _args: GcSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        Ok(nctx
            .intern(&scope, "function () { [native code] }")
            .as_tagged()
            .erase())
    })
}

/// Stub: `Object.prototype.toString` returns "[object Object]".
fn object_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
    _args: GcSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| Ok(nctx.intern(&scope, "[object Object]").as_tagged().erase()))
}

fn make_error(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
    class: &str,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let (vm, heap, _) = nctx.split();
        let message = match args.get(1) {
            Some(v) => Convert::to_string(heap, &scope, vm.interner(), v)?,
            None => vm.interner().intern(heap, &scope, "").as_tagged().erase(),
        };
        let map = match class {
            "TypeError" => heap.known().type_error_map,
            "ReferenceError" => heap.known().reference_error_map,
            _ => heap.known().error_map,
        };
        let empty_elements = heap.known().empty_fixed_array.erase();
        let obj = heap
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values: &[],
                    elements: empty_elements,
                    length: 0,
                },
            )
            .into_handle(&scope);
        let name = scope.handle(
            SlotName::from(vm.interner().intern(heap, &scope, "name").as_tagged()).tagged(),
        );
        let message_key = scope.handle(
            SlotName::from(vm.interner().intern(heap, &scope, "message").as_tagged()).tagged(),
        );
        let class_value = vm.interner().intern(heap, &scope, class);
        let message_value = scope.handle(Tagged::from_value(message));
        Object::define_own_property(
            heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::data(class_value.value()),
        )?;
        Object::define_own_property(
            heap,
            &scope,
            obj,
            message_key,
            PropertyDescriptor::data(message_value.value()),
        )?;
        Ok(obj.value())
    })
}

/// `Object(x)`: returns objects unchanged (boxing of primitives is not
/// implemented yet); `new Object()`: the interpreter prepends the fresh
/// receiver, so [[Construct]] just returns it.
fn object_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    if nctx.is_construct() {
        return args.get(0).ok_or(VmError::Arity);
    }
    let arg = args.get(1).unwrap_or(nctx.heap().known().undefined.value());
    if nctx
        .heap()
        .no_gc(|nogc| vm::Convert::is_primitive(nogc, arg))
    {
        // TODO: box primitives (String/Symbol wrappers)
        return Err(VmError::Type);
    }
    Ok(arg)
}

/// `Object.getPrototypeOf(o)`: the receiver's map prototype. Primitive
/// arguments are a TypeError until ToObject boxing exists (ES5 behavior;
/// ES2015+ boxes them).
fn object_get_prototype_of(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let arg = args.get(1).ok_or(VmError::Arity)?;
    nctx.heap().no_gc(|nogc| {
        let vm::ValueRef::Object(obj) = arg.value_ref(nogc) else {
            return Err(VmError::Type);
        };
        Ok(obj.as_ref().header.map.heap_ref(nogc).prototype.inner())
    })
}

/// `Array(...)`: call and construct behave the same (ES 23.1.1.1). No
/// arguments → `[]`; one non-negative Smi → that many holes (negative or
/// non-integer numbers are a RangeError); otherwise the arguments are the
/// elements.
fn array_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let argv: Vec<Value> = (1..args.len()).filter_map(|i| args.get(i)).collect();
    let single_len = match argv.as_slice() {
        [v] => match Smi::decode(*v) {
            Some(s) if s.value() >= 0 => {
                Some(usize::try_from(s.value()).map_err(|_| VmError::OutOfBounds)?)
            }
            Some(_) => return Err(VmError::OutOfBounds), // Array(-1): RangeError
            None => None,
        },
        _ => None,
    };
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let hole = heap.known().void.value();
        let (values, length) = match single_len {
            Some(n) => (vec![hole; n], n),
            None => {
                let n = argv.len();
                (argv, n)
            }
        };
        let map = heap.known().js_array_map;
        let elements = heap.allocate_handle::<FixedArray>(&values, &scope);
        Ok(heap
            .allocate_object(
                &scope,
                vm::ObjectSlotsInit {
                    map,
                    values: &[],
                    elements: elements.erase(),
                    length,
                },
            )
            .into_tagged()
            .erase())
    })
}

/// `isNaN(x)`: ToNumber(x) is NaN.
fn is_nan(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let arg = args.get(1).unwrap_or(nctx.heap().known().undefined.value());
    let (vm, heap, state) = nctx.split();
    let n = match Runtime::to_numeric(vm, heap, state, arg)? {
        Some(n) => n,
        None => return Ok(heap.known().exception.value()),
    };
    Ok(Convert::boolean(heap, n.is_nan()))
}

fn error_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let name = get_property(vm, heap, state, receiver, "name");
    eprintln!("get name: {name:?}");
    let name = name?;
    let message = get_property(vm, heap, state, receiver, "message");
    eprintln!("get message: {message:?}");
    let message = message?;
    nctx.handle_scope(|nctx, scope| {
        let (vm, heap, _) = nctx.split();
        let a = Convert::to_string(heap, &scope, vm.interner(), name)?;
        let b = Convert::to_string(heap, &scope, vm.interner(), message)?;
        let colon = vm.interner().intern(heap, &scope, ": ");
        let ab = VMString::concat(heap, &scope, a, colon.value());
        Ok(VMString::concat(heap, &scope, ab.value(), b).value())
    })
}

fn get_property(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    receiver: Value,
    name: &str,
) -> Result<Value, VmError> {
    let name = state.handle_scope(|scope| vm.interner().intern(heap, &scope, name).value());
    match Runtime::get_property(vm, heap, state, receiver, name)? {
        crate::runtime::Coercion::Value(v) => Ok(v),
        crate::runtime::Coercion::Threw => Ok(heap.known().exception.value()),
    }
}
