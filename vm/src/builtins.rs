//! Minimal builtin library: the globals the test262 harness needs.
//!
//! Installed once per VM, before any user code runs: `Number`, `Boolean`,
//! `Error`, `TypeError`, `eval`, plus the wrapper/error maps in
//! `WellKnown` and `.prototype`/`.constructor` plumbing for user
//! functions (interpreter `CreateClosure`).

use crate::{
    Convert, FixedArray, GcSlice, HandleScope, Heap, Map, MapInit, MapKind, Object,
    PropertyDescriptor, SlotName, Smi, VMString, Value, VmError,
};
use base_compiler::compile_eval;

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
        object_set_prototype_of: vm.register_native(object_set_prototype_of),
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
    pub object_set_prototype_of: NativeIndex,
    pub array: NativeIndex,
    pub is_nan: NativeIndex,
}

/// Build the builtin objects and install them on the global object.
/// Requires `register_builtin_natives` to have run first.
pub fn install_builtins(vm: &mut VM, idx: &BuiltinIndices) -> Result<(), VmError> {
    let mut thread = vm.attach();
    let wks = thread.heap().known().strings;
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
            .allocate::<crate::Float>(f64::INFINITY)
            .into_global(roots);
        let neg_inf = thread
            .heap()
            .allocate::<crate::Float>(f64::NEG_INFINITY)
            .into_global(roots);
        let max_value = thread
            .heap()
            .allocate::<crate::Float>(f64::MAX)
            .into_global(roots);
        let min_value = thread
            .heap()
            .allocate::<crate::Float>(f64::MIN_POSITIVE)
            .into_global(roots);
        let number_nan = thread
            .heap()
            .allocate::<crate::Float>(f64::NAN)
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
        let known = thread.heap().known();
        let error_name = thread.intern(&scope, "Error");
        define_data(
            thread.heap(),
            &scope,
            error_proto,
            known.strings.name,
            error_name.value(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            error_proto,
            known.strings.message,
            known.strings.empty.value(),
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
            wks.name,
            type_error_name.value(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            type_error_proto,
            wks.message,
            wks.empty.value(),
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
            wks.name,
            reference_error_name.value(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            reference_error_proto,
            wks.message,
            wks.empty.value(),
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
        define_data(
            thread.heap(),
            &scope,
            object_prototype,
            wks.constructor,
            object_fn.value(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            object_fn,
            wks.prototype,
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
        install_method(
            thread,
            &scope,
            roots,
            object_fn,
            "setPrototypeOf",
            idx.object_set_prototype_of,
        )?;

        // ---- Array ---------------------------------------------------------------
        // %Array.prototype% already exists from bootstrap (an array object
        // whose prototype is %Object.prototype%); just link the constructor
        let array_fn = make_native_function(thread, &scope, roots, idx.array)?;
        let array_prototype = thread.heap().known().array_prototype;
        define_data(
            thread.heap(),
            &scope,
            array_prototype,
            wks.constructor,
            array_fn.value(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            array_fn,
            wks.prototype,
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
            .allocate::<crate::Float>(f64::INFINITY)
            .into_global(roots);
        let nan = thread
            .heap()
            .allocate::<crate::Float>(f64::NAN)
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
    roots: &crate::RootHandles,
    kind: MapKind,
    prototype: crate::Global<Object>,
) -> Result<crate::Global<Map>, VmError> {
    alloc_map_with_slots(heap, scope, roots, kind, prototype, 0)
}

fn alloc_map_with_slots<'s>(
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
fn make_native_function<'s>(
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
            &[Smi::new(index.0 as i64).encode(), empty_context.value()],
        )
        .into_global(roots);
    Ok(obj)
}

/// A constructor function + its prototype object (with `.constructor`),
/// the function installed on the global object under `name`.
fn install_constructor<'s>(
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
    let proto = thread.heap().new_object(scope, map, &[]).into_global(roots);

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
    roots: &crate::RootHandles,
    receiver: crate::Global<Object>,
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
    object: crate::Global<Object>,
    name: impl Into<SlotName>,
    value: Value,
) -> Result<(), VmError> {
    let obj = scope.handle(object.as_tagged());
    let name = scope.handle(name.into().tagged());
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
        let (_vm, heap, _) = nctx.split();
        let s = Convert::to_string(heap, &scope, src)?;
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
        nctx.call(closure.value(), GcSlice::EMPTY)
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
        return nctx.handle_scope(|nctx, scope| Ok(nctx.heap().new_number(&scope, n)));
    }
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let value = heap.new_number(&scope, n);
        let map = heap.known().number_wrapper_map;
        Ok(heap.new_object(&scope, map, &[value]).into_tagged().erase())
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
        let (_vm, heap, _) = nctx.split();
        Convert::to_string(heap, &scope, v)
    })
}

/// Read slots[0] of a `PRIMITIVE_WRAPPER` receiver.
fn wrapper_value(heap: &mut Heap, receiver: Value) -> Result<Value, VmError> {
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
        Ok(heap.new_object(&scope, map, &[value]).into_tagged().erase())
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
        let (_vm, heap, _) = nctx.split();
        Convert::to_string(heap, &scope, v)
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
        let (_vm, heap, _) = nctx.split();
        let s = Convert::to_string(heap, &scope, arg)?;
        if !nctx.is_construct() {
            return Ok(s);
        }
        let (_, heap, _) = nctx.split();
        let map = heap.known().string_wrapper_map;
        Ok(heap.new_object(&scope, map, &[s]).into_tagged().erase())
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
    nctx.handle_scope(|nctx, scope| Ok(nctx.intern(&scope, "[object Object]").value()))
}

fn make_error(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
    class: &str,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let (vm, heap, _) = nctx.split();
        let message = match args.get(1) {
            Some(v) => Convert::to_string(heap, &scope, v)?,
            None => vm.interner().intern(heap, &scope, "").value(),
        };
        let map = match class {
            "TypeError" => heap.known().type_error_map,
            "ReferenceError" => heap.known().reference_error_map,
            _ => heap.known().error_map,
        };
        let obj = heap.new_object(&scope, map, &[]).into_handle(&scope);
        let name = heap.known().strings.name;
        let message_key = heap.known().strings.message;
        let class_value = vm.interner().intern(heap, &scope, class);
        let message_value = scope.handle(message);
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
fn object_get_prototype_of(
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
fn object_set_prototype_of(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let target = args.get(1).ok_or(VmError::Arity)?;
    let proto = args.get(2).ok_or(VmError::Arity)?;
    let (nullish, target_is_object, proto_ok) = nctx.heap().no_gc(|nogc| {
        (
            target == nogc.known().null.value()
                || target == nogc.known().undefined.value(),
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
        let hole = heap.known().the_hole.value();
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
                crate::ObjectSlotsInit {
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
        let a = Convert::to_string(heap, &scope, name)?;
        let b = Convert::to_string(heap, &scope, message)?;
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
