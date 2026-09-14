//! Minimal builtin library: the globals the test262 harness needs.
//!
//! Installed once per VM, before any user code runs: `Number`, `Boolean`,
//! `Error`, `TypeError`, `eval`, plus the wrapper/error maps in
//! `WellKnown` and `.prototype`/`.constructor` plumbing for user
//! functions (interpreter `CreateClosure`).

use crate::{
    Context, Convert, FixedArray, GcSlice, HandleScope, Heap, Map, MapInit, MapKind, Object,
    PropertyDescriptor, SlotFlags, SlotName, Smi, Symbol, VMString, Value, VmError,
};
use base_compiler::compile_eval;

use crate::natives::{NativeContext, NativeIndex};
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
        array_values: vm.register_native(array_values),
        array_iterator_next: vm.register_native(array_iterator_next),
        array_iterator_symbol_iterator: vm.register_native(array_iterator_symbol_iterator),
        symbol: vm.register_native(symbol_constructor),
        object_has_own_property: vm.register_native(object_has_own_property),
        object_property_is_enumerable: vm.register_native(object_property_is_enumerable),
        object_get_own_property_names: vm.register_native(object_get_own_property_names),
        object_get_own_property_descriptor: vm.register_native(object_get_own_property_descriptor),
        object_define_property: vm.register_native(object_define_property),
        function_call: vm.register_native(function_call),
        function_apply: vm.register_native(function_apply),
        function_bind: vm.register_native(function_bind),
        function_constructor: vm.register_native(function_constructor),
        array_is_array: vm.register_native(array_is_array),
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
    pub array_values: NativeIndex,
    pub array_iterator_next: NativeIndex,
    pub array_iterator_symbol_iterator: NativeIndex,
    pub symbol: NativeIndex,
    pub object_has_own_property: NativeIndex,
    pub object_property_is_enumerable: NativeIndex,
    pub object_get_own_property_names: NativeIndex,
    pub object_get_own_property_descriptor: NativeIndex,
    pub object_define_property: NativeIndex,
    pub function_call: NativeIndex,
    pub function_apply: NativeIndex,
    pub function_bind: NativeIndex,
    pub function_constructor: NativeIndex,
    pub array_is_array: NativeIndex,
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

        // ---- Function (constructor: dynamic bodies via eval, ES 20.2.1) --------
        let function_prototype = thread.heap().known().function_prototype;
        let function_fn = make_native_function(thread, &scope, roots, idx.function_constructor)?;
        define_method_prop(
            thread.heap(),
            &scope,
            function_prototype,
            SlotName::from_value(wks.constructor.value()),
            function_fn.value(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            function_fn,
            SlotName::from_value(wks.prototype.value()),
            function_prototype.value(),
        )?;
        let function_name = thread.intern(&scope, "Function");
        let global_object = thread.heap().known().global_object;
        define_data(
            thread.heap(),
            &scope,
            global_object,
            SlotName::from(function_name.as_tagged()),
            function_fn.value(),
        )?;

        // ---- Function.prototype toString/call/apply/bind ------------------------
        install_method(
            thread,
            &scope,
            roots,
            function_prototype,
            "toString",
            idx.function_to_string,
        )?;
        install_method(
            thread,
            &scope,
            roots,
            function_prototype,
            "call",
            idx.function_call,
        )?;
        install_method(
            thread,
            &scope,
            roots,
            function_prototype,
            "apply",
            idx.function_apply,
        )?;
        install_method(
            thread,
            &scope,
            roots,
            function_prototype,
            "bind",
            idx.function_bind,
        )?;
        // bind is a JS closure (see BIND_PRELUDE); compile and run it once
        // here, capturing the empty context
        {
            let (vm, heap, state) = thread.split();
            let empty = heap.known().empty_context;
            let closure = {
                let mut p = parser::Parser::new(parser::Utf8SliceStream::new(BIND_PRELUDE));
                p.parse_script().map_err(|e| {
                    eprintln!("bind prelude parse error: {e}");
                    VmError::Type
                })?;
                let ast = p.into_ast();
                let compiled = base_compiler::compile_script(&ast).map_err(|e| {
                    eprintln!("bind prelude compile error: {e}");
                    VmError::Type
                })?;
                materialize_closure_vm(vm, heap, state, &scope, &compiled, empty)?
            };
            let (vm, heap, state) = thread.split();
            let result =
                NativeContext::new(vm, heap, state).call(closure.value(), GcSlice::EMPTY)?;
            if result == heap.known().exception.value() {
                state
                    .take_pending_exception()
                    .map(|ex| eprintln!("bind prelude threw: {ex:?}"));
                return Err(VmError::Type);
            }
        }

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

        // ---- Object.prototype.toString/hasOwnProperty/propertyIsEnumerable ------
        let object_prototype = thread.heap().known().object_prototype;
        install_method(
            thread,
            &scope,
            roots,
            object_prototype,
            "toString",
            idx.object_to_string,
        )?;
        install_method(
            thread,
            &scope,
            roots,
            object_prototype,
            "hasOwnProperty",
            idx.object_has_own_property,
        )?;
        install_method(
            thread,
            &scope,
            roots,
            object_prototype,
            "propertyIsEnumerable",
            idx.object_property_is_enumerable,
        )?;

        // ---- Object / isNaN -----------------------------------------------------
        // %Object.prototype% already exists from bootstrap; link the
        // constructor to it (like Array below)
        let object_prototype = thread.heap().known().object_prototype;
        let object_fn = make_native_function(thread, &scope, roots, idx.object)?;
        define_method_prop(
            thread.heap(),
            &scope,
            object_prototype,
            SlotName::from_value(wks.constructor.value()),
            object_fn.value(),
        )?;
        define_method_prop(
            thread.heap(),
            &scope,
            object_fn,
            SlotName::from_value(wks.prototype.value()),
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
            "getOwnPropertyNames",
            idx.object_get_own_property_names,
        )?;
        install_method(
            thread,
            &scope,
            roots,
            object_fn,
            "getOwnPropertyDescriptor",
            idx.object_get_own_property_descriptor,
        )?;
        install_method(
            thread,
            &scope,
            roots,
            object_fn,
            "defineProperty",
            idx.object_define_property,
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
        define_method_prop(
            thread.heap(),
            &scope,
            array_prototype,
            SlotName::from_value(wks.constructor.value()),
            array_fn.value(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            array_fn,
            SlotName::from_value(wks.prototype.value()),
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

        // Array.isArray
        let is_array_fn = make_native_function(thread, &scope, roots, idx.array_is_array)?;
        let is_array_name = thread.intern(&scope, "isArray");
        define_data(
            thread.heap(),
            &scope,
            array_fn,
            SlotName::from(is_array_name.as_tagged()),
            is_array_fn.value(),
        )?;

        // ---- Array iteration (the iterator protocol minimum) ---------------------
        // %ArrayIteratorPrototype%: next + @@iterator (returns the receiver)
        let array_iterator_prototype = {
            let map = alloc_map(
                thread.heap(),
                &scope,
                roots,
                MapKind::OBJECT.union(MapKind::EXTENDABLE),
                object_prototype,
            )?;
            thread
                .heap()
                .new_object(&scope, map, &[])
                .into_global(roots)
        };
        install_method(
            thread,
            &scope,
            roots,
            array_iterator_prototype,
            "next",
            idx.array_iterator_next,
        )?;
        let sym_iterator_iter =
            make_native_function(thread, &scope, roots, idx.array_iterator_symbol_iterator)?;
        let iterator_symbol = thread.heap().known().iterator_symbol;
        define_method_prop(
            thread.heap(),
            &scope,
            array_iterator_prototype,
            SlotName::from(iterator_symbol.as_tagged()),
            sym_iterator_iter.value(),
        )?;

        // Array.prototype.values === Array.prototype[Symbol.iterator]: a
        // native returning a fresh array-iterator object
        let values_fn = make_native_function(thread, &scope, roots, idx.array_values)?;
        let values_name = thread.intern(&scope, "values");
        define_method_prop(
            thread.heap(),
            &scope,
            array_prototype,
            SlotName::from(values_name.as_tagged()),
            values_fn.value(),
        )?;
        define_method_prop(
            thread.heap(),
            &scope,
            array_prototype,
            SlotName::from(iterator_symbol.as_tagged()),
            values_fn.value(),
        )?;

        // array-iterator map: slots [iterated array, next index], prototype
        // %ArrayIteratorPrototype%
        let array_iterator_map = alloc_map_with_slots(
            thread.heap(),
            &scope,
            roots,
            MapKind::OBJECT.union(MapKind::EXTENDABLE),
            array_iterator_prototype,
            2,
        )?;

        // iterator-result map: { value, done } both {w+, e+, c+}
        let iterator_result_map = {
            let value_name = thread.intern(&scope, "value");
            let done_name = thread.intern(&scope, "done");
            let flags = SlotFlags::WRITABLE
                .union(SlotFlags::CONFIGURABLE)
                .union(SlotFlags::ENUMERABLE);
            let descriptors = vec![
                (
                    SlotName::from(value_name.as_tagged()),
                    flags,
                    Smi::new(0).encode(),
                ),
                (
                    SlotName::from(done_name.as_tagged()),
                    flags,
                    Smi::new(1).encode(),
                ),
            ];
            let proto = scope.handle(object_prototype.value());
            thread
                .heap()
                .allocate::<Map>(MapInit {
                    kind: MapKind::OBJECT.union(MapKind::EXTENDABLE),
                    value_slot_count: 2,
                    descriptors: &descriptors,
                    prototype: proto,
                })
                .into_global(roots)
        };

        let mut known = *thread.heap().known();
        known.array_iterator_map = array_iterator_map;
        known.array_iterator_prototype = array_iterator_prototype;
        known.iterator_result_map = iterator_result_map;
        // for-in enumerator map: slots [level, keys, index, visited],
        // prototype Object.prototype (it must survive `x in Object.prototype`
        // style probes without special cases; the object is never exposed)
        let for_in_enumerator_map = alloc_map_with_slots(
            thread.heap(),
            &scope,
            roots,
            MapKind::OBJECT.union(MapKind::EXTENDABLE),
            object_prototype,
            4,
        )?;
        known.for_in_enumerator_map = for_in_enumerator_map;
        thread.heap().set_known(known);

        let is_nan_fn = make_native_function(thread, &scope, roots, idx.is_nan)?;
        let is_nan_name = thread.intern(&scope, "isNaN");
        define_data(
            thread.heap(),
            &scope,
            global,
            SlotName::from(is_nan_name.as_tagged()),
            is_nan_fn.value(),
        )?;

        // ---- Symbol (minimal: constructor + Symbol.iterator) ---------------
        // enough to author custom iterables; the full Symbol surface stays
        // gated by the test262 feature skip
        let symbol_fn = make_native_function(thread, &scope, roots, idx.symbol)?;
        let symbol_name = thread.intern(&scope, "Symbol");
        define_data(
            thread.heap(),
            &scope,
            global,
            SlotName::from(symbol_name.as_tagged()),
            symbol_fn.value(),
        )?;
        let iterator_symbol = thread.heap().known().iterator_symbol;
        let iter_name = thread.intern(&scope, "iterator");
        define_data(
            thread.heap(),
            &scope,
            symbol_fn,
            SlotName::from(iter_name.as_tagged()),
            iterator_symbol.value(),
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
            // spec attributes are {writable: false, enumerable: false,
            // configurable: false} (ES 19.1.1); non-configurability is
            // what `delete NaN` observes, writable/enumerable stay per
            //missive until non-writable stores stop throwing in sloppy
            // code
            Object::define_own_property(
                thread.heap(),
                &scope,
                scope.handle(global.as_tagged()),
                scope.handle(SlotName::from(n.as_tagged()).tagged()),
                PropertyDescriptor::Data {
                    value,
                    writable: true,
                    enumerable: true,
                    configurable: false,
                },
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
fn define_method_prop(
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
        let context = scope.cast::<Context>(context).ok_or(VmError::Type)?;
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
/// Own enumerable-property keys in specification order: integer indices
/// ascending, then string keys in insertion order (ES 8.6.2, the
/// descriptors array is insertion-ordered).
fn own_property_keys<'a>(nogc: &'a crate::NoGc<'a>, target: Value) -> Vec<Value> {
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
fn object_has_own_property(
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
fn object_property_is_enumerable(
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
fn object_get_own_property_names(
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
        let map = heap.known().js_array_map;
        let elements = heap.allocate_handle::<FixedArray>(&names, &scope);
        Ok(heap
            .allocate_object(
                &scope,
                crate::ObjectSlotsInit {
                    map,
                    values: &[],
                    elements: elements.erase(),
                    length: names.len(),
                },
            )
            .into_tagged()
            .erase())
    })
}

/// Build a plain `{ key: value, ... }` object from static field names.
fn plain_object(
    nctx: &mut crate::natives::NativeContext<'_>,
    fields: &[(&'static str, Value)],
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let map = nctx.heap().known().object_initial_map;
        let obj = nctx.heap().new_object(&scope, map, &[]).into_handle(&scope);
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

/// `Object.getOwnPropertyDescriptor(O, P)` (ES 20.1.2.5).
fn object_get_own_property_descriptor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let target = args.get(1).ok_or(VmError::Arity)?;
    let raw_key = args.get(2).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let Some(key) = crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? else {
        return Ok(heap.known().exception.value());
    };
    enum Desc {
        Data {
            value: Value,
            writable: bool,
            enumerable: bool,
            configurable: bool,
        },
        Accessor {
            get: Value,
            set: Value,
            enumerable: bool,
            configurable: bool,
        },
        /// the JSArray `length` internal slot {w+, e−, c−}
        Length(Value),
        Missing,
    }
    let desc = heap.no_gc(|nogc| {
        let key = crate::SlotName::from_value(key);
        if let crate::Key::Element(i) =
            crate::classify_key(nogc, key.value()).unwrap_or(crate::Key::Name(key))
        {
            if let Some(obj) = target.as_heap_object(nogc)
                && let Some(v) = obj.as_ref().element_value(nogc, i)
            {
                return Desc::Data {
                    value: v,
                    writable: true,
                    enumerable: true,
                    configurable: true,
                };
            }
        }
        if let Some(obj) = target.as_heap_object(nogc)
            && let Some(v) = obj.as_ref().array_length(nogc, key)
        {
            return Desc::Length(v);
        }
        match target.lookup(nogc, key) {
            crate::Lookup::Data { slot, flags, .. } => Desc::Data {
                value: slot.inner(),
                writable: flags.is_writable(),
                enumerable: flags.is_enumerable(),
                configurable: flags.is_configurable(),
            },
            crate::Lookup::Accessor {
                pair,
                holder,
                map_index,
            } => {
                let flags = holder
                    .as_ref()
                    .header
                    .map
                    .heap_ref(nogc)
                    .descriptor(map_index)
                    .flags();
                Desc::Accessor {
                    get: pair.get.inner(),
                    set: pair.set.inner(),
                    enumerable: flags.is_enumerable(),
                    configurable: flags.is_configurable(),
                }
            }
            crate::Lookup::NotFound => Desc::Missing,
        }
    });
    let undefined = nctx.heap().known().undefined.value();
    let true_v = nctx.heap().known().true_object.value();
    let false_v = nctx.heap().known().false_object.value();
    let bool_ = |b| if b { true_v } else { false_v };
    match desc {
        Desc::Data {
            value,
            writable,
            enumerable,
            configurable,
        } => plain_object(
            nctx,
            &[
                ("value", value),
                ("writable", bool_(writable)),
                ("enumerable", bool_(enumerable)),
                ("configurable", bool_(configurable)),
            ],
        ),
        Desc::Length(value) => plain_object(
            nctx,
            &[
                ("value", value),
                ("writable", bool_(true)),
                ("enumerable", bool_(false)),
                ("configurable", bool_(false)),
            ],
        ),
        Desc::Accessor {
            get,
            set,
            enumerable,
            configurable,
        } => plain_object(
            nctx,
            &[
                ("get", get),
                ("set", set),
                ("enumerable", bool_(enumerable)),
                ("configurable", bool_(configurable)),
            ],
        ),
        Desc::Missing => Ok(undefined),
    }
}

/// `Object.defineProperty(O, P, Attributes)` (ES 20.1.2.4): ToPropertyDescriptor
/// + [[DefineOwnProperty]].
fn object_define_property(
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
    if heap.no_gc(|nogc| Convert::is_primitive(nogc, attrs)) {
        return Err(VmError::Type);
    }
    // ToPropertyDescriptor (ES 7.1.6, data descriptors only)
    let read = |nctx: &mut crate::natives::NativeContext<'_>,
                name: &str|
     -> Result<Option<Value>, VmError> {
        // interned: descriptor lookup matches property names by identity
        let s = nctx.handle_scope(|nctx, scope| nctx.intern(&scope, name).value());
        let (vm, heap, state) = nctx.split();
        match crate::runtime::Runtime::get_property(vm, heap, state, attrs, s)? {
            crate::runtime::Coercion::Threw => Err(VmError::Type),
            crate::runtime::Coercion::Value(v) => Ok(Some(v)),
        }
    };
    let undefined = nctx.heap().known().undefined.value();
    let has_value = read(nctx, "value")?.filter(|v| *v != undefined);
    let has_get = read(nctx, "get")?.filter(|v| *v != undefined);
    let has_set = read(nctx, "set")?.filter(|v| *v != undefined);
    let writable = read(nctx, "writable")?
        .map(|v| nctx.heap().no_gc(|nogc| Convert::is_truthy(nogc, v)))
        .unwrap_or(true);
    let enumerable = read(nctx, "enumerable")?
        .map(|v| nctx.heap().no_gc(|nogc| Convert::is_truthy(nogc, v)))
        .unwrap_or(false);
    let configurable = read(nctx, "configurable")?
        .map(|v| nctx.heap().no_gc(|nogc| Convert::is_truthy(nogc, v)))
        .unwrap_or(false);
    let desc = if let Some(get) = has_get.or(has_set.map(|_| nctx.heap().known().undefined.value()))
    {
        // accessor descriptor (ES 7.1.6): the getters/setters must be
        // callable-or-undefined
        let get = if has_get.is_some() {
            get
        } else {
            nctx.heap().known().undefined.value()
        };
        let set = has_set.unwrap_or_else(|| nctx.heap().known().undefined.value());
        for half in [get, set] {
            if half != nctx.heap().known().undefined.value()
                && !crate::runtime::Runtime::is_callable(nctx.heap(), half)
            {
                return Err(VmError::Type);
            }
        }
        PropertyDescriptor::Accessor {
            get,
            set,
            enumerable,
            configurable,
        }
    } else {
        PropertyDescriptor::Data {
            value: has_value.unwrap_or_else(|| nctx.heap().known().undefined.value()),
            writable,
            enumerable,
            configurable,
        }
    };
    nctx.handle_scope(|nctx, scope| {
        let target = scope.cast::<Object>(target).ok_or(VmError::Type)?;
        let key = scope.handle(crate::SlotName::from_value(key).tagged());
        let defined = Object::define_own_property(nctx.heap(), &scope, target, key, desc)?;
        if !defined {
            return Err(VmError::Type);
        }
        Ok(target.value())
    })
}

/// `Function.prototype.call(thisArg, ...args)` (ES 20.2.3.4).
fn function_call(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let f = args.get(0).ok_or(VmError::Arity)?;
    if !crate::runtime::Runtime::is_callable(nctx.heap(), f) {
        return Err(VmError::Type);
    }
    let call_args: Vec<Value> = args.as_slice().iter().skip(1).copied().collect();
    nctx.handle_scope(|nctx, scope| nctx.call(f, scope.stage(&call_args)))
}

/// `Function.prototype.bind(thisArg, ...prepend)` (ES 20.2.3.5): the
/// bound function is the JS closure template installed by BIND_PRELUDE,
/// called with (target, thisArg, prepend-array).
fn function_bind(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let f = args.get(0).ok_or(VmError::Arity)?;
    if !crate::runtime::Runtime::is_callable(nctx.heap(), f) {
        return Err(VmError::Type);
    }
    let this_arg = args
        .get(1)
        .unwrap_or_else(|| nctx.heap().known().undefined.value());
    let prepend: Vec<Value> = args.as_slice().iter().skip(2).copied().collect();
    // Function.prototype.__makeBound (installed by BIND_PRELUDE)
    let make_bound = {
        let name = nctx.handle_scope(|nctx, scope| nctx.intern(&scope, "__makeBound").value());
        let (vm, heap, state) = nctx.split();
        let proto = heap.known().function_prototype.value();
        match crate::runtime::Runtime::get_property(vm, heap, state, proto, name)? {
            crate::runtime::Coercion::Threw => return Ok(heap.known().exception.value()),
            crate::runtime::Coercion::Value(v) => v,
        }
    };
    let array = nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let map = heap.known().js_array_map;
        let elements = heap.allocate_handle::<FixedArray>(&prepend, &scope);
        heap.allocate_object(
            &scope,
            crate::ObjectSlotsInit {
                map,
                values: &[],
                elements: elements.erase(),
                length: prepend.len(),
            },
        )
        .into_tagged()
        .erase()
    });
    nctx.handle_scope(|nctx, scope| {
        // args[0] is the receiver (undefined for the plain call)
        let recv = nctx.heap().known().undefined.value();
        nctx.call(make_bound, scope.stage(&[recv, f, this_arg, array]))
    })
}

/// `Function.prototype.apply(thisArg, argsArray)` (ES 20.2.3.3).
fn function_apply(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let f = args.get(0).ok_or(VmError::Arity)?;
    if !crate::runtime::Runtime::is_callable(nctx.heap(), f) {
        return Err(VmError::Type);
    }
    let this_arg = args
        .get(1)
        .unwrap_or_else(|| nctx.heap().known().undefined.value());
    let array = args
        .get(2)
        .unwrap_or_else(|| nctx.heap().known().undefined.value());
    let call_args: Vec<Value> = if array == nctx.heap().known().undefined.value()
        || array == nctx.heap().known().null.value()
    {
        vec![this_arg]
    } else {
        // array-like: read elements 0..length (holes read as undefined)
        let len = nctx.heap().no_gc(|nogc| {
            array
                .as_heap_object(nogc)
                .map(|o| {
                    o.as_ref()
                        .array_length(
                            nogc,
                            crate::SlotName::from_value(nogc.known().strings.length.value()),
                        )
                        .and_then(|v| Smi::decode(v).map(|s| s.value() as usize))
                        .unwrap_or(0)
                })
                .unwrap_or(0)
        });
        let mut out = Vec::with_capacity(len + 1);
        out.push(this_arg);
        for i in 0..len {
            out.push(nctx.heap().no_gc(|nogc| {
                array
                    .as_heap_object(nogc)
                    .and_then(|o| o.as_ref().element_value(nogc, i))
                    .unwrap_or_else(|| nogc.known().undefined.value())
            }));
        }
        out
    };
    nctx.handle_scope(|nctx, scope| nctx.call(f, scope.stage(&call_args)))
}

/// The bound-function template: a plain JS closure over (target, bound
/// this, prepend array). The native `function_bind` builds the prepend
/// array and delegates here — the native registry holds stateless fn
/// pointers, so the closure state must live in a JS closure.
const BIND_PRELUDE: &str = r#"
Function.prototype.__makeBound = function (f, t, p) {
  return function (...rest) {
    var all = [];
    for (var i = 0; i < p.length; i++) all[all.length] = p[i];
    for (var j = 0; j < rest.length; j++) all[all.length] = rest[j];
    return f.apply(t, all);
  };
};
"#;

fn object_set_prototype_of(
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

/// `Array.prototype.values` / `Array.prototype[@@iterator]` (ES 23.1.3.41):
/// returns a fresh array-iterator over the receiver (CreateArrayIterator).
fn array_values(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    let is_array = nctx.heap().no_gc(|nogc| {
        receiver
            .as_heap_object(nogc)
            .is_some_and(|o| o.as_ref().is_array(nogc))
    });
    if !is_array {
        // Array.prototype[Symbol.iterator] called on a non-array: per spec
        // the iterator operates on any array-like via length + index gets;
        // only real arrays are supported here
        return Err(VmError::Type);
    }
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let map = heap.known().array_iterator_map;
        let zero = Smi::new(0).encode();
        Ok(heap
            .new_object(&scope, map, &[receiver, zero])
            .into_tagged()
            .erase())
    })
}

/// `%ArrayIteratorPrototype%.next` (ES 23.1.5.2.1): one step over the
/// iterated array, producing `{ value, done }`.
fn array_iterator_next(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    nctx.handle_scope(|nctx, scope| {
        let (_, heap, _) = nctx.split();
        let (array, index) = heap.no_gc(|nogc| {
            let Some(obj) = receiver.as_heap_object(nogc) else {
                return Err(VmError::Type);
            };
            let slots = obj.as_ref().slots.heap_ref(nogc);
            if slots.len() < 2 {
                return Err(VmError::Type);
            }
            Ok((slots.at(0), slots.at(1)))
        })?;
        let Some(index) = Smi::decode(index) else {
            return Err(VmError::Type);
        };
        let done = {
            let len = heap.no_gc(|nogc| {
                array
                    .as_heap_object(nogc)
                    .map(|o| o.as_ref().length())
                    .unwrap_or(0)
            });
            index.value() as usize >= len
        };
        let (value, done_value) = if done {
            (
                heap.known().undefined.value(),
                heap.known().true_object.value(),
            )
        } else {
            // element reads see holes as undefined
            let v = heap.no_gc(|nogc| {
                array
                    .as_heap_object(nogc)
                    .and_then(|o| o.as_ref().element_value(nogc, index.value() as usize))
                    .unwrap_or_else(|| nogc.known().undefined.value())
            });
            (v, heap.known().false_object.value())
        };
        // advance the index slot
        heap.no_gc(|nogc| {
            let Some(obj) = receiver.as_heap_object(nogc) else {
                return Err(VmError::Type);
            };
            let slots = obj.as_ref().slots.heap_ref(nogc);
            slots.set(nogc, 1, Smi::new(index.value() + 1).encode());
            Ok(())
        })?;
        let map = heap.known().iterator_result_map;
        Ok(heap
            .new_object(&scope, map, &[value, done_value])
            .into_tagged()
            .erase())
    })
}

/// `%ArrayIteratorPrototype%[@@iterator]`: returns the receiver.
fn array_iterator_symbol_iterator(
    _nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    args.get(0).ok_or(VmError::Arity)
}

/// `Symbol(desc)`: a fresh Symbol primitive (ES 20.4.1.1). This minimal
/// surface exists so user code can author iterables
/// (`obj[Symbol.iterator] = ...`); `Symbol.iterator` is the well-known one.
fn symbol_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let desc = args.get(1);
    let bytes = nctx.heap().no_gc(|nogc| {
        desc.and_then(|d| {
            d.get_as::<VMString>(nogc)
                .map(|s| s.as_slice(nogc).to_vec())
        })
    });
    nctx.handle_scope(|nctx, scope| {
        let mut text = b"Symbol(".to_vec();
        if let Some(d) = &bytes {
            text.extend_from_slice(d);
        }
        text.push(b')');
        Ok(Symbol::new(nctx.heap(), &scope, &text).as_tagged().erase())
    })
}

/// `Function(p0, p1, ..., body)` (ES 20.2.1.1): the dynamic constructor.
/// Builds `function (p0, p1, ...) { body }` and evaluates it in the global
/// scope (approximated with the caller's context; the direct-eval pipeline
/// provides the parsing).
fn function_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let argv: Vec<Value> = (1..args.len()).filter_map(|i| args.get(i)).collect();
    // ToString all arguments (user toString may run)
    let mut parts: Vec<String> = Vec::with_capacity(argv.len());
    for a in argv {
        let s = nctx.handle_scope(|nctx, scope| {
            let (_, heap, _) = nctx.split();
            let _ = heap;
            Convert::to_string(nctx.heap(), &scope, a)
        })?;
        parts.push(nctx.heap().no_gc(|nogc| {
            s.get_as::<VMString>(nogc)
                .map(|x| String::from_utf8_lossy(x.as_slice(nogc)).into_owned())
                .unwrap_or_default()
        }));
    }
    let (params, body) = match parts.split_last() {
        Some((body, params)) => (params.join(", "), body.clone()),
        None => (String::new(), String::new()),
    };
    let source = format!("(function ({params}) {{\n{body}\n}})");
    let context = nctx
        .current_context()
        .unwrap_or_else(|| nctx.heap().known().empty_context.value());
    let mut p = parser::Parser::new(parser::Utf8SliceStream::new(&source));
    if p.parse_script().is_err() {
        nctx.set_pending_exception(VmError::Type);
        return Ok(nctx.heap().known().exception.value());
    }
    let ast = p.into_ast();
    let compiled = match base_compiler::compile_eval(&ast) {
        Ok(c) => c,
        Err(_) => {
            nctx.set_pending_exception(VmError::Type);
            return Ok(nctx.heap().known().exception.value());
        }
    };
    nctx.handle_scope(|nctx, scope| {
        let (vm, heap, state) = nctx.split();
        let context = scope.cast::<Context>(context).ok_or(VmError::Type)?;
        let closure = materialize_closure_vm(vm, heap, state, &scope, &compiled, context)?;
        nctx.call(closure.value(), GcSlice::EMPTY)
    })
}

/// `Array.isArray(arg)` (ES 24.1.2.1).
fn array_is_array(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let arg = args.get(1).ok_or(VmError::Arity)?;
    let is_array = nctx.heap().no_gc(|nogc| {
        arg.as_heap_object(nogc)
            .is_some_and(|o| o.as_ref().is_array(nogc))
    });
    Ok(Convert::boolean(nctx.heap(), is_array))
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
    let name = get_property(vm, heap, state, receiver, "name")?;
    let message = get_property(vm, heap, state, receiver, "message")?;
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
