//! Minimal builtin library: the globals the test262 harness needs.
//!
//! Installed once per VM, before any user code runs: `Number`, `Boolean`,
//! `Error`, `TypeError`, `eval`, plus the wrapper/error maps in
//! `WellKnown` and `.prototype`/`.constructor` plumbing for user
//! functions (interpreter `CreateClosure`).

pub mod array;
pub mod boolean;
pub mod error;
pub mod function;
pub mod global;
pub mod helpers;
pub mod intrinsics;
pub mod number;
pub mod object;
pub mod proxy;
pub mod string;
pub mod symbol;

use array::{
    array_constructor, array_is_array, array_iterator_next, array_iterator_symbol_iterator,
    array_values,
};
use boolean::{boolean_constructor, boolean_to_string, boolean_value_of};
use error::{
    error_constructor, error_to_string, reference_error_constructor, type_error_constructor,
};
use function::{
    BIND_PRELUDE, function_apply, function_bind, function_call, function_constructor,
    function_to_string,
};
use global::{eval_runtime, is_nan};
use helpers::{
    alloc_map, alloc_map_with_slots, define_data, define_method_prop, define_non_enumerable,
    install_constructor, install_method, make_runtime_function, make_runtime_plain_function,
    run_prelude,
};
use number::{number_constructor, number_to_string, number_value_of};
use object::{
    object_constructor, object_define_property, object_freeze, object_get_own_property_descriptor,
    object_get_own_property_names, object_get_prototype_of, object_has_own_property,
    object_is_extensible, object_prevent_extensions, object_property_is_enumerable, object_seal,
    object_set_prototype_of, object_to_string,
};
use proxy::{REVOKE_PRELUDE, proxy_constructor, proxy_revocable, proxy_revoke};
use string::{string_constructor, string_to_string, string_value_of};
use symbol::symbol_constructor;

use crate::{
    Handle, HandleSlice, Map, MapInit, MapKind, Object, PropertyDescriptor, SlotFlags, SlotName,
    Smi, Tagged, Value, VmError,
};

use crate::Float;
use crate::RuntimeIndex;
use crate::VM;

/// Register the builtin runtimes.
pub fn register_builtin_runtimes(vm: &mut VM) -> BuiltinIndices {
    BuiltinIndices {
        eval: vm.register_runtime(eval_runtime),
        string: vm.register_runtime(string_constructor),
        string_value_of: vm.register_runtime(string_value_of),
        string_to_string: vm.register_runtime(string_to_string),
        reference_error: vm.register_runtime(reference_error_constructor),
        function_to_string: vm.register_runtime(function_to_string),
        object_to_string: vm.register_runtime(object_to_string),
        number: vm.register_runtime(number_constructor),
        number_value_of: vm.register_runtime(number_value_of),
        number_to_string: vm.register_runtime(number_to_string),
        boolean: vm.register_runtime(boolean_constructor),
        boolean_value_of: vm.register_runtime(boolean_value_of),
        boolean_to_string: vm.register_runtime(boolean_to_string),
        error: vm.register_runtime(error_constructor),
        type_error: vm.register_runtime(type_error_constructor),
        error_to_string: vm.register_runtime(error_to_string),
        object: vm.register_runtime(object_constructor),
        object_get_prototype_of: vm.register_runtime(object_get_prototype_of),
        object_set_prototype_of: vm.register_runtime(object_set_prototype_of),
        array: vm.register_runtime(array_constructor),
        is_nan: vm.register_runtime(is_nan),
        array_values: vm.register_runtime(array_values),
        array_iterator_next: vm.register_runtime(array_iterator_next),
        array_iterator_symbol_iterator: vm.register_runtime(array_iterator_symbol_iterator),
        symbol: vm.register_runtime(symbol_constructor),
        object_has_own_property: vm.register_runtime(object_has_own_property),
        object_property_is_enumerable: vm.register_runtime(object_property_is_enumerable),
        object_get_own_property_names: vm.register_runtime(object_get_own_property_names),
        object_get_own_property_descriptor: vm.register_runtime(object_get_own_property_descriptor),
        object_define_property: vm.register_runtime(object_define_property),
        function_call: vm.register_runtime(function_call),
        function_apply: vm.register_runtime(function_apply),
        function_bind: vm.register_runtime(function_bind),
        function_constructor: vm.register_runtime(function_constructor),
        array_is_array: vm.register_runtime(array_is_array),
        proxy: vm.register_runtime(proxy_constructor),
        proxy_revocable: vm.register_runtime(proxy_revocable),
        proxy_revoke: vm.register_runtime(proxy_revoke),
        object_prevent_extensions: vm.register_runtime(object_prevent_extensions),
        object_is_extensible: vm.register_runtime(object_is_extensible),
        object_seal: vm.register_runtime(object_seal),
        object_freeze: vm.register_runtime(object_freeze),
    }
}

pub struct BuiltinIndices {
    pub eval: RuntimeIndex,
    pub string: RuntimeIndex,
    pub string_value_of: RuntimeIndex,
    pub string_to_string: RuntimeIndex,
    pub reference_error: RuntimeIndex,
    pub function_to_string: RuntimeIndex,
    pub object_to_string: RuntimeIndex,
    pub number: RuntimeIndex,
    pub number_value_of: RuntimeIndex,
    pub number_to_string: RuntimeIndex,
    pub boolean: RuntimeIndex,
    pub boolean_value_of: RuntimeIndex,
    pub boolean_to_string: RuntimeIndex,
    pub error: RuntimeIndex,
    pub type_error: RuntimeIndex,
    pub error_to_string: RuntimeIndex,
    pub object: RuntimeIndex,
    pub object_get_prototype_of: RuntimeIndex,
    pub object_set_prototype_of: RuntimeIndex,
    pub array: RuntimeIndex,
    pub is_nan: RuntimeIndex,
    pub array_values: RuntimeIndex,
    pub array_iterator_next: RuntimeIndex,
    pub array_iterator_symbol_iterator: RuntimeIndex,
    pub symbol: RuntimeIndex,
    pub object_has_own_property: RuntimeIndex,
    pub object_property_is_enumerable: RuntimeIndex,
    pub object_get_own_property_names: RuntimeIndex,
    pub object_get_own_property_descriptor: RuntimeIndex,
    pub object_define_property: RuntimeIndex,
    pub function_call: RuntimeIndex,
    pub function_apply: RuntimeIndex,
    pub function_bind: RuntimeIndex,
    pub function_constructor: RuntimeIndex,
    pub array_is_array: RuntimeIndex,
    pub proxy: RuntimeIndex,
    pub proxy_revocable: RuntimeIndex,
    pub proxy_revoke: RuntimeIndex,
    pub object_prevent_extensions: RuntimeIndex,
    pub object_is_extensible: RuntimeIndex,
    pub object_seal: RuntimeIndex,
    pub object_freeze: RuntimeIndex,
}

/// Build the builtin objects and install them on the global object.
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
        let pos_inf = roots.create_handle(thread.heap().allocate::<Float>(f64::INFINITY));
        let neg_inf = roots.create_handle(thread.heap().allocate::<Float>(f64::NEG_INFINITY));
        let max_value = roots.create_handle(thread.heap().allocate::<Float>(f64::MAX));
        let min_value = roots.create_handle(thread.heap().allocate::<Float>(f64::MIN_POSITIVE));
        let number_nan = roots.create_handle(thread.heap().allocate::<Float>(f64::NAN));
        for (name, value) in [
            ("POSITIVE_INFINITY", pos_inf),
            ("NEGATIVE_INFINITY", neg_inf),
            ("MAX_VALUE", max_value),
            ("MIN_VALUE", min_value),
            ("NaN", number_nan),
        ] {
            let n = thread.intern(&scope, name);
            // Safety: fresh interned word, rooted below before the define.
            let n = scope.handle(n.as_tagged(&*thread.heap()));
            define_data(thread.heap(), &scope, number_fn, n, value.erase())?;
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
            error_name.erase(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            error_proto,
            known.strings.message,
            known.strings.empty.erase(),
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
            type_error_name.erase(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            type_error_proto,
            wks.message,
            wks.empty.erase(),
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
            reference_error_name.erase(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            reference_error_proto,
            wks.message,
            wks.empty.erase(),
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
        let function_fn = make_runtime_function(thread, &scope, roots, idx.function_constructor)?;
        define_method_prop(
            thread.heap(),
            &scope,
            function_prototype,
            wks.constructor,
            function_fn.erase(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            function_fn,
            wks.prototype,
            function_prototype.erase(),
        )?;
        let function_name = thread.intern(&scope, "Function");
        // Safety: fresh interned word, rooted below before the define.
        let function_name = scope.handle(function_name.as_tagged(&*thread.heap()));
        let global_object = thread.heap().known().global_object;
        define_data(
            thread.heap(),
            &scope,
            global_object,
            function_name,
            function_fn.erase(),
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
        run_prelude(thread, &scope, BIND_PRELUDE, "bind prelude")?;
        // likewise the proxy revoke closure (see REVOKE_PRELUDE)
        run_prelude(thread, &scope, REVOKE_PRELUDE, "revoke prelude")?;

        // ---- eval -------------------------------------------------------------
        let eval_fn = make_runtime_function(thread, &scope, roots, idx.eval)?;
        let global = thread.heap().known().global_object;
        let eval_name = thread.intern(&scope, "eval");
        // Safety: fresh interned word, rooted below before the define.
        let eval_name = scope.handle(eval_name.as_tagged(&*thread.heap()));
        define_data(thread.heap(), &scope, global, eval_name, eval_fn.erase())?;

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
        let object_fn = make_runtime_function(thread, &scope, roots, idx.object)?;
        define_method_prop(
            thread.heap(),
            &scope,
            object_prototype,
            wks.constructor,
            object_fn.erase(),
        )?;
        define_method_prop(
            thread.heap(),
            &scope,
            object_fn,
            wks.prototype,
            object_prototype.erase(),
        )?;
        let object_name = thread.intern(&scope, "Object");
        // Safety: fresh interned word, rooted below before the define.
        let object_name = scope.handle(object_name.as_tagged(&*thread.heap()));
        define_data(
            thread.heap(),
            &scope,
            global,
            object_name,
            object_fn.erase(),
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
        let array_fn = make_runtime_function(thread, &scope, roots, idx.array)?;
        let array_prototype = thread.heap().known().array_prototype;
        define_method_prop(
            thread.heap(),
            &scope,
            array_prototype,
            wks.constructor,
            array_fn.erase(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            array_fn,
            wks.prototype,
            array_prototype.erase(),
        )?;
        let array_name = thread.intern(&scope, "Array");
        // Safety: fresh interned word, rooted below before the define.
        let array_name = scope.handle(array_name.as_tagged(&*thread.heap()));
        define_data(thread.heap(), &scope, global, array_name, array_fn.erase())?;

        // Array.isArray
        let is_array_fn = make_runtime_function(thread, &scope, roots, idx.array_is_array)?;
        let is_array_name = thread.intern(&scope, "isArray");
        // Safety: fresh interned word, rooted below before the define.
        let is_array_name = scope.handle(is_array_name.as_tagged(&*thread.heap()));
        define_data(
            thread.heap(),
            &scope,
            array_fn,
            is_array_name,
            is_array_fn.erase(),
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
            roots.create_handle(thread.heap().new_object(&scope, map, HandleSlice::EMPTY))
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
            make_runtime_function(thread, &scope, roots, idx.array_iterator_symbol_iterator)?;
        let iterator_symbol = thread.heap().known().iterator_symbol;
        // Safety: fresh root-slot word, rooted below before the defines.
        let iterator_name = scope.handle(iterator_symbol.as_tagged(&*thread.heap()));
        define_method_prop(
            thread.heap(),
            &scope,
            array_iterator_prototype,
            iterator_name,
            sym_iterator_iter.erase(),
        )?;

        // Array.prototype.values === Array.prototype[Symbol.iterator]: a
        // runtime returning a fresh array-iterator object
        let values_fn = make_runtime_function(thread, &scope, roots, idx.array_values)?;
        let values_name = thread.intern(&scope, "values");
        // Safety: fresh interned word, rooted below before the defines.
        let values_name = scope.handle(values_name.as_tagged(&*thread.heap()));
        define_method_prop(
            thread.heap(),
            &scope,
            array_prototype,
            values_name,
            values_fn.erase(),
        )?;
        define_method_prop(
            thread.heap(),
            &scope,
            array_prototype,
            iterator_name,
            values_fn.erase(),
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
            // Safety: fresh root-slot word, rooted below before any allocation.
            let proto = scope.handle(unsafe {
                Tagged::<Value>::from_value_unchecked(object_prototype.read_unchecked())
            });
            thread
                .heap()
                .allocate_token_enter_heap(Map::layout_for(2), |token, heap| {
                    // Safety: fresh interned words re-read under the anchor.
                    let descriptors: [(Handle<'_, SlotName>, SlotFlags, Handle<'_, Value>); 2] = [
                        (
                            scope.handle(value_name.as_tagged(heap).erase().as_name()),
                            flags,
                            scope.handle(Smi::new(0).into_tagged()),
                        ),
                        (
                            scope.handle(done_name.as_tagged(heap).erase().as_name()),
                            flags,
                            scope.handle(Smi::new(1).into_tagged()),
                        ),
                    ];
                    roots.create_handle(token.allocate::<Map>(MapInit {
                        kind: MapKind::OBJECT.union(MapKind::EXTENDABLE),
                        value_slot_count: 2,
                        descriptors: &descriptors,
                        prototype: proto,
                    }))
                })
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

        let is_nan_fn = make_runtime_function(thread, &scope, roots, idx.is_nan)?;
        let is_nan_name = thread.intern(&scope, "isNaN");
        // Safety: fresh interned word, rooted below before the define.
        let is_nan_name = scope.handle(is_nan_name.as_tagged(&*thread.heap()));
        define_data(
            thread.heap(),
            &scope,
            global,
            is_nan_name,
            is_nan_fn.erase(),
        )?;

        // ---- Symbol (minimal: constructor + Symbol.iterator) ---------------
        // enough to author custom iterables; the full Symbol surface stays
        // gated by the test262 feature skip
        let symbol_fn = make_runtime_function(thread, &scope, roots, idx.symbol)?;
        let symbol_name = thread.intern(&scope, "Symbol");
        // Safety: fresh interned word, rooted below before the define.
        let symbol_name = scope.handle(symbol_name.as_tagged(&*thread.heap()));
        define_data(
            thread.heap(),
            &scope,
            global,
            symbol_name,
            symbol_fn.erase(),
        )?;
        let iterator_symbol = thread.heap().known().iterator_symbol;
        let iter_name = thread.intern(&scope, "iterator");
        // Safety: fresh interned word, rooted below before the define.
        let iter_name = scope.handle(iter_name.as_tagged(&*thread.heap()));
        define_data(
            thread.heap(),
            &scope,
            symbol_fn,
            iter_name,
            iterator_symbol.erase(),
        )?;

        // ---- Proxy ------------------------------------------------------------
        // The Proxy constructor is a runtime function *without* a
        // `.prototype` property (ES 20.2.1: "Proxy.prototype is
        // undefined"); `install_constructor` cannot be used.
        let proxy_fn = make_runtime_function(thread, &scope, roots, idx.proxy)?;
        let two = scope.handle(Smi::new(2));
        define_non_enumerable(thread.heap(), &scope, proxy_fn, wks.length, two)?;
        let proxy_name = thread.intern(&scope, "Proxy");
        define_non_enumerable(
            thread.heap(),
            &scope,
            proxy_fn,
            wks.name,
            proxy_name.erase(),
        )?;
        // Safety: fresh interned word, rooted below before the define.
        let proxy_name = scope.handle(proxy_name.as_tagged(&*thread.heap()));
        define_data(thread.heap(), &scope, global, proxy_name, proxy_fn.erase())?;
        // Proxy.revocable: a non-constructor function returning
        // { proxy, revoke }; the revoke closure is the JS template
        // installed by REVOKE_PRELUDE (runtimes cannot carry state).
        let revocable_fn = make_runtime_plain_function(thread, &scope, roots, idx.proxy_revocable)?;
        define_non_enumerable(thread.heap(), &scope, revocable_fn, wks.length, two)?;
        let revocable_name = thread.intern(&scope, "revocable");
        define_non_enumerable(
            thread.heap(),
            &scope,
            revocable_fn,
            wks.name,
            revocable_name.erase(),
        )?;
        // Safety: fresh interned word, rooted below before the define.
        let revocable_name = scope.handle(revocable_name.as_tagged(&*thread.heap()));
        define_data(
            thread.heap(),
            &scope,
            proxy_fn,
            revocable_name,
            revocable_fn.erase(),
        )?;
        // hidden revoke runtime used by the REVOKE_PRELUDE closure
        let revoke_fn = make_runtime_plain_function(thread, &scope, roots, idx.proxy_revoke)?;
        let revoke_name = thread.intern(&scope, "__revokeProxy");
        // Safety: fresh interned word, rooted below before the define.
        let revoke_name = scope.handle(revoke_name.as_tagged(&*thread.heap()));
        define_data(
            thread.heap(),
            &scope,
            global,
            revoke_name,
            revoke_fn.erase(),
        )?;

        // ---- Object extensibility statics --------------------------------------
        install_method(
            thread,
            &scope,
            roots,
            object_fn,
            "preventExtensions",
            idx.object_prevent_extensions,
        )?;
        install_method(
            thread,
            &scope,
            roots,
            object_fn,
            "isExtensible",
            idx.object_is_extensible,
        )?;
        install_method(thread, &scope, roots, object_fn, "seal", idx.object_seal)?;
        install_method(
            thread,
            &scope,
            roots,
            object_fn,
            "freeze",
            idx.object_freeze,
        )?;

        // ---- value properties of the global object -----------------------------
        let infinity = roots.create_handle(thread.heap().allocate::<Float>(f64::INFINITY));
        let nan = roots.create_handle(thread.heap().allocate::<Float>(f64::NAN));
        let undefined = thread.heap().known().undefined;
        for name in ["Infinity", "NaN", "undefined"] {
            let n = thread.intern(&scope, name);
            // spec attributes are {writable: false, enumerable: false,
            // configurable: false} (ES 19.1.1); non-configurability is
            // what `delete NaN` observes, writable/enumerable stay per
            //missive until non-writable stores stop throwing in sloppy
            // code
            let value = match name {
                "Infinity" => infinity.erase(),
                "NaN" => nan.erase(),
                _ => undefined.erase(),
            };
            // Safety: fresh root-slot words, rooted below before any
            // allocation.
            let receiver = scope
                .handle(unsafe { Tagged::<Object>::from_value_unchecked(global.read_unchecked()) });
            let key = scope.handle(n.as_tagged(&*thread.heap()));
            Object::define_own_property(
                thread.heap(),
                &scope,
                receiver,
                key,
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
