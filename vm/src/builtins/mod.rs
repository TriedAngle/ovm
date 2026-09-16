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

use array::{array_constructor, array_is_array, array_iterator_next, array_iterator_symbol_iterator, array_values};
use boolean::{boolean_constructor, boolean_to_string, boolean_value_of};
use error::{error_constructor, error_to_string, reference_error_constructor, type_error_constructor};
use function::{BIND_PRELUDE, function_apply, function_bind, function_call, function_constructor, function_to_string};
use global::{eval_native, is_nan};
use helpers::{
    alloc_map, alloc_map_with_slots, define_data, define_method_prop, define_non_enumerable,
    install_constructor, install_method, make_native_function, make_native_plain_function, run_prelude,
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
    Map, MapInit, MapKind,
    Object, PropertyDescriptor, SlotFlags, SlotName, Smi, VmError,
};

use crate::natives::NativeIndex;
use crate::VM;

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
        proxy: vm.register_native(proxy_constructor),
        proxy_revocable: vm.register_native(proxy_revocable),
        proxy_revoke: vm.register_native(proxy_revoke),
        object_prevent_extensions: vm.register_native(object_prevent_extensions),
        object_is_extensible: vm.register_native(object_is_extensible),
        object_seal: vm.register_native(object_seal),
        object_freeze: vm.register_native(object_freeze),
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
    pub proxy: NativeIndex,
    pub proxy_revocable: NativeIndex,
    pub proxy_revoke: NativeIndex,
    pub object_prevent_extensions: NativeIndex,
    pub object_is_extensible: NativeIndex,
    pub object_seal: NativeIndex,
    pub object_freeze: NativeIndex,
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
        run_prelude(thread, &scope, BIND_PRELUDE, "bind prelude")?;
        // likewise the proxy revoke closure (see REVOKE_PRELUDE)
        run_prelude(thread, &scope, REVOKE_PRELUDE, "revoke prelude")?;

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

        // ---- Proxy ------------------------------------------------------------
        // The Proxy constructor is a native function *without* a
        // `.prototype` property (ES 20.2.1: "Proxy.prototype is
        // undefined"); `install_constructor` cannot be used.
        let proxy_fn = make_native_function(thread, &scope, roots, idx.proxy)?;
        let two = Smi::new(2).encode();
        define_non_enumerable(thread.heap(), &scope, proxy_fn, wks.length, two)?;
        let proxy_name = thread.intern(&scope, "Proxy");
        define_non_enumerable(
            thread.heap(),
            &scope,
            proxy_fn,
            wks.name,
            proxy_name.value(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            global,
            SlotName::from(proxy_name.as_tagged()),
            proxy_fn.value(),
        )?;
        // Proxy.revocable: a non-constructor function returning
        // { proxy, revoke }; the revoke closure is the JS template
        // installed by REVOKE_PRELUDE (natives cannot carry state).
        let revocable_fn = make_native_plain_function(thread, &scope, roots, idx.proxy_revocable)?;
        define_non_enumerable(thread.heap(), &scope, revocable_fn, wks.length, two)?;
        let revocable_name = thread.intern(&scope, "revocable");
        define_non_enumerable(
            thread.heap(),
            &scope,
            revocable_fn,
            wks.name,
            revocable_name.value(),
        )?;
        define_data(
            thread.heap(),
            &scope,
            proxy_fn,
            SlotName::from(revocable_name.as_tagged()),
            revocable_fn.value(),
        )?;
        // hidden revoke native used by the REVOKE_PRELUDE closure
        let revoke_fn = make_native_plain_function(thread, &scope, roots, idx.proxy_revoke)?;
        let revoke_name = thread.intern(&scope, "__revokeProxy");
        define_data(
            thread.heap(),
            &scope,
            global,
            SlotName::from(revoke_name.as_tagged()),
            revoke_fn.value(),
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
