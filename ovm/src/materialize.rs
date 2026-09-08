//! Materialization: `base_compiler::CompiledScript` → VM heap objects.
//!
//! Each compiled function becomes a `CallableInfoObject` (bytecode +
//! constants + handler table); closures reference them from the constants
//! table. Constants are converted to heap values: interned strings,
//! `Float`s, oddball singletons, and child callable infos.

use base_compiler::{CompiledScript, Constant};
use parser::FunctionId;
use vm::{
    CallableInfoInit, CallableInfoObject, Context, FixedArray, FixedByteArray, Float, FunctionKind,
    Handle, HandleScope, HandlerEntryInit, HandlerTable, HandlerTableInit, Heap, Object,
    ObjectSlotsInit, ScopeInfo, ScopeInfoInit, Tagged, VmError,
};

use crate::{ContextState, Thread, VM};

/// Materialize a compiled script into a closure object (function map,
/// empty context) ready for `Thread::execute`.
pub fn materialize_script<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    script: &CompiledScript,
) -> Result<Handle<'s, Object>, VmError> {
    let empty = thread.heap().known().empty_context.value();
    materialize_closure(thread, scope, script, empty)
}

/// Materialize a compiled script whose closure captures `context` instead
/// of the empty context (direct eval: the caller's frame context).
pub fn materialize_closure<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    script: &CompiledScript,
    context: vm::Value,
) -> Result<Handle<'s, Object>, VmError> {
    let (vm, heap, state) = thread.split();
    materialize_closure_vm(vm, heap, state, scope, script, context)
}

/// Native-facing variant (the eval builtin has no `Thread`).
pub fn materialize_closure_vm<'s>(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &'s HandleScope<'_>,
    script: &CompiledScript,
    context: vm::Value,
) -> Result<Handle<'s, Object>, VmError> {
    let mut infos: Vec<Option<Handle<'s, CallableInfoObject>>> =
        (0..script.functions.len()).map(|_| None).collect();
    let info = materialize_function(vm, heap, state, scope, script, &mut infos, FunctionId(0))?;

    let map = scope
        .create_handle(heap.known().function_map.as_tagged())
        .expect("function map is strong");
    let context = scope
        .create_handle(unsafe { Tagged::<Context>::from_value_unchecked(context) })
        .expect("context is strong");
    let empty_elements = heap.known().empty_fixed_array.erase();
    let object = heap
        .allocate_object(
            scope,
            ObjectSlotsInit {
                map,
                values: &[info.as_tagged().erase(), context.as_tagged().erase()],
                elements: empty_elements,
                length: 0,
            },
        )
        .into_handle(scope);
    Ok(object)
}

fn materialize_function<'s>(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &'s HandleScope<'_>,
    script: &CompiledScript,
    infos: &mut [Option<Handle<'s, CallableInfoObject>>],
    fid: FunctionId,
) -> Result<Handle<'s, CallableInfoObject>, VmError> {
    if let Some(info) = infos[fid.0 as usize] {
        return Ok(info);
    }
    let function = &script.functions[fid.0 as usize];

    let mut constants = Vec::with_capacity(function.constants.len());
    for constant in &function.constants {
        let value = match constant {
            Constant::String(bytes) => intern(heap, state, scope, vm, bytes),
            Constant::Smi(v) => vm::Smi::new(*v).encode(),
            Constant::Float(f) => heap.allocate_handle::<Float>(*f, scope).value(),
            Constant::Boolean(true) => heap.known().true_object.value(),
            Constant::Boolean(false) => heap.known().false_object.value(),
            Constant::Undefined => heap.known().undefined.value(),
            Constant::Null => heap.known().null.value(),
            Constant::Callable(child) => {
                let info = materialize_function(vm, heap, state, scope, script, infos, *child)?;
                info.as_tagged().erase()
            }
            Constant::ContextNames(names) => {
                let interned: Vec<vm::Value> = names
                    .iter()
                    .map(|n| intern(heap, state, scope, vm, n))
                    .collect();
                let names = heap.allocate_handle::<FixedArray>(&interned, scope);
                // one shared ScopeInfo per compiled scope; every activation
                // of the scope references it from its context
                heap.allocate_handle::<ScopeInfo>(ScopeInfoInit { names }, scope)
                    .value()
            }
        };
        constants.push(value);
    }

    let bytecode = heap.allocate_handle::<FixedByteArray>(&function.bytecode, scope);
    let constants = heap.allocate_handle::<FixedArray>(&constants, scope);
    let handlers = if function.handlers.is_empty() {
        None
    } else {
        let entries: Vec<HandlerEntryInit> = function
            .handlers
            .iter()
            .map(|h| HandlerEntryInit::new(h.try_start, h.try_end, h.handler_pc))
            .collect();
        Some(heap.allocate_handle::<HandlerTable>(HandlerTableInit { entries: &entries }, scope))
    };

    let info = heap.allocate_handle::<CallableInfoObject>(
        CallableInfoInit {
            bytecode,
            constants,
            register_count: function.register_count as usize,
            handlers,
        },
        scope,
    );
    let name = function
        .name
        .as_deref()
        .map(|name| intern(heap, state, scope, vm, name));
    let kind = match function.kind {
        parser::FunctionKind::Normal => FunctionKind::Normal,
        parser::FunctionKind::Generator => FunctionKind::Generator,
        parser::FunctionKind::Arrow => FunctionKind::Arrow,
        parser::FunctionKind::Method => FunctionKind::Method,
        parser::FunctionKind::Getter => FunctionKind::Getter,
        parser::FunctionKind::Setter => FunctionKind::Setter,
        parser::FunctionKind::BaseClassConstructor => FunctionKind::BaseClassConstructor,
        parser::FunctionKind::DerivedClassConstructor => FunctionKind::DerivedClassConstructor,
    };
    heap.no_gc(|nogc| {
        info.heap_ref(nogc).set_metadata(
            nogc,
            name,
            function.formal_parameter_count as usize,
            kind,
            function.strict,
        );
    });
    infos[fid.0 as usize] = Some(info);
    Ok(info)
}

fn intern(
    heap: &mut Heap,
    _state: &ContextState,
    scope: &HandleScope<'_>,
    vm: &VM,
    s: &[u8],
) -> vm::Value {
    // constant strings are WTF-8 (lone surrogates as the 3-byte pattern);
    // the interner is byte-based, so they round-trip without loss
    vm.interner().intern(heap, scope, s).as_tagged().erase()
}
