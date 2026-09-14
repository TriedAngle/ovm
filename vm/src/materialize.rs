//! Materialization: `base_compiler::CompiledScript` → VM heap objects.
//!
//! Each compiled function becomes a `CallableInfoObject` (bytecode +
//! constants + handler table); closures reference them from the constants
//! table. Constants are converted to heap values: interned strings,
//! `Float`s, oddball singletons, and child callable infos.

use crate::{
    CallableInfoInit, CallableInfoObject, Context, FixedArray, FixedByteArray, FunctionKind,
    Handle, HandleScope, HandlerEntryInit, HandlerTable, HandlerTableInit, Heap, Object, ScopeInfo,
    ScopeInfoInit, VmError,
};

use base_compiler::{CompiledScript, Constant};
use parser::FunctionId;

use crate::{ContextState, Thread, VM};

/// Materialize a compiled script into a closure object (function map,
/// empty context) ready for `Thread::execute`.
pub fn materialize_script<'s>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    script: &CompiledScript,
) -> Result<Handle<'s, Object>, VmError> {
    let empty = thread.heap().known().empty_context;
    materialize_closure(thread, scope, script, empty)
}

pub fn materialize_closure<'s, 'c>(
    thread: &mut Thread,
    scope: &'s HandleScope<'_>,
    script: &CompiledScript,
    context: Handle<'c, Context>,
) -> Result<Handle<'s, Object>, VmError>
where
    'c: 's,
{
    let (vm, heap, state) = thread.split();
    materialize_closure_vm(vm, heap, state, scope, script, context)
}

pub fn materialize_closure_vm<'s>(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &'s HandleScope<'_>,
    script: &CompiledScript,
    context: Handle<'s, Context>,
) -> Result<Handle<'s, Object>, VmError> {
    let mut infos: Vec<Option<Handle<'s, CallableInfoObject>>> =
        (0..script.functions.len()).map(|_| None).collect();
    let info = materialize_function(vm, heap, state, scope, script, &mut infos, FunctionId(0))?;

    let map = heap.known().function_map;
    let object = heap
        .new_object(scope, map, &[info.value(), context.value()])
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
            Constant::Smi(v) => crate::Smi::new(*v).encode(),
            Constant::Float(f) => heap.new_number(scope, *f),
            Constant::Boolean(true) => heap.known().true_object.value(),
            Constant::Boolean(false) => heap.known().false_object.value(),
            Constant::Undefined => heap.known().undefined.value(),
            Constant::Null => heap.known().null.value(),
            Constant::Callable(child) => {
                let info = materialize_function(vm, heap, state, scope, script, infos, *child)?;
                info.value()
            }
            Constant::ContextNames(names) => {
                let interned: Vec<crate::Value> = names
                    .iter()
                    .map(|n| intern(heap, state, scope, vm, n))
                    .collect();
                let names = heap.allocate_handle::<FixedArray>(&interned, scope);
                // one shared ScopeInfo per compiled scope; every activation
                // of the scope references it from its context
                heap.allocate_handle::<ScopeInfo>(ScopeInfoInit { names }, scope)
                    .value()
            }
            Constant::ObjectPrototype => heap.known().object_prototype.value(),
            Constant::FunctionPrototype => heap.known().function_prototype.value(),
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
        parser::FunctionKind::DefaultDerivedConstructor => FunctionKind::DefaultDerivedConstructor,
    };
    heap.no_gc(|nogc| {
        info.heap_ref(nogc).set_metadata_full(
            nogc,
            name,
            function.formal_parameter_count as usize,
            function.formal_length as usize,
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
) -> crate::Value {
    // constant strings are WTF-8 (lone surrogates as the 3-byte pattern);
    // the interner is byte-based, so they round-trip without loss
    vm.interner().intern(heap, scope, s).value()
}
