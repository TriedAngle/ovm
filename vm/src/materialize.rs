//! Materialization: `base_compiler::CompiledScript` → VM heap objects.
//!
//! Each compiled function becomes a `CallableInfoObject` (bytecode +
//! constants + handler table); closures reference them from the constants
//! table. Constants are converted to heap values: interned strings,
//! `Float`s, oddball singletons, and child callable infos.

use crate::{
    CallableInfoInit, CallableInfoObject, Context, FixedArray, FixedByteArray, FunctionKind,
    Handle, HandleScope, HandlerEntryInit, HandlerTable, HandlerTableInit, Heap, Object, ScopeInfo,
    ScopeInfoInit, Tagged, Value, VmError,
};

use base_compiler::{CompiledScript, Constant};
use parser::FunctionId;

use crate::DenseString;
use crate::Smi;
use crate::StringData;
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
    let slots = scope.stage(&[
        info.as_tagged(heap).erase_type(),
        context.as_tagged(heap).erase_type(),
    ]);
    let object = heap.new_object(scope, map, slots).into_handle(scope);
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

    let mut constants: Vec<Handle<'s, Value>> = Vec::with_capacity(function.constants.len());
    for constant in &function.constants {
        let value = match constant {
            Constant::String(bytes) => intern(heap, state, scope, vm, bytes).erase(),
            Constant::Smi(v) => scope.handle(Smi::new(*v)),
            Constant::Float(f) => scope.handle(heap.new_number(scope, *f)),
            Constant::Callable(child) => {
                let info = materialize_function(vm, heap, state, scope, script, infos, *child)?;
                info.erase()
            }
            Constant::ContextNames(names) => {
                let interned: Vec<Handle<'s, Value>> = names
                    .iter()
                    .map(|n| intern(heap, state, scope, vm, n).erase())
                    .collect();
                let words: Vec<Tagged<'_, Value>> =
                    interned.iter().map(|h| h.as_tagged(heap)).collect();
                let names = heap.allocate_handle::<FixedArray>(scope.stage(&words), scope);
                // one shared ScopeInfo per compiled scope; every activation
                // of the scope references it from its context
                heap.allocate_handle::<ScopeInfo>(ScopeInfoInit { names }, scope)
                    .erase()
            }
            Constant::ObjectPrototype => heap.known().object_prototype.erase(),
            Constant::FunctionPrototype => heap.known().function_prototype.erase(),
        };
        constants.push(value);
    }

    let bytecode = heap.allocate_handle::<FixedByteArray>(&function.bytecode, scope);
    let words: Vec<Tagged<'_, Value>> = constants.iter().map(|h| h.as_tagged(heap)).collect();
    let constants = heap.allocate_handle::<FixedArray>(scope.stage(&words), scope);
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
    let name: Option<Handle<'s, DenseString>> = function
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
    heap.no_gc(|heap| {
        info.heap_ref(heap).set_metadata_full(
            heap,
            name.map(|h| h.as_tagged(heap).erase_type()),
            function.formal_parameter_count as usize,
            function.formal_length as usize,
            kind,
            function.strict,
        );
    });
    infos[fid.0 as usize] = Some(info);
    Ok(info)
}

fn intern<'s>(
    heap: &mut Heap,
    _state: &ContextState,
    scope: &'s HandleScope<'_>,
    vm: &VM,
    s: &[u8],
) -> Handle<'s, DenseString> {
    let units = crate::decode_wtf8(s).expect("parser produces valid WTF-8 string constants");
    vm.interner().intern(heap, scope, StringData::Utf16(&units))
}
