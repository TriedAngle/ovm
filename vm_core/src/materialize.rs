//! Materialization: `bytecode::Program` → VM heap objects.
//!
//! Each compiled function becomes a `CallableInfoObject` (bytecode +
//! constants + handler table); closures reference them from the constants
//! table. Constants are converted to heap values: interned strings,
//! `Float`s, oddball singletons, and child callable infos.
//!
//! The materializer is frontend-agnostic: it consumes the compiled program
//! function table and knows nothing about the language that produced it.

use crate::{
    CallableInfoInit, CallableInfoObject, Context, FixedArray, FixedByteArray, FunctionKind,
    Handle, HandleScope, HandlerEntryInit, HandlerTable, HandlerTableInit, Heap, Object, ScopeInfo,
    ScopeInfoInit, Tagged, Value, VmError, decode_wtf8, new_feedback_vector,
};

use bytecode::{CallableKind, Constant, FunctionId, Program};

use crate::DenseString;
use crate::Smi;
use crate::StringData;
use crate::{ContextState, Thread, VM};

/// Namespace for turning compiled programs into VM heap objects.
pub struct Materialize;

impl Materialize {
    /// Materialize a compiled program into a closure object (function map,
    /// empty context) ready for `Thread::execute`.
    pub fn script<'s>(
        thread: &mut Thread,
        scope: &'s HandleScope<'_>,
        program: &Program,
    ) -> Result<Handle<'s, Object>, VmError> {
        let empty = thread.heap().known().empty_context;
        Self::closure(thread, scope, program, empty)
    }

    pub fn closure<'s, 'c>(
        thread: &mut Thread,
        scope: &'s HandleScope<'_>,
        program: &Program,
        context: Handle<'c, Context>,
    ) -> Result<Handle<'s, Object>, VmError>
    where
        'c: 's,
    {
        let (vm, heap, state) = thread.split();
        Self::closure_vm(vm, heap, state, scope, program, context)
    }

    pub fn closure_vm<'s>(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        scope: &'s HandleScope<'_>,
        program: &Program,
        context: Handle<'s, Context>,
    ) -> Result<Handle<'s, Object>, VmError> {
        let _span = trace::info_span!("vm::materialize").entered();
        let mut infos: Vec<Option<Handle<'s, CallableInfoObject>>> =
            (0..program.len()).map(|_| None).collect();
        let info = materialize_function(
            vm,
            heap,
            state,
            scope,
            program,
            &mut infos,
            FunctionId::SCRIPT,
        )?;

        let map = heap.known().function_map;
        let slots = scope.stage(&[
            info.as_tagged(heap).erase(),
            context.as_tagged(heap).erase(),
        ]);
        let object = heap.new_object(scope, map, slots).as_handle(scope);
        Ok(object)
    }
}

fn materialize_function<'s>(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &'s HandleScope<'_>,
    program: &Program,
    infos: &mut [Option<Handle<'s, CallableInfoObject>>],
    fid: FunctionId,
) -> Result<Handle<'s, CallableInfoObject>, VmError> {
    if let Some(info) = infos[fid.index()] {
        return Ok(info);
    }
    let _span = trace::debug_span!("vm::materialize_function", fid = fid.0).entered();
    let function = program.function(fid);

    let constants = program.constants(function);
    let mut materialized: Vec<Handle<'s, Value>> = Vec::with_capacity(constants.len());
    for constant in constants {
        let value = match constant {
            Constant::String(bytes) => intern(heap, state, scope, vm, bytes).erase(),
            Constant::Smi(v) => scope.handle(Smi::new(*v)),
            Constant::Float(f) => scope.handle(heap.new_number(*f)),
            Constant::Callable(child) => {
                let info = materialize_function(vm, heap, state, scope, program, infos, *child)?;
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
        materialized.push(value);
    }

    let bytecode = heap.allocate_handle::<FixedByteArray>(program.code(function), scope);
    let words: Vec<Tagged<'_, Value>> = materialized.iter().map(|h| h.as_tagged(heap)).collect();
    let constants = heap.allocate_handle::<FixedArray>(scope.stage(&words), scope);
    let handlers = if program.handlers(function).is_empty() {
        None
    } else {
        let entries: Vec<HandlerEntryInit> = program
            .handlers(function)
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
    if let Some(feedback) = new_feedback_vector(heap, scope, function.feedback_count as usize) {
        info.as_tagged(heap).set_feedback(heap, feedback);
    }
    let name: Option<Handle<'s, DenseString>> = program
        .name(function)
        .map(|name| intern(heap, state, scope, vm, name));
    let kind = match function.kind {
        CallableKind::Normal => FunctionKind::Normal,
        CallableKind::Generator => FunctionKind::Generator,
        CallableKind::Arrow => FunctionKind::Arrow,
        CallableKind::Method => FunctionKind::Method,
        CallableKind::Getter => FunctionKind::Getter,
        CallableKind::Setter => FunctionKind::Setter,
        CallableKind::BaseClassConstructor => FunctionKind::BaseClassConstructor,
        CallableKind::DerivedClassConstructor => FunctionKind::DerivedClassConstructor,
        CallableKind::DefaultDerivedConstructor => FunctionKind::DefaultDerivedConstructor,
    };
    info.as_tagged(heap).set_metadata_full(
        heap,
        name.map(|h| h.as_tagged(heap).erase()),
        function.arity as usize,
        function.length as usize,
        kind,
        function.strict,
    );
    infos[fid.index()] = Some(info);
    Ok(info)
}

fn intern<'s>(
    heap: &mut Heap,
    _state: &ContextState,
    scope: &'s HandleScope<'_>,
    vm: &VM,
    s: &[u8],
) -> Handle<'s, DenseString> {
    let units = decode_wtf8(s).expect("frontends produce valid WTF-8 string constants");
    vm.interner().intern(heap, scope, StringData::Utf16(&units))
}
