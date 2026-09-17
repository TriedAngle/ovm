//! The native calling convention: the context natives run in, the fn
//! pointer type, the index handle, and the registry mapping indices to
//! implementations (the fixed RuntimeFn table first, then dynamically
//! registered builtins).

use core::ptr::NonNull;

use crate::builtins::intrinsics::runtime_fn;
use crate::errors::error_from_vm_error;
use crate::interpreter::execute;
use crate::{
    ContextState, DenseString, GcSlice, Handle, HandleScope, Heap, Object, Tagged, Thread, VM,
    Value, VmError,
};

pub struct NativeContext<'a> {
    vm: &'a VM,
    heap: &'a mut Heap,
    state: &'a ContextState,
    /// `new.target` of the active [[Construct]] call (ES 9.2.2): the
    /// invoked constructor (possibly an exotic object like a proxy),
    /// or `None` when called via [[Call]] (where `new.target` is
    /// undefined).
    new_target: Option<Handle<'a, Value>>,
}

impl<'a> NativeContext<'a> {
    pub fn new(vm: &'a VM, heap: &'a mut Heap, state: &'a ContextState) -> Self {
        Self {
            vm,
            heap,
            state,
            new_target: None,
        }
    }

    pub fn with_new_target(
        vm: &'a VM,
        heap: &'a mut Heap,
        state: &'a ContextState,
        new_target: Option<Handle<'a, Value>>,
    ) -> Self {
        Self {
            vm,
            heap,
            state,
            new_target,
        }
    }

    pub fn is_construct(&self) -> bool {
        self.new_target.is_some()
    }

    /// The raw `new.target` word: callers must consume it in the same
    /// statement (store to a register / rooted slot) or anchor it
    /// themselves.
    pub fn new_target(&self) -> Option<Value> {
        self.new_target.map(|h| unsafe { h.read_unchecked() })
    }

    pub fn vm(&self) -> &VM {
        self.vm
    }

    /// Split the context into its parts (for multi-borrow calls).
    pub fn split(&mut self) -> (&VM, &mut Heap, &ContextState) {
        (self.vm, &mut self.heap, self.state)
    }

    pub fn heap(&mut self) -> &mut Heap {
        self.heap
    }

    pub fn intern<'s>(&mut self, scope: &'s HandleScope<'_>, s: &str) -> Handle<'s, DenseString> {
        self.vm.interner().intern_str(self.heap, scope, s)
    }

    pub fn set_pending_exception(&mut self, err: VmError) {
        let ex = error_from_vm_error(self.vm, self.heap, self.state, err)
            .expect("error materialization must not fail");
        self.state.set_pending_exception(ex);
    }

    pub fn handle_scope<R>(&mut self, f: impl FnOnce(&mut Self, HandleScope<'_>) -> R) -> R {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        f(self, scope)
    }

    /// Invoke `callable` ([[Call]]). The anchored callable must be valid
    /// under a borrow of the heap that is still live for this call; in
    /// `&mut Heap` contexts the only constructible anchors are loan-free
    /// (`Smi::into_tagged()` or `Tagged::from_value_unchecked` on a word
    /// freshly read from rooted memory).
    pub fn call<'s>(
        &mut self,
        callable: Tagged<'_, Value>,
        args: GcSlice<'s>,
    ) -> Result<Value, VmError> {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        let Some(callable) = scope.cast::<Object>(callable) else {
            return Err(VmError::Type);
        };
        execute(self.vm, self.heap, self.state, callable, args, None)
    }

    /// Invoke a callable that is already rooted in a handle. This is the
    /// `&mut Heap`-context entry point: it needs no caller-side anchor.
    pub fn call_rooted<'s>(
        &mut self,
        callable: Handle<'_, Value>,
        args: GcSlice<'s>,
    ) -> Result<Value, VmError> {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        let callable = scope.cast::<Object>(callable.as_tagged(&*self.heap));
        let Some(callable) = callable else {
            return Err(VmError::Type);
        };
        execute(self.vm, self.heap, self.state, callable, args, None)
    }

    /// Invoke `callable` as a constructor with `new.target` = `new_target`:
    /// native callees see `is_construct()` and the receiver's prototype
    /// comes from `new_target.prototype` (ES 9.2.2). See [`Self::call`]
    /// for the callable anchoring rules.
    pub fn call_construct<'s>(
        &mut self,
        callable: Tagged<'_, Value>,
        new_target: Tagged<'_, Value>,
        args: GcSlice<'s>,
    ) -> Result<Value, VmError> {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        let Some(callable) = scope.cast::<Object>(callable) else {
            return Err(VmError::Type);
        };
        // new.target may be exotic (a constructor proxy)
        if !new_target.is_strong_ptr() {
            return Err(VmError::Type);
        }
        let new_target = scope.handle(new_target);
        execute(
            self.vm,
            self.heap,
            self.state,
            callable,
            args,
            Some(new_target),
        )
    }

    /// Rooted [`Self::call_construct`]: callable and new.target are handles,
    /// so no caller-side anchor is needed.
    pub fn call_construct_rooted<'s>(
        &mut self,
        callable: Handle<'_, Value>,
        new_target: Handle<'_, Value>,
        args: GcSlice<'s>,
    ) -> Result<Value, VmError> {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        let Some(callable) = scope.cast::<Object>(callable.as_tagged(&*self.heap)) else {
            return Err(VmError::Type);
        };
        // new.target may be exotic (a constructor proxy)
        let new_target = new_target.as_tagged(&*self.heap);
        if !new_target.is_strong_ptr() {
            return Err(VmError::Type);
        }
        let new_target = scope.handle(new_target);
        execute(
            self.vm,
            self.heap,
            self.state,
            callable,
            args,
            Some(new_target),
        )
    }

    pub fn take_pending_exception(&self) -> Option<Value> {
        self.state.take_pending_exception()
    }

    pub fn has_pending_exception(&self) -> bool {
        self.state.has_pending_exception()
    }

    /// The current frame's context (direct eval chains to it).
    pub fn current_context(&self) -> Option<Value> {
        self.state.current_context()
    }
}

pub type NativeFn = for<'a, 's> fn(&mut NativeContext<'a>, GcSlice<'s>) -> Result<Value, VmError>;

// TODO: this is still not a sound C interface (Rust-ABI `NativeFn` with
// reference/slice arguments); decide on the C ABI before exporting.
//
// Invokes the registered native `index` on `thread` with `argc` raw
// argument words: errors materialize into the pending exception and the
// exception sentinel is returned.
pub unsafe extern "C" fn native_trampoline(
    index: usize,
    thread: *mut Thread,
    args: *const Value,
    argc: u32,
) -> Value {
    let thread = unsafe { &mut *thread };
    let f = thread.vm().native(NativeIndex(index));
    // SAFETY: caller-owned C memory; the native must not read it after
    // allocating (see GcSlice)
    let args = unsafe { GcSlice::from_slice(core::slice::from_raw_parts(args, argc as usize)) };
    let mut nctx = NativeContext::new(&thread.vm, &mut thread.heap, &thread.state);
    match f(&mut nctx, args) {
        Ok(v) => v,
        Err(e) => {
            let ex = error_from_vm_error(&thread.vm, &mut thread.heap, &thread.state, e)
                .expect("error materialization must not fail");
            thread.state.set_pending_exception(ex);
            // Safety: singleton word read for immediate return; the
            // singletons are promoted old-gen and never move.
            unsafe { thread.heap.known().exception.read_unchecked() }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeIndex(pub usize);

impl NativeIndex {
    /// the fixed `RuntimeFn` table occupies 0..COUNT; dynamically
    /// registered natives (builtins) append after it
    pub const RUNTIME_TABLE_END: Self = Self(bytecode::RuntimeFn::COUNT as usize);
}

pub struct NativeRegistry {
    entries: Vec<NativeFn>,
}

impl NativeRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            entries: Vec::new(),
        };
        // the runtime-helper table owns indices 0..COUNT: CallRuntime
        // operands carry RuntimeFn discriminants, so registration order
        // must (and is asserted to) match
        for (i, id) in bytecode::RuntimeFn::ALL.iter().enumerate() {
            debug_assert_eq!(*id as u16 as usize, i, "ALL order must match discriminants");
            let idx = registry.insert(runtime_fn(*id));
            debug_assert_eq!(idx.0, i, "runtime table must start at index 0");
        }
        registry
    }

    pub fn insert(&mut self, f: NativeFn) -> NativeIndex {
        let index = NativeIndex(self.entries.len());
        self.entries.push(f);
        index
    }

    pub fn get(&self, index: NativeIndex) -> Option<NativeFn> {
        self.entries.get(index.0).copied()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for NativeRegistry {
    fn default() -> Self {
        Self::new()
    }
}
