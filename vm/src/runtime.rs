use core::ptr::NonNull;

use crate::builtins::intrinsics::runtime_fn;
use crate::interpreter::execute;
use crate::{
    ContextState, DenseString, Errors, Handle, HandleScope, HandleSlice, Heap, Object, Tagged, VM,
    Value, VmError,
};

/// ToPrimitive hint (ES 7.1.1).
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Hint {
    Default,
    Number,
    String,
}

/// A coerced value anchored at the heap borrow that produced it (so it can
/// be consumed, stored, or rooted before the next allocation), or a pending
/// exception.
pub enum Coercion<'a> {
    Value(Tagged<'a, Value>),
    Threw,
}

// -- runtime calling convention -----------------------------------------

pub struct RuntimeContext<'a> {
    vm: &'a VM,
    heap: &'a mut Heap,
    state: &'a ContextState,
    new_target: Option<Handle<'a, Value>>,
}

impl<'a> RuntimeContext<'a> {
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
        let ex = Errors::from_vm_error(self.vm, self.heap, self.state, err)
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
        args: HandleSlice<'s>,
    ) -> Result<Value, VmError> {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        let Some(callable) = scope.cast::<Object>(callable) else {
            return Err(VmError::Type);
        };
        execute(self.vm, self.heap, self.state, callable, args, None).map(|v| v.raw())
    }

    /// Invoke a callable that is already rooted in a handle. This is the
    /// `&mut Heap`-context entry point: it needs no caller-side anchor.
    pub fn call_rooted<'s>(
        &mut self,
        callable: Handle<'_, Value>,
        args: HandleSlice<'s>,
    ) -> Result<Value, VmError> {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        let callable = scope.cast::<Object>(callable.as_tagged(&*self.heap));
        let Some(callable) = callable else {
            return Err(VmError::Type);
        };
        execute(self.vm, self.heap, self.state, callable, args, None).map(|v| v.raw())
    }

    /// Invoke `callable` as a constructor with `new.target` = `new_target`:
    /// runtime callees see `is_construct()` and the receiver's prototype
    /// comes from `new_target.prototype` (ES 9.2.2). See [`Self::call`]
    /// for the callable anchoring rules.
    pub fn call_construct<'s>(
        &mut self,
        callable: Tagged<'_, Value>,
        new_target: Tagged<'_, Value>,
        args: HandleSlice<'s>,
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
        .map(|v| v.raw())
    }

    /// Rooted [`Self::call_construct`]: callable and new.target are handles,
    /// so no caller-side anchor is needed. Returns an anchored result so the
    /// caller needs no unchecked re-read.
    pub fn call_construct_rooted<'h, 's>(
        vm: &'h VM,
        heap: &'h mut Heap,
        state: &'h ContextState,
        callable: Handle<'_, Value>,
        new_target: Handle<'_, Value>,
        args: HandleSlice<'s>,
    ) -> Result<Tagged<'h, Value>, VmError> {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&state.handles)) };
        let Some(callable) = scope.cast::<Object>(callable.as_tagged(&*heap)) else {
            return Err(VmError::Type);
        };
        // new.target may be exotic (a constructor proxy)
        let new_target = new_target.as_tagged(&*heap);
        if !new_target.is_strong_ptr() {
            return Err(VmError::Type);
        }
        let new_target = scope.handle(new_target);
        execute(vm, heap, state, callable, args, Some(new_target))
    }

    pub fn take_pending_exception(&self) -> Option<Value> {
        self.state.take_pending_exception()
    }

    pub fn has_pending_exception(&self) -> bool {
        self.state.has_pending_exception()
    }

    /// The current frame's context (direct eval chains to it).
    pub fn current_context<'h>(&self, heap: &'h Heap) -> Option<Tagged<'h, Value>> {
        self.state.current_context(heap)
    }
}

pub type RuntimeCall =
    for<'a, 's> fn(&mut RuntimeContext<'a>, HandleSlice<'s>) -> Result<Value, VmError>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RuntimeIndex(pub usize);

impl RuntimeIndex {
    /// the fixed `RuntimeFn` table occupies 0..COUNT; dynamically
    /// registered runtimes (builtins) append after it
    pub const RUNTIME_TABLE_END: Self = Self(bytecode::RuntimeFn::COUNT as usize);
}

pub struct RuntimeRegistry {
    entries: Vec<RuntimeCall>,
}

impl RuntimeRegistry {
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

    pub fn insert(&mut self, f: RuntimeCall) -> RuntimeIndex {
        let index = RuntimeIndex(self.entries.len());
        self.entries.push(f);
        index
    }

    pub fn get(&self, index: RuntimeIndex) -> Option<RuntimeCall> {
        self.entries.get(index.0).copied()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for RuntimeRegistry {
    fn default() -> Self {
        Self::new()
    }
}
