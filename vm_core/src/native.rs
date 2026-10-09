use crate::api::Runtime;
use crate::errors::Errors;
use crate::intrinsics::native_fn;
use crate::{
    Args, Handle, HandleScope, HandleSlice, Heap, Object, Tagged, ThreadState, VM, Value, VmError,
};
use core::ptr::NonNull;

// -- native calling convention --

pub struct HostCtx<'a> {
    pub vm: &'a VM,
    pub heap: &'a mut Heap,
    pub state: &'a ThreadState,
}

impl<'a> HostCtx<'a> {
    pub fn new(vm: &'a VM, heap: &'a mut Heap, state: &'a ThreadState) -> Self {
        Self { vm, heap, state }
    }

    /// The state runtime `R` stored during `VM::add`.
    pub fn runtime_state<R: Runtime>(&self) -> &R::State {
        self.vm.runtime_state::<R>()
    }

    /// Root a handle scope and hand the closure the context parts at their
    /// real (heap) lifetime, so values produced inside are `Tagged<'a>`.
    pub fn handle_scope<R>(
        self,
        f: impl FnOnce(&'a VM, &'a mut Heap, &'a ThreadState, HandleScope<'_>) -> R,
    ) -> R {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        let HostCtx {
            vm, heap, state, ..
        } = self;
        f(vm, heap, state, scope)
    }

    /// Invoke `callable` ([[Call]]) when `new_target` is `None`, or
    /// [[Construct]] with `new.target` = that handle when `Some`
    /// (ES 9.2.2). The callable is a rooted handle; the result is anchored to
    /// the `heap` borrow and `heap` stays usable.
    pub fn enter<'h, 'r>(
        vm: &'h VM,
        heap: &'h mut Heap,
        state: &'h ThreadState,
        callable: Handle<'_, Value>,
        args: HandleSlice<'r>,
        new_target: Option<Handle<'_, Value>>,
    ) -> Result<Tagged<'h, Value>, VmError> {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&state.handles)) };
        let Some(callable) = scope.cast::<Object>(heap, callable.as_tagged(&*heap)) else {
            return Err(VmError::Type);
        };
        let new_target = match new_target {
            None => None,
            Some(nt) => {
                // new.target may be exotic (a constructor proxy)
                let nt = nt.as_tagged(&*heap);
                if !nt.is_strong_ptr() {
                    return Err(VmError::Type);
                }
                Some(scope.handle(nt))
            }
        };
        (vm.shared.execute)(vm, heap, state, callable, args, new_target)
    }
}

/// The runtime-call ABI (tier 1): a single tagged word in the first
/// return register. Errors are the exception sentinel with the pending
/// exception set — one error channel at the boundary. `Result` survives only above `execute`.
pub type NativeFn =
    for<'a, 'nt> fn(HostCtx<'a>, Option<Handle<'nt, Value>>, Args) -> Tagged<'a, Value>;

/// Materialize `err` as the pending exception and return the sentinel:
/// the single-channel bridge for runtime bodies.
pub fn raise_runtime<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ThreadState,
    err: VmError,
) -> Tagged<'a, Value> {
    let ex =
        Errors::from_vm_error(vm, heap, state, err).expect("error materialization must not fail");
    state.set_pending_exception(ex);
    state.set_last_error(err);
    heap.known().exception.as_tagged(heap).erase()
}

/// `?` for runtime bodies: fold an internal `Result` into the sentinel
/// channel via [`raise_runtime`].
#[macro_export]
macro_rules! rt_try {
    ($vm:expr, $heap:expr, $state:expr, $e:expr) => {
        match $e {
            Ok(v) => v,
            Err(err) => return $crate::raise_runtime($vm, $heap, $state, err),
        }
    };
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeIndex(pub usize);

impl NativeIndex {
    /// the fixed `BuiltinFn` table occupies 0..COUNT; dynamically
    /// registered runtimes (builtins) append after it
    pub const NATIVE_TABLE_END: Self = Self(bytecode::BuiltinFn::COUNT as usize);
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
        // operands carry BuiltinFn discriminants, so registration order
        // must (and is asserted to) match
        for (i, id) in bytecode::BuiltinFn::ALL.iter().enumerate() {
            debug_assert_eq!(*id as u16 as usize, i, "ALL order must match discriminants");
            let idx = registry.insert(native_fn(*id));
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
