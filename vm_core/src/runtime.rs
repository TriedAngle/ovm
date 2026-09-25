use core::any::Any;
use core::ptr::NonNull;

use crate::intrinsics::runtime_fn;
use crate::{
    ContextState, EdgeVisitable, Handle, HandleScope, HandleSlice, Heap, Object, Tagged, VM, Value,
    VmError,
};

/// The interpreter entry: [[Call]]/[[Construct]] on a bytecode callable.
/// Stored as a plain fn pointer in `SharedVM` (set once from the
/// `I: Interpreter` type parameter) so core re-entry points —
/// `Thread::execute`, `RuntimeContext::call` — pay one indirect call,
/// never per opcode.
pub type ExecuteFn = for<'a, 'v, 's, 'c, 'r, 'n> fn(
    vm: &'v VM,
    heap: &'a mut Heap,
    state: &'s ContextState,
    callable: Handle<'c, Object>,
    args: HandleSlice<'r>,
    new_target: Option<Handle<'n, Value>>,
) -> Result<Tagged<'a, Value>, VmError>;

pub trait Interpreter {
    const EXECUTE: ExecuteFn;
}

/// A language runtime contributing native functions and globals to a VM
/// (registered and installed once by `VM::add`, before any code runs).
pub trait Runtime: 'static {
    /// Per-VM runtime state, rooted for the GC; fetched back by native
    /// functions via `RuntimeContext::runtime_state`.
    type State: EdgeVisitable + Default + Send + Sync + Any;

    fn setup(vm: &mut VM, state: &mut Self::State) -> Result<(), VmError>;
}

pub trait ErasedRuntimeState: EdgeVisitable + Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

impl<T: EdgeVisitable + Any + Send + Sync> ErasedRuntimeState for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

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
    pub vm: &'a VM,
    pub heap: &'a mut Heap,
    pub state: &'a ContextState,
    pub new_target: Option<Handle<'a, Value>>,
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

    /// The state runtime `R` stored during `VM::add`.
    pub fn runtime_state<R: Runtime>(&self) -> &R::State {
        self.vm.runtime_state::<R>()
    }

    /// Root a handle scope and hand the closure the context parts at their
    /// real (heap) lifetime, so values produced inside are `Tagged<'a>`.
    pub fn handle_scope<R>(
        self,
        f: impl FnOnce(&'a VM, &'a mut Heap, &'a ContextState, HandleScope<'_>) -> R,
    ) -> R {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        let RuntimeContext {
            vm, heap, state, ..
        } = self;
        f(vm, heap, state, scope)
    }

    /// Invoke `callable` ([[Call]]) when `new_target` is `None`, or
    /// [[Construct]] with `new.target` = that handle when `Some`
    /// (ES 9.2.2). The callable is a rooted handle; the result is anchored to
    /// the `heap` borrow and `heap` stays usable.
    pub fn call<'h, 'r>(
        vm: &'h VM,
        heap: &'h mut Heap,
        state: &'h ContextState,
        callable: Handle<'_, Value>,
        args: HandleSlice<'r>,
        new_target: Option<Handle<'_, Value>>,
    ) -> Result<Tagged<'h, Value>, VmError> {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&state.handles)) };
        let Some(callable) = scope.cast::<Object>(callable.as_tagged(&*heap)) else {
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

pub type RuntimeCall =
    for<'a, 'r> fn(RuntimeContext<'a>, HandleSlice<'r>) -> Result<Tagged<'a, Value>, VmError>;

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
