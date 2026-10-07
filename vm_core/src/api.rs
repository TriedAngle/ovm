//! The extension API: choosing an interpreter and plugging in a language
//! runtime.

use core::any::Any;

use crate::{
    EdgeVisitable, Handle, HandleSlice, Heap, Object, Tagged, ThreadState, VM, Value, VmError,
};

/// The interpreter entry: [[Call]]/[[Construct]] on a bytecode callable.
/// Stored as a plain fn pointer in `SharedVM` (set once from the
/// `I: Interpreter` type parameter) so core re-entry points —
/// `Thread::execute`, `HostCtx::enter` — pay one indirect call,
/// never per opcode.
pub type EntryFn = for<'a, 'v, 's, 'c, 'r, 'n> fn(
    vm: &'v VM,
    heap: &'a mut Heap,
    state: &'s ThreadState,
    callable: Handle<'c, Object>,
    args: HandleSlice<'r>,
    new_target: Option<Handle<'n, Value>>,
) -> Result<Tagged<'a, Value>, VmError>;

pub trait Interpreter {
    const EXECUTE: EntryFn;
}

/// A language runtime contributing native functions and globals to a VM
/// (registered and installed once by `VM::add`, before any code runs).
pub trait Runtime: 'static {
    /// Per-VM runtime state, rooted for the GC; fetched back by native
    /// functions via `HostCtx::runtime_state`.
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
