use core::ptr::NonNull;

use vm::{Float, GcSlice, Handle, HandleScope, Heap, InternedString, Object, Smi, Value, VmError};

use crate::{ContextState, Thread, VM};

pub struct NativeContext<'a> {
    vm: &'a VM,
    heap: &'a mut Heap,
    state: &'a ContextState,
    /// `new.target` of the active [[Construct]] call (ES 9.2.2): the
    /// invoked constructor, or `None` when called via [[Call]] (where
    /// `new.target` is undefined).
    // TODO: the Error family constructors read this once they exist
    new_target: Option<Handle<'a, Object>>,
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

    pub(crate) fn with_new_target(
        vm: &'a VM,
        heap: &'a mut Heap,
        state: &'a ContextState,
        new_target: Option<Handle<'a, Object>>,
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

    pub fn new_target(&self) -> Option<Value> {
        self.new_target.map(|h| h.value())
    }

    pub fn vm(&self) -> &VM {
        self.vm
    }

    /// Split the context into its parts (for multi-borrow calls).
    pub(crate) fn split(&mut self) -> (&VM, &mut Heap, &ContextState) {
        (&self.vm, &mut self.heap, &self.state)
    }

    pub fn heap(&mut self) -> &mut Heap {
        self.heap
    }

    pub fn intern<'s>(
        &mut self,
        scope: &'s HandleScope<'_>,
        s: impl AsRef<[u8]>,
    ) -> Handle<'s, InternedString> {
        self.vm.interner().intern(self.heap, scope, s)
    }

    pub fn set_pending_exception(&mut self, err: VmError) {
        let ex = crate::errors::error_from_vm_error(self.vm, self.heap, self.state, err)
            .expect("error materialization must not fail");
        self.state.set_pending_exception(ex);
    }

    pub fn handle_scope<R>(&mut self, f: impl FnOnce(&mut Self, HandleScope<'_>) -> R) -> R {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        f(self, scope)
    }

    pub fn call<'s>(&mut self, callable: Value, args: GcSlice<'s>) -> Result<Value, VmError> {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        let Some(callable) = scope.cast::<Object>(callable) else {
            return Err(VmError::Type);
        };
        crate::interpreter::execute(self.vm, self.heap, self.state, callable, args, None)
    }

    /// Invoke `callable` as a constructor with `new.target` = `new_target`:
    /// native callees see `is_construct()` and the receiver's prototype
    /// comes from `new_target.prototype` (ES 9.2.2).
    pub fn call_construct<'s>(
        &mut self,
        callable: Value,
        new_target: Value,
        args: GcSlice<'s>,
    ) -> Result<Value, VmError> {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        let Some(callable) = scope.cast::<Object>(callable) else {
            return Err(VmError::Type);
        };
        let Some(new_target) = scope.cast::<Object>(new_target) else {
            return Err(VmError::Type);
        };
        crate::interpreter::execute(
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
#[allow(improper_ctypes_definitions)]
pub extern "C" fn native_trampoline(
    f: NativeFn,
    thread: *mut Thread,
    args: *const Value,
    argc: u32,
) -> Value {
    let thread = unsafe { &mut *thread };
    // SAFETY: caller-owned C memory; the native must not read it after
    // allocating (see GcSlice)
    let args = unsafe { GcSlice::from_slice(core::slice::from_raw_parts(args, argc as usize)) };
    let mut nctx = NativeContext::new(&thread.vm, &mut thread.heap, &thread.state);
    match f(&mut nctx, args) {
        Ok(v) => v,
        Err(e) => {
            let ex =
                crate::errors::error_from_vm_error(&thread.vm, &mut thread.heap, &thread.state, e)
                    .expect("error materialization must not fail");
            thread.state.set_pending_exception(ex);
            thread.heap.known().exception.value()
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeIndex(pub usize);

impl NativeIndex {
    pub const SMI_ADD: Self = Self(0);
    pub const FLOAT_ADD: Self = Self(1);
}

pub struct NativeRegistry {
    entries: Vec<NativeFn>,
}

impl NativeRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            entries: Vec::new(),
        };
        registry.insert(smi_add);
        registry.insert(float_add);
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

fn smi_add(_nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let (a, b) = match (args.get(1), args.get(2)) {
        (Some(a), Some(b)) => (a, b),
        _ => return Err(VmError::Arity),
    };
    let a = Smi::decode(a).ok_or(VmError::Type)?;
    let b = Smi::decode(b).ok_or(VmError::Type)?;
    let r = a.value().checked_add(b.value()).ok_or(VmError::Overflow)?;
    if !Smi::in_range(r) {
        return Err(VmError::Overflow);
    }
    Ok(Smi::new(r).encode())
}

fn float_add(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let (a, b) = match (args.get(1), args.get(2)) {
        (Some(a), Some(b)) => (a, b),
        _ => return Err(VmError::Arity),
    };
    let sum = nctx.heap().no_gc(|nogc| {
        let fa = a.get_as::<Float>(nogc).ok_or(VmError::Type)?;
        let fb = b.get_as::<Float>(nogc).ok_or(VmError::Type)?;
        Ok(fa.value.get() + fb.value.get())
    })?;
    nctx.handle_scope(|nctx, scope| Ok(nctx.heap().new_number(&scope, sum)))
}
