use vm::{Float, Handle, HandleScope, InternedString, LocalHeap, Smi, Value};

use crate::{ContextState, Heap, Thread, VM};

pub const EXCEPTION_SENTINEL: Value = Value::CLEARED;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum VmError {
    Arity,
    Type,
    Overflow,
    StackOverflow,
}

pub struct NativeContext<'a, H: Heap> {
    vm: &'a VM<H>,
    heap: &'a mut H::Local,
    state: &'a ContextState,
}

impl<'a, H: Heap> NativeContext<'a, H> {
    pub fn new(vm: &'a VM<H>, heap: &'a mut H::Local, state: &'a ContextState) -> Self {
        Self { vm, heap, state }
    }

    pub fn vm(&self) -> &VM<H> {
        self.vm
    }

    pub fn heap(&mut self) -> &mut H::Local {
        self.heap
    }

    pub fn intern<'s>(
        &mut self,
        scope: &'s HandleScope<'_>,
        s: impl AsRef<str>,
    ) -> Handle<'s, InternedString> {
        self.vm.interner().intern(self.heap, scope, s)
    }

    pub fn set_pending_exception(&self, err: VmError) {
        self.state.set_pending_exception(err);
    }

    pub fn take_pending_exception(&self) -> Option<VmError> {
        self.state.take_pending_exception()
    }

    pub fn has_pending_exception(&self) -> bool {
        self.state.has_pending_exception()
    }
}

pub type NativeFn<H> = for<'a> fn(&mut NativeContext<'a, H>, &[Value]) -> Result<Value, VmError>;

#[allow(improper_ctypes_definitions)]
pub extern "C" fn native_trampoline<H: Heap>(
    f: NativeFn<H>,
    thread: *mut Thread<H>,
    args: *const Value,
    argc: u32,
) -> Value {
    let thread = unsafe { &mut *thread };
    let args = unsafe { core::slice::from_raw_parts(args, argc as usize) };
    let mut nctx = NativeContext::new(&thread.vm, &mut thread.heap, &thread.state);
    match f(&mut nctx, args) {
        Ok(v) => v,
        Err(e) => {
            nctx.set_pending_exception(e);
            EXCEPTION_SENTINEL
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeIndex(pub usize);

impl NativeIndex {
    pub const SMI_ADD: Self = Self(0);
    pub const FLOAT_ADD: Self = Self(1);
}

pub struct NativeRegistry<H: Heap> {
    entries: Vec<NativeFn<H>>,
}

impl<H: Heap> NativeRegistry<H> {
    pub fn new() -> Self {
        let mut registry = Self {
            entries: Vec::new(),
        };
        debug_assert_eq!(registry.insert(smi_add), NativeIndex::SMI_ADD);
        debug_assert_eq!(registry.insert(float_add), NativeIndex::FLOAT_ADD);
        registry
    }

    pub fn insert(&mut self, f: NativeFn<H>) -> NativeIndex {
        let index = NativeIndex(self.entries.len());
        self.entries.push(f);
        index
    }

    pub fn get(&self, index: NativeIndex) -> Option<NativeFn<H>> {
        self.entries.get(index.0).copied()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<H: Heap> Default for NativeRegistry<H> {
    fn default() -> Self {
        Self::new()
    }
}

fn smi_add<H: Heap>(_nctx: &mut NativeContext<'_, H>, args: &[Value]) -> Result<Value, VmError> {
    let (a, b) = match args {
        [_, a, b] => (*a, *b),
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

fn float_add<H: Heap>(nctx: &mut NativeContext<'_, H>, args: &[Value]) -> Result<Value, VmError> {
    let (a, b) = match args {
        [_, a, b] => (*a, *b),
        _ => return Err(VmError::Arity),
    };
    let sum = nctx.heap().no_gc(|nogc, heap| {
        let float_map = heap.known().float_map;
        let fa = nogc.get_as::<Float>(a, float_map).ok_or(VmError::Type)?;
        let fb = nogc.get_as::<Float>(b, float_map).ok_or(VmError::Type)?;
        Ok(fa.value.get() + fb.value.get())
    })?;
    Ok(nctx.heap().allocate::<Float>(sum).erase())
}
