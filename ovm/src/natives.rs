use vm::{Float, LocalHeap, Smi, Value};

use crate::{Context, Heap};

pub const EXCEPTION_SENTINEL: Value = Value::CLEARED;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum VmError {
    Arity,
    Type,
    Overflow,
}

pub type NativeFn<H> = fn(&mut Context<H>, &[Value]) -> Result<Value, VmError>;

#[allow(improper_ctypes_definitions)]
pub extern "C" fn native_trampoline<H: Heap>(
    f: NativeFn<H>,
    ctx: *mut Context<H>,
    args: *const Value,
    argc: u32,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let args = unsafe { core::slice::from_raw_parts(args, argc as usize) };
    match f(ctx, args) {
        Ok(v) => v,
        Err(e) => {
            ctx.set_pending_exception(e);
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

fn smi_add<H: Heap>(_ctx: &mut Context<H>, args: &[Value]) -> Result<Value, VmError> {
    let (a, b) = match args {
        [_, a, b] => (*a, *b),
        _ => return Err(VmError::Arity),
    };
    let a = Smi::decode(a).ok_or(VmError::Type)?;
    let b = Smi::decode(b).ok_or(VmError::Type)?;
    let r = a.value().checked_add(b.value()).ok_or(VmError::Overflow)?;
    Smi::new(r).map(|s| s.encode()).ok_or(VmError::Overflow)
}

fn float_add<H: Heap>(ctx: &mut Context<H>, args: &[Value]) -> Result<Value, VmError> {
    let (a, b) = match args {
        [_, a, b] => (*a, *b),
        _ => return Err(VmError::Arity),
    };
    let sum = ctx.heap().no_gc(|nogc, heap| {
        let float_map = heap.known().float_map;
        let fa = nogc.get_as::<Float>(a, float_map).ok_or(VmError::Type)?;
        let fb = nogc.get_as::<Float>(b, float_map).ok_or(VmError::Type)?;
        Ok(fa.value.get() + fb.value.get())
    })?;
    Ok(ctx.heap().allocate::<Float>(sum).erase())
}
