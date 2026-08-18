use core::cell::UnsafeCell;

use vm::{
    CallableInfoObject, EdgeVisitable, FixedArray, FixedByteArray, HeapPtr, HeapRef, NoGc,
    Register, Tagged, Value, Visitor,
};

use crate::{FrameMeta, Stack};

pub struct StackCache(UnsafeCell<StackCacheImpl>);

struct StackCacheImpl {
    acc: Register,
    acc_spilled: bool,
    code: Register,
    constants: Register,
    pc: usize,
    base: usize,
    register_count: usize,
    active: bool,
    void: Value,
}

impl StackCache {
    pub fn new(void: Value) -> Self {
        Self(UnsafeCell::new(StackCacheImpl {
            acc: unsafe { Register::from_value(void) },
            acc_spilled: false,
            code: unsafe { Register::from_value(void) },
            constants: unsafe { Register::from_value(void) },
            pc: 0,
            base: 0,
            register_count: 0,
            active: false,
            void,
        }))
    }

    fn get(&self) -> &mut StackCacheImpl {
        unsafe { &mut *self.0.get() }
    }

    pub fn is_active(&self) -> bool {
        self.get().active
    }

    pub fn enter(&self, stack: &Stack, frame: FrameMeta) {
        debug_assert!(!self.is_active(), "re-entrant interpreter run");
        self.load(stack, frame);
        self.get().active = true;
    }

    pub fn load(&self, stack: &Stack, frame: FrameMeta) {
        let callable = stack.callable(&frame);
        let ptr = HeapPtr::decode_strong(callable).expect("callable must be strong");
        let obj = unsafe { ptr.cast::<CallableInfoObject>().as_ref() };
        let cache = self.get();
        cache.code.store(obj.bytecode.get().erase());
        cache.constants.store(obj.constants.get().erase());
        cache.pc = frame.pc;
        cache.base = frame.base;
        cache.register_count = frame.register_count;
    }

    pub fn deactivate(&self) {
        let cache = self.get();
        let void = cache.void;
        cache.acc.store(void);
        cache.acc_spilled = false;
        cache.code.store(void);
        cache.constants.store(void);
        cache.active = false;
    }

    pub fn frame_meta(&self) -> FrameMeta {
        let cache = self.get();
        FrameMeta {
            base: cache.base,
            pc: cache.pc,
            register_count: cache.register_count,
        }
    }

    pub fn pc(&self) -> usize {
        self.get().pc
    }

    pub fn set_pc(&self, pc: usize) {
        self.get().pc = pc;
    }

    fn code(&self) -> Value {
        self.get().code.inner()
    }

    pub fn code_ref<'a>(&self, nogc: &'a NoGc<'a>) -> HeapRef<'a, FixedByteArray> {
        debug_assert!(self.is_active(), "bytecode read from inactive cache");
        unsafe { nogc.get_unchecked(Tagged::from_value_unchecked(self.code())) }
    }

    fn constants(&self) -> Value {
        self.get().constants.inner()
    }

    pub fn constants_ref<'a>(&self, nogc: &'a NoGc<'a>) -> HeapRef<'a, FixedArray> {
        debug_assert!(self.is_active(), "constants read from inactive cache");
        unsafe { nogc.get_unchecked(Tagged::from_value_unchecked(self.constants())) }
    }

    pub fn spill_acc(&self, acc: Value) {
        let cache = self.get();
        debug_assert!(!cache.acc_spilled, "accumulator spilled twice");
        cache.acc.store(acc);
        cache.acc_spilled = true;
    }

    pub fn take_acc(&self) -> Value {
        let cache = self.get();
        debug_assert!(cache.acc_spilled, "accumulator taken without spill");
        cache.acc_spilled = false;
        cache.acc.inner()
    }
}

impl EdgeVisitable for StackCache {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        let cache = self.get();
        debug_assert!(
            !cache.active || cache.acc_spilled,
            "GC visited an active cache with an unspilled accumulator"
        );
        if cache.acc_spilled {
            visitor.visit_register(&cache.acc);
        }
        visitor.visit_register(&cache.code);
        visitor.visit_register(&cache.constants);
    }
}
