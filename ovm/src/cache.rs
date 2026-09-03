use core::cell::UnsafeCell;

use vm::{
    EdgeVisitable, FixedArray, FixedByteArray, HeapRef, LocalHeap, NoGc, Register, Value, ValueRef,
    Visitor,
};

use crate::{FrameMeta, Stack};

// TODO: consider finding a way to not need this.
// the reason this needs an UnsafeCell is because the GC needs to be aware of this
// to give the awareness this is attached to a SharedVMInstance and to access it mutably
// we need exclusive borrow
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

    pub fn enter(&self, stack: &Stack, frame: FrameMeta, heap: &mut impl LocalHeap) {
        self.load(stack, frame, heap);
        self.get().active = true;
    }

    pub fn load(&self, stack: &Stack, frame: FrameMeta, heap: &mut impl LocalHeap) {
        heap.no_gc(|nogc, heap| {
            let ValueRef::Object(obj) = stack.callable_slot(&frame).inner().value_ref(nogc) else {
                panic!("frame callable must be an object");
            };
            let info = obj
                .as_ref()
                .callable_info(nogc, heap)
                .expect("frame callable must have a callable info");
            let cache = self.get();
            cache.code.store(info.bytecode.get().erase());
            cache.constants.store(info.constants.get().erase());
            cache.pc = frame.pc;
            cache.base = frame.base;
            cache.register_count = frame.register_count;
        });
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

    pub fn code_ref<'a>(&self, nogc: &'a NoGc<'a>) -> HeapRef<'a, FixedByteArray> {
        debug_assert!(self.is_active(), "bytecode read from inactive cache");
        self.get().code.heap_ref(nogc)
    }

    pub fn constants_ref<'a>(&self, nogc: &'a NoGc<'a>) -> HeapRef<'a, FixedArray> {
        debug_assert!(self.is_active(), "constants read from inactive cache");
        self.get().constants.heap_ref(nogc)
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

    pub fn is_acc_spilled(&self) -> bool {
        self.get().acc_spilled
    }

    pub fn reset_acc_spill(&self) {
        let cache = self.get();
        let void = cache.void;
        cache.acc.store(void);
        cache.acc_spilled = false;
    }

    pub fn restore_acc_spill(&self, was_spilled: bool) {
        let cache = self.get();
        let void = cache.void;
        cache.acc.store(void);
        cache.acc_spilled = was_spilled;
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
            visitor.visit(cache.acc.as_raw());
        }
        visitor.visit(cache.code.as_raw());
        visitor.visit(cache.constants.as_raw());
    }
}
