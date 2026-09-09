use core::cell::UnsafeCell;

use crate::{
    EdgeVisitable, FixedArray, FixedByteArray, Heap, HeapRef, NoGc, Register, Value, Visitor,
};

use crate::{FrameMeta, Stack};

// TODO: consider finding a way to not need this.
// the reason this needs an UnsafeCell is because the GC needs to be aware of this
// to give the awareness this is attached to a SharedVMInstance and to access it mutably
// we need exclusive borrow
pub struct StackCache(UnsafeCell<StackCacheImpl>);

struct StackCacheImpl {
    /// The interpreter accumulator: permanently rooted (visited by the GC),
    /// so it stays valid across allocations without spill/take bookkeeping.
    acc: Register,
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

    pub fn enter(&self, stack: &Stack, frame: FrameMeta, heap: &mut Heap) {
        self.load(stack, frame, heap);
        let cache = self.get();
        cache.active = true;
        // the accumulator is undefined on frame entry
        cache.acc.store(heap.known().undefined.value());
    }

    pub fn load(&self, stack: &Stack, frame: FrameMeta, heap: &mut Heap) {
        heap.no_gc(|nogc| {
            let Some(obj) = stack.callable_slot(&frame).inner().as_heap_object(nogc) else {
                panic!("frame callable must be an object");
            };
            let info = obj
                .as_ref()
                .callable_info(nogc)
                .expect("frame callable must have callable info");
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
            // Placeholder
            handler_pc: 0,
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

    pub fn acc(&self) -> Value {
        self.get().acc.inner()
    }

    pub fn set_acc(&self, v: Value) {
        self.get().acc.store(v);
    }
}

impl EdgeVisitable for StackCache {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        let cache = self.get();
        visitor.visit(cache.acc.as_raw());
        visitor.visit(cache.code.as_raw());
        visitor.visit(cache.constants.as_raw());
    }
}
