use core::cell::UnsafeCell;

use crate::{
    EdgeVisitable, FeedbackVector, FixedArray, FixedByteArray, Heap, Register, Tagged, Value,
    Visitor,
};

use crate::{FrameMeta, Stack};

pub struct StackCache(UnsafeCell<StackCacheImpl>);

struct StackCacheImpl {
    acc: Register,
    code: Register,
    constants: Register,
    feedback: Register,
    pc: usize,
    base: usize,
    register_count: usize,
    active: bool,
    the_hole: Register,
}

impl StackCache {
    pub fn new(the_hole: Value) -> Self {
        Self(UnsafeCell::new(StackCacheImpl {
            acc: unsafe { Register::from_value(the_hole) },
            code: unsafe { Register::from_value(the_hole) },
            constants: unsafe { Register::from_value(the_hole) },
            feedback: unsafe { Register::from_value(the_hole) },
            pc: 0,
            base: 0,
            register_count: 0,
            active: false,
            the_hole: unsafe { Register::from_value(the_hole) },
        }))
    }

    #[allow(clippy::mut_from_ref)]
    fn get(&self) -> &mut StackCacheImpl {
        unsafe { &mut *self.0.get() }
    }

    pub fn is_active(&self) -> bool {
        self.get().active
    }

    #[inline]
    pub fn enter(&self, stack: &Stack, frame: FrameMeta, heap: &mut Heap) {
        self.load(stack, frame, heap);
        let cache = self.get();
        cache.active = true;
        // the accumulator is undefined on frame entry
        cache.acc.store(heap.known().undefined.as_tagged(heap));
    }

    #[inline]
    pub fn load(&self, stack: &Stack, frame: FrameMeta, _heap: &mut Heap) {
        let cache = self.get();
        let copy = |dst: &Register, offset: isize| {
            dst.as_raw()
                .store_raw(stack.header_slot(frame.base, offset).raw().to_bits());
        };
        copy(&cache.code, crate::stack::CODE_OFFSET);
        copy(&cache.constants, crate::stack::CONSTANTS_OFFSET);
        copy(&cache.feedback, crate::stack::FEEDBACK_OFFSET);
        cache.pc = frame.pc;
        cache.base = frame.base;
        cache.register_count = frame.register_count;
    }

    pub fn deactivate(&self, heap: &Heap) {
        let cache = self.get();
        let the_hole = cache.the_hole.get(heap);
        cache.acc.store(the_hole);
        cache.code.store(the_hole);
        cache.constants.store(the_hole);
        cache.feedback.store(the_hole);
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

    pub fn base(&self) -> usize {
        self.get().base
    }

    pub fn register_count(&self) -> usize {
        self.get().register_count
    }

    pub fn code_raw(&self) -> Value {
        self.get().code.raw()
    }

    #[inline]
    pub fn code_ref<'a>(&self, heap: &'a Heap) -> Tagged<'a, FixedByteArray> {
        debug_assert!(self.is_active(), "bytecode read from inactive cache");
        // Only `load`/`enter` write this slot, always with the frame's
        // bytecode: the kind re-check is redundant.
        unsafe { self.get().code.get(heap).cast() }
    }

    #[inline]
    pub fn constants_ref<'a>(&self, heap: &'a Heap) -> Tagged<'a, FixedArray> {
        debug_assert!(self.is_active(), "constants read from inactive cache");
        // Only `load`/`enter` write this slot, always the constant pool.
        unsafe { self.get().constants.get(heap).cast() }
    }

    /// The current frame's feedback vector, or `None` for functions without
    /// feedback slots (or while inactive).
    #[inline]
    pub fn feedback_ref<'a>(&self, heap: &'a Heap) -> Option<Tagged<'a, FeedbackVector>> {
        // Only `load`/`enter`/`deactivate` write this slot: it is always a
        // `FeedbackVector` or the hole, so the kind re-check is redundant.
        let word = self.get().feedback.get(heap);
        (word.raw() != heap.known().the_hole.raw()).then(|| unsafe { word.cast() })
    }

    pub fn acc<'a>(&self, heap: &'a Heap) -> Tagged<'a, Value> {
        self.get().acc.get(heap)
    }

    pub fn acc_mut(&self) -> Acc<'_> {
        Acc(&self.get().acc)
    }

    pub fn set_acc<'a, T: 'a>(&self, v: Tagged<'a, T>) {
        self.get().acc.store(v);
    }
}

pub struct Acc<'a>(&'a Register);

impl Acc<'_> {
    fn word_ptr(&self) -> *mut Value {
        self.0.as_raw().as_ptr().cast::<Value>()
    }
}

impl core::ops::Deref for Acc<'_> {
    type Target = Value;

    fn deref(&self) -> &Value {
        unsafe { &*self.word_ptr() }
    }
}

impl Acc<'_> {
    pub fn get<'a>(&self, heap: &'a Heap) -> Tagged<'a, Value> {
        self.0.get(heap)
    }

    /// Store into the accumulator register. Only an anchored `Tagged` may be
    /// stored, so a stale raw `Value` cannot cross a GC safepoint.
    pub fn store<'x, T: 'x>(&self, value: Tagged<'x, T>) {
        self.0.store(value);
    }
}

impl EdgeVisitable for StackCache {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        let cache = self.get();
        visitor.visit(cache.acc.as_raw());
        visitor.visit(cache.code.as_raw());
        visitor.visit(cache.constants.as_raw());
        visitor.visit(cache.feedback.as_raw());
        visitor.visit(cache.the_hole.as_raw());
    }
}
