use core::cell::UnsafeCell;

use crate::{
    EdgeVisitable, FixedArray, FixedByteArray, Heap, HeapRef, Register, Tagged, Value, Visitor,
};

use crate::{FrameMeta, Stack};

pub struct StackCache(UnsafeCell<StackCacheImpl>);

struct StackCacheImpl {
    acc: Register,
    code: Register,
    constants: Register,
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

    pub fn enter(&self, stack: &Stack, frame: FrameMeta, heap: &mut Heap) {
        self.load(stack, frame, heap);
        let cache = self.get();
        cache.active = true;
        // the accumulator is undefined on frame entry
        // Safety: fresh root-slot read stored immediately.
        cache
            .acc
            .store(unsafe { heap.known().undefined.read_unchecked() });
    }

    pub fn load(&self, stack: &Stack, frame: FrameMeta, heap: &mut Heap) {
        let tagged = stack.callable(heap, &frame);
        // Safety: frame callable slots hold strong object pointers.
        let obj = unsafe {
            HeapRef::from_ptr(tagged.as_ptr().expect("frame callable must be an object"))
        };
        let info = obj
            .as_ref()
            .callable_info(heap)
            .expect("frame callable must have callable info");
        let cache = self.get();
        cache.code.store(info.bytecode.get(heap).raw());
        cache.constants.store(info.constants.get(heap).raw());
        cache.pc = frame.pc;
        cache.base = frame.base;
        cache.register_count = frame.register_count;
    }

    pub fn deactivate(&self) {
        let cache = self.get();
        let the_hole = cache.the_hole.inner();
        cache.acc.store(the_hole);
        cache.code.store(the_hole);
        cache.constants.store(the_hole);
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

    pub fn code_ref<'a>(&self, heap: &'a Heap) -> HeapRef<'a, FixedByteArray> {
        debug_assert!(self.is_active(), "bytecode read from inactive cache");
        // Safety: the cached register holds a strong FixedByteArray.
        unsafe {
            HeapRef::from_ptr(
                self.get()
                    .code
                    .read(heap)
                    .get_as::<FixedByteArray>()
                    .expect("strong cache slot")
                    .into_ptr(),
            )
        }
    }

    pub fn constants_ref<'a>(&self, heap: &'a Heap) -> HeapRef<'a, FixedArray> {
        debug_assert!(self.is_active(), "constants read from inactive cache");
        // Safety: the cached register holds a strong FixedArray.
        unsafe {
            HeapRef::from_ptr(
                self.get()
                    .constants
                    .read(heap)
                    .get_as::<FixedArray>()
                    .expect("strong cache slot")
                    .into_ptr(),
            )
        }
    }

    pub fn acc<'a>(&self, heap: &'a Heap) -> Tagged<'a, Value> {
        self.get().acc.read(heap)
    }

    pub fn acc_mut(&self) -> Acc<'_> {
        Acc(&self.get().acc)
    }

    pub fn set_acc<'a, T: 'a>(&self, v: impl Into<Tagged<'a, T>>) {
        self.get().acc.store(v.into().raw());
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

impl core::ops::DerefMut for Acc<'_> {
    fn deref_mut(&mut self) -> &mut Value {
        unsafe { &mut *self.word_ptr() }
    }
}

impl Acc<'_> {
    pub fn read<'a>(&self, heap: &'a Heap) -> Tagged<'a, Value> {
        self.0.read(heap)
    }
}

impl EdgeVisitable for StackCache {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        let cache = self.get();
        visitor.visit(cache.acc.as_raw());
        visitor.visit(cache.code.as_raw());
        visitor.visit(cache.constants.as_raw());
        visitor.visit(cache.the_hole.as_raw());
    }
}
