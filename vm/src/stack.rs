use core::cell::{Cell, RefCell};

use crate::{EdgeVisitable, HandleSlice, Heap, Object, Register, Smi, Tagged, Value, Visitor};

use crate::VmError;

/// Number of slots in a thread's interpreter stack.
pub const STACK_SLOTS: usize = 16 * 1024;

/// Fixed header slots between the register file and the parameter region
pub const HEADER_SLOTS: usize = 4;
pub const CALLABLE_OFFSET: usize = 0;
pub const ARGC_OFFSET: usize = 1;
/// The frame's current context (the chain LoadContextSlot walks); unlike
/// the function object's closure-context slot it is per-frame, so
/// recursion cannot clobber a suspended frame's context.
pub const CONTEXT_OFFSET: usize = 2;
/// new.target of the active [[Construct]] (ES 9.2.2): the constructor, or
/// undefined when the function was called. `super()` forwards this value.
pub const NEW_TARGET_OFFSET: usize = 3;

/// Frame layout:
/// index                    content
/// base + 0 .. +rc-1        register file (r0..rn)
/// base + rc + 0            header: callable
/// base + rc + 1            header: argc (receiver included)
/// base + rc + 2            header: current context
/// base + rc + 3            header: new.target (or undefined)
/// base + rc + H .. +argc   parameters (param 0 = receiver)
///
/// Parameters are addressed with negative register indices.
///
/// The stack only tracks *suspended* frames: the frame currently being executed
/// lives in the [`StackCache`](StackCache) and is pushed here when a call suspends it.
pub struct Stack {
    slots: Box<[Register]>,
    top: Cell<usize>,
    frames: RefCell<Vec<FrameMeta>>,
    /// Register files of fresh frames are initialized to this value
    /// (the hole: uninitialized `let`/`const` reads must be TDZ errors).
    fill: Register,
    undefined: Register,
}

impl Stack {
    pub fn new(capacity: usize, fill: Value, undefined: Value) -> Self {
        Self {
            slots: (0..capacity)
                .map(|_| unsafe { Register::from_value(fill) })
                .collect(),
            top: Cell::new(0),
            frames: RefCell::new(Vec::new()),
            fill: unsafe { Register::from_value(fill) },
            undefined: unsafe { Register::from_value(undefined) },
        }
    }

    pub fn top(&self) -> usize {
        self.top.get()
    }

    pub fn set_top(&self, top: usize) {
        self.top.set(top);
    }

    pub fn slot_unchecked(&self, index: usize) -> &Register {
        debug_assert!(index < self.slots.len());
        unsafe { &*self.slots.as_ptr().add(index) }
    }

    pub fn value_slice(&self, base: usize, count: usize) -> HandleSlice<'_> {
        let slots = &self.slots[base..base + count];
        // Safety: stack slots are GC-visited, so the words stay current for
        // as long as the returned slice is alive.
        unsafe {
            HandleSlice::from_slice(core::slice::from_raw_parts(
                slots.as_ptr() as *const Value,
                count,
            ))
        }
    }

    /// The frame's callable, re-read under a heap borrow.
    pub fn callable<'a>(&self, heap: &'a Heap, meta: &FrameMeta) -> Tagged<'a, Object> {
        // Safety: frame callable slots hold strong object pointers.
        unsafe { self.callable_slot(meta).read(heap).cast::<Object>() }
    }

    pub fn callable_slot(&self, meta: &FrameMeta) -> &Register {
        self.slot_unchecked(meta.base + meta.register_count + CALLABLE_OFFSET)
    }

    /// The frame's current context, re-read under a heap borrow.
    pub fn context<'a>(&self, heap: &'a Heap, meta: &FrameMeta) -> Tagged<'a, Value> {
        self.context_slot(meta).read(heap)
    }

    pub fn context_slot(&self, meta: &FrameMeta) -> &Register {
        self.slot_unchecked(meta.base + meta.register_count + CONTEXT_OFFSET)
    }

    /// The frame's `new.target`, re-read under a heap borrow.
    pub fn new_target<'a>(&self, heap: &'a Heap, meta: &FrameMeta) -> Tagged<'a, Value> {
        self.new_target_slot(meta).read(heap)
    }

    pub fn new_target_slot(&self, meta: &FrameMeta) -> &Register {
        self.slot_unchecked(meta.base + meta.register_count + NEW_TARGET_OFFSET)
    }

    /// The frame's actual argument count, receiver included.
    pub fn argc(&self, meta: &FrameMeta) -> usize {
        self.slot_unchecked(meta.base + meta.register_count + ARGC_OFFSET)
            .read_smi()
            .value() as usize
    }

    fn reg_index(meta: &FrameMeta, i: i32) -> usize {
        if i >= 0 {
            meta.base + i as usize
        } else {
            meta.base + meta.register_count + HEADER_SLOTS + (-i - 1) as usize
        }
    }

    /// Read a register under a heap borrow: rooted memory is updated in
    /// place by the GC, so the word is current and valid for `'a`.
    pub fn reg<'a>(&self, _heap: &'a Heap, meta: &FrameMeta, i: i32) -> Tagged<'a, Value> {
        self.slot_unchecked(Self::reg_index(meta, i)).read(_heap)
    }

    pub fn set_reg<'x, T: 'x>(&self, meta: &FrameMeta, i: i32, v: Tagged<'x, T>) {
        self.slot_unchecked(Self::reg_index(meta, i)).store(v);
    }

    pub fn args(&self, meta: &FrameMeta, reg_base: i32, count: usize) -> HandleSlice<'_> {
        self.value_slice(Self::reg_index(meta, reg_base), count)
    }

    pub fn push_initial_frame(
        &self,
        heap: &Heap,
        callable: Tagged<'_, Value>,
        register_count: usize,
        context: Tagged<'_, Value>,
        new_target: Tagged<'_, Value>,
        args: HandleSlice<'_>,
        formal_min: usize,
    ) -> Result<FrameMeta, VmError> {
        let argc = args.len();
        let padded = args.len().max(formal_min);
        let base = self.reserve(heap, register_count, padded)?;
        let dst = base + register_count + HEADER_SLOTS;
        debug_assert!(args.raw().iter().all(|v| !v.is_weak_ptr()));
        unsafe {
            core::ptr::copy_nonoverlapping(
                args.raw().as_ptr(),
                self.slots.as_ptr().add(dst) as *mut Value,
                args.len(),
            );
            // missing arguments are undefined (registers are the hole)
            let undefined = self.undefined.read(heap);
            for i in args.len()..padded {
                self.slot_unchecked(dst + i).store(undefined);
            }
        }
        Ok(self.init_frame_header(base, register_count, callable, context, new_target, argc))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_frame(
        &self,
        heap: &Heap,
        caller: FrameMeta,
        handler_pc: usize,
        callable: Tagged<'_, Value>,
        register_count: usize,
        context: Tagged<'_, Value>,
        src_reg_base: i32,
        count: usize,
        new_target: Tagged<'_, Value>,
        formal_min: usize,
    ) -> Result<FrameMeta, VmError> {
        let padded = count.max(formal_min);
        let base = self.reserve(heap, register_count, padded)?;
        let src = Self::reg_index(&caller, src_reg_base);
        let dst = base + register_count + HEADER_SLOTS;
        debug_assert!(
            self.slots[src..src + count]
                .iter()
                .all(|r| !r.inner().is_weak_ptr())
        );
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.slots.as_ptr().add(src) as *const Value,
                self.slots.as_ptr().add(dst) as *mut Value,
                count,
            );
            let undefined = self.undefined.read(heap);
            for i in count..padded {
                self.slot_unchecked(dst + i).store(undefined);
            }
        }
        let callee =
            self.init_frame_header(base, register_count, callable, context, new_target, count);
        let mut caller = caller;
        caller.handler_pc = handler_pc;
        self.frames.borrow_mut().push(caller);
        Ok(callee)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_frame_with_args(
        &self,
        heap: &Heap,
        caller: FrameMeta,
        handler_pc: usize,
        callable: Tagged<'_, Value>,
        register_count: usize,
        context: Tagged<'_, Value>,
        args: HandleSlice<'_>,
        new_target: Tagged<'_, Value>,
        formal_min: usize,
    ) -> Result<FrameMeta, VmError> {
        let padded = args.len().max(formal_min);
        let base = self.reserve(heap, register_count, padded)?;
        let dst = base + register_count + HEADER_SLOTS;
        debug_assert!(args.raw().iter().all(|v| !v.is_weak_ptr()));
        unsafe {
            core::ptr::copy_nonoverlapping(
                args.raw().as_ptr(),
                self.slots.as_ptr().add(dst) as *mut Value,
                args.len(),
            );
            let undefined = self.undefined.read(heap);
            for i in args.len()..padded {
                self.slot_unchecked(dst + i).store(undefined);
            }
        }
        let callee = self.init_frame_header(
            base,
            register_count,
            callable,
            context,
            new_target,
            args.len(),
        );
        let mut caller = caller;
        caller.handler_pc = handler_pc;
        self.frames.borrow_mut().push(caller);
        Ok(callee)
    }

    /// Copy `args` into a fresh register region above the current top and
    /// return a `HandleSlice` over it (GC-visited: reads stay fresh across
    /// allocations). Rewind the region with `set_top(saved_top)` when done.
    pub fn stage_args(
        &self,
        heap: &Heap,
        args: HandleSlice<'_>,
    ) -> Result<(usize, HandleSlice<'_>), VmError> {
        let saved_top = self.top();
        let base = self.reserve(heap, 0, args.len())?;
        let dst = base + HEADER_SLOTS;
        // the staged region reserves frame-header slots it never writes:
        // they sit below `top`, so the GC would scan whatever stale words
        // previous frames left there — fill them like fresh registers
        let fill = self.fill.read(heap);
        for i in 0..HEADER_SLOTS {
            self.slot_unchecked(base + i).store(fill);
        }
        unsafe {
            core::ptr::copy_nonoverlapping(
                args.raw().as_ptr(),
                self.slots.as_ptr().add(dst) as *mut Value,
                args.len(),
            )
        }
        // SAFETY: the destination slots are GC roots (Stack is EdgeVisitable)
        let staged = self.value_slice(dst, args.len());
        Ok((saved_top, staged))
    }

    pub fn pop_frame(&self, current_base: usize) -> Option<FrameMeta> {
        self.set_top(current_base);
        self.frames.borrow_mut().pop()
    }

    pub fn frame_depth(&self) -> usize {
        self.frames.borrow().len()
    }

    pub fn suspend_frame(&self, frame: FrameMeta) {
        self.frames.borrow_mut().push(frame);
    }

    pub fn truncate_frames(&self, depth: usize) {
        self.frames.borrow_mut().truncate(depth);
    }

    fn reserve(&self, heap: &Heap, register_count: usize, argc: usize) -> Result<usize, VmError> {
        let base = self.top();
        let size = register_count + HEADER_SLOTS + argc;
        if base + size > self.slots.len() {
            return Err(VmError::StackOverflow);
        }
        let fill = self.fill.read(heap);
        for i in 0..register_count {
            self.slot_unchecked(base + i).store(fill);
        }
        self.set_top(base + size);
        Ok(base)
    }

    fn init_frame_header(
        &self,
        base: usize,
        register_count: usize,
        callable: Tagged<'_, Value>,
        context: Tagged<'_, Value>,
        new_target: Tagged<'_, Value>,
        argc: usize,
    ) -> FrameMeta {
        self.slot_unchecked(base + register_count + CALLABLE_OFFSET)
            .store(callable);
        self.slot_unchecked(base + register_count + ARGC_OFFSET)
            .store(Smi::new(argc as i64).into_tagged());
        self.slot_unchecked(base + register_count + CONTEXT_OFFSET)
            .store(context);
        self.slot_unchecked(base + register_count + NEW_TARGET_OFFSET)
            .store(new_target);
        FrameMeta {
            base,
            pc: 0,
            register_count,
            handler_pc: 0,
        }
    }
}

impl EdgeVisitable for Stack {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.fill.as_raw());
        visitor.visit(self.undefined.as_raw());
        for slot in &self.slots[..self.top()] {
            visitor.visit(slot.as_raw());
        }
    }
}

#[derive(Copy, Clone)]
pub struct FrameMeta {
    pub base: usize,
    /// Resume pc for normal returns.
    pub pc: usize,
    pub register_count: usize,
    /// Pc used for exception handler lookup when unwinding
    pub handler_pc: usize,
}
