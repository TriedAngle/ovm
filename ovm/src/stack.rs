use core::cell::{Cell, RefCell};

use vm::{EdgeVisitable, GcSlice, Object, Register, Smi, Tagged, Value, Visitor};

use crate::VmError;

/// Number of slots in a thread's interpreter stack.
pub const STACK_SLOTS: usize = 16 * 1024;

/// Fixed header slots between the register file and the parameter region
pub const HEADER_SLOTS: usize = 3;
pub const CALLABLE_OFFSET: usize = 0;
pub const ARGC_OFFSET: usize = 1;
/// The frame's current context (the chain LoadContextSlot walks); unlike
/// the function object's closure-context slot it is per-frame, so
/// recursion cannot clobber a suspended frame's context.
pub const CONTEXT_OFFSET: usize = 2;

/// Frame layout:
/// index                    content
/// base + 0 .. +rc-1        register file (r0..rn)
/// base + rc + 0            header: callable
/// base + rc + 1            header: argc
/// base + rc + 2            header: current context
/// base + rc + H .. +argc   parameters (param 0 = receiver)
///
/// Parameters are addressed with negative register indices.
///
/// The stack only tracks *suspended* frames: the frame currently being executed
/// lives in the [`StackCache`](crate::StackCache) and is pushed here when a call suspends it.
pub struct Stack {
    slots: Box<[Register]>,
    top: Cell<usize>,
    frames: RefCell<Vec<FrameMeta>>,
    /// Register files of fresh frames are initialized to this value
    /// (the hole: uninitialized `let`/`const` reads must be TDZ errors).
    fill: Value,
}

impl Stack {
    pub fn new(capacity: usize, fill: Value) -> Self {
        Self {
            slots: (0..capacity)
                .map(|_| unsafe { Register::from_value(fill) })
                .collect(),
            top: Cell::new(0),
            frames: RefCell::new(Vec::new()),
            fill,
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

    pub fn value_slice(&self, base: usize, count: usize) -> &[Value] {
        let slots = &self.slots[base..base + count];
        unsafe { core::slice::from_raw_parts(slots.as_ptr() as *const Value, count) }
    }

    pub fn callable_slot(&self, meta: &FrameMeta) -> &Register {
        self.slot_unchecked(meta.base + meta.register_count + CALLABLE_OFFSET)
    }

    pub fn context_slot(&self, meta: &FrameMeta) -> &Register {
        self.slot_unchecked(meta.base + meta.register_count + CONTEXT_OFFSET)
    }

    fn reg_index(meta: &FrameMeta, i: i32) -> usize {
        if i >= 0 {
            meta.base + i as usize
        } else {
            meta.base + meta.register_count + HEADER_SLOTS + (-i - 1) as usize
        }
    }

    pub fn reg(&self, meta: &FrameMeta, i: i32) -> Value {
        self.slot_unchecked(Self::reg_index(meta, i)).inner()
    }

    pub fn set_reg(&self, meta: &FrameMeta, i: i32, v: Value) {
        self.slot_unchecked(Self::reg_index(meta, i)).store(v);
    }

    pub fn args(&self, meta: &FrameMeta, reg_base: i32, count: usize) -> GcSlice<'_> {
        unsafe { GcSlice::from_slice(self.value_slice(Self::reg_index(meta, reg_base), count)) }
    }

    /// `callable` slots[0] is the CallableInfoObject; `context` is the
    /// closure context the frame starts executing with.
    pub fn push_initial_frame(
        &self,
        callable: Tagged<Object>,
        register_count: usize,
        context: Value,
        args: GcSlice<'_>,
    ) -> Result<FrameMeta, VmError> {
        let base = self.reserve(register_count, args.len())?;
        let dst = base + register_count + HEADER_SLOTS;
        debug_assert!(args.as_slice().iter().all(|v| !v.is_weak_ptr()));
        unsafe {
            core::ptr::copy_nonoverlapping(
                args.as_slice().as_ptr(),
                self.slots.as_ptr().add(dst) as *mut Value,
                args.len(),
            )
        }
        Ok(self.init_frame_header(base, register_count, callable, context, args.len()))
    }

    pub fn push_frame(
        &self,
        caller: FrameMeta,
        handler_pc: usize,
        callable: Tagged<Object>,
        register_count: usize,
        context: Value,
        src_reg_base: i32,
        count: usize,
    ) -> Result<FrameMeta, VmError> {
        let base = self.reserve(register_count, count)?;
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
            )
        }
        let callee = self.init_frame_header(base, register_count, callable, context, count);
        let mut caller = caller;
        caller.handler_pc = handler_pc;
        self.frames.borrow_mut().push(caller);
        Ok(callee)
    }

    pub fn push_frame_with_args(
        &self,
        caller: FrameMeta,
        handler_pc: usize,
        callable: Tagged<Object>,
        register_count: usize,
        context: Value,
        args: &[Value],
    ) -> Result<FrameMeta, VmError> {
        let base = self.reserve(register_count, args.len())?;
        let dst = base + register_count + HEADER_SLOTS;
        debug_assert!(args.iter().all(|v| !v.is_weak_ptr()));
        unsafe {
            core::ptr::copy_nonoverlapping(
                args.as_ptr(),
                self.slots.as_ptr().add(dst) as *mut Value,
                args.len(),
            )
        }
        let callee = self.init_frame_header(base, register_count, callable, context, args.len());
        let mut caller = caller;
        caller.handler_pc = handler_pc;
        self.frames.borrow_mut().push(caller);
        Ok(callee)
    }

    /// Copy `args` into a fresh register region above the current top and
    /// return a `GcSlice` over it (GC-visited: reads stay fresh across
    /// allocations). Rewind the region with `set_top(saved_top)` when done.
    pub fn stage_args(&self, args: GcSlice<'_>) -> Result<(usize, GcSlice<'_>), VmError> {
        let saved_top = self.top();
        let base = self.reserve(0, args.len())?;
        let dst = base + HEADER_SLOTS;
        unsafe {
            core::ptr::copy_nonoverlapping(
                args.as_slice().as_ptr(),
                self.slots.as_ptr().add(dst) as *mut Value,
                args.len(),
            )
        }
        // SAFETY: the destination slots are GC roots (Stack is EdgeVisitable)
        let staged = unsafe { GcSlice::from_slice(self.value_slice(dst, args.len())) };
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

    fn reserve(&self, register_count: usize, argc: usize) -> Result<usize, VmError> {
        let base = self.top();
        let size = register_count + HEADER_SLOTS + argc;
        if base + size > self.slots.len() {
            return Err(VmError::StackOverflow);
        }
        // registers are born as the hole (TDZ); arguments are copied over
        // them afterwards
        for i in 0..register_count {
            self.slot_unchecked(base + i).store(self.fill);
        }
        self.set_top(base + size);
        Ok(base)
    }

    fn init_frame_header(
        &self,
        base: usize,
        register_count: usize,
        callable: Tagged<Object>,
        context: Value,
        argc: usize,
    ) -> FrameMeta {
        self.slot_unchecked(base + register_count + CALLABLE_OFFSET)
            .store(callable.erase());
        self.slot_unchecked(base + register_count + ARGC_OFFSET)
            .store(Smi::new(argc as i64).encode());
        self.slot_unchecked(base + register_count + CONTEXT_OFFSET)
            .store(context);
        FrameMeta {
            base,
            pc: 0,
            register_count,
            handler_pc: 0,
        }
    }
}

impl EdgeVisitable for Stack {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
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
