use core::cell::{Cell, RefCell};

use crate::{EdgeVisitable, HandleSlice, Heap, Object, Register, Smi, Tagged, Value, Visitor};

use crate::VmError;

/// Number of slots in a thread's interpreter stack.
pub const STACK_SLOTS: usize = 16 * 1024;

/// Fixed header slots between the register file and the parameter region
pub const HEADER_SLOTS: usize = 4;
/// Header slots sit directly below the anchor at these (negative) offsets.
pub const CALLABLE_OFFSET: isize = -1;
pub const ARGC_OFFSET: isize = -2;
/// The frame's current context (the chain LoadContextSlot walks); unlike
/// the function object's closure-context slot it is per-frame, so
/// recursion cannot clobber a suspended frame's context.
pub const CONTEXT_OFFSET: isize = -3;
/// new.target of the active [[Construct]] (ES 9.2.2): the constructor, or
/// undefined when the function was called. `super()` forwards this value.
pub const NEW_TARGET_OFFSET: isize = -4;

const _: () = assert!(
    -bytecode::REGISTER_FILE_START as usize == HEADER_SLOTS + 1,
    "the bytecode register file start must sit directly below the header"
);

fn offset_slot(base: usize, offset: isize) -> usize {
    (base as isize + offset) as usize
}

/// Frame layout — single-anchor layout (anchor = `base`, the analogue
/// of the frame pointer):
/// index                    content
/// base-HEADER-rc .. base-5 register file (r0 at base-5 = directly below
///                          the header, r_i at base-5-i, descending)
/// base-4 .. base-1         header: new.target, context, argc, callable
/// base   + 0               parameter 0 (the receiver)
/// base   + 1 .. +padded-1  parameters (formal j at base+j), padded to
///                          formal_min with undefined
///
/// A register operand IS the anchor-relative slot offset: local `i`
/// encodes as `-(HEADER_SLOTS+1) - i` (descending below the header),
/// parameter `j` as `+j` (ascending above the anchor, receiver = 0).
/// Every register access is one addition — no branch, no runtime frame
/// size. Call argument windows store element 0 (receiver) at the lowest
/// address of the window, so `args` slices are element-ordered and frame
/// pushes are forward memcpys.
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

    /// The register backing store (frame bases index into it; the pointer
    /// is invalidated by frame pushes that grow the store).
    pub fn slots_ptr(&self) -> *mut Register {
        self.slots.as_ptr() as *mut Register
    }

    /// The frame's lowest slot: below the register file and header.
    pub fn frame_low(meta: &FrameMeta) -> usize {
        meta.base - HEADER_SLOTS - meta.register_count
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
        unsafe { self.callable_slot(meta).get(heap).cast::<Object>() }
    }

    pub fn callable_slot(&self, meta: &FrameMeta) -> &Register {
        self.slot_unchecked(offset_slot(meta.base, CALLABLE_OFFSET))
    }

    /// The frame's current context, re-read under a heap borrow.
    pub fn context<'a>(&self, heap: &'a Heap, meta: &FrameMeta) -> Tagged<'a, Value> {
        self.context_slot(meta).get(heap)
    }

    pub fn context_slot(&self, meta: &FrameMeta) -> &Register {
        self.slot_unchecked(offset_slot(meta.base, CONTEXT_OFFSET))
    }

    /// The frame's `new.target`, re-read under a heap borrow.
    pub fn new_target<'a>(&self, heap: &'a Heap, meta: &FrameMeta) -> Tagged<'a, Value> {
        self.new_target_slot(meta).get(heap)
    }

    pub fn new_target_slot(&self, meta: &FrameMeta) -> &Register {
        self.slot_unchecked(offset_slot(meta.base, NEW_TARGET_OFFSET))
    }

    /// The frame's actual argument count, receiver included.
    pub fn argc(&self, meta: &FrameMeta) -> usize {
        self.slot_unchecked(offset_slot(meta.base, ARGC_OFFSET))
            .read_smi()
            .value() as usize
    }

    /// The operand is the anchor-relative slot offset:
    /// one addition, sign-agnostic.
    fn reg_index(meta: &FrameMeta, operand: i32) -> usize {
        (meta.base as isize + operand as isize) as usize
    }

    /// Read a register under a heap borrow: rooted memory is updated in
    /// place by the GC, so the word is current and valid for `'a`.
    pub fn reg<'a>(&self, _heap: &'a Heap, meta: &FrameMeta, i: i32) -> Tagged<'a, Value> {
        self.slot_unchecked(Self::reg_index(meta, i)).get(_heap)
    }

    pub fn set_reg<'x, T: 'x>(&self, meta: &FrameMeta, i: i32, v: Tagged<'x, T>) {
        self.slot_unchecked(Self::reg_index(meta, i)).store(v);
    }

    /// The call-argument window `[reg_base .. reg_base+count)`: element 0
    /// (the receiver) is the base register's window slot at
    /// `reg_base - count + 1`, so the ascending slice is element-ordered.
    pub fn args(&self, meta: &FrameMeta, reg_base: i32, count: usize) -> HandleSlice<'_> {
        if count == 0 {
            return HandleSlice::EMPTY;
        }
        self.value_slice(Self::reg_index(meta, reg_base - count as i32 + 1), count)
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
        let anchor = self.reserve(heap, register_count, padded)?;
        debug_assert!(args.raw().iter().all(|v| !v.is_weak_ptr()));
        // params ascend above the anchor: element j (receiver = 0) at
        // anchor+j — element-ordered, so one forward copy
        unsafe {
            core::ptr::copy_nonoverlapping(
                args.raw().as_ptr(),
                self.slots.as_ptr().add(anchor) as *mut Value,
                args.len(),
            );
            // missing arguments are undefined (registers are the hole)
            let undefined = self.undefined.get(heap);
            for i in args.len()..padded {
                self.slot_unchecked(anchor + i).store(undefined);
            }
        }
        Ok(self.init_frame_header(anchor, register_count, callable, context, new_target, argc))
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
        let anchor = self.reserve(heap, register_count, padded)?;
        // element 0 (receiver) sits at the window's lowest slot: the base
        // operand minus count-1 — one forward copy into the params
        let src = Self::reg_index(&caller, src_reg_base - count as i32 + 1);
        debug_assert!(
            self.slots[src..src + count]
                .iter()
                .all(|r| !r.raw().is_weak_ptr())
        );
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.slots.as_ptr().add(src) as *const Value,
                self.slots.as_ptr().add(anchor) as *mut Value,
                count,
            );
            let undefined = self.undefined.get(heap);
            for i in count..padded {
                self.slot_unchecked(anchor + i).store(undefined);
            }
        }
        let callee =
            self.init_frame_header(anchor, register_count, callable, context, new_target, count);
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
        let anchor = self.reserve(heap, register_count, padded)?;
        debug_assert!(args.raw().iter().all(|v| !v.is_weak_ptr()));
        // params ascend above the anchor, element-ordered: one forward copy
        unsafe {
            core::ptr::copy_nonoverlapping(
                args.raw().as_ptr(),
                self.slots.as_ptr().add(anchor) as *mut Value,
                args.len(),
            );
            let undefined = self.undefined.get(heap);
            for i in args.len()..padded {
                self.slot_unchecked(anchor + i).store(undefined);
            }
        }
        let callee = self.init_frame_header(
            anchor,
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
        let size = HEADER_SLOTS + args.len();
        if saved_top + size > self.slots.len() {
            return Err(VmError::StackOverflow);
        }
        let base = saved_top;
        let dst = base + HEADER_SLOTS;
        // the staged region reserves frame-header slots it never writes:
        // they sit below `top`, so the GC would scan whatever stale words
        // previous frames left there — fill them like fresh registers
        let fill = self.fill.get(heap);
        for i in 0..HEADER_SLOTS {
            self.slot_unchecked(base + i).store(fill);
        }
        self.set_top(base + size);
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

    /// Allocates a frame block `[registers][header][params]` and returns
    /// the anchor.
    fn reserve(&self, heap: &Heap, register_count: usize, padded: usize) -> Result<usize, VmError> {
        let low = self.top();
        let size = register_count + HEADER_SLOTS + padded;
        if low + size > self.slots.len() {
            return Err(VmError::StackOverflow);
        }
        let anchor = low + register_count + HEADER_SLOTS;
        let fill = self.fill.get(heap);
        for i in 0..register_count {
            self.slot_unchecked(low + i).store(fill);
        }
        self.set_top(low + size);
        Ok(anchor)
    }

    fn init_frame_header(
        &self,
        anchor: usize,
        register_count: usize,
        callable: Tagged<'_, Value>,
        context: Tagged<'_, Value>,
        new_target: Tagged<'_, Value>,
        argc: usize,
    ) -> FrameMeta {
        self.slot_unchecked(offset_slot(anchor, CALLABLE_OFFSET))
            .store(callable);
        self.slot_unchecked(offset_slot(anchor, ARGC_OFFSET))
            .store(Smi::new(argc as i64).into_tagged());
        self.slot_unchecked(offset_slot(anchor, CONTEXT_OFFSET))
            .store(context);
        self.slot_unchecked(offset_slot(anchor, NEW_TARGET_OFFSET))
            .store(new_target);
        FrameMeta {
            base: anchor,
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
    /// The addressing anchor (the frame-pointer analogue): params ascend
    /// above it, the header and register file sit below it.
    pub base: usize,
    /// Resume pc for normal returns.
    pub pc: usize,
    pub register_count: usize,
    /// Pc used for exception handler lookup when unwinding
    pub handler_pc: usize,
}
