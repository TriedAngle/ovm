use core::cell::Cell;

use crate::{
    CallableInfoObject, EdgeVisitable, HandleSlice, Heap, Object, Register, Smi, Tagged, Value,
    Visitor,
};

use crate::VmError;

/// Number of slots in a thread's interpreter stack.
pub const STACK_SLOTS: usize = 16 * 1024;

/// Fixed header slots between the register file and the parameter region
pub const HEADER_SLOTS: usize = 11;
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
/// The frame's own bytecode: dispatch state lives in the frame
pub const CODE_OFFSET: isize = -5;
/// The frame's own constant pool.
pub const CONSTANTS_OFFSET: isize = -6;
/// The frame's own feedback vector (or the hole when absent).
pub const FEEDBACK_OFFSET: isize = -7;
/// The suspended caller's anchor (its frame pointer): returning to a frame
/// is a header read, not a side-table lookup.
pub const SAVED_BASE_OFFSET: isize = -8;
/// The caller's resume pc (the instruction after its call).
pub const SAVED_PC_OFFSET: isize = -9;
/// The caller's register-file size (its frame low is derived from it).
pub const SAVED_REGCOUNT_OFFSET: isize = -10;
/// The caller's exception handler lookup pc (its call site).
pub const SAVED_HANDLER_PC_OFFSET: isize = -11;

const _: () = assert!(
    -bytecode::REGISTER_FILE_START as usize == HEADER_SLOTS + 1,
    "the bytecode register file start must sit directly below the header"
);

fn offset_slot(base: usize, offset: isize) -> usize {
    (base as isize + offset) as usize
}

/// Frame layout
/// index                     content
/// base-HEADER-rc .. base-12 register file (r0 at base-12 = directly below
///                           the header, r_i at base-12-i, descending)
/// base-11 .. base-1         header: new.target, context, argc, callable,
///                           code, constants, feedback, and the suspended
///                           caller's base/pc/register count/handler pc
/// base   + 0                parameter 0 (the receiver)
/// base   + 1 .. +padded-1   parameters (formal j at base+j), padded to
///                           formal_min with undefined
///
/// A register operand IS the anchor-relative slot offset: local `i`
/// encodes as `-(HEADER_SLOTS+1) - i` (descending below the header),
/// parameter `j` as `+j` (ascending above the anchor, receiver = 0).
/// Every register access is one addition — no branch, no runtime frame
/// size. Call argument windows store element 0 (receiver) at the lowest
/// address of the window, so `args` slices are element-ordered and frame
/// pushes are forward memcpys.
///
/// Frames are self-describing: the currently executing frame's dispatch
/// state lives in the [`StackCache`](StackCache), and a call writes the
/// caller's suspended state into the callee's header, so no parallel frame list exists.
pub struct Stack {
    slots: Box<[Register]>,
    top: Cell<usize>,
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
            fill: unsafe { Register::from_value(fill) },
            undefined: unsafe { Register::from_value(undefined) },
        }
    }

    pub fn top(&self) -> usize {
        self.top.get()
    }

    /// The undefined word from the stack's own rooted cell — a single
    /// Register read (GC-updated in place), for interpreter entry paths.
    pub fn undefined_word<'a>(&self, heap: &'a Heap) -> Tagged<'a, Value> {
        self.undefined.get(heap)
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

    #[inline]
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

    #[inline(always)]
    pub fn header_slot(&self, base: usize, offset: isize) -> &Register {
        self.slot_unchecked(offset_slot(base, offset))
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

    /// The operand is the anchor-relative slot offset
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
    #[inline]
    pub fn args(&self, meta: &FrameMeta, reg_base: i32, count: usize) -> HandleSlice<'_> {
        if count == 0 {
            return HandleSlice::EMPTY;
        }
        self.value_slice(Self::reg_index(meta, reg_base - count as i32 + 1), count)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_initial_frame(
        &self,
        heap: &Heap,
        callable: Tagged<'_, Value>,
        info: Tagged<'_, CallableInfoObject>,
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
        Ok(self.init_frame_header(
            heap,
            anchor,
            register_count,
            callable,
            info,
            context,
            new_target,
            argc,
            FrameMeta {
                base: 0,
                pc: 0,
                register_count: 0,
                handler_pc: 0,
            },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn push_frame(
        &self,
        heap: &Heap,
        caller: FrameMeta,
        handler_pc: usize,
        callable: Tagged<'_, Value>,
        info: Tagged<'_, CallableInfoObject>,
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
        let mut caller = caller;
        caller.handler_pc = handler_pc;
        Ok(self.init_frame_header(
            heap,
            anchor,
            register_count,
            callable,
            info,
            context,
            new_target,
            count,
            caller,
        ))
    }

    /// Push a `[[Construct]]` frame: the receiver is *synthesized* (not in
    /// the caller's registers), so it is seeded at the anchor and the
    /// contiguous argument range `base-count+1..=base` follows it.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn push_construct_frame(
        &self,
        heap: &Heap,
        caller: FrameMeta,
        handler_pc: usize,
        callable: Tagged<'_, Value>,
        info: Tagged<'_, CallableInfoObject>,
        register_count: usize,
        context: Tagged<'_, Value>,
        src_reg_base: i32,
        count: usize,
        new_target: Tagged<'_, Value>,
        receiver: Tagged<'_, Value>,
        formal_min: usize,
    ) -> Result<FrameMeta, VmError> {
        let total = count + 1;
        let padded = total.max(formal_min);
        let anchor = self.reserve(heap, register_count, padded)?;
        self.slot_unchecked(anchor)
            .as_raw()
            .store_raw(receiver.raw().to_bits());
        let src = Self::reg_index(&caller, src_reg_base - count as i32 + 1);
        debug_assert!(
            count == 0
                || self.slots[src..src + count]
                    .iter()
                    .all(|r| !r.raw().is_weak_ptr())
        );
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.slots.as_ptr().add(src) as *const Value,
                self.slots.as_ptr().add(anchor + 1) as *mut Value,
                count,
            );
        }
        let undefined = self.undefined.get(heap);
        for i in total..padded {
            self.slot_unchecked(anchor + i).store(undefined);
        }
        let mut caller = caller;
        caller.handler_pc = handler_pc;
        Ok(self.init_frame_header(
            heap,
            anchor,
            register_count,
            callable,
            info,
            context,
            new_target,
            total,
            caller,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_frame_with_args(
        &self,
        heap: &Heap,
        caller: FrameMeta,
        handler_pc: usize,
        callable: Tagged<'_, Value>,
        info: Tagged<'_, CallableInfoObject>,
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
        let mut caller = caller;
        caller.handler_pc = handler_pc;
        Ok(self.init_frame_header(
            heap,
            anchor,
            register_count,
            callable,
            info,
            context,
            new_target,
            args.len(),
            caller,
        ))
    }

    /// Stage a construct argument window for a runtime constructor:
    /// `[receiver(=undefined), args...]` — runtime fns index element 0 as
    /// the receiver, constructs pass none.
    pub fn stage_construct_args(
        &self,
        heap: &Heap,
        args: HandleSlice<'_>,
    ) -> Result<(usize, HandleSlice<'_>), VmError> {
        let saved_top = self.top();
        let size = HEADER_SLOTS + 1 + args.len();
        if saved_top + size > self.slots.len() {
            return Err(VmError::StackOverflow);
        }
        let base = saved_top;
        let dst = base + HEADER_SLOTS;
        let fill = self.fill.get(heap);
        for i in 0..HEADER_SLOTS {
            self.slot_unchecked(base + i).store(fill);
        }
        let undefined = self.undefined.get(heap);
        self.slot_unchecked(dst).store(undefined);
        if !args.is_empty() {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    args.raw().as_ptr(),
                    (self.slots.as_ptr() as *mut Value).add(dst + 1),
                    args.len(),
                )
            }
        }
        self.set_top(base + size);
        let staged = self.value_slice(dst, 1 + args.len());
        Ok((saved_top, staged))
    }

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

    #[inline]
    pub fn stage_args_regs<'s>(
        &'s self,
        heap: &Heap,
        caller: &FrameMeta,
        srcs: &[i32],
    ) -> Result<(usize, HandleSlice<'s>), VmError> {
        let saved_top = self.top();
        let size = HEADER_SLOTS + srcs.len();
        if saved_top + size > self.slots.len() {
            return Err(VmError::StackOverflow);
        }
        let base = saved_top;
        let dst = base + HEADER_SLOTS;
        let fill = self.fill.get(heap);
        for i in 0..HEADER_SLOTS {
            self.slot_unchecked(base + i).store(fill);
        }
        self.set_top(base + size);
        for (i, &operand) in srcs.iter().enumerate() {
            let src = Self::reg_index(caller, operand);
            self.slot_unchecked(dst + i)
                .as_raw()
                .store_raw(self.slot_unchecked(src).raw().to_bits());
        }
        // SAFETY: the destination slots are GC roots (Stack is EdgeVisitable)
        let staged = self.value_slice(dst, srcs.len());
        Ok((saved_top, staged))
    }

    #[inline]
    pub fn stage_function_args<'s>(
        &'s self,
        heap: &Heap,
        caller: &FrameMeta,
        srcs: &[i32],
    ) -> Result<(usize, HandleSlice<'s>), VmError> {
        let saved_top = self.top();
        let size = HEADER_SLOTS + 1 + srcs.len();
        if saved_top + size > self.slots.len() {
            return Err(VmError::StackOverflow);
        }
        let base = saved_top;
        let dst = base + HEADER_SLOTS;
        let fill = self.fill.get(heap);
        for i in 0..HEADER_SLOTS {
            self.slot_unchecked(base + i).store(fill);
        }
        self.set_top(base + size);
        self.slot_unchecked(dst).store(self.undefined.get(heap));
        for (i, &operand) in srcs.iter().enumerate() {
            let src = Self::reg_index(caller, operand);
            self.slot_unchecked(dst + 1 + i)
                .as_raw()
                .store_raw(self.slot_unchecked(src).raw().to_bits());
        }
        // SAFETY: the destination slots are GC roots (Stack is EdgeVisitable)
        let staged = self.value_slice(dst, 1 + srcs.len());
        Ok((saved_top, staged))
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn push_frame_function(
        &self,
        heap: &Heap,
        caller: FrameMeta,
        handler_pc: usize,
        callable: Tagged<'_, Value>,
        info: Tagged<'_, CallableInfoObject>,
        register_count: usize,
        context: Tagged<'_, Value>,
        args: &[i32],
        new_target: Tagged<'_, Value>,
        formal_min: usize,
    ) -> Result<FrameMeta, VmError> {
        let count = 1 + args.len();
        let padded = count.max(formal_min);
        let anchor = self.reserve(heap, register_count, padded)?;
        self.slot_unchecked(anchor).store(self.undefined.get(heap));
        for (i, &operand) in args.iter().enumerate() {
            let src = Self::reg_index(&caller, operand);
            debug_assert!(!self.slot_unchecked(src).raw().is_weak_ptr());
            self.slot_unchecked(anchor + 1 + i)
                .as_raw()
                .store_raw(self.slot_unchecked(src).raw().to_bits());
        }
        let undefined = self.undefined.get(heap);
        for i in count..padded {
            self.slot_unchecked(anchor + i).store(undefined);
        }
        let mut caller = caller;
        caller.handler_pc = handler_pc;
        Ok(self.init_frame_header(
            heap,
            anchor,
            register_count,
            callable,
            info,
            context,
            new_target,
            count,
            caller,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn push_frame_scattered(
        &self,
        heap: &Heap,
        caller: FrameMeta,
        handler_pc: usize,
        callable: Tagged<'_, Value>,
        info: Tagged<'_, CallableInfoObject>,
        register_count: usize,
        context: Tagged<'_, Value>,
        srcs: &[i32],
        new_target: Tagged<'_, Value>,
        formal_min: usize,
    ) -> Result<FrameMeta, VmError> {
        let count = srcs.len();
        let padded = count.max(formal_min);
        let anchor = self.reserve(heap, register_count, padded)?;
        for (i, &operand) in srcs.iter().enumerate() {
            let src = Self::reg_index(&caller, operand);
            debug_assert!(!self.slot_unchecked(src).raw().is_weak_ptr());
            self.slot_unchecked(anchor + i)
                .as_raw()
                .store_raw(self.slot_unchecked(src).raw().to_bits());
        }
        let undefined = self.undefined.get(heap);
        for i in count..padded {
            self.slot_unchecked(anchor + i).store(undefined);
        }
        let mut caller = caller;
        caller.handler_pc = handler_pc;
        Ok(self.init_frame_header(
            heap,
            anchor,
            register_count,
            callable,
            info,
            context,
            new_target,
            count,
            caller,
        ))
    }

    #[inline]
    pub fn pop_frame(&self, frame: &FrameMeta) -> FrameMeta {
        self.set_top(Self::frame_low(frame));
        let read = |offset: isize| {
            self.header_slot(frame.base, offset)
                .read_smi_unchecked()
                .value() as usize
        };
        FrameMeta {
            base: read(SAVED_BASE_OFFSET),
            pc: read(SAVED_PC_OFFSET),
            register_count: read(SAVED_REGCOUNT_OFFSET),
            handler_pc: read(SAVED_HANDLER_PC_OFFSET),
        }
    }

    #[inline(always)]
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

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn init_frame_header(
        &self,
        heap: &Heap,
        anchor: usize,
        register_count: usize,
        callable: Tagged<'_, Value>,
        info: Tagged<'_, CallableInfoObject>,
        context: Tagged<'_, Value>,
        new_target: Tagged<'_, Value>,
        argc: usize,
        caller: FrameMeta,
    ) -> FrameMeta {
        // one held base for the whole header: the fixed header words sit
        // contiguously below the anchor (offsets -1..-11), so the stores
        // compile to plain offset writes instead of re-deriving the slot
        // address through `self` per field
        let info = info.as_ref();
        let bytecode = info.bytecode.get(heap).raw();
        let constants = info.constants.get(heap).raw();
        let feedback = info
            .feedback
            .get(heap)
            .map_or_else(|| heap.known().the_hole.as_tagged(heap).raw(), |v| v.raw());
        let saved = [
            (CALLABLE_OFFSET, callable.raw()),
            (ARGC_OFFSET, Smi::new(argc as i64).encode()),
            (CONTEXT_OFFSET, context.raw()),
            (NEW_TARGET_OFFSET, new_target.raw()),
            (CODE_OFFSET, bytecode),
            (CONSTANTS_OFFSET, constants),
            (FEEDBACK_OFFSET, feedback),
            (SAVED_BASE_OFFSET, Smi::new(caller.base as i64).encode()),
            (SAVED_PC_OFFSET, Smi::new(caller.pc as i64).encode()),
            (
                SAVED_REGCOUNT_OFFSET,
                Smi::new(caller.register_count as i64).encode(),
            ),
            (
                SAVED_HANDLER_PC_OFFSET,
                Smi::new(caller.handler_pc as i64).encode(),
            ),
        ];
        unsafe {
            let base = (self.slots.as_ptr() as *mut Value).add(anchor);
            for (offset, word) in saved {
                *base.offset(offset) = word;
            }
        }
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
