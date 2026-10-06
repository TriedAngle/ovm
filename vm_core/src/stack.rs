use core::cell::Cell;

use crate::{
    CallableInfoObject, EdgeVisitable, Handle, HandleSlice, Heap, Object, Register, Smi, Tagged,
    Value, Visitor,
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
/// The caller's exception handler lookup pc (its call site).
pub const SAVED_HANDLER_PC_OFFSET: isize = -10;
/// The frame's own register-file size (its frame low is derived from it).
pub const REGCOUNT_OFFSET: isize = -11;

const _: () = assert!(
    -bytecode::REGISTER_FILE_START as usize == HEADER_SLOTS + 1,
    "the bytecode register file start must sit directly below the header"
);

fn offset_slot(base: usize, offset: isize) -> usize {
    (base as isize + offset) as usize
}

/// The pushed frame's function facts (from a `CallTarget::Bytecode`
/// destructure or a call-IC hit); the frame facts (`register_count`,
/// `formal_min`) decode from the info object's packed descriptor.
pub struct Callee<'a> {
    pub callable: Tagged<'a, Value>,
    pub info: Tagged<'a, CallableInfoObject>,
    pub context: Tagged<'a, Value>,
}

/// A normalized argument window: `count` words starting at the
/// absolute slot `src`; element 0 is the receiver.
#[derive(Copy, Clone)]
pub struct Args {
    /// absolute slot index of element 0
    pub src: usize,
    pub count: usize,
}

/// What a scattered call site stages at element 0.
#[derive(Copy, Clone)]
pub enum Recv {
    /// the receiver register's word (method call sites)
    Reg(i32),
    /// synthesized undefined (function call sites)
    Undefined,
}

/// Frame layout
/// index                     content
/// base-HEADER-rc .. base-12 register file (r0 at base-12 = directly below
///                           the header, r_i at base-12-i, descending)
/// base-11 .. base-1         header: new.target, context, argc, callable,
///                           code, constants, feedback, and the suspended
///                           caller's base/pc/handler pc
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
/// Frames are self-describing: dispatch state (code, constants,
/// feedback, accumulator, register count) lives in the frame header
/// itself and a call writes the caller's suspended state into the
/// callee's header, so no parallel frame list or shadow cache exists.
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
    pub fn callable<'a>(&self, heap: &'a Heap, base: usize) -> Tagged<'a, Object> {
        // Safety: frame callable slots hold strong object pointers.
        unsafe { self.callable_slot(base).get(heap).cast::<Object>() }
    }

    pub fn callable_slot(&self, base: usize) -> &Register {
        self.header_slot(base, CALLABLE_OFFSET)
    }

    /// The frame's current context, re-read under a heap borrow.
    pub fn context<'a>(&self, heap: &'a Heap, base: usize) -> Tagged<'a, Value> {
        self.context_slot(base).get(heap)
    }

    #[inline(always)]
    pub fn header_slot(&self, base: usize, offset: isize) -> &Register {
        self.slot_unchecked(offset_slot(base, offset))
    }

    pub fn context_slot(&self, base: usize) -> &Register {
        self.header_slot(base, CONTEXT_OFFSET)
    }

    /// The frame's `new.target`, re-read under a heap borrow.
    pub fn new_target<'a>(&self, heap: &'a Heap, base: usize) -> Tagged<'a, Value> {
        self.new_target_slot(base).get(heap)
    }

    pub fn new_target_slot(&self, base: usize) -> &Register {
        self.header_slot(base, NEW_TARGET_OFFSET)
    }

    /// The frame's actual argument count, receiver included.
    pub fn argc(&self, base: usize) -> usize {
        self.header_slot(base, ARGC_OFFSET).read_smi().value() as usize
    }

    /// The frame's register-file size.
    #[inline(always)]
    pub fn regcount(&self, base: usize) -> usize {
        self.header_slot(base, REGCOUNT_OFFSET).read_smi().value() as usize
    }

    /// The operand is the anchor-relative slot offset
    fn slot_of(base: usize, operand: i32) -> usize {
        (base as isize + operand as isize) as usize
    }

    /// Read a register under a heap borrow: rooted memory is updated in
    /// place by the GC, so the word is current and valid for `'a`.
    pub fn reg<'a>(&self, _heap: &'a Heap, base: usize, i: i32) -> Tagged<'a, Value> {
        self.slot_unchecked(Self::slot_of(base, i)).get(_heap)
    }

    pub fn set_reg<'x, T: 'x>(&self, base: usize, i: i32, v: Tagged<'x, T>) {
        self.slot_unchecked(Self::slot_of(base, i)).store(v);
    }

    /// The call-argument window `[reg_base .. reg_base+count)`: element 0
    /// (the receiver) is the base register's window slot at
    /// `reg_base - count + 1`, so the ascending slice is element-ordered.
    #[inline]
    pub fn args(&self, base: usize, reg_base: i32, count: usize) -> HandleSlice<'_> {
        if count == 0 {
            return HandleSlice::EMPTY;
        }
        self.value_slice(Self::slot_of(base, reg_base - count as i32 + 1), count)
    }

    /// Push a frame: reserve its register file and parameter region
    /// (padded to `formal_min` with undefined), write the parameters, and
    /// initialize the header — including the caller's suspended state.
    /// Entry is at pc 0; the accumulator slot is seeded undefined.
    #[inline(always)]
    pub fn push_frame(
        &self,
        heap: &Heap,
        caller: FrameMeta,
        callee: Callee<'_>,
        new_target: Tagged<'_, Value>,
        args: Args,
    ) -> Result<FrameMeta, VmError> {
        let count = args.count;
        self.push_frame_with(
            heap,
            caller,
            callee,
            new_target,
            count,
            |stack, anchor, _| {
                // params first: a staged source at the frame low would be
                // clobbered by the register fill. Nothing allocates in
                // between, so reading not-yet-rooted words here is sound.
                stack.copy_params(anchor, args.src, count);
            },
        )
    }

    /// Push a callee frame from scattered call-site registers: the
    /// parameters are written straight from the caller's operands.
    #[inline(always)]
    pub fn push_scattered_frame(
        &self,
        heap: &Heap,
        caller: FrameMeta,
        callee: Callee<'_>,
        new_target: Tagged<'_, Value>,
        recv: Recv,
        sargs: [i32; 2],
        argc: usize,
    ) -> Result<FrameMeta, VmError> {
        let count = 1 + argc;
        self.push_frame_with(
            heap,
            caller,
            callee,
            new_target,
            count,
            |stack, anchor, base| {
                match recv {
                    Recv::Reg(recv) => {
                        let src = Self::slot_of(base, recv);
                        stack
                            .slot_unchecked(anchor)
                            .as_raw()
                            .store_raw(stack.slot_unchecked(src).raw().to_bits());
                    }
                    Recv::Undefined => {
                        stack
                            .slot_unchecked(anchor)
                            .store(stack.undefined.get(heap));
                    }
                }
                for (i, &operand) in sargs.iter().enumerate().take(argc) {
                    let src = Self::slot_of(base, operand);
                    stack
                        .slot_unchecked(anchor + 1 + i)
                        .as_raw()
                        .store_raw(stack.slot_unchecked(src).raw().to_bits());
                }
            },
        )
    }

    /// The frame-push core: `write` fills `[anchor, anchor+count)`
    /// from the call site's shape; the pad, register fill and header
    /// init are shared by every entry point.
    #[inline(always)]
    fn push_frame_with<P: FnOnce(&Self, usize, usize)>(
        &self,
        heap: &Heap,
        caller: FrameMeta,
        callee: Callee<'_>,
        new_target: Tagged<'_, Value>,
        count: usize,
        write: P,
    ) -> Result<FrameMeta, VmError> {
        let (register_count, formal_min) = callee.info.as_ref().frame_facts();
        let padded = count.max(formal_min);
        let low = self.top();
        if low + register_count + HEADER_SLOTS + padded > self.slots.len() {
            return Err(VmError::StackOverflow);
        }
        let anchor = low + register_count + HEADER_SLOTS;
        write(self, anchor, caller.base);
        let undefined = self.undefined.get(heap);
        for i in count..padded {
            self.slot_unchecked(anchor + i).store(undefined);
        }
        let fill = self.fill.get(heap);
        for i in 0..register_count {
            self.slot_unchecked(low + i).store(fill);
        }
        self.set_top(anchor + padded);
        Ok(self.init_frame_header(
            heap,
            anchor,
            &callee,
            new_target,
            count,
            caller,
            register_count,
        ))
    }

    /// Copy whole parameter words into the arena: the destination slots
    /// are GC roots, the staged source is rooted by its staging scope.
    #[inline(always)]
    fn write_params(&self, dst: usize, values: &[Tagged<'_, Value>]) {
        if values.is_empty() {
            return;
        }
        // Safety: Tagged<Value> and Register are whole-word layout twins.
        unsafe {
            core::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<Register>(),
                (self.slots.as_ptr() as *mut Register).add(dst),
                values.len(),
            );
        }
    }

    /// Whole-word, overlap-safe copy between arena slots.
    #[inline(always)]
    fn copy_params(&self, dst: usize, src: usize, count: usize) {
        unsafe {
            core::ptr::copy(
                self.slots.as_ptr().add(src),
                (self.slots.as_ptr() as *mut Register).add(dst),
                count,
            );
        }
    }

    /// Stage scattered call operands into one contiguous
    /// `[receiver, args...]` window at the incoming top without bumping
    /// it: the words are outside the GC's scan until the consumer bumps
    /// `top` over them — nothing may allocate in between.
    #[inline]
    pub fn stage_scattered(
        &self,
        heap: &Heap,
        caller_base: usize,
        recv: Recv,
        args: [i32; 2],
        argc: usize,
    ) -> Result<Args, VmError> {
        let dst = self.top();
        let count = 1 + argc;
        if dst + count > self.slots.len() {
            return Err(VmError::StackOverflow);
        }
        match recv {
            Recv::Undefined => {
                self.slot_unchecked(dst).store(self.undefined.get(heap));
            }
            Recv::Reg(recv) => {
                let src = Self::slot_of(caller_base, recv);
                self.slot_unchecked(dst)
                    .as_raw()
                    .store_raw(self.slot_unchecked(src).raw().to_bits());
            }
        }
        for (i, &operand) in args.iter().enumerate().take(argc) {
            let src = Self::slot_of(caller_base, operand);
            self.slot_unchecked(dst + 1 + i)
                .as_raw()
                .store_raw(self.slot_unchecked(src).raw().to_bits());
        }
        Ok(Args { src: dst, count })
    }

    /// Stage `[receiver, window...]` at the incoming top (construct
    /// pushes and runtime constructors).
    #[inline]
    pub fn stage_construct(
        &self,
        caller_base: usize,
        receiver: Tagged<'_, Value>,
        base: i32,
        count: usize,
    ) -> Result<Args, VmError> {
        let dst = self.top();
        if dst + 1 + count > self.slots.len() {
            return Err(VmError::StackOverflow);
        }
        self.slot_unchecked(dst).store(receiver);
        if count > 0 {
            let src = Self::slot_of(caller_base, base - count as i32 + 1);
            self.copy_params(dst + 1, src, count);
        }
        Ok(Args {
            src: dst,
            count: 1 + count,
        })
    }

    /// Stage an external slice at the incoming top (execution entry).
    #[inline]
    pub fn stage_slice(&self, values: &[Tagged<'_, Value>]) -> Result<Args, VmError> {
        let dst = self.top();
        if dst + values.len() > self.slots.len() {
            return Err(VmError::StackOverflow);
        }
        self.write_params(dst, values);
        Ok(Args {
            src: dst,
            count: values.len(),
        })
    }

    /// A handle over a caller register slot: rooted by the live frame
    /// for the borrow.
    #[inline]
    pub fn reg_handle<'s>(&'s self, base: usize, i: i32) -> Handle<'s, Value> {
        let slot = self.slot_unchecked(Self::slot_of(base, i));
        // Safety: Register and Value are whole-word layout twins; the
        // slot is a rooted, GC-visited word for `'s`.
        unsafe {
            let word: &Value = &*(slot as *const Register).cast::<Value>();
            Handle::from_location(core::ptr::NonNull::from(word))
        }
    }

    /// The caller's call window `[base-count+1 .. base]` as an [`Args`].
    #[inline]
    pub fn window(&self, caller_base: usize, base: i32, count: usize) -> Args {
        Args {
            src: Self::slot_of(caller_base, base - count as i32 + 1),
            count,
        }
    }

    /// A rooted slice view over a normalized argument window.
    #[inline]
    pub fn slice(&self, args: Args) -> HandleSlice<'_> {
        if args.count == 0 {
            return HandleSlice::EMPTY;
        }
        self.value_slice(args.src, args.count)
    }

    #[inline]
    pub fn pop_frame(&self, base: usize) -> FrameMeta {
        self.set_top(base - HEADER_SLOTS - self.regcount(base));
        let read = |offset: isize| self.header_slot(base, offset).read_smi().value() as usize;
        let caller = read(SAVED_BASE_OFFSET);
        // the caller's own header carries its register count: no saved
        // copy rides in this frame
        FrameMeta {
            base: caller,
            pc: read(SAVED_PC_OFFSET),
            register_count: self.regcount(caller),
            handler_pc: read(SAVED_HANDLER_PC_OFFSET),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn init_frame_header(
        &self,
        heap: &Heap,
        anchor: usize,
        callee: &Callee<'_>,
        new_target: Tagged<'_, Value>,
        argc: usize,
        caller: FrameMeta,
        register_count: usize,
    ) -> FrameMeta {
        let info = callee.info.as_ref();
        let bytecode = info.bytecode.get(heap).raw();
        let constants = info.constants.get(heap).raw();
        let feedback = info
            .feedback
            .get(heap)
            .map_or_else(|| heap.known().the_hole.as_tagged(heap).raw(), |v| v.raw());
        let saved = [
            (CALLABLE_OFFSET, callee.callable.raw()),
            (ARGC_OFFSET, Smi::new(argc as i64).encode()),
            (CONTEXT_OFFSET, callee.context.raw()),
            (NEW_TARGET_OFFSET, new_target.raw()),
            (CODE_OFFSET, bytecode),
            (CONSTANTS_OFFSET, constants),
            (FEEDBACK_OFFSET, feedback),
            (SAVED_BASE_OFFSET, Smi::new(caller.base as i64).encode()),
            (SAVED_PC_OFFSET, Smi::new(caller.pc as i64).encode()),
            (
                SAVED_HANDLER_PC_OFFSET,
                Smi::new(caller.handler_pc as i64).encode(),
            ),
            (REGCOUNT_OFFSET, Smi::new(register_count as i64).encode()),
        ];
        // the fixed header words sit contiguously below the anchor
        // (offsets -1..=-11): plain offset writes through one held base
        // instead of re-deriving the slot address through `self` per field
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

impl FrameMeta {
    /// The bootstrap caller of an execution's initial frame: no state is
    /// ever resumed from it (returning past the anchor escapes).
    pub const ROOT: FrameMeta = FrameMeta {
        base: 0,
        pc: 0,
        register_count: 0,
        handler_pc: 0,
    };
}
