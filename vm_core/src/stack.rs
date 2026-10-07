use core::cell::Cell;

use crate::{
    CallableInfoObject, EdgeVisitable, FeedbackVector, FixedArray, FixedByteArray, Handle,
    HandleSlice, Heap, Object, Register, Smi, Tagged, Value, Visitor,
};

use crate::VmError;

/// Number of slots in a thread's interpreter stack.
pub const STACK_SLOTS: usize = 16 * 1024;

/// Fixed header slots between the register file and the parameter region
pub const HEADER_SLOTS: usize = 11;

const _: () = assert!(
    -bytecode::REGISTER_FILE_START as usize == HEADER_SLOTS + 1,
    "the bytecode register file start must sit directly below the header"
);

/// The pushed frame's function facts (from a `CallTarget::Bytecode`
/// destructure or a call-IC hit); the frame facts (`register_count`,
/// `formal_min`) decode from the info object's packed descriptor.
pub struct Callee<'a> {
    pub callable: Tagged<'a, Value>,
    pub info: Tagged<'a, CallableInfoObject>,
    pub context: Tagged<'a, Value>,
}

/// A rooted argument window: `count` words at `ptr`, element 0 the
/// receiver. The words live in GC-visited slots (the stack arena or a
/// handle scope) which the collector updates in place, so reads are
/// always current across safepoints and the address is stable.
///
/// A missing argument reads as `undefined` (ES clause 18); the actual
/// count rides [`Args::len`] for algorithms that must distinguish
/// "not present".
#[derive(Copy, Clone)]
pub struct Args {
    ptr: core::ptr::NonNull<Value>,
    count: usize,
}

impl Args {
    pub const EMPTY: Args = Args {
        ptr: core::ptr::NonNull::dangling(),
        count: 0,
    };

    /// # Safety
    /// `ptr..ptr+count` must point into GC-visited slots (the stack
    /// arena or a handle scope) that stay rooted for as long as `self`
    /// is used. The safe constructors on `Stack` and `HandleSlice`
    /// uphold this by deriving the pointer from rooted memory.
    #[inline(always)]
    pub unsafe fn from_raw(ptr: core::ptr::NonNull<Value>, count: usize) -> Self {
        debug_assert!(ptr.as_ptr().align_offset(core::mem::align_of::<Value>()) == 0);
        Self { ptr, count }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Element `i`; a missing argument reads as undefined. While the
    /// returned `Tagged` is alive the shared heap borrow holds, so no
    /// `&mut` — no safepoint — can intervene (the GcSlot rule).
    #[inline(always)]
    pub fn get<'h>(&self, heap: &'h Heap, i: usize) -> Tagged<'h, Value> {
        if i >= self.count {
            return heap.known().undefined.as_tagged(heap).erase();
        }
        // Safety: the slot is a rooted, GC-visited word.
        unsafe { Tagged::from_value_unchecked(*self.ptr.as_ptr().add(i)) }
    }

    /// A rooting read: a handle to the argument's own slot in the
    /// window — no scope-slot allocation, valid across safepoints (the
    /// GC updates the window slot in place). Unlike [`Args::get`], the
    /// handle borrows the window, not the heap: it may outlive further
    /// `&mut Heap` calls. A missing argument resolves to the undefined
    /// root handle.
    #[inline(always)]
    pub fn get_handle<'s>(&'s self, heap: &Heap, i: usize) -> Handle<'s, Value> {
        if i >= self.count {
            return heap.known().undefined.erase();
        }
        // Safety: the window slots are GC-visited for the call duration.
        unsafe {
            Handle::from_location(core::ptr::NonNull::new_unchecked(self.ptr.as_ptr().add(i)))
        }
    }

    /// The same window minus its first `n` elements.
    #[inline]
    pub fn slice_from(&self, n: usize) -> Args {
        if n >= self.count {
            return Args::EMPTY;
        }
        Args {
            ptr: unsafe { core::ptr::NonNull::new_unchecked(self.ptr.as_ptr().add(n)) },
            count: self.count - n,
        }
    }

    #[inline]
    pub fn iter<'h>(self, heap: &'h Heap) -> impl Iterator<Item = Tagged<'h, Value>> + 'h {
        (0..self.count).map(move |i| self.get(heap, i))
    }
}

/// What a scattered call site stages at element 0.
#[derive(Copy, Clone)]
pub enum Recv {
    /// the receiver register's word (method call sites)
    Reg(i32),
    /// synthesized undefined (function call sites)
    Undefined,
}

/// The fixed frame header: `HEADER_SLOTS` contiguous slots directly below
/// the anchor. Fields are declared in ascending address order, so the
/// struct starts at `anchor - HEADER_SLOTS` and `callable` lands at
/// `anchor - 1`.
#[repr(C)]
pub struct FrameHeader {
    regcount: Register,
    saved_handler_pc: Register,
    saved_pc: Register,
    saved_base: Register,
    feedback: Register,
    constants: Register,
    code: Register,
    new_target: Register,
    context: Register,
    argc: Register,
    callable: Register,
}

const _: () = assert!(
    core::mem::size_of::<FrameHeader>() == HEADER_SLOTS * core::mem::size_of::<Register>(),
    "the frame header must cover exactly the reserved header slots"
);

impl FrameHeader {
    /// The callable slot, for in-place reads/writes by callers that
    /// cannot route through [`Self::callable`].
    #[inline(always)]
    pub fn callable_slot(&self) -> &Register {
        &self.callable
    }

    /// The context slot, for in-place reads/writes by callers that
    /// cannot route through [`Self::context`].
    #[inline(always)]
    pub fn context_slot(&self) -> &Register {
        &self.context
    }

    /// The `new.target` slot, for in-place reads/writes by callers that
    /// cannot route through [`Self::new_target`].
    #[inline(always)]
    pub fn new_target_slot(&self) -> &Register {
        &self.new_target
    }

    /// The frame's callable.
    #[inline(always)]
    pub fn callable<'h>(&self, heap: &'h Heap) -> Tagged<'h, Object> {
        // Safety: the header slot always holds the frame's callable.
        unsafe { self.callable.get(heap).cast::<Object>() }
    }

    /// The frame's current context.
    #[inline(always)]
    pub fn context<'h>(&self, heap: &'h Heap) -> Tagged<'h, Value> {
        self.context.get(heap)
    }

    /// The active `[[Construct]]` new.target.
    #[inline(always)]
    pub fn new_target<'h>(&self, heap: &'h Heap) -> Tagged<'h, Value> {
        self.new_target.get(heap)
    }

    /// Actual argument count, receiver included.
    #[inline(always)]
    pub fn argc(&self) -> usize {
        self.argc.read_smi().value() as usize
    }

    /// Register-file size.
    #[inline(always)]
    pub fn regcount(&self) -> usize {
        self.regcount.read_smi().value() as usize
    }

    /// The frame's own bytecode.
    #[inline(always)]
    pub fn code<'h>(&self, heap: &'h Heap) -> Tagged<'h, FixedByteArray> {
        // Safety: only `init_frame_header` writes this slot, always the
        // frame's bytecode.
        unsafe { self.code.get(heap).cast() }
    }

    /// The frame's own constant pool.
    #[inline(always)]
    pub fn constants<'h>(&self, heap: &'h Heap) -> Tagged<'h, FixedArray> {
        // Safety: only `init_frame_header` writes this slot, always the
        // constant pool.
        unsafe { self.constants.get(heap).cast() }
    }

    /// The frame's feedback vector, or `None` when absent (the hole).
    #[inline(always)]
    pub fn feedback<'h>(&self, heap: &'h Heap) -> Option<Tagged<'h, FeedbackVector>> {
        // Safety: only `init_frame_header` writes this slot: it is always
        // a `FeedbackVector` or the hole.
        let word = self.feedback.get(heap);
        (word.raw() != heap.known().the_hole.raw()).then(|| unsafe { word.cast() })
    }

    /// The suspended caller's anchor, resume pc and handler pc.
    #[inline(always)]
    pub fn saved(&self) -> (usize, usize, usize) {
        (
            self.saved_base.read_smi().value() as usize,
            self.saved_pc.read_smi().value() as usize,
            self.saved_handler_pc.read_smi().value() as usize,
        )
    }
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

    /// The frame header of the live frame anchored at `base`.
    #[inline(always)]
    pub fn frame_header(&self, base: usize) -> &FrameHeader {
        // Safety: `base` anchors a live frame, whose header occupies the
        // reserved slots directly below it.
        unsafe { &*self.frame_header_ptr(base) }
    }

    /// # Safety
    /// `base` must anchor a live frame (`base >= HEADER_SLOTS`).
    #[inline(always)]
    unsafe fn frame_header_ptr(&self, base: usize) -> *mut FrameHeader {
        debug_assert!(base >= HEADER_SLOTS);
        unsafe {
            (self.slots.as_ptr() as *mut Register)
                .add(base - HEADER_SLOTS)
                .cast::<FrameHeader>()
        }
    }

    /// The frame's callable, re-read under a heap borrow.
    pub fn callable<'a>(&self, heap: &'a Heap, base: usize) -> Tagged<'a, Object> {
        self.frame_header(base).callable(heap)
    }

    /// The frame's current context, re-read under a heap borrow.
    pub fn context<'a>(&self, heap: &'a Heap, base: usize) -> Tagged<'a, Value> {
        self.frame_header(base).context(heap)
    }

    /// The frame's `new.target`, re-read under a heap borrow.
    pub fn new_target<'a>(&self, heap: &'a Heap, base: usize) -> Tagged<'a, Value> {
        self.frame_header(base).new_target(heap)
    }

    /// The frame's actual argument count, receiver included.
    pub fn argc(&self, base: usize) -> usize {
        self.frame_header(base).argc()
    }

    /// The frame's register-file size.
    #[inline(always)]
    pub fn regcount(&self, base: usize) -> usize {
        self.frame_header(base).regcount()
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
    /// `reg_base - count + 1`, so the ascending window is element-ordered.
    #[inline]
    pub fn args(&self, base: usize, reg_base: i32, count: usize) -> Args {
        if count == 0 {
            return Args::EMPTY;
        }
        self.args_at(Self::slot_of(base, reg_base - count as i32 + 1), count)
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
                stack.copy_params(anchor, args.ptr, count);
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

    /// Whole-word, overlap-safe copy from an argument window into the
    /// arena.
    #[inline(always)]
    fn copy_params(&self, dst: usize, src: core::ptr::NonNull<Value>, count: usize) {
        unsafe {
            core::ptr::copy(
                src.as_ptr().cast::<Register>(),
                (self.slots.as_ptr() as *mut Register).add(dst),
                count,
            );
        }
    }

    /// An [`Args`] over `count` words starting at the absolute slot
    /// `src` (rooted by this stack).
    #[inline(always)]
    fn args_at(&self, src: usize, count: usize) -> Args {
        // Safety: derived from this stack's own rooted arena.
        unsafe {
            Args::from_raw(
                core::ptr::NonNull::new(
                    (self.slots.as_ptr() as *const Value).add(src) as *mut Value
                )
                .unwrap_unchecked(),
                count,
            )
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
        Ok(self.args_at(dst, count))
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
            self.copy_params(dst + 1, self.args_at(src, count).ptr, count);
        }
        Ok(self.args_at(dst, 1 + count))
    }

    /// Stage an external slice at the incoming top (execution entry).
    #[inline]
    pub fn stage_slice(&self, values: &[Tagged<'_, Value>]) -> Result<Args, VmError> {
        let dst = self.top();
        if dst + values.len() > self.slots.len() {
            return Err(VmError::StackOverflow);
        }
        self.write_params(dst, values);
        Ok(self.args_at(dst, values.len()))
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
        self.args_at(Self::slot_of(caller_base, base - count as i32 + 1), count)
    }

    /// A rooted slice view over an argument window (the proxy/embedder
    /// boundary).
    #[inline]
    pub fn slice(&self, args: Args) -> HandleSlice<'_> {
        if args.count == 0 {
            return HandleSlice::EMPTY;
        }
        // Safety: `args` covers rooted arena slots for the borrow.
        unsafe {
            HandleSlice::from_slice(core::slice::from_raw_parts(args.ptr.as_ptr(), args.count))
        }
    }

    /// The `top` the stack needs for `args` to sit inside the rooted
    /// region; `None` when it already does (a window inside a frame).
    #[inline]
    pub fn rooting_top(&self, args: Args) -> Option<usize> {
        if args.count == 0 {
            return None;
        }
        let base = self.slots.as_ptr() as usize;
        let end = args.ptr.as_ptr() as usize + args.count * core::mem::size_of::<Value>();
        let needed = (end - base) / core::mem::size_of::<Register>();
        (needed > self.top()).then_some(needed)
    }

    #[inline]
    pub fn pop_frame(&self, base: usize) -> FrameMeta {
        let header = self.frame_header(base);
        self.set_top(base - HEADER_SLOTS - header.regcount());
        let (caller, pc, handler_pc) = header.saved();
        // the caller's own header carries its register count: no saved
        // copy rides in this frame
        FrameMeta {
            base: caller,
            pc,
            register_count: self.regcount(caller),
            handler_pc,
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
        let smi = |v: usize| Smi::new(v as i64).encode();
        // Safety: every header word is a Smi or a strong value; the header
        // region is initialized arena space reserved by `push_frame_with`.
        let header = unsafe {
            FrameHeader {
                regcount: Register::from_value(smi(register_count)),
                saved_handler_pc: Register::from_value(smi(caller.handler_pc)),
                saved_pc: Register::from_value(smi(caller.pc)),
                saved_base: Register::from_value(smi(caller.base)),
                feedback: Register::from_value(feedback),
                constants: Register::from_value(constants),
                code: Register::from_value(bytecode),
                new_target: Register::from_value(new_target.raw()),
                context: Register::from_value(callee.context.raw()),
                argc: Register::from_value(smi(argc)),
                callable: Register::from_value(callee.callable.raw()),
            }
        };
        // Safety: one contiguous write over the reserved header slots.
        unsafe { self.frame_header_ptr(anchor).write(header) };
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
