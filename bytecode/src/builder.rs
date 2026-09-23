use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::program::{Constant, Function, HandlerEntry};
use crate::{CallableKind, Opcode, Operand, emit};

/// One staged argument of a runtime call: where the value for a window
/// slot comes from. [`RtArg::Acc`] captures the accumulator's current
/// value — always before any loads run, so later `Reg`/`Const`/`Smi`
/// entries may clobber the accumulator freely.
#[derive(Debug, Clone, Copy)]
pub enum RtArg {
    Acc,
    Reg(Reg),
    Const(ConstIdx),
    Smi(u32),
}

impl FnBuilder {
    /// Emit a `CallRuntime` with its argument window staged from a
    /// declarative list: `&[RtArg::Acc, RtArg::Reg(obj), RtArg::Const(k)]`
    /// stages the accumulator into slot 0, `obj` into slot 1, and the
    /// pooled constant into slot 2, then calls.
    pub fn call_runtime_staged(&mut self, f: crate::RuntimeFn, args: &[RtArg]) {
        let mark = self.temp_depth();
        let base = self.reserve_temps(args.len() as u32);
        if let Some(i) = args.iter().position(|a| matches!(a, RtArg::Acc)) {
            self.store(Reg::new(base.index() + i as i32));
        }
        for (i, arg) in args.iter().enumerate() {
            let dst = Reg::new(base.index() + i as i32);
            match arg {
                RtArg::Acc => {}
                RtArg::Reg(r) => {
                    self.load(*r);
                    self.store(dst);
                }
                RtArg::Const(c) => {
                    self.load_constant(*c);
                    self.store(dst);
                }
                RtArg::Smi(v) => {
                    self.load_smi(*v as i32);
                    self.store(dst);
                }
            }
        }
        self.call_runtime(f, RegList::new(base, args.len() as u32));
        self.drop_temps(mark);
    }
}

/// Emits an operand-less opcode that writes the accumulator.
macro_rules! acc_void {
    ($name:ident, $opcode:ident) => {
        pub fn $name(&mut self) {
            self.emit_tracked(Opcode::$opcode, &[]);
        }
    };
}

/// Emits an operand-less terminal opcode that reads the accumulator.
macro_rules! acc_terminal {
    ($name:ident, $opcode:ident) => {
        pub fn $name(&mut self) {
            self.emit_tracked(Opcode::$opcode, &[]);
            self.acc = Acc::Dead;
        }
    };
}

/// Emits a well-known-singleton load with elision.
macro_rules! singleton_load {
    ($name:ident, $opcode:ident, $acc:ident) => {
        pub fn $name(&mut self) {
            if self.acc == Acc::$acc {
                return;
            }
            self.emit_tracked(Opcode::$opcode, &[]);
            self.acc = Acc::$acc;
        }
    };
}

/// Emits `acc = acc <op> r`.
macro_rules! acc_reg_op {
    ($name:ident, $opcode:ident) => {
        pub fn $name(&mut self, r: Reg) {
            self.emit_tracked(Opcode::$opcode, &[r.operand()]);
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(i32);

impl Reg {
    pub const fn new(index: i32) -> Self {
        Self(index)
    }

    pub const fn index(self) -> i32 {
        self.0
    }

    /// The operand encoding of this register (two's complement).
    pub const fn operand(self) -> u32 {
        self.0 as u32
    }
}

/// Constant-pool index handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstIdx(u32);

impl ConstIdx {
    pub fn index(self) -> u32 {
        self.0
    }
}

/// Feedback-slot handle: the base of one `[state, handler]` inline-cache pair
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Feedback(u32);

impl Feedback {
    pub fn index(self) -> u32 {
        self.0
    }
}

/// A contiguous register range for call arguments:
/// `base`, `base + 1`, ..., `base + count - 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegList {
    pub base: Reg,
    pub count: u32,
}

impl RegList {
    pub const fn new(base: Reg, count: u32) -> Self {
        Self { base, count }
    }

    fn operands(self) -> [u32; 2] {
        [self.base.operand(), self.count]
    }
}

/// A not-yet-known bytecode position: jump targets are bound with
/// [`FnBuilder::bind`] and resolved during [`FnBuilder::finish`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Label {
    id: u32,
    origin: u32,
}

/// A try/catch range under construction: [`FnBuilder::begin_try`] anchors
/// the region start, [`FnBuilder::end_try`] its exclusive end, and
/// [`FnBuilder::handler_entry`] the pc the exception transfers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TryBlock {
    id: u32,
    origin: u32,
}

/// Execution metadata for the finished [`Function`].
#[derive(Debug, Clone, Default)]
pub struct FunctionMeta {
    pub name: Option<Box<[u8]>>,
    pub kind: CallableKind,
    /// JS-visible `length`: parameters before the first default/rest/pattern.
    pub length: u32,
    pub strict: bool,
}

/// A function failed to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildError {
    /// A jump targets a label that was never bound.
    UnboundLabel,
    /// A try range is missing its `end_try` or `handler_entry` anchor.
    UnfinishedHandler,
    /// Temporary registers were still allocated when `finish` ran.
    UnbalancedTemps,
    /// A jump offset exceeds the widened (16-bit) range.
    JumpOutOfRange,
}

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnboundLabel => f.write_str("jump targets a label that was never bound"),
            Self::UnfinishedHandler => f.write_str("try range missing end_try/handler_entry"),
            Self::UnbalancedTemps => f.write_str("temporary registers still allocated at finish"),
            Self::JumpOutOfRange => f.write_str("jump offset exceeds the 16-bit range"),
        }
    }
}

impl std::error::Error for BuildError {}

/// What the accumulator provably holds at the current emission position.
/// `Dead` means "not defined here"
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Acc {
    Dead,
    Unknown,
    Reg(i32),
    Const(u32),
    Smi(i32),
    Zero,
    Undefined,
    Null,
    True,
    False,
    Hole,
}

#[derive(Debug)]
struct JumpRec {
    /// Emission-buffer position of the (narrow placeholder) jump.
    at: usize,
    op: Opcode,
    target: u32,
}

#[derive(Debug, Default)]
struct LabelRec {
    bind: Option<usize>,
}

#[derive(Debug)]
struct HandlerRec {
    try_start: usize,
    try_end: Option<usize>,
    handler_pc: Option<usize>,
}

/// Layout event: a position whose final offset must be known
#[derive(Clone, Copy)]
enum Ev {
    Label(u32),
    TryStart(u32),
    TryEnd(u32),
    Handler(u32),
    Jump(u32),
}

static BUILDER_GEN: AtomicU32 = AtomicU32::new(1);

/// Emits the bytecode of one function.
#[derive(Debug)]
pub struct FnBuilder {
    origin: u32,
    arity: u32,

    // emission state
    code: Vec<u8>,
    jumps: Vec<JumpRec>,
    labels: Vec<LabelRec>,
    handlers: Vec<HandlerRec>,

    // pools
    constants: Vec<Constant>,
    string_pool: HashMap<Box<[u8]>, u32>,
    other_pool: HashMap<Constant, u32>,
    feedback_slots: u32,

    // accumulator analysis
    acc: Acc,

    // register allocation
    temp_base: u32,
    temp_depth: u32,
    max_reg: i32,
}

impl FnBuilder {
    /// Start a function with `arity` formal parameters (addressable as
    /// `param(0)..param(arity - 1)`, i.e. registers `-2..=-(arity + 1)`;
    /// the receiver is always register `-1`, see [`FnBuilder::this_reg`]).
    pub fn new(arity: u32) -> Self {
        Self {
            origin: BUILDER_GEN.fetch_add(1, Ordering::Relaxed),
            arity,
            code: Vec::new(),
            jumps: Vec::new(),
            labels: Vec::new(),
            handlers: Vec::new(),
            constants: Vec::new(),
            string_pool: HashMap::new(),
            other_pool: HashMap::new(),
            feedback_slots: 0,
            acc: Acc::Dead,
            temp_base: 0,
            temp_depth: 0,
            max_reg: -1,
        }
    }

    /// The first register index handed out by [`FnBuilder::temp`]: set this
    /// to one past the frontend's own locals so temps never collide.
    pub fn set_temp_base(&mut self, base: u32) {
        debug_assert_eq!(self.temp_depth, 0, "temp base set with temps live");
        self.temp_base = base;
    }

    /// The receiver register (`this`): parameter-area slot 0.
    pub fn this_reg(&self) -> Reg {
        Reg(-1)
    }

    /// The register holding formal parameter `i`: `-2`-based within the
    /// parameter area, after the receiver at `-1` (`i` must be below
    /// `arity`).
    pub fn param(&self, i: u32) -> Reg {
        debug_assert!(
            i < self.arity,
            "parameter {i} out of range (arity {})",
            self.arity
        );
        Reg(-(i as i32) - 2)
    }

    /// Intern a constant, reusing an equal existing pool entry.
    ///
    /// Floats compare bit-exact, so `0.0` and `-0.0` stay distinct and a
    /// NaN literal re-dedups with itself.
    pub fn constant(&mut self, c: Constant) -> ConstIdx {
        if let Constant::String(bytes) = &c {
            if let Some(idx) = self.string_pool.get(bytes.as_ref()) {
                return ConstIdx(*idx);
            }
            let idx = self.constants.len() as u32;
            self.string_pool.insert(bytes.clone(), idx);
            self.constants.push(c);
            return ConstIdx(idx);
        }
        if let Some(idx) = self.other_pool.get(&c) {
            return ConstIdx(*idx);
        }
        let idx = self.constants.len() as u32;
        self.other_pool.insert(c.clone(), idx);
        self.constants.push(c);
        ConstIdx(idx)
    }

    /// Intern a string constant by borrowed bytes.
    pub fn name(&mut self, bytes: &[u8]) -> ConstIdx {
        if let Some(idx) = self.string_pool.get(bytes) {
            return ConstIdx(*idx);
        }
        let idx = self.constants.len() as u32;
        self.constants.push(Constant::String(bytes.into()));
        self.string_pool.insert(bytes.into(), idx);
        ConstIdx(idx)
    }

    /// Number of pool entries interned so far.
    pub fn constants_len(&self) -> usize {
        self.constants.len()
    }

    /// Reserve one inline-cache `[state, handler]` feedback pair.
    pub fn new_feedback(&mut self) -> Feedback {
        let slot = self.feedback_slots;
        self.feedback_slots += 2;
        Feedback(slot)
    }

    /// Allocate the next temporary register.
    pub fn temp(&mut self) -> Reg {
        let r = Reg(self.temp_base as i32 + self.temp_depth as i32);
        self.temp_depth += 1;
        r
    }

    /// Allocate `n` consecutive temporary registers and return the first
    /// (`base + 1` ... follow from its index) — the shape call argument
    /// windows need.
    pub fn reserve_temps(&mut self, n: u32) -> Reg {
        let base = self.temp();
        for _ in 1..n {
            self.temp();
        }
        base
    }

    /// Store the accumulator into a fresh temporary and return its register.
    pub fn stage_acc(&mut self) -> Reg {
        let t = self.temp();
        self.store(t);
        t
    }

    /// Release the most recently allocated temporary.
    pub fn drop_temp(&mut self) {
        debug_assert!(self.temp_depth > 0, "temp underflow");
        self.temp_depth -= 1;
    }

    /// Release all temps allocated above `mark` (a saved [`FnBuilder::temp_depth`]).
    pub fn drop_temps(&mut self, mark: u32) {
        debug_assert!(mark <= self.temp_depth, "temp mark above current depth");
        self.temp_depth = mark;
    }

    pub fn temp_depth(&self) -> u32 {
        self.temp_depth
    }

    /// Whether the fall-through path is still reachable: false after
    /// `Return`/`Throw`/an unconditional jump, until the next `bind`.
    /// Frontends use this to stop emitting unreachable statements (whose
    /// accumulator reads would be meaningless).
    pub fn is_live(&self) -> bool {
        self.acc != Acc::Dead
    }

    // -- labels and handlers --------------------------------------------------

    pub fn new_label(&mut self) -> Label {
        let id = self.labels.len() as u32;
        self.labels.push(LabelRec::default());
        Label {
            id,
            origin: self.origin,
        }
    }

    /// Fix `label` at the next instruction's position. Every path into a
    /// bound label joins here, so accumulator knowledge is discarded.
    pub fn bind(&mut self, label: Label) {
        self.check_label(label);
        let rec = &mut self.labels[label.id as usize];
        debug_assert!(rec.bind.is_none(), "label bound twice");
        rec.bind = Some(self.code.len());
        self.acc = Acc::Unknown;
    }

    /// Anchor the start of a try region at the next instruction.
    pub fn begin_try(&mut self) -> TryBlock {
        let id = self.handlers.len() as u32;
        self.handlers.push(HandlerRec {
            try_start: self.code.len(),
            try_end: None,
            handler_pc: None,
        });
        TryBlock {
            id,
            origin: self.origin,
        }
    }

    /// Anchor the exclusive end of a try region. Exceptions raised at pcs
    /// in `[try_start, try_end)` transfer to the handler.
    pub fn end_try(&mut self, t: TryBlock) {
        self.check_try(t);
        let rec = &mut self.handlers[t.id as usize];
        debug_assert!(rec.try_end.is_none(), "try range ended twice");
        rec.try_end = Some(self.code.len());
    }

    /// Anchor the handler entry: control resumes here with the exception
    /// in the accumulator.
    pub fn handler_entry(&mut self, t: TryBlock) {
        self.check_try(t);
        let rec = &mut self.handlers[t.id as usize];
        debug_assert!(rec.handler_pc.is_none(), "handler anchored twice");
        rec.handler_pc = Some(self.code.len());
        self.acc = Acc::Unknown;
    }

    pub fn load(&mut self, r: Reg) {
        if self.acc == Acc::Reg(r.0) {
            return;
        }
        self.emit_tracked(Opcode::Load, &[r.operand()]);
        self.acc = Acc::Reg(r.0);
    }

    /// Load a Smi; values beyond the (widened) 16-bit operand range ride
    /// the constant pool instead.
    pub fn load_smi(&mut self, v: i32) {
        if v >= i16::MIN as i32 && v <= i16::MAX as i32 {
            if self.acc == Acc::Smi(v) {
                return;
            }
            self.emit_tracked(Opcode::LoadSmi, &[v as u32]);
            self.acc = Acc::Smi(v);
        } else {
            let idx = self.constant(Constant::Smi(v as i64));
            self.load_constant(idx);
        }
    }

    pub fn load_constant(&mut self, idx: ConstIdx) {
        if self.acc == Acc::Const(idx.0) {
            return;
        }
        self.emit_tracked(Opcode::LoadConstant, &[idx.0]);
        self.acc = Acc::Const(idx.0);
    }

    /// Intern `name` and load it (shorthand for `load_constant(self.name(name))`).
    pub fn load_name(&mut self, name: &[u8]) {
        let idx = self.name(name);
        self.load_constant(idx);
    }

    singleton_load!(load_zero, LoadZero, Zero);
    singleton_load!(load_undefined, LoadUndefined, Undefined);
    singleton_load!(load_null, LoadNull, Null);
    singleton_load!(load_true, LoadTrue, True);
    singleton_load!(load_false, LoadFalse, False);
    singleton_load!(load_hole, LoadHole, Hole);

    pub fn load_global(&mut self, name: ConstIdx, fb: Feedback) {
        self.emit_tracked(Opcode::LoadGlobal, &[name.0, fb.0]);
    }

    pub fn load_global_no_throw(&mut self, name: ConstIdx, fb: Feedback) {
        self.emit_tracked(Opcode::LoadGlobalNoThrow, &[name.0, fb.0]);
    }

    pub fn load_named_property(&mut self, obj: Reg, name: ConstIdx, fb: Feedback) {
        self.emit_tracked(Opcode::LoadNamedProperty, &[obj.operand(), name.0, fb.0]);
    }

    /// Keyed load: the key is in the accumulator, the result replaces it.
    pub fn load_keyed_property(&mut self, obj: Reg, fb: Feedback) {
        self.emit_tracked(Opcode::LoadKeyedProperty, &[obj.operand(), fb.0]);
    }

    acc_void!(load_new_target, LoadNewTarget);
    acc_void!(load_current_closure, LoadCurrentClosure);
    acc_void!(load_context, LoadContext);

    /// Store the accumulator into `r`. The store is elided when the accumulator already holds `r`
    pub fn store(&mut self, r: Reg) {
        if self.acc == Acc::Reg(r.0) {
            return;
        }
        self.emit_tracked(Opcode::Store, &[r.operand()]);
        self.acc = Acc::Reg(r.0);
    }

    pub fn store_global(&mut self, name: ConstIdx, fb: Feedback) {
        self.emit_tracked(Opcode::StoreGlobal, &[name.0, fb.0]);
    }

    pub fn store_named_property(&mut self, obj: Reg, name: ConstIdx, fb: Feedback) {
        self.emit_tracked(Opcode::StoreNamedProperty, &[obj.operand(), name.0, fb.0]);
    }

    pub fn store_named_property_no_shadow(&mut self, obj: Reg, name: ConstIdx, fb: Feedback) {
        self.emit_tracked(
            Opcode::StoreNamedPropertyNoShadow,
            &[obj.operand(), name.0, fb.0],
        );
    }

    pub fn add_parent(&mut self, obj: Reg, name: ConstIdx) {
        self.emit_tracked(Opcode::AddParent, &[obj.operand(), name.0]);
    }

    pub fn store_keyed_property(&mut self, obj: Reg, key: Reg, fb: Feedback) {
        self.emit_tracked(
            Opcode::StoreKeyedProperty,
            &[obj.operand(), key.operand(), fb.0],
        );
    }

    pub fn store_keyed_property_no_shadow(&mut self, obj: Reg, key: Reg, fb: Feedback) {
        self.emit_tracked(
            Opcode::StoreKeyedPropertyNoShadow,
            &[obj.operand(), key.operand(), fb.0],
        );
    }

    pub fn store_keyed_slot(&mut self, obj: Reg, key: Reg) {
        self.emit_tracked(Opcode::StoreKeyedSlot, &[obj.operand(), key.operand()]);
    }

    /// `dst <- src` (the `Move` opcode takes the destination first).
    pub fn move_reg(&mut self, dst: Reg, src: Reg) {
        self.emit_tracked(Opcode::Move, &[dst.operand(), src.operand()]);
    }

    pub fn load_context_slot(&mut self, slot: u32, depth: u32) {
        self.emit_tracked(Opcode::LoadContextSlot, &[slot, depth]);
    }

    pub fn store_context_slot(&mut self, slot: u32, depth: u32) {
        self.emit_tracked(Opcode::StoreContextSlot, &[slot, depth]);
    }

    pub fn create_function_context(&mut self, scope: ConstIdx) {
        self.emit_tracked(Opcode::CreateFunctionContext, &[scope.0]);
    }

    pub fn create_block_context(&mut self, slot_count: u32) {
        self.emit_tracked(Opcode::CreateBlockContext, &[slot_count]);
    }

    /// Push the context in the accumulator; `save` receives the old one.
    pub fn push_context(&mut self, save: Reg) {
        self.emit_tracked(Opcode::PushContext, &[save.operand()]);
    }

    pub fn pop_context(&mut self, restore: Reg) {
        self.emit_tracked(Opcode::PopContext, &[restore.operand()]);
    }

    acc_void!(throw_reference_error_if_hole, ThrowReferenceErrorIfHole);

    pub fn call(&mut self, callee: Reg, args: RegList, fb: Feedback) {
        let [base, count] = args.operands();
        self.emit_tracked(Opcode::Call, &[callee.operand(), base, count, fb.0]);
    }

    pub fn call_no_feedback(&mut self, callee: Reg, args: RegList) {
        let [base, count] = args.operands();
        self.emit_tracked(Opcode::CallNoFeedback, &[callee.operand(), base, count]);
    }

    pub fn call_runtime(&mut self, f: crate::RuntimeFn, args: RegList) {
        let [base, count] = args.operands();
        self.emit_tracked(Opcode::CallRuntime, &[f as u32, base, count]);
    }

    pub fn construct(&mut self, callee: Reg, args: RegList) {
        let [base, count] = args.operands();
        self.emit_tracked(Opcode::Construct, &[callee.operand(), base, count]);
    }

    acc_void!(create_empty_object_literal, CreateEmptyObjectLiteral);
    acc_void!(create_empty_array_literal, CreateEmptyArrayLiteral);
    acc_void!(create_bare_object_literal, CreateBareObjectLiteral);

    /// Create a closure from a [`Constant::Callable`] template in the pool.
    /// Each execution creates a fresh closure, so unlike
    /// [`FnBuilder::load_constant`] two calls never alias.
    pub fn create_closure(&mut self, template: ConstIdx) {
        self.emit_tracked(Opcode::CreateClosure, &[template.0]);
    }

    // -- binary arithmetic: acc = acc op reg --------------------------------------

    acc_reg_op!(add, Add);
    acc_reg_op!(sub, Sub);
    acc_reg_op!(mul, Mul);
    acc_reg_op!(div, Div);
    acc_reg_op!(mod_, Mod);
    acc_reg_op!(exp, Exp);
    acc_reg_op!(bitwise_or, BitwiseOr);
    acc_reg_op!(bitwise_xor, BitwiseXor);
    acc_reg_op!(bitwise_and, BitwiseAnd);
    acc_reg_op!(shift_left, ShiftLeft);
    acc_reg_op!(shift_right, ShiftRight);
    acc_reg_op!(shift_right_logical, ShiftRightLogical);

    // -- tests and comparisons: acc = acc op reg -----------------------------------

    acc_reg_op!(test_reference_equal, TestReferenceEqual);
    acc_reg_op!(instance_of, InstanceOf);
    acc_reg_op!(equal_strict, EqualStrict);
    acc_reg_op!(equal, Equal);
    acc_reg_op!(less_than, LessThan);
    acc_reg_op!(less_than_or_equal, LessThanOrEqual);
    acc_reg_op!(greater_than, GreaterThan);
    acc_reg_op!(greater_than_or_equal, GreaterThanOrEqual);

    acc_void!(test_typeof, TestTypeof);
    acc_void!(negate, Negate);

    // -- control flow -----------------------------------------------------------------

    pub fn jump(&mut self, target: Label) {
        self.emit_jump_op(Opcode::Jump, target);
    }

    /// Backward loop edge: like [`FnBuilder::jump`], but the interpreter
    /// polls GC safepoints before transferring. Must target an earlier pc.
    pub fn jump_loop(&mut self, target: Label) {
        self.emit_jump_op(Opcode::JumpLoop, target);
    }

    pub fn jump_if_truthy(&mut self, target: Label) {
        self.emit_jump_op(Opcode::JumpIfTruthy, target);
    }

    pub fn jump_if_falsy(&mut self, target: Label) {
        self.emit_jump_op(Opcode::JumpIfFalsy, target);
    }

    pub fn jump_if_not_undefined(&mut self, target: Label) {
        self.emit_jump_op(Opcode::JumpIfNotUndefined, target);
    }

    // -- exceptions ----------------------------------------------------------------------

    /// Return the accumulator to the caller.
    pub fn ret(&mut self) {
        self.emit_tracked(Opcode::Return, &[]);
        self.acc = Acc::Dead;
    }

    acc_terminal!(throw, Throw);
    acc_terminal!(re_throw, ReThrow);

    // -- escape hatch ----------------------------------------------------------------------

    /// Emit an instruction by raw opcode and operands, with generic
    /// accumulator/register tracking only. Jumps and `Wide` are rejected:
    /// they need label bookkeeping the raw path cannot provide.
    pub fn raw(&mut self, op: Opcode, operands: &[u32]) {
        debug_assert!(
            !matches!(
                op,
                Opcode::Wide
                    | Opcode::Jump
                    | Opcode::JumpLoop
                    | Opcode::JumpIfTruthy
                    | Opcode::JumpIfFalsy
                    | Opcode::JumpIfNotUndefined
            ),
            "{op:?} must go through the label/jump methods"
        );
        self.emit_tracked(op, operands);
    }

    /// Resolve labels, lay out the final code (widening long jumps), and
    /// freeze the function.
    pub fn finish(mut self, meta: FunctionMeta) -> Result<Function, BuildError> {
        if self.temp_depth != 0 {
            return Err(BuildError::UnbalancedTemps);
        }
        for jump in &self.jumps {
            if self.labels[jump.target as usize].bind.is_none() {
                return Err(BuildError::UnboundLabel);
            }
        }
        for handler in &self.handlers {
            if handler.try_end.is_none() || handler.handler_pc.is_none() {
                return Err(BuildError::UnfinishedHandler);
            }
        }

        let (code, label_out, jump_out, handler_out) = self.layout()?;

        for (i, jump) in self.jumps.iter().enumerate() {
            if jump.op == Opcode::JumpLoop {
                let offset = label_out[jump.target as usize] as i64 - jump_out[i] as i64;
                debug_assert!(offset <= 0, "JumpLoop must target an earlier pc");
            }
        }

        let handlers = self
            .handlers
            .iter()
            .enumerate()
            .map(|(i, _)| HandlerEntry {
                try_start: handler_out[i][0],
                try_end: handler_out[i][1],
                handler_pc: handler_out[i][2],
            })
            .collect();

        Ok(Function {
            code,
            constants: self.constants,
            handlers,
            name: meta.name,
            register_count: (self.max_reg + 1).max(0) as u32,
            kind: meta.kind,
            arity: self.arity,
            length: meta.length,
            feedback_count: self.feedback_slots,
            strict: meta.strict,
        })
    }

    /// Lay out the final code stream. Phase 1 iterates jump widths (0 =
    /// narrow, 2 = `Wide` prefix + 16-bit immediate) on position arithmetic
    /// only — widening is monotone, so it converges within `jumps` passes,
    /// each O(events) with no copying. Phase 2 copies the emission buffer
    /// once, re-encoding each jump at its final scale and patching the
    /// final offset. Returns the code plus the resolved final positions of
    /// every label bind, jump site, and handler anchor.
    #[allow(clippy::type_complexity)]
    fn layout(&mut self) -> Result<(Vec<u8>, Vec<usize>, Vec<usize>, Vec<[usize; 3]>), BuildError> {
        let mut events: Vec<(usize, bool, Ev)> = Vec::new();
        for (i, label) in self.labels.iter().enumerate() {
            if let Some(pos) = label.bind {
                events.push((pos, false, Ev::Label(i as u32)));
            }
        }
        for (i, handler) in self.handlers.iter().enumerate() {
            events.push((handler.try_start, false, Ev::TryStart(i as u32)));
            events.push((handler.try_end.unwrap(), false, Ev::TryEnd(i as u32)));
            events.push((handler.handler_pc.unwrap(), false, Ev::Handler(i as u32)));
        }
        for (i, jump) in self.jumps.iter().enumerate() {
            events.push((jump.at, true, Ev::Jump(i as u32)));
        }
        // at equal positions labels/anchors come before the jump starting
        // there: they resolve to the jump instruction's start
        events.sort_by_key(|&(pos, is_jump, _)| (pos, is_jump));

        let mut extra = vec![0usize; self.jumps.len()];
        let mut label_out = vec![usize::MAX; self.labels.len()];
        let mut jump_out = vec![usize::MAX; self.jumps.len()];
        let mut handler_out = vec![[usize::MAX; 3]; self.handlers.len()];

        // Phase 1: width fixpoint. A jump widened in one pass shifts every
        // later event by 2 bytes, which can push further jumps out of
        // range; since widths only grow, at most `jumps` widenings occur.
        let mut passes = self.jumps.len() + 1;
        loop {
            let mut shift = 0usize;
            for &(epos, _, ev) in &events {
                let pos = epos + shift;
                match ev {
                    Ev::Label(i) => label_out[i as usize] = pos,
                    Ev::TryStart(i) => handler_out[i as usize][0] = pos,
                    Ev::TryEnd(i) => handler_out[i as usize][1] = pos,
                    Ev::Handler(i) => handler_out[i as usize][2] = pos,
                    Ev::Jump(i) => {
                        jump_out[i as usize] = pos;
                        shift += extra[i as usize];
                    }
                }
            }
            let mut changed = false;
            for (i, jump) in self.jumps.iter().enumerate() {
                let offset = label_out[jump.target as usize] as i64 - jump_out[i] as i64;
                if extra[i] == 0 && !(i8::MIN as i64..=i8::MAX as i64).contains(&offset) {
                    extra[i] = 2;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
            passes -= 1;
            debug_assert!(passes > 0, "jump relaxation did not converge");
        }

        // Phase 2: single copy with final scales and offsets.
        let widened: usize = extra.iter().filter(|&&e| e != 0).count();
        let mut out = Vec::with_capacity(self.code.len() + widened * 2);
        let mut pos = 0usize;
        for &(epos, _, ev) in &events {
            out.extend_from_slice(&self.code[pos..epos]);
            if let Ev::Jump(i) = ev {
                let jump = &self.jumps[i as usize];
                let offset = label_out[jump.target as usize] as i64 - out.len() as i64;
                if extra[i as usize] == 0 {
                    out.push(jump.op as u8);
                    out.push(offset as i8 as u8);
                } else {
                    let off = i16::try_from(offset).map_err(|_| BuildError::JumpOutOfRange)?;
                    out.push(Opcode::Wide as u8);
                    out.push(jump.op as u8);
                    out.extend_from_slice(&off.to_le_bytes());
                }
                pos = epos + 2; // skip the narrow placeholder
            } else {
                pos = epos;
            }
        }
        out.extend_from_slice(&self.code[pos..]);

        Ok((out, label_out, jump_out, handler_out))
    }

    fn emit_tracked(&mut self, op: Opcode, operands: &[u32]) {
        let kinds = op.operands();
        debug_assert_eq!(
            kinds.len(),
            operands.len(),
            "operand count mismatch for {op:?}"
        );
        if op.reads_acc() {
            debug_assert!(
                self.acc != Acc::Dead,
                "{op:?} reads the accumulator, which is not defined here \
                 (missing load, or emission after an unconditional transfer without a bind)"
            );
        }
        for (i, kind) in kinds.iter().enumerate() {
            match kind {
                Operand::Register => self.track_reg(operands[i] as i32),
                Operand::RegisterListStart => {
                    // the element count follows immediately; the whole
                    // window base..base+count must fit the frame
                    debug_assert_eq!(kinds.get(i + 1), Some(&Operand::RegisterCount));
                    let base = operands[i] as i32;
                    let count = operands[i + 1];
                    self.track_reg(base);
                    if count > 0 {
                        self.track_reg(base + count as i32 - 1);
                    }
                }
                _ => {}
            }
        }
        if let Some(i) = op.written_reg()
            && self.acc == Acc::Reg(operands[i] as i32)
        {
            self.acc = Acc::Unknown;
        }
        if op.writes_acc() {
            self.acc = Acc::Unknown;
        }
        emit(&mut self.code, op, operands);
    }

    /// Bounds-check a register operand and grow the frame window.
    fn track_reg(&mut self, reg: i32) {
        if reg < 0 {
            debug_assert!(
                reg >= -(self.arity as i32) - 1,
                "register {reg} below the parameter window (receiver plus arity {})",
                self.arity
            );
        } else if reg > self.max_reg {
            self.max_reg = reg;
        }
    }

    fn emit_jump_op(&mut self, op: Opcode, target: Label) {
        self.check_label(target);
        if op.reads_acc() {
            debug_assert!(
                self.acc != Acc::Dead,
                "{op:?} reads the accumulator, which is not defined here"
            );
        }
        let at = self.code.len();
        self.code.push(op as u8);
        self.code.push(0); // narrow imm placeholder, patched at finish
        self.jumps.push(JumpRec {
            at,
            op,
            target: target.id,
        });
        if op == Opcode::Jump || op == Opcode::JumpLoop {
            self.acc = Acc::Dead;
        }
    }

    fn check_label(&self, label: Label) {
        debug_assert_eq!(
            label.origin, self.origin,
            "label created by a different FnBuilder"
        );
        debug_assert!(
            (label.id as usize) < self.labels.len(),
            "label id out of range"
        );
    }

    fn check_try(&self, t: TryBlock) {
        debug_assert_eq!(
            t.origin, self.origin,
            "try block created by a different FnBuilder"
        );
        debug_assert!(
            (t.id as usize) < self.handlers.len(),
            "try block id out of range"
        );
    }
}
