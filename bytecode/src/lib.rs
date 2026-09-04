#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Wide,
    Return,

    Load,  // reg -> acc
    Store, // acc -> reg

    LoadSmi,      // imm -> acc
    LoadConstant, // idx -> acc

    Move, // reg -> reg

    // I don't think we need this right now, LoadConstant should be enough?
    LoadGlobal,       // idx (constant pool name) idx (feedback) -> acc
    StoreGlobal,      // acc -> idx (constant pool name) idx (feedback)
    LoadContextSlot,  // idx (slot) uimm (depth) -> acc; from the frame context
    StoreContextSlot, // acc -> idx (slot) uimm (depth); frame context

    CreateFunctionContext, // uimm (slot count) -> acc; outer = frame context
    CreateBlockContext,    // uimm (slot count) -> acc; outer = frame context
    CreateCatchContext,    // reg (exception) -> acc; outer = frame context
    PushContext,           // acc (context) -> frame context; reg <- old context
    PopContext,            // reg (context) -> frame context
    ThrowReferenceErrorIfHole, // acc -> throw ReferenceError if the hole

    LoadNamedProperty, // reg (obj) idx (constant pool index string) idx (feedback) -> acc
    StoreNamedProperty, // acc -> reg (obj) idx (constant pool index string) idx (feedback)
    // JS semantics: a writable inherited data property is shadowed with a new
    // own property on the receiver instead of written through to the holder.
    StoreNamedPropertyShadow, // acc -> reg (obj) idx (constant pool index string) idx (feedback)

    LoadKeyedProperty,        // reg (obj) idx (feedback); key in acc -> acc
    StoreKeyedProperty,       // acc -> reg (obj) reg (key) idx (feedback)
    StoreKeyedPropertyShadow, // acc -> reg (obj) reg (key) idx (feedback)

    // reglist is the first register (index) we dont have literally the whole list there.
    // for methods the `self` is the first element in the reglist
    Call,           // reg (callee) reglist (base) regcount (count) idx (feedback) -> acc
    CallNoFeedback, // reg (callee) reglist (base) regcount (count) -> acc
    CallNative, // idx (native index) reglist (base, first element is the receiver) regcount (count) -> acc

    CreateEmptyObjectLiteral, // -> acc (object_initial_map, no slots)
    CreateEmptyArrayLiteral,  // -> acc (js_array_map, empty elements)

    CreateClosure, // idx -> acc

    // binary arithmetic: acc = acc op reg
    Add, // reg
    Sub,
    Mul,
    Div,
    Mod,
    Exp,
    BitwiseOr,
    BitwiseXor,
    BitwiseAnd,
    ShiftLeft,
    ShiftRight,
    ShiftRightLogical,

    Jump,     // imm (offset)
    JumpLoop, // imm (negative offset); safepoint-polls before jumping

    JumpIfTruthy, // imm; jump if ToBoolean(acc) == true
    JumpIfFalsy,  // imm; jump if ToBoolean(acc) == false

    TestReferenceEqual, // reg; acc = true singleton iff bits(reg) == bits(acc), else false

    // exception handling
    Throw,   // acc -> pending exception
    ReThrow, // acc -> pending exception
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Operand {
    Register,          // scalable, signed (negative indices are parameters)
    RegisterListStart, // first register of a start+count range; scalable, signed
    RegisterCount,     // UImmediate but not scalable
    Immediate,         // scalable, signed
    UImmediate,        // scalable
    Index,             // scalable
}

impl Operand {
    pub const fn size_in_stream(self, scale: Scale) -> usize {
        use Operand::*;
        match self {
            RegisterCount => 1, // ALWAYS 1 byte
            Register | RegisterListStart | Immediate | UImmediate | Index => scale as usize,
        }
    }

    pub const fn is_signed(self) -> bool {
        use Operand::*;
        matches!(self, Register | RegisterListStart | Immediate)
    }
}

#[derive(Clone, Copy)]
pub struct Operands {
    raw: [u32; 4],
    kinds: &'static [Operand],
}

impl Operands {
    pub const fn new(raw: [u32; 4], kinds: &'static [Operand]) -> Self {
        Self { raw, kinds }
    }

    #[inline]
    fn at(&self, i: usize, kind: Operand) -> u32 {
        debug_assert_eq!(self.kinds[i], kind);
        self.raw[i]
    }

    #[inline]
    pub fn reg(&self, i: usize) -> i32 {
        self.at(i, Operand::Register) as i32
    }

    #[inline]
    pub fn reg_list(&self, i: usize) -> i32 {
        self.at(i, Operand::RegisterListStart) as i32
    }

    #[inline]
    pub fn reg_count(&self, i: usize) -> usize {
        self.at(i, Operand::RegisterCount) as usize
    }

    #[inline]
    pub fn imm(&self, i: usize) -> i32 {
        self.at(i, Operand::Immediate) as i32
    }

    #[inline]
    pub fn uimm(&self, i: usize) -> u32 {
        self.at(i, Operand::UImmediate)
    }

    #[inline]
    pub fn idx(&self, i: usize) -> usize {
        self.at(i, Operand::Index) as usize
    }
}

impl Opcode {
    // TODO: we should probably use transmute unsafe here
    pub const fn from_byte(byte: u8) -> Option<Self> {
        use Opcode::*;
        Some(match byte {
            b if b == Wide as u8 => Wide,
            b if b == Return as u8 => Return,
            b if b == Load as u8 => Load,
            b if b == Store as u8 => Store,
            b if b == LoadSmi as u8 => LoadSmi,
            b if b == LoadConstant as u8 => LoadConstant,
            b if b == Move as u8 => Move,
            b if b == LoadGlobal as u8 => LoadGlobal,
            b if b == StoreGlobal as u8 => StoreGlobal,
            b if b == LoadContextSlot as u8 => LoadContextSlot,
            b if b == StoreContextSlot as u8 => StoreContextSlot,
            b if b == CreateFunctionContext as u8 => CreateFunctionContext,
            b if b == CreateBlockContext as u8 => CreateBlockContext,
            b if b == CreateCatchContext as u8 => CreateCatchContext,
            b if b == PushContext as u8 => PushContext,
            b if b == PopContext as u8 => PopContext,
            b if b == ThrowReferenceErrorIfHole as u8 => ThrowReferenceErrorIfHole,
            b if b == LoadNamedProperty as u8 => LoadNamedProperty,
            b if b == StoreNamedProperty as u8 => StoreNamedProperty,
            b if b == StoreNamedPropertyShadow as u8 => StoreNamedPropertyShadow,
            b if b == LoadKeyedProperty as u8 => LoadKeyedProperty,
            b if b == StoreKeyedProperty as u8 => StoreKeyedProperty,
            b if b == StoreKeyedPropertyShadow as u8 => StoreKeyedPropertyShadow,
            b if b == Call as u8 => Call,
            b if b == CallNoFeedback as u8 => CallNoFeedback,
            b if b == CallNative as u8 => CallNative,
            b if b == CreateEmptyObjectLiteral as u8 => CreateEmptyObjectLiteral,
            b if b == CreateEmptyArrayLiteral as u8 => CreateEmptyArrayLiteral,
            b if b == CreateClosure as u8 => CreateClosure,
            b if b == Add as u8 => Add,
            b if b == Sub as u8 => Sub,
            b if b == Mul as u8 => Mul,
            b if b == Div as u8 => Div,
            b if b == Mod as u8 => Mod,
            b if b == Exp as u8 => Exp,
            b if b == BitwiseOr as u8 => BitwiseOr,
            b if b == BitwiseXor as u8 => BitwiseXor,
            b if b == BitwiseAnd as u8 => BitwiseAnd,
            b if b == ShiftLeft as u8 => ShiftLeft,
            b if b == ShiftRight as u8 => ShiftRight,
            b if b == ShiftRightLogical as u8 => ShiftRightLogical,
            b if b == Jump as u8 => Jump,
            b if b == JumpLoop as u8 => JumpLoop,
            b if b == JumpIfTruthy as u8 => JumpIfTruthy,
            b if b == JumpIfFalsy as u8 => JumpIfFalsy,
            b if b == TestReferenceEqual as u8 => TestReferenceEqual,
            b if b == Throw as u8 => Throw,
            b if b == ReThrow as u8 => ReThrow,
            _ => return None,
        })
    }

    pub const fn operands(self) -> &'static [Operand] {
        use Operand::*;
        match self {
            Self::Wide | Self::Return | Self::Throw | Self::ReThrow => &[],

            Self::Load => &[Register],
            Self::Store => &[Register],

            Self::LoadSmi => &[Immediate],
            Self::LoadConstant => &[Index],

            Self::Move => &[Register, Register],

            Self::LoadGlobal => &[Index, Index],
            Self::StoreGlobal => &[Index, Index],

            Self::LoadContextSlot => &[Index, UImmediate],
            Self::StoreContextSlot => &[Index, UImmediate],

            Self::CreateFunctionContext | Self::CreateBlockContext => &[UImmediate],
            Self::CreateCatchContext => &[Register],
            Self::PushContext | Self::PopContext => &[Register],
            Self::ThrowReferenceErrorIfHole => &[],

            Self::LoadNamedProperty => &[Register, Index, Index],
            Self::StoreNamedProperty => &[Register, Index, Index],
            Self::StoreNamedPropertyShadow => &[Register, Index, Index],

            Self::LoadKeyedProperty => &[Register, Index],
            Self::StoreKeyedProperty => &[Register, Register, Index],
            Self::StoreKeyedPropertyShadow => &[Register, Register, Index],

            Self::Call => &[Register, RegisterListStart, RegisterCount, Index],
            Self::CallNoFeedback => &[Register, RegisterListStart, RegisterCount],
            Self::CallNative => &[Index, RegisterListStart, RegisterCount],

            Self::CreateEmptyObjectLiteral | Self::CreateEmptyArrayLiteral => &[],
            Self::CreateClosure => &[Index],

            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Mod
            | Self::Exp
            | Self::BitwiseOr
            | Self::BitwiseXor
            | Self::BitwiseAnd
            | Self::ShiftLeft
            | Self::ShiftRight
            | Self::ShiftRightLogical => &[Register],

            Self::Jump | Self::JumpLoop | Self::JumpIfTruthy | Self::JumpIfFalsy => &[Immediate],
            Self::TestReferenceEqual => &[Register],
        }
    }

    pub const fn size(self, scale: Scale) -> usize {
        let ops = self.operands();
        let mut size = 1usize;

        let mut i = 0;
        while i < ops.len() {
            size += ops[i].size_in_stream(scale);
            i += 1;
        }
        size
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    Byte1 = 1, // default
    Byte2 = 2, // wide
}

pub fn emit(code: &mut Vec<u8>, op: Opcode, operands: &[u32]) {
    let kinds = op.operands();
    assert_eq!(kinds.len(), operands.len(), "operand count mismatch");

    let wide = kinds.iter().zip(operands).any(|(kind, value)| {
        if *kind == Operand::RegisterCount {
            return false;
        }
        if kind.is_signed() {
            let v = *value as i32;
            v < i8::MIN as i32 || v > i8::MAX as i32
        } else {
            *value > u8::MAX as u32
        }
    });
    let scale = if wide { Scale::Byte2 } else { Scale::Byte1 };
    if wide {
        code.push(Opcode::Wide as u8);
    }
    code.push(op as u8);

    for (kind, value) in kinds.iter().zip(operands) {
        let size = kind.size_in_stream(scale);
        let fits = if kind.is_signed() {
            let v = *value as i32;
            match size {
                1 => v >= i8::MIN as i32 && v <= i8::MAX as i32,
                2 => v >= i16::MIN as i32 && v <= i16::MAX as i32,
                _ => unreachable!("signed operands are at most 2 bytes"),
            }
        } else {
            (*value as u64) < (1u64 << (size * 8))
        };
        assert!(fits, "operand {value} does not fit in {size} byte(s)");
        code.extend_from_slice(&value.to_le_bytes()[..size]);
    }
}

// TODO: get rid of this much branching and .expect(), use `debug_assert!` instead
pub fn decode(code: &[u8], mut pc: usize) -> (Opcode, Operands, usize) {
    let mut op = read_opcode(code, &mut pc);
    let mut scale = Scale::Byte1;
    if op == Opcode::Wide {
        scale = Scale::Byte2;
        op = read_opcode(code, &mut pc);
    }

    let mut raw = [0u32; 4];
    for (i, kind) in op.operands().iter().enumerate() {
        let size = kind.size_in_stream(scale);
        let bytes = code
            .get(pc..pc + size)
            .expect("truncated instruction stream");
        let mut buf = [0u8; 4];
        buf[..size].copy_from_slice(bytes);
        let value = u32::from_le_bytes(buf);
        raw[i] = if kind.is_signed() {
            // sign-extend
            let shift = 32 - size * 8;
            ((value << shift) as i32 >> shift) as u32
        } else {
            value
        };
        pc += size;
    }
    (op, Operands::new(raw, op.operands()), pc)
}

// TODO: see if force inlining matters
fn read_opcode(code: &[u8], pc: &mut usize) -> Opcode {
    let byte = *code.get(*pc).expect("program counter out of bounds");
    *pc += 1;
    Opcode::from_byte(byte).expect("invalid opcode")
}

pub fn jump_target(pc: usize, offset: i32) -> usize {
    pc.wrapping_add_signed(offset as isize)
}
