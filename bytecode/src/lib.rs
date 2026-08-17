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
    // LoadGlobal, //
    // StoreGlobal, //
    LoadContextSlot,  // idx -> acc
    StoreContextSlot, // acc -> idx

    LoadNamedProperty, // reg (obj) idx (constant pool index string) idx (feedback) -> acc
    StoreNamedProperty, // acc -> reg (obj) idx (constant pool index string) idx (feedback)

    // reglist is the first register (index) we dont have literally the whole list there.
    // for methods the `self` is the first element in the reglist
    Call,           // reg (callee) reglist (base) regcount (count) idx (feedback) -> acc
    CallNoFeedback, // reg (callee) reglist (base) regcount (count) -> acc
    CallNative, // idx (native index) reglist (base, first element is the receiver) regcount (count) -> acc

    CreateObjectFromMap, // idx (constant pool map) reglist regcount (slots) -> acc
    CreateArrayLiteral,  // reglist regcount -> acc

    // here to see if it makes a different over CallNative with Add
    Add, // reg1 reg2 -> acc
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
            b if b == LoadContextSlot as u8 => LoadContextSlot,
            b if b == StoreContextSlot as u8 => StoreContextSlot,
            b if b == LoadNamedProperty as u8 => LoadNamedProperty,
            b if b == StoreNamedProperty as u8 => StoreNamedProperty,
            b if b == Call as u8 => Call,
            b if b == CallNoFeedback as u8 => CallNoFeedback,
            b if b == CallNative as u8 => CallNative,
            b if b == CreateObjectFromMap as u8 => CreateObjectFromMap,
            b if b == CreateArrayLiteral as u8 => CreateArrayLiteral,
            b if b == Add as u8 => Add,
            _ => return None,
        })
    }

    pub const fn operands(self) -> &'static [Operand] {
        use Operand::*;
        match self {
            Self::Wide | Self::Return => &[],

            Self::Load => &[Register],
            Self::Store => &[Register],

            Self::LoadSmi => &[Immediate],
            Self::LoadConstant => &[Index],

            Self::Move => &[Register, Register],

            Self::LoadContextSlot => &[Index],
            Self::StoreContextSlot => &[Index],

            Self::LoadNamedProperty => &[Register, Index, Index],
            Self::StoreNamedProperty => &[Register, Index, Index],

            Self::Call => &[Register, RegisterListStart, RegisterCount, Index],
            Self::CallNoFeedback => &[Register, RegisterListStart, RegisterCount],
            Self::CallNative => &[Index, RegisterListStart, RegisterCount],

            Self::CreateObjectFromMap => &[Index, RegisterListStart, RegisterCount],
            Self::CreateArrayLiteral => &[RegisterListStart, RegisterCount],

            Self::Add => &[Register, Register],
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
