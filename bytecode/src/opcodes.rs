use crate::{Operand, Scale};

/// Which table an [`Operand::Index`] operand addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    /// The function's constant pool.
    ConstantPool,
    /// The function's feedback vector (`[state, handler]` pairs).
    Feedback,
    /// The VM's runtime-function registry (`RuntimeFn` discriminants).
    RuntimeFn,
    /// Not statically checkable (e.g. context slots, bounded by the
    /// runtime scope chain).
    Unchecked,
}

macro_rules! define_opcodes {
    // -- per-entry expansion helpers ---------------------------------------
    (@reads) => { false };
    (@reads none) => { false };
    (@reads reads) => { true };
    (@reads writes) => { false };
    (@reads reads_writes) => { true };
    (@writes) => { false };
    (@writes none) => { false };
    (@writes reads) => { false };
    (@writes writes) => { true };
    (@writes reads_writes) => { true };
    (@wreg) => { None };
    (@wreg $n:literal) => { Some($n) };
    (@ikinds) => { &[] };
    (@ikinds $($i:ident),+ $(,)?) => { &[$(IndexKind::$i),+] };

    // -- the table -----------------------------------------------------------
    ($(
        $(#[$meta:meta])*
        $name:ident {
            operands: [$($operand:ident),* $(,)?]
            $(, acc: $acc:ident)?
            $(, writes_reg: $wr:literal)?
            $(, indices: [$($index:ident),* $(,)?])?
            $(,)?
        }
    ),* $(,)?) => {
        #[repr(u8)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Opcode {
            $($(#[$meta])* $name),*
        }

        impl Opcode {
            pub const fn from_byte(byte: u8) -> Option<Self> {
                Some(match byte {
                    $(b if b == Self::$name as u8 => Self::$name,)*
                    _ => return None,
                })
            }

            pub const fn operands(self) -> &'static [Operand] {
                match self {
                    $(Self::$name => &[$(Operand::$operand),*],)*
                }
            }

            /// The instruction reads the implicit accumulator.
            pub const fn reads_acc(self) -> bool {
                match self {
                    $(Self::$name => define_opcodes!(@reads $($acc)?),)*
                }
            }

            /// The instruction overwrites the implicit accumulator.
            pub const fn writes_acc(self) -> bool {
                match self {
                    $(Self::$name => define_opcodes!(@writes $($acc)?),)*
                }
            }

            /// The operand index of the register this opcode writes, if any.
            pub const fn written_reg(self) -> Option<usize> {
                match self {
                    $(Self::$name => define_opcodes!(@wreg $($wr)?),)*
                }
            }

            /// Which table each operand addresses, parallel to
            /// [`Opcode::operands`] (non-`Index` operands are
            /// [`IndexKind::Unchecked`]).
            pub(crate) const fn index_kinds(self) -> &'static [IndexKind] {
                match self {
                    $(Self::$name => define_opcodes!(@ikinds $($($index),*)?)),*
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

        // `indices` must stay parallel to `operands`.
        const _: () = {
            $(
                assert!(
                    Opcode::$name.operands().len() == Opcode::$name.index_kinds().len(),
                    "index kinds must be parallel to operands"
                );
            )*
        };
    };
}

define_opcodes! {
    Wide { operands: [], indices: [] },
    Return { operands: [], acc: reads, indices: [] },

    // -- accumulator loads -------------------------------------------------
    Load { operands: [Register], acc: writes, indices: [Unchecked] },        // reg -> acc
    LoadSmi { operands: [Immediate], acc: writes, indices: [Unchecked] },    // imm -> acc
    LoadConstant { operands: [Index], acc: writes, indices: [ConstantPool] }, // idx -> acc

    // well-known singletons: the hottest loaded values skip the
    // constant-pool round trip (1-byte instructions, no pool slot)
    LoadZero { operands: [], acc: writes, indices: [] },      // -> acc (Smi 0)
    LoadUndefined { operands: [], acc: writes, indices: [] }, // -> acc
    LoadNull { operands: [], acc: writes, indices: [] },      // -> acc
    LoadTrue { operands: [], acc: writes, indices: [] },      // -> acc
    LoadFalse { operands: [], acc: writes, indices: [] },     // -> acc
    /// acc = TheHole (TDZ staging of non-simple parameter lists)
    LoadHole { operands: [], acc: writes, indices: [] },      // -> acc

    LoadGlobal { operands: [Index, Index], acc: writes, indices: [ConstantPool, Feedback] }, // idx (constant pool name) idx (feedback) -> acc
    /// typeof on an unresolved global yields "undefined" instead of throwing
    LoadGlobalNoThrow { operands: [Index, Index], acc: writes, indices: [ConstantPool, Feedback] },

    // dynamic name resolution (direct eval) walks the frame context chain
    // by name through RuntimeFn::Load/StoreDynamicName
    LoadNamedProperty { operands: [Register, Index, Index], acc: writes, indices: [Unchecked, ConstantPool, Feedback] }, // reg (obj) idx (constant pool index string) idx (feedback) -> acc
    LoadKeyedProperty { operands: [Register, Index], acc: reads_writes, indices: [Unchecked, Feedback] }, // reg (obj) idx (feedback); key in acc -> acc

    /// new.target of the current frame (undefined for plain calls)
    LoadNewTarget { operands: [], acc: writes, indices: [] }, // -> acc

    // -- stores ------------------------------------------------------------
    Store { operands: [Register], acc: reads, writes_reg: 0, indices: [Unchecked] }, // acc -> reg
    StoreGlobal { operands: [Index, Index], acc: reads, indices: [ConstantPool, Feedback] }, // acc -> idx (constant pool name) idx (feedback)

    StoreNamedProperty { operands: [Register, Index, Index], acc: reads, indices: [Unchecked, ConstantPool, Feedback] }, // acc -> reg (obj) idx (constant pool index string) idx (feedback)
    /// write-through variant: inherited data properties are written at the
    /// holder, never shadowed on the receiver
    StoreNamedPropertyNoShadow { operands: [Register, Index, Index], acc: reads, indices: [Unchecked, ConstantPool, Feedback] },

    /// acc -> reg (obj) idx (constant pool name): append a named parent to
    /// the object's parent list. Parents are stored as inline
    /// `[name, parent, ...]` pairs, one [[Prototype]] array per object —
    /// one parent or many, always the pair encoding. Creates no slot.
    AddParent { operands: [Register, Index], acc: reads, indices: [Unchecked, ConstantPool] },

    StoreKeyedProperty { operands: [Register, Register, Index], acc: reads, indices: [Unchecked, Unchecked, Feedback] }, // acc -> reg (obj) reg (key) idx (feedback)
    StoreKeyedPropertyNoShadow { operands: [Register, Register, Index], acc: reads, indices: [Unchecked, Unchecked, Feedback] },

    /// acc -> reg (obj) reg (key): Kette `obj[key] = value`. Element keys
    /// are written in place only — a miss or hole is a RangeError, never
    /// an implicit elements-store growth; name keys use the WriteThrough
    /// store. (JS keeps `StoreKeyedProperty*`.)
    StoreKeyedSlot { operands: [Register, Register], acc: reads, indices: [Unchecked, Unchecked] },

    Move { operands: [Register, Register], writes_reg: 0, indices: [Unchecked, Unchecked] }, // reg (dst) <- reg (src)

    // -- contexts ----------------------------------------------------------
    LoadContextSlot { operands: [Index, UImmediate], acc: writes, indices: [Unchecked, Unchecked] },  // idx (slot) uimm (depth) -> acc; from the frame context
    StoreContextSlot { operands: [Index, UImmediate], acc: reads, indices: [Unchecked, Unchecked] },  // acc -> idx (slot) uimm (depth); frame context

    CreateFunctionContext { operands: [Index], acc: writes, indices: [ConstantPool] }, // idx (constants: the scope's ScopeInfo) -> acc; outer = frame context
    CreateBlockContext { operands: [UImmediate], acc: writes, indices: [Unchecked] },  // uimm (slot count) -> acc; outer = frame context
    PushContext { operands: [Register], acc: reads, writes_reg: 0, indices: [Unchecked] }, // acc (context) -> frame context; reg <- old context
    PopContext { operands: [Register], indices: [Unchecked] },                       // reg (context) -> frame context
    ThrowReferenceErrorIfHole { operands: [], acc: reads, indices: [] },             // acc -> throw ReferenceError if the hole

    // the frame's current context (PushContext/PopContext operand value);
    // lets the compiler snapshot it for absolute restores (try handlers)
    LoadContext { operands: [], acc: writes, indices: [] }, // -> acc

    // -- calls -------------------------------------------------------------

    // for methods the `self` is the first element in the reglist
    Call { operands: [Register, RegisterListStart, RegisterCount, Index], acc: writes, indices: [Unchecked, Unchecked, Unchecked, Feedback] }, // reg (callee) reglist (base) regcount (count) idx (feedback) -> acc
    CallNoFeedback { operands: [Register, RegisterListStart, RegisterCount], acc: writes, indices: [Unchecked, Unchecked, Unchecked] },
    CallRuntime { operands: [Index, RegisterListStart, RegisterCount], acc: writes, indices: [RuntimeFn, Unchecked, Unchecked] }, // idx (RuntimeFn discriminant) reglist (base) regcount (count) -> acc

    Construct { operands: [Register, RegisterListStart, RegisterCount], acc: writes, indices: [Unchecked, Unchecked, Unchecked] }, // reg (callee) reglist (base) regcount (count) -> acc

    // -- literals and closures --------------------------------------------
    CreateEmptyObjectLiteral { operands: [], acc: writes, indices: [] }, // -> acc (object_initial_map, no slots)
    CreateEmptyArrayLiteral { operands: [], acc: writes, indices: [] },  // -> acc (js_array_map, empty elements)
    /// -> acc (plain_object_map: extensible, no [[Prototype]]):
    /// Self-style (Kette) objects, whose parents are pair-encoded
    CreateBareObjectLiteral { operands: [], acc: writes, indices: [] },

    CreateClosure { operands: [Index], acc: writes, indices: [ConstantPool] }, // idx -> acc

    /// the currently executing closure (frame callable)
    LoadCurrentClosure { operands: [], acc: writes, indices: [] }, // -> acc

    // -- binary arithmetic: acc = acc op reg --------------------------------
    Add { operands: [Register], acc: reads_writes, indices: [Unchecked] },
    Sub { operands: [Register], acc: reads_writes, indices: [Unchecked] },
    Mul { operands: [Register], acc: reads_writes, indices: [Unchecked] },
    Div { operands: [Register], acc: reads_writes, indices: [Unchecked] },
    Mod { operands: [Register], acc: reads_writes, indices: [Unchecked] },
    Exp { operands: [Register], acc: reads_writes, indices: [Unchecked] },
    BitwiseOr { operands: [Register], acc: reads_writes, indices: [Unchecked] },
    BitwiseXor { operands: [Register], acc: reads_writes, indices: [Unchecked] },
    BitwiseAnd { operands: [Register], acc: reads_writes, indices: [Unchecked] },
    ShiftLeft { operands: [Register], acc: reads_writes, indices: [Unchecked] },
    ShiftRight { operands: [Register], acc: reads_writes, indices: [Unchecked] },
    ShiftRightLogical { operands: [Register], acc: reads_writes, indices: [Unchecked] },

    // -- control flow (jumps and branches) ----------------------------------
    Jump { operands: [Immediate], indices: [Unchecked] },     // imm (offset)
    JumpLoop { operands: [Immediate], indices: [Unchecked] }, // imm (negative offset); safepoint-polls before jumping

    JumpIfTruthy { operands: [Immediate], acc: reads, indices: [Unchecked] }, // imm; jump if ToBoolean(acc) == true
    JumpIfFalsy { operands: [Immediate], acc: reads, indices: [Unchecked] },  // imm; jump if ToBoolean(acc) == false

    /// jump unless acc is `undefined` (pattern/param default guards)
    JumpIfNotUndefined { operands: [Immediate], acc: reads, indices: [Unchecked] },

    // -- tests and comparisons ---------------------------------------------
    TestReferenceEqual { operands: [Register], acc: reads_writes, indices: [Unchecked] }, // reg; acc = true singleton iff bits(reg) == bits(acc), else false

    TestTypeof { operands: [], acc: reads_writes, indices: [] }, // -> acc = interned type string
    Negate { operands: [], acc: reads_writes, indices: [] },     // -> acc = -acc (Smi fast path, -0.0 preserved, else ToNumber)
    InstanceOf { operands: [Register], acc: reads_writes, indices: [Unchecked] }, // reg; acc = acc instanceof reg

    // comparisons: acc = acc op reg, yielding the true/false singleton
    EqualStrict { operands: [Register], acc: reads_writes, indices: [Unchecked] },        // reg; ES Strict Equality Comparison (===)
    Equal { operands: [Register], acc: reads_writes, indices: [Unchecked] },              // reg; ES IsLooselyEqual (==)
    LessThan { operands: [Register], acc: reads_writes, indices: [Unchecked] },           // reg; ES Abstract Relational Comparison <
    LessThanOrEqual { operands: [Register], acc: reads_writes, indices: [Unchecked] },    // reg; <=
    GreaterThan { operands: [Register], acc: reads_writes, indices: [Unchecked] },        // reg; >
    GreaterThanOrEqual { operands: [Register], acc: reads_writes, indices: [Unchecked] }, // reg; >=

    // exception handling
    Throw { operands: [], acc: reads, indices: [] },   // acc -> pending exception
    ReThrow { operands: [], acc: reads, indices: [] }, // acc -> pending exception
}
