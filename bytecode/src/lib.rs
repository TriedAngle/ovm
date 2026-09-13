#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PropertyFlags {
    /// the data value is not writable
    ReadOnly = 1 << 0,
    /// the property is not enumerable
    DontEnum = 1 << 1,
    /// the property is not configurable
    DontDelete = 1 << 2,
    /// acc holds an `AccessorPair` (get/set) instead of a data value
    Accessor = 1 << 3,
}

impl PropertyFlags {
    /// The operand encoding of this attribute.
    pub const fn bits(self) -> u32 {
        self as u32
    }
}

impl core::ops::BitOr for PropertyFlags {
    type Output = u32;
    fn bitor(self, rhs: Self) -> u32 {
        self.bits() | rhs.bits()
    }
}

impl core::ops::BitOr<PropertyFlags> for u32 {
    type Output = u32;
    fn bitor(self, rhs: PropertyFlags) -> Self::Output {
        self | rhs.bits()
    }
}

/// `Store*PropertyToSuper` semantics-flag operand: ES stores shadow
/// inherited data properties on `this`; the alternative write-through
/// variant writes at the holder (Self-style semantics).
pub const SUPER_STORE_WRITE_THROUGH: u32 = 1;

#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeFn {
    /// (obj) -> iterator — GetProperty(obj, @@iterator) + Call (ES 8.5.4)
    GetIterator,
    /// (iterator) -> result object — Call(GetProperty(iter, "next"), iter)
    IteratorNext,
    /// (result) -> bool — ToBoolean(Get(result, "done"))
    IteratorDone,
    /// (result) -> value — Get(result, "value")
    IteratorValue,
    /// (key, obj) -> bool — HasProperty (the `in` operator, ES 14.11.2)
    HasProperty,
    /// (excluded..., target, source) — CopyDataProperties with an
    /// exclusion list (object rest, ES 8.5.1); `excluded` has
    /// count − 2 entries
    CopyDataProperties,
    /// (description) -> Symbol — a fresh private name
    CreatePrivateName,
    /// (obj, key) -> value — PrivateGet (ES 7.3.30), TypeError if absent
    PrivateGet,
    /// (obj, key, value) — PrivateSet (ES 7.3.31), TypeError if absent
    PrivateSet,
    /// (key, obj) -> bool — `#x in obj` own-private presence
    PrivateIn,
    /// (ctor, fields) — attach the instance-field array to the class
    /// constructor's hidden slot
    SetClassFields,
    /// (ctor, instance) -> instance — run each field initializer with the
    /// instance as receiver, [[DefineOwnProperty]] the results (ES 7.3.33)
    InitInstanceFields,
    /// (value) -> value — RequireObjectCoercible (ES 7.2.2): TypeError on
    /// null/undefined (object destructuring sources)
    RequireObjectCoercible,
    /// (obj, key) -> bool — `delete obj.key` in sloppy code (ES 13.5.1.2
    /// step 4): OrdinaryDelete, false on non-configurable properties
    DeletePropertySloppy,
    /// (obj, key) -> bool — `delete obj.key` in strict code: OrdinaryDelete,
    /// TypeError when the delete fails (ES 13.5.1.2 step 4.h)
    DeletePropertyStrict,
    /// (name) -> bool — sloppy `delete x` on an unresolved (global-object)
    /// name: GlobalEnvironmentRecord.DeleteBinding; declared bindings are
    /// statically known and never reach here
    DeleteIdentifierSloppy,
    /// (key) -> never returns — `delete super.x` (ES 13.5.1.2 step 4.c):
    /// ReferenceError in both language modes; the key is coerced first
    DeleteSuperProperty,
}

impl RuntimeFn {
    /// All variants in discriminant order. The array length is the
    /// variant count (type-checked), and the VM registers its table in
    /// this order so registry indices equal discriminants.
    pub const ALL: [Self; 17] = [
        Self::GetIterator,
        Self::IteratorNext,
        Self::IteratorDone,
        Self::IteratorValue,
        Self::HasProperty,
        Self::CopyDataProperties,
        Self::CreatePrivateName,
        Self::PrivateGet,
        Self::PrivateSet,
        Self::PrivateIn,
        Self::SetClassFields,
        Self::InitInstanceFields,
        Self::RequireObjectCoercible,
        Self::DeletePropertySloppy,
        Self::DeletePropertyStrict,
        Self::DeleteIdentifierSloppy,
        Self::DeleteSuperProperty,
    ];

    pub const COUNT: u16 = Self::ALL.len() as u16;
}

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
    LoadGlobal,  // idx (constant pool name) idx (feedback) -> acc
    StoreGlobal, // acc -> idx (constant pool name) idx (feedback)
    // typeof on an unresolved global yields "undefined" instead of throwing
    LoadGlobalNoThrow, // idx (constant pool name) idx (feedback) -> acc
    LoadContextSlot,   // idx (slot) uimm (depth) -> acc; from the frame context
    StoreContextSlot,  // acc -> idx (slot) uimm (depth); frame context

    // JS [[SetPrototypeOf]]: acc (object) gets reg (prototype) as [[Prototype]]
    SetPrototype, // reg -> (acc stays the object)

    CreateFunctionContext, // idx (constants: the scope's ScopeInfo) -> acc; outer = frame context
    CreateBlockContext,    // uimm (slot count) -> acc; outer = frame context
    CreateCatchContext,    // reg (exception) -> acc; outer = frame context
    PushContext,           // acc (context) -> frame context; reg <- old context
    PopContext,            // reg (context) -> frame context
    ThrowReferenceErrorIfHole, // acc -> throw ReferenceError if the hole
    // dynamic name resolution (direct eval): walk the frame context chain
    // by name; unresolved names fall back to the global object
    LoadDynamicName,  // idx (name constant) -> acc
    StoreDynamicName, // acc -> idx (name constant)

    LoadNamedProperty, // reg (obj) idx (constant pool index string) idx (feedback) -> acc
    StoreNamedProperty, // acc -> reg (obj) idx (constant pool index string) idx (feedback)
    // write-through variant: inherited data properties are written at the
    // holder, never shadowed on the receiver
    StoreNamedPropertyNoShadow, // acc -> reg (obj) idx (constant pool index string) idx (feedback)

    LoadKeyedProperty,          // reg (obj) idx (feedback); key in acc -> acc
    StoreKeyedProperty,         // acc -> reg (obj) reg (key) idx (feedback)
    StoreKeyedPropertyNoShadow, // acc -> reg (obj) reg (key) idx (feedback)

    // [[DefineOwnProperty]] with exact attributes (class member
    // installation): the value in acc is a plain value or, with
    // PropertyFlags::Accessor, an AccessorPair produced by
    // CreateAccessorPair. Define sites are strict-mode code: a rejected
    // define throws a TypeError.
    CreateAccessorPair,     // reg (get) reg (set) -> acc (AccessorPair)
    DefineNamedOwnProperty, // acc -> reg (obj) idx (constant pool name) uimm (PropertyFlags bits) idx (feedback)
    DefineKeyedOwnProperty, // acc -> reg (obj) reg (key) uimm (PropertyFlags bits) idx (feedback)

    // for methods the `self` is the first element in the reglist
    Call,           // reg (callee) reglist (base) regcount (count) idx (feedback) -> acc
    CallNoFeedback, // reg (callee) reglist (base) regcount (count) -> acc
    CallRuntime,    // idx (RuntimeFn discriminant) reglist (base) regcount (count) -> acc

    Construct, // reg (callee) reglist (base) regcount (count) -> acc

    CreateEmptyObjectLiteral, // -> acc (object_initial_map, no slots)
    CreateEmptyArrayLiteral,  // -> acc (js_array_map, empty elements)

    CreateClosure, // idx -> acc

    // -- classes -----------------------------------------------------------

    // ES 15.7.14 ClassDefinitionEvaluation: the extends value must be null
    // or a constructor; the superclass's .prototype must be an object or null
    ThrowIfNotConstructorOrNull, // acc -> throw TypeError otherwise
    ThrowIfNotObjectOrNull,      // acc -> throw TypeError otherwise
    // [[ThisBindingStatus]] guards of derived constructors (ES 10.2.2):
    // `this` starts as the hole and super() initializes it exactly once
    ThrowSuperNotCalledIfHole,        // acc -> ReferenceError if the hole
    ThrowSuperAlreadyCalledIfNotHole, // acc -> ReferenceError if not the hole
    // super property access: home object + split receiver/lookup-start
    LoadNamedPropertyFromSuper, // reg (receiver) idx (name) idx (feedback); home object in acc -> acc
    LoadKeyedPropertyFromSuper, // reg (receiver) reg (key) idx (feedback); home object in acc -> acc
    StoreNamedPropertyToSuper, // acc (value) -> reg (home) reg (receiver) idx (name) uimm (semantics flags) idx (feedback)
    StoreKeyedPropertyToSuper, // acc (value) -> reg (home) reg (receiver) reg (key) uimm (semantics flags) idx (feedback)
    // super(...): construct the current function's [[Prototype]] with the
    // current frame's new.target (ES 15.4.3); result in acc
    ConstructSuper,        // reglist (args) regcount (count) -> acc
    ConstructSuperAllArgs, // forward the current frame's full argument list -> acc
    // arrow-delegated super(): the constructor closure and its new.target
    // come from context slots (threaded through .this_function)
    ConstructSuperVia, // reg (closure) reg (new_target) reglist (args) regcount (count) -> acc
    // new.target of the current frame (undefined for plain calls)
    LdaNewTarget, // -> acc
    // the currently executing closure (frame callable)
    LdaCurrentClosure, // -> acc
    // accessor member installation: define an accessor half, merging with an
    // existing pair under the same key (ES 14.3.10 MethodDefinitionEvaluation)
    InstallNamedAccessor, // reg (target) idx (name) uimm (flags) ; closure in acc
    InstallKeyedAccessor, // reg (target) reg (key) uimm (flags) ; closure in acc
    // ES 8.4.4 SetFunctionName: redefine the closure's `name` ({w−, e−, c+});
    // the closure stays in the accumulator
    SetFunctionNameConst, // idx (constant pool name) ; closure in acc
    SetFunctionNameKey,   // reg (key) uimm (prefix: 0 none, 1 get, 2 set) ; closure in acc

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

    TestTypeof, // -> acc = interned type string
    Negate,     // -> acc = -acc (Smi fast path, -0.0 preserved, else ToNumber)
    InstanceOf, // reg; acc = acc instanceof reg

    // comparisons: acc = acc op reg, yielding the true/false singleton
    EqualStrict,        // reg; ES Strict Equality Comparison (===)
    Equal,              // reg; ES IsLooselyEqual (==)
    LessThan,           // reg; ES Abstract Relational Comparison <
    LessThanOrEqual,    // reg; <=
    GreaterThan,        // reg; >
    GreaterThanOrEqual, // reg; >=

    // exception handling
    Throw,   // acc -> pending exception
    ReThrow, // acc -> pending exception

    /// acc = TheHole (TDZ staging of non-simple parameter lists)
    LdaHole, // -> acc
    /// jump unless acc is `undefined` (pattern/param default guards)
    JumpIfNotUndefined, // imm (offset)
    CreateRestParameter, // uimm (first formal parameter index) -> acc
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
    raw: [u32; 5],
    kinds: &'static [Operand],
}

impl Operands {
    pub const fn new(raw: [u32; 5], kinds: &'static [Operand]) -> Self {
        Self { raw, kinds }
    }

    pub fn kinds(&self) -> &'static [Operand] {
        self.kinds
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
            b if b == LoadGlobalNoThrow as u8 => LoadGlobalNoThrow,
            b if b == LoadContextSlot as u8 => LoadContextSlot,
            b if b == StoreContextSlot as u8 => StoreContextSlot,
            b if b == SetPrototype as u8 => SetPrototype,
            b if b == CreateFunctionContext as u8 => CreateFunctionContext,
            b if b == CreateBlockContext as u8 => CreateBlockContext,
            b if b == CreateCatchContext as u8 => CreateCatchContext,
            b if b == PushContext as u8 => PushContext,
            b if b == PopContext as u8 => PopContext,
            b if b == ThrowReferenceErrorIfHole as u8 => ThrowReferenceErrorIfHole,
            b if b == LoadDynamicName as u8 => LoadDynamicName,
            b if b == StoreDynamicName as u8 => StoreDynamicName,
            b if b == LoadNamedProperty as u8 => LoadNamedProperty,
            b if b == StoreNamedProperty as u8 => StoreNamedProperty,
            b if b == StoreNamedPropertyNoShadow as u8 => StoreNamedPropertyNoShadow,
            b if b == LoadKeyedProperty as u8 => LoadKeyedProperty,
            b if b == StoreKeyedProperty as u8 => StoreKeyedProperty,
            b if b == StoreKeyedPropertyNoShadow as u8 => StoreKeyedPropertyNoShadow,
            b if b == CreateAccessorPair as u8 => CreateAccessorPair,
            b if b == DefineNamedOwnProperty as u8 => DefineNamedOwnProperty,
            b if b == DefineKeyedOwnProperty as u8 => DefineKeyedOwnProperty,
            b if b == Call as u8 => Call,
            b if b == CallNoFeedback as u8 => CallNoFeedback,
            b if b == CallRuntime as u8 => CallRuntime,
            b if b == Construct as u8 => Construct,
            b if b == CreateEmptyObjectLiteral as u8 => CreateEmptyObjectLiteral,
            b if b == CreateEmptyArrayLiteral as u8 => CreateEmptyArrayLiteral,
            b if b == CreateClosure as u8 => CreateClosure,
            b if b == ThrowIfNotConstructorOrNull as u8 => ThrowIfNotConstructorOrNull,
            b if b == ThrowIfNotObjectOrNull as u8 => ThrowIfNotObjectOrNull,
            b if b == ThrowSuperNotCalledIfHole as u8 => ThrowSuperNotCalledIfHole,
            b if b == ThrowSuperAlreadyCalledIfNotHole as u8 => ThrowSuperAlreadyCalledIfNotHole,
            b if b == LoadNamedPropertyFromSuper as u8 => LoadNamedPropertyFromSuper,
            b if b == LoadKeyedPropertyFromSuper as u8 => LoadKeyedPropertyFromSuper,
            b if b == StoreNamedPropertyToSuper as u8 => StoreNamedPropertyToSuper,
            b if b == StoreKeyedPropertyToSuper as u8 => StoreKeyedPropertyToSuper,
            b if b == ConstructSuper as u8 => ConstructSuper,
            b if b == ConstructSuperAllArgs as u8 => ConstructSuperAllArgs,
            b if b == ConstructSuperVia as u8 => ConstructSuperVia,
            b if b == LdaNewTarget as u8 => LdaNewTarget,
            b if b == LdaCurrentClosure as u8 => LdaCurrentClosure,
            b if b == InstallNamedAccessor as u8 => InstallNamedAccessor,
            b if b == InstallKeyedAccessor as u8 => InstallKeyedAccessor,
            b if b == SetFunctionNameConst as u8 => SetFunctionNameConst,
            b if b == SetFunctionNameKey as u8 => SetFunctionNameKey,
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
            b if b == TestTypeof as u8 => TestTypeof,
            b if b == Negate as u8 => Negate,
            b if b == InstanceOf as u8 => InstanceOf,
            b if b == EqualStrict as u8 => EqualStrict,
            b if b == Equal as u8 => Equal,
            b if b == LessThan as u8 => LessThan,
            b if b == LessThanOrEqual as u8 => LessThanOrEqual,
            b if b == GreaterThan as u8 => GreaterThan,
            b if b == GreaterThanOrEqual as u8 => GreaterThanOrEqual,
            b if b == Throw as u8 => Throw,
            b if b == ReThrow as u8 => ReThrow,
            b if b == LdaHole as u8 => LdaHole,
            b if b == JumpIfNotUndefined as u8 => JumpIfNotUndefined,
            b if b == CreateRestParameter as u8 => CreateRestParameter,
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

            Self::LoadGlobal | Self::LoadGlobalNoThrow => &[Index, Index],
            Self::StoreGlobal => &[Index, Index],

            Self::LoadContextSlot => &[Index, UImmediate],
            Self::StoreContextSlot => &[Index, UImmediate],

            Self::SetPrototype => &[Register],

            Self::CreateFunctionContext => &[Index],
            Self::CreateBlockContext => &[UImmediate],
            Self::CreateCatchContext => &[Register],
            Self::PushContext | Self::PopContext => &[Register],
            Self::ThrowReferenceErrorIfHole => &[],
            Self::LoadDynamicName | Self::StoreDynamicName => &[Index],

            Self::LoadNamedProperty => &[Register, Index, Index],
            Self::StoreNamedProperty | Self::StoreNamedPropertyNoShadow => {
                &[Register, Index, Index]
            }

            Self::LoadKeyedProperty => &[Register, Index],
            Self::StoreKeyedProperty | Self::StoreKeyedPropertyNoShadow => {
                &[Register, Register, Index]
            }

            Self::CreateAccessorPair => &[Register, Register],
            Self::DefineNamedOwnProperty => &[Register, Index, UImmediate, Index],
            Self::DefineKeyedOwnProperty => &[Register, Register, UImmediate, Index],

            Self::Call => &[Register, RegisterListStart, RegisterCount, Index],
            Self::CallNoFeedback => &[Register, RegisterListStart, RegisterCount],
            Self::CallRuntime => &[Index, RegisterListStart, RegisterCount],
            Self::Construct => &[Register, RegisterListStart, RegisterCount],

            Self::CreateEmptyObjectLiteral | Self::CreateEmptyArrayLiteral => &[],
            Self::CreateClosure => &[Index],

            Self::ThrowIfNotConstructorOrNull
            | Self::ThrowIfNotObjectOrNull
            | Self::ThrowSuperNotCalledIfHole
            | Self::ThrowSuperAlreadyCalledIfNotHole
            | Self::ConstructSuperAllArgs
            | Self::LdaNewTarget
            | Self::LdaCurrentClosure => &[],
            Self::LoadNamedPropertyFromSuper => &[Register, Index, Index],
            Self::LoadKeyedPropertyFromSuper => &[Register, Register, Index],
            Self::StoreNamedPropertyToSuper => &[Register, Register, Index, UImmediate, Index],
            Self::StoreKeyedPropertyToSuper => &[Register, Register, Register, UImmediate, Index],
            Self::ConstructSuper => &[RegisterListStart, RegisterCount],
            Self::ConstructSuperVia => &[Register, Register, RegisterListStart, RegisterCount],
            Self::InstallNamedAccessor => &[Register, Index, UImmediate],
            Self::InstallKeyedAccessor => &[Register, Register, UImmediate],
            Self::SetFunctionNameConst => &[Index],
            Self::SetFunctionNameKey => &[Register, UImmediate],

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
            Self::JumpIfNotUndefined => &[Immediate],
            Self::TestTypeof | Self::Negate => &[],
            Self::TestReferenceEqual
            | Self::EqualStrict
            | Self::Equal
            | Self::LessThan
            | Self::LessThanOrEqual
            | Self::GreaterThan
            | Self::GreaterThanOrEqual
            | Self::InstanceOf => &[Register],

            Self::LdaHole => &[],
            Self::CreateRestParameter => &[UImmediate],
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

    let mut raw = [0u32; 5];
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
