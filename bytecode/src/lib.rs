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
    /// (obj) -> enumerator | undefined — for-in head (ES 14.7.5.6):
    /// undefined for a null/undefined subject (zero iterations), else a
    /// hidden enumerator walking the prototype chain lazily
    ForInEnumerate,
    /// (enumerator) -> key-string | undefined — the next enumerable
    /// string key (ES 14.7.5.9 EnumerateObjectProperties), or undefined
    /// when the walk is exhausted; deleted-before-visited and
    /// shadowed keys are skipped inside
    ForInNext,
    /// (fn, key, prefix) -> fn — ES 8.4.4 SetFunctionName: redefine the
    /// closure's `name` ({w−, e−, c+}); the prefix discriminant is
    /// 0 none, 1 "get ", 2 "set " (a Smi)
    SetFunctionName,
    /// (target, key, closure, flags) — accessor member installation: define
    /// one accessor half, merging with an existing pair under the same key
    /// (ES 14.3.10); flags bit 0 marks the getter half, PropertyFlags bits
    /// carry enumerability
    InstallAccessor,
    /// (obj, key, value, flags) — [[DefineOwnProperty]] with exact
    /// attributes (class member installation); define sites are
    /// strict-mode code, so a rejected define throws TypeError. flags are
    /// PropertyFlags bits (the Accessor bit: the value is an AccessorPair)
    DefineOwnProperty,
    /// (obj, proto) -> obj — [[SetPrototypeOf]] (class prototype wiring)
    SetPrototype,
    /// (value) -> value — TypeError unless the value is null or a
    /// constructor (class extends validation, ES 15.7.14)
    ThrowIfNotConstructorOrNull,
    /// (value) -> value — TypeError unless the value is an object or null
    /// (superCtor.prototype validation)
    ThrowIfNotObjectOrNull,
    /// (value) -> value — ReferenceError on the hole: `this` access before
    /// super() in a derived constructor (ES 10.2.2)
    ThrowSuperNotCalledIfHole,
    /// (value) -> value — ReferenceError unless the hole: InitializeThisBinding
    /// guard, super() may run exactly once (ES 10.2.2)
    ThrowSuperAlreadyCalledIfNotHole,
    /// (args...) -> instance — super(...): construct the frame's super
    /// constructor with the frame's new.target (ES 15.4.3); derived parents
    /// get the hole receiver
    ConstructSuper,
    /// () -> instance — super() forwarding the frame's full argument list
    /// (synthesized default derived constructors, ES 15.7.13)
    ConstructSuperAllArgs,
    /// (args..., closure, new_target) -> instance — arrow-delegated
    /// super(): the constructor closure and its new.target ride the tail
    /// of the argument window (threaded through .this_function)
    ConstructSuperVia,
    /// (name) -> value — direct-eval name load: walk the frame context
    /// chain by name; unresolved names fall back to the global object
    LoadDynamicName,
    /// (value, name) -> value — direct-eval name store: write through to
    /// the context-chain slot, else the global object
    StoreDynamicName,
    /// (first) -> array — a fresh array of the frame's arguments from
    /// formal index `first` (a Smi)
    CreateRestParameter,
    /// (home, recv, key) -> value — super.x load: GetSuperBase of the home
    /// object, walked with the split receiver/lookup-start (ES 15.4.2)
    SuperGetProperty,
    /// (home, recv, key, value, semantics) -> value — super.x store (ES
    /// 15.4.4); semantics is SUPER_STORE_WRITE_THROUGH or 0 (shadow), a Smi
    SuperSetProperty,
}

impl RuntimeFn {
    /// All variants in discriminant order. The array length is the
    /// variant count (type-checked), and the VM registers its table in
    /// this order so registry indices equal discriminants.
    pub const ALL: [Self; 35] = [
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
        Self::ForInEnumerate,
        Self::ForInNext,
        Self::SetFunctionName,
        Self::InstallAccessor,
        Self::DefineOwnProperty,
        Self::SetPrototype,
        Self::ThrowIfNotConstructorOrNull,
        Self::ThrowIfNotObjectOrNull,
        Self::ThrowSuperNotCalledIfHole,
        Self::ThrowSuperAlreadyCalledIfNotHole,
        Self::ConstructSuper,
        Self::ConstructSuperAllArgs,
        Self::ConstructSuperVia,
        Self::LoadDynamicName,
        Self::StoreDynamicName,
        Self::CreateRestParameter,
        Self::SuperGetProperty,
        Self::SuperSetProperty,
    ];

    pub const COUNT: u16 = Self::ALL.len() as u16;
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Wide,
    Return,

    // -- accumulator loads -------------------------------------------------
    Load,         // reg -> acc
    LoadSmi,      // imm -> acc
    LoadConstant, // idx -> acc

    // well-known singletons: the hottest loaded values skip the
    // constant-pool round trip (1-byte instructions, no pool slot)
    LoadZero,      // -> acc (Smi 0)
    LoadUndefined, // -> acc
    LoadNull,      // -> acc
    LoadTrue,      // -> acc
    LoadFalse,     // -> acc
    /// acc = TheHole (TDZ staging of non-simple parameter lists)
    LoadHole, // -> acc

    // I don't think we need this right now, LoadConstant should be enough?
    LoadGlobal, // idx (constant pool name) idx (feedback) -> acc
    // typeof on an unresolved global yields "undefined" instead of throwing
    LoadGlobalNoThrow, // idx (constant pool name) idx (feedback) -> acc

    // dynamic name resolution (direct eval) walks the frame context chain
    // by name through RuntimeFn::Load/StoreDynamicName
    LoadNamedProperty, // reg (obj) idx (constant pool index string) idx (feedback) -> acc
    LoadKeyedProperty, // reg (obj) idx (feedback); key in acc -> acc

    // new.target of the current frame (undefined for plain calls)
    LoadNewTarget, // -> acc

    // -- stores ------------------------------------------------------------
    Store,       // acc -> reg
    StoreGlobal, // acc -> idx (constant pool name) idx (feedback)

    StoreNamedProperty, // acc -> reg (obj) idx (constant pool index string) idx (feedback)
    // write-through variant: inherited data properties are written at the
    // holder, never shadowed on the receiver
    StoreNamedPropertyNoShadow, // acc -> reg (obj) idx (constant pool index string) idx (feedback)

    /// acc -> reg (obj) idx (constant pool name): append a named parent to
    /// the object's parent list. Parents are stored as inline
    /// `[name, parent, ...]` pairs, one [[Prototype]] array per object —
    /// one parent or many, always the pair encoding. Creates no slot.
    AddParent, // reg (obj) idx (constant pool name)

    StoreKeyedProperty,         // acc -> reg (obj) reg (key) idx (feedback)
    StoreKeyedPropertyNoShadow, // acc -> reg (obj) reg (key) idx (feedback)

    /// acc -> reg (obj) reg (key): Kette `obj[key] = value`. Element keys
    /// are written in place only — a miss or hole is a RangeError, never
    /// an implicit elements-store growth; name keys use the WriteThrough
    /// store. (JS keeps `StoreKeyedProperty*`.)
    StoreKeyedSlot, // reg (obj) reg (key)

    Move, // reg -> reg

    // -- contexts ----------------------------------------------------------
    LoadContextSlot,  // idx (slot) uimm (depth) -> acc; from the frame context
    StoreContextSlot, // acc -> idx (slot) uimm (depth); frame context

    CreateFunctionContext, // idx (constants: the scope's ScopeInfo) -> acc; outer = frame context
    CreateBlockContext,    // uimm (slot count) -> acc; outer = frame context
    PushContext,           // acc (context) -> frame context; reg <- old context
    PopContext,            // reg (context) -> frame context
    ThrowReferenceErrorIfHole, // acc -> throw ReferenceError if the hole

    // the frame's current context (PushContext/PopContext operand value);
    // lets the compiler snapshot it for absolute restores (try handlers)
    LoadContext, // -> acc

    // -- calls -------------------------------------------------------------

    // for methods the `self` is the first element in the reglist
    Call,           // reg (callee) reglist (base) regcount (count) idx (feedback) -> acc
    CallNoFeedback, // reg (callee) reglist (base) regcount (count) -> acc
    CallRuntime,    // idx (RuntimeFn discriminant) reglist (base) regcount (count) -> acc

    Construct, // reg (callee) reglist (base) regcount (count) -> acc

    // -- literals and closures --------------------------------------------
    CreateEmptyObjectLiteral, // -> acc (object_initial_map, no slots)
    CreateEmptyArrayLiteral,  // -> acc (js_array_map, empty elements)
    /// -> acc (plain_object_map: extensible, no [[Prototype]]):
    /// Self-style (Kette) objects, whose parents are pair-encoded
    CreateBareObjectLiteral,

    CreateClosure, // idx -> acc

    // the currently executing closure (frame callable)
    LoadCurrentClosure, // -> acc

    // -- binary arithmetic -------------------------------------------------

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

    // -- control flow (jumps and branches) --------------------------------
    Jump,     // imm (offset)
    JumpLoop, // imm (negative offset); safepoint-polls before jumping

    JumpIfTruthy, // imm; jump if ToBoolean(acc) == true
    JumpIfFalsy,  // imm; jump if ToBoolean(acc) == false

    /// jump unless acc is `undefined` (pattern/param default guards)
    JumpIfNotUndefined, // imm (offset)

    // -- tests and comparisons --------------------------------------------
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
            b if b == LoadSmi as u8 => LoadSmi,
            b if b == LoadConstant as u8 => LoadConstant,
            b if b == LoadZero as u8 => LoadZero,
            b if b == LoadUndefined as u8 => LoadUndefined,
            b if b == LoadNull as u8 => LoadNull,
            b if b == LoadTrue as u8 => LoadTrue,
            b if b == LoadFalse as u8 => LoadFalse,
            b if b == LoadHole as u8 => LoadHole,
            b if b == LoadGlobal as u8 => LoadGlobal,
            b if b == LoadGlobalNoThrow as u8 => LoadGlobalNoThrow,
            b if b == LoadNamedProperty as u8 => LoadNamedProperty,
            b if b == LoadKeyedProperty as u8 => LoadKeyedProperty,
            b if b == LoadNewTarget as u8 => LoadNewTarget,
            b if b == Store as u8 => Store,
            b if b == StoreGlobal as u8 => StoreGlobal,
            b if b == StoreNamedProperty as u8 => StoreNamedProperty,
            b if b == StoreNamedPropertyNoShadow as u8 => StoreNamedPropertyNoShadow,
            b if b == AddParent as u8 => AddParent,
            b if b == StoreKeyedProperty as u8 => StoreKeyedProperty,
            b if b == StoreKeyedPropertyNoShadow as u8 => StoreKeyedPropertyNoShadow,
            b if b == StoreKeyedSlot as u8 => StoreKeyedSlot,
            b if b == Move as u8 => Move,
            b if b == LoadContextSlot as u8 => LoadContextSlot,
            b if b == StoreContextSlot as u8 => StoreContextSlot,
            b if b == CreateFunctionContext as u8 => CreateFunctionContext,
            b if b == CreateBlockContext as u8 => CreateBlockContext,
            b if b == PushContext as u8 => PushContext,
            b if b == PopContext as u8 => PopContext,
            b if b == ThrowReferenceErrorIfHole as u8 => ThrowReferenceErrorIfHole,
            b if b == LoadContext as u8 => LoadContext,
            b if b == Call as u8 => Call,
            b if b == CallNoFeedback as u8 => CallNoFeedback,
            b if b == CallRuntime as u8 => CallRuntime,
            b if b == Construct as u8 => Construct,
            b if b == CreateEmptyObjectLiteral as u8 => CreateEmptyObjectLiteral,
            b if b == CreateEmptyArrayLiteral as u8 => CreateEmptyArrayLiteral,
            b if b == CreateBareObjectLiteral as u8 => CreateBareObjectLiteral,
            b if b == CreateClosure as u8 => CreateClosure,
            b if b == LoadCurrentClosure as u8 => LoadCurrentClosure,
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
            b if b == JumpIfNotUndefined as u8 => JumpIfNotUndefined,
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
            _ => return None,
        })
    }

    pub const fn operands(self) -> &'static [Operand] {
        use Operand::*;
        match self {
            Self::Wide | Self::Return | Self::Throw | Self::ReThrow => &[],

            Self::Load => &[Register],
            Self::LoadSmi => &[Immediate],
            Self::LoadConstant => &[Index],

            Self::LoadZero
            | Self::LoadUndefined
            | Self::LoadNull
            | Self::LoadTrue
            | Self::LoadFalse
            | Self::LoadHole => &[],

            Self::LoadGlobal | Self::LoadGlobalNoThrow => &[Index, Index],
            Self::LoadNamedProperty => &[Register, Index, Index],
            Self::LoadKeyedProperty => &[Register, Index],
            Self::LoadNewTarget => &[],

            Self::Store => &[Register],
            Self::StoreGlobal => &[Index, Index],
            Self::StoreNamedProperty | Self::StoreNamedPropertyNoShadow => {
                &[Register, Index, Index]
            }
            Self::AddParent => &[Register, Index],
            Self::StoreKeyedProperty | Self::StoreKeyedPropertyNoShadow => {
                &[Register, Register, Index]
            }
            Self::StoreKeyedSlot => &[Register, Register],

            Self::Move => &[Register, Register],

            Self::LoadContextSlot => &[Index, UImmediate],
            Self::StoreContextSlot => &[Index, UImmediate],
            Self::CreateFunctionContext => &[Index],
            Self::CreateBlockContext => &[UImmediate],
            Self::PushContext | Self::PopContext => &[Register],
            Self::ThrowReferenceErrorIfHole => &[],
            Self::LoadContext => &[],

            Self::Call => &[Register, RegisterListStart, RegisterCount, Index],
            Self::CallNoFeedback => &[Register, RegisterListStart, RegisterCount],
            Self::CallRuntime => &[Index, RegisterListStart, RegisterCount],
            Self::Construct => &[Register, RegisterListStart, RegisterCount],

            Self::CreateEmptyObjectLiteral
            | Self::CreateEmptyArrayLiteral
            | Self::CreateBareObjectLiteral => &[],
            Self::CreateClosure => &[Index],
            Self::LoadCurrentClosure => &[],

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

            Self::Jump
            | Self::JumpLoop
            | Self::JumpIfTruthy
            | Self::JumpIfFalsy
            | Self::JumpIfNotUndefined => &[Immediate],

            Self::TestReferenceEqual
            | Self::EqualStrict
            | Self::Equal
            | Self::LessThan
            | Self::LessThanOrEqual
            | Self::GreaterThan
            | Self::GreaterThanOrEqual
            | Self::InstanceOf => &[Register],

            Self::TestTypeof | Self::Negate => &[],
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
