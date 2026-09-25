mod builder;
mod opcodes;
mod program;
mod validate;

pub use builder::{
    BuildError, ConstIdx, Feedback, FnBuilder, FunctionMeta, Label, Reg, RegList, RtArg, TryBlock,
};
pub use opcodes::Opcode;
pub use program::{
    CallableKind, CompileFn, Constant, FrontendError, FrontendErrorKind, Function, FunctionId,
    HandlerEntry, Program, SourceMode,
};
pub use validate::{ValidationError, validate, validate_function};

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
    /// () -> undefined — `KetteTools.forceMinorGC()`: force a nursery
    /// collection on the calling thread's heap
    ForceMinorGc,
    /// () -> undefined — `KetteTools.forceMajorGC()`: force a full
    /// collection cycle (nursery evacuation included) on the calling
    /// thread's heap
    ForceMajorGc,
    /// () -> never returns a value — `KetteTools.shutdown()`: pause every
    /// other thread at a safepoint (running the shutdown protocol at the
    /// world-stopped point), cancel their executions, and terminate the
    /// calling thread's execution through the uncatchable channel
    ShutdownVm,
}

impl RuntimeFn {
    /// All variants in discriminant order. The array length is the
    /// variant count (type-checked), and the VM registers its table in
    /// this order so registry indices equal discriminants.
    pub const ALL: [Self; 38] = [
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
        Self::ForceMinorGc,
        Self::ForceMajorGc,
        Self::ShutdownVm,
    ];

    pub const COUNT: u16 = Self::ALL.len() as u16;
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

/// A lazy cursor over one instruction's operands: values are read
/// straight from the code stream on access — positional, idempotent,
/// never materialized. Per-operand offset, size and signedness come
/// from packed per-opcode tables (see the `OPERAND_*` statics).
///
/// The `code` view is anchored to a heap borrow and therefore valid
/// only until the next safepoint (any allocation): fast paths read
/// their operands immediately after [`decode`]; across a safepoint,
/// re-derive the view from the rooted code register (the position
/// words — `pc`, `base` — are plain integers and always safe).
#[derive(Clone, Copy)]
pub struct Operands<'c> {
    code: &'c [u8],
    /// Byte offset of the first operand.
    base: u32,
    row: &'static [u8; 5],
}

impl<'c> Operands<'c> {
    /// Read operand `i` from the stream.
    #[inline]
    fn read(&self, i: usize) -> u32 {
        let e = self.row[i] as usize;
        debug_assert!(e != 0, "no operand {i}");
        let off = self.base as usize + (e & 0x0f) - 1;
        debug_assert!(
            off + if e & 0x10 != 0 { 2 } else { 1 } <= self.code.len(),
            "operand read past the end of the code stream"
        );
        // Safety: the stream was validated at materialization, and the
        // row encodes this operand's exact offset and size.
        unsafe {
            if e & 0x10 != 0 {
                let w = u16::from_le_bytes([
                    *self.code.get_unchecked(off),
                    *self.code.get_unchecked(off + 1),
                ]) as u32;
                if e & 0x20 != 0 {
                    (w as i16) as i32 as u32
                } else {
                    w
                }
            } else {
                let b = *self.code.get_unchecked(off) as u32;
                if e & 0x20 != 0 {
                    (b as i8) as i32 as u32
                } else {
                    b
                }
            }
        }
    }

    #[inline]
    pub fn reg(&self, i: usize) -> i32 {
        self.read(i) as i32
    }

    #[inline]
    pub fn reg_list(&self, i: usize) -> i32 {
        self.read(i) as i32
    }

    #[inline]
    pub fn reg_count(&self, i: usize) -> usize {
        self.read(i) as usize
    }

    #[inline]
    pub fn imm(&self, i: usize) -> i32 {
        self.read(i) as i32
    }

    #[inline]
    pub fn uimm(&self, i: usize) -> u32 {
        self.read(i)
    }

    #[inline]
    pub fn idx(&self, i: usize) -> usize {
        self.read(i) as usize
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

/// Decode the instruction at `pc` into a lazy operand cursor. The stream
/// is trusted — programs are validated at materialization (`validate`) —
/// so release builds read unchecked; debug builds re-verify via
/// [`try_decode`]. The returned view is anchored to `code`'s borrow and
/// valid only until the next safepoint (see [`Operands`]).
pub fn decode<'c>(code: &'c [u8], pc: usize) -> (Opcode, Operands<'c>, usize) {
    debug_assert!(
        try_decode(code, pc).is_some(),
        "invalid or truncated instruction at pc {pc}"
    );
    // Safety: validated stream (see above).
    let byte = unsafe { *code.get_unchecked(pc) };
    let (op, base, wide) = if byte == Opcode::Wide as u8 {
        (
            // Safety: validated stream.
            unsafe { Opcode::from_byte_unchecked(*code.get_unchecked(pc + 1)) },
            pc + 2,
            true,
        )
    } else {
        // Safety: validated stream.
        (unsafe { Opcode::from_byte_unchecked(byte) }, pc + 1, false)
    };
    let idx = op as usize;
    let row = (if wide {
        opcodes::OPERAND_ROWS_WIDE
    } else {
        opcodes::OPERAND_ROWS_NARROW
    })[idx];
    let size = (if wide {
        opcodes::OPERAND_SIZES_WIDE
    } else {
        opcodes::OPERAND_SIZES_NARROW
    })[idx] as usize;
    (
        op,
        Operands {
            code,
            base: base as u32,
            row,
        },
        base + size,
    )
}

/// Fallible [`decode`]: `None` on an unknown opcode byte or a stream that
/// ends inside an instruction. The validator's checked twin.
pub fn try_decode<'c>(code: &'c [u8], mut pc: usize) -> Option<(Opcode, Operands<'c>, usize)> {
    let mut byte = *code.get(pc)?;
    pc += 1;
    let wide = byte == Opcode::Wide as u8;
    if wide {
        byte = *code.get(pc)?;
        pc += 1;
    }
    let op = Opcode::from_byte(byte)?;
    let idx = op as usize;
    let row = (if wide {
        opcodes::OPERAND_ROWS_WIDE
    } else {
        opcodes::OPERAND_ROWS_NARROW
    })[idx];
    let size = (if wide {
        opcodes::OPERAND_SIZES_WIDE
    } else {
        opcodes::OPERAND_SIZES_NARROW
    })[idx] as usize;
    if pc + size > code.len() {
        return None;
    }
    Some((
        op,
        Operands {
            code,
            base: pc as u32,
            row,
        },
        pc + size,
    ))
}

pub fn jump_target(pc: usize, offset: i32) -> usize {
    pc.wrapping_add_signed(offset as isize)
}
