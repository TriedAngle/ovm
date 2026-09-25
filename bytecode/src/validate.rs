use crate::opcodes::IndexKind;
use crate::program::{Constant, Function, Program};
use crate::{Opcode, Operand, RuntimeFn, jump_target, try_decode};

/// A compiled function (or program) is malformed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationError {
    /// An opcode byte has no `Opcode` mapping.
    InvalidOpcode { pc: usize },
    /// The stream ends inside an instruction.
    TruncatedInstruction { pc: usize },
    /// A register operand lies outside `[-(arity + 1), register_count)`
    /// (receiver at `-1`, formals below it).
    RegisterOutOfRange { pc: usize, reg: i32 },
    /// A constant-pool index operand exceeds the pool.
    ConstantIndexOutOfRange { pc: usize, index: u32 },
    /// A feedback operand's `[state, handler]` pair exceeds the vector.
    FeedbackSlotOutOfRange { pc: usize, slot: u32 },
    /// A runtime-call discriminant exceeds `RuntimeFn::COUNT`.
    RuntimeFnOutOfRange { pc: usize, discriminant: u32 },
    /// A jump target is outside the stream or not on an instruction
    /// boundary.
    BadJumpTarget { pc: usize, target: usize },
    /// `JumpLoop` (the safepoint-polling back-edge) targets a later pc.
    ForwardJumpLoop { pc: usize, target: usize },
    /// A handler range endpoint is outside the stream or off an
    /// instruction boundary, or the range is empty/misordered.
    BadHandlerRange { index: usize },
    /// Control can run past the end of the code stream.
    MissingTerminal,
    /// A `Constant::Callable` references a function table slot that does
    /// not exist.
    CallableOutOfRange { function: usize, id: u32 },
}

impl core::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid bytecode: {self:?}")
    }
}

impl std::error::Error for ValidationError {}

/// Validate one function. `program_len` bounds `Constant::Callable`
/// references; pass `0` to skip that check (it belongs to [`validate`]).
pub fn validate_function(f: &Function, program_len: usize) -> Result<(), ValidationError> {
    struct JumpSite {
        pc: usize,
        op: Opcode,
        offset: i32,
    }

    let mut starts: Vec<usize> = Vec::new();
    let mut jumps: Vec<JumpSite> = Vec::new();
    let mut last = None;

    let mut pc = 0usize;
    while pc < f.code.len() {
        let byte = f.code[pc];
        if Opcode::from_byte(byte).is_none() {
            return Err(ValidationError::InvalidOpcode { pc });
        }
        let Some((op, ops, next)) = try_decode(&f.code, pc) else {
            return Err(ValidationError::TruncatedInstruction { pc });
        };
        starts.push(pc);
        last = Some(op);

        for (i, kind) in op.operands().iter().enumerate() {
            // negative indices address the parameter area: the receiver at
            // -1 plus `arity` formals down to -(arity + 1)
            let check_reg = |reg: i32| {
                if reg < -(f.arity as i32) - 1 || reg >= f.register_count as i32 {
                    return Err(ValidationError::RegisterOutOfRange { pc, reg });
                }
                Ok(())
            };
            match kind {
                Operand::Register => check_reg(ops.reg(i))?,
                Operand::RegisterListStart => {
                    let base = ops.reg_list(i);
                    check_reg(base)?;
                    let count = ops.reg_count(i + 1);
                    if count > 0 {
                        check_reg(base + count as i32 - 1)?;
                    }
                }
                Operand::RegisterCount => {}
                Operand::Index => match op.index_kinds()[i] {
                    IndexKind::ConstantPool => {
                        let index = ops.idx(i) as u32;
                        if index as usize >= f.constants.len() {
                            return Err(ValidationError::ConstantIndexOutOfRange { pc, index });
                        }
                    }
                    IndexKind::Feedback => {
                        let slot = ops.idx(i) as u32;
                        if slot + 1 >= f.feedback_count {
                            return Err(ValidationError::FeedbackSlotOutOfRange { pc, slot });
                        }
                    }
                    IndexKind::RuntimeFn => {
                        let discriminant = ops.idx(i) as u32;
                        if discriminant >= RuntimeFn::COUNT as u32 {
                            return Err(ValidationError::RuntimeFnOutOfRange { pc, discriminant });
                        }
                    }
                    IndexKind::Unchecked => {}
                },
                Operand::Immediate | Operand::UImmediate => {}
            }
        }

        match op {
            Opcode::Jump | Opcode::JumpLoop => {
                jumps.push(JumpSite {
                    pc,
                    op,
                    offset: ops.imm(0),
                });
            }
            Opcode::JumpIfTruthy | Opcode::JumpIfFalsy | Opcode::JumpIfNotUndefined => {
                jumps.push(JumpSite {
                    pc,
                    op,
                    offset: ops.imm(0),
                });
            }
            Opcode::CompareJump => {
                jumps.push(JumpSite {
                    pc,
                    op,
                    offset: ops.imm(2),
                });
            }
            _ => {}
        }

        pc = next;
    }

    match last {
        Some(
            Opcode::Return | Opcode::Jump | Opcode::JumpLoop | Opcode::Throw | Opcode::ReThrow,
        ) => {}
        _ => return Err(ValidationError::MissingTerminal),
    }

    for jump in &jumps {
        let target = jump_target(jump.pc, jump.offset);
        if target >= f.code.len() || starts.binary_search(&target).is_err() {
            return Err(ValidationError::BadJumpTarget {
                pc: jump.pc,
                target,
            });
        }
        if jump.op == Opcode::JumpLoop && target > jump.pc {
            return Err(ValidationError::ForwardJumpLoop {
                pc: jump.pc,
                target,
            });
        }
    }

    let on_boundary = |pos: usize| pos == f.code.len() || starts.binary_search(&pos).is_ok();
    for (index, handler) in f.handlers.iter().enumerate() {
        let ok = on_boundary(handler.try_start)
            && on_boundary(handler.try_end)
            && on_boundary(handler.handler_pc)
            && handler.try_start < handler.try_end;
        if !ok {
            return Err(ValidationError::BadHandlerRange { index });
        }
    }

    if program_len > 0 {
        for (function, constant) in f.constants.iter().enumerate() {
            if let Constant::Callable(id) = constant
                && id.index() >= program_len
            {
                return Err(ValidationError::CallableOutOfRange { function, id: id.0 });
            }
        }
    }

    Ok(())
}

/// Validate every function of a program, including cross-references.
pub fn validate(program: &Program) -> Result<(), ValidationError> {
    let len = program.len();
    for function in program.functions() {
        validate_function(function, len)?;
    }
    Ok(())
}
