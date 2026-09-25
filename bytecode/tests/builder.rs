//! Usage and behavior tests for the bytecode builder: constpool dedup,
//! feedback pairs, temp registers, labels with automatic jump widening,
//! handler ranges, accumulator elision, and validation.

use bytecode::{
    BuildError, CallableKind, ConstIdx, Constant, FnBuilder, FunctionId, FunctionMeta, Opcode,
    Operand, Program, Reg, RegList, RtArg, RuntimeFn, ValidationError, try_decode, validate,
    validate_function,
};

#[derive(Debug, PartialEq)]
struct Instr {
    op: Opcode,
    /// Operands in stream order (registers signed, indices unsigned).
    ops: Vec<i64>,
    at: usize,
}

/// Decode a full instruction stream (panics if it does not decode cleanly).
fn decoded(code: &[u8]) -> Vec<Instr> {
    let mut out = Vec::new();
    let mut pc = 0;
    while pc < code.len() {
        let (op, operands, next) = try_decode(code, pc).expect("stream decodes");
        let ops = op
            .operands()
            .iter()
            .enumerate()
            .map(|(i, kind)| match kind {
                Operand::Register => operands.reg(i) as i64,
                Operand::RegisterListStart => operands.reg_list(i) as i64,
                Operand::RegisterCount => operands.reg_count(i) as i64,
                Operand::Immediate => operands.imm(i) as i64,
                Operand::UImmediate => operands.uimm(i) as i64,
                Operand::Index => operands.idx(i) as i64,
            })
            .collect();
        out.push(Instr { op, ops, at: pc });
        pc = next;
    }
    out
}

/// 130 bytes of non-elidable one-byte instructions (alternating
/// singletons), to push jump offsets past the narrow range.
fn emit_130_bytes(b: &mut FnBuilder) {
    for i in 0..130 {
        if i % 2 == 0 {
            b.load_zero();
        } else {
            b.load_true();
        }
    }
}

fn meta() -> FunctionMeta {
    FunctionMeta {
        name: Some(b"test".as_slice().into()),
        kind: CallableKind::Normal,
        length: 0,
        strict: true,
    }
}

// ---------------------------------------------------------------------------
// constant pool
// ---------------------------------------------------------------------------

#[test]
fn constant_pool_dedups() {
    let mut b = FnBuilder::new(0);

    let x = b.name(b"x");
    assert_eq!(b.name(b"x"), x, "same string hits the same pool slot");
    assert_eq!(
        b.constant(Constant::String(b"x".as_slice().into())),
        x,
        "name() and constant(String) share the pool"
    );

    let f = b.constant(Constant::Float(0.5));
    assert_eq!(b.constant(Constant::Float(0.5)), f);

    let zero = b.constant(Constant::Float(0.0));
    let neg_zero = b.constant(Constant::Float(-0.0));
    assert_ne!(zero, neg_zero, "0.0 and -0.0 must not merge");

    let big = b.constant(Constant::Smi(1 << 40));
    assert_eq!(b.constant(Constant::Smi(1 << 40)), big);

    let proto = b.constant(Constant::ObjectPrototype);
    assert_eq!(b.constant(Constant::ObjectPrototype), proto);
    assert_eq!(
        b.constant(Constant::FunctionPrototype),
        b.constant(Constant::FunctionPrototype)
    );

    let names: Vec<Box<[u8]>> = vec![b"a".as_slice().into(), b"b".as_slice().into()];
    let scope = b.constant(Constant::ContextNames(names.clone()));
    assert_eq!(b.constant(Constant::ContextNames(names)), scope);

    let callable = b.constant(Constant::Callable(FunctionId(3)));
    assert_eq!(b.constant(Constant::Callable(FunctionId(3))), callable);

    // every distinct entry is pooled exactly once
    assert_eq!(
        b.constants_len(),
        9,
        "x, 0.5, 0.0, -0.0, smi, obj-proto, fn-proto, names, callable"
    );

    let func = b.finish(meta()).unwrap();
    assert_eq!(func.constants.len(), 9);
    assert!(matches!(
        func.constants[x.index() as usize],
        Constant::String(_)
    ));
}

#[test]
fn staged_runtime_calls_lay_out_the_window_in_order() {
    // SetFunctionName(closure = acc, name, prefix): the accumulator value
    // must be captured into slot 0 even though later loads clobber acc
    let mut b = FnBuilder::new(0);
    b.load_name(b"fn");
    let obj = b.stage_acc();
    let name = b.name(b"key");
    b.call_runtime_staged(
        RuntimeFn::SetFunctionName,
        &[RtArg::Acc, RtArg::Const(name), RtArg::Smi(2)],
    );
    b.load(obj);
    b.drop_temp();
    b.ret();

    let f = b.finish(meta()).unwrap();
    let instrs = decoded(&f.code);
    // obj staged at temp 0; the window occupies temps 1..4
    assert_eq!(
        instrs,
        vec![
            Instr {
                op: Opcode::LoadConstant,
                ops: vec![0],
                at: 0
            }, // "fn"
            Instr {
                op: Opcode::Store,
                ops: vec![0],
                at: 2
            }, // obj = temp 0
            Instr {
                op: Opcode::Store,
                ops: vec![1],
                at: 4
            }, // acc -> window slot 0
            Instr {
                op: Opcode::LoadConstant,
                ops: vec![1],
                at: 6
            }, // "key"
            Instr {
                op: Opcode::Store,
                ops: vec![2],
                at: 8
            },
            Instr {
                op: Opcode::LoadSmi,
                ops: vec![2],
                at: 10
            },
            Instr {
                op: Opcode::Store,
                ops: vec![3],
                at: 12
            },
            Instr {
                op: Opcode::CallRuntime,
                ops: vec![19, 1, 3],
                at: 14
            }, // SetFunctionName discriminant
            Instr {
                op: Opcode::Load,
                ops: vec![0],
                at: 18
            },
            Instr {
                op: Opcode::Return,
                ops: vec![],
                at: 20
            },
        ]
    );
    validate_function(&f, 0).unwrap();
}

#[test]
fn feedback_slots_allocate_pairs() {
    let mut b = FnBuilder::new(0);
    assert_eq!(b.new_feedback().index(), 0);
    assert_eq!(b.new_feedback().index(), 2);
    assert_eq!(b.new_feedback().index(), 4);

    b.load_name(b"global");
    let g = b.name(b"global");
    let fb = b.new_feedback();
    b.store_global(g, fb);
    b.ret();

    let f = b.finish(meta()).unwrap();
    assert_eq!(f.feedback_count, 8);
    validate_function(&f, 0).unwrap();
}

// ---------------------------------------------------------------------------
// smi loading and automatic widening
// ---------------------------------------------------------------------------

#[test]
fn smi_loads_pick_the_cheapest_encoding() {
    let mut b = FnBuilder::new(0);
    b.load_smi(5);
    b.ret();
    let f = b.finish(meta()).unwrap();
    assert_eq!(f.code.len(), 3);
    assert_eq!(
        decoded(&f.code)[0],
        Instr {
            op: Opcode::LoadSmi,
            ops: vec![5],
            at: 0
        }
    );

    let mut b = FnBuilder::new(0);
    b.load_smi(200); // > i8::MAX: auto-widens instead of failing
    b.ret();
    let f = b.finish(meta()).unwrap();
    assert_eq!(f.code[0], Opcode::Wide as u8);
    assert_eq!(
        decoded(&f.code)[0],
        Instr {
            op: Opcode::LoadSmi,
            ops: vec![200],
            at: 0
        }
    );

    let mut b = FnBuilder::new(0);
    b.load_smi(100_000); // beyond i16: rides the constant pool
    b.ret();
    let f = b.finish(meta()).unwrap();
    assert_eq!(
        decoded(&f.code)[0],
        Instr {
            op: Opcode::LoadConstant,
            ops: vec![0],
            at: 0
        }
    );
    assert_eq!(f.constants, vec![Constant::Smi(100_000)]);
}

#[test]
fn wide_operands_scale_whole_instruction() {
    // a constpool index past 255 widens *all* scalable operands of its
    // instruction, not just the index
    let mut b = FnBuilder::new(1);
    for i in 0..300 {
        b.name(format!("n{i}").as_bytes());
    }
    let name = b.name(b"target");
    let fb = b.new_feedback();
    b.load(b.param(0));
    b.load_global(name, fb);
    b.ret();

    let f = b.finish(meta()).unwrap();
    assert_eq!(name.index(), 300);
    assert_eq!(f.code[2], Opcode::Wide as u8);
    assert_eq!(
        decoded(&f.code)[1],
        Instr {
            op: Opcode::LoadGlobal,
            ops: vec![300, 0],
            at: 2
        }
    );
    validate_function(&f, 0).unwrap();
}

// ---------------------------------------------------------------------------
// temps and frame sizing
// ---------------------------------------------------------------------------

#[test]
fn temps_allocate_above_temp_base_and_size_the_frame() {
    let mut b = FnBuilder::new(2);
    b.set_temp_base(3); // frontend owns locals 0..3

    b.load_smi(1);
    let a = b.stage_acc(); // temp 3
    let b_ = b.temp(); // temp 4
    b.load_smi(2);
    b.store(b_);
    b.load(a);
    b.add(b_);
    b.store(Reg::new(0)); // a frontend local
    b.drop_temp();
    b.drop_temp();
    b.ret();

    assert_eq!(a, Reg::new(3));
    assert_eq!(b_, Reg::new(4));

    let f = b.finish(meta()).unwrap();
    assert_eq!(f.register_count, 5, "locals 0..3 plus temps 3,4");
    assert_eq!(f.arity, 2);
    validate_function(&f, 0).unwrap();
}

#[test]
fn unbalanced_temps_fail_finish() {
    let mut b = FnBuilder::new(0);
    b.load_zero();
    let _t = b.stage_acc();
    b.ret();
    let err = b.finish(meta()).unwrap_err();
    assert_eq!(err, BuildError::UnbalancedTemps);
}

#[test]
fn reglist_tail_extends_the_frame() {
    // the frame must cover the whole argument window base..base+count,
    // even registers no other instruction names
    let mut b = FnBuilder::new(0);
    b.call_no_feedback(b.this_reg(), RegList::new(Reg::new(0), 3));
    b.ret();

    let f = b.finish(meta()).unwrap();
    assert_eq!(f.register_count, 3, "args window 0..3 sizes the frame");
    validate_function(&f, 0).unwrap();

    // and the validator checks the window end, not just its base
    let mut f = f.clone();
    f.register_count = 2;
    assert_eq!(
        validate_function(&f, 0).unwrap_err(),
        ValidationError::RegisterOutOfRange { pc: 0, reg: 2 }
    );
}

// ---------------------------------------------------------------------------
// register convention (vm/src/stack.rs frame layout)
// ---------------------------------------------------------------------------

/// Regression: the builder's negative registers must match the VM's frame
/// layout — negative index `i` addresses parameter-area slot `(-i - 1)`,
/// where slot 0 is the receiver and the formals follow in order. An
/// off-by-one here compiles fine everywhere but silently reads `this`
/// instead of the first formal once wired into the VM.
#[test]
fn register_convention_matches_the_vm_frame_layout() {
    let b = FnBuilder::new(2);
    assert_eq!(b.this_reg(), Reg::new(-1), "receiver is parameter slot 0");
    assert_eq!(b.param(0), Reg::new(-2), "formal 0 is parameter slot 1");
    assert_eq!(b.param(1), Reg::new(-3), "formal 1 is parameter slot 2");
}

#[test]
fn emitted_param_loads_use_the_vm_convention() {
    // fn f(a) { return a } — the emitted Load must address formal 0 at -2,
    // not the receiver at -1
    let mut b = FnBuilder::new(1);
    b.load(b.param(0));
    b.ret();
    let f = b.finish(meta()).unwrap();
    assert_eq!(
        decoded(&f.code),
        vec![
            Instr {
                op: Opcode::Load,
                ops: vec![-2],
                at: 0
            },
            Instr {
                op: Opcode::Return,
                ops: vec![],
                at: 2
            },
        ]
    );
    validate_function(&f, 0).unwrap();
}

#[test]
fn validate_window_covers_receiver_and_all_formals() {
    // arity 2: receiver -1 and formals -2, -3 are in window; -4 is not
    let mut b = FnBuilder::new(2);
    b.load(b.this_reg());
    b.load(b.param(1));
    b.ret();
    let f = b.finish(meta()).unwrap();
    validate_function(&f, 0).unwrap();

    // patch the second load's operand byte from -3 to -4
    let mut f = f.clone();
    assert_eq!(f.code[3], (-3i32) as u8);
    f.code[3] = (-4i32) as u8;
    assert_eq!(
        validate_function(&f, 0).unwrap_err(),
        ValidationError::RegisterOutOfRange { pc: 2, reg: -4 }
    );
}

// ---------------------------------------------------------------------------
// labels, jumps, relaxation
// ---------------------------------------------------------------------------

#[test]
fn short_forward_jump_stays_narrow() {
    let mut b = FnBuilder::new(0);
    let end = b.new_label();
    b.jump(end);
    b.load_zero();
    b.load_smi(3);
    b.bind(end);
    b.ret();

    let f = b.finish(meta()).unwrap();
    let instrs = decoded(&f.code);
    assert_eq!(f.code[0], Opcode::Jump as u8, "no Wide prefix needed");
    assert_eq!(
        instrs[0],
        Instr {
            op: Opcode::Jump,
            ops: vec![5],
            at: 0
        }
    );
    assert_eq!(instrs.len(), 4);
    validate_function(&f, 0).unwrap();
}

#[test]
fn long_forward_jump_widens_automatically() {
    let mut b = FnBuilder::new(0);
    let end = b.new_label();
    b.jump(end);
    emit_130_bytes(&mut b);
    b.bind(end);
    b.ret();

    let f = b.finish(meta()).unwrap();
    assert_eq!(f.code[0], Opcode::Wide as u8, "jump grew a Wide prefix");
    assert_eq!(f.code.len(), 4 + 130 + 1, "4-byte jump + body + Return");

    let instrs = decoded(&f.code);
    assert_eq!(
        instrs[0],
        Instr {
            op: Opcode::Jump,
            ops: vec![134],
            at: 0
        }
    );
    assert_eq!(instrs[1].at, 4);
    assert_eq!(
        instrs.last().unwrap().at,
        134,
        "jump lands exactly on Return"
    );
    validate_function(&f, 0).unwrap();
}

#[test]
fn backward_jump_loop_stays_narrow() {
    let mut b = FnBuilder::new(0);
    let head = b.new_label();
    let exit = b.new_label();
    b.bind(head);
    b.load_zero();
    b.load_smi(7);
    b.jump_loop(head);
    b.bind(exit);
    b.ret();

    let f = b.finish(meta()).unwrap();
    let instrs = decoded(&f.code);
    assert_eq!(
        instrs[2],
        Instr {
            op: Opcode::JumpLoop,
            ops: vec![-3],
            at: 3
        }
    );
    validate_function(&f, 0).unwrap();
}

#[test]
fn wide_loop_body_widens_its_back_edge() {
    let mut b = FnBuilder::new(0);
    let head = b.new_label();
    let exit = b.new_label();
    b.bind(head);
    emit_130_bytes(&mut b);
    b.jump_loop(head);
    b.bind(exit);
    b.ret();

    let f = b.finish(meta()).unwrap();
    assert_eq!(f.code[130], Opcode::Wide as u8);
    let instrs = decoded(&f.code);
    assert_eq!(
        instrs[130],
        Instr {
            op: Opcode::JumpLoop,
            ops: vec![-130],
            at: 130
        }
    );
    validate_function(&f, 0).unwrap();
}

#[test]
fn wide_and_narrow_jumps_coexist() {
    // one long forward jump (widened) plus a short loop back-edge (narrow),
    // with two labels bound at the same position
    let mut b = FnBuilder::new(0);
    let far = b.new_label();
    let head = b.new_label();

    b.jump(far); // forward over the long body -> wide
    emit_130_bytes(&mut b);
    b.bind(far);
    b.bind(head); // far and head coincide
    b.load_zero();
    b.load_smi(3);
    b.jump_loop(head); // backward 3 bytes -> narrow

    let f = b.finish(meta()).unwrap();
    assert_eq!(f.code[0], Opcode::Wide as u8);
    let instrs = decoded(&f.code);
    assert_eq!(instrs[0].op, Opcode::Jump);
    assert_eq!(instrs[0].ops[0], 134);
    let back = instrs.last().unwrap();
    assert_eq!(back.op, Opcode::JumpLoop);
    assert_eq!(back.ops[0], -3);
    assert_eq!(back.at, 137);
    validate_function(&f, 0).unwrap();
}

#[test]
fn jumps_to_unbound_label_fail_finish() {
    let mut b = FnBuilder::new(0);
    let nowhere = b.new_label();
    let after = b.new_label();
    b.jump(nowhere);
    b.bind(after);
    b.ret();
    assert_eq!(b.finish(meta()).unwrap_err(), BuildError::UnboundLabel);
}

#[test]
fn conditional_jumps_read_the_accumulator_but_keep_its_state() {
    let mut b = FnBuilder::new(0);
    let end = b.new_label();
    b.load(Reg::new(1));
    b.jump_if_falsy(end);
    b.load(Reg::new(1)); // elided: the conditional jump did not clobber acc
    b.bind(end);
    b.ret();

    let f = b.finish(meta()).unwrap();
    let instrs = decoded(&f.code);
    assert_eq!(instrs.len(), 3);
    assert_eq!(
        instrs[1],
        Instr {
            op: Opcode::JumpIfFalsy,
            ops: vec![2],
            at: 2
        }
    );
}

// ---------------------------------------------------------------------------
// handler ranges
// ---------------------------------------------------------------------------

#[test]
fn handler_anchors_record_final_pcs() {
    let mut b = FnBuilder::new(0);
    let end = b.new_label();

    let t = b.begin_try(); // try_start = 0
    b.load_zero(); // 1 byte of try body
    b.end_try(t); // try_end = 1
    b.jump(end); // 2-byte jump over the handler
    b.handler_entry(t); // handler_pc = 3
    b.load_true(); // handler body
    b.bind(end);
    b.ret();

    let f = b.finish(meta()).unwrap();
    assert_eq!(f.handlers.len(), 1);
    let h = f.handlers[0];
    assert_eq!((h.try_start, h.try_end, h.handler_pc), (0, 1, 3));
    validate_function(&f, 0).unwrap();
}

#[test]
fn handler_anchors_survive_jump_widening() {
    let mut b = FnBuilder::new(0);
    let end = b.new_label();

    let t = b.begin_try();
    emit_130_bytes(&mut b); // long try body
    b.end_try(t);
    b.jump(end); // must widen: it clears the equally long handler body
    b.handler_entry(t);
    emit_130_bytes(&mut b); // long handler body
    b.bind(end);
    b.ret();

    let f = b.finish(meta()).unwrap();
    let h = f.handlers[0];
    assert_eq!(h.try_start, 0);
    assert_eq!(
        h.try_end, 130,
        "handler range is unchanged by later widening"
    );
    assert_eq!(
        h.handler_pc, 134,
        "4-byte jump sits between try_end and handler"
    );
    assert_eq!(f.code[130], Opcode::Wide as u8);
    validate_function(&f, 0).unwrap();
}

#[test]
fn unfinished_handler_fails_finish() {
    let mut b = FnBuilder::new(0);
    let _t = b.begin_try();
    b.load_zero();
    b.ret();
    assert_eq!(b.finish(meta()).unwrap_err(), BuildError::UnfinishedHandler);

    let mut b = FnBuilder::new(0);
    let t = b.begin_try();
    b.load_zero();
    b.end_try(t);
    b.ret();
    assert_eq!(b.finish(meta()).unwrap_err(), BuildError::UnfinishedHandler);
}

// ---------------------------------------------------------------------------
// accumulator tracking
// ---------------------------------------------------------------------------

#[test]
fn redundant_loads_are_elided() {
    let mut b = FnBuilder::new(0);
    b.load(Reg::new(0));
    b.load(Reg::new(0));
    b.ret();
    let f = b.finish(meta()).unwrap();
    assert_eq!(decoded(&f.code).len(), 2, "Load, Return");
}

#[test]
fn store_then_load_costs_only_the_store() {
    let mut b = FnBuilder::new(0);
    b.load(Reg::new(0));
    b.store(Reg::new(1));
    b.load(Reg::new(1));
    b.ret();
    let f = b.finish(meta()).unwrap();
    assert_eq!(
        decoded(&f.code),
        vec![
            Instr {
                op: Opcode::Load,
                ops: vec![0],
                at: 0
            },
            Instr {
                op: Opcode::Store,
                ops: vec![1],
                at: 2
            },
            Instr {
                op: Opcode::Return,
                ops: vec![],
                at: 4
            },
        ]
    );
}

#[test]
fn dead_load_store_pair_vanishes() {
    // from a cold accumulator the Load must stay (the store side drops),
    // but once acc is known to hold r, an entire load/store round trip
    // is free
    let mut b = FnBuilder::new(0);
    b.load(Reg::new(0));
    b.store(Reg::new(0));
    b.store(Reg::new(0)); // elided: acc already == reg 0
    b.load(Reg::new(0)); // elided
    b.store(Reg::new(0)); // elided
    b.ret();
    let f = b.finish(meta()).unwrap();
    assert_eq!(
        decoded(&f.code),
        vec![
            Instr {
                op: Opcode::Load,
                ops: vec![0],
                at: 0
            },
            Instr {
                op: Opcode::Return,
                ops: vec![],
                at: 2
            },
        ]
    );
}

#[test]
fn repeated_constant_and_singleton_loads_are_elided() {
    let mut b = FnBuilder::new(0);
    let k = b.name(b"key");
    b.load_constant(k);
    b.load_constant(k);
    b.load_name(b"key");
    b.load_undefined();
    b.load_undefined();
    b.ret();
    let f = b.finish(meta()).unwrap();
    assert_eq!(
        decoded(&f.code),
        vec![
            Instr {
                op: Opcode::LoadConstant,
                ops: vec![0],
                at: 0
            },
            Instr {
                op: Opcode::LoadUndefined,
                ops: vec![],
                at: 2
            },
            Instr {
                op: Opcode::Return,
                ops: vec![],
                at: 3
            },
        ]
    );
}

#[test]
fn register_writes_invalidate_accumulator_knowledge() {
    // a store to another register drops acc knowledge of reg 0
    let mut b = FnBuilder::new(0);
    b.load(Reg::new(0));
    b.store(Reg::new(1));
    b.load(Reg::new(0));
    b.ret();
    assert_eq!(decoded(&b.finish(meta()).unwrap().code).len(), 4);

    // Move writes its destination: acc knowledge of that register dies
    let mut b = FnBuilder::new(0);
    b.load(Reg::new(0));
    b.move_reg(Reg::new(0), Reg::new(1)); // dst = 0
    b.load(Reg::new(0));
    b.ret();
    assert_eq!(decoded(&b.finish(meta()).unwrap().code).len(), 4);

    // Move only reads its source: acc knowledge of reg 0 survives
    let mut b = FnBuilder::new(0);
    b.load(Reg::new(0));
    b.move_reg(Reg::new(1), Reg::new(0)); // dst = 1
    b.load(Reg::new(0));
    b.ret();
    assert_eq!(
        decoded(&b.finish(meta()).unwrap().code).len(),
        3,
        "Load, Move, Return"
    );

    // PushContext saves the old context into its operand register
    let mut b = FnBuilder::new(0);
    b.load(Reg::new(0));
    b.push_context(Reg::new(0));
    b.load(Reg::new(0));
    b.ret();
    assert_eq!(decoded(&b.finish(meta()).unwrap().code).len(), 4);
}

#[test]
fn bind_joins_paths_and_drops_accumulator_knowledge() {
    let mut b = FnBuilder::new(0);
    let l = b.new_label();
    b.load(Reg::new(0));
    b.bind(l);
    b.load(Reg::new(0)); // must be re-emitted: another path joins here
    b.ret();
    assert_eq!(decoded(&b.finish(meta()).unwrap().code).len(), 3);
}

#[test]
fn arithmetic_and_calls_clobber_the_accumulator() {
    let mut b = FnBuilder::new(0);
    b.load(Reg::new(0));
    b.add(Reg::new(1));
    b.load(Reg::new(0));
    let fb = b.new_feedback();
    b.call(Reg::new(2), RegList::new(Reg::new(3), 2), fb);
    b.load(Reg::new(0));
    b.ret();
    assert_eq!(decoded(&b.finish(meta()).unwrap().code).len(), 6);
}

// ---------------------------------------------------------------------------
// misuse is caught
// ---------------------------------------------------------------------------

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "reads the accumulator")]
fn reading_an_undefined_accumulator_panics() {
    let mut b = FnBuilder::new(0);
    b.store(Reg::new(0));
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "reads the accumulator")]
fn keyed_load_with_undefined_accumulator_panics() {
    let mut b = FnBuilder::new(0);
    let fb = b.new_feedback();
    b.load_keyed_property(Reg::new(0), fb);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "reads the accumulator")]
fn context_slot_store_with_undefined_accumulator_panics() {
    let mut b = FnBuilder::new(0);
    b.store_context_slot(0, 0);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "reads the accumulator")]
fn add_parent_with_undefined_accumulator_panics() {
    let mut b = FnBuilder::new(0);
    let name = b.name(b"parent");
    b.add_parent(Reg::new(0), name);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "must go through the label/jump methods")]
fn raw_jumps_are_rejected() {
    let mut b = FnBuilder::new(0);
    b.raw(Opcode::Jump, &[0]);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "created by a different FnBuilder")]
fn foreign_labels_are_rejected() {
    let mut a = FnBuilder::new(0);
    let l = a.new_label();
    let mut b = FnBuilder::new(0);
    b.jump(l);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "label bound twice")]
fn double_bind_panics() {
    let mut b = FnBuilder::new(0);
    let l = b.new_label();
    b.bind(l);
    b.bind(l);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "JumpLoop must target an earlier pc")]
fn forward_jump_loop_panics() {
    let mut b = FnBuilder::new(0);
    let l = b.new_label();
    b.load_zero();
    b.jump_loop(l);
    b.bind(l);
    b.ret();
    let _ = b.finish(meta()).unwrap();
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "below the parameter window")]
fn register_below_parameter_window_panics() {
    let mut b = FnBuilder::new(1); // receiver: -1, formal 0: -2
    b.load(Reg::new(-3));
}

// ---------------------------------------------------------------------------
// validation of corrupted functions
// ---------------------------------------------------------------------------

/// if (p0) { g = "x" }  — one conditional jump, a global store, feedback.
fn good_function() -> bytecode::Function {
    let mut b = FnBuilder::new(1);
    let skip = b.new_label();
    b.load(b.param(0));
    b.jump_if_falsy(skip);
    b.load_name(b"x");
    let x = b.name(b"x");
    let fb = b.new_feedback();
    b.store_global(x, fb);
    b.bind(skip);
    b.load_undefined();
    b.ret();
    b.finish(meta()).unwrap()
}

#[test]
fn builder_output_validates() {
    let f = good_function();
    validate_function(&f, 0).unwrap();
}

#[test]
fn validate_rejects_invalid_opcode() {
    let mut f = good_function();
    f.code[0] = 0xEE;
    assert_eq!(
        validate_function(&f, 0),
        Err(ValidationError::InvalidOpcode { pc: 0 })
    );
}

#[test]
fn validate_rejects_truncated_stream() {
    let mut f = good_function();
    f.code.truncate(1); // opcode byte without its register operand
    assert_eq!(
        validate_function(&f, 0),
        Err(ValidationError::TruncatedInstruction { pc: 0 })
    );
}

#[test]
fn validate_rejects_mid_instruction_jump_target() {
    let mut f = good_function();
    // jump_if_falsy sits at pc 2; patch its offset to land inside the
    // StoreGlobal operand bytes (6,7,8)
    let instrs = decoded(&f.code);
    let jump = &instrs[1];
    assert_eq!(jump.op, Opcode::JumpIfFalsy);
    f.code[jump.at + 1] = 5; // 2 + 5 = 7 is StoreGlobal's name-index byte
    assert_eq!(
        validate_function(&f, 0),
        Err(ValidationError::BadJumpTarget { pc: 2, target: 7 })
    );
}

#[test]
fn validate_rejects_forward_jump_loop() {
    let mut f = good_function();
    let instrs = decoded(&f.code);
    let jump = &instrs[1];
    f.code[jump.at] = Opcode::JumpLoop as u8; // keep the forward offset
    let target = jump.at + jump.ops[0] as usize;
    assert_eq!(
        validate_function(&f, 0),
        Err(ValidationError::ForwardJumpLoop {
            pc: jump.at,
            target
        })
    );
}

#[test]
fn validate_rejects_out_of_range_operands() {
    let mut f = good_function();

    f.feedback_count = 0; // the StoreGlobal site needs slots 0 and 1
    assert_eq!(
        validate_function(&f, 0).unwrap_err(),
        ValidationError::FeedbackSlotOutOfRange { pc: 6, slot: 0 }
    );

    let mut f = good_function();
    f.constants.clear();
    assert_eq!(
        validate_function(&f, 0).unwrap_err(),
        ValidationError::ConstantIndexOutOfRange { pc: 4, index: 0 }
    );

    let mut f = good_function();
    f.register_count = 0; // only the parameter register is used, still legal
    validate_function(&f, 0).unwrap();
    f.arity = 0; // ...but now Reg(-2) is below the parameter window (receiver only)
    assert_eq!(
        validate_function(&f, 0).unwrap_err(),
        ValidationError::RegisterOutOfRange { pc: 0, reg: -2 }
    );
}

#[test]
fn validate_rejects_bad_handler_ranges() {
    let mut b = FnBuilder::new(0);
    let end = b.new_label();
    let t = b.begin_try();
    b.load_zero();
    b.end_try(t);
    b.jump(end);
    b.handler_entry(t);
    b.load_true();
    b.bind(end);
    b.ret();
    let mut f = b.finish(meta()).unwrap();
    validate_function(&f, 0).unwrap();

    f.handlers[0].handler_pc = 2; // inside the jump-over's offset byte
    assert_eq!(
        validate_function(&f, 0).unwrap_err(),
        ValidationError::BadHandlerRange { index: 0 }
    );

    let mut f2 = f.clone();
    f2.handlers[0].try_end = f2.handlers[0].try_start; // empty range
    assert_eq!(
        validate_function(&f2, 0).unwrap_err(),
        ValidationError::BadHandlerRange { index: 0 }
    );
}

#[test]
fn validate_rejects_falling_off_the_end() {
    let mut b = FnBuilder::new(0);
    b.load_zero(); // no terminal instruction
    let f = b.finish(meta()).unwrap();
    assert_eq!(
        validate_function(&f, 0),
        Err(ValidationError::MissingTerminal)
    );
}

// ---------------------------------------------------------------------------
// program table
// ---------------------------------------------------------------------------

#[test]
fn program_collects_functions_and_checks_cross_references() {
    let mut p = Program::with_capacity(2);

    // script referencing a nested function that does not exist yet
    let mut b = FnBuilder::new(0);
    let tmpl = b.constant(Constant::Callable(FunctionId(1)));
    let tmpl_again = b.constant(Constant::Callable(FunctionId(1)));
    assert_eq!(tmpl, tmpl_again);
    b.create_closure(tmpl);
    b.ret();
    let script = b.finish(meta()).unwrap();
    let script_id = p.add_function(script);
    assert_eq!(script_id, FunctionId::SCRIPT);

    let mut inner = FnBuilder::new(0);
    inner.load_smi(7);
    inner.ret();
    let inner_id = p.add_function(inner.finish(meta()).unwrap());
    assert_eq!(inner_id, FunctionId(1));

    assert_eq!(p.len(), 2);
    assert_eq!(
        p.function_ids().collect::<Vec<_>>(),
        vec![FunctionId(0), FunctionId(1)]
    );
    assert_eq!(p.function(script_id).register_count, 0);
    assert_eq!(
        p.constant(p.function(script_id), 0),
        &Constant::Callable(FunctionId(1))
    );
    assert_eq!(p.name(p.function(script_id)), Some(&b"test"[..]));

    validate(&p).unwrap();

    // a dangling callable reference is caught at the program level
    let mut bad = Program::new();
    let mut b = FnBuilder::new(0);
    b.constant(Constant::Callable(FunctionId(9)));
    b.load_zero();
    b.ret();
    bad.add_function(b.finish(meta()).unwrap());
    assert_eq!(
        validate(&bad).unwrap_err(),
        ValidationError::CallableOutOfRange { function: 0, id: 9 }
    );
}

// ---------------------------------------------------------------------------
// a full worked example: if/else with a global store and a call
// ---------------------------------------------------------------------------

#[test]
fn worked_example_builds_valid_bytecode() {
    // function f(o) {
    //     let sum = 0;
    //     for (let i = 0; i < 10; i = i + 1) {
    //         sum = sum + o.x;      // named load with feedback
    //     }
    //     return sum;
    // }
    let mut b = FnBuilder::new(1);
    b.set_temp_base(2); // locals: sum=0, i=1

    let sum = Reg::new(0);
    let i = Reg::new(1);
    let obj = b.param(0);
    let x = b.name(b"x");

    b.load_zero();
    b.store(sum);
    b.store(i);

    let head = b.new_label();
    let exit = b.new_label();
    b.bind(head);
    b.load(i);
    b.load_smi(10);
    let ten = b.stage_acc();
    b.less_than(ten);
    b.jump_if_falsy(exit);
    b.load(sum);
    let site = b.new_feedback();
    b.load_named_property(obj, x, site);
    b.add(sum);
    b.store(sum);
    b.drop_temp();
    b.load(i);
    b.load_smi(1);
    let one = b.stage_acc();
    b.add(one);
    b.store(i);
    b.drop_temp();
    b.jump_loop(head);

    b.bind(exit);
    b.load(sum);
    b.ret();

    let f = b
        .finish(FunctionMeta {
            name: Some(b"f".as_slice().into()),
            kind: CallableKind::Normal,
            length: 1,
            strict: false,
        })
        .unwrap();

    assert_eq!(f.arity, 1);
    assert_eq!(f.register_count, 3, "locals 0,1 plus the staged temp at 2");
    assert_eq!(f.constants, vec![Constant::String(b"x".as_slice().into())]);
    assert_eq!(f.feedback_count, 2);

    let instrs = decoded(&f.code);
    assert_eq!(instrs[0].op, Opcode::LoadZero);
    assert_eq!(instrs[0].ops, Vec::<i64>::new());
    assert_eq!(instrs.last().unwrap().op, Opcode::Return);

    // every jump lands on an instruction boundary, the back-edge is
    // backward, and the whole function passes validation
    validate_function(&f, 1).unwrap();

    let mut p = Program::new();
    p.add_function(f);
    validate(&p).unwrap();
}
