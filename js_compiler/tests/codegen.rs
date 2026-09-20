//! Golden bytecode tests: compile snippets and assert on the emitted
//! instruction stream.

use bytecode::{Opcode, Operand};
use ir::{Constant, Function, FunctionId, Program};
use js_compiler::{CompileError, compile_script};
use js_parser::{Parser, Utf8SliceStream};

fn compile(src: &str) -> Result<Program, CompileError> {
    let mut p = Parser::new(Utf8SliceStream::new(src));
    p.parse_script().expect("parse failed");
    let ast = p.into_ast();
    compile_script(&ast)
}

fn script_fn(program: &Program) -> &Function {
    program.function(FunctionId::SCRIPT)
}

/// Render the script function's bytecode with constants inlined and jump
/// operands shown as absolute target pcs.
fn disasm(program: &Program) -> Vec<String> {
    let f = script_fn(program);
    let code = program.code(f);
    let mut out = Vec::new();
    let mut pc = 0;
    while pc < code.len() {
        let (op, ops, next) = bytecode::decode(code, pc);
        let mut parts = vec![format!("{op:?}")];
        for (i, kind) in op.operands().iter().enumerate() {
            let v = match kind {
                Operand::Register => ops.reg(i) as i64,
                Operand::RegisterListStart => ops.reg_list(i) as i64,
                Operand::RegisterCount => ops.reg_count(i) as i64,
                Operand::Immediate
                    if matches!(
                        op,
                        Opcode::Jump
                            | Opcode::JumpLoop
                            | Opcode::JumpIfTruthy
                            | Opcode::JumpIfFalsy
                    ) =>
                {
                    pc as i64 + ops.imm(i) as i64
                }
                Operand::Immediate => ops.imm(i) as i64,
                Operand::UImmediate => ops.uimm(i) as i64,
                Operand::Index => ops.idx(i) as i64,
            };
            let v = if *kind == Operand::Index
                && (op == Opcode::LoadConstant || op == Opcode::CreateClosure)
            {
                render_constant(program.constants(f), ops.idx(i))
            } else {
                v.to_string()
            };
            parts.push(v);
        }
        out.push(parts.join(" "));
        pc = next;
    }
    out
}

fn render_constant(constants: &[Constant], idx: usize) -> String {
    match &constants[idx] {
        Constant::String(bytes) => format!("#str[{:?}]", String::from_utf8_lossy(bytes)),
        Constant::Float(f) => format!("#f64[{f}]"),
        Constant::Smi(v) => format!("#smi[{v}]"),
        Constant::Callable(fid) => format!("#fn[{}]", fid.0),
        Constant::ContextNames(names) => format!("#ctxnames[{names:?}]"),
        Constant::ObjectPrototype => "#object-prototype".into(),
        Constant::FunctionPrototype => "#function-prototype".into(),
    }
}

fn body(script: &Program) -> Vec<String> {
    disasm(script)
}

// Layout note: scripts (function 0) carry a completion-value register between
// the context-save slot and the temps: prologue stores `undefined` into it,
// every expression statement overwrites it, the epilogue loads it back.

#[test]
fn add_smi_temps_above_locals() {
    let script = compile("1 + 2;").unwrap();
    let out = body(&script);
    assert_eq!(
        script_fn(&script).register_count,
        4,
        "locals(0) + ctx + completion + 2 temps"
    );
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "LoadSmi 1",
        "Store 2",
        "LoadSmi 2",
        "Store 3",
        "Load 2",
        "Add 3",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn nested_binary_reuses_temps() {
    let script = compile("1 + 2 + 3;").unwrap();
    let out = body(&script);
    assert_eq!(script_fn(&script).register_count, 4);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "LoadSmi 1",
        "Store 2",
        "LoadSmi 2",
        "Store 3",
        "Load 2",
        "Add 3",
        "Store 2",
        "LoadSmi 3",
        "Store 3",
        "Load 2",
        "Add 3",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn strict_equality_yields_singletons_and_not_flips() {
    let script = compile("1 !== 2;").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "LoadSmi 1",
        "Store 2",
        "LoadSmi 2",
        "Store 3",
        "Load 2",
        "EqualStrict 3",
        "JumpIfTruthy 28",
        "LoadTrue",
        "Jump 29",
        "LoadFalse",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn number_literals_outside_smi_range_become_constants() {
    let script = compile("1.5 + 2;").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "LoadConstant #f64[1.5]",
        "Store 2",
        "LoadSmi 2",
        "Store 3",
        "Load 2",
        "Add 3",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn string_literal_is_interned_constant() {
    let script = compile("'a' + 'b';").unwrap();
    let out = body(&script);
    assert_eq!(out[2], "LoadUndefined");
    assert!(out[4].starts_with("LoadConstant #str["));
    assert!(out[6].starts_with("LoadConstant #str["));
}

#[test]
fn var_declaration_stores_into_local() {
    let script = compile("var x = 5; return x;").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 1",
        "LoadUndefined",
        "Store 2",
        "LoadSmi 5",
        "Store 0",
        "Load 0",
        "PopContext 1",
        "Return",
        "PopContext 1",
        "Load 2",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn var_without_init_stores_undefined() {
    let script = compile("var x; return x;").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 1",
        "LoadUndefined",
        "Store 2",
        "LoadUndefined",
        "Store 0",
        "Load 0",
        "PopContext 1",
        "Return",
        "PopContext 1",
        "Load 2",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn let_has_tdz_hole_check() {
    let script = compile("let x; x;").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 1",
        "LoadUndefined",
        "Store 2",
        "Load 0",
        "ThrowReferenceErrorIfHole",
        "Store 2",
        "PopContext 1",
        "Load 2",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn if_else_jumps() {
    let script = compile("if (1) 2; else 3;").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "LoadSmi 1",
        "JumpIfFalsy 21",
        "LoadSmi 2",
        "Store 1",
        "Jump 25",
        "LoadSmi 3",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn while_loop_back_edge() {
    let script = compile("while (0) { 1; }").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "LoadSmi 0",
        "JumpIfFalsy 21",
        "LoadSmi 1",
        "Store 1",
        "JumpLoop 7",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn for_loop_with_update_and_condition() {
    let script = compile("for (var i = 0; i < 3; i++) { 1; }").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 1",
        "LoadUndefined",
        "Store 2",
        "LoadSmi 0",
        "Store 0",
        "Load 0",
        "Store 3",
        "LoadSmi 3",
        "Store 4",
        "Load 3",
        "LessThan 4",
        "JumpIfFalsy 56",
        "LoadSmi 1",
        "Store 2",
        "Load 0",
        "Store 3",
        "LoadSmi 1",
        "Store 4",
        "LoadZero",
        "Store 5",
        "Load 3",
        "Sub 5",
        "Add 4",
        "Store 0",
        "Load 3",
        "JumpLoop 11",
        "PopContext 1",
        "Load 2",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn plain_call_receiver_is_undefined() {
    let script = compile("f(6, 7);").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "LoadGlobal 1 0",
        "Store 5",
        "LoadUndefined",
        "Store 2",
        "LoadSmi 6",
        "Store 3",
        "LoadSmi 7",
        "Store 4",
        "CallNoFeedback 5 2 3",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn method_call_passes_receiver() {
    let script = compile("o.m(6);").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "LoadGlobal 1 0",
        "Store 2",
        "LoadNamedProperty 2 2 2",
        "Store 4",
        "LoadSmi 6",
        "Store 3",
        "CallNoFeedback 4 2 2",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn new_construct_with_args() {
    let script = compile("new C(1);").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "LoadGlobal 1 0",
        "Store 3",
        "LoadSmi 1",
        "Store 2",
        "Construct 3 2 1",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn property_load_and_store() {
    let script = compile("o.x = 1; o.x;").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "LoadGlobal 1 0",
        "Store 2",
        "LoadSmi 1",
        "StoreNamedProperty 2 2 2",
        "Store 1",
        "LoadGlobal 3 4",
        "Store 2",
        "LoadNamedProperty 2 4 6",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn keyed_load_and_store() {
    let script = compile("a[k] = 1; a[k];").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "LoadGlobal 1 0",
        "Store 2",
        "LoadGlobal 2 2",
        "Store 3",
        "LoadSmi 1",
        "StoreKeyedProperty 2 3 4",
        "Store 1",
        "LoadGlobal 3 6",
        "Store 2",
        "LoadGlobal 4 8",
        "LoadKeyedProperty 2 10",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn array_literal_skips_holes() {
    let script = compile("[1, , 2];").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "CreateEmptyArrayLiteral",
        "Store 2",
        "LoadSmi 0",
        "Store 3",
        "LoadSmi 1",
        "StoreKeyedPropertyNoShadow 2 3 0",
        "LoadSmi 2",
        "Store 3",
        "LoadSmi 2",
        "StoreKeyedPropertyNoShadow 2 3 2",
        "Load 2",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn object_literal_shadow_stores() {
    let script = compile("({x: 1, y: 2});").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "CreateEmptyObjectLiteral",
        "Store 2",
        "LoadSmi 1",
        "StoreNamedProperty 2 1 0",
        "LoadSmi 2",
        "StoreNamedProperty 2 2 2",
        "Load 2",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn logical_and_short_circuits() {
    let script = compile("a && b;").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "LoadGlobal 1 0",
        "Store 2",
        "JumpIfFalsy 23",
        "LoadGlobal 2 2",
        "Jump 25",
        "Load 2",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn conditional_expression() {
    let script = compile("c ? 1 : 2;").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "LoadGlobal 1 0",
        "JumpIfFalsy 20",
        "LoadSmi 1",
        "Jump 22",
        "LoadSmi 2",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn throw_and_catch_binds_param() {
    let script = compile("try { throw 42; } catch (e) { return e; }").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 1",
        "LoadUndefined",
        "Store 2",
        "LoadContext",
        "Store 3",
        "LoadSmi 42",
        "Throw",
        "Jump 26",
        "PopContext 3",
        "Store 0",
        "Load 0",
        "PopContext 1",
        "Return",
        "PopContext 1",
        "Load 2",
        "Return",
    ];
    assert_eq!(out, expect);
    let handlers = script.handlers(script_fn(&script));
    assert_eq!(handlers.len(), 1);
    assert_eq!(handlers[0].try_start, 10);
    assert_eq!(handlers[0].try_end, 13);
    assert_eq!(handlers[0].handler_pc, 17);
}

#[test]
fn function_declaration_creates_closure_and_stores() {
    let script = compile("function f() { return 1; } f();").unwrap();
    let out = body(&script);
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 1",
        "LoadUndefined",
        "Store 2",
        "CreateClosure #fn[1]",
        "Store 0",
        "Load 0",
        "Store 4",
        "LoadUndefined",
        "Store 3",
        "CallNoFeedback 4 3 1",
        "Store 2",
        "PopContext 1",
        "Load 2",
        "Return",
    ];
    assert_eq!(out, expect);
    // the nested function has its own prologue/epilogue
    let inner = script.function(FunctionId(1));
    let inner_code = script.code(inner);
    let mut pc = 0;
    let mut inner_ops = Vec::new();
    while pc < inner_code.len() {
        let (op, _, next) = bytecode::decode(inner_code, pc);
        inner_ops.push(format!("{op:?}"));
        pc = next;
    }
    assert_eq!(
        inner_ops,
        [
            "CreateFunctionContext",
            "PushContext",
            "LoadSmi",
            "PopContext",
            "Return",
            "PopContext",
            "LoadUndefined",
            "Return",
        ]
    );
}

#[test]
fn parameters_are_negative_registers() {
    let script = compile("function f(a, b) { return a + b; }").unwrap();
    let inner = script.function(FunctionId(1));
    let inner_code = script.code(inner);
    let mut pc = 0;
    let mut inner_ops = Vec::new();
    while pc < inner_code.len() {
        let (op, ops, next) = bytecode::decode(inner_code, pc);
        let mut s = format!("{op:?}");
        for (i, kind) in op.operands().iter().enumerate() {
            if let Operand::Register = kind {
                s.push_str(&format!(" {}", ops.reg(i)));
            }
        }
        inner_ops.push(s);
        pc = next;
    }
    // param a = reg -2, param b = reg -3
    assert!(inner_ops.contains(&"Load -2".to_string()));
    assert!(inner_ops.contains(&"Load -3".to_string()));
    assert!(inner_ops.iter().any(|s| s.starts_with("Add")));
}

#[test]
fn closures_capture_via_context_slots() {
    let script = compile("var x = 1; function f() { return x; }").unwrap();
    // x is captured by f: lives in the script's context (slot 0)
    let inner = script.function(FunctionId(1));
    let inner_code = script.code(inner);
    let mut found = false;
    let mut pc = 0;
    while pc < inner_code.len() {
        let (op, ops, next) = bytecode::decode(inner_code, pc);
        if op == Opcode::LoadContextSlot {
            assert_eq!(ops.idx(0), 0, "captured slot");
            assert_eq!(ops.uimm(1), 1, "one function hop");
            found = true;
        }
        pc = next;
    }
    assert!(found, "closure must read the captured slot");
    let outer = disasm(&script);
    assert!(outer.contains(&"CreateFunctionContext 0".to_string()));
    assert!(outer.contains(&"StoreContextSlot 0 0".to_string()));
}

#[test]
fn captured_block_var_is_in_function_context() {
    let script = compile("{ let x = 1; function f() { return x; } }").unwrap();
    let outer = disasm(&script);
    assert!(outer.contains(&"CreateFunctionContext 0".to_string()));
    assert!(outer.contains(&"StoreContextSlot 0 0".to_string()));
}

#[test]
fn unsupported_features_report_errors() {
    let err = compile("1n + 1n;").unwrap_err();
    assert_eq!(err.feature, "BigInt literals");

    let err = compile("try {} finally {}").unwrap_err();
    assert_eq!(err.feature, "finally blocks");

    let err = compile("a ?? b;").unwrap_err();
    assert_eq!(err.feature, "nullish coalescing");
}

#[test]
fn switch_compiles_to_strict_equal_chain() {
    let script =
        compile("switch (x) { case 1: 10; break; case 2: 20; break; default: 30; }").unwrap();
    let out = body(&script);
    // discriminant evaluated once into a temp, strict-equal chain, bodies
    // in order with break jumps to the end
    let expect = [
        "CreateFunctionContext 0",
        "PushContext 0",
        "LoadUndefined",
        "Store 1",
        "LoadGlobal 1 0",
        "Store 2",
        "LoadSmi 1",
        "Store 3",
        "Load 2",
        "EqualStrict 3",
        "JumpIfTruthy 40",
        "LoadSmi 2",
        "Store 3",
        "Load 2",
        "EqualStrict 3",
        "JumpIfTruthy 48",
        "Jump 56",
        "LoadSmi 10",
        "Store 1",
        "Jump 60",
        "LoadSmi 20",
        "Store 1",
        "Jump 60",
        "LoadSmi 30",
        "Store 1",
        "PopContext 0",
        "Load 1",
        "Return",
    ];
    assert_eq!(out, expect);
}

#[test]
fn switch_without_default_skips_body() {
    let script = compile("switch (x) { case 1: 1; }").unwrap();
    let out = body(&script);
    // no default: both jumps land past the (empty-fallthrough) body
    assert_eq!(out[10], "JumpIfTruthy 28");
    assert_eq!(
        out[11], "Jump 32",
        "no default: fallthrough jumps past the body"
    );
    assert_eq!(out[12], "LoadSmi 1");
    assert!(out.contains(&"PopContext 0".to_string()));
}
