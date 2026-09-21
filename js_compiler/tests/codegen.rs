//! Frontend tests at the IR level: compile snippets through `compile_js`
//! and assert on function metadata and the emitted instruction stream.
//! (Runtime behavior is covered by the `vm` test suite.)

use bytecode::{Opcode, Operand};
use ir::{FrontendErrorKind, FunctionId, Program, SourceMode};
use js_compiler::compile_js;

fn compile(src: &str, mode: SourceMode) -> Program {
    compile_js(src, mode).expect("compile")
}

fn script_fn(program: &Program) -> &ir::Function {
    program.function(FunctionId::SCRIPT)
}

fn ops(program: &Program, f: &ir::Function) -> Vec<Opcode> {
    let code = program.code(f);
    let mut out = Vec::new();
    let mut pc = 0;
    while pc < code.len() {
        let (op, _, next) = bytecode::decode(code, pc);
        out.push(op);
        pc = next;
    }
    out
}

fn contains(program: &Program, f: &ir::Function, op: Opcode) -> bool {
    ops(program, f).contains(&op)
}

#[test]
fn function_metadata() {
    let p = compile(
        "function add(a, b) { return a + b; }
         function rest(a, b = 1, ...c) {}",
        SourceMode::Script,
    );
    let add = p.function(FunctionId(1));
    assert_eq!(add.arity, 2);
    assert_eq!(add.length, 2);
    assert_eq!(p.name(add), Some(&b"add"[..]));
    let rest = p.function(FunctionId(2));
    assert_eq!(rest.arity, 3, "patterns and rest count one slot each");
    assert_eq!(rest.length, 1, "length stops at the first non-simple param");
}

#[test]
fn script_carries_a_completion_register() {
    let p = compile("1 + 2;", SourceMode::Script);
    let f = script_fn(&p);
    // locals(0) + ctx save + completion = 3 fixed slots below the temps
    assert!(f.register_count >= 3);
    assert!(contains(&p, f, Opcode::Store));
}

#[test]
fn unsupported_constructs_are_compile_errors() {
    for (src, feature) in [
        ("1n + 1n;", "BigInt literals"),
        ("try {} finally {}", "finally blocks"),
        ("a ?? b;", "nullish coalescing"),
    ] {
        let err = compile_js(src, SourceMode::Script).expect_err(src);
        assert_eq!(err.kind, FrontendErrorKind::Compile, "{src}");
        assert!(err.message.contains(feature), "{src} -> {err}");
    }
}

#[test]
fn syntax_errors_are_syntax_errors() {
    let err = compile_js("function {", SourceMode::Script).expect_err("must fail");
    assert_eq!(err.kind, FrontendErrorKind::Syntax);
}

#[test]
fn script_mode_top_level_vars_are_locals() {
    let p = compile("var x = 1; x = 2;", SourceMode::Script);
    let f = script_fn(&p);
    assert!(
        !contains(&p, f, Opcode::StoreGlobal),
        "script decls must not touch the global object"
    );
}

#[test]
fn repl_mode_top_level_vars_are_global_properties() {
    let p = compile("var x = 1; let y = 2; function f() {}", SourceMode::Repl);
    let f = script_fn(&p);
    let code = program_code(&p, f);
    let n_globals = code
        .iter()
        .filter(|(op, _)| *op == Opcode::StoreGlobal)
        .count();
    assert!(n_globals >= 3, "var, let and function decls all go global");
}

#[test]
fn eval_mode_free_names_are_dynamic_lookups() {
    let p = compile("undeclared_name;", SourceMode::Eval);
    let f = script_fn(&p);
    assert!(contains(&p, f, Opcode::CallRuntime));
}

#[test]
fn eval_mode_same_code_resolves_normally() {
    // in Script mode the same free name is a plain global load, not a
    // runtime chain walk
    let p = compile("undeclared_name;", SourceMode::Script);
    let f = script_fn(&p);
    assert!(contains(&p, f, Opcode::LoadGlobal));
}

#[test]
fn strict_directive_propagates() {
    let p = compile("\"use strict\"; function f() {}", SourceMode::Script);
    assert!(script_fn(&p).strict);
    assert!(p.function(FunctionId(1)).strict);
    let p = compile("function f() {}", SourceMode::Script);
    assert!(!script_fn(&p).strict);
}

#[test]
fn hoisted_function_declarations_are_initialized_in_the_prologue() {
    // the closure creation must precede any body code
    let p = compile("var r = f(); function f() { return 1; }", SourceMode::Script);
    let f = script_fn(&p);
    let first_create = position(&p, f, Opcode::CreateClosure);
    let first_call = position(&p, f, Opcode::CallNoFeedback);
    assert!(first_create < first_call, "closure created before the call");
}

#[test]
fn class_declarations_compile_with_synthesized_members() {
    let p = compile(
        "class A { #x = 1; get v() { return this.#x; } }
         new A().v;",
        SourceMode::Script,
    );
    // script + ctor + field initializer + getter = 4
    assert_eq!(p.len(), 4);
}

fn position(program: &Program, f: &ir::Function, op: Opcode) -> usize {
    let code = program.code(f);
    let mut pc = 0;
    while pc < code.len() {
        let (found, _, next) = bytecode::decode(code, pc);
        if found == op {
            return pc;
        }
        pc = next;
    }
    panic!("opcode {op:?} not found");
}

/// (opcode, formatted operands) pairs for targeted assertions.
fn program_code<'p>(program: &'p Program, f: &'p ir::Function) -> Vec<(Opcode, Vec<u32>)> {
    let code = program.code(f);
    let mut out = Vec::new();
    let mut pc = 0;
    while pc < code.len() {
        let (op, ops, next) = bytecode::decode(code, pc);
        let vals = op
            .operands()
            .iter()
            .enumerate()
            .map(|(i, kind)| match kind {
                Operand::Register => ops.reg(i) as u32,
                Operand::RegisterListStart => ops.reg_list(i) as u32,
                Operand::RegisterCount => ops.reg_count(i) as u32,
                Operand::Immediate => ops.imm(i) as u32,
                Operand::UImmediate => ops.uimm(i),
                Operand::Index => ops.idx(i) as u32,
            })
            .collect();
        out.push((op, vals));
        pc = next;
    }
    out
}
