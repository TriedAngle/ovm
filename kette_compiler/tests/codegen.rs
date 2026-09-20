//! Golden bytecode tests: compile snippets and assert on the emitted
//! instruction stream (constants inlined, jump targets absolute).

use bytecode::{Opcode, Operand};
use ir::{Constant, Function, FunctionId, Program};
use kette_compiler::CompileError;

fn compile(src: &str) -> Program {
    let mut parser = kette_parser::Parser::new(kette_parser::Utf8SliceStream::new(src));
    parser.parse_script().expect("parse");
    let mut ast = parser.into_ast();
    kette_compiler::compile_ast(&mut ast).expect("compile")
}

fn script_fn(program: &Program) -> &Function {
    program.function(FunctionId::SCRIPT)
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

fn constant_operand(op: Opcode, i: usize) -> bool {
    match op {
        Opcode::LoadConstant | Opcode::CreateClosure => i == 0,
        Opcode::LoadGlobal | Opcode::StoreGlobal => i == 0,
        Opcode::LoadNamedProperty
        | Opcode::StoreNamedProperty
        | Opcode::StoreNamedPropertyNoShadow
        | Opcode::AddParent => i == 1,
        _ => false,
    }
}

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
                Operand::Immediate => ops.imm(i) as i64,
                Operand::UImmediate => ops.uimm(i) as i64,
                Operand::Index => ops.idx(i) as i64,
            };
            let v = if constant_operand(op, i) {
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

fn contains(program: &Program, needle: &str) -> bool {
    disasm(program).iter().any(|line| line.contains(needle))
}

#[test]
fn unused_compile_error_surface() {
    // `while`/`for`/`match` are not lowered yet
    let mut parser = kette_parser::Parser::new(kette_parser::Utf8SliceStream::new("while c { 1 }"));
    parser.parse_script().unwrap();
    let mut ast = parser.into_ast();
    let err: CompileError = kette_compiler::compile_ast(&mut ast).unwrap_err();
    assert_eq!(err.feature, "while loops");
}

#[test]
fn add_becomes_an_add_send() {
    let program = compile("1 + 2");
    assert!(contains(&program, "LoadNamedProperty"));
    assert!(contains(&program, "#str[\"add\"]"));
    assert!(contains(&program, "CallNoFeedback"));
}

#[test]
fn if_becomes_an_ifelse_send_with_blocks() {
    let program = compile("if c { 1 } else { 2 }");
    assert!(contains(&program, "#str[\"ifElse\"]"));
    assert!(contains(&program, "#fn[1]"));
    assert!(contains(&program, "#fn[2]"));
    assert_eq!(program.len(), 3, "script + two branch blocks");
}

#[test]
fn and_wraps_the_rhs_in_a_block() {
    let program = compile("a && b");
    assert!(contains(&program, "#str[\"and\"]"));
    assert_eq!(program.len(), 2, "script + lazy RHS block");
}

#[test]
fn parent_slots_extend_the_prototype() {
    let program = compile("let P = 1\n{ parent*: P }");
    assert!(contains(&program, "CreateBareObjectLiteral"));
    assert!(contains(&program, "AddParent"));
    assert!(contains(&program, "#str[\"parent\"]"));
}

#[test]
fn element_assignment_is_in_place() {
    let program = compile("a[1] = 5");
    assert!(contains(&program, "StoreKeyedSlot"));
}

#[test]
fn try_catch_uses_a_handler_range() {
    let program = compile("try { 1 } catch e { 2 }");
    assert!(contains(&program, "Jump"));
    assert_eq!(program.len(), 3, "script + try body + catch handler");
    let script = script_fn(&program);
    assert_eq!(
        program.handlers(script).len(),
        1,
        "one handler entry covers the body call"
    );
}

#[test]
fn unary_not_is_a_send() {
    let program = compile("!a");
    assert!(contains(&program, "#str[\"not\"]"));
}
