//! Golden bytecode tests: compile snippets and assert on the emitted
//! instruction stream (constants inlined, jump targets absolute).

use bytecode::{Constant, Function, FunctionId, Opcode, Operand, Program};

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
        Opcode::LoadGlobalFast | Opcode::StoreGlobalFast => i == 0,
        Opcode::LoadNamedProperty
        | Opcode::StoreNamedProperty
        | Opcode::StoreNamedPropertyNoShadow
        | Opcode::LoadNamedPropertyFast
        | Opcode::StoreNamedPropertyFast
        | Opcode::StoreNamedPropertyNoShadowFast
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
fn add_is_the_add_instruction() {
    let program = compile("1 + 2");
    assert!(contains(&program, "Add"));
    assert!(!contains(&program, "#str[\"add\"]"));
}

#[test]
fn comparison_is_the_less_than_instruction() {
    let program = compile("1 < 2");
    assert!(contains(&program, "LessThan"));
}

#[test]
fn if_branches_compile_inline() {
    let program = compile("if 1 < 2 { 7 } else { 9 }");
    assert_eq!(program.len(), 1, "branches are inline, script only");
    assert!(contains(&program, "JumpIfFalsy") || contains(&program, "CompareJump"));
    assert!(contains(&program, "LoadSmi 7"));
    assert!(contains(&program, "LoadSmi 9"));
}

#[test]
fn and_is_short_circuit_jumps() {
    let program = compile("a && b");
    assert!(contains(&program, "JumpIfFalsy"));
    assert_eq!(program.len(), 1, "the rhs is not wrapped in a closure");
}

#[test]
fn a_body_object_becomes_a_closure() {
    let program = compile("let f = { |x| x }\nf(1)");
    assert_eq!(program.len(), 2, "script + one function");
    assert!(contains(&program, "CreateClosure"));
}

#[test]
fn slots_and_code_share_one_object() {
    // the counter: a closure with a named slot stored onto it
    let program = compile("{ count: 0\n|| self.count }");
    assert!(contains(&program, "CreateClosure"));
    assert!(contains(&program, "StoreNamedProperty"));
    assert!(contains(&program, "#str[\"count\"]"));
}

#[test]
fn slots_only_is_a_bare_object() {
    let program = compile("{ x: 1 }");
    assert!(contains(&program, "CreateBareObjectLiteral"));
    assert!(!contains(&program, "CreateClosure"));
}

#[test]
fn element_slots_create_an_array() {
    let program = compile("{ label: 1, [0]: 2 }");
    assert!(contains(&program, "CreateEmptyArrayLiteral"));
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
fn a_call_passes_the_callee_as_the_receiver() {
    // disassemble the call: the receiver window slot is the callee
    // register (moved, not loaded undefined)
    let program = compile("f(1)");
    let script = disasm(&program);
    let call = script
        .iter()
        .position(|l| l.contains("CallNoFeedback"))
        .expect("call");
    assert!(
        script[..call].iter().any(|l| l.contains("Move")),
        "the callee is moved into the receiver slot: {script:?}"
    );
}
