use bytecode::{Constant, Opcode, Operand, SourceMode, decode};
use js_compiler::compile_js;

fn show_const(c: &Constant) -> String {
    match c {
        Constant::String(b) => format!("\"{}\"", String::from_utf8_lossy(b)),
        Constant::Float(f) => format!("{f}"),
        Constant::Smi(v) => format!("{v}"),
        Constant::Callable(id) => format!("<fn#{}>", id.0),
        Constant::ContextNames(n) => format!("<names:{}>", n.len()),
        Constant::ObjectPrototype => "<Object.prototype>".into(),
        Constant::FunctionPrototype => "<Function.prototype>".into(),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: disasm <file.js>");
    let filter = args.next();
    let src = std::fs::read_to_string(&path).expect("read");
    let prog = compile_js(&src, SourceMode::Script).expect("compile");
    for id in prog.function_ids() {
        let f = prog.function(id);
        let name = f
            .name
            .as_deref()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_else(|| "<anonymous>".into());
        if let Some(pat) = &filter
            && !name.contains(pat.as_str())
        {
            continue;
        }
        println!(
            "=== fn#{} {}  regs={} arity={} fb={} constants={} ===",
            id.0,
            name,
            f.register_count,
            f.arity,
            f.feedback_count,
            f.constants.len()
        );
        let mut pc = 0usize;
        while pc < f.code.len() {
            let (op, ops, next) = decode(&f.code, pc);
            let kinds = op.operands();
            let mut rendered = Vec::new();
            for (i, kind) in kinds.iter().enumerate() {
                let raw = match kind {
                    Operand::Register => ops.reg(i),
                    Operand::RegisterListStart => ops.reg_list(i),
                    Operand::RegisterCount => ops.reg_count(i) as i32,
                    Operand::Immediate => ops.imm(i),
                    Operand::UImmediate => ops.uimm(i) as i32,
                    Operand::Index => ops.idx(i) as i32,
                };
                rendered.push(format!("{raw}"));
            }
            println!("  @{pc:>4}: {:<24} {}", format!("{op:?}"), rendered.join(", "));
            pc = next;
        }
        if !f.constants.is_empty() {
            println!("  constants:");
            for (i, c) in f.constants.iter().enumerate() {
                println!("    [{i}] {}", show_const(c));
            }
        }
    }
}
