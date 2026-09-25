//! Disassemble a script: parse → compile → print per-function bytecode.

use bytecode::Constant;
use bytecode::decode;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_bytecode <script.js>");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let program = match js_compiler::compile_js(&src, bytecode::SourceMode::Script) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("compile error: {e}");
            std::process::exit(1);
        }
    };
    for (i, f) in program.functions().enumerate() {
        println!(
            "=== function {i} kind={:?} name={:?} params={} regs={} ===",
            f.kind,
            program.name(f).map(String::from_utf8_lossy),
            f.arity,
            f.register_count
        );
        let code = program.code(f);
        let mut pc = 0usize;
        while pc < code.len() {
            let (op, ops, next) = decode(code, pc);
            print!("  {pc:4}: {op:?}");
            for (i, kind) in op.operands().iter().enumerate() {
                use bytecode::Operand;
                match *kind {
                    Operand::RegisterCount => print!(" {}", ops.reg_count(i)),
                    Operand::RegisterListStart => print!(" {}", ops.reg_list(i)),
                    Operand::Register => print!(" r{}", ops.reg(i)),
                    Operand::Immediate => print!(" #{}", ops.imm(i)),
                    Operand::Index => print!(" idx{}", ops.idx(i)),
                    Operand::UImmediate => print!(" {}", ops.uimm(i)),
                }
            }
            // annotate constant-pool indices
            let idx_ops: Vec<usize> = op
                .operands()
                .iter()
                .enumerate()
                .filter(|(_, k)| matches!(k, bytecode::Operand::Index))
                .map(|(i, _)| ops.idx(i))
                .collect();
            for idx in idx_ops {
                if let Some(c) = program.constants(f).get(idx) {
                    match c {
                        Constant::String(s) => {
                            print!("  ; {:?} = {:?}", idx, String::from_utf8_lossy(s))
                        }
                        Constant::Callable(fid) => print!("  ; {idx} = fn{fid:?}"),
                        other => print!("  ; {idx} = {:?}", other),
                    }
                }
            }
            println!();
            pc = next;
        }
    }
}
