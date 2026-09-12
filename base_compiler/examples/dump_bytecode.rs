//! Disassemble a script: parse → compile → print per-function bytecode.

use bytecode::{Opcode, decode};
use parser::Parser;

fn main() {
    let src = std::env::args()
        .nth(1)
        .expect("usage: dump_bytecode <script.js>");
    let mut p = Parser::new(parser::Utf8SliceStream::new(&src));
    p.parse_script().expect("parse");
    let ast = p.into_ast();
    let compiled = match base_compiler::compile_script(&ast) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("compile error: {e}");
            std::process::exit(1);
        }
    };
    for (i, f) in compiled.functions.iter().enumerate() {
        println!(
            "=== function {i} kind={:?} name={:?} params={} regs={} ===",
            f.kind, f.name, f.formal_parameter_count, f.register_count
        );
        let mut pc = 0usize;
        while pc < f.bytecode.len() {
            let (op, ops, next) = decode(&f.bytecode, pc);
            print!("  {pc:4}: {op:?}");
            for (i, kind) in ops.kinds().iter().enumerate() {
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
                if let Some(c) = f.constants.get(idx) {
                    match c {
                        base_compiler::Constant::String(s) => {
                            print!("  ; {:?} = {:?}", idx, String::from_utf8_lossy(s))
                        }
                        base_compiler::Constant::Callable(fid) => print!("  ; {idx} = fn{fid:?}"),
                        other => print!("  ; {idx} = {:?}", other),
                    }
                }
            }
            println!();
            pc = next;
        }
    }
}
