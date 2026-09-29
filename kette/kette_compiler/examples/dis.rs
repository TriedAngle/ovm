//! Dump compiled bytecode: `cargo run -q -p kette_compiler --example dis -- '1 + 2'`
use bytecode::SourceMode;

fn main() {
    let src = std::env::args().nth(1).unwrap_or_else(|| "10 - 3".into());
    let program = kette_compiler::compile_kette(&src, SourceMode::Script).expect("compile");
    for (fid, f) in program.functions().enumerate() {
        println!("== fn {fid}");
        let code = program.code(f);
        let mut pc = 0;
        while pc < code.len() {
            let (op, ops, next) = bytecode::decode(code, pc);
            let mut line = format!("  {pc:4}: {op:?}");
            for (i, kind) in op.operands().iter().enumerate() {
                let v = match kind {
                    bytecode::Operand::Register => format!("r{}", ops.reg(i)),
                    bytecode::Operand::RegisterListStart => format!("r{}", ops.reg_list(i)),
                    bytecode::Operand::RegisterCount => ops.reg_count(i).to_string(),
                    bytecode::Operand::Immediate => ops.imm(i).to_string(),
                    bytecode::Operand::UImmediate => ops.uimm(i).to_string(),
                    bytecode::Operand::Index => format!("#{}", ops.idx(i)),
                };
                line.push_str(&format!(" {v}"));
            }
            println!("{line}");
            pc = next;
        }
    }
}
