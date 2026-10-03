use bytecode::{Opcode, Operand, SourceMode, try_decode};
use js_compiler::compile_js;

const SAMPLE: &str = r#"
const counter = { value: 0 };

function bump(obj, n) {
  for (let i = 0; i < n; i++) {
    obj.value = obj.value + 1;
  }
  return obj.value;
}

bump(counter, 3);
"#;

fn main() {
    let source = match std::env::args().nth(1) {
        Some(path) => std::fs::read_to_string(path).unwrap(),
        None => SAMPLE.to_string(),
    };
    let program = compile_js(&source, SourceMode::Script).expect("compile");
    for id in program.function_ids() {
        let f = program.function(id);
        let name = f
            .name
            .as_deref()
            .map(|n| String::from_utf8_lossy(n).into_owned())
            .unwrap_or_else(|| "<entry>".into());
        println!(
            "=== {name} (regs={}, arity={}, length={}, kind={:?}) ===",
            f.register_count, f.arity, f.length, f.kind
        );
        if !f.constants.is_empty() {
            for (i, c) in f.constants.iter().enumerate() {
                println!("  const[{i}] = {c:?}");
            }
        }
        let mut pc = 0;
        while pc < f.code.len() {
            let (op, ops, next) = try_decode(&f.code, pc).unwrap();
            let jump_operand = match op {
                Opcode::Jump | Opcode::JumpLoop => Some(0),
                Opcode::JumpIfTruthy | Opcode::JumpIfFalsy | Opcode::JumpIfNotUndefined => Some(0),
                Opcode::CompareJump => Some(2),
                _ => None,
            };
            let mut text = format!("{pc:4}  {op:?}");
            for (i, kind) in op.operands().iter().enumerate() {
                match kind {
                    Operand::Register | Operand::RegisterListStart => {
                        text.push_str(&format!(" r{}", ops.reg(i)))
                    }
                    Operand::RegisterCount => text.push_str(&format!(" #{}", ops.reg_count(i))),
                    Operand::Immediate => {
                        text.push_str(&format!(" {}", ops.imm(i)));
                        if jump_operand == Some(i) {
                            text.push_str(&format!(
                                " -> {}",
                                bytecode::jump_target(pc, ops.imm(i))
                            ));
                        }
                    }
                    Operand::UImmediate => text.push_str(&format!(" {}", ops.uimm(i))),
                    Operand::Index => text.push_str(&format!(" [{}]", ops.idx(i))),
                }
            }
            print!("{text}");
            if op.reads_acc() && op.writes_acc() {
                print!("   ; acc -> acc");
            } else if op.reads_acc() {
                print!("   ; acc in");
            } else if op.writes_acc() {
                print!("   ; -> acc");
            }
            println!();
            pc = next;
        }
        println!();
    }
}
