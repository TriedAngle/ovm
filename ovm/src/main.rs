use bytecode::{Opcode, Operand, Scale};
use dummy_heap::{DummyHeap, DummyHeapConfig};
use ovm::VM;
use ovm::natives::NativeIndex;
use vm::{Array, Float, LocalHeap, Smi};

fn main() {
    let vm = VM::<DummyHeap>::new(DummyHeapConfig::default()).expect("failed to create heap");
    let mut ctx = vm.attach();

    ctx.handle_scope(|ctx, scope| {
        let zero = Smi::new_unchecked(0).encode();
        let _handle = ctx
            .heap()
            .allocate_handle::<Array>(&[zero, zero, zero], &scope);

        let interned = ctx.intern(&scope, "hello, ovm");
        let again = ctx.intern(&scope, "hello, ovm");
        ctx.heap().no_gc(|nogc, _| {
            let s = interned.heap_ref(nogc);
            println!(
                "interned: {:?} (hash {}, deduped: {})",
                s.string().as_str().expect("utf8"),
                s.string().hash(),
                interned.value().to_bits() == again.value().to_bits()
            );
        });
    });
    println!(
        "heap initialized: {} bytes ({} used)",
        vm.heap().capacity(),
        vm.heap().used()
    );

    let receiver = Smi::new_unchecked(0).encode();
    let result = vm.native(NativeIndex::SMI_ADD)(
        &mut ctx,
        &[
            receiver,
            Smi::new_unchecked(6).encode(),
            Smi::new_unchecked(7).encode(),
        ],
    )
    .expect("smi_add failed");
    println!("smi_add(6, 7) = {}", Smi::decode(result).unwrap().value());

    let fa = ctx.heap().allocate::<Float>(1.5).erase();
    let fb = ctx.heap().allocate::<Float>(2.25).erase();
    let result = vm.native(NativeIndex::FLOAT_ADD)(&mut ctx, &[receiver, fa, fb])
        .expect("float_add failed");
    let out = ctx.heap().no_gc(|nogc, heap| {
        nogc.get_as::<Float>(result, heap.known().float_map)
            .expect("float_add returned a float")
            .value
            .get()
    });
    println!("float_add(1.5, 2.25) = {}", out);

    // return 6 + 7
    let mut program = Vec::new();
    emit(&mut program, Opcode::LoadSmi, &[6]);
    emit(&mut program, Opcode::Store, &[0]);
    emit(&mut program, Opcode::LoadSmi, &[7]);
    emit(&mut program, Opcode::Store, &[1]);
    emit(&mut program, Opcode::Add, &[0, 1]);
    emit(&mut program, Opcode::Return, &[]);

    match run(&program) {
        Ok(acc) => println!("result: {acc}"),
        Err(err) => {
            eprintln!("runtime error: {err}");
            std::process::exit(1);
        }
    }
}

fn emit(code: &mut Vec<u8>, op: Opcode, operands: &[u32]) {
    let kinds = op.operands();
    assert_eq!(kinds.len(), operands.len(), "operand count mismatch");

    let wide = kinds
        .iter()
        .zip(operands)
        .any(|(kind, value)| *kind != Operand::RegisterCount && *value > u8::MAX as u32);
    let scale = if wide { Scale::Byte2 } else { Scale::Byte1 };
    if wide {
        code.push(Opcode::Wide as u8);
    }
    code.push(op as u8);

    for (kind, value) in kinds.iter().zip(operands) {
        let size = kind.size_in_stream(scale);
        assert!(
            (*value as u64) < (1u64 << (size * 8)),
            "operand {value} does not fit in {size} byte(s)"
        );
        code.extend_from_slice(&value.to_le_bytes()[..size]);
    }
}

pub fn run(code: &[u8]) -> Result<i64, String> {
    let mut registers = [0i64; 256];
    let mut acc = 0i64;
    let mut pc = 0usize;

    loop {
        let mut op = read_opcode(code, &mut pc)?;
        let mut scale = Scale::Byte1;
        if op == Opcode::Wide {
            scale = Scale::Byte2;
            op = read_opcode(code, &mut pc)?;
        }

        let mut operands = [0u32; 4];
        let mut sizes = [0usize; 4];
        for (i, kind) in op.operands().iter().enumerate() {
            let size = kind.size_in_stream(scale);
            let bytes = code
                .get(pc..pc + size)
                .ok_or("truncated instruction stream")?;
            let mut buf = [0u8; 4];
            buf[..size].copy_from_slice(bytes);
            operands[i] = u32::from_le_bytes(buf);
            sizes[i] = size;
            pc += size;
        }

        match op {
            Opcode::LoadSmi => {
                acc = match sizes[0] {
                    1 => operands[0] as u8 as i8 as i64,
                    _ => operands[0] as u16 as i16 as i64,
                };
            }
            Opcode::Load => acc = read_reg(&registers, operands[0])?,
            Opcode::Store => *reg_mut(&mut registers, operands[0])? = acc,
            Opcode::Move => {
                let value = read_reg(&registers, operands[1])?;
                *reg_mut(&mut registers, operands[0])? = value;
            }
            Opcode::Add => {
                let lhs = read_reg(&registers, operands[0])?;
                let rhs = read_reg(&registers, operands[1])?;
                acc = lhs.checked_add(rhs).ok_or("arithmetic overflow")?;
            }
            Opcode::Return => return Ok(acc),
            other => return Err(format!("unsupported opcode {other:?}")),
        }
    }
}

fn read_opcode(code: &[u8], pc: &mut usize) -> Result<Opcode, String> {
    let byte = *code.get(*pc).ok_or("program counter out of bounds")?;
    *pc += 1;
    Opcode::from_byte(byte).ok_or_else(|| format!("invalid opcode {byte:#04x}"))
}

fn read_reg(registers: &[i64], idx: u32) -> Result<i64, String> {
    registers
        .get(idx as usize)
        .copied()
        .ok_or_else(|| format!("register r{idx} out of bounds"))
}

fn reg_mut(registers: &mut [i64], idx: u32) -> Result<&mut i64, String> {
    registers
        .get_mut(idx as usize)
        .ok_or_else(|| format!("register r{idx} out of bounds"))
}
