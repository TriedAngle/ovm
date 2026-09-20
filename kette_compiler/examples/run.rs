//! Run Kette source: `cargo run -p kette_compiler --example run -- file.ktt`
//! (reads stdin when no path is given). Prints the script's value.

use ir::SourceMode;
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{DenseString, Float, Smi, Thread, VM, Value};

fn main() {
    let source = match std::env::args().nth(1) {
        Some(path) => std::fs::read_to_string(&path).expect("read the source file"),
        None => {
            use std::io::Read;
            let mut src = String::new();
            std::io::stdin()
                .read_to_string(&mut src)
                .expect("read stdin");
            src
        }
    };
    let vm = VM::new::<MarkSweep>(MarkSweepConfig::default()).expect("heap");
    let mut thread = vm.attach();
    match thread.run_source(&source, kette_compiler::compile_kette, SourceMode::Script) {
        Ok(value) => {
            if let Some(ex) = thread.take_pending_exception() {
                eprintln!("uncaught exception: {}", show_value(&mut thread, ex));
                std::process::exit(1);
            }
            println!("{}", show_value(&mut thread, value));
        }
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}

fn show_value(thread: &mut Thread, v: Value) -> String {
    if let Some(smi) = Smi::decode(v) {
        return smi.value().to_string();
    }
    let heap = &*thread.heap();
    let known = heap.known();
    if v == known.undefined.as_tagged(heap).raw() {
        return "undefined".into();
    }
    if v == known.null.as_tagged(heap).raw() {
        return "null".into();
    }
    if v == known.true_object.as_tagged(heap).raw() {
        return "true".into();
    }
    if v == known.false_object.as_tagged(heap).raw() {
        return "false".into();
    }
    if let Some(f) = unsafe { v.assume_valid(heap) }.get_as::<Float>() {
        return f.value.get().to_string();
    }
    if let Some(s) = unsafe { v.assume_valid(heap) }.get_as::<DenseString>() {
        return s.to_rust_string(heap);
    }
    format!("{v:?}")
}
