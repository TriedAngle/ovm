//! Feedback-vector plumbing: slot allocation in codegen, transport through
//! the IR, and materialization into a hole-filled `FeedbackVector`.

use bytecode::{Opcode, decode};
use ir::{FunctionId, Program, SourceMode};
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::VM;

fn compile(src: &str) -> Program {
    js_compiler::compile_js(src, SourceMode::Script).expect("compile")
}

fn bytecode_of(program: &Program, fid: FunctionId) -> Vec<u8> {
    program.code(program.function(fid)).to_vec()
}

/// Decode `code` and return the feedback operand of every property-access
/// instruction (all of them carry it as their last operand).
fn feedback_sites(code: &[u8]) -> Vec<(Opcode, usize)> {
    let mut sites = Vec::new();
    let mut pc = 0;
    while pc < code.len() {
        let (op, ops, next) = decode(code, pc);
        let slot = match op {
            Opcode::LoadNamedProperty
            | Opcode::LoadKeyedProperty
            | Opcode::StoreNamedProperty
            | Opcode::StoreNamedPropertyNoShadow
            | Opcode::StoreKeyedProperty
            | Opcode::StoreKeyedPropertyNoShadow => Some(ops.idx(op.operands().len() - 1)),
            _ => None,
        };
        if let Some(slot) = slot {
            sites.push((op, slot));
        }
        pc = next;
    }
    sites
}

#[test]
fn sites_get_distinct_slot_pairs() {
    // three property sites in `f`: named load, named store, keyed load
    let program = compile("function f(o, k) { o.x; o.x = 1; return o[k]; }");
    let fid = program
        .function_ids()
        .find(|&id| program.function(id).feedback_count > 0)
        .expect("inner function carries feedback slots");
    assert_eq!(program.function(fid).feedback_count, 6);

    let sites = feedback_sites(&bytecode_of(&program, fid));
    assert_eq!(sites.len(), 3);
    let slots: Vec<usize> = sites.iter().map(|(_, s)| *s).collect();
    assert_eq!(slots, vec![0, 2, 4], "sites consume slot pairs in order");
}

#[test]
fn functions_without_property_access_have_no_feedback() {
    let program = compile("function f(a, b) { return a + b; }");
    assert!(
        program.functions().all(|f| f.feedback_count == 0),
        "no property sites, no slots"
    );
}

#[test]
fn materialized_vector_is_hole_filled() {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    // named load + method-call load + named store sites live in the script
    let program = compile("var o = {x: 1}; function g(p) { return p.x; } o.x = g(o);");
    let mut thread = vm.attach();
    thread.handle_scope(|thread, scope| {
        let closure =
            vm::materialize::materialize_script(thread, &scope, &program).expect("materialize");
        // script closure: slots[0] = CallableInfoObject
        let heap = thread.heap();
        let info = closure
            .as_tagged(heap)
            .slot(heap, 0)
            .get(heap)
            .get_as::<vm::CallableInfoObject>()
            .expect("script closure info");
        let vector = info
            .as_ref()
            .feedback(heap)
            .expect("script has property sites");
        let len = vector.as_ref().len();
        assert_eq!(len % 2, 0, "slots come in [state, handler] pairs");
        assert!(len > 0);
        let hole = heap.known().the_hole.as_tagged(heap).raw();
        for i in 0..len {
            assert_eq!(
                vector.as_ref().inner(i),
                hole,
                "slot {i} must start as the uninitialized hole"
            );
        }
    });
}

#[test]
fn running_a_script_with_feedback_still_works() {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    let result = thread
        .run_script("var o = {x: 1}; function g(p) { return p.x; } o.x = g(o) + o.x;")
        .expect("run");
    let heap = thread.heap();
    let smi = vm::Smi::decode(result).expect("smi result");
    let _ = smi;
    let _ = heap;
}
