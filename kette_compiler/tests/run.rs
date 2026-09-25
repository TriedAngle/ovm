//! End-to-end tests: compile Kette source to the shared IR, materialize
//! it in a real VM, and run it.

use bytecode::SourceMode;
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{Smi, ThreadedInterpreter, VM};

fn run(source: &str) -> i64 {
    let vm = VM::new::<MarkSweep, ThreadedInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    let value = thread
        .run_source(source, kette_compiler::compile_kette, SourceMode::Script)
        .expect("script runs");
    Smi::decode(value).expect("script result is a smi").value()
}

#[test]
fn object_slot_and_send() {
    assert_eq!(
        run("let obj = { x: 10\n read: { self.x } }\nobj.read()"),
        10
    );
}

#[test]
fn nested_closure_captures() {
    assert_eq!(run("let mk = { |a| { |b| a } }\nlet f = mk(1)\nf(2)"), 1);
}

#[test]
fn element_literal_and_store() {
    assert_eq!(run("let a = [10, 20]\na[1] = 5\na[1]"), 5);
}

#[test]
fn element_write_never_grows() {
    let vm = VM::new::<MarkSweep, ThreadedInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    let result = thread.run_source(
        "let a = [10, 20]\na[2] = 5",
        kette_compiler::compile_kette,
        SourceMode::Script,
    );
    assert!(
        result.is_ok(),
        "the failure is a runtime throw, not a compile error"
    );
    assert!(
        thread.take_pending_exception().is_some(),
        "`a[2] = 5` past the end must throw"
    );
}

#[test]
fn try_catch_returns_the_body_value() {
    assert_eq!(run("try { 7 } catch e { 42 }"), 7);
}

#[test]
fn try_catch_catches_an_out_of_bounds_write() {
    assert_eq!(
        run("let a = [10, 20]\ntry { a[2] = 5\n 7 } catch e { 42 }"),
        42
    );
}

#[test]
fn try_catch_unwinds_through_callee_frames() {
    assert_eq!(
        run("let boom = { |a| a[2] = 5 }\ntry { boom([10, 20]) } catch e { 42 }"),
        42
    );
}

#[test]
fn explicit_return_returns_from_block() {
    assert_eq!(run("let f = { |x| return x }\nf(3)"), 3);
}

#[test]
fn and_is_lazy_and_a_send() {
    // `a && b` → `a.and({ b })`; the block runs when the receiver evaluates it
    assert_eq!(run("let gate = { and: { |b| b() } }\ngate && 5"), 5);
}

#[test]
fn binary_operators_are_sends() {
    // `one + two` → `one.add(two)`
    assert_eq!(
        run("let one = { add: { |o| o.x } }\nlet two = { x: 41 }\none + two"),
        41
    );
}

#[test]
fn if_desugars_to_ifelse() {
    assert_eq!(
        run("let box = { ifElse: { |t, f| t() } }\nif box { 7 } else { 9 }"),
        7
    );
}

#[test]
fn parent_slots_are_multi_parent_prototypes() {
    assert_eq!(run("let P = { x: 7 }\nlet C = { parent*: P }\nC.x"), 7);
}

#[test]
fn second_parent_in_priority_order() {
    assert_eq!(
        run("let P1 = { a: 1 }\nlet P2 = { b: 2 }\nlet C = { parent*: P1\n parent*: P2 }\nC.b"),
        2
    );
}

#[test]
fn assignment_writes_through_to_parent_holder() {
    assert_eq!(
        run("let P = { x: 1 }\nlet C = { parent*: P }\nC.x = 9\nP.x"),
        9
    );
}
