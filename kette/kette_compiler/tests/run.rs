//! End-to-end tests: compile Kette source to the shared IR, materialize
//! it in a real VM, and run it.

use bytecode::SourceMode;
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{MatchLoopInterpreter, Smi, VM};

fn run(source: &str) -> i64 {
    let vm = VM::new::<MarkSweep, MatchLoopInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    let value = thread
        .run_source(source, kette_compiler::compile_kette, SourceMode::Script)
        .expect("script runs");
    Smi::decode(value).expect("script result is a smi").value()
}

fn runs_clean(source: &str) -> bool {
    let vm = VM::new::<MarkSweep, MatchLoopInterpreter>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    let result = thread.run_source(source, kette_compiler::compile_kette, SourceMode::Script);
    result.is_ok() && thread.take_pending_exception().is_none()
}

// -- the unified object ------------------------------------------------------

#[test]
fn object_slot_and_send() {
    assert_eq!(
        run("let obj = { x: 10\n read: { self.x } }\nobj.read()"),
        10
    );
}

#[test]
fn zero_arg_block_runs_its_body() {
    assert_eq!(run("let g = { 41 }\ng()"), 41);
}

#[test]
fn slots_params_and_code_in_one_object() {
    assert_eq!(
        run("let counter = {
    count: 0
    ||
        self.count = self.count + 1
        self.count
}
counter()
counter()"),
        2
    );
}

#[test]
fn a_call_binds_the_callee_as_self() {
    assert_eq!(run("let f = { n: 7\n || self.n }\nf()"), 7);
}

#[test]
fn nested_closure_captures() {
    assert_eq!(run("let mk = { |a| { |b| a } }\nlet f = mk(1)\nf(2)"), 1);
}

// -- operators -----------------------------------------------------------------

#[test]
fn arithmetic_operators_are_instructions() {
    assert_eq!(run("1 + 2 * 3"), 7);
    assert_eq!(run("10 - 3"), 7);
    assert_eq!(run("3 - 10"), -7);
    assert_eq!(run("5 % 3"), 2);
    assert_eq!(run("10 / 2"), 5);
}

#[test]
fn comparisons() {
    assert_eq!(run("if 1 < 2 { 1 } else { 0 }"), 1);
    assert_eq!(run("if 2 < 1 { 1 } else { 0 }"), 0);
    assert_eq!(run("if 3 >= 3 { 1 } else { 0 }"), 1);
    assert_eq!(run("if 1 == 2 { 1 } else { 0 }"), 0);
    assert_eq!(run("if 1 != 2 { 1 } else { 0 }"), 1);
}

#[test]
fn and_or_are_lazy() {
    assert_eq!(run("true && 5"), 5);
    assert_eq!(run("false || 7"), 7);
    assert_eq!(
        run("let z = 0\nlet f = { z = 1\n true }\nfalse && f()\nz"),
        0,
        "the rhs block must not run"
    );
    assert_eq!(
        run("let z = 0\nlet f = { z = 1\n false }\ntrue || f()\nz"),
        0
    );
}

#[test]
fn unary_operators() {
    assert_eq!(run("let x = 3\n-x"), -3);
    assert_eq!(run("if !(1 == 2) { 1 } else { 0 }"), 1);
}

// -- if --------------------------------------------------------------------------

#[test]
fn if_without_else_yields_the_branch_or_null() {
    assert_eq!(run("if true { 9 }"), 9);
    assert_eq!(run("if false { 9 }\n1"), 1);
}

#[test]
fn else_if_chain() {
    assert_eq!(
        run("let n = 2\nif n < 1 { 0 } else { if n < 2 { 1 } else { 2 } }"),
        2
    );
}

#[test]
fn branch_lets_belong_to_the_enclosing_scope() {
    assert_eq!(
        run("let y = if true { let t = 5\n t * 2 } else { 1 }\ny"),
        10
    );
}

// -- objects, slots, elements ------------------------------------------------------

#[test]
fn element_literal_and_store() {
    assert_eq!(run("let a = [10, 20]\na[1] = 5\na[1]"), 5);
}

#[test]
fn element_write_never_grows() {
    let vm = VM::new::<MarkSweep, MatchLoopInterpreter>(MarkSweepConfig::default()).unwrap();
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
fn object_with_element_slots_is_an_array() {
    assert_eq!(
        run("let t = { label: 2\n [0]: 10\n [1]: 20 }\nt[1] + t.label"),
        22
    );
}

#[test]
fn explicit_return_returns_from_block() {
    assert_eq!(run("let f = { |x| return x }\nf(3)"), 3);
    assert_eq!(run("let f = { |x| return x\n 99 }\nf(3)"), 3);
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

#[test]
fn methods_receive_the_send_receiver() {
    assert_eq!(
        run("let v = { x: 1\n add: { |o| self.x + o.x } }
let w = { x: 2 }
v.add(w)"),
        3
    );
}

#[test]
fn closures_capture_branch_lets() {
    assert_eq!(
        run("let f = { |x| if x { let y = 10\n { y } } else { { 2 } } }
f(true)()"),
        10
    );
}

#[test]
fn the_language_example_composes() {
    assert!(runs_clean(
        "let mk = { |n| { |x| x + n } }
let inc = mk(1)
let dec = mk(0 - 1)
inc(inc(dec(10)))"
    ));
    assert_eq!(
        run("let mk = { |n| { |x| x + n } }\nlet inc = mk(1)\ninc(inc(41))"),
        43
    );
}
