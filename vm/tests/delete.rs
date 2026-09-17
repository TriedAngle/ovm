//! `delete` (ES 13.5.1): OrdinaryDelete over the hidden-class object
//! model, exotic receivers, references vs values, and language-mode
//! splits.

use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{DenseString, ScriptError, Smi, Value};
use vm::{Thread, VM};

fn run(src: &str) -> Result<Value, ScriptError> {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    thread.run_script(src)
}

fn run_smi(src: &str) -> i64 {
    Smi::decode(run(src).unwrap()).unwrap().value()
}

fn run_bool(src: &str) -> bool {
    let (result, mut thread) = run_value(src);
    thread.heap().no_gc(|heap| {
        if result == heap.known().true_object.as_tagged(heap).raw() {
            true
        } else if result == heap.known().false_object.as_tagged(heap).raw() {
            false
        } else {
            panic!("expected boolean result, got {result:?}");
        }
    })
}

fn run_value(src: &str) -> (Value, Thread) {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    let result = thread.run_script(src).unwrap();
    (result, thread)
}

fn run_str(src: &str) -> String {
    let (result, mut thread) = run_value(src);
    thread.heap().no_gc(|heap| {
        let s = unsafe { result.assume_valid(heap) }
            .get_as::<DenseString>()
            .expect("string result");
        s.to_rust_string(heap)
    })
}

// -- ordinary objects -------------------------------------------------------

#[test]
fn removes_own_property() {
    assert_eq!(
        run_smi(
            "var o = {a: 1, b: 2};
             delete o.a;
             ('a' in o) + (o.a === undefined) + o.b + ('b' in o);"
        ),
        4
    );
}

#[test]
fn absent_and_present_return_values() {
    assert!(run_bool("var o = {}; delete o.nope;"));
    assert!(run_bool("var o = {p: 1}; delete o.p;"));
}

#[test]
fn slot_compaction_keeps_survivors() {
    assert_eq!(
        run_smi(
            "var o = {a: 1, b: 2, c: 3};
             delete o.b;
             o.a * 100 + o.c;"
        ),
        103
    );
}

#[test]
fn same_shaped_objects_delete_independently() {
    assert_eq!(
        run_smi(
            "var a = {x: 1, y: 2, z: 3};
             var b = {x: 4, y: 5, z: 6};
             delete a.y;
             delete b.x;
             a.x * 1000 + a.z * 100 + b.y * 10 + b.z;"
        ),
        1356
    );
}

#[test]
fn readd_lands_at_the_end() {
    assert_eq!(
        run_smi(
            "var o = {a: 1, b: 2};
             delete o.a;
             o.a = 3;
             var names = Object.getOwnPropertyNames(o);
             (names.length === 2) * 100
               + (names[0] === 'b' ? 10 : 0)
               + (names[1] === 'a' ? 1 : 0);"
        ),
        111
    );
}

#[test]
fn computed_key_and_coercion() {
    assert_eq!(
        run_smi(
            "var o = {p: 1};
             var log = { n: 0 };
             var k = { toString() { log.n = log.n + 1; return 'p'; } };
             var r = delete o[k];
             r + 10 * (log.n + ('p' in o ? 1 : 0));"
        ),
        11
    );
}

#[test]
fn delete_does_not_run_getters() {
    assert_eq!(
        run_smi(
            "var ran = 0;
             var o = { get p() { ran = 1; return 42; } };
             var r = delete o.p;
             r * 10 + ran;"
        ),
        10
    );
}

#[test]
fn accessors_are_removable() {
    assert_eq!(
        run_smi(
            "var ran = 0;
             var o = {
               set p(v) { ran = 1; },
               q: 7
             };
             var r = delete o.p;
             o.p = 9; // setter gone: plain store, does not run
             r * 100 + ran * 10 + o.q;"
        ),
        107
    );
}

#[test]
fn nonconfigurable_is_not_removed() {
    assert_eq!(
        run_smi(
            "var o = {};
             Object.defineProperty(o, 'x', {value: 1, configurable: false});
             var r = delete o.x;
             r * 10 + o.x;"
        ),
        1
    );
}

#[test]
fn symbol_keys_delete() {
    assert!(run_bool(
        "var s = Symbol();
         var o = {};
         o[s] = 1;
         delete o[s] && !(s in o);"
    ));
}

#[test]
fn property_introspection_reflects_the_delete() {
    assert_eq!(
        run_smi(
            "var o = {a: 1, b: 2};
             delete o.a;
             var names = Object.getOwnPropertyNames(o);
             var d = Object.getOwnPropertyDescriptor(o, 'a');
             names.length * 10 + (d === undefined ? 1 : 0) + (o.hasOwnProperty('b') ? 0 : 5);"
        ),
        11
    );
}

// -- references vs values ---------------------------------------------------

#[test]
fn sloppy_delete_of_var_returns_false_and_keeps_it() {
    assert_eq!(
        run_smi(
            "var x = 1;
             var r = delete x;
             r * 10 + x;"
        ),
        1
    );
}

#[test]
fn sloppy_delete_of_undeclared_returns_true() {
    assert!(run_bool("delete definitelyNotDeclaredAnywhere;"));
}

#[test]
fn sloppy_delete_of_lexical_returns_false() {
    assert_eq!(
        run_smi(
            "let l = 1;
             const c = 2;
             (delete l === false) + (delete c === false) + l + c;"
        ),
        5
    );
}

#[test]
fn sloppy_delete_of_undeclared_assignment_deletes_it() {
    assert_eq!(
        run_smi(
            "createdOnTheFly = 5;
             var r = delete createdOnTheFly;
             var gone = false;
             try { createdOnTheFly; } catch (e) { gone = e.name === 'ReferenceError'; }
             r * 10 + gone;"
        ),
        11
    );
}

#[test]
fn strict_delete_identifier_is_a_parse_error() {
    let err = run("\"use strict\"; delete x;").unwrap_err();
    assert!(matches!(err, ScriptError::Parse(_)), "got {err:?}");
}

#[test]
fn strict_delete_in_parens_is_still_a_parse_error() {
    let err = run("\"use strict\"; delete (((x)));").unwrap_err();
    assert!(matches!(err, ScriptError::Parse(_)), "got {err:?}");
}

#[test]
fn strict_delete_of_property_is_fine() {
    assert!(run_bool("\"use strict\"; var o = {p: 1}; delete o.p;"));
}

#[test]
fn non_reference_operands_yield_true() {
    assert!(run_bool("delete 1;"));
    // `undefined` is a non-configurable global property (ES 19.1.1), so
    // unlike other literals its delete observes declarative semantics
    assert!(!run_bool("delete undefined;"));
    assert!(run_bool("delete (function f() {});"));
    assert_eq!(
        run_smi("var ran = 0; function f() { ran = 1; } delete (f(), 0) + ran;"),
        2
    );
}

#[test]
fn sequence_delete_does_not_delete() {
    assert_eq!(
        run_smi(
            "var o = {p: 1};
             var r = delete (0, o.p);
             r * 10 + o.p;"
        ),
        11
    );
}

// -- strict failures throw --------------------------------------------------

#[test]
fn strict_delete_nonconfigurable_throws() {
    assert_eq!(
        run_str(
            "\"use strict\";
             var o = {};
             Object.defineProperty(o, 'x', {value: 1, configurable: false});
             try { delete o.x; return 'no'; }
             catch (e) { e.name; }"
        ),
        "TypeError"
    );
}

// -- arrays -----------------------------------------------------------------

#[test]
fn element_delete_punches_a_hole() {
    assert_eq!(
        run_smi(
            "var a = [1, 2, 3];
             var r = delete a[1];
             r * 1000
               + (a[1] === undefined ? 100 : 0)
               + (1 in a ? 50 : 0)
               + a.length * 10
               + a[0] + a[2];"
        ),
        1134
    );
}

#[test]
fn delete_past_length_is_a_noop_true() {
    assert!(run_bool("var a = [1]; delete a[5];"));
    assert!(run_bool("var a = [1]; a.length = 3; delete a[2];"));
}

#[test]
fn array_length_is_not_deletable() {
    assert_eq!(
        run_smi(
            "var a = [1, 2];
             (delete a.length === false) + a.length;"
        ),
        3
    );
}

#[test]
fn array_named_properties_delete() {
    assert_eq!(
        run_smi(
            "var a = [1];
             a.extra = 9;
             var r = delete a.extra;
             r * 10 + a.length + (a.extra === undefined ? 4 : 0);"
        ),
        15
    );
}

#[test]
fn plain_object_numeric_keys_delete() {
    assert_eq!(
        run_smi(
            "var o = {};
             o[2] = 'two';
             o[1] = 'one';
             var r = delete o[2];
             r * 10 + (o[2] === undefined ? 5 : 0) + (o[1] === 'one' ? 1 : 0);"
        ),
        16
    );
}

// -- primitives and exotics -------------------------------------------------

#[test]
fn string_length_and_indices_are_not_deletable() {
    assert_eq!(
        run_smi(
            "var s = 'ab';
             (delete s.length === false)
               + (delete s[0] === false)
               + (delete s[1] === false)
               + (delete s[2] === true)
               + (delete s.x === true);"
        ),
        5
    );
}

#[test]
fn noncanonical_string_indices_delete() {
    assert!(run_bool("var s = 'ab'; delete s['01'];"));
    assert!(run_bool("var s = 'ab'; delete s['-0'];"));
}

#[test]
fn other_primitives_delete_as_true() {
    assert!(run_bool("delete (5).anything;"));
    assert!(run_bool("delete true.x;"));
}

#[test]
fn nullish_base_throws_typeerror() {
    assert_eq!(
        run_str("try { delete null.x; return 'no'; } catch (e) { e.name; }"),
        "TypeError"
    );
    assert_eq!(
        run_str("try { delete undefined.x; return 'no'; } catch (e) { e.name; }"),
        "TypeError"
    );
}

#[test]
fn key_coercion_runs_before_base_check() {
    assert_eq!(
        run_smi(
            "var log = { n: 0 };
             var k = { toString() { log.n = log.n + 1; return 'x'; } };
             var threw = 0;
             try { delete null[k]; } catch (e) { threw = e.name === 'TypeError'; }
             log.n * 10 + threw;"
        ),
        11
    );
}

// -- super ------------------------------------------------------------------

#[test]
fn delete_super_property_is_referenceerror() {
    assert_eq!(
        run_str(
            "class B { }
             class C extends B {
               m() { try { delete super.x; return 'no'; } catch (e) { return e.name; } }
             }
             new C().m();"
        ),
        "ReferenceError"
    );
}

// -- descriptors after mutation ----------------------------------------------

#[test]
fn delete_then_store_reshapes_correctly() {
    assert_eq!(
        run_smi(
            "var o = {a: 1, b: 2};
             delete o.a;
             o.a = 10;
             o.b = 20;
             o.a + o.b + Object.getOwnPropertyNames(o).length;"
        ),
        32
    );
}
