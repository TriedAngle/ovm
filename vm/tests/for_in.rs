//! `for-in` (ES 14.7.5): enumeration order, prototype-chain walking,
//! mutation during enumeration, head forms, and the per-iteration
//! assignment target.

use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{ScriptError, Smi, VM, VMString, Value};

fn run(src: &str) -> Result<Value, ScriptError> {
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    thread.run_script(src)
}

fn run_smi(src: &str) -> i64 {
    Smi::decode(run(src).unwrap()).unwrap().value()
}

fn run_str(src: &str) -> String {
    run(src).unwrap();
    let vm = VM::with_builtins::<MarkSweep>(MarkSweepConfig::default()).unwrap();
    let mut thread = vm.attach();
    // re-run under a live heap to read the string back
    let result = thread.run_script(src).unwrap();
    thread.heap().no_gc(|nogc| {
        let s = result.get_as::<VMString>(nogc).expect("string result");
        String::from_utf8(s.as_slice(nogc).to_vec()).unwrap()
    })
}

// -- order ------------------------------------------------------------------

#[test]
fn own_property_order() {
    assert_eq!(
        run_str(
            "var o = {p2: 1, p1: 2};
             o.z = 3;
             var keys = '';
             for (var k in o) { keys = keys + k; }
             keys;"
        ),
        "p2p1z"
    );
}

#[test]
fn integer_keys_ascending_before_strings() {
    assert_eq!(
        run_str(
            "var o = {};
             o[2] = 1; o[0] = 1; o[1] = 1;
             o.p = 1;
             var keys = '';
             for (var k in o) { keys = keys + k + ','; }
             keys;"
        ),
        "0,1,2,p,"
    );
}

#[test]
fn integer_keys_in_object_literals_sort() {
    // numeric literal keys land in descriptors in insertion order; the
    // snapshot reorders canonical indices ascending
    assert_eq!(
        run_str(
            "var o = { 2: 1, 0: 1, 1: 1, x: 1 };
             var keys = '';
             for (var k in o) { keys = keys + k + ','; }
             keys;"
        ),
        "0,1,2,x,"
    );
}

#[test]
fn noncanonical_numeric_strings_stay_insertional() {
    assert_eq!(
        run_str(
            "var o = {};
             o['01'] = 1; o['1'] = 1; o['0'] = 1;
             var keys = '';
             for (var k in o) { keys = keys + k + ','; }
             keys;"
        ),
        "0,1,01,"
    );
}

#[test]
fn deleted_and_readded_key_moves_to_the_end() {
    assert_eq!(
        run_str(
            "var o = {a: 1, b: 2};
             delete o.a;
             o.a = 3;
             var keys = '';
             for (var k in o) { keys = keys + k; }
             keys;"
        ),
        "ba"
    );
}

// -- prototype chain ----------------------------------------------------------

#[test]
fn walks_the_prototype_chain() {
    assert_eq!(
        run_str(
            "function Base() {}
             Base.prototype.inherited = 1;
             var o = new Base();
             o.own = 2;
             var keys = '';
             for (var k in o) { keys = keys + k; }
             keys;"
        ),
        "owninherited"
    );
}

#[test]
fn own_keys_shadow_proto_keys() {
    assert_eq!(
        run_str(
            "function Base() {}
             Base.prototype.x = 1;
             Base.prototype.y = 2;
             var o = new Base();
             o.x = 3;
             var keys = '';
             for (var k in o) { keys = keys + k; }
             keys;"
        ),
        "xy"
    );
}

#[test]
fn nonenumerable_own_still_shadows_proto() {
    // shadowing ignores [[Enumerable]]: a non-enumerable own property
    // blocks the proto's key without yielding it (ES 14.7.5.9)
    assert_eq!(
        run_smi(
            "function Base() {}
             Base.prototype.x = 1;
             var o = new Base();
             Object.defineProperty(o, 'x', {value: 2, enumerable: false});
             var seen = 0;
             for (var k in o) { seen = seen + 1; }
             seen * 10 + (('x' in o) ? 1 : 0);"
        ),
        1
    );
}

#[test]
fn nonenumerable_proto_keys_are_not_yielded() {
    assert_eq!(
        run_str(
            "var base = {};
             Object.defineProperty(base, 'hidden', {value: 1, enumerable: false});
             base.shown = 2;
             var keys = '';
             for (var k in base) { keys = keys + k; }
             keys;"
        ),
        "shown"
    );
}

#[test]
fn chain_terminates_at_object_prototype() {
    // Object.prototype's own built-ins are non-enumerable; only
    // user-added enumerable props there show up at the chain's end
    assert_eq!(
        run_str(
            "var keys = '';
             for (var k in Object.prototype) { keys = keys + k; }
             keys;"
        ),
        ""
    );
}

#[test]
fn proto_added_props_are_visible_when_their_level_is_reached() {
    // the chain is walked lazily: mutations of the prototype between
    // iterations are observable (ES 14.7.5.9)
    assert_eq!(
        run_str(
            "function Base() {}
             var base = Base.prototype;
             var o = new Base();
             o.a = 1;
             var keys = '';
             for (var k in o) {
               keys = keys + k;
               base.late = 1; // added before the walk reaches the proto
             }
             keys;"
        ),
        "alate"
    );
}

// -- mutation during enumeration ---------------------------------------------

#[test]
fn deleted_before_visited_is_skipped() {
    assert_eq!(
        run_str(
            "var o = {a: 1, b: 2, c: 3};
             var keys = '';
             for (var k in o) {
               if (k === 'a') { delete o.b; }
               keys = keys + k;
             }
             keys;"
        ),
        "ac"
    );
}

#[test]
fn added_during_enumeration_not_revisited() {
    assert_eq!(
        run_str(
            "var o = {a: 1};
             var keys = '';
             for (var k in o) {
               keys = keys + k;
               o.new = 1; // not in the level-0 snapshot
             }
             keys;"
        ),
        "a"
    );
}

#[test]
fn each_key_yielded_at_most_once() {
    assert_eq!(
        run_smi(
            "var o = {a: 1, b: 2};
             var n = 0;
             for (var k in o) { n = n + 1; }
             n;"
        ),
        2
    );
}

// -- head forms ---------------------------------------------------------------

#[test]
fn null_and_undefined_run_zero_iterations() {
    assert_eq!(
        run_smi(
            "var n = 0;
             for (var k in null) { n = n + 1; }
             for (k in undefined) { n = n + 1; }
             n;"
        ),
        0
    );
}

#[test]
fn let_and_const_heads() {
    assert_eq!(
        run_str(
            "var keys = '';
             for (let k in {a: 1}) { keys = keys + k; }
             for (const c in {b: 1}) { keys = keys + c; }
             keys;"
        ),
        "ab"
    );
}

#[test]
fn member_assignment_targets() {
    assert_eq!(
        run_str(
            "var target = {slot: ''};
             for (target.slot in {a: 1, b: 2}) {}
             var arr = [];
             for (arr[0] in {c: 3}) {}
             target.slot + arr[0];"
        ),
        "bc"
    );
}

#[test]
fn assignment_target_reevaluates_each_iteration() {
    // ForIn/OfBodyEvaluation: the lhs reference "may be evaluated
    // repeatedly" — o[i++] advances once per yielded key
    assert_eq!(
        run_smi(
            "var o = {a: 1, b: 2};
             var receiver = {s: 0};
             var i = 0;
             for (o.k in receiver) { i = i + 1; }
             // no-op enumeration below uses the same effect
             var j = 0;
             var box = {};
             for (box[j++] in {x: 1, y: 1, z: 1}) {}
             j;"
        ),
        3
    );
}

#[test]
fn pattern_heads_destructure_the_key() {
    assert_eq!(
        run_str(
            "var out = '';
             for (var {length: n} in {ab: 1, c: 2}) { out = out + n; }
             out;"
        ),
        "21"
    );
}

#[test]
fn strings_enumerate_indices() {
    assert_eq!(
        run_str(
            "var keys = '';
             for (var k in 'ab') { keys = keys + k; }
             keys;"
        ),
        "01"
    );
}

#[test]
fn arrays_enumerate_elements_then_names() {
    assert_eq!(
        run_str(
            "var a = [1, 2, 3];
             delete a[1];
             a.extra = 9;
             var keys = '';
             for (var k in a) { keys = keys + k + ','; }
             keys;"
        ),
        "0,2,extra,"
    );
}

// -- control flow ---------------------------------------------------------------

#[test]
fn break_and_continue() {
    assert_eq!(
        run_str(
            "var o = {a: 1, b: 2, c: 3, d: 4};
             var keys = '';
             for (var k in o) {
               if (k === 'a') { continue; }
               if (k === 'd') { break; }
               keys = keys + k;
             }
             keys;"
        ),
        "bc"
    );
}

#[test]
fn labelled_break() {
    // the label must lead the script: ovm's scanner mis-detects labels
    // that follow another statement (pre-existing, tracked separately)
    assert_eq!(
        run_str(
            "outer: for (var k in {a: 1, b: 2}) {
               for (var j in {x: 1, y: 1}) {
                 break outer;
               }
             }
             'broke';"
        ),
        "broke"
    );
}

#[test]
fn for_in_over_numeric_and_string_keys_agree() {
    // the canonicalization fix: o[2] and o['2'] are one property
    assert_eq!(
        run_smi(
            "var o = {};
             o[2] = 'a';
             o['3'] = 'b';
             var n = 0;
             for (var k in o) { n = n + 1; }
             (o['2'] === 'a') * 100 + (o[3] === 'b') * 10 + n;"
        ),
        112
    );
}

#[test]
fn lexical_declaration_body_is_a_parse_error() {
    let err = run("for (var k in {}) let x = 1;").unwrap_err();
    assert!(matches!(err, ScriptError::Parse(_)), "got {err:?}");
    let err = run("for (var k in {}) const x = 1;").unwrap_err();
    assert!(matches!(err, ScriptError::Parse(_)), "got {err:?}");
    let err = run("while (0) let x = 1;").unwrap_err();
    assert!(matches!(err, ScriptError::Parse(_)), "got {err:?}");
}

#[test]
fn invalid_assignment_targets_are_parse_errors() {
    let err = run("for (f() in {}) {}").unwrap_err();
    assert!(matches!(err, ScriptError::Parse(_)), "got {err:?}");
    let err = run("for (1 in {}) {}").unwrap_err();
    assert!(matches!(err, ScriptError::Parse(_)), "got {err:?}");
}

#[test]
fn c_style_for_still_parses_after_disambiguation() {
    assert_eq!(
        run_smi("var n = 0; for (var i = 0; i < 3; i = i + 1) { n = n + 1; } n;"),
        3
    );
    assert_eq!(
        run_smi("var x = 0; for (x = 1; x < 2;) { x = x + 1; } x;"),
        2
    );
    assert_eq!(run_smi("for (;;) { break; } 7;"), 7);
}

#[test]
fn in_operator_still_works_inside_c_style_for() {
    // `in` inside the C-style head stays a binary operator
    assert_eq!(
        run_smi(
            "var o = {a: 1}; var n = 0; for (var i = ('a' in o) ? 0 : 5; i < 2; i = i + 1) { n = n + 1; } n;"
        ),
        2
    );
}

// -- string index loads and primitive chain walks (compliance follow-ups) --

#[test]
fn string_index_loads() {
    assert_eq!(run_str("var s = 'ab'; s[1];"), "b");
    assert_eq!(run_str("'ab'[0];"), "a");
    // out of range: undefined
    assert_eq!(run_smi("var s = 'ab'; (s[5] === undefined) ? 1 : 0;"), 1);
    // surrogate pairs: indices are UTF-16 code units
    assert_eq!(
        run_smi(
            "var s = '\\u{1D400}x';
             (s.length === 3) * 100 + (typeof s[0] === 'string') * 10 + (s[2] === 'x') * 1;"
        ),
        111
    );
}

#[test]
fn primitives_walk_their_constructor_prototypes() {
    assert_eq!(
        run_str(
            "Number.prototype.extra = 1;
             var out = '';
             for (var k in 5) { out = out + k; }
             out;"
        ),
        "extra"
    );
    assert_eq!(
        run_str(
            "Boolean.prototype.be = 1;
             var out = '';
             for (var k in true) { out = out + k; }
             out;"
        ),
        "be"
    );
    // clean prototypes enumerate nothing
    assert_eq!(
        run_smi(
            "var n = 0;
             for (var k in 7) { n = n + 1; }
             for (var k in false) { n = n + 1; }
             n;"
        ),
        0
    );
}
