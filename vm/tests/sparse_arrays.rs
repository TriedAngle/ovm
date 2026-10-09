// Sparse (dictionary-elements) array tests: normalization heuristics,
// hole semantics through dictionaries, accessor entries, denormalize
// round trips, and the memory blowup the dictionary exists to fix.

use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{DefaultInterpreter, JSRuntime, JavascriptCompiler, VM};

fn eval_ok(src: &str) {
    let vm = VM::new::<MarkSweep, DefaultInterpreter>(MarkSweepConfig::default())
        .expect("vm")
        .add::<JSRuntime>()
        .expect("js runtime");
    let mut thread = vm.attach();
    let result = thread.eval::<JavascriptCompiler>(src);
    match result {
        Ok(v) => {
            let heap = &*thread.heap();
            assert_ne!(
                v,
                heap.known().exception.as_tagged(heap).raw(),
                "script threw"
            );
        }
        Err(e) => panic!("script failed: {e}"),
    }
}

#[test]
fn far_store_does_not_materialize_the_range() {
    eval_ok(
        r#"
        var a = [];
        a[5000000] = 42;
        if (a[5000000] !== 42) throw "far read";
        if (a.length !== 5000001) throw "length";
        if (a[0] !== undefined) throw "hole";
        if (a[4999999] !== undefined) throw "hole 2";
        if (a[5000002] !== undefined) throw "hole 3";
        1
        "#,
    );
}

#[test]
fn sparse_roundtrip_and_delete() {
    eval_ok(
        r#"
        var a = [];
        a[100000] = 1;
        a[200000] = 2;
        a[300000] = 3;
        if (a[100000] + a[200000] + a[300000] !== 6) throw "sum";
        delete a[200000];
        if (a[200000] !== undefined) throw "deleted";
        if (a[300000] !== 3) throw "neighbor";
        delete a[100000];
        delete a[300000];
        if (a.length !== 300001) throw "length survives deletes";
        if (a[300000] !== undefined) throw "gone";
        1
        "#,
    );
}

#[test]
fn overwrite_keeps_entry() {
    eval_ok(
        r#"
        var a = [];
        a[100000] = 1;
        a[100000] = 2;
        a[100000] = 3;
        if (a[100000] !== 3) throw "overwrite";
        if (a.length !== 100001) throw "length";
        1
        "#,
    );
}

#[test]
fn huge_length_does_not_normalize_contents() {
    eval_ok(
        r#"
        var a = [1, 2, 3];
        a.length = 100000000;
        if (a[0] !== 1 || a[2] !== 3) throw "dense start";
        if (a[99999999] !== undefined) throw "hole";
        a[99999999] = 9;
        if (a[99999999] !== 9) throw "far store";
        1
        "#,
    );
}

#[test]
fn accessor_entry_in_sparse_array() {
    eval_ok(
        r#"
        var a = [];
        a[100000] = 1;
        var t = 0;
        Object.defineProperty(a, "200000", {
            set: function (v) { t = v; },
            get: function () { return 55; },
            configurable: true
        });
        if (a[200000] !== 55) throw "getter";
        a[200000] = 7;
        if (t !== 7) throw "setter not called";
        if (a[100000] !== 1) throw "data neighbor";
        var d = Object.getOwnPropertyDescriptor(a, "200000");
        if (!d.get || !d.set) throw "descriptor";
        // defineProperty defaults unspecified attributes to false
        if (d.enumerable !== false || d.configurable !== true) throw "descriptor flags";
        delete a[200000];
        if (a[200000] !== undefined) throw "deleted accessor";
        1
        "#,
    );
}

#[test]
fn readonly_entry_blocks_assignment() {
    eval_ok(
        r#"
        var a = [];
        a[100000] = 1;
        Object.defineProperty(a, "5", { value: 11, writable: false, configurable: true });
        if (a[5] !== 11) throw "read";
        a[5] = 99;
        if (a[5] !== 11) throw "readonly ignored";
        var d = Object.getOwnPropertyDescriptor(a, "5");
        if (d.writable !== false) throw "descriptor writable";
        1
        "#,
    );
}

#[test]
fn prototype_chain_still_consulted_for_sparse_holes() {
    eval_ok(
        r#"
        var a = [];
        a[100000] = 1;
        if (a[7] !== undefined) throw "pre";
        Array.prototype[7] = 42;
        if (a[7] !== 42) throw "post";
        var b = [];
        b[100000] = 1;
        if (b[7] !== 42) throw "fresh sparse array";
        delete Array.prototype[7];
        if (a[7] !== undefined) throw "gone";
        1
        "#,
    );
}

#[test]
fn dense_small_gaps_stay_dense() {
    eval_ok(
        r#"
        var a = [1, 2, 3];
        a[10] = 4;
        if (a[0] !== 1 || a[3] !== undefined || a[10] !== 4) throw "holes";
        if (a.length !== 11) throw "length";
        1
        "#,
    );
}

#[test]
fn densify_back_after_low_stores() {
    eval_ok(
        r#"
        var a = [];
        a[100000] = 1;
        // now fill the low range heavily: the dictionary stops paying
        for (var i = 0; i < 200; i++) { a[i] = i; }
        var sum = 0;
        for (var i = 0; i < 200; i++) { sum += a[i]; }
        if (sum !== 19900) throw "sum " + sum;
        if (a[100000] !== 1) throw "far entry survived";
        1
        "#,
    );
}

#[test]
fn enumeration_is_ascending() {
    eval_ok(
        r#"
        var a = [];
        a[300] = 3;
        a[100] = 1;
        a[200] = 2;
        var seen = [];
        for (var k in a) { seen.push(k); }
        if (seen.length !== 3) throw "count";
        if (seen[0] !== "100" || seen[1] !== "200" || seen[2] !== "300") throw "order " + seen;
        1
        "#,
    );
}

#[test]
fn sparse_builtins() {
    eval_ok(
        r#"
        var a = [];
        a[2] = 30;
        a[0] = 10;
        if (a.join(",") !== "10,,30") throw "join " + a.join(",");
        var s = 0;
        for (var i = 0; i < a.length; i++) { s += a[i] | 0; }
        if (s !== 40) throw "sum";
        var last = a.pop();
        if (last !== 30) throw "pop " + last;
        if (a[2] !== undefined) throw "popped hole";
        a.push(99);
        if (a[2] !== 99) throw "push";
        if (a.length !== 3) throw "push length";
        var b = a.slice(0, 3);
        if (b[0] !== 10 || b[1] !== undefined || b[2] !== 99) throw "slice";
        a.sort();
        if (a[0] !== 10 || a[1] !== 99 || a[2] !== undefined) throw "sort " + a.join(",");
        1
        "#,
    );
}

#[test]
fn shrink_length_removes_sparse_entries() {
    eval_ok(
        r#"
        var a = [];
        a[500000] = 1;
        a.length = 100;
        if (a[500000] !== undefined) throw "entry removed";
        if (a.length !== 100) throw "length";
        1
        "#,
    );
}

#[test]
fn delete_length_is_rejected_by_the_descriptor_row() {
    eval_ok(
        r#"
        var dense = [1, 2, 3];
        if (delete dense.length !== false) throw "dense delete";
        if (dense.length !== 3) throw "dense length";

        var sparse = [];
        sparse[100000] = 1;
        if (delete sparse.length !== false) throw "sparse delete";
        if (sparse.length !== 100001) throw "sparse length";
        if (sparse[100000] !== 1) throw "sparse entry";

        var proto_changed = [1];
        Object.setPrototypeOf(proto_changed, null);
        if (delete proto_changed.length !== false) throw "null-proto delete";
        if (proto_changed.length !== 1) throw "null-proto length";

        function strictDelete() {
            "use strict";
            var arr = [1];
            delete arr.length;
        }
        var threw = false;
        try { strictDelete(); } catch (e) { threw = true; }
        if (!threw) throw "strict delete must throw";
        1
        "#,
    );
}
