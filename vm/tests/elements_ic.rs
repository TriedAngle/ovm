// Element IC regression tests: prototype validity cells guard the
// hole/OOB/hole-store fast paths instead of a global latch. Each test
// warms an IC first, so a stale handler would produce the wrong answer.

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
            // uncaught throws surface as the exception sentinel value
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
fn hole_load_sees_array_prototype_element_after_warmup() {
    eval_ok(
        r#"
        var a = [1, , 3];
        var sink;
        for (var i = 0; i < 200; i++) { sink = a[1]; }
        if (a[1] !== undefined) throw "pre";
        Array.prototype[1] = 42;
        if (a[1] !== 42) throw "post";
        var b = [1, , 3];
        if (b[1] !== 42) throw "fresh array";
        1
        "#,
    );
}

#[test]
fn oob_load_sees_array_prototype_element_after_warmup() {
    eval_ok(
        r#"
        var a = [1, , 3];
        var sink;
        for (var i = 0; i < 200; i++) { sink = a[7]; }
        if (a[7] !== undefined) throw "pre";
        Array.prototype[7] = 42;
        if (a[7] !== 42) throw "post";
        1
        "#,
    );
}

#[test]
fn hole_load_sees_object_prototype_integer_property() {
    eval_ok(
        r#"
        var a = [1, , 3];
        var sink;
        for (var i = 0; i < 200; i++) { sink = a[1]; }
        Object.prototype[1] = 9;
        if (a[1] !== 9) throw "post";
        1
        "#,
    );
}

#[test]
fn string_oob_sees_string_prototype_element() {
    eval_ok(
        r#"
        var s = "abc";
        var sink;
        for (var i = 0; i < 200; i++) { sink = s[5]; }
        if (s[5] !== undefined) throw "pre";
        String.prototype[5] = "x";
        if (s[5] !== "x") throw "post";
        1
        "#,
    );
}

#[test]
fn deleted_element_disappears_from_chain() {
    eval_ok(
        r#"
        var a = [1, , 3];
        var sink;
        for (var i = 0; i < 200; i++) { sink = a[7]; }
        Array.prototype[7] = 42;
        if (a[7] !== 42) throw "present";
        delete Array.prototype[7];
        if (a[7] !== undefined) throw "gone";
        1
        "#,
    );
}

#[test]
fn hole_store_fills_on_clean_chain() {
    eval_ok(
        r#"
        for (var i = 0; i < 200; i++) {
            var c = [1, , 3];
            c[1] = i;
            if (c[1] !== i) throw "fill";
        }
        var b = [1, , 3];
        b[1] = 5;
        if (b[1] !== 5) throw "fill after warm";
        if (b.length !== 3) throw "length";
        1
        "#,
    );
}

#[test]
fn hole_store_calls_prototype_setter_after_warmup() {
    eval_ok(
        r#"
        for (var i = 0; i < 200; i++) {
            var c = [1, , 3];
            c[1] = i;
        }
        var t = 0;
        Object.defineProperty(Object.prototype, "1", {
            set: function (v) { t = v; },
            get: function () { return 77; },
            configurable: true
        });
        var b = [1, , 3];
        b[1] = 5;
        if (t !== 5) throw "setter not called";
        if (b[1] !== 77) throw "getter not consulted";
        1
        "#,
    );
}

#[test]
fn append_store_calls_prototype_setter_after_warmup() {
    eval_ok(
        r#"
        for (var i = 0; i < 200; i++) {
            var c = [1];
            c[c.length] = 2;
        }
        var t = 0;
        Object.defineProperty(Array.prototype, "1", {
            set: function (v) { t = v; },
            get: function () { return 77; },
            configurable: true
        });
        var d = [1];
        d[1] = 9;
        if (t !== 9) throw "setter not called";
        if (d.length !== 1) throw "append should not extend";
        1
        "#,
    );
}

#[test]
fn own_accessor_at_index_calls_setter() {
    eval_ok(
        r#"
        var t = 0;
        var q = [1];
        Object.defineProperty(q, "1", {
            set: function (v) { t = v; },
            get: function () { return 77; },
            configurable: true
        });
        q[1] = 9;
        if (t !== 9) throw "setter not called";
        if (q[1] !== 77) throw "getter not consulted";
        // per ArrayDefineOwnProperty, defining index 1 extends length
        if (q.length !== 2) throw "define extends length";
        1
        "#,
    );
}

#[test]
fn holey_oob_loads_stay_on_fast_path() {
    let src = r#"
        var a = [1, , 3];
        var sink = 0;
        for (var i = 0; i < 300000; i++) {
            sink += a[7] === undefined ? 1 : 0;
        }
        if (sink !== 300000) throw "bad";
        1
    "#;
    let start = std::time::Instant::now();
    eval_ok(src);
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 5000,
        "holey OOB load path regressed: {elapsed:?}"
    );
}
