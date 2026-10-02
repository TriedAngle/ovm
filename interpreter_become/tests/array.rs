use interpreter_become::BecomeInterpreter;
use js_compiler::JavascriptCompiler;
use js_runtime::JSRuntime;
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm_core::VM;

fn run(label: &str, src: &str) {
    let vm = VM::new::<MarkSweep, BecomeInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<JSRuntime>()
        .unwrap();
    vm.arm_gc_stress();
    let mut thread = vm.attach();
    thread.eval::<JavascriptCompiler>(src).expect("runs");
    if let Some(ex) = thread.take_pending_exception() {
        panic!("{label}: pending exception: {ex:?}");
    }
}

const CHECK: &str = "function check(c, m) { if (!c) { throw new Error(m); } }\n";

/// Array `length` is a real accessor: stores truncate/extend and reject
/// invalid lengths.
#[test]
fn length_setter() {
    run(
        "length",
        &format!(
            "{CHECK}
            var a = [1,2,3];
            a.length = 1;
            check(a.length == 1, 'truncate');
            check(a[1] == undefined, 'dropped');
            check(Object.getOwnPropertyNames(a).join(',') == '0,length', 'keys');
            a.length = 4;
            check(a.length == 4, 'grow');
            var b = [];
            b[0] = 1; b[5] = 2;
            b.length = 1;
            check(b.length == 1, 'sparse truncate');
            var threw = false;
            try {{ a.length = -1; }} catch (e) {{ threw = true; }}
            check(threw, 'negative throws');
            threw = false;
            try {{ a.length = 1.5; }} catch (e) {{ threw = true; }}
            check(threw, 'fraction throws');
            a.length = '3';
            check(a.length == 3, 'coerce');"
        ),
    );
}

/// `slice` copies a range (holes preserved) and `sort` is stable and
/// in-place with `undefined`/holes last.
#[test]
fn slice_and_sort() {
    run(
        "slice",
        &format!(
            "{CHECK}
            check([1,2,3,4,5].slice().join(',') == '1,2,3,4,5', 'slice all');
            check([1,2,3,4,5].slice(1).join(',') == '2,3,4,5', 'slice from');
            check([1,2,3,4,5].slice(1,3).join(',') == '2,3', 'slice range');
            check([1,2,3,4,5].slice(-2).join(',') == '4,5', 'slice neg');
            check([1,2,3].slice(3,1).length == 0, 'slice empty');
            var r = [];
            for (var i = 0; i < 20; i++) if (i % 10 == 9) r[i] = i;
            var s = r.slice();
            check(s.length == 20, 'holey length');
            check((9 in s) && !(10 in s), 'holes preserved');"
        ),
    );
    run(
        "sort",
        &format!(
            "{CHECK}
            check([3,1,2].sort().join(',') == '1,2,3', 'default');
            check([10,9,1,100].sort().join(',') == '1,10,100,9', 'lexicographic');
            check([3,1,2].sort(function(a,b){{ return a-b; }}).join(',') == '1,2,3', 'cmp asc');
            check([3,1,2].sort(function(a,b){{ return b-a; }}).join(',') == '3,2,1', 'cmp desc');
            var a = [3,1,2];
            check(a.sort() == a, 'in place');
            check([5,undefined,3,undefined,1].sort().join(',') == '1,3,5,,', 'undefined last');
            var threw = false;
            try {{ [1,2].sort(function(){{ throw 1; }}); }} catch (e) {{ threw = true; }}
            check(threw, 'comparator throw propagates');"
        ),
    );
}
