use interpreter_become::BecomeInterpreter;
use js_compiler::JavascriptCompiler;
use js_runtime::JSRuntime;
use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm_core::{Interpreter, VM};

fn run(label: &str, src: &str) {
    let vm = VM::new::<MarkSweep, BecomeInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<JSRuntime>()
        .unwrap();
    let mut thread = vm.attach();
    thread.eval::<JavascriptCompiler>(src).expect("runs");
    if let Some(ex) = thread.take_pending_exception() {
        panic!("{label}: pending exception: {ex:?}");
    }
}

const CHECK: &str = "function check(c, m) { if (!c) { throw new Error(m); } }\n";

/// The keyed element fast paths (dense arrays, holes, gaps, append,
/// string indices) behave like the reference interpreter.
#[test]
fn element_fast_paths() {
    run(
        "array_n",
        &format!(
            "{CHECK}
      var e = Array(3);
      check(e[0] == undefined, 'hole');
      check(e.length == 3, 'length');
      e[1] = 5;
      check(e[1] == 5, 'store');
      check(e[0] == undefined, 'hole after store');"
        ),
    );
    run(
        "fill",
        &format!(
            "{CHECK}
      var a = Array(30);
      var i;
      for (i = 0; i < 30; i++) a[i] = i * 2;
      for (i = 0; i < 30; i++) check(a[i] == i * 2, 'fill ' + i);
      check(a[30] == undefined, 'oob undefined');"
        ),
    );
    run(
        "gap",
        &format!(
            "{CHECK}
      var g = Array(2);
      g[0] = 1; g[1] = 2;
      g[5] = 6;
      check(g.length == 6, 'gap length');
      check(g[3] == undefined, 'gap hole');
      check(g[5] == 6, 'gap value');"
        ),
    );
    run(
        "append",
        &format!(
            "{CHECK}
      var b = Array(0);
      var i;
      for (i = 0; i < 40; i++) b[i] = i;
      for (i = 0; i < 40; i++) check(b[i] == i, 'append ' + i);
      check(b.length == 40, 'append length');"
        ),
    );
    run(
        "string",
        &format!(
            "{CHECK}
      var s = 'hello';
      var acc = '';
      var i;
      for (i = 0; i < s.length; i++) acc += s[i];
      check(acc == 'hello', 'string loop');
      check('abc'[1] == 'b', 'string 1');
      check('abc'[5] == undefined, 'string oob');"
        ),
    );
}
