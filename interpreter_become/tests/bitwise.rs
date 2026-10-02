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
    vm.arm_gc_stress();
    let mut thread = vm.attach();
    thread.eval::<JavascriptCompiler>(src).expect("runs");
    if let Some(ex) = thread.take_pending_exception() {
        panic!("{label}: pending exception: {ex:?}");
    }
}

const CHECK: &str = "function check(c, m) { if (!c) { throw new Error(m); } }\n";

/// Bitwise/shift operators coerce non-Smi operands via ToInt32/ToUint32
/// instead of throwing, in both the binary and immediate forms.
#[test]
fn bitwise_coercion() {
    run(
        "bitwise",
        &format!(
            "{CHECK}
            var x = 1.5;
            check((x | 0) == 1, 'or float');
            check((x | x) == 1, 'or float binary');
            var s = '12';
            check((s | 0) == 12, 'or string');
            var a = 5.5, b = 3.2;
            check((a ^ b) == 6, 'xor float');
            check((a & 3) == 1, 'and float immediate');
            var c = 2.9;
            check((c << 1) == 4, 'shl float immediate');
            check((x >> 0) == 1, 'shr float immediate');
            check((-1 >>> 0) == 4294967295, 'shr logical');
            var big = 4294967296;
            check((big | 0) == 0, 'or 2^32');
            check(((3 * 1.1 + 1) >> 0) == 4, 'expr shr');
            var n = NaN;
            check((n | 0) == 0, 'or NaN');
            check((null | 0) == 0, 'or null');
            check((true | 0) == 1, 'or true');"
        ),
    );
}

/// Date formatting and parsing round-trip each other, including the
/// extended-year forms at the range extremes.
#[test]
fn date_round_trip() {
    run(
        "date",
        &format!(
            "{CHECK}
            check(new Date(0).toISOString() == '1970-01-01T00:00:00.000Z', 'iso epoch');
            check(new Date(0).toGMTString() == 'Thu, 01 Jan 1970 00:00:00 GMT', 'gmt epoch');
            check(new Date(8.64e15).toISOString() == '+275760-09-13T00:00:00.000Z', 'iso max');
            check(new Date(-8.64e15).toISOString() == '-271821-04-20T00:00:00.000Z', 'iso min');
            check(Date.parse(new Date(1609459200123).toISOString()) == 1609459200123, 'parse iso');
            check(Date.parse(new Date(0).toGMTString()) == 0, 'parse gmt');
            check(Date.parse(new Date(0).toString()) == 0, 'parse datestring');
            check(Date.parse(new Date(-8.64e15).toGMTString()) == -8.64e15, 'parse gmt min');
            check(Date.parse('2021-01-01') == 1609459200000, 'date only');
            check(isNaN(Date.parse('not a date')), 'parse garbage');"
        ),
    );
}
