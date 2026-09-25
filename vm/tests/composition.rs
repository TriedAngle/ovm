use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::{
    JSRuntime, JavascriptCompiler, KetteCompiler, KetteRuntime, Smi, ThreadedInterpreter, VM,
    VmEval,
};

#[test]
fn composed_vm_runs_js_and_kette() {
    let vm = VM::new::<MarkSweep, ThreadedInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<JSRuntime>()
        .unwrap()
        .add::<KetteRuntime>()
        .unwrap();

    let js = vm.eval::<JavascriptCompiler>("1 + 2").expect("js runs");
    assert_eq!(Smi::decode(js).unwrap().value(), 3);

    let kette = vm
        .eval::<KetteCompiler>("let obj = { x: 10\n read: { self.x } }\nobj.read()")
        .expect("kette runs");
    assert_eq!(Smi::decode(kette).unwrap().value(), 10);
}

#[test]
fn runtime_state_is_retrievable() {
    let vm = vm::VM::new::<MarkSweep, vm::ThreadedInterpreter>(MarkSweepConfig::default())
        .unwrap()
        .add::<vm::JSRuntime>()
        .unwrap();
    vm.arm_gc_stress();
    let _indices = &vm.runtime_state::<JSRuntime>().indices;
}
