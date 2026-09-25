pub use vm_core::*;

pub use interpreter::{Interpreter, ThreadedInterpreter};
pub use js_runtime::JSRuntime;
pub use kette_runtime::KetteRuntime;

pub use js_compiler::JavascriptCompiler;
pub use kette_compiler::KetteCompiler;

pub trait VmEval {
    fn eval<C: Compiler>(&self, source: &str) -> Result<Value, ScriptError>;
}

impl VmEval for VM {
    fn eval<C: Compiler>(&self, source: &str) -> Result<Value, ScriptError> {
        let mut thread = self.attach();
        thread.eval::<C>(source)
    }
}
