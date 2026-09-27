pub use vm_core::*;

pub use interpreter_match_loop::{Interpreter, MatchLoopInterpreter};

#[cfg(feature = "fast")]
pub use interpreter_become::BecomeInterpreter;

/// The interpreter the `ovm` binary runs on. Enabling the `fast` feature
/// (nightly-only) picks the tail-call interpreter; otherwise the portable
/// match-loop interpreter is used.
#[cfg(feature = "fast")]
pub type DefaultInterpreter = interpreter_become::BecomeInterpreter;
#[cfg(not(feature = "fast"))]
pub type DefaultInterpreter = interpreter_match_loop::MatchLoopInterpreter;

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
