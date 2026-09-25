mod threaded;

pub use vm_core::{ExecuteFn, Interpreter};

pub struct ThreadedInterpreter;

impl Interpreter for ThreadedInterpreter {
    const EXECUTE: ExecuteFn = threaded::execute;
}
