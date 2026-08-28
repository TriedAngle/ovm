/// Errors surfaced by VM operations (interpreter, natives, transitions).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum VmError {
    Arity,
    Type,
    Overflow,
    OutOfBounds,
    StackOverflow,
    /// Tried to add a property to a non-extensible object (spec: TypeError).
    NotExtensible,
}
