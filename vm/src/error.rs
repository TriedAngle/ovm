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
    /// Unresolvable binding or TDZ access (spec: ReferenceError).
    Reference,
}

impl VmError {
    /// Name of the ECMAScript error class this VM error materializes as §20.5.3.2
    pub const fn name(self) -> &'static str {
        match self {
            Self::Arity | Self::Type | Self::NotExtensible => "TypeError",
            Self::Overflow | Self::OutOfBounds | Self::StackOverflow => "RangeError",
            Self::Reference => "ReferenceError",
        }
    }

    /// Default message used when materializing this error as an object.
    pub const fn message(self) -> &'static str {
        match self {
            Self::Arity => "wrong number of arguments",
            Self::Type => "invalid operand type",
            Self::Overflow => "value out of range",
            Self::OutOfBounds => "index out of bounds",
            Self::StackOverflow => "maximum call stack size exceeded",
            Self::NotExtensible => "object is not extensible",
            Self::Reference => "cannot access variable before initialization",
        }
    }
}
