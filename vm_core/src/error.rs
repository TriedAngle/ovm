/// Errors surfaced by VM operations (interpreter, runtimes, transitions).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
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
    /// TypeError with a specific message, by registry id.
    Message(Message),
}

/// Interned custom error messages: `VmError::Message(Message::...)`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u16)]
pub enum Message {
    /// `constructor Proxy requires 'new'`
    ProxyRequiresNew,
    /// `cannot create proxy with a non-object target or handler`
    ProxyBadTargetOrHandler,
    /// `proxy trap is not a function`
    ProxyTrapNotFunction,
    /// `proxy construct trap must return an object`
    ProxyConstructNotObject,
    /// `proxy get trap must match a non-writable, non-configurable property`
    ProxyGetMismatch,
    /// `proxy get trap must return undefined for an accessor without a getter`
    ProxyGetAccessorUndefined,
    /// `proxy set trap must match a non-writable, non-configurable property`
    ProxySetMismatch,
    /// `proxy set trap may not report success for an accessor without a setter`
    ProxySetAccessorNoSetter,
    /// `proxy has trap may not hide a non-configurable property`
    ProxyHasHiddenConfigurable,
    /// `proxy has trap may not hide a property of a non-extensible target`
    ProxyHasHiddenNonExtensible,
    /// `proxy deleteProperty trap may not delete a non-configurable property`
    ProxyDeleteConfigurable,
    /// `proxy deleteProperty trap may not delete a property of a non-extensible target`
    ProxyDeleteNonExtensible,
    /// `proxy defineProperty trap may not add a property to a non-extensible target`
    ProxyDefineNonExtensible,
    /// `proxy defineProperty trap may not claim a non-configurable new property`
    ProxyDefineNewConfigurable,
    /// `proxy defineProperty trap returned an incompatible descriptor`
    ProxyDefineIncompatible,
    /// `proxy defineProperty trap may not make a configurable property non-configurable`
    ProxyDefineUnconfigure,
    /// `proxy defineProperty trap may not make a non-configurable property non-writable`
    ProxyDefineUnwritable,
    /// `proxy preventExtensions trap returned true for an extensible target`
    ProxyPreventExtensionsTrue,
    /// `proxy isExtensible trap must match the target's extensibility`
    ProxyExtensibilityMismatch,
    /// `cannot perform '<trap>' on a proxy that has been revoked`
    ProxyRevokedGet,
    ProxyRevokedSet,
    ProxyRevokedHas,
    ProxyRevokedDeleteProperty,
    ProxyRevokedGetOwnPropertyDescriptor,
    ProxyRevokedDefineProperty,
    ProxyRevokedGetPrototypeOf,
    ProxyRevokedSetPrototypeOf,
    ProxyRevokedIsExtensible,
    ProxyRevokedPreventExtensions,
    ProxyRevokedOwnKeys,
    ProxyRevokedApply,
}

impl VmError {
    /// Name of the ECMAScript error class this VM error materializes as §20.5.3.2
    pub const fn name(self) -> &'static str {
        match self {
            Self::Arity
            | Self::Type
            | Self::NotExtensible
            | Self::Message(_) => "TypeError",
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
            Self::Message(m) => m.text(),
        }
    }
}

impl Message {
    pub const fn text(self) -> &'static str {
        match self {
            Self::ProxyRequiresNew => "constructor Proxy requires 'new'",
            Self::ProxyBadTargetOrHandler => {
                "cannot create proxy with a non-object target or handler"
            }
            Self::ProxyTrapNotFunction => "proxy trap is not a function",
            Self::ProxyConstructNotObject => "proxy construct trap must return an object",
            Self::ProxyGetMismatch => {
                "proxy get trap must match a non-writable, non-configurable property"
            }
            Self::ProxyGetAccessorUndefined => {
                "proxy get trap must return undefined for an accessor without a getter"
            }
            Self::ProxySetMismatch => {
                "proxy set trap must match a non-writable, non-configurable property"
            }
            Self::ProxySetAccessorNoSetter => {
                "proxy set trap may not report success for an accessor without a setter"
            }
            Self::ProxyHasHiddenConfigurable => {
                "proxy has trap may not hide a non-configurable property"
            }
            Self::ProxyHasHiddenNonExtensible => {
                "proxy has trap may not hide a property of a non-extensible target"
            }
            Self::ProxyDeleteConfigurable => {
                "proxy deleteProperty trap may not delete a non-configurable property"
            }
            Self::ProxyDeleteNonExtensible => {
                "proxy deleteProperty trap may not delete a property of a non-extensible target"
            }
            Self::ProxyDefineNonExtensible => {
                "proxy defineProperty trap may not add a property to a non-extensible target"
            }
            Self::ProxyDefineNewConfigurable => {
                "proxy defineProperty trap may not claim a non-configurable new property"
            }
            Self::ProxyDefineIncompatible => {
                "proxy defineProperty trap returned an incompatible descriptor"
            }
            Self::ProxyDefineUnconfigure => {
                "proxy defineProperty trap may not make a configurable property non-configurable"
            }
            Self::ProxyDefineUnwritable => {
                "proxy defineProperty trap may not make a non-configurable property non-writable"
            }
            Self::ProxyPreventExtensionsTrue => {
                "proxy preventExtensions trap returned true for an extensible target"
            }
            Self::ProxyExtensibilityMismatch => {
                "proxy isExtensible trap must match the target's extensibility"
            }
            Self::ProxyRevokedGet => "cannot perform 'get' on a proxy that has been revoked",
            Self::ProxyRevokedSet => "cannot perform 'set' on a proxy that has been revoked",
            Self::ProxyRevokedHas => "cannot perform 'has' on a proxy that has been revoked",
            Self::ProxyRevokedDeleteProperty => {
                "cannot perform 'deleteProperty' on a proxy that has been revoked"
            }
            Self::ProxyRevokedGetOwnPropertyDescriptor => {
                "cannot perform 'getOwnPropertyDescriptor' on a proxy that has been revoked"
            }
            Self::ProxyRevokedDefineProperty => {
                "cannot perform 'defineProperty' on a proxy that has been revoked"
            }
            Self::ProxyRevokedGetPrototypeOf => {
                "cannot perform 'getPrototypeOf' on a proxy that has been revoked"
            }
            Self::ProxyRevokedSetPrototypeOf => {
                "cannot perform 'setPrototypeOf' on a proxy that has been revoked"
            }
            Self::ProxyRevokedIsExtensible => {
                "cannot perform 'isExtensible' on a proxy that has been revoked"
            }
            Self::ProxyRevokedPreventExtensions => {
                "cannot perform 'preventExtensions' on a proxy that has been revoked"
            }
            Self::ProxyRevokedOwnKeys => {
                "cannot perform 'ownKeys' on a proxy that has been revoked"
            }
            Self::ProxyRevokedApply => {
                "cannot perform 'apply' on a proxy that has been revoked"
            }
        }
    }
}
