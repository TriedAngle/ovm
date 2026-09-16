//! ES 28.2: the Proxy constructor, Proxy.revocable, and the
//! revoke-closure prelude.

use crate::natives::NativeContext;
use crate::{GcSlice, Value, VmError};
use super::object::plain_object;

/// `new Proxy(target, handler)` (ES 20.2.1.1): both must be JSReceivers;
/// the map's capability bits mirror the target's so callability is
/// observable (`typeof`, future `Call`/`Construct` dispatch).
pub(crate) fn proxy_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    if !nctx.is_construct() {
        return Err(VmError::Message("constructor Proxy requires 'new'"));
    }
    let target = args.get(1).ok_or(VmError::Type)?;
    let handler = args.get(2).ok_or(VmError::Type)?;
    let ok = nctx.heap().no_gc(|nogc| {
        crate::proxy::is_js_receiver(nogc, target) && crate::proxy::is_js_receiver(nogc, handler)
    });
    if !ok {
        return Err(VmError::Message(
            "cannot create proxy with a non-object target or handler",
        ));
    }
    nctx.handle_scope(|nctx, scope| {
        let target = scope.handle(target);
        let handler = scope.handle(handler);
        Ok(crate::proxy::allocate(
            nctx.heap(),
            target.value(),
            handler.value(),
        ))
    })
}

/// `Proxy.revocable(target, handler)` (ES 20.2.2.1): returns
/// `{ proxy, revoke }`; the revoke closure is the JS template installed
/// by REVOKE_PRELUDE (it keeps the idempotence flag and calls the
/// hidden `__revokeProxy` native).
pub(crate) fn proxy_revocable(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let target = args.get(1).ok_or(VmError::Type)?;
    let handler = args.get(2).ok_or(VmError::Type)?;
    let ok = nctx.heap().no_gc(|nogc| {
        crate::proxy::is_js_receiver(nogc, target) && crate::proxy::is_js_receiver(nogc, handler)
    });
    if !ok {
        return Err(VmError::Message(
            "cannot create proxy with a non-object target or handler",
        ));
    }
    nctx.handle_scope(|nctx, scope| {
        let target = scope.handle(target);
        let handler = scope.handle(handler);
        let proxy = scope.handle(crate::proxy::allocate(
            nctx.heap(),
            target.value(),
            handler.value(),
        ));
        // Function.prototype.__makeRevoke (installed by REVOKE_PRELUDE)
        let make_revoke = {
            let (vm, heap, state) = nctx.split();
            let name = vm
                .interner()
                .intern_str(heap, &scope, "__makeRevoke")
                .value();
            let proto = heap.known().function_prototype.value();
            match crate::runtime::Runtime::get_property(vm, heap, state, proto, name)? {
                crate::runtime::Coercion::Threw => {
                    return Ok(heap.known().exception.value());
                }
                crate::runtime::Coercion::Value(v) => v,
            }
        };
        let make_revoke = scope.handle(make_revoke);
        let undefined = nctx.heap().known().undefined.value();
        let (vm, heap, state) = nctx.split();
        let revoke = NativeContext::new(vm, heap, state).call(
            make_revoke.value(),
            scope.stage(&[undefined, proxy.value()]),
        )?;
        if revoke == heap.known().exception.value() {
            return Ok(heap.known().exception.value());
        }
        let revoke = scope.handle(revoke);
        plain_object(
            nctx,
            &[("proxy", proxy.value()), ("revoke", revoke.value())],
        )
    })
}

/// Hidden `__revokeProxy(p)`: nulls the proxy's target/handler slots
/// (idempotent — a null handler already means revoked). Called only by
/// the REVOKE_PRELUDE closure, which guards it with a done-flag.
pub(crate) fn proxy_revoke(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let proxy = args.get(1).ok_or(VmError::Arity)?;
    crate::proxy::revoke(nctx.heap(), proxy);
    Ok(nctx.heap().known().undefined.value())
}

/// The revoke-closure template: `done` plays [[RevocableProxy]]'s
/// cleared-slot role (idempotent revoke), the captured `p` keeps the
/// proxy reachable.
pub(crate) const REVOKE_PRELUDE: &str = r#"
Function.prototype.__makeRevoke = function (p) {
  var done = false;
  return function revoke() {
    if (!done) {
      done = true;
      __revokeProxy(p);
    }
  };
};
"#;
