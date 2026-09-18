//! ES 28.2: the Proxy constructor, Proxy.revocable, and the
//! revoke-closure prelude.

use super::object::plain_object;
use crate::natives::NativeContext;
use crate::proxy::Proxy;
use crate::runtime::Coercion;
use crate::runtime::Runtime;
use crate::{HandleSlice, Tagged, Value, VmError};

/// `new Proxy(target, handler)` (ES 20.2.1.1): both must be JSReceivers;
/// the map's capability bits mirror the target's so callability is
/// observable (`typeof`, future `Call`/`Construct` dispatch).
pub fn proxy_constructor(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    if !nctx.is_construct() {
        return Err(VmError::Message("constructor Proxy requires 'new'"));
    }
    let (target, handler) = {
        let heap = &*nctx.heap();
        (
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Type)?
                .raw(),
            args.get(2)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Type)?
                .raw(),
        )
    };
    let ok = {
        let heap = &*nctx.heap();
        Proxy::is_js_receiver(heap, unsafe { target.assume_valid(heap) })
            && Proxy::is_js_receiver(heap, unsafe { handler.assume_valid(heap) })
    };
    if !ok {
        return Err(VmError::Message(
            "cannot create proxy with a non-object target or handler",
        ));
    }
    nctx.handle_scope(|nctx, scope| {
        Ok(Proxy::allocate(
            nctx.heap(),
            &scope,
            // Safety: fresh argument words, consumed by the allocation.
            unsafe { Tagged::<Value>::from_value_unchecked(target) },
            unsafe { Tagged::<Value>::from_value_unchecked(handler) },
        )
        .raw())
    })
}

/// `Proxy.revocable(target, handler)` (ES 20.2.2.1): returns
/// `{ proxy, revoke }`; the revoke closure is the JS template installed
/// by REVOKE_PRELUDE (it keeps the idempotence flag and calls the
/// hidden `__revokeProxy` native).
pub fn proxy_revocable(
    nctx: &mut NativeContext<'_>,
    args: HandleSlice<'_>,
) -> Result<Value, VmError> {
    let (target, handler) = {
        let heap = &*nctx.heap();
        (
            args.get(1)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Type)?
                .raw(),
            args.get(2)
                .map(|h| h.as_tagged(heap))
                .ok_or(VmError::Type)?
                .raw(),
        )
    };
    let ok = {
        let heap = &*nctx.heap();
        Proxy::is_js_receiver(heap, unsafe { target.assume_valid(heap) })
            && Proxy::is_js_receiver(heap, unsafe { handler.assume_valid(heap) })
    };
    if !ok {
        return Err(VmError::Message(
            "cannot create proxy with a non-object target or handler",
        ));
    }
    nctx.handle_scope(|nctx, scope| {
        let proxy = scope.handle(Proxy::allocate(
            nctx.heap(),
            &scope,
            // Safety: fresh argument words, consumed by the allocation.
            unsafe { Tagged::<Value>::from_value_unchecked(target) },
            unsafe { Tagged::<Value>::from_value_unchecked(handler) },
        ));
        // Function.prototype.__makeRevoke (installed by REVOKE_PRELUDE)
        let make_revoke = {
            let (vm, heap, state) = nctx.split();
            // Safety: fresh interned word, consumed by the lookup.
            let name = vm
                .interner()
                .intern_str(heap, &scope, "__makeRevoke")
                .erase();
            let proto = heap.known().function_prototype.erase();
            match Runtime::get_property(vm, heap, state, proto, name)? {
                Coercion::Threw => {
                    // Safety: fresh root-slot word read for the return.
                    return Ok(unsafe { heap.known().exception.read_unchecked() });
                }
                Coercion::Value(v) => scope.handle(v),
            }
        };
        // Safety: fresh rooted-slot words staged for the call.
        let undefined = unsafe { nctx.heap().known().undefined.read_unchecked() };
        let proxy_word = unsafe { proxy.read_unchecked() };
        let (vm, heap, state) = nctx.split();
        let revoke = NativeContext::new(vm, heap, state).call(
            // Safety: fresh rooted-slot word, consumed by the call.
            unsafe { Tagged::<Value>::from_value_unchecked(make_revoke.read_unchecked()) },
            scope.stage(&[
                unsafe { Tagged::<Value>::from_value_unchecked(undefined) },
                unsafe { Tagged::<Value>::from_value_unchecked(proxy_word) },
            ]),
        )?;
        if revoke == unsafe { heap.known().exception.read_unchecked() } {
            return Ok(revoke);
        }
        // Safety: fresh call result, rooted below.
        let revoke = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(revoke) });
        plain_object(nctx, &[("proxy", proxy.erase()), ("revoke", revoke)])
    })
}

/// Hidden `__revokeProxy(p)`: nulls the proxy's target/handler slots
/// (idempotent — a null handler already means revoked). Called only by
/// the REVOKE_PRELUDE closure, which guards it with a done-flag.
pub fn proxy_revoke(nctx: &mut NativeContext<'_>, args: HandleSlice<'_>) -> Result<Value, VmError> {
    let proxy = {
        let heap = &*nctx.heap();
        args.get(1)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?
            .raw()
    };
    Proxy::revoke(
        nctx.heap(),
        // Safety: fresh argument word, consumed with no allocation delay.
        unsafe { Tagged::<Value>::from_value_unchecked(proxy) },
    );
    // Safety: fresh root-slot word read for the immediate return.
    Ok(unsafe { nctx.heap().known().undefined.read_unchecked() })
}

/// The revoke-closure template: `done` plays [[RevocableProxy]]'s
/// cleared-slot role (idempotent revoke), the captured `p` keeps the
/// proxy reachable.
pub const REVOKE_PRELUDE: &str = r#"
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
