//! ES 28.2: the Proxy constructor, Proxy.revocable, and the
//! revoke-closure prelude.

use crate::Lookup;
use crate::Object;
use crate::PropertyDescriptor;
use crate::RuntimeContext;
use crate::proxy::Proxy;
use crate::runtime::Coercion;
use crate::{HandleSlice, Tagged, Value, VmError};

/// `new Proxy(target, handler)` (ES 20.2.1.1): both must be JSReceivers;
/// the map's capability bits mirror the target's so callability is
/// observable (`typeof`, future `Call`/`Construct` dispatch).
pub fn proxy_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    if !nctx.is_construct() {
        return Err(VmError::Message("constructor Proxy requires 'new'"));
    }
    let RuntimeContext { heap, state, .. } = nctx;
    let (target, handler) = (
        args.get(1)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Type)?
            .raw(),
        args.get(2)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Type)?
            .raw(),
    );
    let ok = {
        Proxy::is_js_receiver(heap, unsafe { target.assume_valid(heap) })
            && Proxy::is_js_receiver(heap, unsafe { handler.assume_valid(heap) })
    };
    if !ok {
        return Err(VmError::Message(
            "cannot create proxy with a non-object target or handler",
        ));
    }
    state.handle_scope(|scope| {
        Ok(Proxy::allocate(
            heap,
            &scope,
            // Safety: fresh argument words, consumed by the allocation.
            unsafe { Tagged::<Value>::from_value_unchecked(target) },
            unsafe { Tagged::<Value>::from_value_unchecked(handler) },
        ))
    })
}

/// `Proxy.revocable(target, handler)` (ES 20.2.2.1): returns
/// `{ proxy, revoke }`; the revoke closure is the JS template installed
/// by REVOKE_PRELUDE (it keeps the idempotence flag and calls the
/// hidden `__revokeProxy` runtime).
pub fn proxy_revocable<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let (target, handler) = (
        args.get(1)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Type)?
            .raw(),
        args.get(2)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Type)?
            .raw(),
    );
    let ok = {
        Proxy::is_js_receiver(heap, unsafe { target.assume_valid(heap) })
            && Proxy::is_js_receiver(heap, unsafe { handler.assume_valid(heap) })
    };
    if !ok {
        return Err(VmError::Message(
            "cannot create proxy with a non-object target or handler",
        ));
    }
    state.handle_scope(|scope| {
        let proxy = scope.handle(Proxy::allocate(
            heap,
            &scope,
            // Safety: fresh argument words, consumed by the allocation.
            unsafe { Tagged::<Value>::from_value_unchecked(target) },
            unsafe { Tagged::<Value>::from_value_unchecked(handler) },
        ));
        // Function.prototype.__makeRevoke (installed by REVOKE_PRELUDE)
        let make_revoke = {
            // Safety: fresh interned word, consumed by the lookup.
            let name = vm
                .interner()
                .intern_str(heap, &scope, "__makeRevoke")
                .erase();
            let proto = heap.known().function_prototype.erase();
            match Lookup::get_property_on(vm, heap, state, proto, proto, name)? {
                Coercion::Threw => {
                    return Ok(heap.known().exception.as_tagged(heap).erase());
                }
                Coercion::Value(v) => scope.handle(v),
            }
        };
        // Safety: fresh rooted-slot words staged for the call.
        let exception = heap.known().exception.as_tagged(heap).raw();
        let staged = scope.stage(&[
            heap.known().undefined.as_tagged(heap).erase(),
            proxy.as_tagged(heap).erase(),
        ]);
        let revoke = {
            let r = RuntimeContext::call(vm, &mut *heap, state, make_revoke, staged, None)?;
            if r.raw() == exception {
                return Ok(heap.known().exception.as_tagged(heap).erase());
            }
            // Safety: fresh call result, rooted below.
            scope.handle(r)
        };
        let map = heap.known().object_initial_map;
        let obj = heap
            .new_object(&scope, map, HandleSlice::EMPTY)
            .into_handle(&scope);
        let proxy_name = vm.interner().intern_str(heap, &scope, "proxy");
        let proxy_name = scope.handle(proxy_name.as_tagged(heap));
        Object::define_own_property(
            heap,
            &scope,
            obj,
            proxy_name,
            PropertyDescriptor::data(proxy.erase()),
        )?;
        let revoke_name = vm.interner().intern_str(heap, &scope, "revoke");
        let revoke_name = scope.handle(revoke_name.as_tagged(heap));
        Object::define_own_property(
            heap,
            &scope,
            obj,
            revoke_name,
            PropertyDescriptor::data(revoke.erase()),
        )?;
        Ok(obj.as_tagged(heap).erase())
    })
}

/// Hidden `__revokeProxy(p)`: nulls the proxy's target/handler slots
/// (idempotent — a null handler already means revoked). Called only by
/// the REVOKE_PRELUDE closure, which guards it with a done-flag.
pub fn proxy_revoke<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext { heap, .. } = nctx;
    let proxy = args
        .get(1)
        .map(|h| h.as_tagged(heap))
        .ok_or(VmError::Arity)?
        .raw();
    Proxy::revoke(
        heap,
        // Safety: fresh argument word, consumed with no allocation delay.
        unsafe { Tagged::<Value>::from_value_unchecked(proxy) },
    );
    Ok(heap.known().undefined.as_tagged(heap).erase())
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
