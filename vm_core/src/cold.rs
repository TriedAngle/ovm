//! The shared interpreter slow-path bodies: one semantic implementation
//! per operation, used by every interpreter shell. These return a plain `Tagged<Value>` where every
//! failure is the exception sentinel (the pending exception is set);
//! `Result<VmError>` survives only *inside* primitive helpers and is
//! folded to a sentinel (`Ctx::raise_tag`) at the first Tagged boundary.

use crate::convert::Convert;
use crate::handle::{Handle, HandleScope};
use crate::ic::{InlineCache, StoreHit, StoreOutcomeKind};
use crate::interp::Ctx;
use crate::lookup::{Key, LoadOutcome, Lookup};
use crate::objects::map::SlotName;
use crate::objects::object::Object;
use crate::objects::proxy::Proxy;
use crate::objects::string::DenseString;
use crate::runtime::{Coercion, Hint, RuntimeContext};
use crate::transition::{PropertyDescriptor, StoreOutcome, StoreSemantics};
use crate::value::{Smi, Tagged, Value};
use crate::{Compare, Errors, VmError};

/// `Add` cold body: ToPrimitive both operands, string concatenation when
/// either side is a string, numeric addition otherwise (ES 13.15.3).
pub fn add<'a>(
    ctx: &Ctx<'a>,
    lhs: Tagged<'_, Value>,
    rhs: Tagged<'_, Value>,
) -> Tagged<'a, Value> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let lhs = scope.handle(lhs);
        let lhs = match Object::to_primitive(vm, heap, state, lhs, Hint::Default) {
            Ok(Coercion::Threw) => return ctx.exception_word(),
            Ok(Coercion::Value(v)) => scope.handle(v),
            Err(e) => return ctx.raise_tag(e),
        };
        let rhs = scope.handle(rhs);
        let rhs = match Object::to_primitive(vm, heap, state, rhs, Hint::Default) {
            Ok(Coercion::Threw) => return ctx.exception_word(),
            Ok(Coercion::Value(v)) => scope.handle(v),
            Err(e) => return ctx.raise_tag(e),
        };
        let is_string = (
            lhs.as_tagged(heap).get_as::<DenseString>().is_some(),
            rhs.as_tagged(heap).get_as::<DenseString>().is_some(),
        );
        if is_string.0 || is_string.1 {
            let a = match Convert::to_string(heap, &scope, lhs) {
                Ok(v) => scope.handle(v),
                Err(e) => return ctx.raise_tag(e),
            };
            let b = match Convert::to_string(heap, &scope, rhs) {
                Ok(v) => scope.handle(v),
                Err(e) => return ctx.raise_tag(e),
            };
            DenseString::concat(heap, &scope, a, b)
                .as_tagged(heap)
                .erase()
        } else {
            let a = match Convert::to_number(heap, lhs.as_tagged(heap)) {
                Ok(v) => v,
                Err(e) => return ctx.raise_tag(e),
            };
            let b = match Convert::to_number(heap, rhs.as_tagged(heap)) {
                Ok(v) => v,
                Err(e) => return ctx.raise_tag(e),
            };
            // IEEE `-0 + -0` yields +0; the spec demands -0
            let r = a + b;
            let r = if r == 0.0 && a.is_sign_negative() && b.is_sign_negative() {
                -0.0
            } else {
                r
            };
            heap.new_number(r)
        }
    })
}

/// `Sub`/`Mul`/`Div`/`Mod` cold body: full `numeric_op` semantics.
pub fn numeric<'a>(
    ctx: &Ctx<'a>,
    lhs: Tagged<'_, Value>,
    rhs: Tagged<'_, Value>,
    f: fn(f64, f64) -> f64,
) -> Tagged<'a, Value> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let a = scope.handle(lhs);
        let b = scope.handle(rhs);
        match Object::numeric_op(vm, heap, state, a, b, f) {
            Ok(Some(v)) => v,
            Ok(None) => ctx.exception_word(),
            Err(e) => ctx.raise_tag(e),
        }
    })
}

/// `Negate` cold body: ToNumeric and flip (with `-0`).
pub fn negate<'a>(ctx: &Ctx<'a>, v: Tagged<'_, Value>) -> Tagged<'a, Value> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let v = scope.handle(v);
        match Object::to_numeric(vm, heap, state, v) {
            Ok(Some(n)) => heap.new_number(-n),
            Ok(None) => ctx.exception_word(),
            Err(e) => ctx.raise_tag(e),
        }
    })
}

/// `Bitwise*`/`Shift*` cold body: ToInt32/ToUint32 both operands and apply
/// the op. `kind`: 0 = `|`, 1 = `^`, 2 = `&`, 3 = `<<`, 4 = `>>`,
/// 5 = `>>>`.
pub fn bitwise<'a>(
    ctx: &Ctx<'a>,
    kind: u8,
    lhs: Tagged<'_, Value>,
    rhs: Tagged<'_, Value>,
) -> Tagged<'a, Value> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let lhs = scope.handle(lhs);
        let rhs = scope.handle(rhs);
        let a = match Object::to_numeric(vm, heap, state, lhs) {
            Ok(Some(n)) => n,
            Ok(None) => return ctx.exception_word(),
            Err(e) => return ctx.raise_tag(e),
        };
        let b = match Object::to_numeric(vm, heap, state, rhs) {
            Ok(Some(n)) => n,
            Ok(None) => return ctx.exception_word(),
            Err(e) => return ctx.raise_tag(e),
        };
        let (ai, bi) = (Convert::number_to_int32(a), Convert::number_to_uint32(b));
        let r: i64 = match kind {
            0 => (ai | (bi as i32)) as i64,
            1 => (ai ^ (bi as i32)) as i64,
            2 => (ai & (bi as i32)) as i64,
            3 => ai.wrapping_shl(bi & 31) as i64,
            4 => ai.wrapping_shr(bi & 31) as i64,
            // >>>: the uint32 result is always non-negative
            _ => Convert::number_to_uint32(a).wrapping_shr(bi & 31) as i64,
        };
        Smi::new(r).into_tagged()
    })
}

/// The compare cold body: `cmp` selects the relation (0 = loose `==`,
/// 1 = strict `===`, 2 = `<`, 3 = `<=`, 4 = `>`, 5 = `>=`). Returns the
/// boolean result value or the sentinel (a coercion threw).
pub fn compare<'a>(
    ctx: &Ctx<'a>,
    cmp: u8,
    lhs: Tagged<'_, Value>,
    rhs: Tagged<'_, Value>,
) -> Tagged<'a, Value> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let x = scope.handle(lhs);
        let y = scope.handle(rhs);
        if cmp == 1 {
            // IsStrictEqual (ES 7.2.15): no coercion, ever
            let b = Compare::strict_equal(heap, x.as_tagged(heap), y.as_tagged(heap));
            return Convert::boolean(heap, b);
        }
        // IsLooselyEqual: object ↔ object compares by identity; an
        // object ↔ primitive pair coerces only the object. The
        // relational operators coerce both sides with hint Number.
        let loose = cmp == 0;
        let x_obj = Compare::is_object_operand(heap, x.as_tagged(heap));
        let y_obj = Compare::is_object_operand(heap, y.as_tagged(heap));
        if loose && x_obj && y_obj {
            let b = Compare::strict_equal(heap, x.as_tagged(heap), y.as_tagged(heap));
            return Convert::boolean(heap, b);
        }
        let hint = if loose { Hint::Default } else { Hint::Number };
        let x = if x_obj {
            match Object::to_primitive(vm, heap, state, x, hint) {
                Ok(Coercion::Threw) => return ctx.exception_word(),
                Ok(Coercion::Value(v)) => scope.handle(v),
                Err(e) => return ctx.raise_tag(e),
            }
        } else {
            x
        };
        let y = if y_obj {
            match Object::to_primitive(vm, heap, state, y, hint) {
                Ok(Coercion::Threw) => return ctx.exception_word(),
                Ok(Coercion::Value(v)) => scope.handle(v),
                Err(e) => return ctx.raise_tag(e),
            }
        } else {
            y
        };
        let b = match cmp {
            0 => Compare::equal(heap, x.as_tagged(heap), y.as_tagged(heap)),
            2 => Compare::less_than(heap, x.as_tagged(heap), y.as_tagged(heap)),
            3 => Compare::less_than_or_equal(heap, x.as_tagged(heap), y.as_tagged(heap)),
            4 => Compare::greater_than(heap, x.as_tagged(heap), y.as_tagged(heap)),
            _ => Compare::greater_than_or_equal(heap, x.as_tagged(heap), y.as_tagged(heap)),
        };
        match b {
            Ok(v) => Convert::boolean(heap, v),
            Err(e) => ctx.raise_tag(e),
        }
    })
}

/// `LoadNamedProperty*` cold body: proxy trap, lookup, IC fill only for
/// plain values (getters are never cached).
pub fn named_load<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    name_idx: usize,
    fb_slot: usize,
) -> Tagged<'a, Value> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let recv = scope.handle(recv);
        let name = scope.handle(
            ctx.stack()
                .callable(heap, ctx.frame_base())
                .as_ref()
                .constant_slot_name(heap, name_idx),
        );
        // ES 20.2.5.8: a proxy receiver runs its `get` trap (with the
        // proxy as `this`) instead of a descriptor walk. No IC update:
        // traps are dynamic, a proxy map must never be cached as a
        // field handler.
        if Proxy::is_proxy(heap, recv.as_tagged(heap)) {
            return match Proxy::get(vm, heap, state, recv, recv, name.erase()) {
                Ok(Coercion::Threw) => ctx.exception_word(),
                Ok(Coercion::Value(v)) => v,
                Err(e) => ctx.raise_tag(e),
            };
        }
        let out = match Lookup::load_outcome(heap, recv.as_tagged(heap), name.as_tagged(heap)) {
            Ok(LoadOutcome::Value(v)) => {
                let v = scope.handle(v);
                InlineCache::update_load(
                    heap,
                    &scope,
                    ctx.feedback_ref(heap).map(|v| scope.handle(v)),
                    fb_slot,
                    recv.as_tagged(heap)
                        .as_heap_object()
                        .map(|o| scope.handle(o)),
                    scope.handle(name.as_tagged(heap).erase().as_name()),
                    true,
                );
                v
            }
            Ok(LoadOutcome::Getter(getter)) => {
                let getter = scope.handle(getter);
                let args = scope.stage(&[recv.as_tagged(heap).erase()]);
                match RuntimeContext::call(vm, heap, state, getter, args, None) {
                    Ok(v) => scope.handle(v),
                    Err(e) => return ctx.raise_tag(e),
                }
            }
            Err(e) => return ctx.raise_tag(e),
        };
        out.as_tagged(heap).erase()
    })
}

/// `LoadKeyedProperty`/`LoadKeyedPropertyReg` cold body: property-key
/// coercion, proxies, string receivers, getters.
pub fn keyed_load<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    key: Tagged<'_, Value>,
    fb_slot: Option<usize>,
) -> Tagged<'a, Value> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let recv = scope.handle(recv);
        let raw_key = scope.handle(key);
        let key = match Object::to_property_key(vm, heap, state, raw_key) {
            Ok(Some(k)) => scope.handle(k),
            Ok(None) => return ctx.exception_word(),
            Err(e) => return ctx.raise_tag(e),
        };
        // ES 20.2.5.8 (keyed form): the `get` trap with the coerced
        // key. No IC update: traps are dynamic.
        if Proxy::is_proxy(heap, recv.as_tagged(heap)) {
            return match Proxy::get(vm, heap, state, recv, recv, key.erase()) {
                Ok(Coercion::Threw) => ctx.exception_word(),
                Ok(Coercion::Value(v)) => v,
                Err(e) => ctx.raise_tag(e),
            };
        }
        let element_index = match Lookup::classify_key(heap, key.as_tagged(heap).erase()) {
            Ok(Key::Element(i)) => Some(i),
            _ => None,
        };
        if let Some(i) = element_index {
            if let Some(fb) = fb_slot {
                InlineCache::update_load_element(
                    heap,
                    &scope,
                    ctx.feedback_ref(heap).map(|v| scope.handle(v)),
                    fb,
                    Some(recv),
                    i,
                );
            }
            if let Some(unit) = DenseString::index_element(heap, &scope, recv, key) {
                return scope.handle(unit).as_tagged(heap).erase();
            }
        }
        let out = match Lookup::load_outcome_keyed(heap, recv.as_tagged(heap), key.as_tagged(heap))
        {
            Ok(LoadOutcome::Value(v)) => scope.handle(v),
            Ok(LoadOutcome::Getter(getter)) => {
                let getter = scope.handle(getter);
                let args = scope.stage(&[recv.as_tagged(heap).erase()]);
                match RuntimeContext::call(vm, heap, state, getter, args, None) {
                    Ok(v) => scope.handle(v),
                    Err(e) => return ctx.raise_tag(e),
                }
            }
            Err(e) => return ctx.raise_tag(e),
        };
        out.as_tagged(heap).erase()
    })
}

/// `LoadElementImm` cold body: the keyed-load tail with the constant
/// index as the key (string receivers, proxies, named fallback).
pub fn keyed_load_imm<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    idx: usize,
) -> Tagged<'a, Value> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let recv = scope.handle(recv);
        let smi = Smi::new(idx as i64).into_tagged();
        let key: Handle<'_, SlotName> = scope.handle(smi.as_name());
        if let Some(unit) = DenseString::index_element(heap, &scope, recv, key) {
            return scope.handle(unit).as_tagged(heap).erase();
        }
        if Proxy::is_proxy(heap, recv.as_tagged(heap)) {
            return match Proxy::get(vm, heap, state, recv, recv, key.erase()) {
                Ok(Coercion::Threw) => ctx.exception_word(),
                Ok(Coercion::Value(v)) => v,
                Err(e) => ctx.raise_tag(e),
            };
        }
        let out = match Lookup::load_outcome_keyed(heap, recv.as_tagged(heap), key.as_tagged(heap))
        {
            Ok(LoadOutcome::Value(v)) => scope.handle(v),
            Ok(LoadOutcome::Getter(getter)) => {
                let getter = scope.handle(getter);
                let args = scope.stage(&[recv.as_tagged(heap).erase()]);
                match RuntimeContext::call(vm, heap, state, getter, args, None) {
                    Ok(v) => scope.handle(v),
                    Err(e) => return ctx.raise_tag(e),
                }
            }
            Err(e) => return ctx.raise_tag(e),
        };
        out.as_tagged(heap).erase()
    })
}

/// `StoreKeyedProperty` cold body: proxy traps first, then key coercion,
/// element growth with the store-element IC fill, and named-property
/// transitions.
pub fn keyed_store<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    key: Tagged<'_, Value>,
    value: Tagged<'_, Value>,
    fb_slot: Option<usize>,
    semantics: StoreSemantics,
) -> Tagged<'a, Value> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let recv = scope.handle(recv);
        let raw_key = scope.handle(key);
        let value = scope.handle(value);
        if Proxy::is_proxy(heap, recv.as_tagged(heap)) {
            return match Proxy::set(vm, heap, state, recv, raw_key, value, recv) {
                Ok(Coercion::Threw) => ctx.exception_word(),
                Ok(Coercion::Value(_)) => value.as_tagged(heap).erase(),
                Err(e) => ctx.raise_tag(e),
            };
        }
        let key = match Object::to_property_key(vm, heap, state, raw_key) {
            Ok(Some(k)) => scope.handle(k),
            Ok(None) => return ctx.exception_word(),
            Err(e) => return ctx.raise_tag(e),
        };
        let name: Handle<'_, SlotName> =
            match Lookup::classify_key(heap, key.as_tagged(heap).erase()) {
                Ok(Key::Element(i)) => {
                    if recv
                        .as_tagged(heap)
                        .as_heap_object()
                        .is_some_and(|obj| obj.as_ref().is_array(heap))
                    {
                        let obj = scope
                            .cast::<Object>(recv.as_tagged(heap))
                            .expect("array receiver is an object");
                        let grew = i >= obj.as_tagged(heap).as_ref().length();
                        if let Err(e) = Object::store_array_element(heap, &scope, &obj, i, &value) {
                            return ctx.raise_tag(e);
                        }
                        if let Some(fb) = fb_slot {
                            InlineCache::update_store_element(
                                heap,
                                &scope,
                                ctx.feedback_ref(heap).map(|v| scope.handle(v)),
                                fb,
                                Some(recv),
                                grew,
                            );
                        }
                        return value.as_tagged(heap).erase();
                    }
                    scope.handle(Tagged::<SlotName>::from(Smi::new(i as i64)))
                }
                Ok(Key::Name(key)) => scope.handle(key),
                Err(e) => return ctx.raise_tag(e),
            };
        let outcome = match recv.as_tagged(heap).erase().store_lookup(
            heap,
            &scope,
            name.as_tagged(heap),
            value.as_tagged(heap),
            semantics,
        ) {
            Ok(o) => o,
            Err(e) => return ctx.raise_tag(e),
        };
        match apply_store_outcome(ctx, recv, outcome, value) {
            Ok(()) => value.as_tagged(heap).erase(),
            Err(()) => ctx.exception_word(),
        }
    })
}

/// `LoadGlobal*` cold body: straight to the global lookup (no
/// second-tier IC re-probe), IC fill only for plain data properties
/// (getters and absent globals are never cached).
pub fn global_load<'a>(
    ctx: &Ctx<'a>,
    name_idx: usize,
    fb_slot: usize,
    throws: bool,
) -> Tagged<'a, Value> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let global = heap.known().global_object;
        let name = scope.handle(ctx.constants_ref(heap).at(heap, name_idx).erase().as_name());
        let out = match global.as_tagged(heap).lookup(heap, name.as_tagged(heap)) {
            Lookup::Data { slot, .. } => {
                let v = scope.handle(slot.get(heap));
                InlineCache::update_load(
                    heap,
                    &scope,
                    ctx.feedback_ref(heap).map(|v| scope.handle(v)),
                    fb_slot,
                    Some(scope.handle(global.as_tagged(heap))),
                    scope.handle(name.as_tagged(heap).erase().as_name()),
                    false,
                );
                v
            }
            Lookup::Accessor { pair, .. } => {
                let getter = scope.handle(pair.get.get(heap));
                if getter.as_tagged(heap) == heap.known().undefined.as_tagged(heap) {
                    // a setter-only accessor: the read yields undefined
                    return ctx.undefined_word();
                }
                let args = scope.stage(&[global.as_tagged(heap).erase()]);
                match RuntimeContext::call(vm, heap, state, getter, args, None) {
                    Ok(v) => scope.handle(v),
                    Err(e) => return ctx.raise_tag(e),
                }
            }
            Lookup::NotFound => {
                if throws {
                    // unresolvable reference: GetValue throws a
                    // ReferenceError naming the binding
                    let text = name
                        .as_tagged(heap)
                        .erase()
                        .get_as::<DenseString>()
                        .map(|s| s.to_rust_string(heap))
                        .unwrap_or_default();
                    let ex = Errors::not_defined(vm, heap, state, &text)
                        .expect("error materialization must not fail");
                    state.set_pending_exception(ex);
                    return ctx.exception_word();
                } else {
                    return ctx.undefined_word();
                }
            }
        };
        out.as_tagged(heap).erase()
    })
}

/// `StoreNamedProperty` cold body: the store IC, transitions and setter
/// invocation (Shadow semantics).
pub fn store_named<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    name_idx: usize,
    fb_slot: usize,
    value: Tagged<'_, Value>,
) -> Tagged<'a, Value> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let recv = scope.handle(recv);
        let value = scope.handle(value);
        let name = scope.handle(ctx.constants_ref(heap).at(heap, name_idx).erase().as_name());

        // ES 20.2.5.10: a proxy receiver runs its `set` trap; no IC
        // update (dynamic traps must not be cached), and the probe is
        // skipped entirely so an armed site never matches a proxy map.
        if Proxy::is_proxy(heap, recv.as_tagged(heap)) {
            return match Proxy::set(vm, heap, state, recv, name.erase(), value, recv) {
                Ok(Coercion::Threw) => ctx.exception_word(),
                Ok(Coercion::Value(_)) => value.as_tagged(heap).erase(),
                Err(e) => ctx.raise_tag(e),
            };
        }

        if let Some(hit) = InlineCache::try_store(
            heap,
            &scope,
            ctx.feedback_ref(heap).map(|v| scope.handle(v)),
            fb_slot,
            recv.erase(),
            name,
            value,
        ) {
            return match hit {
                StoreHit::Done => value.as_tagged(heap).erase(),
                StoreHit::Setter(setter) => {
                    let setter = scope.handle(setter);
                    let args = scope.stage(&[recv.as_tagged(heap).erase(), value.as_tagged(heap)]);
                    // the setter's return value is ignored (store result
                    // semantics: the accumulator keeps the stored value);
                    // a throwing setter routes the sentinel to the caller
                    let v = match RuntimeContext::call(vm, heap, state, setter, args, None) {
                        Ok(v) => scope.handle(v),
                        Err(e) => return ctx.raise_tag(e),
                    };
                    v.as_tagged(heap)
                }
            };
        }

        store_named_tail(
            ctx,
            recv,
            name,
            value,
            Some(fb_slot),
            StoreSemantics::Shadow,
        )
    })
}

/// `StoreNamedPropertyNoShadow` cold body: the named-store tail without
/// the IC (WriteThrough stores into parent-pair arrays that no map
/// describes).
pub fn store_named_no_shadow<'a>(
    ctx: &Ctx<'a>,
    recv: Tagged<'_, Value>,
    name_idx: usize,
    value: Tagged<'_, Value>,
) -> Tagged<'a, Value> {
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let recv = scope.handle(recv);
        let value = scope.handle(value);
        let name = scope.handle(ctx.constants_ref(heap).at(heap, name_idx).erase().as_name());
        store_named_tail(ctx, recv, name, value, None, StoreSemantics::WriteThrough)
    })
}

/// `StoreGlobal` cold body: the named-store tail against the global
/// object (WriteThrough, no IC).
pub fn store_global<'a>(
    ctx: &Ctx<'a>,
    name_idx: usize,
    value: Tagged<'_, Value>,
) -> Tagged<'a, Value> {
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let value = scope.handle(value);
        let name = scope.handle(ctx.constants_ref(heap).at(heap, name_idx).erase().as_name());
        let global = scope.handle(heap.known().global_object.as_tagged(heap).erase());
        store_named_tail(ctx, global, name, value, None, StoreSemantics::WriteThrough)
    })
}

/// The shared named-store tail: generic `store_lookup`, outcome
/// application, then the IC fill (the become interpreter's ordering: a
/// transition is cached against the map the store produced, a setter ran
/// before the fill).
fn store_named_tail<'a>(
    ctx: &Ctx<'a>,
    recv: Handle<'_, Value>,
    name: Handle<'_, SlotName>,
    value: Handle<'_, Value>,
    fb_slot: Option<usize>,
    semantics: StoreSemantics,
) -> Tagged<'a, Value> {
    let state = ctx.state();
    state.handle_scope(|scope| {
        store_named_tail_scoped(ctx, scope, recv, name, value, fb_slot, semantics)
    })
}

fn store_named_tail_scoped<'a>(
    ctx: &Ctx<'a>,
    scope: HandleScope<'_>,
    recv: Handle<'_, Value>,
    name: Handle<'_, SlotName>,
    value: Handle<'_, Value>,
    fb_slot: Option<usize>,
    semantics: StoreSemantics,
) -> Tagged<'a, Value> {
    let heap = unsafe { ctx.heap_mut() };
    // the receiver's map before the store: the state key when
    // the outcome transitions it
    let prev = recv
        .as_tagged(heap)
        .as_heap_object()
        .map(|o| scope.handle(o.as_ref().map_ref(heap)));
    let outcome = match recv.as_tagged(heap).erase().store_lookup(
        heap,
        &scope,
        name.as_tagged(heap),
        value.as_tagged(heap),
        semantics,
    ) {
        Ok(o) => o,
        Err(e) => return ctx.raise_tag(e),
    };
    let kind = match &outcome {
        StoreOutcome::Done => StoreOutcomeKind::Done,
        StoreOutcome::Transition { .. } => StoreOutcomeKind::Transition,
        StoreOutcome::CallSetter { .. } => StoreOutcomeKind::CallSetter,
    };
    if let Err(()) = apply_store_outcome(ctx, recv, outcome, value) {
        return ctx.exception_word();
    }
    if let (Some(fb), Some(prev)) = (fb_slot, prev) {
        InlineCache::update_store(
            heap,
            &scope,
            ctx.feedback_ref(heap).map(|v| scope.handle(v)),
            fb,
            recv.erase(),
            name,
            prev,
            kind,
        );
    }
    value.as_tagged(heap).erase()
}

/// Apply a store outcome: transitions define the own property, setters
/// run as a blocking call (their return value is ignored; the
/// accumulator keeps the stored value). `Err(())` = threw (pending set).
pub fn apply_store_outcome(
    ctx: &Ctx<'_>,
    receiver: Handle<'_, Value>,
    outcome: StoreOutcome<'_>,
    value: Handle<'_, Value>,
) -> Result<(), ()> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    match outcome {
        StoreOutcome::Transition {
            receiver: recv,
            name,
        } => state.handle_scope(|scope| {
            match Object::add_own_property(
                heap,
                &scope,
                recv,
                name,
                PropertyDescriptor::data(value),
            ) {
                // TODO(strict-mode): a false result must throw in strict
                // code; the current store path preserves its existing
                // sloppy result.
                Ok(_) => Ok(()),
                Err(e) => {
                    ctx.raise_tag(e);
                    Err(())
                }
            }
        }),
        StoreOutcome::CallSetter { setter } => state.handle_scope(|scope| {
            let args = scope.stage(&[receiver.as_tagged(heap).erase(), value.as_tagged(heap)]);
            match RuntimeContext::call(vm, heap, state, setter, args, None) {
                Ok(v) => {
                    // the setter's return value is ignored: the stored
                    // value stays the expression's result
                    if ctx.is_throw(v) { Err(()) } else { Ok(()) }
                }
                Err(e) => {
                    ctx.raise_tag(e);
                    Err(())
                }
            }
        }),
        StoreOutcome::Done => Ok(()),
    }
}

/// `Construct` cold body: [[Construct]] with receiver synthesis,
/// derived-class handling and proxy traps (ES 9.2.2).
pub fn construct<'a>(
    ctx: &Ctx<'a>,
    callee: Tagged<'_, Value>,
    args_base: i32,
    count: usize,
) -> Tagged<'a, Value> {
    let vm = ctx.vm();
    let heap = unsafe { ctx.heap_mut() };
    let state = ctx.state();
    state.handle_scope(|scope| {
        let callee = scope.handle(callee);
        let Some(obj) = callee.as_tagged(heap).as_heap_object() else {
            return ctx.raise_tag(VmError::Type);
        };
        if !obj.as_ref().header.map.get(heap).kind().is_constructor() {
            return ctx.raise_tag(VmError::Type);
        }
        let args = ctx.stack().args(ctx.frame_base(), args_base, count);
        if Proxy::is_proxy(heap, callee.as_tagged(heap)) {
            return match Proxy::construct(vm, heap, state, callee, args, callee) {
                Ok(Coercion::Threw) => ctx.exception_word(),
                Ok(Coercion::Value(v)) => v,
                Err(e) => ctx.raise_tag(e),
            };
        }
        // a derived constructor's `this` is the hole until `super()` binds
        // it; ordinary and
        // base-class constructors synthesize the receiver
        let derived = obj
            .as_ref()
            .header
            .map
            .get(heap)
            .kind()
            .is_class_constructor()
            && obj
                .as_ref()
                .callable_info(heap)
                .is_some_and(|info| info.function_kind().is_derived_class_constructor());
        let callee = scope
            .cast::<Object>(callee.as_tagged(heap))
            .expect("constructible callee is an object");
        let receiver = if derived {
            scope.handle(heap.known().the_hole.as_tagged(heap).erase())
        } else {
            match Object::create_construct_receiver_value(vm, heap, state, callee.erase()) {
                Ok(Some(r)) => scope.handle(r),
                Ok(None) => return ctx.exception_word(),
                Err(e) => return ctx.raise_tag(e),
            }
        };
        let mut staged: Vec<Tagged<'_, Value>> = Vec::with_capacity(count + 1);
        staged.push(receiver.as_tagged(heap).erase());
        staged.extend(args.iter().map(|h| h.as_tagged(heap)));
        let staged = scope.stage(&staged);
        let result = match RuntimeContext::call(
            vm,
            heap,
            state,
            callee.erase(),
            staged,
            Some(callee.erase()),
        ) {
            Ok(v) => scope.handle(v),
            Err(e) => return ctx.raise_tag(e),
        };
        if result.as_tagged(heap) == ctx.exception_word() {
            return ctx.exception_word();
        }
        if Convert::is_primitive(heap, result.as_tagged(heap)) {
            if derived {
                // a derived constructor returned a primitive (ES 9.2.2.1)
                return ctx.raise_tag(VmError::Type);
            }
            receiver.as_tagged(heap).erase()
        } else {
            result.as_tagged(heap).erase()
        }
    })
}
