//! ES 22.1 (Math): function properties of the Math namespace object.

use vm_core::Object;
use vm_core::RuntimeContext;
use vm_core::{ContextState, HandleSlice, Heap, Tagged, VM, Value, VmError};
use vm_core::{raise_runtime, rt_try};

/// `Math.sqrt(x)` (ES 22.1.2.29): ToNumber, then the IEEE-754 square root
/// (NaN/negative input → NaN, ±0 → ±0).
pub fn math_sqrt<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let n = rt_try!(vm, heap, state, state.handle_scope(|scope| {
        let arg = args
            .get(1)
            .map(|h| h.as_tagged(heap))
            .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
        let arg = scope.handle(arg);
        Object::to_numeric(vm, heap, state, arg)
    }));
    let Some(n) = n else {
        return heap.known().exception.as_tagged(heap).erase();
    };
    heap.new_number(n.sqrt())
}

/// ToNumeric the argument at `i` (defaulting to NaN), or `None` when user
/// code threw (a pending exception holds the cause).
fn numeric_arg(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    args: &HandleSlice<'_>,
    i: usize,
) -> Result<Option<f64>, VmError> {
    state.handle_scope(|scope| {
        let arg = args
            .get(i)
            .map(|h| h.as_tagged(heap))
            .unwrap_or_else(|| heap.known().undefined.as_tagged(heap).erase());
        let arg = scope.handle(arg);
        Object::to_numeric(vm, heap, state, arg)
    })
}

/// `Math.log(x)` (ES 22.1.2.15): natural logarithm.
pub fn math_log<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let Some(x) = rt_try!(vm, heap, state, numeric_arg(vm, heap, state, &args, 1)) else {
        return heap.known().exception.as_tagged(heap).erase();
    };
    heap.new_number(x.ln())
}

/// `Math.pow(base, exponent)` (ES 22.1.2.20).
pub fn math_pow<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let Some(base) = rt_try!(vm, heap, state, numeric_arg(vm, heap, state, &args, 1)) else {
        return heap.known().exception.as_tagged(heap).erase();
    };
    let Some(exp) = rt_try!(vm, heap, state, numeric_arg(vm, heap, state, &args, 2)) else {
        return heap.known().exception.as_tagged(heap).erase();
    };
    heap.new_number(base.powf(exp))
}
