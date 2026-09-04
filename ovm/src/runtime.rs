use vm::{Convert, Heap, LoadOutcome, SlotName, Value, ValueRef, VmError, load_outcome};

use crate::{ContextState, NativeContext, VM};

/// ToPrimitive hint (ES 7.1.1).
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Hint {
    Default,
    Number,
    String,
}

pub enum Coercion {
    Value(Value),
    Threw,
}

/// ES 7.1.1 ToPrimitive. Primitives pass through untouched
pub fn to_primitive(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    value: Value,
    hint: Hint,
) -> Result<Coercion, VmError> {
    if heap.no_gc(|nogc, heap| Convert::is_primitive(nogc, heap, value)) {
        return Ok(Coercion::Value(value));
    }
    let known = heap.known();
    let exception = known.exception.value();

    // 1. exotic @@toPrimitive (GetMethod)
    let exotic = get_property(vm, heap, state, value, known.to_primitive_symbol.value())?;
    let exotic = match exotic {
        Coercion::Threw => return Ok(Coercion::Threw),
        Coercion::Value(v) => v,
    };
    if exotic != known.undefined.value() && exotic != known.null.value() {
        if !is_callable(heap, exotic) {
            // GetMethod: a non-callable, non-nullish method is a TypeError
            return Err(VmError::Type);
        }
        let hint_name = match hint {
            Hint::Default => "default",
            Hint::Number => "number",
            Hint::String => "string",
        };
        let hint_string =
            state.handle_scope(|scope| vm.interner().intern(heap, &scope, hint_name).value());
        let result = NativeContext::new(vm, heap, state).call(exotic, &[hint_string])?;
        if result == exception {
            return Ok(Coercion::Threw);
        }
        return if heap.no_gc(|nogc, heap| Convert::is_primitive(nogc, heap, result)) {
            Ok(Coercion::Value(result))
        } else {
            Err(VmError::Type)
        };
    }

    // 2. OrdinaryToPrimitive: hint string → toString first, else valueOf first
    let method_names: [Value; 2] = if hint == Hint::String {
        [
            intern_value(vm, heap, state, "toString"),
            intern_value(vm, heap, state, "valueOf"),
        ]
    } else {
        [
            intern_value(vm, heap, state, "valueOf"),
            intern_value(vm, heap, state, "toString"),
        ]
    };
    for name in method_names {
        let method = get_property(vm, heap, state, value, name)?;
        let method = match method {
            Coercion::Threw => return Ok(Coercion::Threw),
            Coercion::Value(v) => v,
        };
        if !is_callable(heap, method) {
            continue;
        }
        let result = NativeContext::new(vm, heap, state).call(method, &[value])?;
        if result == exception {
            return Ok(Coercion::Threw);
        }
        if heap.no_gc(|nogc, heap| Convert::is_primitive(nogc, heap, result)) {
            return Ok(Coercion::Value(result));
        }
        // object result: try the next method name
    }
    Err(VmError::Type)
}

/// ToNumeric (ES 7.1.3): ToPrimitive with hint Number, then ToNumber.
/// `None` means user code threw (pending exception holds it).
pub fn to_numeric(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    value: Value,
) -> Result<Option<f64>, VmError> {
    match to_primitive(vm, heap, state, value, Hint::Number)? {
        Coercion::Threw => Ok(None),
        Coercion::Value(v) => Ok(Some(
            heap.no_gc(|nogc, heap| Convert::to_number(nogc, heap, v))?,
        )),
    }
}

pub fn numeric_op(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    a: Value,
    b: Value,
    op: fn(f64, f64) -> f64,
) -> Result<Option<Value>, VmError> {
    let a = to_numeric(vm, heap, state, a)?;
    let Some(a) = a else { return Ok(None) };
    let b = to_numeric(vm, heap, state, b)?;
    let Some(b) = b else { return Ok(None) };
    let r = op(a, b);
    Ok(Some(
        state.handle_scope(|scope| Convert::to_value(heap, &scope, r)),
    ))
}

/// Get a property value with full [[Get]] semantics: accessor getters are
/// called (nested run), missing properties yield undefined.
fn get_property(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    receiver: Value,
    name: Value,
) -> Result<Coercion, VmError> {
    let outcome =
        heap.no_gc(|nogc, heap| load_outcome(nogc, heap, receiver, SlotName::from_value(name)))?;
    match outcome {
        LoadOutcome::Value(v) => Ok(Coercion::Value(v)),
        LoadOutcome::Getter(getter) => {
            let exception = heap.known().exception.value();
            let result = NativeContext::new(vm, heap, state).call(getter, &[receiver])?;
            if result == exception {
                Ok(Coercion::Threw)
            } else {
                Ok(Coercion::Value(result))
            }
        }
    }
}

fn is_callable(heap: &mut Heap, v: Value) -> bool {
    heap.no_gc(|nogc, _heap| {
        let ValueRef::Object(obj) = v.value_ref(nogc) else {
            return false;
        };
        obj.as_ref().header.map.heap_ref(nogc).kind().is_callable()
    })
}

fn intern_value(vm: &VM, heap: &mut Heap, state: &ContextState, s: &str) -> Value {
    state.handle_scope(|scope| vm.interner().intern(heap, &scope, s).value())
}
