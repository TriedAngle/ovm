use vm::{
    Convert, FixedArray, Float, GcSlice, Handle, Heap, LoadOutcome, NoGc, Object, ObjectSlotsInit,
    SlotName, Symbol, Tagged, VMString, Value, ValueRef, VmError, load_outcome, set_prototype,
};

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
        let args = [hint_string];
        // SAFETY: hint_string was interned immediately above; the snapshot
        // is consumed (staged/copied into the frame) before any GC
        let args = unsafe { GcSlice::from_slice(&args) };
        let result = NativeContext::new(vm, heap, state).call(exotic, args)?;
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
        let args = [value];
        // SAFETY: `value` is a fresh snapshot read above; consumed
        // (staged/copied into the frame) before any GC in the call
        let args = unsafe { GcSlice::from_slice(&args) };
        let result = NativeContext::new(vm, heap, state).call(method, args)?;
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
pub(crate) fn get_property(
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
            let args = [receiver];
            // SAFETY: `receiver` is a fresh snapshot read above; consumed
            // (staged/copied into the frame) before any GC in the call
            let args = unsafe { GcSlice::from_slice(&args) };
            let result = NativeContext::new(vm, heap, state).call(getter, args)?;
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

/// ES 13.5.3 typeof: the interned type string for a value. `null` reports
/// `"object"`; callables report `"function"`.
pub(crate) fn type_of(vm: &VM, heap: &mut Heap, state: &ContextState, v: Value) -> Value {
    let name = heap.no_gc(|nogc, heap| {
        let known = heap.known();
        if v.is_smi() || v.get_as::<Float>(nogc, known.float_map).is_some() {
            "number"
        } else if v == known.undefined.value() || v == known.void.value() {
            "undefined"
        } else if v == known.null.value() {
            "object"
        } else if v == known.true_object.value() || v == known.false_object.value() {
            "boolean"
        } else if v.get_as::<VMString>(nogc, known.string_map).is_some() {
            "string"
        } else if v.get_as::<Symbol>(nogc, known.symbol_map).is_some() {
            "symbol"
        } else if let ValueRef::Object(obj) = v.value_ref(nogc) {
            if obj.as_ref().header.map.heap_ref(nogc).kind().is_callable() {
                "function"
            } else {
                "object"
            }
        } else {
            "object"
        }
    });
    intern_value(vm, heap, state, name)
}

/// ES 13.10.2 instanceof / 7.3.20 OrdinaryHasInstance: `Get(C, "prototype")`
/// must yield an object (else TypeError), then walk the object's prototype
/// chain for it. `None` means user code threw.
pub(crate) fn instance_of(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    object: Value,
    callable: Value,
) -> Result<Option<bool>, VmError> {
    if !is_callable(heap, callable) {
        return Err(VmError::Type);
    }
    // 4. P = Get(C, "prototype") — full [[Get]], getters may run user code
    let proto_name = intern_value(vm, heap, state, "prototype");
    let proto = get_property(vm, heap, state, callable, proto_name)?;
    let proto = match proto {
        Coercion::Threw => return Ok(None),
        Coercion::Value(v) => v,
    };
    // 5. P must be an object
    if heap.no_gc(|nogc, heap| Convert::is_primitive(nogc, heap, proto)) {
        return Err(VmError::Type);
    }
    Ok(Some(heap.no_gc(|nogc, heap| {
        has_proto_in_chain(nogc, heap, object, proto)
    })))
}

/// OrdinaryHasInstance step 6: walk the prototype chain of `object`.
fn has_proto_in_chain<'a>(nogc: &'a NoGc<'a>, heap: &Heap, object: Value, target: Value) -> bool {
    let ValueRef::Object(obj) = object.value_ref(nogc) else {
        return false;
    };
    let proto = obj.as_ref().header.map.heap_ref(nogc).prototype.inner();
    if proto == target {
        return true;
    }
    if proto == heap.known().null.value() {
        return false;
    }
    if let Some(parents) = proto.get_as::<FixedArray>(nogc, heap.known().array_map) {
        for i in 0..parents.len() {
            if has_proto_in_chain(nogc, heap, parents.at(i), target) {
                return true;
            }
        }
        return false;
    }
    has_proto_in_chain(nogc, heap, proto, target)
}

/// The Construct receiver (ES 9.2.2 step 5): a fresh `{}` whose
/// [[Prototype]] is `new.target.prototype` when that is an object, else
/// the ordinary object prototype (GetPrototypeFromConstructor,
/// ES 9.1.14). The callee is irrelevant here — `this` always comes from
/// new.target. `None` means user code threw.
pub(crate) fn create_construct_receiver(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    new_target: Handle<'_, Object>,
) -> Result<Option<Value>, VmError> {
    let proto_name = intern_value(vm, heap, state, "prototype");
    state.handle_scope(|scope| {
        let proto = get_property(vm, heap, state, new_target.value(), proto_name)?;
        let proto = match proto {
            Coercion::Threw => return Ok(None),
            Coercion::Value(v) => v,
        };
        // root the prototype before allocating below (GC may move it)
        let proto = if heap.no_gc(|nogc, heap| Convert::is_primitive(nogc, heap, proto)) {
            None
        } else {
            Some(
                scope
                    .create_handle(unsafe { Tagged::<Object>::from_value_unchecked(proto) })
                    .expect("object prototype must be strong"),
            )
        };
        let known = heap.known();
        let obj = heap
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map: known.object_initial_map,
                    values: &[],
                    elements: known.empty_fixed_array.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope)
            .as_tagged()
            .erase();
        if let Some(proto) = proto {
            set_prototype(heap, &scope, obj, proto.value())?;
        }
        Ok(Some(obj))
    })
}
