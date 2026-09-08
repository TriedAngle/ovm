use vm::{
    CallableInfoObject, Context, Convert, FixedArray, Float, GcSlice, Handle, HandleScope, Heap,
    LoadOutcome, NoGc, Object, ObjectSlotsInit, PropertyDescriptor, SlotName, Smi, Symbol, Tagged,
    VMString, Value, ValueRef, VmError, load_outcome,
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

pub struct Runtime;

impl Runtime {
    pub fn create_closure(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        info: Handle<'_, CallableInfoObject>,
        context: Value,
    ) -> Result<Value, VmError> {
        let (kind, source_name, formal_parameter_count) = heap.no_gc(|nogc| {
            let info = info.heap_ref(nogc);
            (
                info.function_kind(),
                info.name(nogc),
                info.formal_parameter_count(),
            )
        });
        let function_name = match source_name {
            Some(name) => scope.handle(Tagged::from_value(name)),
            None => scope.handle(Tagged::from_value(heap.known().strings.empty.value())),
        };
        let map = match kind {
            kind if kind.is_class_constructor() => heap.known().class_constructor_map,
            kind if kind.is_constructible() => heap.known().function_map,
            _ => heap.known().non_constructor_function_map,
        };
        let context = unsafe { scope.handle_value::<Context>(context) };
        let function = heap
            .allocate_object(
                scope,
                ObjectSlotsInit {
                    map,
                    values: &[info.as_tagged().erase(), context.as_tagged().erase()],
                    elements: heap.known().empty_fixed_array.erase(),
                    length: 0,
                },
            )
            .into_handle(scope);

        let length_key = heap.known().strings.length;
        let name_key = heap.known().strings.name;
        let defined = Object::define_own_property(
            heap,
            scope,
            function,
            length_key,
            PropertyDescriptor::Data {
                value: Smi::new(formal_parameter_count as i64).encode(),
                writable: false,
                enumerable: false,
                configurable: true,
            },
        )?;
        if !defined {
            return Err(VmError::Type);
        }
        let defined = Object::define_own_property(
            heap,
            scope,
            function,
            name_key,
            PropertyDescriptor::Data {
                value: function_name.value(),
                writable: false,
                enumerable: false,
                configurable: true,
            },
        )?;
        if !defined {
            return Err(VmError::Type);
        }

        if kind.needs_prototype() {
            let proto = heap
                .allocate_object(
                    scope,
                    ObjectSlotsInit {
                        map: heap.known().object_initial_map,
                        values: &[],
                        elements: heap.known().empty_fixed_array.erase(),
                        length: 0,
                    },
                )
                .into_handle(scope);
            let constructor = heap.known().strings.constructor;
            let prototype = heap.known().strings.prototype;
            let defined = Object::define_own_property(
                heap,
                scope,
                proto,
                constructor,
                PropertyDescriptor::Data {
                    value: function.value(),
                    writable: true,
                    enumerable: false,
                    configurable: true,
                },
            )?;
            if !defined {
                return Err(VmError::Type);
            }
            let defined = Object::define_own_property(
                heap,
                scope,
                function,
                prototype,
                PropertyDescriptor::Data {
                    value: proto.value(),
                    writable: true,
                    enumerable: false,
                    configurable: false,
                },
            )?;
            if !defined {
                return Err(VmError::Type);
            }
        }

        Ok(function.value())
    }

    /// ES 7.1.1 ToPrimitive. Primitives pass through untouched
    pub fn to_primitive(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        value: Value,
        hint: Hint,
    ) -> Result<Coercion, VmError> {
        if heap.no_gc(|nogc| Convert::is_primitive(nogc, value)) {
            return Ok(Coercion::Value(value));
        }
        let known = heap.known();
        let exception = known.exception.value();

        // 1. exotic @@toPrimitive (GetMethod)
        let exotic = Self::get_property(vm, heap, state, value, known.to_primitive_symbol.value())?;
        let exotic = match exotic {
            Coercion::Threw => return Ok(Coercion::Threw),
            Coercion::Value(v) => v,
        };
        if exotic != known.undefined.value() && exotic != known.null.value() {
            if !Self::is_callable(heap, exotic) {
                // GetMethod: a non-callable, non-nullish method is a TypeError
                return Err(VmError::Type);
            }
            let hint_string = match hint {
                Hint::Default => known.strings.default.value(),
                Hint::Number => known.strings.number.value(),
                Hint::String => known.strings.string.value(),
            };
            let args = [hint_string];
            // SAFETY: hint_string was interned immediately above; the snapshot
            // is consumed (staged/copied into the frame) before any GC
            let args = unsafe { GcSlice::from_slice(&args) };
            let result = NativeContext::new(vm, heap, state).call(exotic, args)?;
            if result == exception {
                return Ok(Coercion::Threw);
            }
            return if heap.no_gc(|nogc| Convert::is_primitive(nogc, result)) {
                Ok(Coercion::Value(result))
            } else {
                Err(VmError::Type)
            };
        }

        // 2. OrdinaryToPrimitive: hint string → toString first, else valueOf first
        let method_names: [Value; 2] = if hint == Hint::String {
            [
                known.strings.to_string.value(),
                known.strings.value_of.value(),
            ]
        } else {
            [
                known.strings.value_of.value(),
                known.strings.to_string.value(),
            ]
        };
        for name in method_names {
            let method = Self::get_property(vm, heap, state, value, name)?;
            let method = match method {
                Coercion::Threw => return Ok(Coercion::Threw),
                Coercion::Value(v) => v,
            };
            if !Self::is_callable(heap, method) {
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
            if heap.no_gc(|nogc| Convert::is_primitive(nogc, result)) {
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
        match Self::to_primitive(vm, heap, state, value, Hint::Number)? {
            Coercion::Threw => Ok(None),
            Coercion::Value(v) => Ok(Some(heap.no_gc(|nogc| Convert::to_number(nogc, v))?)),
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
        let a = Self::to_numeric(vm, heap, state, a)?;
        let Some(a) = a else { return Ok(None) };
        let b = Self::to_numeric(vm, heap, state, b)?;
        let Some(b) = b else { return Ok(None) };
        let r = op(a, b);
        Ok(Some(
            state.handle_scope(|scope| Convert::to_value(heap, &scope, r)),
        ))
    }

    /// Get a property value with full [[Get]] semantics: accessor getters are
    /// called (nested run), missing properties yield undefined.
    pub fn get_property(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        receiver: Value,
        name: Value,
    ) -> Result<Coercion, VmError> {
        let outcome =
            heap.no_gc(|nogc| load_outcome(nogc, receiver, SlotName::from_value(name)))?;
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

    pub fn is_callable(heap: &mut Heap, v: Value) -> bool {
        heap.no_gc(|nogc| {
            let ValueRef::Object(obj) = v.value_ref(nogc) else {
                return false;
            };
            obj.as_ref().header.map.heap_ref(nogc).kind().is_callable()
        })
    }

    /// ES 13.5.3 typeof: the well-known type string for a value. `null`
    /// reports `"object"`; callables report `"function"`.
    pub fn type_of(heap: &mut Heap, v: Value) -> Value {
        heap.no_gc(|nogc| {
            let strings = nogc.known().strings;
            if v.is_smi() || v.get_as::<Float>(nogc).is_some() {
                strings.number.value()
            } else if v == nogc.known().undefined.value() || v == nogc.known().void.value() {
                strings.undefined.value()
            } else if v == nogc.known().null.value() {
                strings.object.value()
            } else if v == nogc.known().true_object.value()
                || v == nogc.known().false_object.value()
            {
                strings.boolean.value()
            } else if v.get_as::<VMString>(nogc).is_some() {
                strings.string.value()
            } else if v.get_as::<Symbol>(nogc).is_some() {
                strings.symbol.value()
            } else if let ValueRef::Object(obj) = v.value_ref(nogc) {
                if obj.as_ref().header.map.heap_ref(nogc).kind().is_callable() {
                    strings.function.value()
                } else {
                    strings.object.value()
                }
            } else {
                strings.object.value()
            }
        })
    }

    /// ES 13.10.2 instanceof / 7.3.20 OrdinaryHasInstance: `Get(C, "prototype")`
    /// must yield an object (else TypeError), then walk the object's prototype
    /// chain for it. `None` means user code threw.
    pub fn instance_of(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        object: Value,
        callable: Value,
    ) -> Result<Option<bool>, VmError> {
        if !Self::is_callable(heap, callable) {
            return Err(VmError::Type);
        }
        // 4. P = Get(C, "prototype") — full [[Get]], getters may run user code
        let proto_name = heap.known().strings.prototype.value();
        let proto = Self::get_property(vm, heap, state, callable, proto_name)?;
        let proto = match proto {
            Coercion::Threw => return Ok(None),
            Coercion::Value(v) => v,
        };
        // 5. P must be an object
        if heap.no_gc(|nogc| Convert::is_primitive(nogc, proto)) {
            return Err(VmError::Type);
        }
        Ok(Some(heap.no_gc(|nogc| {
            Self::has_proto_in_chain(nogc, object, proto)
        })))
    }

    /// OrdinaryHasInstance step 6: walk the prototype chain of `object`.
    pub fn has_proto_in_chain<'a>(nogc: &'a NoGc<'a>, object: Value, target: Value) -> bool {
        let ValueRef::Object(obj) = object.value_ref(nogc) else {
            return false;
        };
        let proto = obj.as_ref().header.map.heap_ref(nogc).prototype.inner();
        if proto == target {
            return true;
        }
        if proto == nogc.known().null.value() {
            return false;
        }
        if let Some(parents) = proto.get_as::<FixedArray>(nogc) {
            for i in 0..parents.len() {
                if Self::has_proto_in_chain(nogc, parents.at(i), target) {
                    return true;
                }
            }
            return false;
        }
        Self::has_proto_in_chain(nogc, proto, target)
    }

    /// The Construct receiver (ES 9.2.2 step 5): a fresh `{}` whose
    /// [[Prototype]] is `new.target.prototype` when that is an object, else
    /// the ordinary object prototype (GetPrototypeFromConstructor,
    /// ES 9.1.14). The callee is irrelevant here — `this` always comes from
    /// new.target. `None` means user code threw.
    pub fn create_construct_receiver(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        new_target: Handle<'_, Object>,
    ) -> Result<Option<Value>, VmError> {
        let proto_name = heap.known().strings.prototype.value();
        state.handle_scope(|scope| {
            let proto = Self::get_property(vm, heap, state, new_target.value(), proto_name)?;
            let proto = match proto {
                Coercion::Threw => return Ok(None),
                Coercion::Value(v) => v,
            };
            // root the prototype before allocating below (GC may move it)
            let proto = if heap.no_gc(|nogc| Convert::is_primitive(nogc, proto)) {
                None
            } else {
                Some(unsafe { scope.handle_value::<Object>(proto) })
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
                Object::set_prototype(heap, &scope, obj, proto.value())?;
            }
            Ok(Some(obj))
        })
    }
}
