use crate::{
    CallableInfoObject, Context, Convert, DenseString, FixedArray, Float, GcSlice, Handle,
    HandleScope, Heap, LoadOutcome, Object, PartialDescriptor, PropertyDescriptor, SlotName, Smi,
    Symbol, Tagged, Value, VmError, load_outcome_on,
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
    /// The returned closure is anchored at the `&mut Heap` borrow; callers
    /// store the word (or root it) before the next allocation.
    pub fn create_closure<'a>(
        heap: &'a mut Heap,
        scope: &HandleScope<'_>,
        info: Handle<'_, CallableInfoObject>,
        context: Tagged<'_, Value>,
    ) -> Result<Tagged<'a, Value>, VmError> {
        let (kind, source_name, formal_length) = heap.no_gc(|heap| {
            let info = info.heap_ref(heap);
            (
                info.function_kind(),
                info.name(heap).map(|name| name.erase()),
                info.formal_length(),
            )
        });
        let function_name = match source_name {
            // Safety: the word was read under an anchor one statement ago;
            // no allocation has run since.
            Some(name) => scope.handle(unsafe { name.assume_valid(&*heap) }),
            None => heap.no_gc(|heap| {
                scope.handle(heap.known().strings.empty.as_tagged(heap).erase_type())
            }),
        };
        let map = match kind {
            kind if kind.is_class_constructor() => heap.known().class_constructor_map,
            kind if kind.is_constructible() => heap.known().function_map,
            _ => heap.known().non_constructor_function_map,
        };
        let context = scope.cast::<Context>(context).ok_or(VmError::Type)?;
        // class constructors carry a third hidden slot: the instance-field
        // array ([key0, init0, ...]); undefined until SetClassFields
        let function = if kind.is_class_constructor() {
            let values = heap.no_gc(|heap| {
                [
                    info.as_tagged(heap).erase(),
                    context.as_tagged(heap).erase(),
                    heap.known().undefined.as_tagged(heap).erase(),
                ]
            });
            heap.new_object(scope, map, scope.stage_words(&values))
                .into_handle(scope)
        } else {
            let values = heap.no_gc(|heap| {
                [
                    info.as_tagged(heap).erase(),
                    context.as_tagged(heap).erase(),
                ]
            });
            heap.new_object(scope, map, scope.stage_words(&values))
                .into_handle(scope)
        };

        let length_key = heap.known().strings.length;
        let name_key = heap.known().strings.name;
        let defined = Object::define_own_property(
            heap,
            scope,
            function,
            length_key,
            PropertyDescriptor::Data {
                value: Smi::new(formal_length as i64).encode(),
                writable: false,
                enumerable: false,
                configurable: true,
            },
        )?;
        if !defined {
            return Err(VmError::Type);
        }
        // fresh word out of the rooted slot, stored below before any
        // allocation completes (the define path roots its inputs first)
        let function_name_word = function_name.as_tagged(&*heap).erase();
        let defined = Object::define_own_property(
            heap,
            scope,
            function,
            name_key,
            PropertyDescriptor::Data {
                value: function_name_word,
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
                .new_object(scope, heap.known().object_initial_map, GcSlice::EMPTY)
                .into_handle(scope);
            let constructor = heap.known().strings.constructor;
            let prototype = heap.known().strings.prototype;
            let function_word = function.as_tagged(&*heap).erase();
            let defined = Object::define_own_property(
                heap,
                scope,
                proto,
                constructor,
                PropertyDescriptor::Data {
                    value: function_word,
                    writable: true,
                    enumerable: false,
                    configurable: true,
                },
            )?;
            if !defined {
                return Err(VmError::Type);
            }
            let proto_word = proto.as_tagged(&*heap).erase();
            let defined = Object::define_own_property(
                heap,
                scope,
                function,
                prototype,
                PropertyDescriptor::Data {
                    value: proto_word,
                    writable: true,
                    enumerable: false,
                    configurable: false,
                },
            )?;
            if !defined {
                return Err(VmError::Type);
            }
        }

        // fresh word under a final shared reborrow; anchored at the `&mut`
        // borrow, so callers must store or root it before allocating
        Ok(function.as_tagged(&*heap).erase_type())
    }

    /// ES 7.1.1 ToPrimitive. Primitives pass through untouched
    pub fn to_primitive(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        value: Value,
        hint: Hint,
    ) -> Result<Coercion, VmError> {
        if heap.no_gc(|heap| Convert::is_primitive(heap, unsafe { value.assume_valid(heap) })) {
            return Ok(Coercion::Value(value));
        }
        // the receiver stays rooted throughout: method lookups and calls
        // below run user code (getters, valueOf/toString), which allocates
        // and would leave a raw copy dangling
        state.handle_scope(|scope| {
            // Safety: caller-supplied word, fresh at entry.
            let value = scope.handle(unsafe { value.assume_valid(&*heap) });
            let (exception, to_primitive_symbol, undefined, null) = heap.no_gc(|heap| {
                let known = heap.known();
                (
                    known.exception.as_tagged(heap).erase(),
                    known.to_primitive_symbol.as_tagged(heap).erase(),
                    known.undefined.as_tagged(heap).erase(),
                    known.null.as_tagged(heap).erase(),
                )
            });

            // 1. exotic @@toPrimitive (GetMethod)
            let receiver_word = value.as_tagged(&*heap).erase();
            let exotic = Self::get_property(vm, heap, state, receiver_word, to_primitive_symbol)?;
            let exotic = match exotic {
                Coercion::Threw => return Ok(Coercion::Threw),
                // Safety: fresh word from the call, no allocation since.
                Coercion::Value(v) => scope.handle(unsafe { v.assume_valid(&*heap) }),
            };
            let exotic_word = exotic.as_tagged(&*heap).erase();
            if exotic_word != undefined && exotic_word != null {
                if !Self::is_callable(heap, exotic_word) {
                    // GetMethod: a non-callable, non-nullish method is a TypeError
                    return Err(VmError::Type);
                }
                let hint_string = heap.no_gc(|heap| {
                    let s = heap.known().strings;
                    match hint {
                        Hint::Default => s.default.as_tagged(heap).erase(),
                        Hint::Number => s.number.as_tagged(heap).erase(),
                        Hint::String => s.string.as_tagged(heap).erase(),
                    }
                });
                let result = state.handle_scope(|scope| {
                    NativeContext::new(vm, heap, state).call(
                        // Safety: word freshly read from a rooted slot, no
                        // allocation since.
                        unsafe { Tagged::from_value_unchecked(exotic_word) },
                        scope.stage_words(&[hint_string]),
                    )
                })?;
                if result == exception {
                    return Ok(Coercion::Threw);
                }
                return if heap
                    .no_gc(|heap| Convert::is_primitive(heap, unsafe { result.assume_valid(heap) }))
                {
                    Ok(Coercion::Value(result))
                } else {
                    Err(VmError::Type)
                };
            }

            // 2. OrdinaryToPrimitive: hint string → toString first, else valueOf first
            let method_names: [Value; 2] = heap.no_gc(|heap| {
                let s = heap.known().strings;
                if hint == Hint::String {
                    [
                        s.to_string.as_tagged(heap).erase(),
                        s.value_of.as_tagged(heap).erase(),
                    ]
                } else {
                    [
                        s.value_of.as_tagged(heap).erase(),
                        s.to_string.as_tagged(heap).erase(),
                    ]
                }
            });
            for name in method_names {
                let receiver_word = value.as_tagged(&*heap).erase();
                let method = Self::get_property(vm, heap, state, receiver_word, name)?;
                let method = match method {
                    Coercion::Threw => return Ok(Coercion::Threw),
                    // Safety: fresh word from the call, no allocation since.
                    Coercion::Value(v) => scope.handle(unsafe { v.assume_valid(&*heap) }),
                };
                let method_word = method.as_tagged(&*heap).erase();
                if !Self::is_callable(heap, method_word) {
                    continue;
                }
                let receiver = value.as_tagged(&*heap).erase();
                let result = state.handle_scope(|scope| {
                    NativeContext::new(vm, heap, state).call(
                        // Safety: word freshly read from a rooted slot, no
                        // allocation since.
                        unsafe { Tagged::from_value_unchecked(method_word) },
                        scope.stage_words(&[receiver]),
                    )
                })?;
                if result == exception {
                    return Ok(Coercion::Threw);
                }
                if heap
                    .no_gc(|heap| Convert::is_primitive(heap, unsafe { result.assume_valid(heap) }))
                {
                    return Ok(Coercion::Value(result));
                }
                // object result: try the next method name
            }
            Err(VmError::Type)
        })
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
            // Safety: fresh word from the coercion, consumed here.
            Coercion::Value(v) => {
                Ok(Some(heap.no_gc(|heap| {
                    Convert::to_number(heap, unsafe { v.assume_valid(heap) })
                })?))
            }
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
        // to_numeric runs user code (valueOf): the second operand must
        // stay rooted across the first one's coercion
        state.handle_scope(|scope| {
            // Safety: caller-supplied word, fresh at entry.
            let b = scope.handle(unsafe { b.assume_valid(&*heap) });
            let a = Self::to_numeric(vm, heap, state, a)?;
            let Some(a) = a else { return Ok(None) };
            let b_word = b.as_tagged(&*heap).erase();
            let b = Self::to_numeric(vm, heap, state, b_word)?;
            let Some(b) = b else { return Ok(None) };
            let r = op(a, b);
            let v = heap.new_number(&scope, r);
            Ok(Some(v.erase()))
        })
    }

    /// Get a property value with full [[Get]] semantics: accessor getters are
    /// called (nested run), missing properties yield undefined. Proxy
    /// receivers run their `get` trap (ES 20.2.5.8).
    pub fn get_property(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        receiver: Value,
        name: Value,
    ) -> Result<Coercion, VmError> {
        Self::get_property_on(vm, heap, state, receiver, receiver, name)
    }

    /// Same, with the lookup start (`holder`) split from the getter
    /// receiver — the proxy forward shape: lookup on the target,
    /// `this` = the proxy.
    pub fn get_property_on(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        holder: Value,
        receiver: Value,
        name: Value,
    ) -> Result<Coercion, VmError> {
        if heap.no_gc(|heap| crate::proxy::is_proxy(heap, unsafe { holder.assume_valid(heap) })) {
            return crate::proxy::get(
                vm,
                heap,
                state,
                // Safety: caller-supplied words, fresh at entry.
                unsafe { Tagged::from_value_unchecked(holder) },
                unsafe { Tagged::from_value_unchecked(receiver) },
                unsafe { Tagged::from_value_unchecked(name) },
            );
        }
        // direct anchor: the outcome is consumed (erased / rooted) before
        // any allocation below runs
        let (outcome, exception) = {
            let heap = &*heap;
            let exception = heap.known().exception.as_tagged(heap).erase();
            let outcome = load_outcome_on(
                heap,
                unsafe { holder.assume_valid(heap) },
                SlotName::from_value(name),
            )?;
            (outcome, exception)
        };
        match outcome {
            LoadOutcome::Value(v) => Ok(Coercion::Value(v.erase())),
            LoadOutcome::Getter(getter) => {
                // erase before the first allocation (the nested call)
                let getter = getter.erase();
                let result = state.handle_scope(|scope| {
                    NativeContext::new(vm, heap, state).call(
                        // Safety: freshly erased anchored read, no
                        // allocation since.
                        unsafe { Tagged::from_value_unchecked(getter) },
                        scope.stage_words(&[receiver]),
                    )
                })?;
                if result == exception {
                    Ok(Coercion::Threw)
                } else {
                    Ok(Coercion::Value(result))
                }
            }
        }
    }

    pub fn to_property_descriptor(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        attrs: Value,
    ) -> Result<Option<PartialDescriptor>, VmError> {
        if heap.no_gc(|heap| Convert::is_primitive(heap, unsafe { attrs.assume_valid(heap) })) {
            return Err(VmError::Type);
        }
        state.handle_scope(|scope| {
            // Safety: caller-supplied word, fresh at entry.
            let attrs = scope.handle(unsafe { attrs.assume_valid(&*heap) });
            let names = heap.no_gc(|heap| {
                let s = heap.known().strings;
                [
                    s.value,
                    s.get,
                    s.set,
                    s.writable,
                    s.enumerable,
                    s.configurable,
                ]
                .map(|n| n.as_tagged(heap).erase())
            });
            let mut reads: Vec<Handle<'_, Value>> = Vec::new();
            for name in names {
                // re-read per iteration: the getters below may run user
                // code (and move the receiver)
                let attrs_word = attrs.as_tagged(&*heap).erase();
                match Self::get_property(vm, heap, state, attrs_word, name)? {
                    Coercion::Threw => return Ok(None),
                    // Safety: fresh word from the call, no allocation since.
                    Coercion::Value(v) => {
                        reads.push(scope.handle(unsafe { v.assume_valid(&*heap) }))
                    }
                }
            }
            // fresh words out of rooted slots, consumed within these
            // statements (no allocation in between)
            let (value_w, get_w, set_w, writable_w, enumerable_w, configurable_w) =
                heap.no_gc(|heap| {
                    (
                        reads[0].as_tagged(heap).erase(),
                        reads[1].as_tagged(heap).erase(),
                        reads[2].as_tagged(heap).erase(),
                        reads[3].as_tagged(heap).erase(),
                        reads[4].as_tagged(heap).erase(),
                        reads[5].as_tagged(heap).erase(),
                    )
                });
            let undefined = heap.no_gc(|heap| heap.known().undefined.as_tagged(heap).erase());
            let value = (value_w != undefined).then_some(value_w);
            let get = (get_w != undefined).then_some(get_w);
            let set = (set_w != undefined).then_some(set_w);
            // accessor halves must be callable or undefined
            if get.is_some() || set.is_some() {
                for half in [get, set] {
                    if let Some(h) = half
                        && h != undefined
                        && !Self::is_callable(heap, h)
                    {
                        return Err(VmError::Type);
                    }
                }
            }
            Ok(Some(PartialDescriptor {
                value,
                get,
                set,
                writable: (writable_w != undefined).then(|| {
                    heap.no_gc(|heap| {
                        Convert::is_truthy(heap, unsafe { writable_w.assume_valid(heap) })
                    })
                }),
                enumerable: (enumerable_w != undefined).then(|| {
                    heap.no_gc(|heap| {
                        Convert::is_truthy(heap, unsafe { enumerable_w.assume_valid(heap) })
                    })
                }),
                configurable: (configurable_w != undefined).then(|| {
                    heap.no_gc(|heap| {
                        Convert::is_truthy(heap, unsafe { configurable_w.assume_valid(heap) })
                    })
                }),
            }))
        })
    }

    pub fn is_callable(heap: &Heap, v: Value) -> bool {
        // Safety: caller-supplied word, fresh at entry.
        let Some(obj) = unsafe { v.assume_valid(heap) }.as_heap_object() else {
            return false;
        };
        obj.as_ref().header.map.heap_ref(heap).kind().is_callable()
    }

    pub fn to_property_key(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        v: Value,
    ) -> Result<Option<Value>, VmError> {
        if v.is_smi()
            || heap.no_gc(|heap| unsafe { v.assume_valid(heap) }.get_as::<Symbol>().is_some())
        {
            return Ok(Some(v));
        }
        let is_string = heap.no_gc(|heap| {
            unsafe { v.assume_valid(heap) }
                .get_as::<DenseString>()
                .is_some()
        });
        let primitive = if is_string {
            v
        } else {
            match Self::to_primitive(vm, heap, state, v, Hint::String)? {
                Coercion::Threw => return Ok(None),
                Coercion::Value(p) => p,
            }
        };
        // named lookup compares interned strings by pointer: canonicalize
        // exactly once here, then every downstream bits-compare is sound.
        // to_string allocates (Float → string, ToString of objects), so
        // its result must be rooted and re-cast inside one scope — a raw
        // copy crossing the scope boundary goes stale
        let interned = state.handle_scope(|scope| {
            let stringified = Convert::to_string(
                heap,
                &scope,
                // Safety: caller-supplied / freshly coerced word.
                unsafe { Tagged::from_value_unchecked(primitive) },
            )?;
            match scope.cast::<DenseString>(stringified) {
                Some(s) => Ok(vm
                    .interner()
                    .intern_value(heap, &scope, &s)
                    .as_tagged(&*heap)
                    .erase()),
                // ToString of a symbol primitive throws (ES 6.1.7.1)
                None => Err(VmError::Type),
            }
        })?;
        Ok(Some(interned))
    }

    /// ES 13.5.3 typeof: the well-known type string for a value. `null`
    /// reports `"object"`; callables report `"function"`.
    pub fn type_of<'a>(heap: &'a Heap, v: Tagged<'a, Value>) -> Tagged<'a, Value> {
        let strings = heap.known().strings;
        if v.is_smi() || v.get_as::<Float>().is_some() {
            strings.number.as_tagged(heap).erase_type()
        } else if v.erase() == heap.known().undefined.as_tagged(heap).erase()
            || v.erase() == heap.known().the_hole.as_tagged(heap).erase()
        {
            strings.undefined.as_tagged(heap).erase_type()
        } else if v.erase() == heap.known().null.as_tagged(heap).erase() {
            strings.object.as_tagged(heap).erase_type()
        } else if v.erase() == heap.known().true_object.as_tagged(heap).erase()
            || v.erase() == heap.known().false_object.as_tagged(heap).erase()
        {
            strings.boolean.as_tagged(heap).erase_type()
        } else if v.get_as::<DenseString>().is_some() {
            strings.string.as_tagged(heap).erase_type()
        } else if v.get_as::<Symbol>().is_some() {
            strings.symbol.as_tagged(heap).erase_type()
        } else if let Some(obj) = v.as_heap_object() {
            if obj.as_ref().header.map.heap_ref(heap).kind().is_callable() {
                strings.function.as_tagged(heap).erase_type()
            } else {
                strings.object.as_tagged(heap).erase_type()
            }
        } else {
            strings.object.as_tagged(heap).erase_type()
        }
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
        // 4. P = Get(C, "prototype") — full [[Get]], getters may run user
        // code: the object must stay rooted across it
        state.handle_scope(|scope| {
            // Safety: caller-supplied word, fresh at entry.
            let object = scope.handle(unsafe { object.assume_valid(&*heap) });
            let proto_name =
                heap.no_gc(|heap| heap.known().strings.prototype.as_tagged(heap).erase());
            let proto = Self::get_property(vm, heap, state, callable, proto_name)?;
            let proto = match proto {
                Coercion::Threw => return Ok(None),
                Coercion::Value(v) => v,
            };
            // 5. P must be an object
            // Safety: fresh word from the call, consumed here.
            if heap.no_gc(|heap| Convert::is_primitive(heap, unsafe { proto.assume_valid(heap) })) {
                return Err(VmError::Type);
            }
            Ok(Some(heap.no_gc(|heap| {
                // Safety: fresh word from the call, consumed here.
                Self::has_proto_in_chain(heap, object.as_tagged(heap), unsafe {
                    proto.assume_valid(heap)
                })
            })))
        })
    }

    /// OrdinaryHasInstance step 6: walk the prototype chain of `object`.
    pub fn has_proto_in_chain<'a>(
        heap: &'a Heap,
        object: Tagged<'a, Value>,
        target: Tagged<'a, Value>,
    ) -> bool {
        let Some(obj) = object.as_heap_object() else {
            return false;
        };
        let proto = obj.as_ref().header.map.heap_ref(heap).prototype.get(heap);
        if proto.ptr_eq(target) {
            return true;
        }
        if proto.erase() == heap.known().null.as_tagged(heap).erase() {
            return false;
        }
        if let Some(parents) = proto.get_as::<FixedArray>() {
            for i in 0..parents.len() {
                if Self::has_proto_in_chain(heap, parents.at(heap, i), target) {
                    return true;
                }
            }
            return false;
        }
        Self::has_proto_in_chain(heap, proto, target)
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
        let new_target = new_target.as_tagged(&*heap).erase();
        Self::create_construct_receiver_value(vm, heap, state, new_target)
    }

    /// Same, for `new.target` values that may be exotic (a constructor
    /// proxy): only `Get(new.target, "prototype")` is observed.
    pub fn create_construct_receiver_value(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        new_target: Value,
    ) -> Result<Option<Value>, VmError> {
        let proto_name = heap.no_gc(|heap| heap.known().strings.prototype.as_tagged(heap).erase());
        state.handle_scope(|scope| {
            let proto = Self::get_property(vm, heap, state, new_target, proto_name)?;
            let proto = match proto {
                Coercion::Threw => return Ok(None),
                Coercion::Value(v) => v,
            };
            // root the prototype before allocating below (GC may move it)
            // non-object prototypes fall back to the ordinary prototype
            // Safety: fresh word from the call, no allocation since.
            let proto = scope.cast::<Object>(unsafe { proto.assume_valid(&*heap) });
            let known = heap.known();
            let obj = heap
                .new_object(&scope, known.object_initial_map, GcSlice::EMPTY)
                .into_handle(&scope);
            if let Some(proto) = proto {
                // fresh words under a shared reborrow, stored by the call
                // before any allocation completes
                let proto_word = proto.as_tagged(&*heap).erase();
                let obj_word = obj.as_tagged(&*heap).erase();
                Object::set_prototype(heap, &scope, obj_word, proto_word)?;
            }
            // fresh word at return: callers store it to a register (rooted
            // memory) immediately
            Ok(Some(obj.as_tagged(&*heap).erase()))
        })
    }
}
