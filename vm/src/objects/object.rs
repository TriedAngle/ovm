use core::alloc::Layout;

use crate::{
    CallableInfoObject, Coercion, Context, ContextState, Convert, DenseString, EdgeVisitable,
    FixedArray, Float, FunctionKind, GcSlot, Handle, HandleScope, HandleSlice, Header, Heap,
    HeapObject, Hint, Lookup, Map, ObjectKind, PropertyDescriptor, RuntimeContext, SlotName, Smi,
    Symbol, Tagged, VM, Value, Visitor, VmError,
};

#[repr(C)]
pub struct Object {
    pub header: Header,
    pub slots: GcSlot<FixedArray>,
    pub elements: GcSlot,
    pub length: GcSlot<Smi>,
}

impl Object {
    pub fn layout_for() -> Layout {
        Layout::new::<Self>()
    }

    pub fn callable_info<'a>(&'a self, heap: &'a Heap) -> Option<Tagged<'a, CallableInfoObject>> {
        if !self.header.map.get(heap).kind().is_callable() {
            return None;
        }
        let info = self.slots.get(heap).at(heap, 0);
        info.get_as::<CallableInfoObject>()
    }

    pub fn closure_context<'a>(&'a self, heap: &'a Heap) -> Option<Tagged<'a, Context>> {
        if !self.header.map.get(heap).kind().is_callable() {
            return None;
        }
        self.slots.get(heap).at(heap, 1).get_as::<Context>()
    }

    /// The `idx`-th entry of the callable's constant pool, as a slot name.
    pub fn constant_slot_name<'a>(&'a self, heap: &'a Heap, idx: usize) -> Tagged<'a, SlotName> {
        self.callable_info(heap)
            .expect("callable must have callable info")
            .constant_slot_name(heap, idx)
    }

    pub fn runtime_index<'a>(&'a self, heap: &'a Heap) -> Option<usize> {
        if !self.header.map.get(heap).kind().is_runtime() {
            return None;
        }
        let idx = Smi::decode(self.slots.get(heap).at(heap, 0).raw())?.value();
        usize::try_from(idx).ok()
    }

    pub fn is_array<'a>(&'a self, heap: &'a Heap) -> bool {
        self.header.map.get(heap).kind().kind() == ObjectKind::Array
    }

    /// The JSArray `length` internal slot, when `self` is an array named
    /// `name`: it lives outside the map descriptors, so descriptor walks
    /// must consult this first. `None` for any other name or non-array.
    pub fn array_length<'a>(
        &'a self,
        heap: &'a Heap,
        name: Tagged<'a, SlotName>,
    ) -> Option<Tagged<'a, Value>> {
        if !self.is_array(heap) {
            return None;
        }
        let s = name.erase().get_as::<DenseString>()?.as_ref();
        s.data(heap)
            .matches_ascii(b"length")
            .then(|| self.length.get(heap).erase())
    }

    /// The object's map (shape).
    pub fn map_ref<'a>(&self, heap: &'a Heap) -> Tagged<'a, Map> {
        self.header.map.get(heap)
    }

    /// Whether the object's map allows adding new properties.
    pub fn is_extendable(&self, heap: &Heap) -> bool {
        self.map_ref(heap).kind().is_extendable()
    }

    /// The slot holding the value of the data slot at `offset`.
    pub fn slot<'a>(&self, heap: &'a Heap, offset: usize) -> &'a GcSlot {
        self.slots.get(heap).as_ref().element_slot(offset)
    }

    pub fn length(&self) -> usize {
        self.length.to_smi().value() as usize
    }

    pub fn elements_array<'a>(&'a self, heap: &'a Heap) -> Option<Tagged<'a, FixedArray>> {
        self.elements.get(heap).get_as::<FixedArray>()
    }

    /// Fast element read for array objects: `None` if `self` is not an
    /// array, the index is past the end, or the slot is a hole — the
    /// caller must fall back to a named property lookup.
    pub fn element_value<'a>(&'a self, heap: &'a Heap, i: usize) -> Option<Tagged<'a, Value>> {
        if !self.is_array(heap) || i >= self.length() {
            return None;
        }
        let elements = self.elements_array(heap)?;
        if i >= elements.len() {
            // `length` can exceed the backing store: those indices are holes
            return None;
        }
        let v = elements.at(heap, i);
        if v == heap.known().the_hole.as_tagged(heap) {
            return None;
        }
        Some(v)
    }
}

/// What kind of callable a value refers to.
pub enum CallTarget<'a> {
    Bytecode(Tagged<'a, Object>, usize, FunctionKind),
    Runtime(usize),
}

impl Object {
    /// Classify a value as a callable: a bytecode function (with register
    /// count and kind) or a runtime (with registry index).
    pub fn call_target<'a>(heap: &'a Heap, f: Tagged<'a, Value>) -> Option<CallTarget<'a>> {
        let obj = f.as_heap_object()?;
        let kind = obj.as_ref().header.map.get(heap).kind();
        if !kind.is_callable() {
            return None;
        }
        if kind.is_runtime() {
            return Some(CallTarget::Runtime(obj.as_ref().runtime_index(heap)?));
        }
        let info = obj.as_ref().callable_info(heap)?;
        let register_count = info.register_count.to_smi().value() as usize;
        Some(CallTarget::Bytecode(
            obj,
            register_count,
            info.function_kind(),
        ))
    }

    /// Store `value` at element index `i` of an array object, growing the
    /// elements backing store and updating `length` when `i` is past the end.
    /// Both arguments are rooted handles, so the grow path may allocate.
    pub fn store_array_element(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: &Handle<'_, Object>,
        i: usize,
        value: &Handle<'_, Value>,
    ) -> Result<(), VmError> {
        let new_len = i.checked_add(1).ok_or(VmError::OutOfBounds)?;

        let grows = {
            let obj = receiver.as_tagged(heap);
            if !obj.as_ref().is_array(heap) {
                return Err(VmError::Type);
            }
            // grow only when the store index is past the physical backing
            // store (its capacity), not the logical length: sequential appends
            // with headroom must not reallocate every time
            let capacity = obj
                .as_ref()
                .elements_array(heap)
                .map(|e| e.len())
                .unwrap_or(0);
            i >= capacity
        };

        if grows {
            let staged = {
                let heap_ref: &Heap = heap;
                let obj = receiver.as_tagged(heap_ref);
                let elements = obj.as_ref().elements_array(heap_ref).ok_or(VmError::Type)?;
                let keep = obj.as_ref().length().min(elements.len());
                let capacity = (new_len + (new_len >> 1) + 16).max(elements.len());
                let mut values: Vec<Tagged<'_, Value>> = Vec::with_capacity(capacity);
                for k in 0..keep {
                    values.push(elements.at(heap_ref, k));
                }
                values.resize(
                    capacity,
                    heap_ref.known().the_hole.as_tagged(heap_ref).erase(),
                );
                values[i] = value.as_tagged(heap_ref).erase();
                scope.stage(&values)
            };
            let elements = heap.allocate_handle::<FixedArray>(staged, scope);
            let obj = receiver.as_tagged(heap);
            obj.elements
                .set(heap, obj.erase(), elements.as_tagged(heap).erase());
            obj.length.set(heap, obj.erase(), Smi::new(new_len as i64));
        } else {
            let obj = receiver.as_tagged(heap);
            let elements = obj.as_ref().elements_array(heap).ok_or(VmError::Type)?;
            elements.set(heap, i, value.as_tagged(heap));
            // a store inside the physical capacity but past the logical
            // length still extends the array
            if i >= obj.as_ref().length() {
                obj.length.set(heap, obj.erase(), Smi::new(new_len as i64));
            }
        }
        Ok(())
    }

    pub fn store_array_element_in_place(
        heap: &mut Heap,
        receiver: &Handle<'_, Object>,
        i: usize,
        value: &Handle<'_, Value>,
    ) -> Result<(), VmError> {
        let obj = receiver.as_tagged(heap);
        if !obj.as_ref().is_array(heap) {
            return Err(VmError::Type);
        }
        if obj.as_ref().element_value(heap, i).is_none() {
            return Err(VmError::OutOfBounds);
        }
        let elements = obj.as_ref().elements_array(heap).ok_or(VmError::Type)?;
        elements.set(heap, i, value.as_tagged(heap));
        Ok(())
    }
}

pub struct ObjectInit<'a> {
    pub map: Handle<'a, Map>,
    pub slots: Handle<'a, FixedArray>,
    pub elements: Handle<'a, Value>,
    pub length: usize,
}

pub struct ObjectSlotsInit<'m, 'v> {
    pub map: Handle<'m, Map>,
    pub values: HandleSlice<'v>,
    pub elements: Handle<'m, Value>,
    pub length: usize,
}

impl HeapObject for Object {
    const KIND: ObjectKind = ObjectKind::Object;

    fn matches_kind(kind: ObjectKind) -> bool {
        matches!(
            kind,
            ObjectKind::Object
                | ObjectKind::Array
                | ObjectKind::ByteArray
                | ObjectKind::String
                | ObjectKind::Oddball
        )
    }
    type Init<'a> = ObjectInit<'a>;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Self::layout_for()
    }

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        let host = self.tagged(heap);
        self.header.map.set(heap, host, config.map.as_tagged(heap));
        self.slots.set(heap, host, config.slots.as_tagged(heap));
        self.elements
            .set(heap, host, config.elements.as_tagged(heap));
        self.length.set(heap, host, Smi::new(config.length as i64));
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for()
    }
}

impl EdgeVisitable for Object {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.slots.as_raw());
        visitor.visit(self.elements.as_raw());
    }
}

impl Object {
    pub fn create_closure<'a>(
        heap: &'a mut Heap,
        scope: &HandleScope<'_>,
        info: Handle<'_, CallableInfoObject>,
        context: Handle<'_, Context>,
    ) -> Result<Tagged<'a, Object>, VmError> {
        let info_ref = info.as_tagged(heap);
        let kind = info_ref.function_kind();
        let formal_length = info_ref.formal_length();
        let function_name = match info_ref.name(heap) {
            Some(name) => scope.handle(name),
            None => scope.handle(heap.known().strings.empty.as_tagged(heap).erase()),
        };
        let map = if kind.is_class_constructor() {
            heap.known().class_constructor_map
        } else if kind.is_constructible() {
            heap.known().function_map
        } else {
            heap.known().non_constructor_function_map
        };
        // class constructors carry a third hidden slot: the instance-field
        // array ([key0, init0, ...]); undefined until SetClassFields
        let args = if kind.is_class_constructor() {
            scope.stage(&[
                info.as_tagged(heap).erase(),
                context.as_tagged(heap).erase(),
                heap.known().undefined.as_tagged(heap).erase(),
            ])
        } else {
            scope.stage(&[
                info.as_tagged(heap).erase(),
                context.as_tagged(heap).erase(),
            ])
        };
        let function = heap.new_object(scope, map, args).as_handle(scope);

        let length_key = heap.known().strings.length;
        let name_key = heap.known().strings.name;
        if !Object::define_own_property(
            heap,
            scope,
            function,
            length_key,
            PropertyDescriptor::Data {
                value: scope.handle(Smi::new(formal_length as i64)),
                writable: false,
                enumerable: false,
                configurable: true,
            },
        )? {
            return Err(VmError::Type);
        }
        if !Object::define_own_property(
            heap,
            scope,
            function,
            name_key,
            PropertyDescriptor::Data {
                value: function_name.erase(),
                writable: false,
                enumerable: false,
                configurable: true,
            },
        )? {
            return Err(VmError::Type);
        }

        if kind.needs_prototype() {
            let proto = heap
                .new_object(scope, heap.known().object_initial_map, HandleSlice::EMPTY)
                .as_handle(scope);
            let constructor = heap.known().strings.constructor;
            let prototype = heap.known().strings.prototype;
            if !Object::define_own_property(
                heap,
                scope,
                proto,
                constructor,
                PropertyDescriptor::Data {
                    value: function.erase(),
                    writable: true,
                    enumerable: false,
                    configurable: true,
                },
            )? {
                return Err(VmError::Type);
            }
            if !Object::define_own_property(
                heap,
                scope,
                function,
                prototype,
                PropertyDescriptor::Data {
                    value: proto.erase(),
                    writable: true,
                    enumerable: false,
                    configurable: false,
                },
            )? {
                return Err(VmError::Type);
            }
        }

        Ok(function.as_tagged(heap))
    }

    /// ES 7.1.1 ToPrimitive. Primitives pass through untouched. `value` is a
    /// rooted handle; the result is anchored at the `&mut Heap` borrow, so the
    /// caller must consume or root it before the next allocation.
    pub fn to_primitive<'a>(
        vm: &'a VM,
        heap: &'a mut Heap,
        state: &'a ContextState,
        value: Handle<'_, Value>,
        hint: Hint,
    ) -> Result<Coercion<'a>, VmError> {
        if Convert::is_primitive(heap, value.as_tagged(heap)) {
            return Ok(Coercion::Value(value.as_tagged(heap)));
        }
        // the receiver stays rooted throughout: method lookups and calls
        // below run user code (getters, valueOf/toString), which allocates
        // and would leave a raw copy dangling
        RuntimeContext::new(vm, heap, state).handle_scope(|vm, heap, state, scope| {
            // 1. exotic @@toPrimitive (GetMethod)
            let exotic = match Lookup::get_property_on(
                vm,
                heap,
                state,
                value,
                value,
                heap.known().to_primitive_symbol.erase(),
            )? {
                Coercion::Threw => return Ok(Coercion::Threw),
                Coercion::Value(v) => scope.handle(v),
            };
            if exotic.as_tagged(heap) != heap.known().undefined.as_tagged(heap)
                && exotic.as_tagged(heap) != heap.known().null.as_tagged(heap)
            {
                if !Self::is_callable(heap, exotic.as_tagged(heap)) {
                    // GetMethod: a non-callable, non-nullish method is a TypeError
                    return Err(VmError::Type);
                }
                let hint_string = {
                    let s = heap.known().strings;
                    match hint {
                        Hint::Default => s.default.as_tagged(heap).erase(),
                        Hint::Number => s.number.as_tagged(heap).erase(),
                        Hint::String => s.string.as_tagged(heap).erase(),
                    }
                };
                let args = scope.stage(&[hint_string]);
                let result = scope.handle(RuntimeContext::call(
                    vm, &mut *heap, state, exotic, args, None,
                )?);
                if result.as_tagged(heap) == heap.known().exception.as_tagged(heap) {
                    return Ok(Coercion::Threw);
                }
                return if Convert::is_primitive(heap, result.as_tagged(heap)) {
                    Ok(Coercion::Value(result.as_tagged(heap)))
                } else {
                    Err(VmError::Type)
                };
            }

            // 2. OrdinaryToPrimitive: hint string → toString first, else valueOf first
            let method_names: [Handle<'_, Value>; 2] = {
                let s = heap.known().strings;
                if hint == Hint::String {
                    [s.to_string.erase(), s.value_of.erase()]
                } else {
                    [s.value_of.erase(), s.to_string.erase()]
                }
            };
            for name in method_names {
                let method = match Lookup::get_property_on(vm, heap, state, value, value, name)? {
                    Coercion::Threw => return Ok(Coercion::Threw),
                    Coercion::Value(v) => scope.handle(v),
                };
                if !Self::is_callable(heap, method.as_tagged(heap)) {
                    continue;
                }
                let args = scope.stage(&[value.as_tagged(heap)]);
                let result = scope.handle(RuntimeContext::call(
                    vm, &mut *heap, state, method, args, None,
                )?);
                if result.as_tagged(heap) == heap.known().exception.as_tagged(heap) {
                    return Ok(Coercion::Threw);
                }
                if Convert::is_primitive(heap, result.as_tagged(heap)) {
                    return Ok(Coercion::Value(result.as_tagged(heap)));
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
        value: Handle<'_, Value>,
    ) -> Result<Option<f64>, VmError> {
        state.handle_scope(|scope| {
            match Self::to_primitive(vm, heap, state, value, Hint::Number)? {
                Coercion::Threw => Ok(None),
                // root the anchored result to release the heap borrow
                Coercion::Value(v) => {
                    let v = scope.handle(v);
                    Ok(Some(Convert::to_number(heap, v.as_tagged(heap))?))
                }
            }
        })
    }

    pub fn numeric_op<'a>(
        vm: &'a VM,
        heap: &'a mut Heap,
        state: &'a ContextState,
        a: Handle<'_, Value>,
        b: Handle<'_, Value>,
        op: fn(f64, f64) -> f64,
    ) -> Result<Option<Tagged<'a, Value>>, VmError> {
        // to_numeric runs user code (valueOf): both operands are rooted by
        // the caller's handles, so the sibling survives the first coercion.
        let Some(a) = Self::to_numeric(vm, heap, state, a)? else {
            return Ok(None);
        };
        let Some(b) = Self::to_numeric(vm, heap, state, b)? else {
            return Ok(None);
        };
        Ok(Some(heap.new_number(op(a, b))))
    }

    pub fn is_callable<'a>(heap: &'a Heap, v: Tagged<'a, Value>) -> bool {
        let Some(obj) = v.as_heap_object() else {
            return false;
        };
        obj.as_ref().header.map.get(heap).kind().is_callable()
    }

    /// ES 7.1.18 ToPropertyKey: smis and symbols pass through, everything
    /// else is coerced to its interned canonical string. The returned
    /// name is anchored at the `&mut Heap` borrow; callers store the word
    /// (or root it) before the next allocation. `None` means user code
    /// threw (the pending exception holds it).
    pub fn to_property_key<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        v: Handle<'_, Value>,
    ) -> Result<Option<Tagged<'a, SlotName>>, VmError> {
        // Smis are pointer-free: valid at any lifetime, no rooting needed.
        if let Some(smi) = Smi::decode(v.as_tagged(heap).raw()) {
            return Ok(Some(Tagged::from(smi)));
        }
        state.handle_scope(|scope| {
            let cond_5 = v.as_tagged(heap).get_as::<Symbol>().is_some();
            if cond_5 {
                return Ok(Some(v.as_tagged(heap).as_name()));
            }
            let is_string = v.as_tagged(heap).get_as::<DenseString>().is_some();
            let primitive: Handle<'_, Value> = if is_string {
                v
            } else {
                match Self::to_primitive(vm, heap, state, v, Hint::String)? {
                    Coercion::Threw => return Ok(None),
                    Coercion::Value(p) => scope.handle(p),
                }
            };
            // Named lookup compares interned strings by pointer:
            // canonicalize exactly once here, then every downstream
            // bits-compare is sound.
            let stringified = Convert::to_string(heap, &scope, primitive)?;
            let interned = match scope.cast::<DenseString>(stringified) {
                Some(s) => vm.interner().intern_value(heap, &scope, &s),
                // ToString of a symbol primitive throws (ES 6.1.7.1)
                None => return Err(VmError::Type),
            };
            // fresh anchored re-read of the rooted interned word
            Ok(Some(interned.as_tagged(heap).into()))
        })
    }

    /// ES 13.5.3 typeof: the well-known type string for a value. `null`
    /// reports `"object"`; callables report `"function"`.
    pub fn type_of<'a>(heap: &'a Heap, v: Tagged<'a, Value>) -> Tagged<'a, Value> {
        let known = heap.known();
        let s = known.strings;
        // `null` needs no arm of its own: it is a non-callable object, and
        // so is anything that is not one of the primitive kinds above
        let type_name = if v.is_smi() || v.get_as::<Float>().is_some() {
            s.number
        } else if v == known.undefined.as_tagged(heap) || v == known.the_hole.as_tagged(heap) {
            s.undefined
        } else if v == known.true_object.as_tagged(heap) || v == known.false_object.as_tagged(heap)
        {
            s.boolean
        } else if v.get_as::<DenseString>().is_some() {
            s.string
        } else if v.get_as::<Symbol>().is_some() {
            s.symbol
        } else if Self::is_callable(heap, v) {
            s.function
        } else {
            s.object
        };
        type_name.as_tagged(heap).erase()
    }

    /// ES 13.10.2 instanceof / 7.3.20 OrdinaryHasInstance: `Get(C, "prototype")`
    /// must yield an object (else TypeError), then walk the object's prototype
    /// chain for it. `None` means user code threw.
    pub fn instance_of(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        object: Handle<'_, Value>,
        callable: Handle<'_, Value>,
    ) -> Result<Option<bool>, VmError> {
        if !Self::is_callable(heap, callable.as_tagged(heap)) {
            return Err(VmError::Type);
        }
        // 4. P = Get(C, "prototype") — full [[Get]]; a getter may run user
        // code and move the heap, so only its result has to be rooted here
        state.handle_scope(|scope| {
            let proto = match Lookup::get_property_on(
                vm,
                heap,
                state,
                callable,
                callable,
                heap.known().strings.prototype.erase(),
            )? {
                Coercion::Threw => return Ok(None),
                Coercion::Value(v) => scope.handle(v),
            };
            // 5. P must be an object
            if Convert::is_primitive(heap, proto.as_tagged(heap)) {
                return Err(VmError::Type);
            }
            Ok(Some(Self::has_proto_in_chain(
                heap,
                object.as_tagged(heap),
                proto.as_tagged(heap),
            )))
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
        let proto = obj.as_ref().header.map.get(heap).prototype.get(heap);
        if proto.ptr_eq(target) {
            return true;
        }
        if proto == heap.known().null.as_tagged(heap) {
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

    /// Same, for `new.target` values that may be exotic (a constructor
    /// proxy): only `Get(new.target, "prototype")` is observed.
    pub fn create_construct_receiver_value<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        new_target: Handle<'_, Value>,
    ) -> Result<Option<Tagged<'a, Value>>, VmError> {
        state.handle_scope(|scope| {
            let proto_name = scope.handle(heap.known().strings.prototype.as_tagged(heap).erase());
            let proto =
                Lookup::get_property_on(vm, heap, state, new_target, new_target, proto_name)?;
            let proto = match proto {
                Coercion::Threw => return Ok(None),
                Coercion::Value(v) => scope.handle(v),
            };
            // non-object prototypes fall back to the ordinary prototype
            let proto = scope.cast::<Object>(proto.as_tagged(heap));
            let known = heap.known();
            let obj = heap
                .new_object(&scope, known.object_initial_map, HandleSlice::EMPTY)
                .as_handle(&scope);
            if let Some(proto) = proto {
                Object::set_prototype(heap, &scope, obj, proto.erase())?;
            }
            Ok(Some(obj.as_tagged(heap).erase()))
        })
    }
}
