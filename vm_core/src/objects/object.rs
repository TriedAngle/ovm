use core::alloc::Layout;

use crate::{
    CallableInfoObject, Coercion, ContextObject, Convert, DenseString, EdgeVisitable, ElementsKind,
    FixedArray, Float, FunctionKind, GcSlot, Handle, HandleScope, HandleSlice, Header, Heap,
    HeapObject, Hint, HostCtx, Lookup, Map, NativeIndex, ObjectKind, PropertyDescriptor, Prototype,
    SiblingChange, SlotName, Smi, Symbol, Tagged, ThreadState, Transition, VM, Value, Visitor,
    VmError,
};

#[repr(C)]
pub struct Object {
    pub header: Header,
    pub slots: GcSlot<FixedArray>,
    pub elements: GcSlot<FixedArray>,
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
        // the callable map's slot 0 is always its CallableInfoObject
        Some(unsafe { info.cast() })
    }

    pub fn closure_context<'a>(&'a self, heap: &'a Heap) -> Option<Tagged<'a, ContextObject>> {
        if !self.header.map.get(heap).kind().is_callable() {
            return None;
        }
        self.slots
            .get(heap)
            .at(heap, 1)
            .get_as::<ContextObject>(heap)
    }

    /// The `idx`-th entry of the callable's constant pool, as a slot name.
    pub fn constant_slot_name<'a>(&'a self, heap: &'a Heap, idx: usize) -> Tagged<'a, SlotName> {
        self.callable_info(heap)
            .expect("callable must have callable info")
            .constant_slot_name(heap, idx)
    }

    #[inline]
    pub fn runtime_index<'a>(&'a self, heap: &'a Heap) -> Option<usize> {
        if !self.header.map.get(heap).kind().is_native() {
            return None;
        }
        let idx = Smi::decode(self.slots.get(heap).at(heap, 0).raw())?.value();
        usize::try_from(idx).ok()
    }

    /// The NATIVE-kind slot-0 Smi decoded to a call target half: the
    /// value is the callee's registry runtime index.
    #[inline]
    pub fn runtime_call_target<'a>(&'a self, heap: &'a Heap) -> Option<CallTarget<'a>> {
        if !self.header.map.get(heap).kind().is_native() {
            return None;
        }
        let idx = Smi::decode(self.slots.get(heap).at(heap, 0).raw())?.value();
        Some(CallTarget::Native(NativeIndex(usize::try_from(idx).ok()?)))
    }

    #[inline(always)]
    pub fn is_array<'a>(&'a self, heap: &'a Heap) -> bool {
        self.header.map.get(heap).kind().is_array()
    }

    /// The JSArray `length` internal slot, when `self` is an array named
    /// `name`: it lives outside the map descriptors, so descriptor walks
    /// must consult this first. `None` for any other name or non-array.
    #[inline]
    pub fn array_length<'a>(
        &'a self,
        heap: &'a Heap,
        name: Tagged<'a, SlotName>,
    ) -> Option<Tagged<'a, Value>> {
        if !self.is_array(heap) {
            return None;
        }
        name.erase()
            .ptr_eq(heap.known().strings.length.as_tagged(heap))
            .then(|| self.length.get(heap).erase())
    }

    /// ES 7.3.26 PrivateElementFind restricted to fields: an own data
    /// descriptor matching the private Symbol key (no prototype walk — private
    /// elements live only on the instance itself).
    pub fn private_find<'a>(
        heap: &'a Heap,
        obj: Tagged<'a, Value>,
        key: Tagged<'a, Value>,
    ) -> Option<&'a GcSlot> {
        let o = obj.as_heap_object()?;
        let map = o.as_ref().header.map.get(heap);
        for d in map.descriptors() {
            if d.name(heap).ptr_eq(key.as_name()) && !d.flags().is_accessor() {
                return Some(o.as_ref().slot(heap, d.offset()));
            }
        }
        None
    }

    /// The object's map (shape).
    #[inline]
    pub fn map_ref<'a>(&self, heap: &'a Heap) -> Tagged<'a, Map> {
        self.header.map.get(heap)
    }

    /// Whether the object's map allows adding new properties.
    pub fn is_extendable(&self, heap: &Heap) -> bool {
        self.map_ref(heap).kind().is_extendable()
    }

    /// The slot holding the value of the data slot at `offset`.
    #[inline]
    pub fn slot<'a>(&self, heap: &'a Heap, offset: usize) -> &'a GcSlot {
        self.slots.get(heap).as_ref().element_slot(offset)
    }

    #[inline(always)]
    pub fn length(&self) -> usize {
        self.length.to_smi_unchecked().value() as usize
    }

    #[inline(always)]
    pub fn elements_array<'a>(&'a self, heap: &'a Heap) -> Option<Tagged<'a, FixedArray>> {
        let elements = self.elements.get(heap);
        elements.is_strong_ptr().then_some(elements)
    }

    pub fn promote_holey(heap: &mut Heap, scope: &HandleScope<'_>, receiver: &Handle<'_, Object>) {
        let map = {
            let obj = receiver.as_tagged(heap);
            let map = obj.as_ref().map_ref(heap);
            // one-way lattice: only packed maps promote; holey and
            // dictionary maps are already past this rung
            if map.as_ref().kind().elements() != ElementsKind::Packed {
                return;
            }
            scope.handle(map)
        };
        // the map swap must invalidate chains keyed on the old map:
        // later shape changes fire on the new map lineage only
        Prototype::shape_changed(heap, map.as_tagged(heap));
        let target =
            Transition::sibling_target(heap, scope, |h| map.as_tagged(h), SiblingChange::Holey);
        let obj = receiver.as_tagged(heap);
        obj.as_ref()
            .header
            .map
            .set(heap, obj.erase(), target.as_tagged(heap));
    }

    #[inline]
    pub fn element_value<'a>(&'a self, heap: &'a Heap, i: usize) -> Option<Tagged<'a, Value>> {
        let kind = self.header.map.get(heap).as_ref().kind();
        if !kind.is_array() || !kind.is_dense_elements() || i >= self.length() {
            return None;
        }
        let elements = self.elements.get(heap);
        if kind.is_holey() {
            // `length` can exceed the backing store: those indices are holes
            if i >= elements.as_ref().len() {
                return None;
            }
            let v = elements.as_ref().at(heap, i);
            if v == heap.known().the_hole.as_tagged(heap) {
                return None;
            }
            Some(v)
        } else {
            // packed: capacity covers `length`, and no slot is a hole
            Some(elements.as_ref().at(heap, i))
        }
    }
}

pub enum CallTarget<'a> {
    Bytecode {
        target: Tagged<'a, Object>,
        info: Tagged<'a, CallableInfoObject>,
        context: Tagged<'a, ContextObject>,
        kind: FunctionKind,
    },
    Native(NativeIndex),
    /// A callable proxy: `[[Call]]` dispatches through the `apply` trap.
    Proxy(Tagged<'a, Object>),
}

impl Object {
    #[inline(always)]
    pub fn call_target<'a>(heap: &'a Heap, f: Tagged<'a, Value>) -> Option<CallTarget<'a>> {
        let obj = f.as_heap_object()?;
        let kind = obj.as_ref().header.map.get(heap).kind();
        if kind.is_proxy() {
            return Some(CallTarget::Proxy(obj));
        }
        if !kind.is_callable() {
            return None;
        }
        if kind.is_native() {
            return obj.as_ref().runtime_call_target(heap);
        }
        // every non-runtime callable object is laid out with
        // `[callable info, context]` in its first two slots
        let slots = obj.as_ref().slots.get(heap);
        let info = unsafe { slots.at(heap, 0).cast::<CallableInfoObject>() };
        let context = unsafe { slots.at(heap, 1).cast::<ContextObject>() };
        let descriptor = info.descriptor.to_smi_unchecked().value() as u64;
        let register_count = (descriptor & 0xffff) as usize;
        let formal_min = ((descriptor >> 16) & 0xffff) as usize;
        let kind = FunctionKind::decode(((descriptor >> 32) & 0xf) as i64);
        debug_assert_eq!(
            register_count,
            info.register_count.to_smi().value() as usize
        );
        debug_assert_eq!(formal_min, info.formal_parameter_count() + 1);
        debug_assert_eq!(kind, info.function_kind());
        Some(CallTarget::Bytecode {
            target: obj,
            info,
            context,
            kind,
        })
    }

    #[inline]
    pub fn bytecode_target<'a>(heap: &'a Heap, f: Tagged<'a, Value>) -> Option<CallTarget<'a>> {
        let obj = f.as_heap_object()?;
        let slots = obj.as_ref().slots.get(heap);
        let info = unsafe { slots.at(heap, 0).cast::<CallableInfoObject>() };
        let context = unsafe { slots.at(heap, 1).cast::<ContextObject>() };
        let descriptor = info.descriptor.to_smi_unchecked().value() as u64;
        Some(CallTarget::Bytecode {
            target: obj,
            info,
            context,
            kind: FunctionKind::decode(((descriptor >> 32) & 0xf) as i64),
        })
    }

    pub fn store_array_element(
        heap: &mut Heap,
        scope: &HandleScope<'_>,
        receiver: &Handle<'_, Object>,
        i: usize,
        value: &Handle<'_, Value>,
    ) -> Result<(), VmError> {
        let new_len = i.checked_add(1).ok_or(VmError::OutOfBounds)?;

        let (grows, old_len) = {
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
            (i >= capacity, obj.as_ref().length())
        };

        // writing past the end leaves holes behind: the map must stop
        // promising packed elements
        if i > old_len {
            Self::promote_holey(heap, scope, receiver);
        }

        if grows {
            let (capacity, keep) = {
                let heap_ref: &Heap = heap;
                let obj = receiver.as_tagged(heap_ref);
                let elements = obj.as_ref().elements_array(heap_ref).ok_or(VmError::Type)?;
                let keep = obj.as_ref().length().min(elements.len());
                let capacity = (new_len + (new_len >> 1) + 16).max(elements.len());
                (capacity, keep)
            };
            // one allocation, one copy: the old elements go straight into
            // the fresh backing store (no intermediate Vec or staging block)
            let elements = heap.allocate_hole_array(capacity).as_handle(scope);
            {
                let heap_ref: &Heap = heap;
                let obj = receiver.as_tagged(heap_ref);
                let old = obj.as_ref().elements_array(heap_ref).ok_or(VmError::Type)?;
                let new = elements.as_tagged(heap_ref);
                for k in 0..keep {
                    new.as_ref().set(heap_ref, k, old.at(heap_ref, k));
                }
                new.as_ref().set(heap_ref, i, value.as_tagged(heap_ref));
            }
            let obj = receiver.as_tagged(heap);
            obj.elements
                .set(heap, obj.erase(), elements.as_tagged(heap));
            obj.length.set(heap, obj.erase(), Smi::new(new_len as i64));
            Prototype::element_mutated(heap, obj);
        } else {
            let obj = receiver.as_tagged(heap);
            let elements = obj.as_ref().elements_array(heap).ok_or(VmError::Type)?;
            elements.set(heap, i, value.as_tagged(heap));
            // a store inside the physical capacity but past the logical
            // length still extends the array
            if i >= obj.as_ref().length() {
                obj.length.set(heap, obj.erase(), Smi::new(new_len as i64));
            }
            Prototype::element_mutated(heap, obj);
        }
        Ok(())
    }

    /// Write an existing element slot in place: never allocates, never
    /// grows, no handle scope required — takes anchored `Tagged` words
    /// under a single shared heap borrow (safe for fast paths).
    #[inline]
    pub fn store_array_element_in_place(
        heap: &Heap,
        receiver: Tagged<'_, Value>,
        i: usize,
        value: Tagged<'_, Value>,
    ) -> Result<(), VmError> {
        let Some(obj) = receiver.as_heap_object() else {
            return Err(VmError::Type);
        };
        let kind = obj.as_ref().header.map.get(heap).as_ref().kind();
        if !kind.is_array() || !kind.is_dense_elements() || i >= obj.as_ref().length() {
            return Err(VmError::OutOfBounds);
        }
        let elements = obj.as_ref().elements_array(heap).ok_or(VmError::Type)?;
        if i >= elements.len() {
            return Err(VmError::OutOfBounds);
        }
        if kind.is_holey() && elements.at(heap, i) == heap.known().the_hole.as_tagged(heap) {
            return Err(VmError::OutOfBounds);
        }
        elements.set(heap, i, value);
        Prototype::element_mutated(heap, obj);
        Ok(())
    }
}

pub struct ObjectInit<'a> {
    pub map: Handle<'a, Map>,
    pub slots: Handle<'a, FixedArray>,
    pub elements: Handle<'a, FixedArray>,
    pub length: usize,
}

pub struct ObjectSlotsInit<'m, 'v> {
    pub map: Handle<'m, Map>,
    pub values: HandleSlice<'v>,
    pub elements: Handle<'m, FixedArray>,
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
        context: Handle<'_, ContextObject>,
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

    pub fn to_primitive<'a>(
        vm: &'a VM,
        heap: &'a mut Heap,
        state: &'a ThreadState,
        value: Handle<'_, Value>,
        hint: Hint,
    ) -> Result<Coercion<'a>, VmError> {
        if Convert::is_primitive(heap, value.as_tagged(heap)) {
            return Ok(Coercion::Value(value.as_tagged(heap)));
        }
        HostCtx::new(vm, heap, state).handle_scope(|vm, heap, state, scope| {
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
                let result =
                    scope.handle(HostCtx::enter(vm, &mut *heap, state, exotic, args, None)?);
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
                let result =
                    scope.handle(HostCtx::enter(vm, &mut *heap, state, method, args, None)?);
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

    pub fn to_numeric(
        vm: &VM,
        heap: &mut Heap,
        state: &ThreadState,
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

    pub fn to_string<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ThreadState,
        value: Handle<'_, Value>,
    ) -> Result<Option<Tagged<'a, Value>>, VmError> {
        state.handle_scope(|scope| {
            let primitive = if Convert::is_primitive(heap, value.as_tagged(heap)) {
                value
            } else {
                match Self::to_primitive(vm, heap, state, value, Hint::String)? {
                    Coercion::Threw => return Ok(None),
                    Coercion::Value(v) => scope.handle(v),
                }
            };
            let s = scope.handle(Convert::to_string(heap, &scope, primitive)?);
            Ok(Some(s.as_tagged(heap)))
        })
    }

    pub fn numeric_op<'a>(
        vm: &'a VM,
        heap: &'a mut Heap,
        state: &'a ThreadState,
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

    pub fn to_property_key<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ThreadState,
        v: Handle<'_, Value>,
    ) -> Result<Option<Tagged<'a, SlotName>>, VmError> {
        // Smis are pointer-free: valid at any lifetime, no rooting needed.
        if let Some(smi) = Smi::decode(v.as_tagged(heap).raw()) {
            return Ok(Some(Tagged::from(smi)));
        }
        state.handle_scope(|scope| {
            let cond_5 = v.as_tagged(heap).get_as::<Symbol>(heap).is_some();
            if cond_5 {
                return Ok(Some(v.as_tagged(heap).as_name()));
            }
            let is_string = v.as_tagged(heap).get_as::<DenseString>(heap).is_some();
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
            // Root the word: the `Tagged<'a>` keeps the `&mut Heap` borrow
            // alive, which would conflict with the cast's shared map read.
            let stringified = scope.handle(stringified);
            let interned = match scope.cast::<DenseString>(heap, stringified.as_tagged(heap)) {
                Some(s) => vm.interner().intern_value(heap, &scope, &s),
                // ToString of a symbol primitive throws (ES 6.1.7.1)
                None => return Err(VmError::Type),
            };
            let word = interned.as_tagged(heap);
            // canonical index strings name the same property as their
            // numeric form (ES 6.1.7): canonicalize to the Smi spelling
            // exactly once here, so every downstream bits-compare is sound
            if let Some(i) = Lookup::canonical_index(word.as_ref().data(heap))
                && i < u32::MAX as usize
            {
                return Ok(Some(Tagged::from(Smi::new(i as i64))));
            }
            // fresh anchored re-read of the rooted interned word
            Ok(Some(word.into()))
        })
    }

    pub fn type_of<'a>(heap: &'a Heap, v: Tagged<'a, Value>) -> Tagged<'a, Value> {
        let known = heap.known();
        let s = known.strings;
        // `null` needs no arm of its own: it is a non-callable object, and
        // so is anything that is not one of the primitive kinds above
        let type_name = if v.is_smi() || v.get_as::<Float>(heap).is_some() {
            s.number
        } else if v == known.undefined.as_tagged(heap) || v == known.the_hole.as_tagged(heap) {
            s.undefined
        } else if v == known.true_object.as_tagged(heap) || v == known.false_object.as_tagged(heap)
        {
            s.boolean
        } else if v.get_as::<DenseString>(heap).is_some() {
            s.string
        } else if v.get_as::<Symbol>(heap).is_some() {
            s.symbol
        } else if Self::is_callable(heap, v) {
            s.function
        } else {
            s.object
        };
        type_name.as_tagged(heap).erase()
    }

    pub fn instance_of(
        vm: &VM,
        heap: &mut Heap,
        state: &ThreadState,
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
        if let Some(parents) = proto.get_as::<FixedArray>(heap) {
            for i in 0..parents.len() {
                if Self::has_proto_in_chain(heap, parents.at(heap, i), target) {
                    return true;
                }
            }
            return false;
        }
        Self::has_proto_in_chain(heap, proto, target)
    }

    pub fn create_construct_receiver_value<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ThreadState,
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
            let proto = scope.cast::<Object>(heap, proto.as_tagged(heap));
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
