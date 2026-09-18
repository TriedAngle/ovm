use core::alloc::Layout;

use crate::{
    CallableInfoObject, Context, DenseString, EdgeVisitable, FixedArray, FunctionKind, GcSlot,
    Handle, HandleScope, HandleSlice, Header, Heap, HeapObject, HeapRef, Map, ObjectKind, SlotName,
    Smi, Tagged, Value, Visitor, VmError,
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

    pub fn callable_info<'a>(&'a self, heap: &'a Heap) -> Option<HeapRef<'a, CallableInfoObject>> {
        if !self.header.map.heap_ref(heap).kind().is_callable() {
            return None;
        }
        let info = self.slots.heap_ref(heap).at(heap, 0);
        info.get_as::<CallableInfoObject>()
    }

    pub fn closure_context<'a>(&'a self, heap: &'a Heap) -> Option<Tagged<'a, Context>> {
        if !self.header.map.heap_ref(heap).kind().is_callable() {
            return None;
        }
        self.slots
            .heap_ref(heap)
            .at(heap, 1)
            .get_as_tagged::<Context>()
    }

    /// The `idx`-th entry of the callable's constant pool, as a slot name.
    pub fn constant_slot_name<'a>(&'a self, heap: &'a Heap, idx: usize) -> Tagged<'a, SlotName> {
        self.callable_info(heap)
            .expect("callable must have callable info")
            .constant_slot_name(heap, idx)
    }

    pub fn native_index<'a>(&'a self, heap: &'a Heap) -> Option<usize> {
        if !self.header.map.heap_ref(heap).kind().is_native() {
            return None;
        }
        let idx = Smi::decode(self.slots.heap_ref(heap).at(heap, 0).raw())?.value();
        usize::try_from(idx).ok()
    }

    pub fn is_array<'a>(&'a self, heap: &'a Heap) -> bool {
        self.header.map.heap_ref(heap).kind().kind() == ObjectKind::Array
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
    pub fn map_ref<'a>(&self, heap: &'a Heap) -> HeapRef<'a, Map> {
        self.header.map.heap_ref(heap)
    }

    /// Whether the object's map allows adding new properties.
    pub fn is_extendable(&self, heap: &Heap) -> bool {
        self.map_ref(heap).kind().is_extendable()
    }

    /// The slot holding the value of the data slot at `offset`.
    pub fn slot<'a>(&self, heap: &'a Heap, offset: usize) -> &'a GcSlot {
        self.slots.heap_ref(heap).as_ref().element_slot(offset)
    }

    pub fn length(&self) -> usize {
        self.length.to_smi().value() as usize
    }

    pub fn elements_array<'a>(&'a self, heap: &'a Heap) -> Option<HeapRef<'a, FixedArray>> {
        // Safety: fresh slot read under the anchor.
        unsafe { self.elements.inner().assume_valid(heap) }.get_as::<FixedArray>()
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
        // Safety: fresh root-slot read under the anchor.
        if v == unsafe { heap.known().the_hole.read_unchecked() } {
            return None;
        }
        Some(v)
    }
}

/// What kind of callable a value refers to.
pub enum CallTarget<'a> {
    Bytecode(Tagged<'a, Object>, usize, FunctionKind),
    Native(usize),
}

impl Object {
    /// Classify a value as a callable: a bytecode function (with register
    /// count and kind) or a native (with registry index).
    pub fn call_target<'a>(heap: &'a Heap, f: Tagged<'a, Value>) -> Option<CallTarget<'a>> {
        let obj = f.as_heap_object()?;
        let kind = obj.as_ref().header.map.heap_ref(heap).kind();
        if !kind.is_callable() {
            return None;
        }
        if kind.is_native() {
            return Some(CallTarget::Native(obj.as_ref().native_index(heap)?));
        }
        let info = obj.as_ref().callable_info(heap)?;
        let register_count = info.register_count.to_smi().value() as usize;
        Some(CallTarget::Bytecode(
            obj.into_tagged(),
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
            let obj = receiver.heap_ref(heap);
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
                let obj = receiver.heap_ref(heap_ref);
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
            let obj = receiver.heap_ref(heap);
            obj.elements
                .set(heap, obj.erase(), elements.as_tagged(heap).erase());
            obj.length.set(heap, obj.erase(), Smi::new(new_len as i64));
        } else {
            let obj = receiver.heap_ref(heap);
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
        let host = self.erase();
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
