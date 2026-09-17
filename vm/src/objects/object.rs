use core::alloc::Layout;

use crate::{
    CallableInfoObject, Context, DenseString, EdgeVisitable, FixedArray, FunctionKind, GcSlice, GcSlot,
    Handle, HandleScope, Header, Heap, HeapObject, HeapRef, Map, NoGc, ObjectKind, SlotName, Smi,
    Tagged, Value, Visitor, VmError,
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

    pub fn callable_info<'a>(
        &'a self,
        guard: &'a NoGc<'a>,
    ) -> Option<HeapRef<'a, CallableInfoObject>> {
        if !self.header.map.heap_ref(guard).kind().is_callable() {
            return None;
        }
        let info = self.slots.heap_ref(guard).at(0);
        info.get_as(guard)
    }

    pub fn closure_context<'a>(&'a self, guard: &'a NoGc<'a>) -> Option<HeapRef<'a, Context>> {
        if !self.header.map.heap_ref(guard).kind().is_callable() {
            return None;
        }
        self.slots.heap_ref(guard).at(1).get_as(guard)
    }

    pub fn native_index<'a>(&'a self, guard: &'a NoGc<'a>) -> Option<usize> {
        if !self.header.map.heap_ref(guard).kind().is_native() {
            return None;
        }
        let idx = Smi::decode(self.slots.heap_ref(guard).at(0))?.value();
        usize::try_from(idx).ok()
    }

    pub fn is_array<'a>(&'a self, nogc: &'a NoGc<'a>) -> bool {
        self.header.map.heap_ref(nogc).kind().kind() == ObjectKind::Array
    }

    /// The JSArray `length` internal slot, when `self` is an array named
    /// `name`: it lives outside the map descriptors, so descriptor walks
    /// must consult this first. `None` for any other name or non-array.
    pub fn array_length<'a>(&'a self, nogc: &'a NoGc<'a>, name: SlotName) -> Option<Value> {
        if !self.is_array(nogc) {
            return None;
        }
        let s = name.value().get_as::<DenseString>(nogc)?.as_ref();
        s.data(nogc)
            .matches_ascii(b"length")
            .then(|| self.length.inner())
    }

    /// The object's map (shape).
    pub fn map_ref<'a>(&self, nogc: &'a NoGc<'a>) -> HeapRef<'a, Map> {
        self.header.map.heap_ref(nogc)
    }

    /// Whether the object's map allows adding new properties.
    pub fn is_extendable<'a>(&self, nogc: &'a NoGc<'a>) -> bool {
        self.map_ref(nogc).kind().is_extendable()
    }

    /// The slot holding the value of the data slot at `offset`.
    pub fn slot<'a>(&self, nogc: &'a NoGc<'a>, offset: usize) -> &'a GcSlot {
        self.slots.heap_ref(nogc).as_ref().element_slot(offset)
    }

    pub fn length(&self) -> usize {
        self.length.to_smi().value() as usize
    }

    pub fn elements_array<'a>(&'a self, nogc: &'a NoGc<'a>) -> Option<HeapRef<'a, FixedArray>> {
        self.elements.inner().get_as::<FixedArray>(nogc)
    }

    /// Fast element read for array objects: `None` if `self` is not an
    /// array, the index is past the end, or the slot is a hole — the
    /// caller must fall back to a named property lookup.
    pub fn element_value<'a>(&'a self, nogc: &'a NoGc<'a>, i: usize) -> Option<Value> {
        if !self.is_array(nogc) || i >= self.length() {
            return None;
        }
        let elements = self.elements_array(nogc)?;
        if i >= elements.len() {
            // `length` can exceed the backing store: those indices are holes
            return None;
        }
        let v = elements.at(i);
        if v == nogc.known().the_hole.value() {
            return None;
        }
        Some(v)
    }
}

/// What kind of callable a value refers to.
pub enum CallTarget {
    Bytecode(Tagged<Object>, usize, FunctionKind),
    Native(usize),
}

pub fn call_target<'a>(nogc: &'a NoGc<'a>, f: Value) -> Option<CallTarget> {
    let obj = f.as_heap_object(nogc)?;
    let kind = obj.as_ref().header.map.heap_ref(nogc).kind();
    if !kind.is_callable() {
        return None;
    }
    if kind.is_native() {
        return Some(CallTarget::Native(obj.as_ref().native_index(nogc)?));
    }
    let info = obj.as_ref().callable_info(nogc)?;
    let register_count = info.register_count.to_smi().value() as usize;
    Some(CallTarget::Bytecode(
        obj.into_tagged(),
        register_count,
        info.function_kind(),
    ))
}

/// The callee's function kind, when it is a bytecode function.
pub fn function_kind_of<'a>(nogc: &'a NoGc<'a>, v: Value) -> Option<FunctionKind> {
    let obj = v.as_heap_object(nogc)?;
    let info = obj.as_ref().callable_info(nogc)?;
    Some(info.function_kind())
}

/// Store `value` at element index `i` of an array object, growing the
/// elements backing store and updating `length` when `i` is past the end.
pub fn store_array_element(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    receiver: Value,
    i: usize,
    value: Value,
) -> Result<(), VmError> {
    let new_len = i.checked_add(1).ok_or(VmError::OutOfBounds)?;
    let receiver = scope.cast::<Object>(receiver).ok_or(VmError::Type)?;

    let grows = heap.no_gc(|nogc| {
        let obj = receiver.heap_ref(nogc);
        if !obj.as_ref().is_array(nogc) {
            return Err(VmError::Type);
        }
        // grow only when the store index is past the physical backing
        // store (its capacity), not the logical length: sequential appends
        // with headroom must not reallocate every time
        let capacity = obj
            .as_ref()
            .elements_array(nogc)
            .map(|e| e.len())
            .unwrap_or(0);
        Ok(i >= capacity)
    })?;

    if grows {
        let mut values = heap.no_gc(|nogc| {
            let obj = receiver.heap_ref(nogc);
            let elements = obj.as_ref().elements_array(nogc).ok_or(VmError::Type)?;
            let keep = obj.as_ref().length().min(elements.len());
            let capacity = (new_len + (new_len >> 1) + 16).max(elements.len());
            let mut values = Vec::with_capacity(capacity);
            for k in 0..keep {
                values.push(elements.at(k));
            }
            values.resize(capacity, nogc.known().the_hole.value());
            Ok::<_, VmError>(values)
        })?;
        values[i] = value;
        let elements = heap.allocate_handle::<FixedArray>(scope.stage(&values), scope);
        heap.no_gc(|nogc| {
            let obj = receiver.heap_ref(nogc);
            obj.elements
                .set(nogc, obj.erase(), elements.as_tagged().erase());
            obj.length.set(nogc, obj.erase(), Smi::new(new_len as i64));
        });
    } else {
        heap.no_gc(|nogc| {
            let obj = receiver.heap_ref(nogc);
            let elements = obj.as_ref().elements_array(nogc).ok_or(VmError::Type)?;
            elements.set(nogc, i, value);
            // a store inside the physical capacity but past the logical
            // length still extends the array
            if i >= obj.as_ref().length() {
                obj.length.set(nogc, obj.erase(), Smi::new(new_len as i64));
            }
            Ok::<_, VmError>(())
        })?;
    }
    Ok(())
}

pub struct ObjectInit<'a> {
    pub map: Handle<'a, Map>,
    pub slots: Handle<'a, FixedArray>,
    pub elements: Handle<'a, Value>,
    pub length: usize,
}

pub struct ObjectSlotsInit<'m, 'v> {
    pub map: Handle<'m, Map>,
    pub values: GcSlice<'v>,
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

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header.map.set(nogc, host, config.map.as_tagged());
        self.slots.set(nogc, host, config.slots.as_tagged());
        self.elements.set(nogc, host, config.elements.erase());
        self.length.set(nogc, host, Smi::new(config.length as i64));
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
