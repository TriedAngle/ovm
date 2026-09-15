use core::{
    alloc::Layout,
    cell::{Cell, UnsafeCell},
    ptr::NonNull,
};

use crate::{
    Compare, EdgeVisitable, GcSlot, Handle, HandleScope, Heap, HeapPtr, HeapRef, NoGc,
    OptionGcSlot, STRONG_PTR, Smi, Tagged, TransitionGuard, Value, Visitor, VmError, WEAK_PTR,
    Word,
};

pub trait HeapObject: 'static {
    type Init<'a>;

    const KIND: ObjectKind;

    fn matches_kind(kind: ObjectKind) -> bool {
        kind == Self::KIND
    }

    fn layout_for(config: &Self::Init<'_>) -> Layout;

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>);

    fn header(&self) -> &Header;

    fn layout(&self) -> Layout;

    fn erase(&self) -> Value
    where
        Self: Sized,
    {
        let addr = self as *const Self as *const Word as Word;
        Value::from_bits(addr | STRONG_PTR)
    }

    fn erase_weak(&self) -> Value
    where
        Self: Sized,
    {
        let addr = self as *const Self as *const Word as Word;
        Value::from_bits(addr | WEAK_PTR)
    }
}

#[repr(C)]
pub struct Header {
    pub map: GcSlot<Map>,
}

impl Header {
    pub fn map(&self) -> Tagged<Map> {
        self.map.get()
    }
}

#[repr(C)]
pub struct Map {
    pub header: Header,
    pub value_slot_count: GcSlot<Smi>,
    pub descriptor_count: GcSlot<Smi>,
    /// Object kind tag (low byte) and capability flags (second byte).
    pub kind: GcSlot<Smi>,
    /// The prototype(s) for property lookup:
    /// - an object: single parent (JS `[[Prototype]]`)
    /// - a `FixedArray` of objects: multiple parents in priority order (Self-style `parent*`)
    /// - the hole: no parents (null-proto root)
    pub prototype: GcSlot,
    /// Empty, or a `FixedArray` of flat `[name, target_map]` transition pairs.
    // TODO: make transition targets weak (V8 does this so unused shape subtrees die)
    pub transitions: OptionGcSlot<FixedArray>,
    pub descriptors: [SlotDescriptor; 0],
}

impl Map {
    pub fn layout_for(descriptor_count: usize) -> Layout {
        let descriptors_layout =
            Layout::array::<SlotDescriptor>(descriptor_count).expect("descriptors layout");
        Layout::new::<Self>()
            .extend(descriptors_layout)
            .expect("map layout")
            .0
    }

    pub fn value_slot_count(&self) -> usize {
        self.value_slot_count.to_smi().value() as usize
    }

    pub fn descriptor_count(&self) -> usize {
        self.descriptor_count.to_smi().value() as usize
    }

    pub fn kind(&self) -> MapKind {
        MapKind::new(self.kind.to_smi().value() as u64)
    }

    fn data_ptr(&self) -> *mut SlotDescriptor {
        self.descriptors.as_ptr() as *mut SlotDescriptor
    }

    pub fn descriptors(&self) -> &[SlotDescriptor] {
        unsafe { core::slice::from_raw_parts(self.data_ptr(), self.descriptor_count()) }
    }

    pub fn descriptor(&self, i: usize) -> &SlotDescriptor {
        debug_assert!(i < self.descriptor_count());
        unsafe { &*self.data_ptr().add(i) }
    }

    pub fn find_transition<'a>(
        &self,
        nogc: &'a NoGc<'a>,
        name: SlotName,
        flags: SlotFlags,
        pair: Option<(Value, Value)>,
    ) -> Option<HeapRef<'a, Map>> {
        let lock = nogc.transition_lock();
        let guard = lock.acquire();
        self.find_transition_locked(nogc, name, flags, pair, &guard)
    }

    pub fn find_transition_locked<'a>(
        &self,
        nogc: &'a NoGc<'a>,
        name: SlotName,
        flags: SlotFlags,
        pair: Option<(Value, Value)>,
        _guard: &TransitionGuard<'_>,
    ) -> Option<HeapRef<'a, Map>> {
        let array = self.transitions.heap_ref(nogc)?;
        let pairs = array.as_slice();
        debug_assert!(
            pairs.len() % 2 == 0,
            "transition pairs are flat [name, map]"
        );
        for entry in pairs.chunks_exact(2) {
            if entry[0].inner() != name.value() {
                continue;
            }

            let target = entry[1]
                .inner()
                .get_as::<Map>(nogc)
                .expect("transition target must be a map");

            // adds append the property (last descriptor), redefines keep
            // its index: either way the descriptor row for `name` must
            // carry the requested flags. This lets adds and redefines
            // share one transition tree — identical shapes, identical maps.
            let Some(row) = target
                .descriptors()
                .iter()
                .find(|d| d.name() == name && d.flags() == flags)
            else {
                continue;
            };

            // accessor rows embed the AccessorPair: the cached map is only
            // reusable when the pair is identical (same-name accessors with
            // different pairs get separate tree entries)
            if let Some((get, set)) = pair {
                let matches = row
                    .value
                    .inner()
                    .get_as::<AccessorPair>(nogc)
                    .is_some_and(|p| {
                        Compare::same_value(nogc, get, p.get.inner())
                            && Compare::same_value(nogc, set, p.set.inner())
                    });
                if !matches {
                    continue;
                }
            }
            return Some(target);
        }
        None
    }

    /// Find the recorded remove-transition for `name`: a child map that
    /// lacks the descriptor and holds exactly one fewer. Add and redefine
    /// transitions key their pair by the target's own descriptor row for
    /// `name`; a removal target has no such row, so the two pair
    /// populations sharing one name never collide.
    pub fn find_remove_transition_locked<'a>(
        &self,
        nogc: &'a NoGc<'a>,
        name: SlotName,
        _guard: &TransitionGuard<'_>,
    ) -> Option<HeapRef<'a, Map>> {
        let array = self.transitions.heap_ref(nogc)?;
        let pairs = array.as_slice();
        debug_assert!(
            pairs.len() % 2 == 0,
            "transition pairs are flat [name, map]"
        );
        for entry in pairs.chunks_exact(2) {
            if entry[0].inner() != name.value() {
                continue;
            }
            let target = entry[1]
                .inner()
                .get_as::<Map>(nogc)
                .expect("transition target must be a map");
            if target.descriptor_count() + 1 == self.descriptor_count()
                && !target.descriptors().iter().any(|d| d.name() == name)
            {
                return Some(target);
            }
        }
        None
    }
}

pub struct MapInit<'a> {
    pub kind: MapKind,
    pub value_slot_count: usize,
    pub descriptors: &'a [(SlotName, SlotFlags, Value)],
    /// the hole = no prototype (null-proto for JS maps).
    /// Handled because `Map` allocation may move the prototype.
    pub prototype: Handle<'a, Value>,
}

impl HeapObject for Map {
    const KIND: ObjectKind = ObjectKind::Map;
    type Init<'a> = MapInit<'a>;

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.descriptors.len())
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().map_map.as_tagged());
        self.value_slot_count
            .set(nogc, host, Smi::new(config.value_slot_count as i64));
        self.descriptor_count
            .set(nogc, host, Smi::new(config.descriptors.len() as i64));
        self.kind
            .set(nogc, host, Smi::new(config.kind.bits() as i64));
        self.prototype.set(nogc, host, config.prototype.value());
        self.transitions.clear(nogc.heap());
        for (i, (name, flags, value)) in config.descriptors.iter().enumerate() {
            let d = self.descriptor(i);
            d.name.set(nogc, host, name.tagged());
            d.flags.set(nogc, host, Smi::new(flags.bits() as i64));
            d.value.set(nogc, host, *value);
        }
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.descriptor_count())
    }
}

impl EdgeVisitable for Map {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.prototype.as_raw());
        visitor.visit(self.transitions.as_raw());
        for d in self.descriptors() {
            visitor.visit(d.name.as_raw());
            visitor.visit(d.value.as_raw());
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u64)]
pub enum ObjectKind {
    BuiltinStart = 0,
    Map = 1,
    FixedArray = 2,
    FixedByteArray = 3,
    VMString = 4,
    AccessorPair = 5,
    CallableInfo = 6,
    Float = 7,
    Symbol = 8,
    HandlerTable = 9,
    Context = 10,
    ScopeInfo = 11,
    BuiltinEnd = 12,

    /// `elements` points to the well-known `empty_fixed_array`, `len` is 0
    Object = 13,
    /// `elements` points to a `FixedArray`.
    Array = 14,
    /// `elements` points to a `FixedByteArray`.
    ByteArray = 15,
    /// `elements` points to a `VMString`.
    String = 16,
    /// A Proxy exotic object (`ProxyObject`): no own properties, all
    /// internal methods dispatch through handler traps.
    Proxy = 17,
    /// Sentinel and odd heap values
    Oddball = 18,
}

impl ObjectKind {
    pub const BUILTIN_START: u64 = ObjectKind::BuiltinStart as u64;
    pub const BUILTIN_END: u64 = ObjectKind::BuiltinEnd as u64;

    /// Whether values of this kind are ECMAScript receivers (JSReceiver)
    pub const fn is_js_receiver(self) -> bool {
        matches!(
            self,
            ObjectKind::Object
                | ObjectKind::Array
                | ObjectKind::ByteArray
                | ObjectKind::String
                | ObjectKind::Proxy
        )
    }
}

/// Low byte: the `ObjectKind`. Second byte: capability flags
/// (extendable, callable, constructor, native). Constructor implies callable.
/// NATIVE is only valid together with CALLABLE and means slots[0] of the
/// object is a Smi native registry index instead of a `CallableInfoObject`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MapKind(u64);

impl MapKind {
    const KIND_MASK: u64 = 0xff;

    pub const EXTENDABLE: MapKind = MapKind(1 << 8);
    pub const CALLABLE: MapKind = MapKind(1 << 9);
    pub const CONSTRUCTOR: MapKind = MapKind(1 << 10);
    pub const NATIVE: MapKind = MapKind(1 << 11);
    pub const PRIMITIVE_WRAPPER: MapKind = MapKind(1 << 12);
    pub const CLASS_CONSTRUCTOR: MapKind = MapKind(1 << 13);

    pub const MAP: MapKind = MapKind(ObjectKind::Map as u64);
    pub const FIXED_ARRAY: MapKind = MapKind(ObjectKind::FixedArray as u64);
    pub const FIXED_BYTE_ARRAY: MapKind = MapKind(ObjectKind::FixedByteArray as u64);
    pub const VM_STRING: MapKind = MapKind(ObjectKind::VMString as u64);
    pub const ACCESSOR_PAIR: MapKind = MapKind(ObjectKind::AccessorPair as u64);
    pub const CALLABLE_INFO: MapKind = MapKind(ObjectKind::CallableInfo as u64);
    pub const FLOAT: MapKind = MapKind(ObjectKind::Float as u64);
    pub const SYMBOL: MapKind = MapKind(ObjectKind::Symbol as u64);
    pub const HANDLER_TABLE: MapKind = MapKind(ObjectKind::HandlerTable as u64);
    pub const CONTEXT: MapKind = MapKind(ObjectKind::Context as u64);
    pub const SCOPE_INFO: MapKind = MapKind(ObjectKind::ScopeInfo as u64);
    pub const OBJECT: MapKind = MapKind(ObjectKind::Object as u64);
    pub const ARRAY: MapKind = MapKind(ObjectKind::Array as u64);
    pub const BYTE_ARRAY: MapKind = MapKind(ObjectKind::ByteArray as u64);
    pub const STRING: MapKind = MapKind(ObjectKind::String as u64);
    pub const PROXY: MapKind = MapKind(ObjectKind::Proxy as u64);
    pub const ODDBALL: MapKind = MapKind(ObjectKind::Oddball as u64);

    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, flags: Self) -> bool {
        self.0 & flags.0 == flags.0
    }

    // TODO: consider transmute with debug assert
    pub const fn kind(self) -> ObjectKind {
        match Self(self.0 & Self::KIND_MASK) {
            Self::MAP => ObjectKind::Map,
            Self::FIXED_ARRAY => ObjectKind::FixedArray,
            Self::FIXED_BYTE_ARRAY => ObjectKind::FixedByteArray,
            Self::VM_STRING => ObjectKind::VMString,
            Self::ACCESSOR_PAIR => ObjectKind::AccessorPair,
            Self::CALLABLE_INFO => ObjectKind::CallableInfo,
            Self::FLOAT => ObjectKind::Float,
            Self::SYMBOL => ObjectKind::Symbol,
            Self::HANDLER_TABLE => ObjectKind::HandlerTable,
            Self::CONTEXT => ObjectKind::Context,
            Self::SCOPE_INFO => ObjectKind::ScopeInfo,
            Self::OBJECT => ObjectKind::Object,
            Self::ARRAY => ObjectKind::Array,
            Self::BYTE_ARRAY => ObjectKind::ByteArray,
            Self::STRING => ObjectKind::String,
            Self::PROXY => ObjectKind::Proxy,
            Self::ODDBALL => ObjectKind::Oddball,
            _ => panic!("invalid object kind"),
        }
    }

    pub const fn is_builtin(self) -> bool {
        let kind = self.0 & Self::KIND_MASK;
        kind > ObjectKind::BUILTIN_START && kind < ObjectKind::BUILTIN_END
    }

    pub const fn is_extendable(self) -> bool {
        self.0 & Self::EXTENDABLE.0 != 0
    }

    pub const fn is_callable(self) -> bool {
        self.0 & Self::CALLABLE.0 != 0
    }

    pub const fn is_native(self) -> bool {
        self.0 & Self::NATIVE.0 != 0
    }

    pub const fn is_constructor(self) -> bool {
        self.0 & Self::CONSTRUCTOR.0 != 0
    }

    pub const fn is_class_constructor(self) -> bool {
        self.0 & Self::CLASS_CONSTRUCTOR.0 != 0
    }
}

/// Descriptor flags. Data values live in the object's slots (the descriptor
/// holds a Smi offset); accessors embed the `AccessorPair` in the descriptor.
/// Writability is the WRITABLE attribute bit — there is no separate const kind.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SlotFlags(u64);

impl SlotFlags {
    pub const ACCESSOR: SlotFlags = SlotFlags(1 << 0);
    pub const WRITABLE: SlotFlags = SlotFlags(1 << 1);
    pub const CONFIGURABLE: SlotFlags = SlotFlags(1 << 2);
    pub const ENUMERABLE: SlotFlags = SlotFlags(1 << 3);

    /// The plain data slot: no flags set.
    pub const VALUE: SlotFlags = SlotFlags(0);

    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn is_accessor(self) -> bool {
        self.0 & Self::ACCESSOR.0 != 0
    }

    pub const fn is_writable(self) -> bool {
        self.0 & Self::WRITABLE.0 != 0
    }

    pub const fn is_configurable(self) -> bool {
        self.0 & Self::CONFIGURABLE.0 != 0
    }

    pub const fn is_enumerable(self) -> bool {
        self.0 & Self::ENUMERABLE.0 != 0
    }
}

#[repr(C)]
pub struct SlotDescriptor {
    pub name: GcSlot<SlotName>,
    pub flags: GcSlot<Smi>,
    pub value: GcSlot,
}

impl SlotDescriptor {
    pub fn name(&self) -> SlotName {
        SlotName(self.name.inner())
    }

    pub fn flags(&self) -> SlotFlags {
        SlotFlags::new(self.flags.to_smi().value() as u64)
    }

    pub fn offset(&self) -> usize {
        Smi::decode(self.value.inner())
            .expect("slot offset")
            .value() as usize
    }
}

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
        let s = name.value().get_as::<VMString>(nogc)?;
        (s.as_slice(nogc) == b"length").then(|| self.length.inner())
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

#[repr(C)]
pub struct ProxyObject {
    pub header: Header,
    pub target: GcSlot,
    pub handler: GcSlot,
}

pub struct ProxyInit<'a> {
    pub map: Handle<'a, Map>,
    pub target: Value,
    pub handler: Value,
}

impl ProxyObject {
    pub fn layout_for() -> Layout {
        Layout::new::<Self>()
    }

    /// Whether the proxy has been revoked (handler nulled).
    pub fn is_revoked<'a>(&self, nogc: &'a NoGc<'a>) -> bool {
        self.handler.inner() == nogc.known().null.value()
    }

    /// (target, handler) as raw values; caller checks revocation.
    pub fn parts(&self) -> (Value, Value) {
        (self.target.inner(), self.handler.inner())
    }
}

impl HeapObject for ProxyObject {
    const KIND: ObjectKind = ObjectKind::Proxy;
    type Init<'a> = ProxyInit<'a>;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Self::layout_for()
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header.map.set(nogc, host, config.map.as_tagged());
        self.target.set(nogc, host, config.target);
        self.handler.set(nogc, host, config.handler);
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for()
    }
}

impl EdgeVisitable for ProxyObject {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.target.as_raw());
        visitor.visit(self.handler.as_raw());
    }
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
        let elements = heap.allocate_handle::<FixedArray>(&values, scope);
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
    pub values: &'v [Value],
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

#[repr(C)]
pub struct FixedArray {
    pub header: Header,
    pub size: GcSlot<Smi>,
    pub values: [GcSlot; 0],
}

impl FixedArray {
    pub fn layout_for(len: usize) -> Layout {
        let values_layout = Layout::array::<GcSlot>(len).expect("values layout");
        Layout::new::<Self>()
            .extend(values_layout)
            .expect("array layout")
            .0
    }

    pub fn len(&self) -> usize {
        self.size.to_smi().value() as usize
    }

    pub fn data_ptr(&self) -> *mut GcSlot {
        self.values.as_ptr() as *mut GcSlot
    }

    pub fn as_slice(&self) -> &[GcSlot] {
        unsafe { core::slice::from_raw_parts(self.data_ptr(), self.len()) }
    }

    pub fn element_slot(&self, i: usize) -> &GcSlot {
        debug_assert!(i < self.len());
        unsafe { &*self.values.as_ptr().add(i) }
    }

    pub fn at(&self, i: usize) -> Value {
        self.element_slot(i).get().erase()
    }

    pub fn set(&self, nogc: &NoGc<'_>, i: usize, v: Value) {
        self.element_slot(i).set(nogc, self.erase(), v);
    }
}

impl HeapObject for FixedArray {
    const KIND: ObjectKind = ObjectKind::FixedArray;
    type Init<'a> = &'a [Value];

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.len())
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().array_map.as_tagged());
        self.size.set(nogc, host, Smi::new(config.len() as i64));
        for (i, v) in config.iter().enumerate() {
            self.element_slot(i).set(nogc, host, *v);
        }
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.len())
    }
}

impl EdgeVisitable for FixedArray {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        let size = self.size.to_smi().value() as usize;
        for i in 0..size {
            visitor.visit(self.element_slot(i).as_raw());
        }
    }
}

#[repr(C)]
pub struct FixedByteArray {
    pub header: Header,
    pub size: GcSlot<Smi>,
    pub values: [UnsafeCell<u8>; 0],
}

impl FixedByteArray {
    pub fn layout_for(len: usize) -> Layout {
        let values_layout = Layout::array::<u8>(len).expect("values layout");
        Layout::new::<Self>()
            .extend(values_layout)
            .expect("bytearray layout")
            .0
    }

    pub fn len(&self) -> usize {
        self.size.to_smi().value() as usize
    }

    fn data_ptr(&self) -> *mut u8 {
        UnsafeCell::raw_get(self.values.as_ptr())
    }

    pub fn get(&self, i: usize) -> u8 {
        debug_assert!(i < self.len());
        unsafe { *self.data_ptr().add(i) }
    }
    pub fn set(&self, i: usize, b: u8) {
        debug_assert!(i < self.len());
        unsafe { *self.data_ptr().add(i) = b }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.data_ptr(), self.len()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.data_ptr(), self.len()) }
    }
}

impl HeapObject for FixedByteArray {
    const KIND: ObjectKind = ObjectKind::FixedByteArray;
    type Init<'a> = &'a [u8];

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.len())
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().byte_array_map.as_tagged());
        self.size.set(nogc, host, Smi::new(config.len() as i64));
        for (i, b) in config.iter().enumerate() {
            self.set(i, *b);
        }
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.len())
    }
}

impl EdgeVisitable for FixedByteArray {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
    }
}

#[repr(C)]
pub struct VMString {
    pub header: Header,
    pub backing: GcSlot<FixedByteArray>,
    pub hash: GcSlot<Smi>,
}

/// Content hash for strings (FNV-1a, masked into smi range).
/// TODO: decide on a hash algorithm
pub fn string_content_hash(bytes: &[u8]) -> i64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h & ((1 << 62) - 1)) as i64
}

impl VMString {
    pub fn from_bytes<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        bytes: &[u8],
    ) -> Handle<'s, VMString> {
        let backing = heap.allocate_handle::<FixedByteArray>(bytes, scope);
        heap.allocate_handle::<VMString>((backing, string_content_hash(bytes)), scope)
    }

    pub fn concat<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        a: Value,
        b: Value,
    ) -> Handle<'s, VMString> {
        let bytes = heap.no_gc(|nogc| {
            let sa = a
                .get_as::<VMString>(nogc)
                .expect("concat operand must be a string");
            let sb = b
                .get_as::<VMString>(nogc)
                .expect("concat operand must be a string");
            let mut out = Vec::with_capacity(sa.len(nogc) + sb.len(nogc));
            out.extend_from_slice(sa.as_slice(nogc));
            out.extend_from_slice(sb.as_slice(nogc));
            out
        });
        Self::from_bytes(heap, scope, &bytes)
    }

    pub fn backing<'a>(&self, nogc: &'a NoGc<'a>) -> HeapRef<'a, FixedByteArray> {
        self.backing.heap_ref(nogc)
    }

    pub fn hash(&self) -> i64 {
        self.hash.to_smi().value()
    }

    pub fn len<'a>(&self, nogc: &'a NoGc<'a>) -> usize {
        self.backing(nogc).len()
    }

    pub fn as_slice<'a>(&self, nogc: &'a NoGc<'a>) -> &'a [u8] {
        self.backing(nogc).as_ref().as_slice()
    }

    pub fn as_str<'a>(&self, nogc: &'a NoGc<'a>) -> Option<&'a str> {
        core::str::from_utf8(self.as_slice(nogc)).ok()
    }
}

impl HeapObject for VMString {
    const KIND: ObjectKind = ObjectKind::VMString;
    type Init<'a> = (Handle<'a, FixedByteArray>, i64);

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().string_map.as_tagged());
        self.backing.set(nogc, host, config.0.as_tagged());
        self.hash.set(nogc, host, Smi::new(config.1));
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for VMString {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.backing.as_raw());
    }
}

#[repr(C)]
pub struct InternedString(pub VMString);

impl InternedString {
    pub fn string(&self) -> &VMString {
        &self.0
    }
}

impl HeapObject for InternedString {
    const KIND: ObjectKind = ObjectKind::VMString;
    type Init<'a> = (Handle<'a, FixedByteArray>, i64);

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        self.0.init(nogc, config);
    }

    fn header(&self) -> &Header {
        self.0.header()
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for InternedString {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        self.0.visit_edges(visitor);
    }
}

#[repr(C)]
pub struct Symbol {
    pub header: Header,
    pub backing: GcSlot<FixedByteArray>,
}

impl Symbol {
    pub fn new<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        description: &[u8],
    ) -> Handle<'s, Symbol> {
        let backing = heap.allocate_handle::<FixedByteArray>(description, scope);
        heap.allocate_handle::<Symbol>(backing, scope)
    }

    pub fn backing<'a>(&self, nogc: &'a NoGc<'a>) -> HeapRef<'a, FixedByteArray> {
        self.backing.heap_ref(nogc)
    }

    pub fn len<'a>(&self, nogc: &'a NoGc<'a>) -> usize {
        self.backing(nogc).len()
    }

    pub fn as_slice<'a>(&self, nogc: &'a NoGc<'a>) -> &'a [u8] {
        self.backing(nogc).as_ref().as_slice()
    }

    pub fn as_str<'a>(&self, nogc: &'a NoGc<'a>) -> Option<&'a str> {
        core::str::from_utf8(self.as_slice(nogc)).ok()
    }
}

impl HeapObject for Symbol {
    const KIND: ObjectKind = ObjectKind::Symbol;
    type Init<'a> = Handle<'a, FixedByteArray>;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().symbol_map.as_tagged());
        self.backing.set(nogc, host, *config);
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for Symbol {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.backing.as_raw());
    }
}

/// A property name: an interned string, a symbol, or a smi index.
#[repr(transparent)]
#[derive(Debug, Copy, Clone)]
pub struct SlotName(Value);

impl SlotName {
    pub fn value(self) -> Value {
        self.0
    }

    pub fn from_value(value: Value) -> Self {
        Self(value)
    }

    pub fn tagged(self) -> Tagged<SlotName> {
        unsafe { Tagged::from_value_unchecked(self.0) }
    }
}

impl From<Tagged<VMString>> for SlotName {
    fn from(string: Tagged<VMString>) -> Self {
        Self(string.erase())
    }
}

impl From<Tagged<Symbol>> for SlotName {
    fn from(symbol: Tagged<Symbol>) -> Self {
        Self(symbol.erase())
    }
}

impl From<Tagged<InternedString>> for SlotName {
    fn from(string: Tagged<InternedString>) -> Self {
        Self(string.erase())
    }
}

impl From<Tagged<Smi>> for SlotName {
    fn from(smi: Tagged<Smi>) -> Self {
        Self(smi.erase())
    }
}

impl From<Handle<'_, SlotName>> for SlotName {
    fn from(name: Handle<'_, SlotName>) -> Self {
        Self::from_value(name.value())
    }
}

impl PartialEq for SlotName {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for SlotName {}

#[repr(C)]
pub struct AccessorPair {
    pub header: Header,
    pub get: GcSlot,
    pub set: GcSlot,
}

impl HeapObject for AccessorPair {
    const KIND: ObjectKind = ObjectKind::AccessorPair;
    type Init<'a> = (Value, Value);

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().accessor_pair_map.as_tagged());
        self.get.set(nogc, host, config.0);
        self.set.set(nogc, host, config.1);
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for AccessorPair {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.get.as_raw());
        visitor.visit(self.set.as_raw());
    }
}

#[repr(C)]
pub struct CallableInfoObject {
    pub header: Header,
    pub bytecode: GcSlot<FixedByteArray>,
    pub constants: GcSlot<FixedArray>,
    pub register_count: GcSlot<Smi>,
    pub handlers: OptionGcSlot<HandlerTable>,
    pub name: GcSlot,
    pub formal_parameter_count: GcSlot<Smi>,
    /// JS-visible `length` (differs from `formal_parameter_count` when the
    /// parameter list has defaults / patterns / a rest parameter)
    pub formal_length: GcSlot<Smi>,
    pub kind: GcSlot<Smi>,
    /// Language mode is preserved now; strict-sensitive call/store/delete
    /// branches are intentionally deferred.
    pub strict: GcSlot<Smi>,
}

#[repr(i64)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FunctionKind {
    #[default]
    Normal,
    Generator,
    Arrow,
    Method,
    Getter,
    Setter,
    BaseClassConstructor,
    DerivedClassConstructor,
    /// synthesized default constructor of a derived class:
    /// `constructor(...args) { super(...args) }`
    DefaultDerivedConstructor,
}

impl FunctionKind {
    pub const fn is_constructible(self) -> bool {
        matches!(
            self,
            Self::Normal
                | Self::BaseClassConstructor
                | Self::DerivedClassConstructor
                | Self::DefaultDerivedConstructor
        )
    }

    pub const fn is_class_constructor(self) -> bool {
        matches!(
            self,
            Self::BaseClassConstructor
                | Self::DerivedClassConstructor
                | Self::DefaultDerivedConstructor
        )
    }

    pub const fn is_derived_class_constructor(self) -> bool {
        matches!(
            self,
            Self::DerivedClassConstructor | Self::DefaultDerivedConstructor
        )
    }

    pub const fn needs_prototype(self) -> bool {
        matches!(self, Self::Normal)
    }

    fn decode(value: i64) -> Self {
        match value {
            x if x == Self::Normal as i64 => Self::Normal,
            x if x == Self::Generator as i64 => Self::Generator,
            x if x == Self::Arrow as i64 => Self::Arrow,
            x if x == Self::Method as i64 => Self::Method,
            x if x == Self::Getter as i64 => Self::Getter,
            x if x == Self::Setter as i64 => Self::Setter,
            x if x == Self::BaseClassConstructor as i64 => Self::BaseClassConstructor,
            x if x == Self::DerivedClassConstructor as i64 => Self::DerivedClassConstructor,
            x if x == Self::DefaultDerivedConstructor as i64 => Self::DefaultDerivedConstructor,
            _ => panic!("invalid function kind"),
        }
    }
}

pub struct CallableInfoInit<'a> {
    pub bytecode: Handle<'a, FixedByteArray>,
    pub constants: Handle<'a, FixedArray>,
    pub register_count: usize,
    pub handlers: Option<Handle<'a, HandlerTable>>,
}

impl HeapObject for CallableInfoObject {
    const KIND: ObjectKind = ObjectKind::CallableInfo;
    type Init<'a> = CallableInfoInit<'a>;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().callable_map.as_tagged());
        self.bytecode.set(nogc, host, config.bytecode.as_tagged());
        self.constants.set(nogc, host, config.constants.as_tagged());
        self.register_count
            .set(nogc, host, Smi::new(config.register_count as i64));
        match config.handlers {
            Some(handlers) => self.handlers.set(nogc, host, handlers),
            None => self.handlers.clear(nogc.heap()),
        }
        self.name.set(nogc, host, nogc.known().the_hole.value());
        self.formal_parameter_count.set(nogc, host, Smi::new(0));
        self.formal_length.set(nogc, host, Smi::new(0));
        self.kind
            .set(nogc, host, Smi::new(FunctionKind::Normal as i64));
        self.strict.set(nogc, host, Smi::new(0));
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for CallableInfoObject {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.bytecode.as_raw());
        visitor.visit(self.constants.as_raw());
        visitor.visit(self.handlers.as_raw());
        visitor.visit(self.name.as_raw());
    }
}

impl CallableInfoObject {
    pub fn set_metadata(
        &self,
        nogc: &NoGc<'_>,
        name: Option<Value>,
        formal_parameter_count: usize,
        kind: FunctionKind,
        strict: bool,
    ) {
        self.set_metadata_full(
            nogc,
            name,
            formal_parameter_count,
            formal_parameter_count,
            kind,
            strict,
        )
    }

    pub fn set_metadata_full(
        &self,
        nogc: &NoGc<'_>,
        name: Option<Value>,
        formal_parameter_count: usize,
        formal_length: usize,
        kind: FunctionKind,
        strict: bool,
    ) {
        let host = self.erase();
        self.name.set(
            nogc,
            host,
            name.unwrap_or_else(|| nogc.known().the_hole.value()),
        );
        self.formal_parameter_count
            .set(nogc, host, Smi::new(formal_parameter_count as i64));
        self.formal_length
            .set(nogc, host, Smi::new(formal_length as i64));
        self.kind.set(nogc, host, Smi::new(kind as i64));
        self.strict.set(nogc, host, Smi::new(i64::from(strict)));
    }

    pub fn name<'a>(&self, nogc: &'a NoGc<'a>) -> Option<Value> {
        let name = self.name.inner();
        name.get_as::<VMString>(nogc).map(|_| name)
    }

    pub fn formal_parameter_count(&self) -> usize {
        self.formal_parameter_count.to_smi().value() as usize
    }

    /// JS-visible `length`
    pub fn formal_length(&self) -> usize {
        self.formal_length.to_smi().value() as usize
    }

    pub fn function_kind(&self) -> FunctionKind {
        FunctionKind::decode(self.kind.to_smi().value())
    }

    pub fn is_strict(&self) -> bool {
        self.strict.to_smi().value() != 0
    }

    /// Decode a constant-pool property name.
    // TODO: this must handle also non constants and non interned strings and symbols
    pub fn constant_slot_name<'a>(&self, nogc: &'a NoGc<'a>, idx: usize) -> SlotName {
        let v = self.constants.heap_ref(nogc).at(idx);
        let name = v
            .get_as::<InternedString>(nogc)
            .expect("property name constant must be an interned string");
        SlotName::from(name.into_tagged())
    }
}

/// layout `[range_start, range_end, handler_offset]`: a half-open bytecode region
#[repr(C)]
pub struct HandlerEntry {
    pub try_start: GcSlot<Smi>,
    pub try_end: GcSlot<Smi>,
    pub handler_pc: GcSlot<Smi>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct HandlerEntryInit {
    pub try_start: usize,
    pub try_end: usize,
    pub handler_pc: usize,
}

impl HandlerEntryInit {
    pub const fn new(try_start: usize, try_end: usize, handler_pc: usize) -> Self {
        Self {
            try_start,
            try_end,
            handler_pc,
        }
    }
}

#[repr(C)]
pub struct HandlerTable {
    pub header: Header,
    pub size: GcSlot<Smi>,
    pub entries: [HandlerEntry; 0],
}

pub struct HandlerTableInit<'a> {
    pub entries: &'a [HandlerEntryInit],
}

impl HandlerTable {
    pub fn layout_for(entry_count: usize) -> Layout {
        let entries_layout =
            Layout::array::<HandlerEntry>(entry_count).expect("handler table layout");
        Layout::new::<Self>()
            .extend(entries_layout)
            .expect("handler table layout")
            .0
    }

    pub fn len(&self) -> usize {
        self.size.to_smi().value() as usize
    }

    fn entry_ptr(&self) -> *mut HandlerEntry {
        self.entries.as_ptr() as *mut HandlerEntry
    }

    pub fn entry(&self, i: usize) -> HandlerEntryInit {
        debug_assert!(i < self.len());
        let e = unsafe { &*self.entry_ptr().add(i) };
        HandlerEntryInit {
            try_start: e.try_start.to_smi().value() as usize,
            try_end: e.try_end.to_smi().value() as usize,
            handler_pc: e.handler_pc.to_smi().value() as usize,
        }
    }

    pub fn lookup(&self, pc: usize) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for i in 0..self.len() {
            let e = self.entry(i);
            if e.try_start <= pc && pc < e.try_end {
                match best {
                    Some((start, _)) if start >= e.try_start => {}
                    _ => best = Some((e.try_start, e.handler_pc)),
                }
            }
        }
        best.map(|(_, handler_pc)| handler_pc)
    }
}

impl HeapObject for HandlerTable {
    const KIND: ObjectKind = ObjectKind::HandlerTable;
    type Init<'a> = HandlerTableInit<'a>;

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.entries.len())
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().handler_table_map.as_tagged());
        self.size
            .set(nogc, host, Smi::new(config.entries.len() as i64));
        for (i, e) in config.entries.iter().enumerate() {
            let slot = unsafe { &*self.entry_ptr().add(i) };
            slot.try_start.set(nogc, host, Smi::new(e.try_start as i64));
            slot.try_end.set(nogc, host, Smi::new(e.try_end as i64));
            slot.handler_pc
                .set(nogc, host, Smi::new(e.handler_pc as i64));
        }
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.len())
    }
}

impl EdgeVisitable for HandlerTable {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
    }
}

// TODO: more info
#[repr(C)]
pub struct ScopeInfo {
    pub header: Header,
    /// parallel to the context's slots
    pub names: GcSlot<FixedArray>,
}

pub struct ScopeInfoInit<'a> {
    pub names: Handle<'a, FixedArray>,
}

impl HeapObject for ScopeInfo {
    const KIND: ObjectKind = ObjectKind::ScopeInfo;
    type Init<'a> = ScopeInfoInit<'a>;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().scope_info_map.as_tagged());
        self.names.set(nogc, host, config.names.as_tagged());
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for ScopeInfo {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.names.as_raw());
    }
}

#[repr(C)]
pub struct Context {
    pub header: Header,
    pub outer: OptionGcSlot<Context>,
    pub slots: GcSlot<FixedArray>,
    pub scope_info: GcSlot<ScopeInfo>,
}

pub struct ContextInit<'a> {
    pub outer: Option<Handle<'a, Context>>,
    pub slots: Handle<'a, FixedArray>,
    pub scope_info: Handle<'a, ScopeInfo>,
}

impl HeapObject for Context {
    const KIND: ObjectKind = ObjectKind::Context;
    type Init<'a> = ContextInit<'a>;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().context_map.as_tagged());
        match config.outer {
            Some(outer) => self.outer.set(nogc, host, outer),
            None => self.outer.clear(nogc.heap()),
        }
        self.slots.set(nogc, host, config.slots.as_tagged());
        self.scope_info
            .set(nogc, host, config.scope_info.as_tagged());
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for Context {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.outer.as_raw());
        visitor.visit(self.slots.as_raw());
        visitor.visit(self.scope_info.as_raw());
    }
}

#[repr(C)]
pub struct Float {
    pub header: Header,
    pub value: Cell<f64>,
}

impl HeapObject for Float {
    const KIND: ObjectKind = ObjectKind::Float;
    type Init<'a> = f64;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().float_map.as_tagged());
        self.value.set(*config);
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for Float {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
    }
}

pub unsafe fn object_kind(addr: NonNull<()>) -> ObjectKind {
    let header = unsafe { &*addr.cast::<Header>().as_ptr() };
    let map = header.map.get();
    let map_ref = unsafe { HeapPtr::<Map>::from(map).as_ref() };
    map_ref.kind().kind()
}

pub unsafe fn object_layout(addr: NonNull<()>) -> Layout {
    let kind = unsafe { object_kind(addr) };
    unsafe {
        match kind {
            ObjectKind::Map => (*addr.cast::<Map>().as_ptr()).layout(),
            ObjectKind::FixedArray => (*addr.cast::<FixedArray>().as_ptr()).layout(),
            ObjectKind::FixedByteArray => (*addr.cast::<FixedByteArray>().as_ptr()).layout(),
            ObjectKind::VMString => (*addr.cast::<VMString>().as_ptr()).layout(),
            ObjectKind::AccessorPair => (*addr.cast::<AccessorPair>().as_ptr()).layout(),
            ObjectKind::CallableInfo => (*addr.cast::<CallableInfoObject>().as_ptr()).layout(),
            ObjectKind::Float => (*addr.cast::<Float>().as_ptr()).layout(),
            ObjectKind::Symbol => (*addr.cast::<Symbol>().as_ptr()).layout(),
            ObjectKind::HandlerTable => (*addr.cast::<HandlerTable>().as_ptr()).layout(),
            ObjectKind::Context => (*addr.cast::<Context>().as_ptr()).layout(),
            ObjectKind::ScopeInfo => (*addr.cast::<ScopeInfo>().as_ptr()).layout(),
            ObjectKind::Object
            | ObjectKind::Array
            | ObjectKind::ByteArray
            | ObjectKind::String
            | ObjectKind::Oddball => (*addr.cast::<Object>().as_ptr()).layout(),
            ObjectKind::Proxy => (*addr.cast::<ProxyObject>().as_ptr()).layout(),
            ObjectKind::BuiltinStart | ObjectKind::BuiltinEnd => {
                unreachable!("sentinel kind in object header")
            }
        }
    }
}

pub unsafe fn visit_object(addr: NonNull<()>, visitor: &mut dyn Visitor) {
    let kind = unsafe { object_kind(addr) };
    unsafe {
        match kind {
            ObjectKind::Map => (*addr.cast::<Map>().as_ptr()).visit_edges(visitor),
            ObjectKind::FixedArray => (*addr.cast::<FixedArray>().as_ptr()).visit_edges(visitor),
            ObjectKind::FixedByteArray => {
                (*addr.cast::<FixedByteArray>().as_ptr()).visit_edges(visitor)
            }
            ObjectKind::VMString => (*addr.cast::<VMString>().as_ptr()).visit_edges(visitor),
            ObjectKind::AccessorPair => {
                (*addr.cast::<AccessorPair>().as_ptr()).visit_edges(visitor)
            }
            ObjectKind::CallableInfo => {
                (*addr.cast::<CallableInfoObject>().as_ptr()).visit_edges(visitor)
            }
            ObjectKind::Float => (*addr.cast::<Float>().as_ptr()).visit_edges(visitor),
            ObjectKind::Symbol => (*addr.cast::<Symbol>().as_ptr()).visit_edges(visitor),
            ObjectKind::HandlerTable => {
                (*addr.cast::<HandlerTable>().as_ptr()).visit_edges(visitor)
            }
            ObjectKind::Context => (*addr.cast::<Context>().as_ptr()).visit_edges(visitor),
            ObjectKind::ScopeInfo => (*addr.cast::<ScopeInfo>().as_ptr()).visit_edges(visitor),
            ObjectKind::Object
            | ObjectKind::Array
            | ObjectKind::ByteArray
            | ObjectKind::String
            | ObjectKind::Oddball => (*addr.cast::<Object>().as_ptr()).visit_edges(visitor),
            ObjectKind::Proxy => (*addr.cast::<ProxyObject>().as_ptr()).visit_edges(visitor),
            ObjectKind::BuiltinStart | ObjectKind::BuiltinEnd => {
                unreachable!("sentinel kind in object header")
            }
        }
    }
}
