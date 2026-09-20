use core::alloc::Layout;

use crate::{
    DenseString, EdgeVisitable, FeedbackVector, FixedArray, FixedByteArray, GcSlot, Handle,
    HandlerTable, Header, Heap, HeapObject, ObjectKind, OptionGcSlot, SlotName, Smi, Tagged, Value,
    Visitor,
};

#[repr(C)]
pub struct CallableInfoObject {
    pub header: Header,
    pub bytecode: GcSlot<FixedByteArray>,
    pub constants: GcSlot<FixedArray>,
    pub register_count: GcSlot<Smi>,
    pub handlers: OptionGcSlot<HandlerTable>,
    /// Inline-cache state for this function's property-access sites;
    /// the hole while the function has no feedback slots.
    pub feedback: OptionGcSlot<FeedbackVector>,
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

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(heap, host, heap.known().callable_map.as_tagged(heap));
        self.bytecode
            .set(heap, host, config.bytecode.as_tagged(heap));
        self.constants
            .set(heap, host, config.constants.as_tagged(heap));
        self.register_count
            .set(heap, host, Smi::new(config.register_count as i64));
        match config.handlers {
            Some(handlers) => self.handlers.set(heap, host, handlers.as_tagged(heap)),
            None => self.handlers.clear(heap),
        }
        self.feedback.clear(heap);
        self.name
            .set(heap, host, heap.known().the_hole.as_tagged(heap).erase());
        self.formal_parameter_count.set(heap, host, Smi::new(0));
        self.formal_length.set(heap, host, Smi::new(0));
        self.kind
            .set(heap, host, Smi::new(FunctionKind::Normal as i64));
        self.strict.set(heap, host, Smi::new(0));
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
        visitor.visit(self.feedback.as_raw());
        visitor.visit(self.name.as_raw());
    }
}

impl CallableInfoObject {
    pub fn set_metadata(
        &self,
        heap: &Heap,
        name: Option<Tagged<'_, Value>>,
        formal_parameter_count: usize,
        kind: FunctionKind,
        strict: bool,
    ) {
        self.set_metadata_full(
            heap,
            name,
            formal_parameter_count,
            formal_parameter_count,
            kind,
            strict,
        )
    }

    pub fn set_metadata_full(
        &self,
        heap: &Heap,
        name: Option<Tagged<'_, Value>>,
        formal_parameter_count: usize,
        formal_length: usize,
        kind: FunctionKind,
        strict: bool,
    ) {
        let host = self.erase();
        self.name.set(
            heap,
            host,
            name.unwrap_or_else(|| heap.known().the_hole.as_tagged(heap).erase()),
        );
        self.formal_parameter_count
            .set(heap, host, Smi::new(formal_parameter_count as i64));
        self.formal_length
            .set(heap, host, Smi::new(formal_length as i64));
        self.kind.set(heap, host, Smi::new(kind as i64));
        self.strict.set(heap, host, Smi::new(i64::from(strict)));
    }

    /// Attach the (already allocated) inline-cache state.
    pub fn set_feedback(&self, heap: &Heap, vector: Handle<'_, FeedbackVector>) {
        self.feedback
            .set(heap, self.erase(), vector.as_tagged(heap));
    }

    pub fn feedback<'a>(&self, heap: &'a Heap) -> Option<Tagged<'a, FeedbackVector>> {
        self.feedback.get(heap)
    }

    pub fn name<'a>(&self, heap: &'a Heap) -> Option<Tagged<'a, Value>> {
        self.name.get(heap).get_as::<DenseString>().map(|_| {
            // Safety: fresh slot read under the anchor.
            unsafe { Tagged::from_value_unchecked(self.name.inner()) }
        })
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
    pub fn constant_slot_name<'a>(&self, heap: &'a Heap, idx: usize) -> Tagged<'a, SlotName> {
        let v = self.constants.get(heap).at(heap, idx);
        let name = v
            .get_as::<DenseString>()
            .expect("property name constant must be an interned string");
        name.into()
    }
}
