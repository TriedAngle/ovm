use core::alloc::Layout;

use crate::lookup::has_property;
use crate::runtime::{Coercion, Runtime};
use crate::{
    Compare, ContextState, Convert, EdgeVisitable, FixedArray, GcSlice, GcSlot, Handle,
    HandleScope, Header, Heap, HeapObject, Key, Lookup, Map, NativeContext, Object, ObjectKind,
    PartialDescriptor, PropertyDescriptor, SlotName, Smi, Tagged, VM, Value, Visitor, VmError,
    is_compatible_property_descriptor,
};

#[repr(C)]
pub struct ProxyObject {
    pub header: Header,
    pub target: GcSlot,
    pub handler: GcSlot,
}

pub struct ProxyInit<'a> {
    pub map: Handle<'a, Map>,
    pub target: Handle<'a, Value>,
    pub handler: Handle<'a, Value>,
}

impl ProxyObject {
    pub fn layout_for() -> Layout {
        Layout::new::<Self>()
    }

    /// Whether the proxy has been revoked (handler nulled).
    pub fn is_revoked(&self, heap: &Heap) -> bool {
        self.handler.inner() == heap.known().null.as_tagged(heap).erase()
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

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header.map.set(heap, host, config.map.as_tagged(heap));
        self.target
            .set(heap, host, config.target.as_tagged(heap).erase_type());
        self.handler
            .set(heap, host, config.handler.as_tagged(heap).erase_type());
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

/// `Ok(Flow::Threw)` means user code threw and the pending exception is
/// set (the caller surfaces the exception sentinel).
pub enum Flow<T> {
    Threw,
    Value(T),
}

/// The thirteen proxy traps (ES 20.2).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Trap {
    Get,
    Set,
    Has,
    DeleteProperty,
    GetOwnPropertyDescriptor,
    DefineProperty,
    GetPrototypeOf,
    SetPrototypeOf,
    IsExtensible,
    PreventExtensions,
    OwnKeys,
    Apply,
    Construct,
}

impl Trap {
    fn name(self, heap: &Heap) -> SlotName {
        let s = heap.known().strings;
        match self {
            Self::Get => SlotName::from_value(unsafe { s.get.read_unchecked() }),
            Self::Set => SlotName::from_value(unsafe { s.set.read_unchecked() }),
            Self::Has => SlotName::from_value(unsafe { s.has.read_unchecked() }),
            Self::DeleteProperty => {
                SlotName::from_value(unsafe { s.delete_property.read_unchecked() })
            }
            Self::GetOwnPropertyDescriptor => {
                SlotName::from_value(unsafe { s.get_own_property_descriptor.read_unchecked() })
            }
            Self::DefineProperty => {
                SlotName::from_value(unsafe { s.define_property.read_unchecked() })
            }
            Self::GetPrototypeOf => {
                SlotName::from_value(unsafe { s.get_prototype_of.read_unchecked() })
            }
            Self::SetPrototypeOf => {
                SlotName::from_value(unsafe { s.set_prototype_of.read_unchecked() })
            }
            Self::IsExtensible => SlotName::from_value(unsafe { s.is_extensible.read_unchecked() }),
            Self::PreventExtensions => {
                SlotName::from_value(unsafe { s.prevent_extensions.read_unchecked() })
            }
            Self::OwnKeys => SlotName::from_value(unsafe { s.own_keys.read_unchecked() }),
            Self::Apply => SlotName::from_value(unsafe { s.apply.read_unchecked() }),
            Self::Construct => SlotName::from_value(unsafe { s.construct.read_unchecked() }),
        }
    }
}

/// Fast proxy check: one map-kind read.
pub fn is_proxy<'a>(_heap: &'a Heap, v: Tagged<'a, Value>) -> bool {
    v.get_as::<ProxyObject>().is_some()
}

/// Whether `v` is a valid ECMAScript receiver ([[ProxyTarget]] /
/// [[ProxyHandler]] validation, `Reflect.*` / `Object.*` argument
/// checks): a pure map-kind check, like V8's instance-type range test —
/// the oddball singletons (`null`, `undefined`, `true`, `false`, …)
/// carry `ODDBALL`-kind maps and fall out naturally.
pub fn is_js_receiver<'a>(heap: &'a Heap, v: Tagged<'a, Value>) -> bool {
    let Some(obj) = v.as_heap_object() else {
        return false;
    };
    obj.as_ref()
        .header
        .map
        .heap_ref(heap)
        .kind()
        .kind()
        .is_js_receiver()
}

/// (target, handler) of a proxy, read under a leaf scope. The raw words
/// must be rooted by the caller before any allocation.
fn parts<'a>(_heap: &'a Heap, proxy: Tagged<'a, Value>) -> Option<(Value, Value)> {
    let p = proxy.get_as::<ProxyObject>()?;
    Some(p.as_ref().parts())
}

pub fn allocate<'a>(
    heap: &'a mut Heap,
    scope: &HandleScope<'_>,
    target: Tagged<'_, Value>,
    handler: Tagged<'_, Value>,
) -> Tagged<'a, Value> {
    let target = scope.handle(target);
    let handler = scope.handle(handler);
    let known = heap.known();
    let map: Handle<'_, Map> = heap.no_gc(|heap| {
        let kind = target
            .as_tagged(heap)
            .as_heap_object()
            .expect("proxy target must be a JSReceiver")
            .as_ref()
            .header
            .map
            .heap_ref(heap)
            .kind();
        if kind.is_constructor() {
            known.proxy_constructor_map
        } else if kind.is_callable() {
            known.proxy_callable_map
        } else {
            known.proxy_map
        }
    });
    heap.allocate::<ProxyObject>(ProxyInit {
        map,
        target,
        handler,
    })
    .erase_type()
}

pub fn revoke(heap: &mut Heap, proxy: Tagged<'_, Value>) {
    heap.no_gc(|heap| {
        let Some(p) = proxy.get_as::<ProxyObject>() else {
            return;
        };
        if p.as_ref().is_revoked(heap) {
            return;
        }
        let host = proxy.erase();
        let null = heap.known().null.as_tagged(heap).erase_type();
        p.as_ref().target.set(heap, host, null);
        p.as_ref().handler.set(heap, host, null);
    });
}

enum TrapLookup<'s> {
    /// undefined/null: forward to the target.
    None,
    /// A callable... verified by `call_trap`.
    Trap(Handle<'s, Value>),
    Threw,
}

fn get_trap<'s>(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &'s HandleScope<'_>,
    handler: &Handle<'_, Value>,
    trap: Trap,
) -> Result<TrapLookup<'s>, VmError> {
    let name = heap.no_gc(|heap| trap.name(heap));
    match Runtime::get_property(
        vm,
        heap,
        state,
        unsafe { handler.read_unchecked() },
        name.value(),
    )? {
        Coercion::Threw => Ok(TrapLookup::Threw),
        Coercion::Value(v) => {
            let nullish = heap.no_gc(|heap| {
                v == heap.known().undefined.as_tagged(heap).erase()
                    || v == heap.known().null.as_tagged(heap).erase()
            });
            if nullish {
                Ok(TrapLookup::None)
            } else {
                Ok(TrapLookup::Trap(
                    scope.handle(unsafe { v.assume_valid(&*heap) }),
                ))
            }
        }
    }
}

fn call_trap(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    trap: &Handle<'_, Value>,
    args: &[Handle<'_, Value>],
) -> Result<Coercion, VmError> {
    if !Runtime::is_callable(heap, unsafe { trap.read_unchecked() }) {
        return Err(VmError::Message("proxy trap is not a function"));
    }
    let words: Vec<Value> = args.iter().map(|h| unsafe { h.read_unchecked() }).collect();
    let result = NativeContext::new(vm, heap, state)
        // Safety: fresh handle-slot word under the live scope borrow.
        .call(
            unsafe { Tagged::from_value_unchecked(trap.read_unchecked()) },
            scope.stage_words(&words),
        )?;
    let exception = heap.no_gc(|heap| heap.known().exception.as_tagged(heap).erase());
    if result == exception {
        Ok(Coercion::Threw)
    } else {
        Ok(Coercion::Value(result))
    }
}

/// Revocation check + (target, handler) extraction — the shared entry
/// of every internal method. Both values come back rooted in `scope`.
fn enter_trap<'s>(
    scope: &'s HandleScope<'_>,
    heap: &mut Heap,
    proxy: &Handle<'_, Value>,
    trap: Trap,
) -> Result<(Handle<'s, Value>, Handle<'s, Value>), VmError> {
    let (target, handler) = heap
        .no_gc(|heap| parts(heap, proxy.as_tagged(heap)))
        .expect("caller verified a proxy");
    let revoked = heap.no_gc(|heap| handler == heap.known().null.as_tagged(heap).erase());
    if revoked {
        return Err(revoked_error(trap));
    }
    Ok((
        scope.handle(unsafe { target.assume_valid(&*heap) }),
        scope.handle(unsafe { handler.assume_valid(&*heap) }),
    ))
}

fn revoked_error(trap: Trap) -> VmError {
    VmError::Message(match trap {
        Trap::Get => "cannot perform 'get' on a proxy that has been revoked",
        Trap::Set => "cannot perform 'set' on a proxy that has been revoked",
        Trap::Has => "cannot perform 'has' on a proxy that has been revoked",
        Trap::DeleteProperty => "cannot perform 'deleteProperty' on a proxy that has been revoked",
        Trap::GetOwnPropertyDescriptor => {
            "cannot perform 'getOwnPropertyDescriptor' on a proxy that has been revoked"
        }
        Trap::DefineProperty => "cannot perform 'defineProperty' on a proxy that has been revoked",
        Trap::GetPrototypeOf => "cannot perform 'getPrototypeOf' on a proxy that has been revoked",
        Trap::SetPrototypeOf => "cannot perform 'setPrototypeOf' on a proxy that has been revoked",
        Trap::IsExtensible => "cannot perform 'isExtensible' on a proxy that has been revoked",
        Trap::PreventExtensions => {
            "cannot perform 'preventExtensions' on a proxy that has been revoked"
        }
        Trap::OwnKeys => "cannot perform 'ownKeys' on a proxy that has been revoked",
        Trap::Apply => "cannot perform 'apply' on a proxy that has been revoked",
        Trap::Construct => "cannot perform 'construct' on a proxy that has been revoked",
    })
}

pub fn internal_own_descriptor(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    obj: Tagged<'_, Value>,
    key: Tagged<'_, Value>,
) -> Result<Flow<Option<PartialDescriptor>>, VmError> {
    state.handle_scope(|scope| {
        let obj = scope.handle(obj);
        let key = scope.handle(key);
        own_descriptor_h(vm, heap, state, &scope, &obj, &key)
    })
}

fn own_descriptor_h(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    obj: &Handle<'_, Value>,
    key: &Handle<'_, Value>,
) -> Result<Flow<Option<PartialDescriptor>>, VmError> {
    if !heap.no_gc(|heap| is_proxy(heap, obj.as_tagged(heap))) {
        let desc = heap.no_gc(|heap| {
            crate::lookup::ordinary_own_descriptor(heap, obj.as_tagged(heap), key.as_tagged(heap))
        });
        return Ok(Flow::Value(desc.as_ref().map(PartialDescriptor::from)));
    }
    let (target, handler) = heap
        .no_gc(|heap| parts(heap, obj.as_tagged(heap)))
        .expect("checked proxy above");
    let revoked = heap.no_gc(|heap| handler == heap.known().null.as_tagged(heap).erase());
    if revoked {
        return Err(revoked_error(Trap::GetOwnPropertyDescriptor));
    }
    let target = scope.handle(unsafe { target.assume_valid(&*heap) });
    let handler = scope.handle(unsafe { handler.assume_valid(&*heap) });
    match get_trap(
        vm,
        heap,
        state,
        scope,
        &handler,
        Trap::GetOwnPropertyDescriptor,
    )? {
        TrapLookup::Threw => Ok(Flow::Threw),
        TrapLookup::None => own_descriptor_h(vm, heap, state, scope, &target, key),
        TrapLookup::Trap(t) => {
            let result = call_trap(vm, heap, state, scope, &t, &[handler, target, *key])?;
            let Coercion::Value(result) = result else {
                return Ok(Flow::Threw);
            };
            let undefined = heap.no_gc(|heap| heap.known().undefined.as_tagged(heap).erase());
            if result == undefined {
                return Ok(Flow::Value(None));
            }
            match Runtime::to_property_descriptor(vm, heap, state, result)? {
                Some(partial) => Ok(Flow::Value(Some(partial))),
                None => Ok(Flow::Threw),
            }
        }
    }
}

/// [[IsExtensible]] through proxies, *without* the trap-must-match
/// invariant (that check belongs to the `Object.isExtensible` entry
/// point; other internal methods only read the value).
pub fn internal_is_extensible(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    obj: Tagged<'_, Value>,
) -> Result<Flow<bool>, VmError> {
    state.handle_scope(|scope| {
        let obj = scope.handle(obj);
        is_extensible_h(vm, heap, state, &scope, &obj)
    })
}

fn is_extensible_h(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    obj: &Handle<'_, Value>,
) -> Result<Flow<bool>, VmError> {
    if !heap.no_gc(|heap| is_proxy(heap, obj.as_tagged(heap))) {
        return Ok(Flow::Value(heap.no_gc(|heap| {
            obj.as_tagged(heap)
                .as_heap_object()
                .is_some_and(|o| o.as_ref().map_ref(heap).kind().is_extendable())
        })));
    }
    let (target, handler) = heap
        .no_gc(|heap| parts(heap, obj.as_tagged(heap)))
        .expect("checked proxy above");
    let revoked = heap.no_gc(|heap| handler == heap.known().null.as_tagged(heap).erase());
    if revoked {
        return Err(revoked_error(Trap::IsExtensible));
    }
    let target = scope.handle(unsafe { target.assume_valid(&*heap) });
    let handler = scope.handle(unsafe { handler.assume_valid(&*heap) });
    match get_trap(vm, heap, state, scope, &handler, Trap::IsExtensible)? {
        TrapLookup::Threw => Ok(Flow::Threw),
        TrapLookup::None => is_extensible_h(vm, heap, state, scope, &target),
        TrapLookup::Trap(t) => {
            let result = call_trap(vm, heap, state, scope, &t, &[handler, target])?;
            let Coercion::Value(result) = result else {
                return Ok(Flow::Threw);
            };
            Ok(Flow::Value(heap.no_gc(|heap| {
                Convert::is_truthy(heap, unsafe { result.assume_valid(heap) })
            })))
        }
    }
}

// ---- forwards (trap absent → run the operation on the target) ---------------

/// A rooted partial descriptor: value-ish fields are handle-rooted so
/// the payload survives trap calls and descriptor-object allocation.
struct RootedPartial<'s> {
    value: Option<Handle<'s, Value>>,
    get: Option<Handle<'s, Value>>,
    set: Option<Handle<'s, Value>>,
    writable: Option<bool>,
    enumerable: Option<bool>,
    configurable: Option<bool>,
}

impl<'s> RootedPartial<'s> {
    fn new(scope: &'s HandleScope<'_>, heap: &Heap, partial: &PartialDescriptor) -> Self {
        Self {
            value: partial
                .value
                .map(|v| scope.handle(unsafe { v.assume_valid(heap) })),
            get: partial
                .get
                .map(|v| scope.handle(unsafe { v.assume_valid(heap) })),
            set: partial
                .set
                .map(|v| scope.handle(unsafe { v.assume_valid(heap) })),
            writable: partial.writable,
            enumerable: partial.enumerable,
            configurable: partial.configurable,
        }
    }

    fn is_data_descriptor(&self) -> bool {
        self.partial().is_data_descriptor()
    }

    /// The words are fresh reads out of rooted slots; the result must be
    /// consumed within the same expression (no allocation in between).
    fn partial(&self) -> PartialDescriptor {
        PartialDescriptor {
            value: self.value.map(|h| unsafe { h.read_unchecked() }),
            get: self.get.map(|h| unsafe { h.read_unchecked() }),
            set: self.set.map(|h| unsafe { h.read_unchecked() }),
            writable: self.writable,
            enumerable: self.enumerable,
            configurable: self.configurable,
        }
    }
}

/// CreateDataProperty-style full descriptor ({w,e,c} true) — the payload
/// of OrdinarySet's receiver-define step.
fn create_data_partial<'s>(
    _scope: &'s HandleScope<'_>,
    value: &Handle<'s, Value>,
) -> RootedPartial<'s> {
    RootedPartial {
        value: Some(*value),
        get: None,
        set: None,
        writable: Some(true),
        enumerable: Some(true),
        configurable: Some(true),
    }
}

/// FromPropertyDescriptor (ES 6.2.6.4): a descriptor object carrying
/// exactly the fields present in the partial (what a
/// `defineProperty`/`getOwnPropertyDescriptor` trap receives/returns).
/// Allocates, so the payload must be rooted (`RootedPartial`).
fn descriptor_object(
    heap: &mut Heap,
    state: &ContextState,
    partial: &RootedPartial<'_>,
) -> Result<Value, VmError> {
    state.handle_scope(|scope| {
        let obj = heap
            .new_object(&scope, heap.known().object_initial_map, GcSlice::EMPTY)
            .into_handle(&scope);
        let s = heap.known().strings;
        // spec field order: value, writable, get, set, enumerable, configurable
        let mut fields: Vec<(SlotName, Handle<'_, Value>)> = Vec::new();
        if let Some(v) = &partial.value {
            fields.push((
                SlotName::from_value(unsafe { s.value.read_unchecked() }),
                *v,
            ));
        }
        if let Some(b) = partial.writable {
            fields.push((
                SlotName::from_value(unsafe { s.writable.read_unchecked() }),
                scope.handle(Convert::boolean(&*heap, b)),
            ));
        }
        if let Some(v) = &partial.get {
            fields.push((SlotName::from_value(unsafe { s.get.read_unchecked() }), *v));
        }
        if let Some(v) = &partial.set {
            fields.push((SlotName::from_value(unsafe { s.set.read_unchecked() }), *v));
        }
        if let Some(b) = partial.enumerable {
            fields.push((
                SlotName::from_value(unsafe { s.enumerable.read_unchecked() }),
                scope.handle(Convert::boolean(&*heap, b)),
            ));
        }
        if let Some(b) = partial.configurable {
            fields.push((
                SlotName::from_value(unsafe { s.configurable.read_unchecked() }),
                scope.handle(Convert::boolean(&*heap, b)),
            ));
        }
        for (name, value) in fields {
            let name = scope.handle(unsafe { name.tagged(&*heap) });
            crate::Object::define_own_property(
                heap,
                &scope,
                obj,
                name,
                PropertyDescriptor::data(unsafe { value.read_unchecked() }),
            )?;
        }
        Ok(unsafe { obj.read_unchecked() })
    })
}

/// Store `value` at element index `i` of an array object, growing the
/// elements backing store and updating `length` when `i` is past the
/// end. Handle-rooted variant of `objects::object::store_array_element`
/// (the exported one takes `Tagged` args, which cannot be re-read from
/// handles under a `&mut Heap`).
fn define_array_element(
    heap: &mut Heap,
    scope: &HandleScope<'_>,
    receiver: &Handle<'_, Object>,
    i: usize,
    value: &Handle<'_, Value>,
) -> Result<(), VmError> {
    let new_len = i.checked_add(1).ok_or(VmError::OutOfBounds)?;
    let grows = heap.no_gc(|heap| {
        let obj = receiver.heap_ref(heap);
        if !obj.as_ref().is_array(heap) {
            return Err(VmError::Type);
        }
        // grow only when the store index is past the physical backing
        // store (its capacity), not the logical length: sequential
        // appends with headroom must not reallocate every time
        let capacity = obj
            .as_ref()
            .elements_array(heap)
            .map(|e| e.len())
            .unwrap_or(0);
        Ok(i >= capacity)
    })?;

    if grows {
        let mut values = Vec::new();
        heap.no_gc(|heap| -> Result<(), VmError> {
            let obj = receiver.heap_ref(heap);
            let elements = obj.as_ref().elements_array(heap).ok_or(VmError::Type)?;
            let keep = obj.as_ref().length().min(elements.len());
            let capacity = (new_len + (new_len >> 1) + 16).max(elements.len());
            values = Vec::with_capacity(capacity);
            for k in 0..keep {
                values.push(elements.at(heap, k).erase());
            }
            // Safety: fresh root-slot read under the anchor.
            values.resize(capacity, unsafe { heap.known().the_hole.read_unchecked() });
            Ok(())
        })?;
        values[i] = unsafe { value.read_unchecked() };
        let elements = heap.allocate_handle::<FixedArray>(scope.stage_words(&values), scope);
        heap.no_gc(|heap| {
            let obj = receiver.heap_ref(heap);
            obj.as_ref().elements.set(
                heap,
                obj.as_ref().erase(),
                elements.as_tagged(heap).erase_type(),
            );
            obj.as_ref()
                .length
                .set(heap, obj.as_ref().erase(), Smi::new(new_len as i64));
        });
    } else {
        heap.no_gc(|heap| -> Result<(), VmError> {
            let obj = receiver.heap_ref(heap);
            let elements = obj.as_ref().elements_array(heap).ok_or(VmError::Type)?;
            elements.set(heap, i, value.as_tagged(heap));
            // a store inside the physical capacity but past the logical
            // length still extends the array
            if i >= obj.as_ref().length() {
                obj.as_ref()
                    .length
                    .set(heap, obj.as_ref().erase(), Smi::new(new_len as i64));
            }
            Ok(())
        })?;
    }
    Ok(())
}

/// `[[DefineOwnProperty]]` dispatch: proxies run their `defineProperty`
/// trap (with full spec validation), ordinary objects complete the
/// partial against their current descriptor and define through the
/// transition machinery.
pub fn define_internal(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    obj: Tagged<'_, Value>,
    name: Tagged<'_, Value>,
    partial: PartialDescriptor,
) -> Result<Flow<bool>, VmError> {
    state.handle_scope(|scope| {
        let obj = scope.handle(obj);
        let name = scope.handle(name);
        define_internal_h(vm, heap, state, &scope, &obj, &name, partial)
    })
}

fn define_internal_h(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    obj: &Handle<'_, Value>,
    name: &Handle<'_, Value>,
    partial: PartialDescriptor,
) -> Result<Flow<bool>, VmError> {
    if heap.no_gc(|heap| is_proxy(heap, obj.as_tagged(heap))) {
        return proxy_define_h(vm, heap, state, scope, obj, name, partial);
    }
    let undefined = heap.no_gc(|heap| heap.known().undefined.as_tagged(heap).erase());
    // dense array elements: element defines land in the backing
    // store, not as descriptors
    let element = heap.no_gc(|heap| {
        let on_array = obj
            .as_tagged(heap)
            .as_heap_object()
            .is_some_and(|o| o.as_ref().is_array(heap));
        if !on_array {
            return None;
        }
        match crate::classify_key(heap, name.as_tagged(heap)) {
            Ok(Key::Element(i)) => Some(i),
            _ => None,
        }
    });
    if let Some(i) = element {
        let completed = partial.complete_against(undefined, None);
        let PropertyDescriptor::Data { value, .. } = completed else {
            return Err(VmError::Type);
        };
        let value = scope.handle(unsafe { value.assume_valid(&*heap) });
        let array = scope
            .cast::<Object>(obj.as_tagged(&*heap))
            .expect("array checked above");
        define_array_element(heap, scope, &array, i, &value)?;
        return Ok(Flow::Value(true));
    }
    let current = heap.no_gc(|heap| {
        crate::lookup::ordinary_own_descriptor(heap, obj.as_tagged(heap), name.as_tagged(heap))
    });
    // no allocation between reading `current` and the define call
    // (the define path roots its inputs before allocating)
    let full = partial.complete_against(undefined, current.as_ref());
    let defined = crate::Object::define_own_property_values(
        heap,
        scope,
        unsafe { obj.read_unchecked() },
        SlotName::from_value(unsafe { name.read_unchecked() }),
        full,
    )?;
    Ok(Flow::Value(defined))
}

/// OrdinarySet (receiver-aware) for an ordinary target with a proxy
/// (or arbitrary) receiver: the spec's define-on-receiver shape.
fn ordinary_set_forward(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    target: &Handle<'_, Value>,
    receiver: &Handle<'_, Value>,
    name: &Handle<'_, Value>,
    value: &Handle<'_, Value>,
) -> Result<Coercion, VmError> {
    // array element keys on arrays route through the define path
    // (which stores into the backing store)
    let classified = heap.no_gc(|heap| {
        (
            matches!(
                crate::classify_key(heap, name.as_tagged(heap)),
                Ok(Key::Element(_))
            ),
            target
                .as_tagged(heap)
                .as_heap_object()
                .is_some_and(|o| o.as_ref().is_array(heap)),
        )
    });
    if let (true, true) = classified {
        let partial = create_data_partial(scope, value);
        let flow = define_internal_h(vm, heap, state, scope, receiver, name, partial.partial())?;
        return define_flow_to_coercion(heap, flow);
    }

    // owned summary of the chain lookup starting at the target
    // (Lookup borrows its no-GC scope, so it is consumed inside)
    enum SetLookup {
        DataWritable,
        DataReadonly,
        Setter(Value),
        NotFound,
    }
    let lookup = heap.no_gc(|heap| {
        match target
            .as_tagged(heap)
            .lookup(heap, SlotName::from_value(unsafe { name.read_unchecked() }))
        {
            Lookup::Data { flags, .. } => {
                if flags.is_writable() {
                    SetLookup::DataWritable
                } else {
                    SetLookup::DataReadonly
                }
            }
            Lookup::Accessor { pair, .. } => SetLookup::Setter(pair.as_ref().set.inner()),
            Lookup::NotFound => SetLookup::NotFound,
        }
    });
    match lookup {
        SetLookup::DataReadonly => Ok(Coercion::Value(Convert::boolean(heap, false).erase())),
        SetLookup::DataWritable | SetLookup::NotFound => {
            let partial = create_data_partial(scope, value);
            let flow =
                define_internal_h(vm, heap, state, scope, receiver, name, partial.partial())?;
            define_flow_to_coercion(heap, flow)
        }
        SetLookup::Setter(setter) => {
            let undefined = heap.no_gc(|heap| heap.known().undefined.as_tagged(heap).erase());
            if setter == undefined {
                return Ok(Coercion::Value(Convert::boolean(heap, false).erase()));
            }
            let setter = scope.handle(unsafe { setter.assume_valid(&*heap) });
            let words = [unsafe { receiver.read_unchecked() }, unsafe {
                value.read_unchecked()
            }];
            let result = NativeContext::new(vm, heap, state).call(
                // Safety: fresh handle-slot word.
                unsafe { Tagged::from_value_unchecked(setter.read_unchecked()) },
                scope.stage_words(&words),
            )?;
            let exception = heap.no_gc(|heap| heap.known().exception.as_tagged(heap).erase());
            if result == exception {
                Ok(Coercion::Threw)
            } else {
                Ok(Coercion::Value(Convert::boolean(heap, true).erase()))
            }
        }
    }
}

fn define_flow_to_coercion(heap: &Heap, flow: Flow<bool>) -> Result<Coercion, VmError> {
    Ok(match flow {
        Flow::Threw => Coercion::Threw,
        Flow::Value(b) => Coercion::Value(Convert::boolean(heap, b).erase()),
    })
}

// ---- the internal methods (ES 20.2.5.x) ------------------------------------

/// Proxy `[[Get]]` (ES 20.2.5.8). `receiver` is the [[Get]] receiver —
/// the proxy itself at entry points, the *original* receiver when
/// forwarding through a chain of proxies.
pub fn get(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    proxy: Tagged<'_, Value>,
    receiver: Tagged<'_, Value>,
    name: Tagged<'_, Value>,
) -> Result<Coercion, VmError> {
    state.handle_scope(|scope| {
        let proxy = scope.handle(proxy);
        let receiver = scope.handle(receiver);
        let name = scope.handle(name);
        get_h(vm, heap, state, &scope, &proxy, &receiver, &name)
    })
}

fn get_h(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    proxy: &Handle<'_, Value>,
    receiver: &Handle<'_, Value>,
    name: &Handle<'_, Value>,
) -> Result<Coercion, VmError> {
    let (target, handler) = enter_trap(scope, heap, proxy, Trap::Get)?;
    match get_trap(vm, heap, state, scope, &handler, Trap::Get)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => {
            // forward: lookup on the target, getter `this` = the
            // original receiver (a proxy target re-enters its own
            // `get` trap with the same receiver)
            Runtime::get_property_on(
                vm,
                heap,
                state,
                // Safety: fresh handle-slot words, rooted until consumed.
                unsafe { target.read_unchecked() },
                unsafe { receiver.read_unchecked() },
                unsafe { name.read_unchecked() },
            )
        }
        TrapLookup::Trap(t) => {
            let result = call_trap(
                vm,
                heap,
                state,
                scope,
                &t,
                &[handler, target, *name, *receiver],
            )?;
            let Coercion::Value(result) = result else {
                return Ok(Coercion::Threw);
            };
            let result = scope.handle(unsafe { result.assume_valid(&*heap) });
            // invariant (steps 9-11): a trap cannot lie about
            // non-configurable data / accessor properties
            let desc = own_descriptor_h(vm, heap, state, scope, &target, name)?;
            let Flow::Value(desc) = desc else {
                return Ok(Coercion::Threw);
            };
            if let Some(d) = desc.as_ref().filter(|d| d.configurable == Some(false)) {
                heap.no_gc(|heap| -> Result<(), VmError> {
                    let undefined = heap.known().undefined.as_tagged(heap).erase();
                    if d.is_data_descriptor()
                        && d.writable == Some(false)
                        && !Compare::same_value(heap, result.as_tagged(heap), unsafe {
                            d.value.unwrap_or(undefined).assume_valid(heap)
                        })
                    {
                        return Err(VmError::Message(
                            "proxy get trap must match a non-writable, non-configurable property",
                        ));
                    }
                    if d.is_accessor_descriptor()
                        && d.get.as_ref().is_some_and(|g| *g == undefined)
                        && result.as_tagged(heap).erase() != undefined
                    {
                        return Err(VmError::Message(
                            "proxy get trap must return undefined for an accessor without a getter",
                        ));
                    }
                    Ok(())
                })?;
            }
            Ok(Coercion::Value(unsafe { result.read_unchecked() }))
        }
    }
}

/// Proxy `[[Set]]` (ES 20.2.5.10).
pub fn set(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    proxy: Tagged<'_, Value>,
    name: Tagged<'_, Value>,
    value: Tagged<'_, Value>,
    receiver: Tagged<'_, Value>,
) -> Result<Coercion, VmError> {
    state.handle_scope(|scope| {
        let proxy = scope.handle(proxy);
        let name = scope.handle(name);
        let value = scope.handle(value);
        let receiver = scope.handle(receiver);
        set_h(vm, heap, state, &scope, &proxy, &name, &value, &receiver)
    })
}

fn set_h(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    proxy: &Handle<'_, Value>,
    name: &Handle<'_, Value>,
    value: &Handle<'_, Value>,
    receiver: &Handle<'_, Value>,
) -> Result<Coercion, VmError> {
    let (target, handler) = enter_trap(scope, heap, proxy, Trap::Set)?;
    match get_trap(vm, heap, state, scope, &handler, Trap::Set)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => {
            if heap.no_gc(|heap| is_proxy(heap, target.as_tagged(heap))) {
                set_h(vm, heap, state, scope, &target, name, value, receiver)
            } else {
                ordinary_set_forward(vm, heap, state, scope, &target, receiver, name, value)
            }
        }
        TrapLookup::Trap(t) => {
            let result = call_trap(
                vm,
                heap,
                state,
                scope,
                &t,
                &[handler, target, *name, *value, *receiver],
            )?;
            let Coercion::Value(result) = result else {
                return Ok(Coercion::Threw);
            };
            let truthy =
                heap.no_gc(|heap| Convert::is_truthy(heap, unsafe { result.assume_valid(heap) }));
            if !truthy {
                // [[Set]] returned false; sloppy stores ignore it
                return Ok(Coercion::Value(Convert::boolean(heap, false).erase()));
            }
            // invariant (steps 11-13)
            let desc = own_descriptor_h(vm, heap, state, scope, &target, name)?;
            let Flow::Value(desc) = desc else {
                return Ok(Coercion::Threw);
            };
            if let Some(d) = desc.as_ref().filter(|d| d.configurable == Some(false)) {
                heap.no_gc(|heap| -> Result<(), VmError> {
                    let undefined = heap.known().undefined.as_tagged(heap).erase();
                    if d.is_data_descriptor()
                        && d.writable == Some(false)
                        && !Compare::same_value(
                            heap,
                            value.as_tagged(heap),
                            unsafe { d.value.unwrap_or(undefined).assume_valid(heap) },
                        )
                    {
                        return Err(VmError::Message(
                            "proxy set trap must match a non-writable, non-configurable property",
                        ));
                    }
                    if d.is_accessor_descriptor()
                        && d.set.as_ref().is_some_and(|s| *s == undefined)
                    {
                        return Err(VmError::Message(
                            "proxy set trap may not report success for an accessor without a setter",
                        ));
                    }
                    Ok(())
                })?;
            }
            Ok(Coercion::Value(Convert::boolean(heap, true).erase()))
        }
    }
}

/// Proxy `[[HasProperty]]` (ES 20.2.5.9).
pub fn has(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    proxy: Tagged<'_, Value>,
    name: Tagged<'_, Value>,
) -> Result<Coercion, VmError> {
    state.handle_scope(|scope| {
        let proxy = scope.handle(proxy);
        let name = scope.handle(name);
        has_h(vm, heap, state, &scope, &proxy, &name)
    })
}

fn has_h(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    proxy: &Handle<'_, Value>,
    name: &Handle<'_, Value>,
) -> Result<Coercion, VmError> {
    let (target, handler) = enter_trap(scope, heap, proxy, Trap::Has)?;
    match get_trap(vm, heap, state, scope, &handler, Trap::Has)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => {
            if heap.no_gc(|heap| is_proxy(heap, target.as_tagged(heap))) {
                has_h(vm, heap, state, scope, &target, name)
            } else {
                let found = heap.no_gc(|heap| {
                    has_property(
                        heap,
                        target.as_tagged(heap),
                        SlotName::from_value(unsafe { name.read_unchecked() }),
                    )
                });
                Ok(Coercion::Value(Convert::boolean(heap, found).erase()))
            }
        }
        TrapLookup::Trap(t) => {
            let result = call_trap(vm, heap, state, scope, &t, &[handler, target, *name])?;
            let Coercion::Value(result) = result else {
                return Ok(Coercion::Threw);
            };
            let truthy =
                heap.no_gc(|heap| Convert::is_truthy(heap, unsafe { result.assume_valid(heap) }));
            if truthy {
                return Ok(Coercion::Value(Convert::boolean(heap, true).erase()));
            }
            // invariant (steps 10-12): cannot hide non-configurable
            // or existing-on-non-extensible-target properties
            let desc = own_descriptor_h(vm, heap, state, scope, &target, name)?;
            let Flow::Value(desc) = desc else {
                return Ok(Coercion::Threw);
            };
            if let Some(d) = desc {
                if d.configurable == Some(false) {
                    return Err(VmError::Message(
                        "proxy has trap may not hide a non-configurable property",
                    ));
                }
                let ext = is_extensible_h(vm, heap, state, scope, &target)?;
                let Flow::Value(extensible) = ext else {
                    return Ok(Coercion::Threw);
                };
                if !extensible {
                    return Err(VmError::Message(
                        "proxy has trap may not hide a property of a non-extensible target",
                    ));
                }
            }
            Ok(Coercion::Value(Convert::boolean(heap, false).erase()))
        }
    }
}

/// Proxy `[[Delete]]` (ES 20.2.5.4). The strict-mode false→TypeError
/// translation is the caller's (`delete` sloppy/strict natives).
pub fn delete(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    proxy: Tagged<'_, Value>,
    key: Tagged<'_, Value>,
) -> Result<Coercion, VmError> {
    state.handle_scope(|scope| {
        let proxy = scope.handle(proxy);
        let key = scope.handle(key);
        delete_h(vm, heap, state, &scope, &proxy, &key)
    })
}

fn delete_h(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    proxy: &Handle<'_, Value>,
    key: &Handle<'_, Value>,
) -> Result<Coercion, VmError> {
    let (target, handler) = enter_trap(scope, heap, proxy, Trap::DeleteProperty)?;
    match get_trap(vm, heap, state, scope, &handler, Trap::DeleteProperty)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => {
            if heap.no_gc(|heap| is_proxy(heap, target.as_tagged(heap))) {
                delete_h(vm, heap, state, scope, &target, key)
            } else {
                let target_obj = scope
                    .cast::<Object>(target.as_tagged(&*heap))
                    .expect("ordinary target");
                let deleted =
                    crate::Object::delete_own_property(heap, scope, target_obj, unsafe {
                        key.read_unchecked()
                    })?;
                Ok(Coercion::Value(Convert::boolean(heap, deleted).erase()))
            }
        }
        TrapLookup::Trap(t) => {
            let result = call_trap(vm, heap, state, scope, &t, &[handler, target, *key])?;
            let Coercion::Value(result) = result else {
                return Ok(Coercion::Threw);
            };
            let truthy =
                heap.no_gc(|heap| Convert::is_truthy(heap, unsafe { result.assume_valid(heap) }));
            if !truthy {
                return Ok(Coercion::Value(Convert::boolean(heap, false).erase()));
            }
            // invariant: cannot claim deletion of non-configurable
            // (or existing-on-non-extensible-target) properties
            let desc = own_descriptor_h(vm, heap, state, scope, &target, key)?;
            let Flow::Value(desc) = desc else {
                return Ok(Coercion::Threw);
            };
            if let Some(d) = desc {
                if d.configurable == Some(false) {
                    return Err(VmError::Message(
                        "proxy deleteProperty trap may not delete a non-configurable property",
                    ));
                }
                let ext = is_extensible_h(vm, heap, state, scope, &target)?;
                let Flow::Value(extensible) = ext else {
                    return Ok(Coercion::Threw);
                };
                if !extensible {
                    return Err(VmError::Message(
                        "proxy deleteProperty trap may not delete a property of a non-extensible target",
                    ));
                }
            }
            Ok(Coercion::Value(Convert::boolean(heap, true).erase()))
        }
    }
}

/// Proxy `[[DefineOwnProperty]]` (ES 20.2.5.6).
pub fn proxy_define(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    proxy: Tagged<'_, Value>,
    name: Tagged<'_, Value>,
    partial: PartialDescriptor,
) -> Result<Flow<bool>, VmError> {
    state.handle_scope(|scope| {
        let proxy = scope.handle(proxy);
        let name = scope.handle(name);
        proxy_define_h(vm, heap, state, &scope, &proxy, &name, partial)
    })
}

fn proxy_define_h(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    proxy: &Handle<'_, Value>,
    name: &Handle<'_, Value>,
    partial: PartialDescriptor,
) -> Result<Flow<bool>, VmError> {
    let (target, handler) = enter_trap(scope, heap, proxy, Trap::DefineProperty)?;
    let partial = RootedPartial::new(scope, &*heap, &partial);

    match get_trap(vm, heap, state, scope, &handler, Trap::DefineProperty)? {
        TrapLookup::Threw => Ok(Flow::Threw),
        TrapLookup::None => {
            define_internal_h(vm, heap, state, scope, &target, name, partial.partial())
        }
        TrapLookup::Trap(t) => {
            let desc_obj = descriptor_object(heap, state, &partial)?;
            let desc_obj = scope.handle(unsafe { desc_obj.assume_valid(&*heap) });
            let result = call_trap(
                vm,
                heap,
                state,
                scope,
                &t,
                &[handler, target, *name, desc_obj],
            )?;
            let Coercion::Value(result) = result else {
                return Ok(Flow::Threw);
            };
            if !heap.no_gc(|heap| Convert::is_truthy(heap, unsafe { result.assume_valid(heap) })) {
                return Ok(Flow::Value(false));
            }
            // extensible first (a proxy target's trap allocates)
            let ext = is_extensible_h(vm, heap, state, scope, &target)?;
            let Flow::Value(extensible) = ext else {
                return Ok(Flow::Threw);
            };
            let desc = own_descriptor_h(vm, heap, state, scope, &target, name)?;
            let Flow::Value(desc) = desc else {
                return Ok(Flow::Threw);
            };
            let setting_config_false = partial.configurable == Some(false);
            match desc.as_ref() {
                None => {
                    if !extensible {
                        return Err(VmError::Message(
                            "proxy defineProperty trap may not add a property to a non-extensible target",
                        ));
                    }
                    if setting_config_false {
                        return Err(VmError::Message(
                            "proxy defineProperty trap may not claim a non-configurable new property",
                        ));
                    }
                }
                Some(d) => {
                    let compatible = heap.no_gc(|heap| {
                        let p = partial.partial();
                        is_compatible_property_descriptor(heap, extensible, &p, Some(d))
                    });
                    if !compatible {
                        return Err(VmError::Message(
                            "proxy defineProperty trap returned an incompatible descriptor",
                        ));
                    }
                    if setting_config_false && d.configurable != Some(false) {
                        return Err(VmError::Message(
                            "proxy defineProperty trap may not make a configurable property non-configurable",
                        ));
                    }
                    if d.configurable == Some(false)
                        && d.is_data_descriptor()
                        && d.writable == Some(true)
                        && partial.is_data_descriptor()
                        && partial.writable == Some(false)
                    {
                        return Err(VmError::Message(
                            "proxy defineProperty trap may not make a non-configurable property non-writable",
                        ));
                    }
                }
            }
            // the trap owns the define; its true result stands
            Ok(Flow::Value(true))
        }
    }
}

/// Proxy `[[Call]]` (ES 20.2.5.15): the `apply` trap, or a plain call
/// of the target. `args` includes the receiver (`this`) at index 0.
pub fn apply(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    proxy: Tagged<'_, Value>,
    args: &[Tagged<'_, Value>],
) -> Result<Coercion, VmError> {
    state.handle_scope(|scope| {
        let proxy = scope.handle(proxy);
        let args: Vec<Handle<'_, Value>> = args.iter().map(|a| scope.handle(*a)).collect();
        apply_h(vm, heap, state, &scope, &proxy, &args)
    })
}

fn apply_h(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    proxy: &Handle<'_, Value>,
    args: &[Handle<'_, Value>],
) -> Result<Coercion, VmError> {
    let (target, handler) = enter_trap(scope, heap, proxy, Trap::Apply)?;
    let this_arg = match args.first() {
        Some(h) => *h,
        None => {
            scope.handle(unsafe { heap.known().undefined.read_unchecked().assume_valid(&*heap) })
        }
    };
    match get_trap(vm, heap, state, scope, &handler, Trap::Apply)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => {
            let mut all: Vec<Value> = Vec::with_capacity(args.len());
            all.push(unsafe { this_arg.read_unchecked() });
            for h in &args[1..] {
                all.push(unsafe { h.read_unchecked() });
            }
            let result = NativeContext::new(vm, heap, state).call(
                // Safety: fresh handle-slot word.
                unsafe { Tagged::from_value_unchecked(target.read_unchecked()) },
                scope.stage_words(&all),
            )?;
            let exception = heap.no_gc(|heap| heap.known().exception.as_tagged(heap).erase());
            if result == exception {
                Ok(Coercion::Threw)
            } else {
                Ok(Coercion::Value(result))
            }
        }
        TrapLookup::Trap(t) => {
            // args: (target, thisArg, argumentsList)
            let words: Vec<Value> = args[1..]
                .iter()
                .map(|h| unsafe { h.read_unchecked() })
                .collect();
            let arr = heap
                .new_array(scope, scope.stage_words(&words))
                .into_handle(scope)
                .erase();
            let result = call_trap(
                vm,
                heap,
                state,
                scope,
                &t,
                &[handler, target, this_arg, arr],
            )?;
            Ok(result)
        }
    }
}

/// Proxy `[[Construct]]` (ES 20.2.5.5): the `construct` trap, or a
/// construct of the target with the original `new.target`. `args` are
/// the constructor arguments (no synthesized receiver).
pub fn construct(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    proxy: Tagged<'_, Value>,
    args: &[Tagged<'_, Value>],
    new_target: Tagged<'_, Value>,
) -> Result<Coercion, VmError> {
    state.handle_scope(|scope| {
        let proxy = scope.handle(proxy);
        let args: Vec<Handle<'_, Value>> = args.iter().map(|a| scope.handle(*a)).collect();
        let new_target = scope.handle(new_target);
        construct_h(vm, heap, state, &scope, &proxy, &args, &new_target)
    })
}

fn construct_h(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    proxy: &Handle<'_, Value>,
    args: &[Handle<'_, Value>],
    new_target: &Handle<'_, Value>,
) -> Result<Coercion, VmError> {
    let (target, handler) = enter_trap(scope, heap, proxy, Trap::Construct)?;
    match get_trap(vm, heap, state, scope, &handler, Trap::Construct)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => {
            // forward: the full target [[Construct]] — synthesize the
            // receiver from new.target (the hole for derived class
            // constructors), run, and prefer an object result
            let derived = heap.no_gc(|heap| {
                target
                    .as_tagged(heap)
                    .as_heap_object()
                    .and_then(|obj| {
                        obj.as_ref()
                            .header
                            .map
                            .heap_ref(heap)
                            .kind()
                            .is_class_constructor()
                            .then(|| obj.as_ref().callable_info(heap))
                    })
                    .flatten()
                    .is_some_and(|info| info.function_kind().is_derived_class_constructor())
            });
            let receiver = if derived {
                let hole = heap.no_gc(|heap| heap.known().the_hole.as_tagged(heap).erase());
                scope.handle(unsafe { hole.assume_valid(&*heap) })
            } else {
                let Some(r) = Runtime::create_construct_receiver_value(vm, heap, state, unsafe {
                    new_target.read_unchecked()
                })?
                else {
                    return Ok(Coercion::Threw);
                };
                scope.handle(unsafe { r.assume_valid(&*heap) })
            };
            let mut all: Vec<Value> = Vec::with_capacity(args.len() + 1);
            all.push(unsafe { receiver.read_unchecked() });
            for h in args {
                all.push(unsafe { h.read_unchecked() });
            }
            let result = NativeContext::new(vm, heap, state).call_construct(
                // Safety: fresh handle-slot word.
                unsafe { Tagged::from_value_unchecked(target.read_unchecked()) },
                // Safety: fresh handle-slot word.
                unsafe { Tagged::from_value_unchecked(new_target.read_unchecked()) },
                scope.stage_words(&all),
            )?;
            let exception = heap.no_gc(|heap| heap.known().exception.as_tagged(heap).erase());
            if result == exception {
                return Ok(Coercion::Threw);
            }
            let result = scope.handle(unsafe { result.assume_valid(&*heap) });
            if heap.no_gc(|heap| Convert::is_primitive(heap, result.as_tagged(heap))) {
                if derived {
                    // a derived constructor may only return objects
                    return Err(VmError::Type);
                }
                return Ok(Coercion::Value(unsafe { receiver.read_unchecked() }));
            }
            Ok(Coercion::Value(unsafe { result.read_unchecked() }))
        }
        TrapLookup::Trap(t) => {
            // args: (target, argumentsList, newTarget)
            let words: Vec<Value> = args.iter().map(|h| unsafe { h.read_unchecked() }).collect();
            let arr = heap
                .new_array(scope, scope.stage_words(&words))
                .into_handle(scope)
                .erase();
            let result = call_trap(
                vm,
                heap,
                state,
                scope,
                &t,
                &[handler, target, arr, *new_target],
            )?;
            let Coercion::Value(result) = result else {
                return Ok(Coercion::Threw);
            };
            let result = scope.handle(unsafe { result.assume_valid(&*heap) });
            // invariant: the trap must return an object
            if heap.no_gc(|heap| Convert::is_primitive(heap, result.as_tagged(heap))) {
                return Err(VmError::Message(
                    "proxy construct trap must return an object",
                ));
            }
            Ok(Coercion::Value(unsafe { result.read_unchecked() }))
        }
    }
}

/// Ordinary `[[PreventExtensions]]`: clone the map without the
/// EXTENDABLE bit (maps are shared, so the clone isolates the object).
fn ordinary_prevent_extensions(heap: &mut Heap, scope: &HandleScope<'_>, obj: Handle<'_, Object>) {
    use crate::{MapInit, MapKind};
    let (kind, prototype, descriptors) = heap.no_gc(|heap| {
        let map = obj.heap_ref(heap).map_ref(heap);
        (
            map.kind(),
            map.prototype.inner(),
            map.descriptors()
                .iter()
                .map(|d| {
                    (
                        d.name(),
                        d.flags(),
                        scope.handle(unsafe { d.value.inner().assume_valid(heap) }),
                    )
                })
                .collect::<Vec<_>>(),
        )
    });
    if !kind.is_extendable() {
        return; // already non-extensible (idempotent)
    }
    let prototype = scope.handle(unsafe { prototype.assume_valid(&*heap) });
    heap.allocate_token_enter_heap(Map::layout_for(descriptors.len()), |token, heap| {
        let obj_ref = obj.heap_ref(heap);
        let new_map = token.allocate::<Map>(MapInit {
            kind: MapKind::new(kind.bits() & !MapKind::EXTENDABLE.bits()),
            value_slot_count: obj_ref.map_ref(heap).value_slot_count(),
            descriptors: &descriptors,
            prototype,
        });
        obj_ref.header.map.set(heap, obj_ref.erase(), new_map);
    });
}

/// Proxy `[[PreventExtensions]]` (ES 20.2.5.3), plus the ordinary path —
/// the `Object.preventExtensions` implementation.
pub fn prevent_extensions(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    obj: Tagged<'_, Value>,
) -> Result<Coercion, VmError> {
    state.handle_scope(|scope| {
        let obj = scope.handle(obj);
        prevent_extensions_h(vm, heap, state, &scope, &obj)
    })
}

fn prevent_extensions_h(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    obj: &Handle<'_, Value>,
) -> Result<Coercion, VmError> {
    if !heap.no_gc(|heap| is_proxy(heap, obj.as_tagged(heap))) {
        let Some(obj) = scope.cast::<Object>(obj.as_tagged(&*heap)) else {
            return Err(VmError::Type);
        };
        ordinary_prevent_extensions(heap, scope, obj);
        return Ok(Coercion::Value(Convert::boolean(heap, true).erase()));
    }
    let (target, handler) = heap
        .no_gc(|heap| parts(heap, obj.as_tagged(heap)))
        .expect("checked proxy above");
    let revoked = heap.no_gc(|heap| handler == heap.known().null.as_tagged(heap).erase());
    if revoked {
        return Err(revoked_error(Trap::PreventExtensions));
    }
    let target = scope.handle(unsafe { target.assume_valid(&*heap) });
    let handler = scope.handle(unsafe { handler.assume_valid(&*heap) });
    match get_trap(vm, heap, state, scope, &handler, Trap::PreventExtensions)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => prevent_extensions_h(vm, heap, state, scope, &target),
        TrapLookup::Trap(t) => {
            let result = call_trap(vm, heap, state, scope, &t, &[handler, target])?;
            let Coercion::Value(result) = result else {
                return Ok(Coercion::Threw);
            };
            if !heap.no_gc(|heap| Convert::is_truthy(heap, unsafe { result.assume_valid(heap) })) {
                return Ok(Coercion::Value(Convert::boolean(heap, false).erase()));
            }
            // invariant: returning true requires a non-extensible target
            let ext = is_extensible_h(vm, heap, state, scope, &target)?;
            let Flow::Value(extensible) = ext else {
                return Ok(Coercion::Threw);
            };
            if extensible {
                return Err(VmError::Message(
                    "proxy preventExtensions trap returned true for an extensible target",
                ));
            }
            Ok(Coercion::Value(Convert::boolean(heap, true).erase()))
        }
    }
}

/// `Object.isExtensible` (ES 20.1.2.14) including the proxy trap and
/// its must-match-target invariant (ES 20.2.5.2 step 8).
pub fn is_extensible(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    obj: Tagged<'_, Value>,
) -> Result<Coercion, VmError> {
    state.handle_scope(|scope| {
        let obj = scope.handle(obj);
        is_extensible_entry_h(vm, heap, state, &scope, &obj)
    })
}

fn is_extensible_entry_h(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    obj: &Handle<'_, Value>,
) -> Result<Coercion, VmError> {
    if !heap.no_gc(|heap| is_proxy(heap, obj.as_tagged(heap))) {
        let extensible = heap.no_gc(|heap| {
            obj.as_tagged(heap)
                .as_heap_object()
                .is_some_and(|o| o.as_ref().map_ref(heap).kind().is_extendable())
        });
        return Ok(Coercion::Value(Convert::boolean(heap, extensible).erase()));
    }
    let (target, handler) = heap
        .no_gc(|heap| parts(heap, obj.as_tagged(heap)))
        .expect("checked proxy above");
    let revoked = heap.no_gc(|heap| handler == heap.known().null.as_tagged(heap).erase());
    if revoked {
        return Err(revoked_error(Trap::IsExtensible));
    }
    let target = scope.handle(unsafe { target.assume_valid(&*heap) });
    let handler = scope.handle(unsafe { handler.assume_valid(&*heap) });
    match get_trap(vm, heap, state, scope, &handler, Trap::IsExtensible)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => is_extensible_entry_h(vm, heap, state, scope, &target),
        TrapLookup::Trap(t) => {
            let result = call_trap(vm, heap, state, scope, &t, &[handler, target])?;
            let Coercion::Value(result) = result else {
                return Ok(Coercion::Threw);
            };
            let trap_bool =
                heap.no_gc(|heap| Convert::is_truthy(heap, unsafe { result.assume_valid(heap) }));
            let target_bool = is_extensible_h(vm, heap, state, scope, &target)?;
            let Flow::Value(target_bool) = target_bool else {
                return Ok(Coercion::Threw);
            };
            if trap_bool != target_bool {
                return Err(VmError::Message(
                    "proxy isExtensible trap must match the target's extensibility",
                ));
            }
            Ok(Coercion::Value(Convert::boolean(heap, trap_bool).erase()))
        }
    }
}
