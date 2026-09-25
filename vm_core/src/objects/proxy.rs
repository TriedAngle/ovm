use core::alloc::Layout;

use crate::runtime::Coercion;
use crate::{
    Compare, ContextState, Convert, EdgeVisitable, GcSlot, Handle, HandleScope, HandleSlice,
    Header, Heap, HeapObject, Key, Lookup, Map, Object, ObjectKind, PartialDescriptor,
    PropertyDescriptor, RuntimeContext, SlotName, Tagged, Transition, VM, Value, Visitor, VmError,
};

pub struct Proxy;

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
        self.handler.raw() == heap.known().null.as_tagged(heap)
    }

    /// (target, handler) anchored to `heap`; caller checks revocation.
    pub fn parts<'a>(&self, heap: &'a Heap) -> (Tagged<'a, Value>, Tagged<'a, Value>) {
        (self.target.get(heap), self.handler.get(heap))
    }
}

impl HeapObject for ProxyObject {
    const KIND: ObjectKind = ObjectKind::Proxy;
    type Init<'a> = ProxyInit<'a>;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Self::layout_for()
    }

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        let host = self.tagged(heap);
        self.header.map.set(heap, host, config.map.as_tagged(heap));
        self.target
            .set(heap, host, config.target.as_tagged(heap).erase());
        self.handler
            .set(heap, host, config.handler.as_tagged(heap).erase());
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
    fn name<'a>(self, heap: &'a Heap) -> Tagged<'a, SlotName> {
        let s = heap.known().strings;
        match self {
            Self::Get => s.get.as_tagged(heap),
            Self::Set => s.set.as_tagged(heap),
            Self::Has => s.has.as_tagged(heap),
            Self::DeleteProperty => s.delete_property.as_tagged(heap),
            Self::GetOwnPropertyDescriptor => s.get_own_property_descriptor.as_tagged(heap),
            Self::DefineProperty => s.define_property.as_tagged(heap),
            Self::GetPrototypeOf => s.get_prototype_of.as_tagged(heap),
            Self::SetPrototypeOf => s.set_prototype_of.as_tagged(heap),
            Self::IsExtensible => s.is_extensible.as_tagged(heap),
            Self::PreventExtensions => s.prevent_extensions.as_tagged(heap),
            Self::OwnKeys => s.own_keys.as_tagged(heap),
            Self::Apply => s.apply.as_tagged(heap),
            Self::Construct => s.construct.as_tagged(heap),
        }
    }
}

/// (target, handler) of a proxy, read under a leaf scope. The raw words
/// must be rooted by the caller before any allocation.
fn parts<'a>(
    heap: &'a Heap,
    proxy: Tagged<'a, Value>,
) -> Option<(Tagged<'a, Value>, Tagged<'a, Value>)> {
    let p = proxy.get_as::<ProxyObject>()?;
    Some(p.as_ref().parts(heap))
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
    let name = scope.handle(trap.name(heap).erase());
    match Lookup::get_property_on(vm, heap, state, *handler, *handler, name)? {
        Coercion::Threw => Ok(TrapLookup::Threw),
        Coercion::Value(v) => {
            let v = scope.handle(v);
            let nullish = v.as_tagged(heap) == heap.known().undefined.as_tagged(heap)
                || v.as_tagged(heap) == heap.known().null.as_tagged(heap);
            if nullish {
                Ok(TrapLookup::None)
            } else {
                Ok(TrapLookup::Trap(v))
            }
        }
    }
}

fn call_trap<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    trap: &Handle<'_, Value>,
    args: &[Handle<'_, Value>],
) -> Result<Coercion<'a>, VmError> {
    if !Object::is_callable(heap, trap.as_tagged(heap)) {
        return Err(VmError::Message("proxy trap is not a function"));
    }
    let words: Vec<Tagged<'_, Value>> = args.iter().map(|h| h.as_tagged(heap)).collect();
    let staged = scope.stage(&words);
    let result = scope.handle(RuntimeContext::call(
        vm, &mut *heap, state, *trap, staged, None,
    )?);
    let exception = heap.known().exception.as_tagged(heap);
    if result.as_tagged(heap) == exception {
        Ok(Coercion::Threw)
    } else {
        Ok(Coercion::Value(result.as_tagged(heap)))
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
    let (target, handler) = parts(heap, proxy.as_tagged(heap)).expect("caller verified a proxy");
    let revoked = handler == heap.known().null.as_tagged(heap);
    if revoked {
        return Err(revoked_error(trap));
    }
    Ok((scope.handle(target), scope.handle(handler)))
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

fn own_descriptor_h<'s>(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &'s HandleScope<'_>,
    obj: &Handle<'_, Value>,
    key: &Handle<'_, Value>,
) -> Result<Flow<Option<PartialDescriptor<'s>>>, VmError> {
    let cond_11 = Proxy::is_proxy(heap, obj.as_tagged(heap));
    if !cond_11 {
        let desc =
            Lookup::ordinary_own_descriptor(heap, scope, obj.as_tagged(heap), key.as_tagged(heap));
        return Ok(Flow::Value(desc.as_ref().map(PartialDescriptor::from)));
    }
    let (target, handler) = parts(heap, obj.as_tagged(heap)).expect("checked proxy above");
    let revoked = handler.ptr_eq(heap.known().null.as_tagged(heap).erase());
    if revoked {
        return Err(revoked_error(Trap::GetOwnPropertyDescriptor));
    }
    let target = scope.handle(target);
    let handler = scope.handle(handler);
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
            let result = match result {
                Coercion::Threw => return Ok(Flow::Threw),
                Coercion::Value(v) => scope.handle(v),
            };
            let is_undefined = result
                .as_tagged(heap)
                .ptr_eq(heap.known().undefined.as_tagged(heap).erase());
            if is_undefined {
                return Ok(Flow::Value(None));
            }
            match Lookup::to_property_descriptor(vm, heap, state, scope, result)? {
                Some(partial) => Ok(Flow::Value(Some(partial))),
                None => Ok(Flow::Threw),
            }
        }
    }
}

fn is_extensible_h(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    obj: &Handle<'_, Value>,
) -> Result<Flow<bool>, VmError> {
    let cond_12 = Proxy::is_proxy(heap, obj.as_tagged(heap));
    if !cond_12 {
        return Ok(Flow::Value({
            obj.as_tagged(heap)
                .as_heap_object()
                .is_some_and(|o| o.as_ref().map_ref(heap).kind().is_extendable())
        }));
    }
    let (target, handler) = parts(heap, obj.as_tagged(heap)).expect("checked proxy above");
    let revoked = handler == heap.known().null.as_tagged(heap);
    if revoked {
        return Err(revoked_error(Trap::IsExtensible));
    }
    let target = scope.handle(target);
    let handler = scope.handle(handler);
    match get_trap(vm, heap, state, scope, &handler, Trap::IsExtensible)? {
        TrapLookup::Threw => Ok(Flow::Threw),
        TrapLookup::None => is_extensible_h(vm, heap, state, scope, &target),
        TrapLookup::Trap(t) => {
            let result = call_trap(vm, heap, state, scope, &t, &[handler, target])?;
            let result = match result {
                Coercion::Threw => return Ok(Flow::Threw),
                Coercion::Value(v) => scope.handle(v),
            };
            Ok(Flow::Value({
                Convert::is_truthy(heap, result.as_tagged(heap))
            }))
        }
    }
}

// ---- forwards (trap absent → run the operation on the target) ---------------

/// CreateDataProperty-style full descriptor ({w,e,c} true) — the payload
/// of OrdinarySet's receiver-define step.
fn create_data_partial<'s>(
    _scope: &'s HandleScope<'_>,
    value: &Handle<'s, Value>,
) -> PartialDescriptor<'s> {
    PartialDescriptor {
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
/// Allocates, so the payload is handle-rooted (`PartialDescriptor`).
fn descriptor_object<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    partial: &PartialDescriptor<'_>,
) -> Result<Handle<'s, Value>, VmError> {
    {
        let obj = heap
            .new_object(scope, heap.known().object_initial_map, HandleSlice::EMPTY)
            .as_handle(scope);
        // the well-known field names are persistent roots: they cross the
        // define allocations without further rooting
        let s = heap.known().strings;
        // spec field order: value, writable, get, set, enumerable, configurable
        let mut fields: Vec<(Handle<'_, SlotName>, Handle<'_, Value>)> = Vec::new();
        if let Some(v) = partial.value {
            fields.push((s.value, v));
        }
        if let Some(b) = partial.writable {
            fields.push((s.writable, scope.handle(Convert::boolean(heap, b))));
        }
        if let Some(v) = partial.get {
            fields.push((s.get, v));
        }
        if let Some(v) = partial.set {
            fields.push((s.set, v));
        }
        if let Some(b) = partial.enumerable {
            fields.push((s.enumerable, scope.handle(Convert::boolean(heap, b))));
        }
        if let Some(b) = partial.configurable {
            fields.push((s.configurable, scope.handle(Convert::boolean(heap, b))));
        }
        for (name, value) in fields {
            Object::define_own_property(heap, scope, obj, name, PropertyDescriptor::data(value))?;
        }
        Ok(obj.erase())
    }
}

fn define_internal_h<'s>(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &'s HandleScope<'_>,
    obj: &Handle<'_, Value>,
    name: &Handle<'_, Value>,
    partial: PartialDescriptor<'s>,
) -> Result<Flow<bool>, VmError> {
    let cond_13 = Proxy::is_proxy(heap, obj.as_tagged(heap));
    if cond_13 {
        return proxy_define_h(vm, heap, state, scope, obj, name, partial);
    }
    let undefined = scope.handle(heap.known().undefined.as_tagged(heap).erase());
    // dense array elements: element defines land in the backing
    // store, not as descriptors
    let element = 'element: {
        let on_array = obj
            .as_tagged(heap)
            .as_heap_object()
            .is_some_and(|o| o.as_ref().is_array(heap));
        if !on_array {
            break 'element None;
        }
        match Lookup::classify_key(heap, name.as_tagged(heap)) {
            Ok(Key::Element(i)) => Some(i),
            _ => None,
        }
    };
    if let Some(i) = element {
        let completed = partial.complete_against(undefined, None);
        let PropertyDescriptor::Data { value, .. } = completed else {
            return Err(VmError::Type);
        };
        let array = scope
            .cast::<Object>(obj.as_tagged(heap))
            .expect("array checked above");
        Object::store_array_element(heap, scope, &array, i, &value)?;
        return Ok(Flow::Value(true));
    }
    let current =
        Lookup::ordinary_own_descriptor(heap, scope, obj.as_tagged(heap), name.as_tagged(heap));
    let full = partial.complete_against(undefined, current.as_ref());
    let obj_ref = scope
        .cast::<Object>(obj.as_tagged(heap))
        .expect("non-proxy target is an object");
    let name_ref: Handle<'_, SlotName> = scope.handle(name.as_tagged(heap).as_name());
    let defined = Object::define_own_property(heap, scope, obj_ref, name_ref, full)?;
    Ok(Flow::Value(defined))
}

/// OrdinarySet (receiver-aware) for an ordinary target with a proxy
/// (or arbitrary) receiver: the spec's define-on-receiver shape.
fn ordinary_set_forward<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    target: &Handle<'_, Value>,
    receiver: &Handle<'_, Value>,
    name: &Handle<'_, Value>,
    value: &Handle<'_, Value>,
) -> Result<Coercion<'a>, VmError> {
    // array element keys on arrays route through the define path
    // (which stores into the backing store)
    let classified = (
        matches!(
            Lookup::classify_key(heap, name.as_tagged(heap)),
            Ok(Key::Element(_))
        ),
        target
            .as_tagged(heap)
            .as_heap_object()
            .is_some_and(|o| o.as_ref().is_array(heap)),
    );
    if let (true, true) = classified {
        let partial = create_data_partial(scope, value);
        let flow = define_internal_h(vm, heap, state, scope, receiver, name, partial)?;
        return define_flow_to_coercion(heap, flow);
    }

    // owned summary of the chain lookup starting at the target
    // (Lookup borrows its non-allocating region, so it is consumed inside)
    enum SetLookup<'s> {
        DataWritable,
        DataReadonly,
        Setter(Handle<'s, Value>),
        NotFound,
    }
    let lookup = match target.lookup(heap, name.as_tagged(heap).as_name()) {
        Lookup::Data { flags, .. } => {
            if flags.is_writable() {
                SetLookup::DataWritable
            } else {
                SetLookup::DataReadonly
            }
        }
        Lookup::Accessor { pair, .. } => {
            SetLookup::Setter(scope.handle(pair.as_ref().set.get(heap)))
        }
        Lookup::NotFound => SetLookup::NotFound,
    };
    match lookup {
        SetLookup::DataReadonly => Ok(Coercion::Value(Convert::boolean(heap, false))),
        SetLookup::DataWritable | SetLookup::NotFound => {
            let partial = create_data_partial(scope, value);
            let flow = define_internal_h(vm, heap, state, scope, receiver, name, partial)?;
            define_flow_to_coercion(heap, flow)
        }
        SetLookup::Setter(setter) => {
            if setter.as_tagged(heap) == heap.known().undefined.as_tagged(heap) {
                return Ok(Coercion::Value(Convert::boolean(heap, false)));
            }
            let words = [
                receiver.as_tagged(heap).erase(),
                value.as_tagged(heap).erase(),
            ];
            let staged = scope.stage(&words);
            let result = scope.handle(RuntimeContext::call(
                vm, &mut *heap, state, setter, staged, None,
            )?);
            let exception = heap.known().exception.as_tagged(heap);
            if result.as_tagged(heap) == exception {
                Ok(Coercion::Threw)
            } else {
                Ok(Coercion::Value(Convert::boolean(heap, true)))
            }
        }
    }
}

fn define_flow_to_coercion<'a>(heap: &'a Heap, flow: Flow<bool>) -> Result<Coercion<'a>, VmError> {
    Ok(match flow {
        Flow::Threw => Coercion::Threw,
        Flow::Value(b) => Coercion::Value(Convert::boolean(heap, b)),
    })
}

// ---- the internal methods (ES 20.2.5.x) ------------------------------------

fn get_h<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    proxy: &Handle<'_, Value>,
    receiver: &Handle<'_, Value>,
    name: &Handle<'_, Value>,
) -> Result<Coercion<'a>, VmError> {
    let (target, handler) = enter_trap(scope, heap, proxy, Trap::Get)?;
    match get_trap(vm, heap, state, scope, &handler, Trap::Get)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => {
            // forward: lookup on the target, getter `this` = the
            // original receiver (a proxy target re-enters its own
            // `get` trap with the same receiver)
            let forwarded = Lookup::get_property_on(vm, heap, state, target, *receiver, *name)?;
            match forwarded {
                Coercion::Threw => Ok(Coercion::Threw),
                Coercion::Value(v) => {
                    let v = scope.handle(v);
                    Ok(Coercion::Value(v.as_tagged(heap)))
                }
            }
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
            let result = match result {
                Coercion::Threw => return Ok(Coercion::Threw),
                Coercion::Value(v) => scope.handle(v),
            };
            // invariant (steps 9-11): a trap cannot lie about
            // non-configurable data / accessor properties
            let desc = own_descriptor_h(vm, heap, state, scope, &target, name)?;
            let Flow::Value(desc) = desc else {
                return Ok(Coercion::Threw);
            };
            if let Some(d) = desc.as_ref().filter(|d| d.configurable == Some(false)) {
                {
                    let undefined = heap.known().undefined.as_tagged(heap).erase();
                    if d.is_data_descriptor()
                        && d.writable == Some(false)
                        && !Compare::same_value(
                            heap,
                            result.as_tagged(heap),
                            d.value.map_or(undefined, |h| h.as_tagged(heap)),
                        )
                    {
                        return Err(VmError::Message(
                            "proxy get trap must match a non-writable, non-configurable property",
                        ));
                    }
                    if d.is_accessor_descriptor()
                        && d.get.is_some_and(|g| g.as_tagged(heap).ptr_eq(undefined))
                        && !result.as_tagged(heap).ptr_eq(undefined)
                    {
                        return Err(VmError::Message(
                            "proxy get trap must return undefined for an accessor without a getter",
                        ));
                    }
                    Ok(())
                }?;
            }
            Ok(Coercion::Value(result.as_tagged(heap)))
        }
    }
}

fn set_h<'a, 's>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    scope: &'s HandleScope<'_>,
    proxy: &Handle<'_, Value>,
    name: &Handle<'_, Value>,
    value: &Handle<'_, Value>,
    receiver: &Handle<'_, Value>,
) -> Result<Coercion<'a>, VmError> {
    let (target, handler) = enter_trap(scope, heap, proxy, Trap::Set)?;
    match get_trap(vm, heap, state, scope, &handler, Trap::Set)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => {
            let cond_14 = Proxy::is_proxy(heap, target.as_tagged(heap));
            if cond_14 {
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
            let result = match result {
                Coercion::Threw => return Ok(Coercion::Threw),
                Coercion::Value(v) => scope.handle(v),
            };
            let truthy = Convert::is_truthy(heap, result.as_tagged(heap));
            if !truthy {
                // [[Set]] returned false; sloppy stores ignore it
                return Ok(Coercion::Value(Convert::boolean(heap, false)));
            }
            // invariant (steps 11-13)
            let desc = own_descriptor_h(vm, heap, state, scope, &target, name)?;
            let Flow::Value(desc) = desc else {
                return Ok(Coercion::Threw);
            };
            if let Some(d) = desc.as_ref().filter(|d| d.configurable == Some(false)) {
                {
                    let undefined = heap.known().undefined.as_tagged(heap).erase();
                    if d.is_data_descriptor()
                        && d.writable == Some(false)
                        && !Compare::same_value(
                            heap,
                            value.as_tagged(heap),
                            d.value.map_or(undefined, |h| h.as_tagged(heap)),
                        )
                    {
                        return Err(VmError::Message(
                            "proxy set trap must match a non-writable, non-configurable property",
                        ));
                    }
                    if d.is_accessor_descriptor()
                        && d.set.is_some_and(|s| s.as_tagged(heap).ptr_eq(undefined))
                    {
                        return Err(VmError::Message(
                            "proxy set trap may not report success for an accessor without a setter",
                        ));
                    }
                    Ok(())
                }?;
            }
            Ok(Coercion::Value(Convert::boolean(heap, true)))
        }
    }
}

fn has_h<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    proxy: &Handle<'_, Value>,
    name: &Handle<'_, Value>,
) -> Result<Coercion<'a>, VmError> {
    let (target, handler) = enter_trap(scope, heap, proxy, Trap::Has)?;
    match get_trap(vm, heap, state, scope, &handler, Trap::Has)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => {
            let cond_15 = Proxy::is_proxy(heap, target.as_tagged(heap));
            if cond_15 {
                has_h(vm, heap, state, scope, &target, name)
            } else {
                let found = Lookup::has_property(
                    heap,
                    target.as_tagged(heap),
                    name.as_tagged(heap).as_name(),
                );
                Ok(Coercion::Value(Convert::boolean(heap, found)))
            }
        }
        TrapLookup::Trap(t) => {
            let result = call_trap(vm, heap, state, scope, &t, &[handler, target, *name])?;
            let result = match result {
                Coercion::Threw => return Ok(Coercion::Threw),
                Coercion::Value(v) => scope.handle(v),
            };
            let truthy = Convert::is_truthy(heap, result.as_tagged(heap));
            if truthy {
                return Ok(Coercion::Value(Convert::boolean(heap, true)));
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
            Ok(Coercion::Value(Convert::boolean(heap, false)))
        }
    }
}

fn delete_h<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    proxy: &Handle<'_, Value>,
    key: &Handle<'_, Value>,
) -> Result<Coercion<'a>, VmError> {
    let (target, handler) = enter_trap(scope, heap, proxy, Trap::DeleteProperty)?;
    match get_trap(vm, heap, state, scope, &handler, Trap::DeleteProperty)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => {
            let cond_16 = Proxy::is_proxy(heap, target.as_tagged(heap));
            if cond_16 {
                delete_h(vm, heap, state, scope, &target, key)
            } else {
                let target_obj = scope
                    .cast::<Object>(target.as_tagged(heap))
                    .expect("ordinary target");
                let deleted = Object::delete_own_property(heap, scope, target_obj, *key)?;
                Ok(Coercion::Value(Convert::boolean(heap, deleted)))
            }
        }
        TrapLookup::Trap(t) => {
            let result = call_trap(vm, heap, state, scope, &t, &[handler, target, *key])?;
            let result = match result {
                Coercion::Threw => return Ok(Coercion::Threw),
                Coercion::Value(v) => scope.handle(v),
            };
            let truthy = Convert::is_truthy(heap, result.as_tagged(heap));
            if !truthy {
                return Ok(Coercion::Value(Convert::boolean(heap, false)));
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
            Ok(Coercion::Value(Convert::boolean(heap, true)))
        }
    }
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

    match get_trap(vm, heap, state, scope, &handler, Trap::DefineProperty)? {
        TrapLookup::Threw => Ok(Flow::Threw),
        TrapLookup::None => define_internal_h(vm, heap, state, scope, &target, name, partial),
        TrapLookup::Trap(t) => {
            let desc_obj = descriptor_object(heap, scope, &partial)?;
            let result = call_trap(
                vm,
                heap,
                state,
                scope,
                &t,
                &[handler, target, *name, desc_obj],
            )?;
            let result = match result {
                Coercion::Threw => return Ok(Flow::Threw),
                Coercion::Value(v) => scope.handle(v),
            };
            let cond_17 = Convert::is_truthy(heap, result.as_tagged(heap));
            if !cond_17 {
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
                    let compatible = {
                        let p = partial;
                        Transition::is_compatible_property_descriptor(heap, extensible, &p, Some(d))
                    };
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

fn apply_h<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    proxy: &Handle<'_, Value>,
    args: &[Handle<'_, Value>],
) -> Result<Coercion<'a>, VmError> {
    let (target, handler) = enter_trap(scope, heap, proxy, Trap::Apply)?;
    let this_arg = match args.first() {
        Some(h) => *h,
        None => scope.handle(heap.known().undefined.as_tagged(heap).erase()),
    };
    match get_trap(vm, heap, state, scope, &handler, Trap::Apply)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => {
            let mut all: Vec<Tagged<'_, Value>> = Vec::with_capacity(args.len());
            all.push(this_arg.as_tagged(heap).erase());
            for h in &args[1..] {
                all.push(h.as_tagged(heap).erase());
            }
            let staged = scope.stage(&all);
            let result = scope.handle(RuntimeContext::call(
                vm, &mut *heap, state, target, staged, None,
            )?);
            let exception = heap.known().exception.as_tagged(heap);
            if result.as_tagged(heap) == exception {
                Ok(Coercion::Threw)
            } else {
                Ok(Coercion::Value(result.as_tagged(heap)))
            }
        }
        TrapLookup::Trap(t) => {
            // args: (target, thisArg, argumentsList)
            let words: Vec<Tagged<'_, Value>> =
                args[1..].iter().map(|h| h.as_tagged(heap)).collect();
            let arr = heap
                .new_array(scope, scope.stage(&words))
                .as_handle(scope)
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

fn construct_h<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    proxy: &Handle<'_, Value>,
    args: &[Handle<'_, Value>],
    new_target: &Handle<'_, Value>,
) -> Result<Coercion<'a>, VmError> {
    let (target, handler) = enter_trap(scope, heap, proxy, Trap::Construct)?;
    match get_trap(vm, heap, state, scope, &handler, Trap::Construct)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => {
            // forward: the full target [[Construct]] — synthesize the
            // receiver from new.target (the hole for derived class
            // constructors), run, and prefer an object result
            let derived = target
                .as_tagged(heap)
                .as_heap_object()
                .and_then(|obj| {
                    obj.as_ref()
                        .header
                        .map
                        .get(heap)
                        .kind()
                        .is_class_constructor()
                        .then(|| obj.as_ref().callable_info(heap))
                })
                .flatten()
                .is_some_and(|info| info.function_kind().is_derived_class_constructor());
            let receiver = if derived {
                scope.handle(heap.known().the_hole.as_tagged(heap).erase())
            } else {
                let Some(receiver) =
                    Object::create_construct_receiver_value(vm, heap, state, *new_target)?
                else {
                    return Ok(Coercion::Threw);
                };
                scope.handle(receiver)
            };
            let mut all: Vec<Tagged<'_, Value>> = Vec::with_capacity(args.len() + 1);
            all.push(receiver.as_tagged(heap).erase());
            for h in args {
                all.push(h.as_tagged(heap).erase());
            }
            let staged = scope.stage(&all);
            let result = scope.handle(RuntimeContext::call(
                vm,
                heap,
                state,
                target,
                staged,
                Some(*new_target),
            )?);
            let exception = heap.known().exception.as_tagged(heap);
            if result.as_tagged(heap) == exception {
                return Ok(Coercion::Threw);
            }
            let cond_18 = Convert::is_primitive(heap, result.as_tagged(heap));
            if cond_18 {
                if derived {
                    // a derived constructor may only return objects
                    return Err(VmError::Type);
                }
                return Ok(Coercion::Value(receiver.as_tagged(heap)));
            }
            Ok(Coercion::Value(result.as_tagged(heap)))
        }
        TrapLookup::Trap(t) => {
            // args: (target, argumentsList, newTarget)
            let words: Vec<Tagged<'_, Value>> = args.iter().map(|h| h.as_tagged(heap)).collect();
            let arr = heap
                .new_array(scope, scope.stage(&words))
                .as_handle(scope)
                .erase();
            let result = call_trap(
                vm,
                heap,
                state,
                scope,
                &t,
                &[handler, target, arr, *new_target],
            )?;
            let result = match result {
                Coercion::Threw => return Ok(Coercion::Threw),
                Coercion::Value(v) => scope.handle(v),
            };
            // invariant: the trap must return an object
            {
                let cond_19 = Convert::is_primitive(heap, result.as_tagged(heap));
                if cond_19 {
                    return Err(VmError::Message(
                        "proxy construct trap must return an object",
                    ));
                }
            }
            Ok(Coercion::Value(result.as_tagged(heap)))
        }
    }
}

/// Ordinary `[[PreventExtensions]]`: clone the map without the
/// EXTENDABLE bit (maps are shared, so the clone isolates the object).
fn ordinary_prevent_extensions(heap: &mut Heap, scope: &HandleScope<'_>, obj: Handle<'_, Object>) {
    use crate::{MapInit, MapKind};
    let (kind, prototype, rows) = {
        let map = obj.as_tagged(heap).map_ref(heap);
        (
            map.kind(),
            scope.handle(map.prototype.get(heap)),
            // names are rooted here and re-anchored in the allocating
            // closure: a Tagged cannot escape this non-allocating region
            map.descriptors()
                .iter()
                .map(|d| {
                    (
                        scope.handle(d.name(heap)),
                        d.flags(),
                        scope.handle(d.value.get(heap)),
                    )
                })
                .collect::<Vec<_>>(),
        )
    };
    if !kind.is_extendable() {
        return; // already non-extensible (idempotent)
    }
    heap.allocate_token_enter_heap(Map::layout_for(rows.len()), |token, heap| {
        let obj_ref = obj.as_tagged(heap);
        let new_map = token.allocate::<Map>(MapInit {
            kind: MapKind::new(kind.bits() & !MapKind::EXTENDABLE.bits()),
            value_slot_count: obj_ref.map_ref(heap).value_slot_count(),
            descriptors: &rows,
            prototype,
        });
        obj_ref.header.map.set(heap, obj_ref.erase(), new_map);
    });
}

fn prevent_extensions_h<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    obj: &Handle<'_, Value>,
) -> Result<Coercion<'a>, VmError> {
    let cond_20 = Proxy::is_proxy(heap, obj.as_tagged(heap));
    if !cond_20 {
        let Some(obj) = scope.cast::<Object>(obj.as_tagged(heap)) else {
            return Err(VmError::Type);
        };
        ordinary_prevent_extensions(heap, scope, obj);
        return Ok(Coercion::Value(Convert::boolean(heap, true)));
    }
    let (target, handler) = parts(heap, obj.as_tagged(heap)).expect("checked proxy above");
    let revoked = handler == heap.known().null.as_tagged(heap);
    if revoked {
        return Err(revoked_error(Trap::PreventExtensions));
    }
    let target = scope.handle(target);
    let handler = scope.handle(handler);
    match get_trap(vm, heap, state, scope, &handler, Trap::PreventExtensions)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => prevent_extensions_h(vm, heap, state, scope, &target),
        TrapLookup::Trap(t) => {
            let result = call_trap(vm, heap, state, scope, &t, &[handler, target])?;
            let result = match result {
                Coercion::Threw => return Ok(Coercion::Threw),
                Coercion::Value(v) => scope.handle(v),
            };
            let cond_21 = Convert::is_truthy(heap, result.as_tagged(heap));
            if !cond_21 {
                return Ok(Coercion::Value(Convert::boolean(heap, false)));
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
            Ok(Coercion::Value(Convert::boolean(heap, true)))
        }
    }
}

fn is_extensible_entry_h<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    obj: &Handle<'_, Value>,
) -> Result<Coercion<'a>, VmError> {
    let cond_22 = Proxy::is_proxy(heap, obj.as_tagged(heap));
    if !cond_22 {
        let extensible = obj
            .as_tagged(heap)
            .as_heap_object()
            .is_some_and(|o| o.as_ref().map_ref(heap).kind().is_extendable());
        return Ok(Coercion::Value(Convert::boolean(heap, extensible)));
    }
    let (target, handler) = parts(heap, obj.as_tagged(heap)).expect("checked proxy above");
    let revoked = handler == heap.known().null.as_tagged(heap);
    if revoked {
        return Err(revoked_error(Trap::IsExtensible));
    }
    let target = scope.handle(target);
    let handler = scope.handle(handler);
    match get_trap(vm, heap, state, scope, &handler, Trap::IsExtensible)? {
        TrapLookup::Threw => Ok(Coercion::Threw),
        TrapLookup::None => is_extensible_entry_h(vm, heap, state, scope, &target),
        TrapLookup::Trap(t) => {
            let result = call_trap(vm, heap, state, scope, &t, &[handler, target])?;
            let result = match result {
                Coercion::Threw => return Ok(Coercion::Threw),
                Coercion::Value(v) => scope.handle(v),
            };
            let trap_bool = Convert::is_truthy(heap, result.as_tagged(heap));
            let target_bool = is_extensible_h(vm, heap, state, scope, &target)?;
            let Flow::Value(target_bool) = target_bool else {
                return Ok(Coercion::Threw);
            };
            if trap_bool != target_bool {
                return Err(VmError::Message(
                    "proxy isExtensible trap must match the target's extensibility",
                ));
            }
            Ok(Coercion::Value(Convert::boolean(heap, trap_bool)))
        }
    }
}

impl Proxy {
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
            .get(heap)
            .kind()
            .kind()
            .is_js_receiver()
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
        let map: Handle<'_, Map> = {
            let kind = target
                .as_tagged(heap)
                .as_heap_object()
                .expect("proxy target must be a JSReceiver")
                .as_ref()
                .header
                .map
                .get(heap)
                .kind();
            if kind.is_constructor() {
                known.proxy_constructor_map
            } else if kind.is_callable() {
                known.proxy_callable_map
            } else {
                known.proxy_map
            }
        };
        heap.allocate::<ProxyObject>(ProxyInit {
            map,
            target,
            handler,
        })
        .erase()
    }

    pub fn revoke(heap: &mut Heap, proxy: Tagged<'_, Value>) {
        let Some(p) = proxy.get_as::<ProxyObject>() else {
            return;
        };
        if p.as_ref().is_revoked(heap) {
            return;
        }
        let host = proxy.erase();
        let null = heap.known().null.as_tagged(heap).erase();
        p.as_ref().target.set(heap, host, null);
        p.as_ref().handler.set(heap, host, null);
    }

    pub fn internal_own_descriptor<'s>(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        scope: &'s HandleScope<'_>,
        obj: Tagged<'_, Value>,
        key: Tagged<'_, Value>,
    ) -> Result<Flow<Option<PartialDescriptor<'s>>>, VmError> {
        let obj = scope.handle(obj);
        let key = scope.handle(key);
        own_descriptor_h(vm, heap, state, scope, &obj, &key)
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

    /// `[[DefineOwnProperty]]` dispatch: proxies run their `defineProperty`
    /// trap (with full spec validation), ordinary objects complete the
    /// partial against their current descriptor and define through the
    /// transition machinery.
    pub fn define_internal<'s>(
        vm: &VM,
        heap: &mut Heap,
        state: &ContextState,
        scope: &'s HandleScope<'_>,
        obj: Handle<'_, Value>,
        name: Handle<'_, Value>,
        partial: PartialDescriptor<'s>,
    ) -> Result<Flow<bool>, VmError> {
        define_internal_h(vm, heap, state, scope, &obj, &name, partial)
    }

    /// Proxy `[[Get]]` (ES 20.2.5.8). `receiver` is the [[Get]] receiver —
    /// the proxy itself at entry points, the *original* receiver when
    /// forwarding through a chain of proxies.
    pub fn get<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        proxy: Handle<'_, Value>,
        receiver: Handle<'_, Value>,
        name: Handle<'_, Value>,
    ) -> Result<Coercion<'a>, VmError> {
        state.handle_scope(|scope| get_h(vm, heap, state, &scope, &proxy, &receiver, &name))
    }

    /// Proxy `[[Set]]` (ES 20.2.5.10).
    pub fn set<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        proxy: Handle<'_, Value>,
        name: Handle<'_, Value>,
        value: Handle<'_, Value>,
        receiver: Handle<'_, Value>,
    ) -> Result<Coercion<'a>, VmError> {
        state.handle_scope(|scope| set_h(vm, heap, state, &scope, &proxy, &name, &value, &receiver))
    }

    /// Proxy `[[HasProperty]]` (ES 20.2.5.9).
    pub fn has<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        proxy: Handle<'_, Value>,
        name: Handle<'_, Value>,
    ) -> Result<Coercion<'a>, VmError> {
        state.handle_scope(|scope| has_h(vm, heap, state, &scope, &proxy, &name))
    }

    /// Proxy `[[Delete]]` (ES 20.2.5.4). The strict-mode false→TypeError
    /// translation is the caller's (`delete` sloppy/strict runtimes).
    pub fn delete<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        proxy: Handle<'_, Value>,
        key: Handle<'_, Value>,
    ) -> Result<Coercion<'a>, VmError> {
        state.handle_scope(|scope| delete_h(vm, heap, state, &scope, &proxy, &key))
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

    /// Proxy `[[Call]]` (ES 20.2.5.15): the `apply` trap, or a plain call
    /// of the target. `args` includes the receiver (`this`) at index 0.
    pub fn apply<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        proxy: Handle<'_, Value>,
        args: HandleSlice<'_>,
    ) -> Result<Coercion<'a>, VmError> {
        state.handle_scope(|scope| {
            let args: Vec<Handle<'_, Value>> = args.iter().collect();
            apply_h(vm, heap, state, &scope, &proxy, &args)
        })
    }

    /// Proxy `[[Construct]]` (ES 20.2.5.5): the `construct` trap, or a
    /// construct of the target with the original `new.target`. `args` are
    /// the constructor arguments (no synthesized receiver).
    pub fn construct<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        proxy: Handle<'_, Value>,
        args: HandleSlice<'_>,
        new_target: Handle<'_, Value>,
    ) -> Result<Coercion<'a>, VmError> {
        state.handle_scope(|scope| {
            let args: Vec<Handle<'_, Value>> = args.iter().collect();
            construct_h(vm, heap, state, &scope, &proxy, &args, &new_target)
        })
    }

    /// Proxy `[[PreventExtensions]]` (ES 20.2.5.3), plus the ordinary path —
    /// the `Object.preventExtensions` implementation.
    pub fn prevent_extensions<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        obj: Tagged<'_, Value>,
    ) -> Result<Coercion<'a>, VmError> {
        state.handle_scope(|scope| {
            let obj = scope.handle(obj);
            prevent_extensions_h(vm, heap, state, &scope, &obj)
        })
    }

    /// `Object.isExtensible` (ES 20.1.2.14) including the proxy trap and
    /// its must-match-target invariant (ES 20.2.5.2 step 8).
    pub fn is_extensible<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        obj: Tagged<'_, Value>,
    ) -> Result<Coercion<'a>, VmError> {
        state.handle_scope(|scope| {
            let obj = scope.handle(obj);
            is_extensible_entry_h(vm, heap, state, &scope, &obj)
        })
    }
}
