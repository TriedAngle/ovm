use core::alloc::Layout;

use crate::{
    Compare, ContextState, Convert, EdgeVisitable, GcSlot, GcSlice, Handle, HandleScope, Header, Heap,
    HeapObject, Key, Lookup, Map, NativeContext, NoGc, Object, ObjectKind, PartialDescriptor,
    PropertyDescriptor, SlotName, VM, Value, Visitor, VmError, is_compatible_property_descriptor,
    lookup::has_property,
    runtime::{Coercion, Runtime},
    store_array_element,
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
        self.target.set(nogc, host, config.target.value());
        self.handler.set(nogc, host, config.handler.value());
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
    fn name(self, heap: &Heap) -> Value {
        let s = heap.known().strings;
        match self {
            Self::Get => s.get.value(),
            Self::Set => s.set.value(),
            Self::Has => s.has.value(),
            Self::DeleteProperty => s.delete_property.value(),
            Self::GetOwnPropertyDescriptor => s.get_own_property_descriptor.value(),
            Self::DefineProperty => s.define_property.value(),
            Self::GetPrototypeOf => s.get_prototype_of.value(),
            Self::SetPrototypeOf => s.set_prototype_of.value(),
            Self::IsExtensible => s.is_extensible.value(),
            Self::PreventExtensions => s.prevent_extensions.value(),
            Self::OwnKeys => s.own_keys.value(),
            Self::Apply => s.apply.value(),
            Self::Construct => s.construct.value(),
        }
    }
}

/// Fast proxy check: one map-kind read.
pub fn is_proxy<'a>(nogc: &'a NoGc<'a>, v: Value) -> bool {
    v.get_as::<ProxyObject>(nogc).is_some()
}

/// Whether `v` is a valid ECMAScript receiver ([[ProxyTarget]] /
/// [[ProxyHandler]] validation, `Reflect.*` / `Object.*` argument
/// checks): a pure map-kind check, like V8's instance-type range test —
/// the oddball singletons (`null`, `undefined`, `true`, `false`, …)
/// carry `ODDBALL`-kind maps and fall out naturally.
pub fn is_js_receiver<'a>(nogc: &'a NoGc<'a>, v: Value) -> bool {
    let Some(obj) = v.as_heap_object(nogc) else {
        return false;
    };
    obj.as_ref()
        .header
        .map
        .heap_ref(nogc)
        .kind()
        .kind()
        .is_js_receiver()
}

/// (target, handler) of a proxy, read under a leaf scope.
fn parts<'a>(nogc: &'a NoGc<'a>, proxy: Value) -> Option<(Value, Value)> {
    let p = proxy.get_as::<ProxyObject>(nogc)?;
    Some(p.as_ref().parts())
}

pub fn allocate(heap: &mut Heap, scope: &HandleScope<'_>, target: Value, handler: Value) -> Value {
    let target = scope.handle(target);
    let handler = scope.handle(handler);
    let known = heap.known();
    let map: Handle<'_, Map> = heap.no_gc(|nogc| {
        let kind = target
            .value()
            .as_heap_object(nogc)
            .expect("proxy target must be a JSReceiver")
            .as_ref()
            .header
            .map
            .heap_ref(nogc)
            .kind();
        if kind.is_constructor() {
            known.proxy_constructor_map
        } else if kind.is_callable() {
            known.proxy_callable_map
        } else {
            known.proxy_map
        }
    });
    heap.allocate::<ProxyObject>(crate::ProxyInit {
        map,
        target,
        handler,
    })
    .erase()
}

pub fn revoke(heap: &mut Heap, proxy: Value) {
    heap.no_gc(|nogc| {
        let Some(p) = proxy.get_as::<ProxyObject>(nogc) else {
            return;
        };
        if p.as_ref().is_revoked(nogc) {
            return;
        }
        let null = nogc.known().null.value();
        p.as_ref().target.set(nogc, proxy, null);
        p.as_ref().handler.set(nogc, proxy, null);
    });
}

enum TrapLookup {
    /// undefined/null: forward to the target.
    None,
    /// A callable... verified by `call_trap`.
    Trap(Value),
    Threw,
}

fn get_trap(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    handler: Value,
    trap: Trap,
) -> Result<TrapLookup, VmError> {
    let name = trap.name(heap);
    match Runtime::get_property(vm, heap, state, handler, name)? {
        Coercion::Threw => Ok(TrapLookup::Threw),
        Coercion::Value(v) => {
            let nullish = heap.no_gc(|nogc| {
                v == nogc.known().undefined.value() || v == nogc.known().null.value()
            });
            if nullish {
                Ok(TrapLookup::None)
            } else {
                Ok(TrapLookup::Trap(v))
            }
        }
    }
}

fn call_trap(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    scope: &HandleScope<'_>,
    trap: Value,
    args: &[Value],
) -> Result<Coercion, VmError> {
    if !Runtime::is_callable(heap, trap) {
        return Err(VmError::Message("proxy trap is not a function"));
    }
    let trap = scope.handle(trap);
    let result = NativeContext::new(vm, heap, state).call(trap.value(), scope.stage(args))?;
    if result == heap.known().exception.value() {
        Ok(Coercion::Threw)
    } else {
        Ok(Coercion::Value(result))
    }
}

/// Revocation check + (target, handler) extraction — the shared entry
/// of every internal method. The caller roots both values before any
/// trap can run.
fn enter_trap(heap: &mut Heap, proxy: Value, trap: Trap) -> Result<(Value, Value), VmError> {
    let (target, handler) = heap
        .no_gc(|nogc| parts(nogc, proxy))
        .expect("caller verified a proxy");
    if handler == heap.known().null.value() {
        return Err(revoked_error(trap));
    }
    Ok((target, handler))
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
    obj: Value,
    key: Value,
) -> Result<Flow<Option<PartialDescriptor>>, VmError> {
    if !heap.no_gc(|nogc| is_proxy(nogc, obj)) {
        let desc = heap.no_gc(|nogc| crate::lookup::ordinary_own_descriptor(nogc, obj, key));
        return Ok(Flow::Value(desc.as_ref().map(PartialDescriptor::from)));
    }
    let (target, handler) = heap
        .no_gc(|nogc| parts(nogc, obj))
        .expect("checked proxy above");
    if handler == heap.known().null.value() {
        return Err(revoked_error(Trap::GetOwnPropertyDescriptor));
    }
    state.handle_scope(|scope| {
        let target = scope.handle(target);
        let handler = scope.handle(handler);
        let key = scope.handle(key);
        match get_trap(
            vm,
            heap,
            state,
            handler.value(),
            Trap::GetOwnPropertyDescriptor,
        )? {
            TrapLookup::Threw => Ok(Flow::Threw),
            TrapLookup::None => {
                internal_own_descriptor(vm, heap, state, target.value(), key.value())
            }
            TrapLookup::Trap(t) => {
                let result = call_trap(
                    vm,
                    heap,
                    state,
                    &scope,
                    t,
                    &[handler.value(), target.value(), key.value()],
                )?;
                let Coercion::Value(result) = result else {
                    return Ok(Flow::Threw);
                };
                if result == heap.known().undefined.value() {
                    return Ok(Flow::Value(None));
                }
                match Runtime::to_property_descriptor(vm, heap, state, result)? {
                    Some(partial) => Ok(Flow::Value(Some(partial))),
                    None => Ok(Flow::Threw),
                }
            }
        }
    })
}

/// [[IsExtensible]] through proxies, *without* the trap-must-match
/// invariant (that check belongs to the `Object.isExtensible` entry
/// point; other internal methods only read the value).
pub fn internal_is_extensible(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    obj: Value,
) -> Result<Flow<bool>, VmError> {
    if !heap.no_gc(|nogc| is_proxy(nogc, obj)) {
        return Ok(Flow::Value(heap.no_gc(|nogc| {
            obj.as_heap_object(nogc)
                .is_some_and(|o| o.as_ref().map_ref(nogc).kind().is_extendable())
        })));
    }
    let (target, handler) = heap
        .no_gc(|nogc| parts(nogc, obj))
        .expect("checked proxy above");
    if handler == heap.known().null.value() {
        return Err(revoked_error(Trap::IsExtensible));
    }
    state.handle_scope(|scope| {
        let target = scope.handle(target);
        let handler = scope.handle(handler);
        match get_trap(vm, heap, state, handler.value(), Trap::IsExtensible)? {
            TrapLookup::Threw => Ok(Flow::Threw),
            TrapLookup::None => internal_is_extensible(vm, heap, state, target.value()),
            TrapLookup::Trap(t) => {
                let result = call_trap(
                    vm,
                    heap,
                    state,
                    &scope,
                    t,
                    &[handler.value(), target.value()],
                )?;
                let Coercion::Value(result) = result else {
                    return Ok(Flow::Threw);
                };
                Ok(Flow::Value(
                    heap.no_gc(|nogc| Convert::is_truthy(nogc, result)),
                ))
            }
        }
    })
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
    fn new(scope: &'s HandleScope<'_>, partial: &PartialDescriptor) -> Self {
        Self {
            value: partial.value.map(|v| scope.handle(v)),
            get: partial.get.map(|v| scope.handle(v)),
            set: partial.set.map(|v| scope.handle(v)),
            writable: partial.writable,
            enumerable: partial.enumerable,
            configurable: partial.configurable,
        }
    }

    fn is_data_descriptor(&self) -> bool {
        self.partial().is_data_descriptor()
    }

    fn partial(&self) -> PartialDescriptor {
        PartialDescriptor {
            value: self.value.map(|h| h.value()),
            get: self.get.map(|h| h.value()),
            set: self.set.map(|h| h.value()),
            writable: self.writable,
            enumerable: self.enumerable,
            configurable: self.configurable,
        }
    }
}

/// CreateDataProperty-style full descriptor ({w,e,c} true) — the payload
/// of OrdinarySet's receiver-define step.
fn create_data_partial<'s>(scope: &'s HandleScope<'_>, value: Value) -> RootedPartial<'s> {
    RootedPartial {
        value: Some(scope.handle(value)),
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
        let mut fields: Vec<(Value, Value)> = Vec::new();
        if let Some(v) = &partial.value {
            fields.push((s.value.value(), v.value()));
        }
        if let Some(b) = partial.writable {
            fields.push((s.writable.value(), Convert::boolean(heap, b)));
        }
        if let Some(v) = &partial.get {
            fields.push((s.get.value(), v.value()));
        }
        if let Some(v) = &partial.set {
            fields.push((s.set.value(), v.value()));
        }
        if let Some(b) = partial.enumerable {
            fields.push((s.enumerable.value(), Convert::boolean(heap, b)));
        }
        if let Some(b) = partial.configurable {
            fields.push((s.configurable.value(), Convert::boolean(heap, b)));
        }
        for (name, value) in fields {
            let name = scope.handle(SlotName::from_value(name).tagged());
            crate::Object::define_own_property(
                heap,
                &scope,
                obj,
                name,
                PropertyDescriptor::data(value),
            )?;
        }
        Ok(obj.value())
    })
}

/// `[[DefineOwnProperty]]` dispatch: proxies run their `defineProperty`
/// trap (with full spec validation), ordinary objects complete the
/// partial against their current descriptor and define through the
/// transition machinery.
pub fn define_internal(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    obj: Value,
    name: Value,
    partial: PartialDescriptor,
) -> Result<Flow<bool>, VmError> {
    if heap.no_gc(|nogc| is_proxy(nogc, obj)) {
        return proxy_define(vm, heap, state, obj, name, partial);
    }
    state.handle_scope(|scope| {
        let obj = scope.handle(obj);
        let name = scope.handle(name);
        let undefined = heap.known().undefined.value();
        // dense array elements: element defines land in the backing
        // store, not as descriptors
        let is_element_on_array = heap.no_gc(|nogc| {
            crate::classify_key(nogc, name.value()).is_ok_and(|k| {
                matches!(k, Key::Element(_))
                    && obj
                        .value()
                        .as_heap_object(nogc)
                        .is_some_and(|o| o.as_ref().is_array(nogc))
            })
        });
        if is_element_on_array {
            let key = heap
                .no_gc(|nogc| crate::classify_key(nogc, name.value()))
                .expect("element classification cannot fail here");
            let Key::Element(i) = key else {
                unreachable!("checked above")
            };
            let completed = partial.complete_against(undefined, None);
            let PropertyDescriptor::Data { value, .. } = completed else {
                return Err(VmError::Type);
            };
            let value = scope.handle(value);
            store_array_element(heap, &scope, obj.value(), i, value.value())?;
            return Ok(Flow::Value(true));
        }
        let current = heap
            .no_gc(|nogc| crate::lookup::ordinary_own_descriptor(nogc, obj.value(), name.value()));
        // no allocation between reading `current` and the define call
        // (the define path roots its inputs before allocating)
        let full = partial.complete_against(undefined, current.as_ref());
        let defined = crate::Object::define_own_property_values(
            heap,
            &scope,
            obj.value(),
            SlotName::from_value(name.value()),
            full,
        )?;
        Ok(Flow::Value(defined))
    })
}

/// OrdinarySet (receiver-aware) for an ordinary target with a proxy
/// (or arbitrary) receiver: the spec's define-on-receiver shape.
fn ordinary_set_forward(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    target: Value,
    receiver: Value,
    name: Value,
    value: Value,
) -> Result<Coercion, VmError> {
    state.handle_scope(|scope| {
        let target = scope.handle(target);
        let receiver = scope.handle(receiver);
        let name = scope.handle(name);
        let value = scope.handle(value);

        // array element keys on arrays route through the define path
        // (which stores into the backing store)
        let classified = heap.no_gc(|nogc| {
            (
                crate::classify_key(nogc, name.value()),
                target
                    .value()
                    .as_heap_object(nogc)
                    .is_some_and(|o| o.as_ref().is_array(nogc)),
            )
        });
        if let (Ok(Key::Element(_)), true) = classified {
            let partial = create_data_partial(&scope, value.value());
            let flow = define_internal(
                vm,
                heap,
                state,
                receiver.value(),
                name.value(),
                partial.partial(),
            )?;
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
        let lookup = heap.no_gc(|nogc| {
            match target
                .value()
                .lookup(nogc, SlotName::from_value(name.value()))
            {
                Lookup::Data { flags, .. } => {
                    if flags.is_writable() {
                        SetLookup::DataWritable
                    } else {
                        SetLookup::DataReadonly
                    }
                }
                Lookup::Accessor { pair, .. } => SetLookup::Setter(pair.set.inner()),
                Lookup::NotFound => SetLookup::NotFound,
            }
        });
        match lookup {
            SetLookup::DataReadonly => Ok(Coercion::Value(Convert::boolean(heap, false))),
            SetLookup::DataWritable | SetLookup::NotFound => {
                let partial = create_data_partial(&scope, value.value());
                let flow = define_internal(
                    vm,
                    heap,
                    state,
                    receiver.value(),
                    name.value(),
                    partial.partial(),
                )?;
                define_flow_to_coercion(heap, flow)
            }
            SetLookup::Setter(setter) => {
                if setter == heap.known().undefined.value() {
                    return Ok(Coercion::Value(Convert::boolean(heap, false)));
                }
                let setter = scope.handle(setter);
                let result = NativeContext::new(vm, heap, state).call(
                    setter.value(),
                    scope.stage(&[receiver.value(), value.value()]),
                )?;
                if result == heap.known().exception.value() {
                    Ok(Coercion::Threw)
                } else {
                    Ok(Coercion::Value(Convert::boolean(heap, true)))
                }
            }
        }
    })
}

fn define_flow_to_coercion(heap: &Heap, flow: Flow<bool>) -> Result<Coercion, VmError> {
    Ok(match flow {
        Flow::Threw => Coercion::Threw,
        Flow::Value(b) => Coercion::Value(Convert::boolean(heap, b)),
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
    proxy: Value,
    receiver: Value,
    name: Value,
) -> Result<Coercion, VmError> {
    let (target, handler) = enter_trap(heap, proxy, Trap::Get)?;
    state.handle_scope(|scope| {
        let target = scope.handle(target);
        let handler = scope.handle(handler);
        let name = scope.handle(name);
        let receiver = scope.handle(receiver);
        match get_trap(vm, heap, state, handler.value(), Trap::Get)? {
            TrapLookup::Threw => Ok(Coercion::Threw),
            TrapLookup::None => {
                // forward: lookup on the target, getter `this` = the
                // original receiver (a proxy target re-enters its own
                // `get` trap with the same receiver)
                Runtime::get_property_on(
                    vm,
                    heap,
                    state,
                    target.value(),
                    receiver.value(),
                    name.value(),
                )
            }
            TrapLookup::Trap(t) => {
                let result = call_trap(
                    vm,
                    heap,
                    state,
                    &scope,
                    t,
                    &[handler.value(), target.value(), name.value(), receiver.value()],
                )?;
                let Coercion::Value(result) = result else {
                    return Ok(Coercion::Threw);
                };
                let result = scope.handle(result);
                // invariant (steps 9-11): a trap cannot lie about
                // non-configurable data / accessor properties
                let desc = internal_own_descriptor(vm, heap, state, target.value(), name.value())?;
                let Flow::Value(desc) = desc else {
                    return Ok(Coercion::Threw);
                };
                if let Some(d) = desc.as_ref().filter(|d| d.configurable == Some(false)) {
                    heap.no_gc(|nogc| {
                        let undefined = nogc.known().undefined.value();
                        if d.is_data_descriptor()
                            && d.writable == Some(false)
                            && !Compare::same_value(
                                nogc,
                                result.value(),
                                d.value.unwrap_or(undefined),
                            )
                        {
                            return Err(VmError::Message(
                                "proxy get trap must match a non-writable, non-configurable property",
                            ));
                        }
                        if d.is_accessor_descriptor()
                            && d.get.as_ref().is_some_and(|g| *g == undefined)
                            && result.value() != undefined
                        {
                            return Err(VmError::Message(
                                "proxy get trap must return undefined for an accessor without a getter",
                            ));
                        }
                        Ok(())
                    })?;
                }
                Ok(Coercion::Value(result.value()))
            }
        }
    })
}

/// Proxy `[[Set]]` (ES 20.2.5.10).
pub fn set(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    proxy: Value,
    name: Value,
    value: Value,
    receiver: Value,
) -> Result<Coercion, VmError> {
    let (target, handler) = enter_trap(heap, proxy, Trap::Set)?;
    state.handle_scope(|scope| {
        let target = scope.handle(target);
        let handler = scope.handle(handler);
        let name = scope.handle(name);
        let value = scope.handle(value);
        let receiver = scope.handle(receiver);
        match get_trap(vm, heap, state, handler.value(), Trap::Set)? {
            TrapLookup::Threw => Ok(Coercion::Threw),
            TrapLookup::None => {
                if heap.no_gc(|nogc| is_proxy(nogc, target.value())) {
                    set(vm, heap, state, target.value(), name.value(), value.value(), receiver.value())
                } else {
                    ordinary_set_forward(
                        vm,
                        heap,
                        state,
                        target.value(),
                        receiver.value(),
                        name.value(),
                        value.value(),
                    )
                }
            }
            TrapLookup::Trap(t) => {
                let result = call_trap(
                    vm,
                    heap,
                    state,
                    &scope,
                    t,
                    &[
                        handler.value(),
                        target.value(),
                        name.value(),
                        value.value(),
                        receiver.value(),
                    ],
                )?;
                let Coercion::Value(result) = result else {
                    return Ok(Coercion::Threw);
                };
                let truthy = heap.no_gc(|nogc| Convert::is_truthy(nogc, result));
                if !truthy {
                    // [[Set]] returned false; sloppy stores ignore it
                    return Ok(Coercion::Value(Convert::boolean(heap, false)));
                }
                // invariant (steps 11-13)
                let desc = internal_own_descriptor(vm, heap, state, target.value(), name.value())?;
                let Flow::Value(desc) = desc else {
                    return Ok(Coercion::Threw);
                };
                if let Some(d) = desc.as_ref().filter(|d| d.configurable == Some(false)) {
                    heap.no_gc(|nogc| {
                        let undefined = nogc.known().undefined.value();
                        if d.is_data_descriptor()
                            && d.writable == Some(false)
                            && !Compare::same_value(
                                nogc,
                                value.value(),
                                d.value.unwrap_or(undefined),
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
                Ok(Coercion::Value(Convert::boolean(heap, true)))
            }
        }
    })
}

/// Proxy `[[HasProperty]]` (ES 20.2.5.9).
pub fn has(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    proxy: Value,
    name: Value,
) -> Result<Coercion, VmError> {
    let (target, handler) = enter_trap(heap, proxy, Trap::Has)?;
    state.handle_scope(|scope| {
        let target = scope.handle(target);
        let handler = scope.handle(handler);
        let name = scope.handle(name);
        match get_trap(vm, heap, state, handler.value(), Trap::Has)? {
            TrapLookup::Threw => Ok(Coercion::Threw),
            TrapLookup::None => {
                if heap.no_gc(|nogc| is_proxy(nogc, target.value())) {
                    has(vm, heap, state, target.value(), name.value())
                } else {
                    let found = heap.no_gc(|nogc| {
                        has_property(nogc, target.value(), SlotName::from_value(name.value()))
                    });
                    Ok(Coercion::Value(Convert::boolean(heap, found)))
                }
            }
            TrapLookup::Trap(t) => {
                let result = call_trap(
                    vm,
                    heap,
                    state,
                    &scope,
                    t,
                    &[handler.value(), target.value(), name.value()],
                )?;
                let Coercion::Value(result) = result else {
                    return Ok(Coercion::Threw);
                };
                let truthy = heap.no_gc(|nogc| Convert::is_truthy(nogc, result));
                if truthy {
                    return Ok(Coercion::Value(Convert::boolean(heap, true)));
                }
                // invariant (steps 10-12): cannot hide non-configurable
                // or existing-on-non-extensible-target properties
                let desc = internal_own_descriptor(vm, heap, state, target.value(), name.value())?;
                let Flow::Value(desc) = desc else {
                    return Ok(Coercion::Threw);
                };
                if let Some(d) = desc {
                    if d.configurable == Some(false) {
                        return Err(VmError::Message(
                            "proxy has trap may not hide a non-configurable property",
                        ));
                    }
                    let ext = internal_is_extensible(vm, heap, state, target.value())?;
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
    })
}

/// Proxy `[[Delete]]` (ES 20.2.5.4). The strict-mode false→TypeError
/// translation is the caller's (`delete` sloppy/strict natives).
pub fn delete(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    proxy: Value,
    key: Value,
) -> Result<Coercion, VmError> {
    let (target, handler) = enter_trap(heap, proxy, Trap::DeleteProperty)?;
    state.handle_scope(|scope| {
        let target = scope.handle(target);
        let handler = scope.handle(handler);
        let key = scope.handle(key);
        match get_trap(vm, heap, state, handler.value(), Trap::DeleteProperty)? {
            TrapLookup::Threw => Ok(Coercion::Threw),
            TrapLookup::None => {
                if heap.no_gc(|nogc| is_proxy(nogc, target.value())) {
                    delete(vm, heap, state, target.value(), key.value())
                } else {
                    let target_obj = scope.cast::<Object>(target.value()).expect("ordinary target");
                    let deleted = crate::Object::delete_own_property(
                        heap,
                        &scope,
                        target_obj,
                        key.value(),
                    )?;
                    Ok(Coercion::Value(Convert::boolean(heap, deleted)))
                }
            }
            TrapLookup::Trap(t) => {
                let result = call_trap(
                    vm,
                    heap,
                    state,
                    &scope,
                    t,
                    &[handler.value(), target.value(), key.value()],
                )?;
                let Coercion::Value(result) = result else {
                    return Ok(Coercion::Threw);
                };
                let truthy = heap.no_gc(|nogc| Convert::is_truthy(nogc, result));
                if !truthy {
                    return Ok(Coercion::Value(Convert::boolean(heap, false)));
                }
                // invariant: cannot claim deletion of non-configurable
                // (or existing-on-non-extensible-target) properties
                let desc = internal_own_descriptor(vm, heap, state, target.value(), key.value())?;
                let Flow::Value(desc) = desc else {
                    return Ok(Coercion::Threw);
                };
                if let Some(d) = desc {
                    if d.configurable == Some(false) {
                        return Err(VmError::Message(
                            "proxy deleteProperty trap may not delete a non-configurable property",
                        ));
                    }
                    let ext = internal_is_extensible(vm, heap, state, target.value())?;
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
    })
}

/// Proxy `[[DefineOwnProperty]]` (ES 20.2.5.6).
pub fn proxy_define(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    proxy: Value,
    name: Value,
    partial: PartialDescriptor,
) -> Result<Flow<bool>, VmError> {
    let (target, handler) = enter_trap(heap, proxy, Trap::DefineProperty)?;
    state.handle_scope(|scope| {
        let target = scope.handle(target);
        let handler = scope.handle(handler);
        let name = scope.handle(name);
        let partial = RootedPartial::new(&scope, &partial);

        match get_trap(vm, heap, state, handler.value(), Trap::DefineProperty)? {
            TrapLookup::Threw => Ok(Flow::Threw),
            TrapLookup::None => define_internal(
                vm,
                heap,
                state,
                target.value(),
                name.value(),
                partial.partial(),
            ),
            TrapLookup::Trap(t) => {
                let desc_obj = descriptor_object(heap, state, &partial)?;
                let result = call_trap(
                    vm,
                    heap,
                    state,
                    &scope,
                    t,
                    &[handler.value(), target.value(), name.value(), desc_obj],
                )?;
                let Coercion::Value(result) = result else {
                    return Ok(Flow::Threw);
                };
                if !heap.no_gc(|nogc| Convert::is_truthy(nogc, result)) {
                    return Ok(Flow::Value(false));
                }
                // extensible first (a proxy target's trap allocates)
                let ext = internal_is_extensible(vm, heap, state, target.value())?;
                let Flow::Value(extensible) = ext else {
                    return Ok(Flow::Threw);
                };
                let desc = internal_own_descriptor(vm, heap, state, target.value(), name.value())?;
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
                        let compatible = heap.no_gc(|nogc| {
                            is_compatible_property_descriptor(
                                nogc,
                                extensible,
                                &partial.partial(),
                                Some(d),
                            )
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
    })
}

/// Proxy `[[Call]]` (ES 20.2.5.15): the `apply` trap, or a plain call
/// of the target. `args` includes the receiver (`this`) at index 0.
pub fn apply(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    proxy: Value,
    args: &[Value],
) -> Result<Coercion, VmError> {
    let (target, handler) = enter_trap(heap, proxy, Trap::Apply)?;
    state.handle_scope(|scope| {
        let target = scope.handle(target);
        let handler = scope.handle(handler);
        let this_arg = scope.handle(*args.first().unwrap_or(&heap.known().undefined.value()));
        match get_trap(vm, heap, state, handler.value(), Trap::Apply)? {
            TrapLookup::Threw => Ok(Coercion::Threw),
            TrapLookup::None => {
                let all: Vec<Value> = std::iter::once(this_arg.value())
                    .chain(args.iter().skip(1).copied())
                    .collect();
                let result =
                    NativeContext::new(vm, heap, state).call(target.value(), scope.stage(&all))?;
                if result == heap.known().exception.value() {
                    Ok(Coercion::Threw)
                } else {
                    Ok(Coercion::Value(result))
                }
            }
            TrapLookup::Trap(t) => {
                // args: (target, thisArg, argumentsList)
                let staged = scope.stage(&args[1..]);
                let arr = heap.new_array(&scope, staged).into_tagged().erase();
                let arr = scope.handle(arr);
                let result = call_trap(
                    vm,
                    heap,
                    state,
                    &scope,
                    t,
                    &[
                        handler.value(),
                        target.value(),
                        this_arg.value(),
                        arr.value(),
                    ],
                )?;
                Ok(result)
            }
        }
    })
}

/// Proxy `[[Construct]]` (ES 20.2.5.5): the `construct` trap, or a
/// construct of the target with the original `new.target`. `args` are
/// the constructor arguments (no synthesized receiver).
pub fn construct(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    proxy: Value,
    args: &[Value],
    new_target: Value,
) -> Result<Coercion, VmError> {
    let (target, handler) = enter_trap(heap, proxy, Trap::Construct)?;
    state.handle_scope(|scope| {
        let target = scope.handle(target);
        let handler = scope.handle(handler);
        let new_target = scope.handle(new_target);
        match get_trap(vm, heap, state, handler.value(), Trap::Construct)? {
            TrapLookup::Threw => Ok(Coercion::Threw),
            TrapLookup::None => {
                // forward: the full target [[Construct]] — synthesize the
                // receiver from new.target (the hole for derived class
                // constructors), run, and prefer an object result
                let derived = heap.no_gc(|nogc| {
                    target
                        .value()
                        .as_heap_object(nogc)
                        .and_then(|obj| {
                            obj.as_ref()
                                .header
                                .map
                                .heap_ref(nogc)
                                .kind()
                                .is_class_constructor()
                                .then(|| obj.as_ref().callable_info(nogc))
                        })
                        .flatten()
                        .is_some_and(|info| info.function_kind().is_derived_class_constructor())
                });
                let receiver = if derived {
                    heap.known().the_hole.value()
                } else {
                    let Some(r) = Runtime::create_construct_receiver_value(
                        vm,
                        heap,
                        state,
                        new_target.value(),
                    )?
                    else {
                        return Ok(Coercion::Threw);
                    };
                    r
                };
                let receiver = scope.handle(receiver);
                let mut all: Vec<Value> = Vec::with_capacity(args.len() + 1);
                all.push(receiver.value());
                all.extend_from_slice(args);
                let result = NativeContext::new(vm, heap, state).call_construct(
                    target.value(),
                    new_target.value(),
                    scope.stage(&all),
                )?;
                if result == heap.known().exception.value() {
                    return Ok(Coercion::Threw);
                }
                if heap.no_gc(|nogc| Convert::is_primitive(nogc, result)) {
                    if derived {
                        // a derived constructor may only return objects
                        return Err(VmError::Type);
                    }
                    return Ok(Coercion::Value(receiver.value()));
                }
                Ok(Coercion::Value(result))
            }
            TrapLookup::Trap(t) => {
                // args: (target, argumentsList, newTarget)
                let staged = scope.stage(args);
                let arr = heap.new_array(&scope, staged).into_tagged().erase();
                let arr = scope.handle(arr);
                let result = call_trap(
                    vm,
                    heap,
                    state,
                    &scope,
                    t,
                    &[
                        handler.value(),
                        target.value(),
                        arr.value(),
                        new_target.value(),
                    ],
                )?;
                let Coercion::Value(result) = result else {
                    return Ok(Coercion::Threw);
                };
                let result = scope.handle(result);
                // invariant: the trap must return an object
                if heap.no_gc(|nogc| Convert::is_primitive(nogc, result.value())) {
                    return Err(VmError::Message(
                        "proxy construct trap must return an object",
                    ));
                }
                Ok(Coercion::Value(result.value()))
            }
        }
    })
}

/// Ordinary `[[PreventExtensions]]`: clone the map without the
/// EXTENDABLE bit (maps are shared, so the clone isolates the object).
fn ordinary_prevent_extensions(heap: &mut Heap, scope: &HandleScope<'_>, obj: Handle<'_, Object>) {
    use crate::{
    GcSlice,
    MapInit,
    MapKind,
};
    let (kind, prototype, descriptors) = heap.no_gc(|nogc| {
        let map = obj.heap_ref(nogc).map_ref(nogc);
        (
            map.kind(),
            map.prototype.inner(),
            map.descriptors()
                .iter()
                .map(|d| (d.name(), d.flags(), scope.handle(d.value.inner())))
                .collect::<Vec<_>>(),
        )
    });
    if !kind.is_extendable() {
        return; // already non-extensible (idempotent)
    }
    let prototype = scope.handle(prototype);
    heap.allocate_token_enter_nogc(crate::Map::layout_for(descriptors.len()), |token, nogc| {
        let obj_ref = obj.heap_ref(nogc);
        let new_map = token.allocate::<Map>(MapInit {
            kind: MapKind::new(kind.bits() & !MapKind::EXTENDABLE.bits()),
            value_slot_count: obj_ref.map_ref(nogc).value_slot_count(),
            descriptors: &descriptors,
            prototype,
        });
        obj_ref
            .header
            .map
            .set(nogc, obj.value(), new_map.into_tagged());
    });
}

/// Proxy `[[PreventExtensions]]` (ES 20.2.5.3), plus the ordinary path —
/// the `Object.preventExtensions` implementation.
pub fn prevent_extensions(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    obj: Value,
) -> Result<Coercion, VmError> {
    if !heap.no_gc(|nogc| is_proxy(nogc, obj)) {
        return state.handle_scope(|scope| {
            let Some(obj) = scope.cast::<Object>(obj) else {
                return Err(VmError::Type);
            };
            ordinary_prevent_extensions(heap, &scope, obj);
            Ok(Coercion::Value(Convert::boolean(heap, true)))
        });
    }
    let (target, handler) = heap
        .no_gc(|nogc| parts(nogc, obj))
        .expect("checked proxy above");
    if handler == heap.known().null.value() {
        return Err(revoked_error(Trap::PreventExtensions));
    }
    state.handle_scope(|scope| {
        let target = scope.handle(target);
        let handler = scope.handle(handler);
        match get_trap(vm, heap, state, handler.value(), Trap::PreventExtensions)? {
            TrapLookup::Threw => Ok(Coercion::Threw),
            TrapLookup::None => prevent_extensions(vm, heap, state, target.value()),
            TrapLookup::Trap(t) => {
                let result = call_trap(
                    vm,
                    heap,
                    state,
                    &scope,
                    t,
                    &[handler.value(), target.value()],
                )?;
                let Coercion::Value(result) = result else {
                    return Ok(Coercion::Threw);
                };
                if !heap.no_gc(|nogc| Convert::is_truthy(nogc, result)) {
                    return Ok(Coercion::Value(Convert::boolean(heap, false)));
                }
                // invariant: returning true requires a non-extensible target
                let ext = internal_is_extensible(vm, heap, state, target.value())?;
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
    })
}

/// `Object.isExtensible` (ES 20.1.2.14) including the proxy trap and
/// its must-match-target invariant (ES 20.2.5.2 step 8).
pub fn is_extensible(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    obj: Value,
) -> Result<Coercion, VmError> {
    if !heap.no_gc(|nogc| is_proxy(nogc, obj)) {
        let extensible = heap.no_gc(|nogc| {
            obj.as_heap_object(nogc)
                .is_some_and(|o| o.as_ref().map_ref(nogc).kind().is_extendable())
        });
        return Ok(Coercion::Value(Convert::boolean(heap, extensible)));
    }
    let (target, handler) = heap
        .no_gc(|nogc| parts(nogc, obj))
        .expect("checked proxy above");
    if handler == heap.known().null.value() {
        return Err(revoked_error(Trap::IsExtensible));
    }
    state.handle_scope(|scope| {
        let target = scope.handle(target);
        let handler = scope.handle(handler);
        match get_trap(vm, heap, state, handler.value(), Trap::IsExtensible)? {
            TrapLookup::Threw => Ok(Coercion::Threw),
            TrapLookup::None => is_extensible(vm, heap, state, target.value()),
            TrapLookup::Trap(t) => {
                let result = call_trap(
                    vm,
                    heap,
                    state,
                    &scope,
                    t,
                    &[handler.value(), target.value()],
                )?;
                let Coercion::Value(result) = result else {
                    return Ok(Coercion::Threw);
                };
                let trap_bool = heap.no_gc(|nogc| Convert::is_truthy(nogc, result));
                let target_bool = internal_is_extensible(vm, heap, state, target.value())?;
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
    })
}
