use core::ptr::NonNull;

use crate::{
    Convert, GcSlice, Handle, HandleScope, Heap, InternedString, Lookup, NoGc, Object,
    PropertyDescriptor, SlotName, Smi, Symbol, VMString, Value, VmError, private_find,
    runtime::Coercion,
};

use crate::{ContextState, Thread, VM};

pub struct NativeContext<'a> {
    vm: &'a VM,
    heap: &'a mut Heap,
    state: &'a ContextState,
    /// `new.target` of the active [[Construct]] call (ES 9.2.2): the
    /// invoked constructor, or `None` when called via [[Call]] (where
    /// `new.target` is undefined).
    // TODO: the Error family constructors read this once they exist
    new_target: Option<Handle<'a, Object>>,
}

impl<'a> NativeContext<'a> {
    pub fn new(vm: &'a VM, heap: &'a mut Heap, state: &'a ContextState) -> Self {
        Self {
            vm,
            heap,
            state,
            new_target: None,
        }
    }

    pub(crate) fn with_new_target(
        vm: &'a VM,
        heap: &'a mut Heap,
        state: &'a ContextState,
        new_target: Option<Handle<'a, Object>>,
    ) -> Self {
        Self {
            vm,
            heap,
            state,
            new_target,
        }
    }

    pub fn is_construct(&self) -> bool {
        self.new_target.is_some()
    }

    pub fn new_target(&self) -> Option<Value> {
        self.new_target.map(|h| h.value())
    }

    pub fn vm(&self) -> &VM {
        self.vm
    }

    /// Split the context into its parts (for multi-borrow calls).
    pub(crate) fn split(&mut self) -> (&VM, &mut Heap, &ContextState) {
        (&self.vm, &mut self.heap, &self.state)
    }

    pub fn heap(&mut self) -> &mut Heap {
        self.heap
    }

    pub fn intern<'s>(
        &mut self,
        scope: &'s HandleScope<'_>,
        s: impl AsRef<[u8]>,
    ) -> Handle<'s, InternedString> {
        self.vm.interner().intern(self.heap, scope, s)
    }

    pub fn set_pending_exception(&mut self, err: VmError) {
        let ex = crate::errors::error_from_vm_error(self.vm, self.heap, self.state, err)
            .expect("error materialization must not fail");
        self.state.set_pending_exception(ex);
    }

    pub fn handle_scope<R>(&mut self, f: impl FnOnce(&mut Self, HandleScope<'_>) -> R) -> R {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        f(self, scope)
    }

    pub fn call<'s>(&mut self, callable: Value, args: GcSlice<'s>) -> Result<Value, VmError> {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        let Some(callable) = scope.cast::<Object>(callable) else {
            return Err(VmError::Type);
        };
        crate::interpreter::execute(self.vm, self.heap, self.state, callable, args, None)
    }

    /// Invoke `callable` as a constructor with `new.target` = `new_target`:
    /// native callees see `is_construct()` and the receiver's prototype
    /// comes from `new_target.prototype` (ES 9.2.2).
    pub fn call_construct<'s>(
        &mut self,
        callable: Value,
        new_target: Value,
        args: GcSlice<'s>,
    ) -> Result<Value, VmError> {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        let Some(callable) = scope.cast::<Object>(callable) else {
            return Err(VmError::Type);
        };
        let Some(new_target) = scope.cast::<Object>(new_target) else {
            return Err(VmError::Type);
        };
        crate::interpreter::execute(
            self.vm,
            self.heap,
            self.state,
            callable,
            args,
            Some(new_target),
        )
    }

    pub fn take_pending_exception(&self) -> Option<Value> {
        self.state.take_pending_exception()
    }

    pub fn has_pending_exception(&self) -> bool {
        self.state.has_pending_exception()
    }

    /// The current frame's context (direct eval chains to it).
    pub fn current_context(&self) -> Option<Value> {
        self.state.current_context()
    }
}

pub type NativeFn = for<'a, 's> fn(&mut NativeContext<'a>, GcSlice<'s>) -> Result<Value, VmError>;

// TODO: this is still not a sound C interface (Rust-ABI `NativeFn` with
// reference/slice arguments); decide on the C ABI before exporting.
#[allow(improper_ctypes_definitions)]
pub extern "C" fn native_trampoline(
    f: NativeFn,
    thread: *mut Thread,
    args: *const Value,
    argc: u32,
) -> Value {
    let thread = unsafe { &mut *thread };
    // SAFETY: caller-owned C memory; the native must not read it after
    // allocating (see GcSlice)
    let args = unsafe { GcSlice::from_slice(core::slice::from_raw_parts(args, argc as usize)) };
    let mut nctx = NativeContext::new(&thread.vm, &mut thread.heap, &thread.state);
    match f(&mut nctx, args) {
        Ok(v) => v,
        Err(e) => {
            let ex =
                crate::errors::error_from_vm_error(&thread.vm, &mut thread.heap, &thread.state, e)
                    .expect("error materialization must not fail");
            thread.state.set_pending_exception(ex);
            thread.heap.known().exception.value()
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeIndex(pub usize);

impl NativeIndex {
    /// the fixed `RuntimeFn` table occupies 0..COUNT; dynamically
    /// registered natives (builtins) append after it
    pub const RUNTIME_TABLE_END: Self = Self(bytecode::RuntimeFn::COUNT as usize);
}

/// The fixed runtime-helper table: one implementation per
/// `bytecode::RuntimeFn`. The exhaustive match is the compile-time link
/// between the ABI ids and their implementations — adding a variant
/// without an entry here is a compile error, and `NativeRegistry::new`
/// registers them in `RuntimeFn::ALL` order so registry indices equal
/// discriminants.
fn runtime_fn(id: bytecode::RuntimeFn) -> NativeFn {
    match id {
        bytecode::RuntimeFn::GetIterator => get_iterator,
        bytecode::RuntimeFn::IteratorNext => iterator_next,
        bytecode::RuntimeFn::IteratorDone => iterator_done,
        bytecode::RuntimeFn::IteratorValue => iterator_value,
        bytecode::RuntimeFn::HasProperty => has_property,
        bytecode::RuntimeFn::CopyDataProperties => copy_data_properties,
        bytecode::RuntimeFn::CreatePrivateName => create_private_name,
        bytecode::RuntimeFn::PrivateGet => private_get,
        bytecode::RuntimeFn::PrivateSet => private_set,
        bytecode::RuntimeFn::PrivateIn => private_in,
        bytecode::RuntimeFn::SetClassFields => set_class_fields,
        bytecode::RuntimeFn::InitInstanceFields => init_instance_fields,
        bytecode::RuntimeFn::RequireObjectCoercible => require_object_coercible,
        bytecode::RuntimeFn::DeletePropertySloppy => delete_property_sloppy,
        bytecode::RuntimeFn::DeletePropertyStrict => delete_property_strict,
        bytecode::RuntimeFn::DeleteIdentifierSloppy => delete_identifier_sloppy,
        bytecode::RuntimeFn::DeleteSuperProperty => delete_super_property,
    }
}

pub struct NativeRegistry {
    entries: Vec<NativeFn>,
}

impl NativeRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            entries: Vec::new(),
        };
        // the runtime-helper table owns indices 0..COUNT: CallRuntime
        // operands carry RuntimeFn discriminants, so registration order
        // must (and is asserted to) match
        for (i, id) in bytecode::RuntimeFn::ALL.iter().enumerate() {
            debug_assert_eq!(*id as u16 as usize, i, "ALL order must match discriminants");
            let idx = registry.insert(runtime_fn(*id));
            debug_assert_eq!(idx.0, i, "runtime table must start at index 0");
        }
        registry
    }

    pub fn insert(&mut self, f: NativeFn) -> NativeIndex {
        let index = NativeIndex(self.entries.len());
        self.entries.push(f);
        index
    }

    pub fn get(&self, index: NativeIndex) -> Option<NativeFn> {
        self.entries.get(index.0).copied()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for NativeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---- runtime helpers ------------------------------------------------------

/// RequireObjectCoercible (ES 7.2.2): (value) -> value, TypeError on
/// null/undefined (object destructuring sources).
fn require_object_coercible(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    // internal-native convention: the register list IS the argument list
    // (no receiver slot)
    let arg = args.get(0).ok_or(VmError::Arity)?;
    let nullish = nctx
        .heap()
        .no_gc(|nogc| arg == nogc.known().null.value() || arg == nogc.known().undefined.value());
    if nullish {
        return Err(VmError::Type);
    }
    Ok(arg)
}

// ---- delete (ES 13.5.1) ----------------------------------------------------

/// `delete obj.key` in sloppy code: (obj, key) -> bool.
fn delete_property_sloppy(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    delete_property(nctx, args, false)
}

/// `delete obj.key` in strict code: (obj, key) -> bool, TypeError when
/// the delete fails (ES 13.5.1.2 step 4.h).
fn delete_property_strict(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    delete_property(nctx, args, true)
}

fn delete_property(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
    strict: bool,
) -> Result<Value, VmError> {
    let target = args.get(0).ok_or(VmError::Arity)?;
    let raw_key = args.get(1).ok_or(VmError::Arity)?;
    // the reference's key is coerced before the base is touched (ES
    // 13.15.5 EvaluatePropertyAccess: user toString/valueOf of a
    // computed key runs even when the delete afterwards throws)
    let (vm, heap, state) = nctx.split();
    let Some(key) = crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? else {
        return Ok(nctx.heap().known().exception.value());
    };
    let ok = delete_property_core(nctx, target, key)?;
    if strict && !ok {
        return Err(VmError::Type);
    }
    Ok(Convert::boolean(nctx.heap(), ok))
}

fn delete_property_core(
    nctx: &mut NativeContext<'_>,
    target: Value,
    key: Value,
) -> Result<bool, VmError> {
    // ToObject (ES 7.2.3): a null/undefined base throws
    let nullish = nctx.heap().no_gc(|nogc| {
        target == nogc.known().null.value() || target == nogc.known().undefined.value()
    });
    if nullish {
        return Err(VmError::Type);
    }
    // primitives: ToObject creates a fresh wrapper whose only own
    // properties are a string's non-configurable length/indices
    if nctx
        .heap()
        .no_gc(|nogc| Convert::is_primitive(nogc, target))
    {
        let owned = nctx
            .heap()
            .no_gc(|nogc| string_exotic_own(nogc, target, key));
        return Ok(!owned);
    }
    nctx.handle_scope(|nctx, scope| {
        let receiver = scope
            .cast::<Object>(target)
            .expect("non-primitive receivers are objects");
        Object::delete_own_property(nctx.heap(), &scope, receiver, key)
    })
}

/// Whether a ToObject'd primitive owns `key` non-configurably: only
/// String wrappers own anything — "length" and their indices (ES
/// 10.4.3.3/4 StringGetOwnProperty). Deleting those yields false; every
/// other primitive property deletes as absent (true).
fn string_exotic_own<'a>(nogc: &'a NoGc<'a>, target: Value, key: Value) -> bool {
    let Some(s) = target.get_as::<VMString>(nogc) else {
        return false;
    };
    if let Some(idx) = Smi::decode(key) {
        let i = idx.value();
        return i >= 0 && (i as u64) < utf16_length(s.as_slice(nogc)) as u64;
    }
    let Some(name) = key.get_as::<InternedString>(nogc) else {
        return false; // symbols own nothing on primitives
    };
    let bytes = name.string().as_slice(nogc);
    bytes == b"length" || canonical_index(bytes).is_some_and(|i| i < utf16_length(s.as_slice(nogc)))
}

/// Canonical array-index strings ("0", "1", "42"): digits only, no
/// leading zeros (ES 6.1.7). "01", "-0" and "1e2" are not indices.
fn canonical_index(bytes: &[u8]) -> Option<usize> {
    if bytes.is_empty() || bytes.len() > 10 {
        return None;
    }
    if bytes[0] == b'0' {
        return (bytes.len() == 1).then_some(0);
    }
    let mut n: usize = 0;
    for &b in bytes {
        let d = b.checked_sub(b'0')?;
        n = n.checked_mul(10)?.checked_add(d as usize)?;
    }
    Some(n)
}

/// WTF-8 bytes → UTF-16 code-unit count: BMP code points (1-3 byte
/// sequences) are one unit, supplementary code points (4-byte sequences)
/// are two.
fn utf16_length(bytes: &[u8]) -> usize {
    let mut units = 0;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let width = match b {
            0x00..=0x7F => 1,
            0x80..=0xDF => 2,
            0xE0..=0xEF => 3,
            _ => 4,
        };
        units += usize::from(width == 4) + 1;
        i += width;
    }
    units
}

/// Sloppy `delete x` on an unresolved name (ES 13.5.1.2 step 5 →
/// GlobalEnvironmentRecord.DeleteBinding): (name) -> bool. Declared
/// bindings resolve statically and compile to `false`; only global-object
/// properties reach here, and sloppy references never throw on failure.
fn delete_identifier_sloppy(
    nctx: &mut NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let name = args.get(0).ok_or(VmError::Arity)?;
    let global = nctx.heap().known().global_object.value();
    let ok = delete_property_core(nctx, global, name)?;
    Ok(Convert::boolean(nctx.heap(), ok))
}

/// `delete super.x` (ES 13.5.1.2 step 4.c): ReferenceError in both
/// language modes. The reference has already been evaluated (including
/// the uninitialized-`this` check and the key expression); the key is
/// never coerced — delete-super fails before any ToPropertyKey.
fn delete_super_property(
    _nctx: &mut NativeContext<'_>,
    _args: GcSlice<'_>,
) -> Result<Value, VmError> {
    Err(VmError::Reference)
}

/// GetIterator (ES 8.5.4): (obj) -> iterator.
fn get_iterator(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let obj = args.get(0).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let symbol = heap.known().iterator_symbol.value();
    let method = crate::runtime::Runtime::get_property(vm, heap, state, obj, symbol)?;
    let method = match method {
        Coercion::Threw => return Ok(nctx.heap().known().exception.value()),
        Coercion::Value(v) => v,
    };
    if method == nctx.heap().known().undefined.value()
        || method == nctx.heap().known().null.value()
        || !crate::runtime::Runtime::is_callable(nctx.heap(), method)
    {
        return Err(VmError::Type); // "obj is not iterable"
    }
    nctx.handle_scope(|nctx, scope| nctx.call(method, scope.stage(&[obj])))
}

/// IteratorNext (ES 8.5.6): (iterator) -> result object.
fn iterator_next(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let iter = args.get(0).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let next_name = heap.known().strings.next.value();
    let next = crate::runtime::Runtime::get_property(vm, heap, state, iter, next_name)?;
    let next = match next {
        Coercion::Threw => return Ok(nctx.heap().known().exception.value()),
        Coercion::Value(v) => v,
    };
    let result = nctx.handle_scope(|nctx, scope| nctx.call(next, scope.stage(&[iter])))?;
    if result == nctx.heap().known().exception.value() {
        return Ok(result);
    }
    if nctx
        .heap()
        .no_gc(|nogc| Convert::is_primitive(nogc, result))
    {
        return Err(VmError::Type); // IteratorNext result must be an Object
    }
    Ok(result)
}

/// IteratorComplete (ES 8.5.7): (result) -> bool.
fn iterator_done(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let result = args.get(0).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let done_name = heap.known().strings.done.value();
    let done = crate::runtime::Runtime::get_property(vm, heap, state, result, done_name)?;
    match done {
        Coercion::Threw => Ok(nctx.heap().known().exception.value()),
        Coercion::Value(v) => {
            let truthy = nctx.heap().no_gc(|nogc| Convert::is_truthy(nogc, v));
            Ok(Convert::boolean(nctx.heap(), truthy))
        }
    }
}

/// IteratorValue (ES 8.5.8): (result) -> value.
fn iterator_value(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let result = args.get(0).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let value_name = heap.known().strings.value.value();
    match crate::runtime::Runtime::get_property(vm, heap, state, result, value_name)? {
        Coercion::Threw => Ok(nctx.heap().known().exception.value()),
        Coercion::Value(v) => Ok(v),
    }
}

/// The `in` operator (ES 14.11.2): (key, obj) -> bool.
fn has_property(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let raw_key = args.get(0).ok_or(VmError::Arity)?;
    let obj = args.get(1).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let Some(key) = crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? else {
        return Ok(nctx.heap().known().exception.value());
    };
    let has = heap.no_gc(|nogc| crate::lookup::has_property(nogc, obj, SlotName::from_value(key)));
    Ok(Convert::boolean(nctx.heap(), has))
}

/// CopyDataProperties (ES 8.5.1) with an exclusion list (object rest):
/// (excluded..., target, source); `excluded` has count−2 entries.
fn copy_data_properties(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let n = args.len();
    if n < 2 {
        return Err(VmError::Arity);
    }
    let target = args.as_slice()[n - 2];
    let source = args.as_slice()[n - 1];
    let excluded: &[Value] = &args.as_slice()[..n - 2];
    let nullish = nctx.heap().no_gc(|nogc| {
        source == nogc.known().null.value() || source == nogc.known().undefined.value()
    });
    if nullish {
        return Ok(target);
    }
    // only heap objects contribute (string sources need boxing)
    if nctx
        .heap()
        .no_gc(|nogc| Convert::is_primitive(nogc, source))
    {
        return Ok(target);
    }
    // enumerate own enumerable keys: element indices ascending, then named
    // descriptors in insertion order
    let mut keys: Vec<Value> = Vec::new();
    nctx.heap().no_gc(|nogc| {
        let Some(obj) = source.as_heap_object(nogc) else {
            return;
        };
        if obj.as_ref().is_array(nogc) {
            let len = obj.as_ref().length().min(
                obj.as_ref()
                    .elements_array(nogc)
                    .map(|e| e.len())
                    .unwrap_or(0),
            );
            for i in 0..len {
                if obj.as_ref().element_value(nogc, i).is_some() {
                    keys.push(Smi::new(i as i64).encode());
                }
            }
        }
        for d in obj.as_ref().header.map.heap_ref(nogc).descriptors() {
            if d.flags().is_enumerable() {
                keys.push(d.name().value());
            }
        }
    });
    // canonicalize the excluded keys (interning strings) so a plain bits
    // comparison suffices against the source's descriptor names
    let excluded: Vec<Value> = {
        let (vm, heap, state) = nctx.split();
        let mut out = Vec::with_capacity(excluded.len());
        for &k in excluded {
            match crate::runtime::Runtime::to_property_key(vm, heap, state, k)? {
                Some(k) => out.push(k),
                None => return Ok(nctx.heap().known().exception.value()),
            }
        }
        out
    };
    let threw = nctx.handle_scope(|nctx, scope| -> Result<bool, VmError> {
        let (vm, heap, state) = nctx.split();
        for key in keys {
            if excluded.contains(&key) {
                continue;
            }
            let key = scope.handle(key);
            // full [[Get]] (getters may run)
            let value = match crate::runtime::Runtime::get_property(
                vm,
                heap,
                state,
                source,
                key.value(),
            )? {
                Coercion::Threw => return Ok(true),
                Coercion::Value(v) => v,
            };
            // CreateDataProperty: skipped when already present
            let exists = heap.no_gc(|nogc| {
                !matches!(
                    target.lookup(nogc, SlotName::from_value(key.value())),
                    Lookup::NotFound
                )
            });
            if exists {
                continue;
            }
            let value = scope.handle(value);
            let target = scope.handle(target);
            Object::add_own_property_values(
                heap,
                &scope,
                target.value(),
                SlotName::from_value(key.value()),
                PropertyDescriptor::data(value.value()),
            )?;
        }
        Ok(false)
    })?;
    if threw {
        return Ok(nctx.heap().known().exception.value());
    }
    Ok(target)
}

/// A fresh private name: (description) -> Symbol.
fn create_private_name(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let desc = args.get(0).ok_or(VmError::Arity)?;
    let bytes = nctx.heap().no_gc(|nogc| {
        desc.get_as::<VMString>(nogc)
            .map(|s| s.as_slice(nogc).to_vec())
    });
    nctx.handle_scope(|nctx, scope| {
        let desc = bytes.unwrap_or_default();
        Ok(Symbol::new(nctx.heap(), &scope, &desc).as_tagged().erase())
    })
}

/// PrivateGet (ES 7.3.30): (obj, key) -> value, TypeError when absent.
fn private_get(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let obj = args.get(0).ok_or(VmError::Arity)?;
    let key = args.get(1).ok_or(VmError::Arity)?;
    match nctx
        .heap()
        .no_gc(|nogc| private_find(nogc, obj, key).map(|s| s.inner()))
    {
        Some(v) => Ok(v),
        None => Err(VmError::Type),
    }
}

/// PrivateSet (ES 7.3.31): (obj, key, value), TypeError when absent.
fn private_set(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let obj = args.get(0).ok_or(VmError::Arity)?;
    let key = args.get(1).ok_or(VmError::Arity)?;
    let value = args.get(2).ok_or(VmError::Arity)?;
    let stored = nctx
        .heap()
        .no_gc(|nogc| match private_find(nogc, obj, key) {
            Some(slot) => {
                slot.set(nogc, obj, value);
                true
            }
            None => false,
        });
    if !stored {
        return Err(VmError::Type);
    }
    Ok(value)
}

/// `#x in obj`: (key, obj) -> bool (own private presence only).
fn private_in(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let key = args.get(0).ok_or(VmError::Arity)?;
    let obj = args.get(1).ok_or(VmError::Arity)?;
    let has = nctx
        .heap()
        .no_gc(|nogc| private_find(nogc, obj, key).is_some());
    Ok(Convert::boolean(nctx.heap(), has))
}

/// Attach the instance-field array to the class constructor:
/// (ctor, fields).
fn set_class_fields(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let ctor = args.get(0).ok_or(VmError::Arity)?;
    let fields = args.get(1).ok_or(VmError::Arity)?;
    let ok = nctx.heap().no_gc(|nogc| {
        let Some(obj) = ctor.as_heap_object(nogc) else {
            return false;
        };
        if !obj
            .as_ref()
            .header
            .map
            .heap_ref(nogc)
            .kind()
            .is_class_constructor()
        {
            return false;
        }
        let slots = obj.as_ref().slots.heap_ref(nogc);
        if slots.len() < 3 {
            return false;
        }
        slots.as_ref().element_slot(2).set(nogc, ctor, fields);
        true
    });
    if !ok {
        return Err(VmError::Type);
    }
    Ok(ctor)
}

/// InitializeInstanceElements (ES 7.3.33): (ctor, instance) -> instance.
/// Runs each field initializer with the instance as receiver and defines
/// the result onto it ({w+, e+, c+}).
fn init_instance_fields(nctx: &mut NativeContext<'_>, args: GcSlice<'_>) -> Result<Value, VmError> {
    let ctor = args.get(0).ok_or(VmError::Arity)?;
    let instance = args.get(1).ok_or(VmError::Arity)?;
    let fields = nctx.heap().no_gc(|nogc| {
        let Some(obj) = ctor.as_heap_object(nogc) else {
            return None;
        };
        let slots = obj.as_ref().slots.heap_ref(nogc);
        (slots.len() >= 3).then(|| slots.at(2))
    });
    let Some(fields) = fields else {
        return Err(VmError::Type);
    };
    if fields == nctx.heap().known().undefined.value() {
        return Ok(instance);
    }
    let count = nctx.heap().no_gc(|nogc| {
        fields
            .as_heap_object(nogc)
            .map(|o| o.as_ref().length())
            .unwrap_or(0)
    });
    let threw_or_failed = nctx.handle_scope(|nctx, scope| -> Result<bool, VmError> {
        let instance = scope.handle(instance);
        let fields = scope.handle(fields);
        let exception = nctx.heap().known().exception.value();
        let mut i = 0;
        while i + 1 < count {
            let raw_key = nctx.heap().no_gc(|nogc| {
                fields
                    .value()
                    .as_heap_object(nogc)
                    .and_then(|o| o.as_ref().element_value(nogc, i))
                    .unwrap_or_else(|| nogc.known().undefined.value())
            });
            let init = nctx.heap().no_gc(|nogc| {
                fields
                    .value()
                    .as_heap_object(nogc)
                    .and_then(|o| o.as_ref().element_value(nogc, i + 1))
                    .unwrap_or_else(|| nogc.known().undefined.value())
            });
            // computed keys need ToPropertyKey canonicalization
            let key = {
                let (vm, heap, state) = nctx.split();
                match crate::runtime::Runtime::to_property_key(vm, heap, state, raw_key)? {
                    Some(k) => k,
                    None => return Ok(true),
                }
            };
            let value = nctx.call(init, scope.stage(&[instance.value()]))?;
            if value == exception {
                return Ok(true);
            }
            let key = scope.handle(key);
            let value = scope.handle(value);
            let defined = {
                let heap = nctx.heap();
                Object::define_own_property_values(
                    heap,
                    &scope,
                    instance.value(),
                    SlotName::from_value(key.value()),
                    PropertyDescriptor::Data {
                        value: value.value(),
                        writable: true,
                        enumerable: true,
                        configurable: true,
                    },
                )?
            };
            if !defined {
                return Err(VmError::Type);
            }
            i += 2;
        }
        Ok(false)
    })?;
    if threw_or_failed {
        return Ok(nctx.heap().known().exception.value());
    }
    Ok(instance)
}
