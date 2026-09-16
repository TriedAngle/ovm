//! ES 20.5: Error constructors, Error.prototype.toString, and error
//! object materialization.

use crate::{ContextState, Convert, DenseString, GcSlice, Heap, Object, PropertyDescriptor, Value, VM, VmError, runtime::Runtime};

pub(crate) fn error_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    make_error(nctx, args, "Error")
}

pub(crate) fn type_error_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    make_error(nctx, args, "TypeError")
}

pub(crate) fn reference_error_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    make_error(nctx, args, "ReferenceError")
}

pub(crate) fn make_error(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
    class: &str,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let (vm, heap, _) = nctx.split();
        let message = match args.get(1) {
            Some(v) => Convert::to_string(heap, &scope, v)?,
            None => vm.interner().intern_str(heap, &scope, "").value(),
        };
        let map = match class {
            "TypeError" => heap.known().type_error_map,
            "ReferenceError" => heap.known().reference_error_map,
            _ => heap.known().error_map,
        };
        let obj = heap.new_object(&scope, map, &[]).into_handle(&scope);
        let name = heap.known().strings.name;
        let message_key = heap.known().strings.message;
        let class_value = vm.interner().intern_str(heap, &scope, class);
        let message_value = scope.handle(message);
        Object::define_own_property(
            heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::data(class_value.value()),
        )?;
        Object::define_own_property(
            heap,
            &scope,
            obj,
            message_key,
            PropertyDescriptor::data(message_value.value()),
        )?;
        Ok(obj.value())
    })
}

pub(crate) fn error_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    let receiver = args.get(0).ok_or(VmError::Arity)?;
    let (vm, heap, state) = nctx.split();
    let name = get_property(vm, heap, state, receiver, "name")?;
    let message = get_property(vm, heap, state, receiver, "message")?;
    nctx.handle_scope(|nctx, scope| {
        let (vm, heap, _) = nctx.split();
        let a = Convert::to_string(heap, &scope, name)?;
        let b = Convert::to_string(heap, &scope, message)?;
        let colon = vm.interner().intern_str(heap, &scope, ": ");
        let ab = DenseString::concat(heap, &scope, a, colon.value());
        Ok(DenseString::concat(heap, &scope, ab.value(), b).value())
    })
}

pub(crate) fn get_property(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    receiver: Value,
    name: &str,
) -> Result<Value, VmError> {
    let name = state.handle_scope(|scope| vm.interner().intern_str(heap, &scope, name).value());
    match Runtime::get_property(vm, heap, state, receiver, name)? {
        crate::runtime::Coercion::Value(v) => Ok(v),
        crate::runtime::Coercion::Threw => Ok(heap.known().exception.value()),
    }
}
