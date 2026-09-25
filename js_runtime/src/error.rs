use vm_core::Lookup;
use vm_core::RuntimeContext;
use vm_core::runtime::Coercion;

use vm_core::{
    Convert, DenseString, HandleSlice, Object, PropertyDescriptor, Tagged, Value, VmError,
};

pub fn error_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    make_error(nctx, args, "Error")
}

pub fn type_error_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    make_error(nctx, args, "TypeError")
}

pub fn reference_error_constructor<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    make_error(nctx, args, "ReferenceError")
}

pub fn make_error<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
    class: &str,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        // root the message right away: the allocations below (new_object,
        // interning) would leave a raw copy stale
        let message_word = args.get(1).map(|h| h.as_tagged(heap)).map(|v| v.raw());
        let message = match message_word {
            // Safety: fresh argument word, consumed before any allocation.
            Some(v) => {
                let v = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(v) });
                scope.handle(Convert::to_string(heap, &scope, v)?)
            }
            None => scope.handle(
                vm.interner()
                    .intern_str(heap, &scope, "")
                    .as_tagged(heap)
                    .erase(),
            ),
        };
        let map = match class {
            "TypeError" => heap.known().type_error_map,
            "ReferenceError" => heap.known().reference_error_map,
            _ => heap.known().error_map,
        };

        let obj = heap
            .new_object(&scope, map, HandleSlice::EMPTY)
            .as_handle(&scope);
        let name = heap.known().strings.name;
        let message_key = heap.known().strings.message;
        let class_value = vm.interner().intern_str(heap, &scope, class);
        Object::define_own_property(
            heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::data(class_value.erase()),
        )?;
        Object::define_own_property(
            heap,
            &scope,
            obj,
            message_key,
            PropertyDescriptor::data(message.erase()),
        )?;
        Ok(obj.as_tagged(heap).erase())
    })
}

pub fn error_to_string<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let receiver = args.get(0).ok_or(VmError::Arity)?;

        let name_key = vm.interner().intern_str(heap, &scope, "name");
        let name =
            match Lookup::get_property_on(vm, heap, state, receiver, receiver, name_key.erase())? {
                Coercion::Value(v) => scope.handle(v),
                Coercion::Threw => scope.handle(heap.known().exception.as_tagged(heap).erase()),
            };
        let message_key = vm.interner().intern_str(heap, &scope, "message");
        let message = match Lookup::get_property_on(
            vm,
            heap,
            state,
            receiver,
            receiver,
            message_key.erase(),
        )? {
            Coercion::Value(v) => scope.handle(v),
            Coercion::Threw => scope.handle(heap.known().exception.as_tagged(heap).erase()),
        };

        let a = scope.handle(Convert::to_string(heap, &scope, name)?);
        let b = scope.handle(Convert::to_string(heap, &scope, message)?);
        let colon = vm.interner().intern_str(heap, &scope, ": ");
        let ab = DenseString::concat(heap, &scope, a, colon.erase());
        let out = DenseString::concat(heap, &scope, ab.erase(), b);
        Ok(out.as_tagged(heap).erase())
    })
}
