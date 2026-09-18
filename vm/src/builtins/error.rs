//! ES 20.5: Error constructors, Error.prototype.toString, and error
//! object materialization.
use crate::Lookup;
use crate::RuntimeContext;
use crate::runtime::Coercion;

use crate::{
    ContextState, Convert, DenseString, HandleSlice, Heap, Object, PropertyDescriptor, Tagged, VM,
    Value, VmError,
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
            .into_handle(&scope);
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
        // both [[Get]]s below run user code (getters): the receiver must
        // stay rooted across them
        // Safety: fresh argument word, rooted below before any allocation.
        let receiver_word = args
            .get(0)
            .map(|h| h.as_tagged(heap))
            .ok_or(VmError::Arity)?
            .raw();
        let receiver =
            scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(receiver_word) });
        let recv = receiver.as_tagged(heap).raw();
        let name = get_property(vm, heap, state, recv, "name")?;
        let recv = receiver.as_tagged(heap).raw();
        let message = get_property(vm, heap, state, recv, "message")?;
        // each to_string/intern allocates: root both halves before the
        // concats read them
        // Safety: fresh words from the lookups above, consumed before any
        // allocation.
        let name = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(name) });
        let a = scope.handle(Convert::to_string(heap, &scope, name)?);
        let message = scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(message) });
        let b = scope.handle(Convert::to_string(heap, &scope, message)?);
        let colon = vm.interner().intern_str(heap, &scope, ": ");
        let ab = DenseString::concat(heap, &scope, a, colon.erase());
        let ab = scope.handle(ab.as_tagged(heap).erase());
        let out = DenseString::concat(heap, &scope, ab, b);
        Ok(out.as_tagged(heap).erase())
    })
}

pub fn get_property(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    receiver: Value,
    name: &str,
) -> Result<Value, VmError> {
    state.handle_scope(|scope| {
        let name = scope.handle(
            vm.interner()
                .intern_str(heap, &scope, name)
                .as_tagged(heap)
                .erase(),
        );
        // Safety: caller-supplied word, fresh at entry.
        let receiver = scope.handle(unsafe { receiver.assume_valid(heap) });
        match Lookup::get_property_on(vm, heap, state, receiver, receiver, name)? {
            Coercion::Value(v) => Ok(v.raw()),
            Coercion::Threw => Ok(heap.known().exception.as_tagged(heap).raw()),
        }
    })
}
