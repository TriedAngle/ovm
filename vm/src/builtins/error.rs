//! ES 20.5: Error constructors, Error.prototype.toString, and error
//! object materialization.

use crate::{
    ContextState, Convert, DenseString, GcSlice, Heap, Object, PropertyDescriptor, Tagged, VM,
    Value, VmError, runtime::Runtime,
};

pub fn error_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    make_error(nctx, args, "Error")
}

pub fn type_error_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    make_error(nctx, args, "TypeError")
}

pub fn reference_error_constructor(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    make_error(nctx, args, "ReferenceError")
}

pub fn make_error(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
    class: &str,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        let (vm, heap, _) = nctx.split();
        // root the message right away: the allocations below (new_object,
        // interning) would leave a raw copy stale
        let message = match heap.no_gc(|heap| args.get(heap, 1).map(|v| v.erase())) {
            // Safety: fresh argument word, consumed before any allocation.
            Some(v) => {
                let v = unsafe { Tagged::<Value>::from_value_unchecked(v) };
                scope.handle(Convert::to_string(heap, &scope, v)?)
            }
            None => scope.handle(
                vm.interner()
                    .intern_str(heap, &scope, "")
                    .as_tagged(heap)
                    .erase_type(),
            ),
        };
        let map = match class {
            "TypeError" => heap.known().type_error_map,
            "ReferenceError" => heap.known().reference_error_map,
            _ => heap.known().error_map,
        };
        let obj = heap
            .new_object(&scope, map, GcSlice::EMPTY)
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
        // Safety: fresh rooted-slot word, returned without an
        // intervening allocation.
        Ok(unsafe { obj.read_unchecked() })
    })
}

pub fn error_to_string(
    nctx: &mut crate::natives::NativeContext<'_>,
    args: GcSlice<'_>,
) -> Result<Value, VmError> {
    nctx.handle_scope(|nctx, scope| {
        // both [[Get]]s below run user code (getters): the receiver must
        // stay rooted across them
        // Safety: fresh argument word, rooted below before any allocation.
        let receiver_word = nctx
            .heap()
            .no_gc(|heap| Ok(args.get(heap, 0).ok_or(VmError::Arity)?.erase()))?;
        let receiver =
            scope.handle(unsafe { Tagged::<Value>::from_value_unchecked(receiver_word) });
        let (vm, heap, state) = nctx.split();
        let recv = receiver.as_tagged(heap).erase();
        let name = get_property(vm, heap, state, recv, "name")?;
        let recv = receiver.as_tagged(heap).erase();
        let message = get_property(vm, heap, state, recv, "message")?;
        let (vm, heap, _) = nctx.split();
        // each to_string/intern allocates: root both halves before the
        // concats read them
        // Safety: fresh words from the lookups above, consumed before any
        // allocation.
        let a = scope.handle(Convert::to_string(heap, &scope, unsafe {
            Tagged::<Value>::from_value_unchecked(name)
        })?);
        let b = scope.handle(Convert::to_string(heap, &scope, unsafe {
            Tagged::<Value>::from_value_unchecked(message)
        })?);
        let colon = vm.interner().intern_str(heap, &scope, ": ");
        let ab = DenseString::concat(heap, &scope, a, colon.erase());
        let ab = scope.handle(ab.as_tagged(heap).erase_type());
        let out = DenseString::concat(heap, &scope, ab, b);
        // Safety: fresh rooted-slot word, returned without an
        // intervening allocation.
        Ok(unsafe { out.read_unchecked() })
    })
}

pub fn get_property(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    receiver: Value,
    name: &str,
) -> Result<Value, VmError> {
    let name = state.handle_scope(|scope| {
        // Safety: fresh rooted-slot word, consumed below.
        unsafe {
            vm.interner()
                .intern_str(heap, &scope, name)
                .read_unchecked()
        }
    });
    match Runtime::get_property(vm, heap, state, receiver, name)? {
        crate::runtime::Coercion::Value(v) => Ok(v),
        // Safety: fresh root-slot word read for the immediate return.
        crate::runtime::Coercion::Threw => Ok(unsafe { heap.known().exception.read_unchecked() }),
    }
}
