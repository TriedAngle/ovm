use crate::{DenseString, GcSlice, Handle, Heap, Object, PropertyDescriptor, Value, VmError};

use crate::{ContextState, VM};

/// Materialize a VM error as an ECMAScript error object.
///
/// Returns a raw word that is fresh at the return point (the last
/// statement is a read under a shared reborrow, no allocation after):
/// callers store it into rooted memory (the pending-exception register)
/// immediately.
#[cold]
#[inline(never)]
pub fn error_from_vm_error(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    err: VmError,
) -> Result<Value, VmError> {
    state.handle_scope(|scope| {
        let name_value = vm.interner().intern_str(heap, &scope, err.name());
        let message_value = vm.interner().intern_str(heap, &scope, err.message());

        // per-class maps carry the right prototype chain (.constructor etc.)
        let map = match err.name() {
            "TypeError" => heap.known().type_error_map,
            "ReferenceError" => heap.known().reference_error_map,
            "RangeError" => heap.known().range_error_map,
            _ => heap.known().error_map,
        };
        let obj = heap
            .new_object(&scope, map, GcSlice::EMPTY)
            .into_handle(&scope);
        let name = heap.known().strings.name;
        let message = heap.known().strings.message;
        // root fresh copies before the (allocating) defines below
        let name_value: Handle<'_, DenseString> = scope.handle(name_value.as_tagged(heap));
        let message_value: Handle<'_, DenseString> = scope.handle(message_value.as_tagged(heap));
        Object::define_own_property(
            heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::data(name_value.erase()),
        )?;
        Object::define_own_property(
            heap,
            &scope,
            obj,
            message,
            PropertyDescriptor::data(message_value.erase()),
        )?;
        Ok(obj.as_tagged(heap).erase())
    })
}
