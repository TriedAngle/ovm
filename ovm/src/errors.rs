use vm::{Heap, Object, ObjectSlotsInit, PropertyDescriptor, Value, VmError};

use crate::{ContextState, VM};

/// Materialize a VM error as an ECMAScript error object
#[cold]
#[inline(never)]
pub fn error_from_vm_error(
    vm: &VM,
    heap: &mut Heap,
    state: &ContextState,
    err: VmError,
) -> Result<Value, VmError> {
    state.handle_scope(|scope| {
        let name_value = vm.interner().intern(heap, &scope, err.name());
        let message_value = vm.interner().intern(heap, &scope, err.message());

        // per-class maps carry the right prototype chain (.constructor etc.)
        let map = match err.name() {
            "TypeError" => heap.known().type_error_map,
            "ReferenceError" => heap.known().reference_error_map,
            "RangeError" => heap.known().range_error_map,
            _ => heap.known().error_map,
        };
        let obj = heap
            .allocate_object(
                &scope,
                ObjectSlotsInit {
                    map,
                    values: &[],
                    elements: heap.known().empty_fixed_array.erase(),
                    length: 0,
                },
            )
            .into_handle(&scope);
        let name = heap.known().strings.name;
        let message = heap.known().strings.message;
        let name_value = scope.handle(name_value.value());
        let message_value = scope.handle(message_value.value());
        Object::define_own_property(
            heap,
            &scope,
            obj,
            name,
            PropertyDescriptor::data(name_value.value()),
        )?;
        Object::define_own_property(
            heap,
            &scope,
            obj,
            message,
            PropertyDescriptor::data(message_value.value()),
        )?;
        Ok(obj.value())
    })
}
