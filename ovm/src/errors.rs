use vm::{Heap, Object, ObjectSlotsInit, SlotName, Tagged, Value, VmError};

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
        let name_string = vm.interner().intern(heap, &scope, "name");
        let message_string = vm.interner().intern(heap, &scope, "message");
        let name_value = vm.interner().intern(heap, &scope, err.name());
        let message_value = vm.interner().intern(heap, &scope, err.message());

        let map = scope
            .create_handle(heap.known().error_map.as_tagged())
            .expect("error map is strong");
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
        let name = scope
            .create_handle(SlotName::from(name_string.as_tagged()).tagged())
            .expect("name is strong");
        let message = scope
            .create_handle(SlotName::from(message_string.as_tagged()).tagged())
            .expect("message is strong");
        let name_value = scope
            .create_handle(Tagged::from_value(name_value.value()))
            .expect("name value is strong");
        let message_value = scope
            .create_handle(Tagged::from_value(message_value.value()))
            .expect("message value is strong");
        Object::store_new_data_property(heap, obj, name, name_value)?;
        Object::store_new_data_property(heap, obj, message, message_value)?;
        Ok(obj.value())
    })
}
