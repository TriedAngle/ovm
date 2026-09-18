use crate::{
    DenseString, Handle, HandleSlice, Heap, Object, PropertyDescriptor, Tagged, Value, VmError,
};

use crate::{ContextState, VM};

/// Namespace for materializing VM errors as ECMAScript error objects.
pub struct Errors;

impl Errors {
    /// Materialize a VM error as an ECMAScript error object.
    ///
    /// The returned value is anchored to the heap borrow; callers store it
    /// into rooted memory (the pending-exception register) before the next
    /// allocation.
    #[cold]
    #[inline(never)]
    pub fn from_vm_error<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        err: VmError,
    ) -> Result<Tagged<'a, Value>, VmError> {
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
                .new_object(&scope, map, HandleSlice::EMPTY)
                .into_handle(&scope);
            let name = heap.known().strings.name;
            let message = heap.known().strings.message;
            // root fresh copies before the (allocating) defines below
            let name_value: Handle<'_, DenseString> = scope.handle(name_value.as_tagged(heap));
            let message_value: Handle<'_, DenseString> =
                scope.handle(message_value.as_tagged(heap));
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
}
