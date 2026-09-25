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
        Self::with_message(vm, heap, state, err.name(), err.message())
    }

    /// ReferenceError for an unresolvable binding reference (ES 6.2.3.3
    /// GetValue on a baseless reference): `'<name>' is not defined`.
    #[cold]
    #[inline(never)]
    pub fn not_defined<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        name: &str,
    ) -> Result<Tagged<'a, Value>, VmError> {
        Self::with_message(
            vm,
            heap,
            state,
            "ReferenceError",
            &format!("'{name}' is not defined"),
        )
    }

    /// Materialize an error object of the named class with an arbitrary
    /// message — the escape hatch for messages that must name a runtime
    /// value. Classes without a dedicated map fall back to the plain
    /// Error prototype chain.
    #[cold]
    #[inline(never)]
    pub fn with_message<'a>(
        vm: &VM,
        heap: &'a mut Heap,
        state: &ContextState,
        class: &str,
        message: &str,
    ) -> Result<Tagged<'a, Value>, VmError> {
        state.handle_scope(|scope| {
            let name_value = vm.interner().intern_str(heap, &scope, class);
            let message_value = vm.interner().intern_str(heap, &scope, message);

            // per-class maps carry the right prototype chain (.constructor etc.)
            let map = match class {
                "TypeError" => heap.known().type_error_map,
                "ReferenceError" => heap.known().reference_error_map,
                "RangeError" => heap.known().range_error_map,
                _ => heap.known().error_map,
            };
            let obj = heap
                .new_object(&scope, map, HandleSlice::EMPTY)
                .as_handle(&scope);
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
