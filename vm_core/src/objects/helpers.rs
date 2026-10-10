use crate::materialize::Materialize;
use crate::{
    DenseString, Float, Handle, HandleScope, HandleSlice, Heap, HostCtx, Map, MapInit, MapKind,
    NativeIndex, Object, PropertyDescriptor, Smi, Tagged, Thread, Value, VmError,
};

#[derive(Copy, Clone)]
pub enum WrapperKind {
    Number,
    Boolean,
    String,
}

impl WrapperKind {
    #[inline]
    fn matches(self, heap: &Heap, v: Tagged<'_, Value>) -> bool {
        match self {
            WrapperKind::Number => v.is_smi() || v.get_as::<Float>(heap).is_some(),
            WrapperKind::Boolean => {
                let known = heap.known();
                v == known.true_object.as_tagged(heap) || v == known.false_object.as_tagged(heap)
            }
            WrapperKind::String => v.get_as::<DenseString>(heap).is_some(),
        }
    }
}

impl Object {
    pub fn native_function<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        index: NativeIndex,
    ) -> Result<Handle<'s, Object>, VmError> {
        Self::make_native_object(heap, scope, index, true)
    }

    pub fn native_plain_function<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        index: NativeIndex,
    ) -> Result<Handle<'s, Object>, VmError> {
        Self::make_native_object(heap, scope, index, false)
    }

    fn make_native_object<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        index: NativeIndex,
        constructor: bool,
    ) -> Result<Handle<'s, Object>, VmError> {
        let mut kind = MapKind::OBJECT
            .union(MapKind::CALLABLE)
            .union(MapKind::NATIVE)
            .union(MapKind::EXTENDABLE);
        if constructor {
            kind = kind.union(MapKind::CONSTRUCTOR);
        }
        let map = heap.allocate_handle::<Map>(
            MapInit {
                kind,
                value_slot_count: 2,
                descriptors: &[],
                prototype: heap.known().function_prototype.erase(),
            },
            scope,
        );
        let empty_context = heap.known().empty_context;
        let obj = heap.new_object(
            scope,
            map,
            scope.stage(&[
                Smi::new(index.0 as i64).into_tagged(),
                empty_context.as_tagged(heap).erase(),
            ]),
        );
        Ok(scope.handle(obj))
    }

    // TODO: get rid of this.
    pub fn run_prelude(
        thread: &mut Thread,
        scope: &HandleScope<'_>,
        src: &str,
        name: &str,
        compile: bytecode::CompileFn,
    ) -> Result<(), VmError> {
        let (vm, heap, state) = thread.split();
        let empty = heap.known().empty_context;
        let closure = {
            let program = compile(src, bytecode::SourceMode::Script).map_err(|e| {
                eprintln!("{name} prelude compile error: {e}");
                VmError::Type
            })?;
            Materialize::closure_vm(vm, heap, state, scope, &program, empty)?
        };
        let (vm, heap, state) = thread.split();
        let exception = heap.known().exception.as_tagged(heap).raw();
        let result = HostCtx::enter(
            vm,
            &mut *heap,
            state,
            closure.erase(),
            HandleSlice::EMPTY,
            None,
        )?;
        if result.raw() == exception {
            if let Some(ex) = state.take_pending_exception() {
                eprintln!("{name} prelude threw: {ex:?}")
            }
            return Err(VmError::Type);
        }
        Ok(())
    }

    pub fn install_constructor<'s>(
        thread: &mut Thread,
        scope: &'s HandleScope<'_>,
        index: NativeIndex,
        name: &str,
        proto_parent: Handle<'_, Object>,
    ) -> Result<(Handle<'s, Object>, Handle<'s, Object>), VmError> {
        let name_str = thread.intern(scope, name);
        let fn_obj = Self::native_function(thread.heap(), scope, index)?;

        // prototype object: fresh extendable object chained to proto_parent
        let map = thread.heap().allocate_handle::<Map>(
            MapInit {
                kind: MapKind::OBJECT.union(MapKind::EXTENDABLE),
                value_slot_count: 0,
                descriptors: &[],
                prototype: proto_parent.erase(),
            },
            scope,
        );
        let proto = scope.handle(thread.heap().new_object(scope, map, HandleSlice::EMPTY));

        // proto.constructor = fn; fn.prototype = proto
        // (built-in methods/constructor properties are non-enumerable, ES 20+)
        let constructor_str = thread.intern(scope, "constructor");
        // Safety: fresh interned word, rooted below before any allocation.
        let constructor_name = scope.handle(constructor_str.as_tagged(&*thread.heap()));
        Object::define_own_property(
            thread.heap(),
            scope,
            proto,
            constructor_name,
            PropertyDescriptor::method(fn_obj.erase()),
        )?;
        let prototype_str = thread.intern(scope, "prototype");
        // Safety: fresh interned word, rooted below before any allocation.
        let prototype_name = scope.handle(prototype_str.as_tagged(&*thread.heap()));
        Object::define_own_property(
            thread.heap(),
            scope,
            fn_obj,
            prototype_name,
            PropertyDescriptor::method(proto.erase()),
        )?;

        // global.Name = fn
        let global = thread.heap().known().global_object;
        // Safety: fresh interned word, rooted below before any allocation.
        let name = scope.handle(name_str.as_tagged(&*thread.heap()));
        Object::define_own_property(
            thread.heap(),
            scope,
            global,
            name,
            PropertyDescriptor::data(fn_obj.erase()),
        )?;
        Ok((fn_obj, proto))
    }

    pub fn install_method(
        thread: &mut Thread,
        scope: &HandleScope<'_>,
        receiver: Handle<'_, Object>,
        name: &str,
        index: NativeIndex,
    ) -> Result<(), VmError> {
        let method = Self::native_function(thread.heap(), scope, index)?;
        let name_str = thread.intern(scope, name);
        // Safety: fresh interned word, rooted below before any allocation.
        let method_name = scope.handle(name_str.as_tagged(&*thread.heap()));
        Object::define_own_property(
            thread.heap(),
            scope,
            receiver,
            method_name,
            PropertyDescriptor::method(method.erase()),
        )?;
        Ok(())
    }

    /// Install a non-constructor native function (`Function.prototype.call`-
    /// style) as an own method of `receiver`.
    pub fn install_plain_method(
        thread: &mut Thread,
        scope: &HandleScope<'_>,
        receiver: Handle<'_, Object>,
        name: &str,
        index: NativeIndex,
    ) -> Result<(), VmError> {
        let method = Self::native_plain_function(thread.heap(), scope, index)?;
        let name_str = thread.intern(scope, name);
        // Safety: fresh interned word, rooted below before any allocation.
        let method_name = scope.handle(name_str.as_tagged(&*thread.heap()));
        Object::define_own_property(
            thread.heap(),
            scope,
            receiver,
            method_name,
            PropertyDescriptor::method(method.erase()),
        )?;
        Ok(())
    }
}

impl<'a> Tagged<'a, Value> {
    /// Read slots[0] of a `PRIMITIVE_WRAPPER` receiver — or the receiver
    /// itself when it is the matching unboxed primitive: builtin `this`-values
    /// are never auto-boxed (ES 5.2.3), so `Number.prototype.toString` and
    /// friends must accept raw Smi/Float/string/boolean receivers. A receiver
    /// of the wrong primitive type (and any symbol/null/undefined) is a
    /// TypeError.
    pub fn wrapper_value(
        self,
        heap: &'a Heap,
        kind: WrapperKind,
    ) -> Result<Tagged<'a, Value>, VmError> {
        if kind.matches(heap, self) {
            return Ok(self);
        }
        let Some(obj) = self.as_heap_object() else {
            return Err(VmError::Type);
        };
        let map = obj.as_ref().map(heap);
        if !map.kind().contains(MapKind::PRIMITIVE_WRAPPER) {
            return Err(VmError::Type);
        }
        let value = obj.as_ref().slots(heap).at(heap, 0);
        if kind.matches(heap, value) {
            Ok(value)
        } else {
            Err(VmError::Type)
        }
    }
}
