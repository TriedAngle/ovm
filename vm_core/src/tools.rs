use crate::runtime_api::make_runtime_plain_function_in;
use crate::{
    HandleScope, HandleSlice, Heap, Map, MapInit, MapKind, Object, PropertyDescriptor,
    RuntimeContext, RuntimeIndex, StringInterner, Tagged, Value, VmError,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Termination {
    Shutdown,
}

pub struct KetteTools;

impl KetteTools {
    pub fn force_minor_gc<'a>(
        ctx: RuntimeContext<'a>,
        _args: HandleSlice<'_>,
    ) -> Result<Tagged<'a, Value>, VmError> {
        let RuntimeContext { heap, .. } = ctx;
        heap.collect_minor();
        Ok(heap.known().undefined.as_tagged(heap).erase())
    }

    pub fn force_major_gc<'a>(
        ctx: RuntimeContext<'a>,
        _args: HandleSlice<'_>,
    ) -> Result<Tagged<'a, Value>, VmError> {
        let RuntimeContext { heap, .. } = ctx;
        heap.collect();
        Ok(heap.known().undefined.as_tagged(heap).erase())
    }

    pub fn shutdown<'a>(
        ctx: RuntimeContext<'a>,
        _args: HandleSlice<'_>,
    ) -> Result<Tagged<'a, Value>, VmError> {
        let RuntimeContext {
            vm, heap, state, ..
        } = ctx;
        heap.cancel_executions(&|| {});
        vm.note_shutdown();
        state.set_termination(Termination::Shutdown);
        state.set_pending_exception(heap.known().undefined.as_tagged(heap));
        Ok(heap.known().exception.as_tagged(heap).erase())
    }

    pub fn install(
        heap: &mut Heap,
        interner: &StringInterner,
        scope: &HandleScope<'_>,
    ) -> Result<(), VmError> {
        let global = heap.known().global_object;
        let tools = {
            let map = heap.allocate_handle::<Map>(
                MapInit {
                    kind: MapKind::OBJECT.union(MapKind::EXTENDABLE),
                    value_slot_count: 0,
                    descriptors: &[],
                    prototype: heap.known().object_prototype.erase(),
                },
                scope,
            );
            scope.handle(heap.new_object(scope, map, HandleSlice::EMPTY))
        };

        let methods: &[(&str, RuntimeIndex)] = &[
            (
                "forceMinorGC",
                RuntimeIndex(bytecode::RuntimeFn::ForceMinorGc as usize),
            ),
            (
                "forceMajorGC",
                RuntimeIndex(bytecode::RuntimeFn::ForceMajorGc as usize),
            ),
            (
                "shutdown",
                RuntimeIndex(bytecode::RuntimeFn::ShutdownVm as usize),
            ),
        ];
        for (name, index) in methods {
            let method = make_runtime_plain_function_in(heap, scope, *index)?;
            let name_str = interner.intern_str(heap, scope, name);
            let method_name = scope.handle(name_str.as_tagged(heap));
            Object::define_own_property(
                heap,
                scope,
                tools,
                method_name,
                PropertyDescriptor::method(method.erase()),
            )?;
        }

        let name_str = interner.intern_str(heap, scope, "KetteTools");
        let name = scope.handle(name_str.as_tagged(heap));
        Object::define_own_property(
            heap,
            scope,
            global,
            name,
            PropertyDescriptor::data(tools.erase()),
        )?;
        Ok(())
    }
}
