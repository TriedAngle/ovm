use vm_core::runtime_api::install_method;
use vm_core::{
    Convert, DenseString, HandleSlice, Map, MapInit, MapKind, Object, PropertyDescriptor, Runtime,
    RuntimeContext, Tagged, VM, Value, VmError,
};

pub struct KetteRuntime;

impl Runtime for KetteRuntime {
    type State = ();

    fn setup(vm: &mut VM, _state: &mut Self::State) -> Result<(), VmError> {
        let print = vm.register_runtime(console_print);
        let mut thread = vm.attach();
        thread.handle_scope(|thread, scope| {
            let object_prototype = thread.heap().known().object_prototype;
            let map = thread.heap().allocate_handle::<Map>(
                MapInit {
                    kind: MapKind::OBJECT.union(MapKind::EXTENDABLE),
                    value_slot_count: 0,
                    descriptors: &[],
                    prototype: object_prototype.erase(),
                },
                &scope,
            );
            let console = scope.handle(thread.heap().new_object(&scope, map, HandleSlice::EMPTY));
            install_method(thread, &scope, console, "print", print)?;
            let console_name = thread.intern(&scope, "Console");
            let console_name = scope.handle(console_name.as_tagged(&*thread.heap()));
            let global = thread.heap().known().global_object;
            Object::define_own_property(
                thread.heap(),
                &scope,
                global,
                console_name,
                PropertyDescriptor::data(console.erase()),
            )?;
            Ok(())
        })
    }
}

fn console_print<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
) -> Result<Tagged<'a, Value>, VmError> {
    let RuntimeContext {
        vm: _, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        match args.get(1) {
            Some(v) => {
                let text = scope.handle(Convert::to_string(heap, &scope, v)?);
                let text = text
                    .as_tagged(heap)
                    .get_as::<DenseString>()
                    .map(|s| s.to_rust_string(heap))
                    .ok_or(VmError::Type)?;
                println!("{text}");
            }
            None => println!(),
        }
        Ok(heap.known().undefined.as_tagged(heap).erase())
    })
}
