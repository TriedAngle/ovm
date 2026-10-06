use vm_core::raise_runtime;
use vm_core::runtime_api::install_method;
use vm_core::{
    DenseString, Handle, HandleSlice, Map, MapInit, MapKind, Object, PropertyDescriptor, Runtime,
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
            // a bare global `print(x)`, same native as `Console.print`
            install_method(thread, &scope, global, "print", print)?;
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
    _new_target: Option<Handle<'_, Value>>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let Some(v) = args.get(1) else {
        println!();
        return heap.known().undefined.as_tagged(heap).erase();
    };
    let text = match Object::to_string(vm, heap, state, v) {
        Ok(Some(t)) => {
            let word = t.raw();
            // Safety: fresh string word, no allocation since the read.
            unsafe { word.assume_valid(heap) }
                .get_as::<DenseString>()
                .map(|s| s.to_rust_string(heap))
                .unwrap_or_default()
        }
        Ok(None) => return heap.known().exception.as_tagged(heap).erase(),
        Err(err) => return raise_runtime(vm, heap, state, err),
    };
    println!("{text}");
    heap.known().undefined.as_tagged(heap).erase()
}
