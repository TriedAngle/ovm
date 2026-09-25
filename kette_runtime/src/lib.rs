use vm_core::{Runtime, VM, VmError};

pub struct KetteRuntime;

impl Runtime for KetteRuntime {
    type State = ();

    fn setup(_vm: &mut VM, _state: &mut Self::State) -> Result<(), VmError> {
        Ok(())
    }
}
