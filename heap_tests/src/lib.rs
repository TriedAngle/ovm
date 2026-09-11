use vm::{HeapBackend, VM};

pub fn vm<B: HeapBackend>(config: B::Config) -> VM {
    VM::new::<B>(config).expect("failed to create heap")
}

pub fn vm_with_builtins<B: HeapBackend>(config: B::Config) -> VM {
    VM::with_builtins::<B>(config).expect("failed to create heap")
}

/// Instantiates each generic test function once per backend.
#[macro_export]
macro_rules! for_each_backend {
    ($($name:ident),* $(,)?) => {
        mod backend_dummy_heap {
            use super::*;
            $(
                #[test]
                fn $name() {
                    super::$name::<::dummy_heap::DummyHeap>();
                }
            )*
        }
        mod backend_mark_sweep {
            use super::*;
            $(
                #[test]
                fn $name() {
                    super::$name::<::mark_sweep::MarkSweep>();
                }
            )*
        }
    };
}
