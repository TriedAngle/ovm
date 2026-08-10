use std::sync::{Arc, Mutex, Weak};

use core::ptr::NonNull;

use vm::{AllocError, HandleData, HandleScope, Heap};

pub struct SharedVM<H: Heap> {
    heap: H,
    threads: Mutex<Vec<Weak<ContextState>>>,
}

pub struct VM<H: Heap> {
    shared: Arc<SharedVM<H>>,
}

struct ContextState {
    handles: HandleData,
}

pub struct Context<H: Heap> {
    vm: VM<H>,
    heap: H::Local,
    state: Arc<ContextState>,
}

unsafe impl Send for ContextState {}
unsafe impl Sync for ContextState {}

impl<H: Heap> Clone for VM<H> {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl<H: Heap> VM<H> {
    pub fn new(config: H::Config) -> Result<Self, AllocError> {
        Ok(Self {
            shared: Arc::new(SharedVM {
                heap: H::new(config)?,
                threads: Mutex::new(Vec::new()),
            }),
        })
    }

    pub fn heap(&self) -> &H {
        &self.shared.heap
    }

    pub fn attach(&self) -> Context<H> {
        let state = Arc::new(ContextState {
            handles: HandleData::new(),
        });
        let mut threads = self.shared.threads.lock().unwrap();
        threads.retain(|t| t.strong_count() > 0);
        threads.push(Arc::downgrade(&state));
        Context {
            vm: self.clone(),
            heap: self.shared.heap.new_local(),
            state,
        }
    }

    pub fn spawn<F, R>(&self, f: F) -> std::thread::JoinHandle<R>
    where
        F: FnOnce(&mut Context<H>) -> R + Send + 'static,
        R: Send + 'static,
        H: 'static,
    {
        let vm = self.clone();
        std::thread::spawn(move || {
            let mut ctx = vm.attach();
            f(&mut ctx)
        })
    }
}

impl<H: Heap> Context<H> {
    pub fn vm(&self) -> &VM<H> {
        &self.vm
    }

    pub fn heap(&mut self) -> &mut H::Local {
        &mut self.heap
    }

    pub fn handle_scope<R>(
        &mut self,
        f: impl for<'s> FnOnce(&mut Self, HandleScope<'s>) -> R,
    ) -> R {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        f(self, scope)
    }
}
