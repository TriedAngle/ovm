use std::sync::{Arc, Mutex, Weak};

use core::cell::Cell;
use core::ptr::NonNull;

use vm::{AllocError, Handle, HandleData, HandleScope, Heap, InternedString};

pub mod interner;
pub mod natives;

pub use interner::StringInterner;
pub use natives::{EXCEPTION_SENTINEL, NativeFn, NativeIndex, NativeRegistry, VmError};

pub struct SharedVM<H: Heap> {
    heap: H,
    threads: Mutex<Vec<Weak<ContextState>>>,
    interner: StringInterner,
    natives: NativeRegistry<H>,
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
    pending_exception: Cell<Option<VmError>>,
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
                interner: StringInterner::new(),
                natives: NativeRegistry::new(),
            }),
        })
    }

    pub fn heap(&self) -> &H {
        &self.shared.heap
    }

    pub fn interner(&self) -> &StringInterner {
        &self.shared.interner
    }

    pub fn natives(&self) -> &NativeRegistry<H> {
        &self.shared.natives
    }

    pub fn native(&self, index: NativeIndex) -> NativeFn<H> {
        self.shared
            .natives
            .get(index)
            .expect("unknown native index")
    }

    pub fn register_native(&mut self, f: NativeFn<H>) -> NativeIndex {
        Arc::get_mut(&mut self.shared)
            .expect("cannot register natives on a shared VM")
            .natives
            .insert(f)
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
            pending_exception: Cell::new(None),
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

    pub fn intern<'s>(
        &mut self,
        scope: &'s HandleScope<'_>,
        s: impl AsRef<str>,
    ) -> Handle<'s, InternedString> {
        self.vm.interner().intern(&mut self.heap, scope, s)
    }

    /// Record a pending exception (set by failing natives; consumed by
    /// the interpreter's exception path).
    pub fn set_pending_exception(&self, err: VmError) {
        self.pending_exception.set(Some(err));
    }

    pub fn take_pending_exception(&self) -> Option<VmError> {
        self.pending_exception.take()
    }

    pub fn has_pending_exception(&self) -> bool {
        self.pending_exception.get().is_some()
    }

    pub fn handle_scope<R>(
        &mut self,
        f: impl for<'s> FnOnce(&mut Self, HandleScope<'s>) -> R,
    ) -> R {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        f(self, scope)
    }
}
