use std::sync::{Arc, Mutex, Weak};

use core::cell::Cell;
use core::ptr::NonNull;

use vm::{
    AllocError, EdgeVisitable, GlobalHeap, Handle, HandleData, HandleScope, Heap, HeapBackend,
    InternedString, Object, Register, RootVisitor, Value, Visitor,
};

pub mod cache;
pub mod interner;
pub mod interpreter;
pub mod natives;
pub mod stack;

pub use stack::{FrameMeta, STACK_SLOTS, Stack};

pub use cache::StackCache;

pub use interner::StringInterner;
pub use interpreter::error_from_vm_error;
pub use natives::{EXCEPTION_SENTINEL, NativeContext, NativeFn, NativeIndex, NativeRegistry};
pub use vm::VmError;

pub struct SharedVM {
    heap: GlobalHeap,
    // TODO: investiage if Mutex is fine, maybe a lock-free mechanism exists
    threads: Mutex<Vec<Weak<ContextState>>>,
    interner: StringInterner,
    natives: NativeRegistry,
}

pub struct VM {
    shared: Arc<SharedVM>,
}

pub struct ContextState {
    handles: HandleData,
    stack: Stack,
    cache: StackCache,
    pending_exception: Register,
    has_pending_exception: Cell<bool>,
}

impl ContextState {
    pub fn stack(&self) -> &Stack {
        &self.stack
    }

    pub fn set_pending_exception(&self, value: Value) {
        self.pending_exception.store(value);
        self.has_pending_exception.set(true);
    }

    pub fn take_pending_exception(&self) -> Option<Value> {
        if self.has_pending_exception.get() {
            self.has_pending_exception.set(false);
            Some(self.pending_exception.inner())
        } else {
            None
        }
    }

    pub fn has_pending_exception(&self) -> bool {
        self.has_pending_exception.get()
    }

    pub fn handle_scope<R>(&self, f: impl for<'s> FnOnce(HandleScope<'s>) -> R) -> R {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.handles)) };
        f(scope)
    }
}

unsafe impl Send for ContextState {}
// TODO: can we get rid of this somehow?
// in practice we seem to need Sync because Heap needs access,
// in theory full isolation (and passing?) should be possible
unsafe impl Sync for ContextState {}

impl EdgeVisitable for ContextState {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        self.handles.visit_edges(visitor);
        self.stack.visit_edges(visitor);
        self.cache.visit_edges(visitor);
        visitor.visit(self.pending_exception.as_raw());
    }
}

pub struct Thread {
    vm: VM,
    heap: Heap,
    state: Arc<ContextState>,
}

impl Thread {
    pub fn vm(&self) -> &VM {
        &self.vm
    }

    pub fn heap(&mut self) -> &mut Heap {
        &mut self.heap
    }

    pub fn state(&self) -> &ContextState {
        &self.state
    }

    pub fn intern<'s>(
        &mut self,
        scope: &'s HandleScope<'_>,
        s: impl AsRef<str>,
    ) -> Handle<'s, InternedString> {
        self.vm.interner().intern(&mut self.heap, scope, s)
    }

    pub fn set_pending_exception(&mut self, err: VmError) {
        let ex = interpreter::error_from_vm_error(&self.vm, &mut self.heap, &self.state, err)
            .expect("error materialization must not fail");
        self.state.set_pending_exception(ex);
    }

    pub fn take_pending_exception(&self) -> Option<Value> {
        self.state.take_pending_exception()
    }

    pub fn has_pending_exception(&self) -> bool {
        self.state.has_pending_exception()
    }

    pub fn handle_scope<R>(
        &mut self,
        f: impl for<'s> FnOnce(&mut Self, HandleScope<'s>) -> R,
    ) -> R {
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&self.state.handles)) };
        f(self, scope)
    }

    pub fn execute(
        &mut self,
        callable: Handle<'_, Object>,
        args: &[Value],
    ) -> Result<Value, VmError> {
        debug_assert_eq!(self.state.stack.top(), 0);
        debug_assert_eq!(self.state.stack.frame_depth(), 0);
        debug_assert!(!self.state.cache.is_active());

        // don't leak pending exception if it exists
        let _ = self.state.take_pending_exception();
        interpreter::execute(&self.vm, &mut self.heap, &self.state, callable, args)
    }

    pub fn error_object(&mut self, err: VmError) -> Result<Value, VmError> {
        interpreter::error_from_vm_error(&self.vm, &mut self.heap, &self.state, err)
    }

    pub fn run_native(&mut self, f: NativeFn, args: &[Value]) -> Result<Value, VmError> {
        let mut nctx = NativeContext::new(&self.vm, &mut self.heap, &self.state);
        f(&mut nctx, args)
    }
}

impl Clone for VM {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl VM {
    pub fn new<B: HeapBackend>(config: B::Config) -> Result<Self, AllocError> {
        let heap = B::new(config)?.into_global();
        heap.install_well_known_maps();
        let interner = StringInterner::new();
        // canonical empty string
        interner.insert("", heap.known().empty_string);
        Ok(Self {
            shared: Arc::new(SharedVM {
                heap,
                threads: Mutex::new(Vec::new()),
                interner,
                natives: NativeRegistry::new(),
            }),
        })
    }

    pub fn heap(&self) -> &GlobalHeap {
        &self.shared.heap
    }

    pub fn interner(&self) -> &StringInterner {
        &self.shared.interner
    }

    pub fn natives(&self) -> &NativeRegistry {
        &self.shared.natives
    }

    pub fn native(&self, index: NativeIndex) -> NativeFn {
        self.shared
            .natives
            .get(index)
            .expect("unknown native index")
    }

    pub fn register_native(&mut self, f: NativeFn) -> NativeIndex {
        Arc::get_mut(&mut self.shared)
            .expect("cannot register natives on a shared VM")
            .natives
            .insert(f)
    }

    pub fn visit_roots(&self, visitor: &mut impl RootVisitor) {
        self.shared.interner.visit_edges(visitor);
        self.shared.heap.iterate_roots(visitor);
        let threads = self.shared.threads.lock().unwrap();
        for state in threads.iter().filter_map(Weak::upgrade) {
            state.visit_edges(visitor);
        }
    }

    pub fn attach(&self) -> Thread {
        let heap = self.shared.heap.new_local();
        let void = heap.known().void.value();
        let state = Arc::new(ContextState {
            handles: HandleData::new(void),
            stack: Stack::new(STACK_SLOTS, void),
            cache: StackCache::new(void),
            pending_exception: unsafe { Register::from_value(void) },
            has_pending_exception: Cell::new(false),
        });
        let mut threads = self.shared.threads.lock().unwrap();
        // TODO: should we really call this every attach() ?
        threads.retain(|t| t.strong_count() > 0);
        threads.push(Arc::downgrade(&state));
        Thread {
            vm: self.clone(),
            heap,
            state,
        }
    }

    pub fn spawn<F, R>(&self, f: F) -> std::thread::JoinHandle<R>
    where
        F: FnOnce(&mut Thread) -> R + Send + 'static,
        R: Send + 'static,
    {
        let vm = self.clone();
        std::thread::spawn(move || {
            let mut thread = vm.attach();
            f(&mut thread)
        })
    }
}
