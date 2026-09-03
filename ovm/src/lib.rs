use std::sync::{Arc, Mutex, Weak};

use core::cell::Cell;
use core::ptr::NonNull;

use vm::{
    AllocError, EdgeVisitable, Handle, HandleData, HandleScope, Heap, InternedString, LocalHeap,
    Object, RootVisitor, Value, Visitor,
};

pub mod cache;
pub mod interner;
pub mod interpreter;
pub mod natives;
pub mod stack;

pub use stack::{FrameMeta, STACK_SLOTS, Stack};

pub use cache::StackCache;

pub use interner::StringInterner;
pub use natives::{EXCEPTION_SENTINEL, NativeContext, NativeFn, NativeIndex, NativeRegistry};
pub use vm::VmError;

// TODO: get rid of the generic heap, instead make only init generic.
// in runtime we only want a Generic interface with vtable pointers probably
// this is especially problematic with C/native interaces which dont support generics
// also these generics have not shown any benifit really.
// additionnaly without generics we can do runtime GC swapping (whatever the usecase may be)
pub struct SharedVM<H: Heap> {
    heap: H,
    // TODO: investiage if Mutex is fine, maybe a lock-free mechanism exists
    threads: Mutex<Vec<Weak<ContextState>>>,
    interner: StringInterner,
    natives: NativeRegistry<H>,
}

pub struct VM<H: Heap> {
    shared: Arc<SharedVM<H>>,
}

// TODO: implement real exception and handler stack
pub struct ContextState {
    handles: HandleData,
    stack: Stack,
    cache: StackCache,
    pending_exception: Cell<Option<VmError>>,
}

impl ContextState {
    pub fn stack(&self) -> &Stack {
        &self.stack
    }

    pub fn set_pending_exception(&self, err: VmError) {
        self.pending_exception.set(Some(err));
    }

    pub fn take_pending_exception(&self) -> Option<VmError> {
        self.pending_exception.take()
    }

    pub fn has_pending_exception(&self) -> bool {
        self.pending_exception.get().is_some()
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
    }
}

pub struct Thread<H: Heap> {
    vm: VM<H>,
    heap: H::Local,
    state: Arc<ContextState>,
}

impl<H: Heap> Thread<H> {
    pub fn vm(&self) -> &VM<H> {
        &self.vm
    }

    pub fn heap(&mut self) -> &mut H::Local {
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

    pub fn set_pending_exception(&self, err: VmError) {
        self.state.set_pending_exception(err);
    }

    pub fn take_pending_exception(&self) -> Option<VmError> {
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
        interpreter::execute(&self.vm, &mut self.heap, &self.state, callable, args)
    }

    pub fn run_native(&mut self, f: NativeFn<H>, args: &[Value]) -> Result<Value, VmError> {
        let mut nctx = NativeContext::new(&self.vm, &mut self.heap, &self.state);
        f(&mut nctx, args)
    }
}

impl<H: Heap> Clone for VM<H> {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl<H: Heap> VM<H> {
    pub fn new(config: H::Config) -> Result<Self, AllocError> {
        let heap = H::new(config)?;
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

    pub fn visit_roots(&self, visitor: &mut impl RootVisitor) {
        self.shared.interner.visit_edges(visitor);
        let threads = self.shared.threads.lock().unwrap();
        for state in threads.iter().filter_map(Weak::upgrade) {
            state.visit_edges(visitor);
        }
    }

    pub fn attach(&self) -> Thread<H> {
        let heap = self.shared.heap.new_local();
        let void = heap.known().void.value();
        let state = Arc::new(ContextState {
            handles: HandleData::new(void),
            stack: Stack::new(STACK_SLOTS, void),
            cache: StackCache::new(void),
            pending_exception: Cell::new(None),
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
        F: FnOnce(&mut Thread<H>) -> R + Send + 'static,
        R: Send + 'static,
        H: 'static,
    {
        let vm = self.clone();
        std::thread::spawn(move || {
            let mut thread = vm.attach();
            f(&mut thread)
        })
    }
}
