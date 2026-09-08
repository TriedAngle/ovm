use std::sync::{Arc, Mutex, Weak};

use core::cell::Cell;
use core::ptr::NonNull;

use vm::{
    AllocError, EdgeVisitable, GlobalHeap, Handle, HandleData, HandleScope, Heap, HeapBackend,
    InternedString, Object, Register, RootHandles, RootVisitor, Smi, StringInterner, Value,
    Visitor, bootstrap_basics, bootstrap_well_known, intern_well_known_strings,
};

pub mod builtins;
pub mod cache;
pub mod errors;
pub mod interpreter;
pub mod materialize;
pub mod natives;
pub mod runtime;
pub mod stack;

pub use stack::{FrameMeta, STACK_SLOTS, Stack};

pub use cache::StackCache;

pub use errors::error_from_vm_error;
pub use natives::{NativeContext, NativeFn, NativeIndex, NativeRegistry};
pub use vm::VmError;

pub struct SharedVM {
    heap: GlobalHeap,
    roots: RootHandles,
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

    /// The current frame's context (the chain `LoadContextSlot` walks),
    /// for direct eval. `None` when no frame is executing.
    pub fn current_context(&self) -> Option<Value> {
        if !self.cache.is_active() {
            return None;
        }
        Some(self.stack.context_slot(&self.cache.frame_meta()).inner())
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

    /// Split the thread into its parts (multi-borrow calls).
    pub(crate) fn split(&mut self) -> (&VM, &mut Heap, &ContextState) {
        (&self.vm, &mut self.heap, &self.state)
    }

    pub fn state(&self) -> &ContextState {
        &self.state
    }

    pub fn intern<'s>(
        &mut self,
        scope: &'s HandleScope<'_>,
        s: impl AsRef<[u8]>,
    ) -> Handle<'s, InternedString> {
        self.vm.interner().intern(&mut self.heap, scope, s)
    }

    pub fn set_pending_exception(&mut self, err: VmError) {
        let ex = errors::error_from_vm_error(&self.vm, &mut self.heap, &self.state, err)
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
        self.state.handle_scope(|scope| {
            let args = scope.stage(args);
            interpreter::execute(&self.vm, &mut self.heap, &self.state, callable, args, None)
        })
    }

    pub fn error_object(&mut self, err: VmError) -> Result<Value, VmError> {
        errors::error_from_vm_error(&self.vm, &mut self.heap, &self.state, err)
    }

    pub fn run_native(&mut self, f: NativeFn, args: &[Value]) -> Result<Value, VmError> {
        let mut nctx = NativeContext::new(&self.vm, &mut self.heap, &self.state);
        // stage a rooted copy: the native may keep reading it across its
        // own allocations
        self.state
            .handle_scope(|scope| f(&mut nctx, scope.stage(args)))
    }

    /// Parse + compile + materialize + run a script to completion.
    /// The VM must have the builtin library installed (see
    /// [`VM::with_builtins`]).
    pub fn run_script(&mut self, src: &str) -> Result<Value, ScriptError> {
        self.run_compiled(src, base_compiler::compile_script)
    }

    /// Like [`Thread::run_script`], but compiled in REPL mode: top-level
    /// declarations become global object properties, so bindings persist
    /// across calls (and can be redeclared).
    pub fn run_script_repl(&mut self, src: &str) -> Result<Value, ScriptError> {
        self.run_compiled(src, base_compiler::compile_repl)
    }

    fn run_compiled(
        &mut self,
        src: &str,
        compile: fn(
            &parser::Ast,
        ) -> Result<base_compiler::CompiledScript, base_compiler::CompileError>,
    ) -> Result<Value, ScriptError> {
        let mut p = parser::Parser::new(parser::Utf8SliceStream::new(src));
        p.parse_script().map_err(ScriptError::Parse)?;
        let ast = p.into_ast();
        let compiled = compile(&ast).map_err(ScriptError::Compile)?;
        self.handle_scope(|thread, scope| {
            let closure = materialize::materialize_script(thread, &scope, &compiled)
                .map_err(ScriptError::Vm)?;
            thread.execute(closure, &[]).map_err(ScriptError::Vm)
        })
    }
}

/// Failure of any stage of [`Thread::run_script`].
#[derive(Debug)]
pub enum ScriptError {
    Parse(parser::ParseError),
    Compile(base_compiler::CompileError),
    Vm(VmError),
}

impl core::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "parse error: {e}"),
            Self::Compile(e) => write!(f, "compile error: {e}"),
            Self::Vm(e) => write!(f, "runtime error: {e:?}"),
        }
    }
}

impl std::error::Error for ScriptError {}

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
        let mut local = heap.new_local();
        let roots = unsafe { RootHandles::new(256, Smi::new(0).encode()) };
        let interner = StringInterner::new();
        bootstrap_basics(&mut local, &roots);
        intern_well_known_strings(&mut local, &interner, &roots);
        bootstrap_well_known(&mut local, &roots);
        Ok(Self {
            shared: Arc::new(SharedVM {
                heap,
                roots,
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
        self.shared.roots.visit_edges(visitor);
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

    /// Create a VM with the builtin library installed (Number, Boolean,
    /// Error, TypeError, eval, function prototypes).
    pub fn with_builtins<B: HeapBackend>(config: B::Config) -> Result<Self, VmError>
    where
        Self: Sized,
    {
        let mut vm = Self::new::<B>(config).map_err(|_| VmError::OutOfBounds)?;
        let idx = builtins::register_builtin_natives(&mut vm);
        builtins::install_builtins(&mut vm, &idx)?;
        Ok(vm)
    }
}
