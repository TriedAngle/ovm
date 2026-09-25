use std::any::{Any, TypeId};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use core::cell::Cell;
use core::ptr::NonNull;

pub mod bootstrap;
pub mod cache;
pub mod compare;
pub mod convert;
pub mod error;
pub mod errors;
pub mod handle;
pub mod heap;
pub mod ic;
pub mod interner;
pub mod intrinsics;
pub mod lookup;
pub mod materialize;
pub mod objects;
pub mod runtime;
pub mod runtime_api;
pub mod stack;
pub mod tools;
pub mod transition;
pub mod value;

pub use objects::proxy;
pub use objects::string;

pub use bootstrap::{
    KnownCell, WellKnown, WellKnownStrings, bootstrap_basics, bootstrap_well_known,
    intern_well_known_strings,
};
pub use cache::{Acc, StackCache};
pub use compare::Compare;
pub use convert::Convert;
pub use error::VmError;
pub use errors::Errors;
pub use handle::{
    EscapableHandleScope, Handle, HandleData, HandleScope, HandleSet, HandleSlice, RootHandles,
};
pub use heap::{
    AllocToken, AtomicOptionGcSlot, EdgeVisitable, GcSlot, GlobalHeap, Heap, MaybeWeakGcSlot,
    OptionGcSlot, Register, WordType,
};
pub use interner::StringInterner;
pub use lookup::{Key, LoadOutcome, Lookup};
pub use objects::{
    AccessorPair, CallTarget, CallableInfoInit, CallableInfoObject, Context, ContextInit,
    DenseString, Encoding, FeedbackVector, FeedbackVectorInit, FixedArray, FixedByteArray, Float,
    FunctionKind, HandlerEntry, HandlerEntryInit, HandlerTable, HandlerTableInit, Header,
    HeapObject, Map, MapInit, MapKind, Object, ObjectInit, ObjectKind, ObjectSlotsInit, ProxyInit,
    ProxyObject, ScopeInfo, ScopeInfoInit, SlotDescriptor, SlotFlags, SlotName, StringData, Symbol,
    WeakFixedArray, WeakFixedArrayInit, decode_wtf8, new_feedback_vector, object_kind,
    object_layout, string_content_hash, visit_object,
};
pub use runtime::{
    Coercion, ErasedRuntimeState, ExecuteFn, Hint, Interpreter, Runtime, RuntimeCall,
    RuntimeContext, RuntimeIndex, RuntimeRegistry,
};
pub use stack::{FrameMeta, STACK_SLOTS, Stack};
pub use tools::{KetteTools, Termination};
pub use transition::{
    Change, PartialDescriptor, PropertyDescriptor, StoreOutcome, StoreSemantics, Transition,
};
pub use value::{
    HeapPtr, MaybeWeak, PTR_BIT, STRONG_PTR, Smi, TAG_MASK, TAG_SMI, Tagged, Value, WEAK_BIT,
    WEAK_PTR, Word, encode_smi,
};

pub use heap_api::{
    AllocError, GcHost, HeapBackend, HeapStats, LocalHeap, RawCell, SharedHeap, Visitor,
};

pub type Local<'scope, T> = Handle<'scope, T>;
// pseudo-static
pub type Global<T> = Handle<'static, T>;

/// A bytecode frontend, selectable by type at the call site
/// (`vm.eval::<JavascriptCompiler>(..)`).
pub trait Compiler {
    fn compile(
        source: &str,
        mode: bytecode::SourceMode,
    ) -> Result<bytecode::Program, bytecode::FrontendError>;
}

/// Number of slots per handle block.
pub const HANDLE_BLOCK_SIZE: usize = 1024;

const _: () = {
    use core::mem::size_of;
    assert!(size_of::<Value>() == size_of::<Word>());
    assert!(size_of::<Tagged<Value>>() == size_of::<Word>());
    assert!(size_of::<Tagged<DenseString>>() == size_of::<Word>());
    assert!(size_of::<GcSlot>() == size_of::<Word>());
    assert!(size_of::<MaybeWeakGcSlot>() == size_of::<Word>());
    assert!(size_of::<OptionGcSlot<FixedArray>>() == size_of::<Word>());
    assert!(size_of::<GcSlot<Smi>>() == size_of::<Word>());
    assert!(size_of::<GcSlot<DenseString>>() == size_of::<Word>());
    assert!(size_of::<Register>() == size_of::<Word>());
    // SlotName is a newtype over the erased name word; names travel as
    // `Tagged<'_, SlotName>`
    assert!(size_of::<SlotName>() == size_of::<Word>());
};

pub struct SharedVM {
    heap: GlobalHeap,
    roots: RootHandles,
    known: KnownCell,
    // TODO: investiage if Mutex is fine, maybe a lock-free mechanism exists
    threads: Mutex<Vec<Weak<ContextState>>>,
    interner: StringInterner,
    runtimes: RuntimeRegistry,
    states: Mutex<Vec<(TypeId, Box<dyn ErasedRuntimeState>)>>,
    /// Weak slots the GC clears when their targets die; used for tests and
    /// the seed of a weak-registry feature.
    weak_slots: Mutex<Vec<RawCell>>,
    /// Informational only: set once `KetteTools.shutdown()` ran. Cancels
    /// live on the safepoint nodes, not here — the VM stays usable after a
    /// shutdown.
    shutdown_requested: AtomicBool,
    /// Interpreter entry ([[Call]]/[[Construct]] on bytecode callables);
    /// set once at construction from the `I: Interpreter` type parameter.
    execute: ExecuteFn,
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
    /// Why this thread's execution terminated, if it did. Set exactly once,
    /// right before the uncatchable unwind; guest code can never observe or
    /// intercept it.
    termination: Cell<Option<Termination>>,
}

impl ContextState {
    pub fn stack(&self) -> &Stack {
        &self.stack
    }

    pub fn cache(&self) -> &StackCache {
        &self.cache
    }

    pub fn set_pending_exception<'x, T: 'x>(&self, value: Tagged<'x, T>) {
        self.pending_exception.store(value);
        self.has_pending_exception.set(true);
    }

    pub fn take_pending_exception(&self) -> Option<Value> {
        if self.has_pending_exception.get() {
            self.has_pending_exception.set(false);
            Some(self.pending_exception.raw())
        } else {
            None
        }
    }

    /// Like [`Self::take_pending_exception`], but anchored to the heap
    /// borrow: the register is GC-visited, so the read is current.
    pub fn take_pending_exception_tagged<'a>(&self, heap: &'a Heap) -> Option<Tagged<'a, Value>> {
        if self.has_pending_exception.get() {
            self.has_pending_exception.set(false);
            Some(self.pending_exception.get(heap))
        } else {
            None
        }
    }

    pub fn has_pending_exception(&self) -> bool {
        self.has_pending_exception.get()
    }

    /// Why this thread terminated, if it did (uncatchable by guest code).
    pub fn termination(&self) -> Option<Termination> {
        self.termination.get()
    }

    pub fn set_termination(&self, termination: Termination) {
        self.termination.set(Some(termination));
    }

    /// Reset the termination marker for a fresh run: a terminated
    /// execution ends here, the thread itself keeps going.
    pub fn clear_termination(&self) {
        self.termination.set(None);
    }

    /// The current frame's context (the chain `LoadContextSlot` walks),
    /// for direct eval. `None` when no frame is executing.
    pub fn current_context<'a>(&self, heap: &'a Heap) -> Option<Tagged<'a, Value>> {
        if !self.cache.is_active() {
            return None;
        }
        Some(self.stack.context(heap, &self.cache.frame_meta()))
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

impl SharedVM {
    fn visit_roots(&self, visitor: &mut dyn Visitor) {
        self.interner.visit_edges(visitor);
        self.roots.visit_edges(visitor);
        self.heap.iterate_roots(visitor);
        for slot in self.weak_slots.lock().unwrap().iter() {
            visitor.visit(slot);
        }
        for (_, state) in self.states.lock().unwrap().iter() {
            state.visit_edges(visitor);
        }
        let threads = self.threads.lock().unwrap();
        for state in threads.iter().filter_map(Weak::upgrade) {
            state.handles.visit_edges(visitor);
            state.stack.visit_edges(visitor);
            state.cache.visit_edges(visitor);
            visitor.visit(state.pending_exception.as_raw());
        }
    }

    fn gc_host(&self) -> GcHost {
        GcHost {
            ctx: self as *const SharedVM as *const (),
            visit_roots: host_visit_roots,
            layout_of: host_layout_of,
            visit_object: host_visit_object,
        }
    }

    /// Install `KetteTools` on the global object. Owns the bootstrap-style
    /// stack handle scope; the installed object stays reachable through
    /// the global object after it dies.
    fn install_tools(&self, local: &mut Heap) {
        let data = HandleData::new(local.known().the_hole.raw());
        // Safety: `data` outlives every use of the scope below.
        let scope = unsafe { HandleScope::from_raw(NonNull::from(&data)) };
        KetteTools::install(local, &self.interner, &scope)
            .expect("installing KetteTools must not fail");
    }
}

fn host_visit_roots(ctx: *const (), visitor: &mut dyn Visitor) {
    let shared = unsafe { &*(ctx as *const SharedVM) };
    shared.visit_roots(visitor);
}

fn host_layout_of(addr: NonNull<()>) -> core::alloc::Layout {
    unsafe { object_layout(addr) }
}

fn host_visit_object(addr: NonNull<()>, visitor: &mut dyn Visitor) {
    unsafe { visit_object(addr, visitor) }
}

impl EdgeVisitable for ContextState {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
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
    pub fn split(&mut self) -> (&VM, &mut Heap, &ContextState) {
        (&self.vm, &mut self.heap, &self.state)
    }

    pub fn state(&self) -> &ContextState {
        &self.state
    }

    pub fn intern<'s>(&mut self, scope: &'s HandleScope<'_>, s: &str) -> Handle<'s, DenseString> {
        self.vm.interner().intern_str(&mut self.heap, scope, s)
    }

    pub fn set_pending_exception(&mut self, err: VmError) {
        let ex = Errors::from_vm_error(&self.vm, &mut self.heap, &self.state, err)
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
        self.state.clear_termination();
        let _ = self.heap.take_cancel();
        let result = self.state.handle_scope(|scope| {
            let args = scope.stage(
                &args
                    .iter()
                    .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                    .collect::<Vec<_>>(),
            );
            (self.vm.shared.execute)(&self.vm, &mut self.heap, &self.state, callable, args, None)
                .map(|v| v.raw())
        });
        if self.state.termination().is_some() {
            let _ = self.state.take_pending_exception();
            let heap = &self.heap;
            return Ok(heap.known().undefined.as_tagged(heap).raw());
        }
        result
    }

    pub fn error_object(&mut self, err: VmError) -> Result<Value, VmError> {
        Errors::from_vm_error(&self.vm, &mut self.heap, &self.state, err).map(|v| v.raw())
    }

    pub fn run_runtime(&mut self, f: RuntimeCall, args: &[Value]) -> Result<Value, VmError> {
        let nctx = RuntimeContext::new(&self.vm, &mut self.heap, &self.state);
        // stage a rooted copy: the runtime may keep reading it across its
        // own allocations
        // Safety: caller-owned words staged before any allocation.
        self.state.handle_scope(|scope| {
            f(
                nctx,
                scope.stage(
                    &args
                        .iter()
                        .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                        .collect::<Vec<_>>(),
                ),
            )
            .map(|v| v.raw())
        })
    }

    pub fn run_source(
        &mut self,
        src: &str,
        compile: bytecode::CompileFn,
        mode: bytecode::SourceMode,
    ) -> Result<Value, ScriptError> {
        let program = compile(src, mode).map_err(ScriptError::from_frontend)?;
        self.handle_scope(|thread, scope| {
            let closure = materialize::Materialize::script(thread, &scope, &program)
                .map_err(ScriptError::Vm)?;
            thread.execute(closure, &[]).map_err(ScriptError::Vm)
        })
    }

    pub fn eval<C: Compiler>(&mut self, src: &str) -> Result<Value, ScriptError> {
        self.run_source(src, C::compile, bytecode::SourceMode::Script)
    }

    pub fn eval_repl<C: Compiler>(&mut self, src: &str) -> Result<Value, ScriptError> {
        self.run_source(src, C::compile, bytecode::SourceMode::Repl)
    }
}

/// Failure of any stage of [`Thread::eval`].
#[derive(Debug)]
pub enum ScriptError {
    Parse(bytecode::FrontendError),
    Compile(bytecode::FrontendError),
    Vm(VmError),
}

impl ScriptError {
    fn from_frontend(err: bytecode::FrontendError) -> Self {
        match err.kind {
            bytecode::FrontendErrorKind::Syntax => ScriptError::Parse(err),
            bytecode::FrontendErrorKind::Compile => ScriptError::Compile(err),
        }
    }
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
    pub fn new<B: HeapBackend, I: Interpreter>(config: B::Config) -> Result<Self, AllocError> {
        let heap = GlobalHeap::from_backend::<B>(config)?;
        let roots = unsafe { RootHandles::new(512, Smi::new(0).encode()) };
        let interner = StringInterner::new();
        let shared = Arc::new(SharedVM {
            heap,
            known: KnownCell::new(WellKnown::uninit(&roots)),
            roots,
            threads: Mutex::new(Vec::new()),
            interner,
            runtimes: RuntimeRegistry::new(),
            states: Mutex::new(Vec::new()),
            weak_slots: Mutex::new(Vec::new()),
            shutdown_requested: AtomicBool::new(false),
            execute: I::EXECUTE,
        });
        shared.heap.set_host(shared.gc_host());
        let mut local = shared.heap.new_local(&shared.known);
        bootstrap_basics(&mut local, &shared.roots);
        intern_well_known_strings(&mut local, &shared.interner, &shared.roots);
        bootstrap_well_known(&mut local, &shared.roots);
        shared.install_tools(&mut local);
        Ok(Self { shared })
    }

    pub fn heap(&self) -> &GlobalHeap {
        &self.shared.heap
    }

    pub fn is_shutdown(&self) -> bool {
        self.shared.shutdown_requested.load(Ordering::Acquire)
    }

    pub fn note_shutdown(&self) {
        self.shared
            .shutdown_requested
            .store(true, Ordering::Release);
    }

    pub fn known(&self) -> &WellKnown {
        self.shared.known.get()
    }

    pub fn interner(&self) -> &StringInterner {
        &self.shared.interner
    }

    pub fn runtimes(&self) -> &RuntimeRegistry {
        &self.shared.runtimes
    }

    pub fn runtime(&self, index: RuntimeIndex) -> RuntimeCall {
        self.shared
            .runtimes
            .get(index)
            .expect("unknown runtime index")
    }

    pub fn register_runtime(&mut self, f: RuntimeCall) -> RuntimeIndex {
        Arc::get_mut(&mut self.shared)
            .expect("cannot register runtimes on a shared VM")
            .runtimes
            .insert(f)
    }

    pub fn visit_roots(&self, visitor: &mut dyn Visitor) {
        self.shared.visit_roots(visitor)
    }

    /// Register a weak slot for `value` (must be an anchored strong heap
    /// pointer); returns its index. The GC clears the slot once the target
    /// dies.
    pub fn track_weak<'x, T: 'x>(&self, value: Tagged<'x, T>) -> usize {
        let weak = value.erase().as_weak().raw();
        let mut slots = self.shared.weak_slots.lock().unwrap();
        slots.push(unsafe { RawCell::from_word(weak.to_bits()) });
        slots.len() - 1
    }

    /// Current contents of a tracked weak slot: the weak-tagged target, or
    /// the cleared sentinel once the target died.
    pub fn weak_value(&self, index: usize) -> Value {
        Value::from_bits(self.shared.weak_slots.lock().unwrap()[index].load())
    }

    pub fn attach(&self) -> Thread {
        let heap = self.shared.heap.new_local(&self.shared.known);
        // Safety: root-slot reads stored straight into rooted fill cells.
        let the_hole = heap.known().the_hole.raw();
        let undefined = heap.known().undefined.raw();
        let state = Arc::new(ContextState {
            handles: HandleData::new(the_hole),
            stack: Stack::new(STACK_SLOTS, the_hole, undefined),
            cache: StackCache::new(the_hole),
            pending_exception: unsafe { Register::from_value(the_hole) },
            has_pending_exception: Cell::new(false),
            termination: Cell::new(None),
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

    /// Run `R`'s setup (native-fn registration + global installation) and
    /// store its rooted state for later `runtime_state` access. Must run
    /// before the VM is shared: `R::setup` registers runtime functions,
    /// which requires exclusive access.
    pub fn add<R: Runtime>(mut self) -> Result<Self, VmError> {
        let mut state = R::State::default();
        R::setup(&mut self, &mut state)?;
        self.shared
            .states
            .lock()
            .unwrap()
            .push((TypeId::of::<R>(), Box::new(state)));
        Ok(self)
    }

    /// The state a runtime stored during [`VM::add`].
    pub fn runtime_state<R: Runtime>(&self) -> &R::State {
        let states = self.shared.states.lock().unwrap();
        let (_, state) = states
            .iter()
            .find(|(id, _)| *id == TypeId::of::<R>())
            .expect("runtime not added to this VM");
        // Safety: entries are append-only and `Box` targets never move, so
        // the reference stays valid for the life of the VM.
        unsafe { &*(state.as_any() as *const dyn Any as *const R::State) }
    }

    pub fn roots(&self) -> &RootHandles {
        &self.shared.roots
    }

    /// Arm the `stress-minor-gc` knob (no-op without the feature).
    pub fn arm_gc_stress(&self) {
        self.shared.heap.arm_gc_stress();
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
