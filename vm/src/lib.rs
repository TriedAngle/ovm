use std::sync::{Arc, Mutex, Weak};

use core::cell::Cell;
use core::ptr::NonNull;

pub mod bootstrap;
pub mod builtins;
pub mod cache;
pub mod compare;
pub mod convert;
pub mod error;
pub mod errors;
pub mod handle;
pub mod heap;
pub mod interner;
pub mod interpreter;
pub mod lookup;
pub mod materialize;
pub mod objects;
pub mod runtime;
pub mod stack;
pub mod transition;
pub mod value;

pub use objects::proxy;
pub use objects::string;

pub use bootstrap::{
    KnownCell, WellKnown, WellKnownStrings, bootstrap_basics, bootstrap_well_known,
    intern_well_known_strings,
};
pub use cache::StackCache;
pub use compare::Compare;
pub use convert::Convert;
pub use error::VmError;
pub use errors::Errors;
pub use handle::{
    EscapableHandleScope, Handle, HandleData, HandleScope, HandleSet, HandleSlice, RootHandles,
};
pub use heap::{
    AllocToken, EdgeVisitable, GcSlot, GlobalHeap, Heap, MaybeWeakGcSlot, OptionGcSlot, Register,
    WordType,
};
pub use interner::StringInterner;
pub use lookup::{
    Key, LoadOutcome, Lookup, has_property, home_proto, load_outcome_on, lookup_in_parents,
    ordinary_own_descriptor, private_find, super_constructor, super_lookup,
    super_lookup_from_proto,
};
pub use objects::{
    AccessorPair, CallTarget, CallableInfoInit, CallableInfoObject, Context, ContextInit,
    DenseString, Encoding, FixedArray, FixedByteArray, Float, FunctionKind, HandlerEntry,
    HandlerEntryInit, HandlerTable, HandlerTableInit, Header, HeapObject, Map, MapInit, MapKind,
    Object, ObjectInit, ObjectKind, ObjectSlotsInit, ProxyInit, ProxyObject, ScopeInfo,
    ScopeInfoInit, SlotDescriptor, SlotFlags, SlotName, StringData, Symbol, WeakFixedArray,
    WeakFixedArrayInit, decode_wtf8, object_kind, object_layout, string_content_hash, visit_object,
};
pub use runtime::{Coercion, Hint, RuntimeCall, RuntimeContext, RuntimeIndex, RuntimeRegistry};
pub use stack::{FrameMeta, STACK_SLOTS, Stack};
pub use transition::{
    Change, PartialDescriptor, PropertyDescriptor, StoreOutcome, StoreSemantics, Transition,
    TransitionGuard, TransitionLock, is_compatible_property_descriptor, super_store_lookup,
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
    /// Weak slots the GC clears when their targets die; used for tests and
    /// the seed of a weak-registry feature.
    weak_slots: Mutex<Vec<RawCell>>,
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

    pub fn set_pending_exception(&self, value: impl Into<Value>) {
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

    /// Like [`Self::take_pending_exception`], but anchored to the heap
    /// borrow: the register is GC-visited, so the read is current.
    pub fn take_pending_exception_tagged<'a>(&self, heap: &'a Heap) -> Option<Tagged<'a, Value>> {
        if self.has_pending_exception.get() {
            self.has_pending_exception.set(false);
            Some(self.pending_exception.read(heap))
        } else {
            None
        }
    }

    pub fn has_pending_exception(&self) -> bool {
        self.has_pending_exception.get()
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
        // Safety: caller-owned argument words staged into rooted slots
        // before anything can allocate.
        self.state.handle_scope(|scope| {
            let args = scope.stage(
                &args
                    .iter()
                    .map(|v| unsafe { Tagged::<Value>::from_value_unchecked(*v) })
                    .collect::<Vec<_>>(),
            );
            interpreter::execute(&self.vm, &mut self.heap, &self.state, callable, args, None)
                .map(|v| v.raw())
        })
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

    pub fn run_script(&mut self, src: &str) -> Result<Value, ScriptError> {
        self.run_source(src, js_compiler::compile_js, ir::SourceMode::Script)
    }

    pub fn run_script_repl(&mut self, src: &str) -> Result<Value, ScriptError> {
        self.run_source(src, js_compiler::compile_js, ir::SourceMode::Repl)
    }

    pub fn run_source(
        &mut self,
        src: &str,
        compile: ir::CompileFn,
        mode: ir::SourceMode,
    ) -> Result<Value, ScriptError> {
        let program = compile(src, mode).map_err(ScriptError::from_frontend)?;
        self.handle_scope(|thread, scope| {
            let closure = materialize::materialize_script(thread, &scope, &program)
                .map_err(ScriptError::Vm)?;
            thread.execute(closure, &[]).map_err(ScriptError::Vm)
        })
    }
}

/// Failure of any stage of [`Thread::run_script`].
#[derive(Debug)]
pub enum ScriptError {
    Parse(ir::FrontendError),
    Compile(ir::FrontendError),
    Vm(VmError),
}

impl ScriptError {
    fn from_frontend(err: ir::FrontendError) -> Self {
        match err.kind {
            ir::FrontendErrorKind::Syntax => ScriptError::Parse(err),
            ir::FrontendErrorKind::Compile => ScriptError::Compile(err),
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
    pub fn new<B: HeapBackend>(config: B::Config) -> Result<Self, AllocError> {
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
            weak_slots: Mutex::new(Vec::new()),
        });
        shared.heap.set_host(shared.gc_host());
        let mut local = shared.heap.new_local(&shared.known);
        bootstrap_basics(&mut local, &shared.roots);
        intern_well_known_strings(&mut local, &shared.interner, &shared.roots);
        bootstrap_well_known(&mut local, &shared.roots);
        Ok(Self { shared })
    }

    pub fn heap(&self) -> &GlobalHeap {
        &self.shared.heap
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

    /// Register a weak slot for `value` (must be a strong heap pointer);
    /// returns its index. The GC clears the slot once the target dies.
    pub fn track_weak(&self, value: Value) -> usize {
        let weak = Value::from_bits(value.to_bits() | WEAK_BIT);
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
        let the_hole = unsafe { heap.known().the_hole.read_unchecked() };
        let undefined = unsafe { heap.known().undefined.read_unchecked() };
        let state = Arc::new(ContextState {
            handles: HandleData::new(the_hole),
            stack: Stack::new(STACK_SLOTS, the_hole, undefined),
            cache: StackCache::new(the_hole),
            pending_exception: unsafe { Register::from_value(the_hole) },
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
        let idx = builtins::register_builtin_runtimes(&mut vm);
        builtins::install_builtins(&mut vm, &idx)?;
        // Promote the bootstrap singletons (the hole, undefined, the
        // initial maps and prototype objects, ...) out of the young
        // generation before any code runs: the VM caches their addresses
        // (`known().*.value()` snapshots in Rust locals across
        // allocations, handle/stack fill cells), which is only sound
        // while they never move again. One minor collection evacuates
        // every reachable young object — all of them included — into
        // the old generation.
        {
            let mut thread = vm.attach();
            thread.heap().collect_minor();
        }
        #[cfg(feature = "stress-minor-gc")]
        vm.shared.heap.arm_gc_stress();
        Ok(vm)
    }
}
