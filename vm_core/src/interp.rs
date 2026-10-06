//! The per-execution interpreter context shared by the interpreter
//! implementations. Hot accessors are `#[inline(always)]` so they
//! keep inlining into interpreter crates across the crate boundary;
//! raise paths stay cold and out of line.

use core::cell::Cell;
use core::marker::PhantomData;

use crate::errors::Errors;
use crate::heap::{Heap, Register};
use crate::stack::{CODE_OFFSET, CONSTANTS_OFFSET, FEEDBACK_OFFSET, FrameMeta, Stack};
use crate::value::{Tagged, Value};
use crate::{ContextState, FeedbackVector, FixedArray, FixedByteArray, VM, VmError};

/// Loop back-edge ticks between safepoint polls.
pub const SAFEPOINT_INTERVAL: u32 = 1 << 12;

pub struct Ctx<'a> {
    vm: *const VM,
    heap: *mut Heap,
    state: *const ContextState,
    /// Anchor of the frame `execute` entered: a return at this anchor ends
    /// the execution instead of unwinding into a caller.
    base_anchor: usize,
    safepoints: Cell<u32>,
    /// Machine-stack limit for the call paths (one Rust frame per JS
    /// call): a call below this raises StackOverflow. Recomputed at
    /// each `enter` from the current SP — nested enters only deepen it.
    stack_limit: usize,
    _heap: PhantomData<&'a mut Heap>,
}

impl<'a> Ctx<'a> {
    /// Build a context for a fresh `execute` entry.
    ///
    /// Safety: `vm`, `heap` and `state` must stay valid and unaliased
    /// (mutable heap) for `'a`; `base_anchor` must be the frame the
    /// entered execution returns at.
    #[inline(always)]
    pub unsafe fn new(
        vm: &'a VM,
        heap: &'a mut Heap,
        state: &'a ContextState,
        base_anchor: usize,
        stack_limit: usize,
    ) -> Self {
        Ctx {
            vm: vm as *const VM,
            heap: heap as *mut Heap,
            state: state as *const ContextState,
            base_anchor,
            safepoints: Cell::new(SAFEPOINT_INTERVAL),
            stack_limit,
            _heap: PhantomData,
        }
    }

    /// A nested context for a callee frame sharing this execution's
    /// borrows and machine-stack limit (a fresh safepoint cell each).
    ///
    /// Safety: same borrow validity as `self`; `base_anchor` must be the
    /// callee's frame.
    #[inline(always)]
    pub unsafe fn child(&self, base_anchor: usize) -> Ctx<'a> {
        Ctx {
            vm: self.vm,
            heap: self.heap,
            state: self.state,
            base_anchor,
            safepoints: Cell::new(SAFEPOINT_INTERVAL),
            stack_limit: self.stack_limit,
            _heap: PhantomData,
        }
    }

    #[inline(always)]
    pub fn vm(&self) -> &'a VM {
        unsafe { &*self.vm }
    }

    #[inline(always)]
    pub fn heap(&self) -> &'a Heap {
        unsafe { &*self.heap }
    }

    #[inline(always)]
    pub unsafe fn heap_mut(&self) -> &'a mut Heap {
        unsafe { &mut *self.heap }
    }

    #[inline(always)]
    pub fn state(&self) -> &'a ContextState {
        unsafe { &*self.state }
    }

    #[inline(always)]
    pub fn stack(&self) -> &'a Stack {
        self.state().stack()
    }

    #[inline(always)]
    pub fn base_anchor(&self) -> usize {
        self.base_anchor
    }

    /// The current frame's anchor (the frame-pointer analogue). Frame
    /// switches are a single store: the frame header carries every
    /// other dispatch fact.
    #[inline(always)]
    pub fn frame_base(&self) -> usize {
        self.state().frame_base()
    }

    /// Switch the current frame (after a push, pop, or unwind).
    #[inline(always)]
    pub fn set_frame_base(&self, base: usize) {
        self.state().set_frame_base(base);
    }

    /// The caller descriptor for pushing a frame from the current one:
    /// `pc` is the caller's resume point, `handler_pc` its exception
    /// lookup pc.
    #[inline(always)]
    pub fn caller_meta(&self, pc: usize, handler_pc: usize) -> FrameMeta {
        let base = self.frame_base();
        FrameMeta {
            base,
            pc,
            register_count: self.stack().regcount(base),
            handler_pc,
        }
    }

    #[inline(always)]
    pub fn code_ptr(&self) -> *const u8 {
        // Safety: only `init_frame_header` writes this slot, always the
        // frame's bytecode: the kind re-check is redundant.
        let arr = unsafe {
            self.stack()
                .header_slot(self.frame_base(), CODE_OFFSET)
                .get(self.heap())
                .cast::<FixedByteArray>()
        };
        arr.as_ref().as_ptr()
    }

    #[inline(always)]
    pub fn constants_ref<'h>(&self, heap: &'h Heap) -> Tagged<'h, FixedArray> {
        debug_assert!(
            self.state().is_frame_active(),
            "constants read without a frame"
        );
        // Safety: only `init_frame_header` writes this slot, always the
        // constant pool.
        unsafe {
            self.stack()
                .header_slot(self.frame_base(), CONSTANTS_OFFSET)
                .get(heap)
                .cast()
        }
    }

    /// The current frame's feedback vector, or `None` for functions
    /// without feedback slots.
    #[inline(always)]
    pub fn feedback_ref<'h>(&self, heap: &'h Heap) -> Option<Tagged<'h, FeedbackVector>> {
        // Safety: only `init_frame_header` writes this slot: it is always
        // a `FeedbackVector` or the hole.
        let word = self
            .stack()
            .header_slot(self.frame_base(), FEEDBACK_OFFSET)
            .get(heap);
        (word.raw() != heap.known().the_hole.raw()).then(|| unsafe { word.cast() })
    }

    /// The accumulator cell.
    #[inline(always)]
    pub fn acc_slot(&self) -> &Register {
        self.state().acc_slot()
    }

    #[inline(always)]
    pub fn regs_ptr(&self) -> *mut Register {
        unsafe { self.stack().slots_ptr().add(self.frame_base()) }
    }

    #[inline(always)]
    pub fn exception_word(&self) -> Tagged<'a, Value> {
        let heap = self.heap();
        heap.known().exception.as_tagged(heap).erase()
    }

    #[inline(always)]
    pub fn is_throw(&self, v: Tagged<'_, Value>) -> bool {
        v == self.exception_word()
    }

    #[inline(always)]
    pub fn undefined_word(&self) -> Tagged<'a, Value> {
        let heap = self.heap();
        heap.known().undefined.as_tagged(heap).erase()
    }

    /// The machine-stack probe for the call paths: `true` when the
    /// current stack pointer has fallen below this execution's limit.
    #[inline(always)]
    pub fn stack_overflowed(&self) -> bool {
        let probe = 0u8;
        (&probe as *const u8 as usize) < self.stack_limit
    }

    /// One loop back-edge: `true` every `SAFEPOINT_INTERVAL` ticks or when
    /// a collection has been requested since the last reset.
    #[inline(always)]
    pub fn safepoint_tick(&self) -> bool {
        let n = self.safepoints.get();
        if n == 0 {
            self.safepoints.set(SAFEPOINT_INTERVAL);
            true
        } else {
            self.safepoints.set(n - 1);
            false
        }
    }

    /// Materialize a VM error as the pending exception and return `Ok`
    /// with the exception sentinel (the fold for `Result`-returning
    /// helpers).
    #[cold]
    #[inline(never)]
    pub unsafe fn raise(&self, err: VmError) -> Result<Tagged<'a, Value>, VmError> {
        unsafe { Ok(self.raise_tag(err)) }
    }

    /// The single-channel raise for Tagged-returning interpreter fns:
    /// materialize `err`, set the pending exception, return the sentinel
    /// word.
    #[cold]
    #[inline(never)]
    pub unsafe fn raise_tag(&self, err: VmError) -> Tagged<'a, Value> {
        unsafe {
            let heap = self.heap_mut();
            let state = self.state();
            let ex = Errors::from_vm_error(self.vm(), heap, state, err)
                .expect("error materialization must not fail");
            state.set_pending_exception(ex);
            self.exception_word()
        }
    }

    /// Invoke runtime callee `rt` over a normalized argument window.
    ///
    /// `args` may be a window inside the caller's frame (already rooted)
    /// or staged above `top` ([`Stack::stage_scattered`]): the latter is
    /// covered by bumping `top` over it for the call's duration, so the
    /// GC sees the words and a re-entrant execution pushes above them.
    #[inline(always)]
    pub fn call_runtime(&self, rt: crate::RuntimeIndex, args: crate::Args) -> Tagged<'a, Value> {
        let f = self.vm().runtime(rt);
        let stack = self.stack();
        let saved = stack.top();
        let rooted = args.src + args.count;
        let staged = rooted > saved;
        if staged {
            stack.set_top(rooted);
        }
        // Safety: the heap parts carry Ctx's original borrows.
        let nctx =
            crate::RuntimeContext::new(self.vm(), unsafe { self.heap_mut() }, self.state());
        let v = f(nctx, stack.slice(args));
        if staged {
            stack.set_top(saved);
        }
        v
    }
}

/// The exception-dispatch result: a handler was found (its entry pc
/// rides in `Caught` along with the exception, the current frame is the
/// handler's) or the exception escaped this execution.
pub enum Unwind<'a> {
    Caught { pc: usize, ex: Tagged<'a, Value> },
    Escaped,
}

/// Search the handler tables from the faulting frame outward, popping
/// frames until a catch handler covers `fault_pc` or the execution's
/// base anchor escapes. On `Caught` the pending exception is consumed
/// and the current frame is the handler's.
#[cold]
#[inline(never)]
pub unsafe fn unwind<'a>(ctx: &Ctx<'a>, fault_pc: usize) -> Unwind<'a> {
    let mut pc = fault_pc;
    loop {
        let base = ctx.frame_base();
        let handled = if ctx.state().termination().is_some() {
            // A termination is not an exception: no handler (catch or
            // finally in any frame) may observe or intercept it.
            None
        } else {
            let callable = ctx.stack().callable_slot(base).get(ctx.heap());
            callable
                .as_heap_object()
                .and_then(|obj| obj.as_ref().callable_info(ctx.heap()))
                .and_then(|info| info.handlers.get(ctx.heap()))
                .and_then(|handlers| handlers.as_ref().lookup(pc))
        };
        if let Some(handler_pc) = handled {
            let ex = ctx
                .state()
                .take_pending_exception_tagged(ctx.heap())
                .expect("pending exception must be set while unwinding");
            return Unwind::Caught { pc: handler_pc, ex };
        }
        if base == ctx.base_anchor() {
            return Unwind::Escaped;
        }
        let caller = ctx.stack().pop_frame(base);
        ctx.set_frame_base(caller.base);
        pc = caller.handler_pc;
    }
}
