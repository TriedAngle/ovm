use core::{
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
    EdgeVisitable, GcSlot, Global, HANDLE_BLOCK_SIZE, Header, Heap, HeapObject, HeapPtr, Map,
    RawCell, Register, Tagged, Value, Visitor,
};

/// A rooted reference to a `T` that survives relocation by the GC.
/// Weak references cannot be rooted: they live in `MaybeWeakGcSlot`s as
/// `Tagged<MaybeWeak<T>>` words.
///
/// The handle itself is only a location; reading it back as a
/// [`Tagged`] requires a live borrow of the heap
/// ([`Handle::as_tagged`]) so the snapshot cannot outlive the next GC.
pub struct Handle<'scope, T> {
    location: NonNull<Value>,
    _phantom: PhantomData<(&'scope (), T)>,
}

impl<'s, T> Clone for Handle<'s, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'s, T> Copy for Handle<'s, T> {}

impl<'s, T> core::fmt::Debug for Handle<'s, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Safety: raw word read for diagnostics only.
        let value = unsafe { *self.location.as_ptr() };
        f.debug_struct("Handle").field("value", &value).finish()
    }
}

impl<'s, T> Handle<'s, T> {
    /// # Safety
    /// `location` must point at a live GC-visited slot that holds a strong
    /// value and that stays valid for `'s` (i.e. is owned by a handle scope
    /// that outlives `'s`).
    pub unsafe fn from_location(location: NonNull<Value>) -> Self {
        Handle {
            location,
            _phantom: PhantomData,
        }
    }

    /// Re-read the rooted slot under a heap borrow: the returned
    /// snapshot is valid for `'a` because no GC can run while the
    /// borrow lives. This is the only safe `Handle -> Tagged` path.
    pub fn as_tagged<'a>(self, _heap: &'a Heap) -> Tagged<'a, T> {
        // Safety: handle slots only ever hold strong values, and the
        // anchor borrow proves no GC ran since the load.
        unsafe { Tagged::from_value_unchecked(*self.location.as_ptr()) }
    }

    /// The current word without an anchor. It may be moved by a later
    /// collection; only safe to use as an opaque `Value` or re-anchored
    /// (`Value::assume_valid`).
    pub fn raw(self) -> Value {
        unsafe { *self.location.as_ptr() }
    }

    pub fn erase(self) -> Handle<'s, Value> {
        Handle {
            location: self.location,
            _phantom: PhantomData,
        }
    }
}

impl<'s, T: HeapObject> Handle<'s, T> {
    pub fn get(self) -> HeapPtr<T> {
        // Safety: handle slots only ever hold strong values.
        unsafe { Tagged::<T>::from_value_unchecked(self.raw()) }
            .as_ptr()
            .expect("strong local slot must contain strong pointer")
    }
}

pub struct HandleData {
    inner: UnsafeCell<HandleDataImpl>,
}

struct HandleDataImpl {
    // Note: the memory MUST stay stable, this is why its necessary to use additional blocks
    // instead of a single Vec<T>, because we can add/remove blocks without moving any blocks.
    blocks: Vec<Box<[Value]>>,
    next: *mut Value,
    limit: *mut Value,
    level: usize,
    fill: Register,
}

impl HandleDataImpl {
    fn allocate_slot(&mut self) -> *mut Value {
        if self.next == self.limit {
            self.extend();
        }
        let slot = self.next;
        self.next = unsafe { slot.add(1) };
        slot
    }

    fn extend(&mut self) {
        self.extend_sized(HANDLE_BLOCK_SIZE);
    }

    fn extend_sized(&mut self, size: usize) {
        let block = vec![self.fill.raw(); size].into_boxed_slice();
        self.next = block.as_ptr() as *mut Value;
        self.limit = unsafe { self.next.add(block.len()) };
        self.blocks.push(block);
    }

    /// Reserve `n` contiguous slots in the current block, extending first
    /// when the remainder is too small. Oversized requests get a dedicated
    /// block (block memory is boxed and never moves, so any size works).
    fn allocate_block(&mut self, n: usize) -> *mut Value {
        if n > HANDLE_BLOCK_SIZE {
            self.extend_sized(n);
        } else if self.next.addr() + n * core::mem::size_of::<Value>() > self.limit.addr() {
            self.extend();
        }
        let start = self.next;
        self.next = unsafe { start.add(n) };
        start
    }

    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.fill.as_raw());
        for block in &self.blocks {
            let start = block.as_ptr() as *mut Value;
            let end = unsafe { start.add(block.len()) };
            let used_end = if (start..end).contains(&self.next) {
                self.next
            } else {
                end
            };
            let mut slot = start;
            while slot < used_end {
                visitor.visit(unsafe { &*(slot as *const RawCell) });
                slot = unsafe { slot.add(1) };
            }
        }
    }
}

impl HandleData {
    pub fn new(fill: Value) -> Self {
        let mut inner = HandleDataImpl {
            blocks: Vec::new(),
            next: std::ptr::null_mut(),
            limit: std::ptr::null_mut(),
            level: 0,
            fill: unsafe { Register::from_value(fill) },
        };
        inner.extend();
        Self {
            inner: UnsafeCell::new(inner),
        }
    }

    #[allow(clippy::mut_from_ref)]
    fn inner(&self) -> &mut HandleDataImpl {
        unsafe { &mut *self.inner.get() }
    }

    pub fn level(&self) -> usize {
        self.inner().level
    }
}

impl EdgeVisitable for HandleData {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        self.inner().visit_edges(visitor)
    }
}

pub struct HandleScope<'d> {
    data: NonNull<HandleData>,
    prev_next: *mut Value,
    prev_limit: *mut Value,
    prev_block_count: usize,
    _phantom: PhantomData<fn(&'d ()) -> &'d ()>,
}

impl<'d> HandleScope<'d> {
    pub unsafe fn from_raw(data: NonNull<HandleData>) -> Self {
        let inner = unsafe { &*data.as_ptr() }.inner();
        inner.level += 1;
        Self {
            data,
            prev_next: inner.next,
            prev_limit: inner.limit,
            prev_block_count: inner.blocks.len(),
            _phantom: PhantomData,
        }
    }

    pub fn handle<'x, T: 'x>(&self, value: impl Into<Tagged<'x, T>>) -> Handle<'_, T> {
        let value = value.into();
        debug_assert!(
            !value.raw().is_weak_ptr(),
            "weak value cannot be rooted in a handle"
        );
        let slot = unsafe { &*self.data.as_ptr() }.inner().allocate_slot();
        unsafe { *slot = value.raw() };
        unsafe { Handle::from_location(NonNull::new_unchecked(slot)) }
    }

    /// Root a copy of `values` in contiguous scope slots. The values may be
    /// anchored anywhere: rooting only writes words.
    pub fn stage<'x, T>(&self, values: &[Tagged<'x, T>]) -> HandleSlice<'_> {
        let inner = unsafe { &*self.data.as_ptr() }.inner();
        let start = inner.allocate_block(values.len());
        for (i, v) in values.iter().enumerate() {
            debug_assert!(!v.is_weak_ptr(), "weak value staged for a call");
            unsafe { *start.add(i) = v.raw() };
        }
        unsafe { HandleSlice::from_slice(core::slice::from_raw_parts(start, values.len())) }
    }

    pub fn cast<T: HeapObject>(&self, value: Tagged<'_, Value>) -> Option<Handle<'_, T>> {
        let ptr = HeapPtr::decode_strong(value.raw())?;
        // Safety: raw header read for a kind check.
        let map = unsafe { &*(ptr.as_ptr() as *const Header) }.map.raw();
        let kind = unsafe { HeapPtr::<Map>::new(map.raw_addr() as *mut Map).as_ref() }
            .kind()
            .kind();
        if !T::matches_kind(kind) {
            return None;
        }
        Some(self.handle(unsafe { value.cast() }))
    }

    pub fn escapable_scope<'a>(&'a mut self) -> EscapableHandleScope<'a, 'd> {
        let data = unsafe { &*self.data.as_ptr() };
        let escape_slot = data.inner().allocate_slot();
        let inner = data.inner();
        inner.level += 1;
        EscapableHandleScope {
            parent: self,
            escape_slot,
            prev_next: inner.next,
            prev_limit: inner.limit,
            prev_block_count: inner.blocks.len(),
            escaped: Cell::new(false),
        }
    }
}

impl Drop for HandleScope<'_> {
    fn drop(&mut self) {
        let inner = unsafe { &*self.data.as_ptr() }.inner();
        inner.level -= 1;
        inner.next = self.prev_next;
        inner.limit = self.prev_limit;
        inner.blocks.truncate(self.prev_block_count);
    }
}

/// An inner `HandleScope` that can move one handle into its parent scope.
pub struct EscapableHandleScope<'i, 'o> {
    parent: &'i mut HandleScope<'o>,
    escape_slot: *mut Value,
    prev_next: *mut Value,
    prev_limit: *mut Value,
    prev_block_count: usize,
    escaped: Cell<bool>,
}
impl<'d> core::ops::Deref for EscapableHandleScope<'_, 'd> {
    type Target = HandleScope<'d>;

    fn deref(&self) -> &Self::Target {
        self.parent
    }
}

impl<'i, 'o> EscapableHandleScope<'i, 'o> {
    fn data(&self) -> &HandleData {
        unsafe { &*self.parent.data.as_ptr() }
    }

    pub fn escape<T>(&self, handle: Handle<'i, T>) -> Handle<'o, T> {
        debug_assert!(!self.escaped.get(), "only one handle can escape a scope");
        self.escaped.set(true);
        // Safety: moving one rooted word into another rooted slot.
        unsafe { *self.escape_slot = handle.raw() };
        unsafe { Handle::from_location(NonNull::new_unchecked(self.escape_slot)) }
    }
}

impl Drop for EscapableHandleScope<'_, '_> {
    fn drop(&mut self) {
        let inner = self.data().inner();
        inner.level -= 1;
        inner.next = self.prev_next;
        inner.limit = self.prev_limit;
        inner.blocks.truncate(self.prev_block_count);
    }
}

pub struct RootHandles {
    slots: Box<[GcSlot]>,
    next: AtomicUsize,
}

unsafe impl Send for RootHandles {}
unsafe impl Sync for RootHandles {}

impl RootHandles {
    pub unsafe fn new(capacity: usize, fill: Value) -> Self {
        Self {
            slots: (0..capacity)
                .map(|_| unsafe { GcSlot::from_value(fill) })
                .collect(),
            next: AtomicUsize::new(0),
        }
    }

    pub fn create_handle<'x, T: 'x>(&self, value: impl Into<Tagged<'x, T>>) -> Global<T> {
        let value = value.into();
        let i = self.next.fetch_add(1, Ordering::Relaxed);
        assert!(i < self.slots.len(), "root handle table exhausted");
        let slot = GcSlot::raw_get(&self.slots[i]);
        unsafe { *slot = value.raw() };
        unsafe { Handle::from_location(NonNull::new_unchecked(slot)) }
    }
}

impl EdgeVisitable for RootHandles {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        for slot in &self.slots[..self.next.load(Ordering::Relaxed)] {
            visitor.visit(slot.as_raw());
        }
    }
}

pub trait HandleSet {
    fn create_handle<'x, T: 'x>(&self, value: impl Into<Tagged<'x, T>>) -> Handle<'_, T>;
}

impl HandleSet for HandleScope<'_> {
    fn create_handle<'x, T: 'x>(&self, value: impl Into<Tagged<'x, T>>) -> Handle<'_, T> {
        self.handle(value)
    }
}

impl HandleSet for RootHandles {
    fn create_handle<'x, T: 'x>(&self, value: impl Into<Tagged<'x, T>>) -> Handle<'_, T> {
        // Slots in the root table are stable and never reclaimed, so the
        // returned handle is valid for the borrow of `self` (in practice:
        // pseudo-static, see `Global<T>`).
        RootHandles::create_handle(self, value)
    }
}

/// A borrowed view of rooted argument memory: a slice of GC-visited
/// slots (handle scope blocks or the register file). The words are
/// updated in place by the GC, so a [`Handle`] can be taken to any
/// element for free — the slot is already a root, no scope allocation is
/// needed. Anchored reads go through [`Handle::as_tagged`].
#[derive(Copy, Clone)]
pub struct HandleSlice<'a> {
    raw: &'a [Value],
}

impl<'a> HandleSlice<'a> {
    /// The empty argument list.
    pub const EMPTY: HandleSlice<'static> = HandleSlice { raw: &[] };

    /// # Safety
    /// The slice must point at memory the GC visits for as long as it is
    /// alive ([`HandleScope::stage`] slots or the register file), and the
    /// words must be strong (non-weak) values.
    pub unsafe fn from_slice(raw: &'a [Value]) -> Self {
        Self { raw }
    }

    /// A free handle to the element at `index`: the slot is already
    /// rooted, so this allocates nothing and the handle survives
    /// collection for as long as the slice lives.
    pub fn get(&self, index: usize) -> Option<Handle<'a, Value>> {
        let raw: &'a [Value] = self.raw;
        let word = raw.get(index)?;
        // Safety: `from_slice` guarantees a GC-visited, strong slot.
        Some(unsafe { Handle::from_location(NonNull::from(word)) })
    }

    /// Free handles over every element (see [`Self::get`]).
    pub fn iter(&self) -> impl Iterator<Item = Handle<'a, Value>> + 'a {
        let raw: &'a [Value] = self.raw;
        raw.iter().map(|word| {
            // Safety: see `get`.
            unsafe { Handle::from_location(NonNull::from(word)) }
        })
    }

    /// Raw words, for storage copies into fresh objects (no GC can run
    /// mid-`init`).
    pub fn raw(&self) -> &'a [Value] {
        self.raw
    }

    pub fn len(&self) -> usize {
        self.raw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }
}
