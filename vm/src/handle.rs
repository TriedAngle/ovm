use core::{
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
    EdgeVisitable, GcSlot, Global, HANDLE_BLOCK_SIZE, HeapObject, HeapPtr, HeapRef, NoGc, RawCell,
    Tagged, Value, Visitor,
};

/// A rooted reference to a `T` that survives relocation by the GC.
/// Weak references cannot be rooted: they live in `WeakGcCell`s as
/// `Tagged<MaybeWeak<T>>` words.
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
        f.debug_struct("Handle")
            .field("value", &self.value())
            .finish()
    }
}

impl<'s, T> Handle<'s, T> {
    pub fn from_location(location: NonNull<Value>) -> Self {
        Handle {
            location,
            _phantom: PhantomData,
        }
    }

    pub fn value(self) -> Value {
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
        self.as_tagged()
            .as_ptr()
            .expect("strong local slot must contain strong pointer")
    }

    pub fn as_tagged(self) -> Tagged<T> {
        self.into()
    }

    pub fn heap_ref<'a>(self, _guard: &'a NoGc<'a>) -> HeapRef<'a, T> {
        unsafe { HeapRef::from_ptr(self.get()) }
    }
}

impl<'s, T> From<Handle<'s, T>> for Tagged<T> {
    fn from(h: Handle<'s, T>) -> Self {
        // SAFETY: handle slots only ever hold strong values.
        unsafe { Self::from_value_unchecked(h.value()) }
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
    fill: Value,
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
        let block = vec![self.fill; HANDLE_BLOCK_SIZE].into_boxed_slice();
        self.next = block.as_ptr() as *mut Value;
        self.limit = unsafe { self.next.add(block.len()) };
        self.blocks.push(block);
    }

    fn visit_edges(&self, visitor: &mut impl Visitor) {
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
            fill,
        };
        inner.extend();
        Self {
            inner: UnsafeCell::new(inner),
        }
    }

    fn inner(&self) -> &mut HandleDataImpl {
        unsafe { &mut *self.inner.get() }
    }

    pub fn level(&self) -> usize {
        self.inner().level
    }
}

impl EdgeVisitable for HandleData {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
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

    pub fn handle<T>(&self, value: Tagged<T>) -> Handle<'_, T> {
        debug_assert!(
            !value.erase().is_weak_ptr(),
            "weak value cannot be rooted in a handle"
        );
        let slot = unsafe { &*self.data.as_ptr() }.inner().allocate_slot();
        unsafe { *slot = value.erase() };
        Handle::from_location(unsafe { NonNull::new_unchecked(slot) })
    }

    // TODO: get rid of this
    pub unsafe fn handle_value<T>(&self, value: Value) -> Handle<'_, T> {
        self.handle(unsafe { Tagged::from_value_unchecked(value) })
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
        unsafe { *self.escape_slot = handle.value() };
        Handle::from_location(unsafe { NonNull::new_unchecked(self.escape_slot) })
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

    pub fn create_handle<T>(&self, value: Tagged<T>) -> Global<T> {
        let i = self.next.fetch_add(1, Ordering::Relaxed);
        assert!(i < self.slots.len(), "root handle table exhausted");
        let slot = GcSlot::raw_get(&self.slots[i]);
        unsafe { *slot = value.erase() };
        Handle::from_location(unsafe { NonNull::new_unchecked(slot) })
    }
}

impl EdgeVisitable for RootHandles {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        for slot in &self.slots[..self.next.load(Ordering::Relaxed)] {
            visitor.visit(slot.as_raw());
        }
    }
}

pub trait HandleSet {
    fn create_handle<T>(&self, value: Tagged<T>) -> Handle<'_, T>;
}

impl HandleSet for HandleScope<'_> {
    fn create_handle<T>(&self, value: Tagged<T>) -> Handle<'_, T> {
        self.handle(value)
    }
}

impl HandleSet for RootHandles {
    fn create_handle<T>(&self, value: Tagged<T>) -> Handle<'_, T> {
        // Slots in the root table are stable and never reclaimed, so the
        // returned handle is valid for the borrow of `self` (in practice:
        // pseudo-static, see `Global<T>`).
        RootHandles::create_handle(self, value)
    }
}

#[derive(Copy, Clone)]
pub struct GcSlice<'a> {
    slice: &'a [Value],
}

impl<'a> GcSlice<'a> {
    pub unsafe fn from_slice(slice: &'a [Value]) -> Self {
        Self { slice }
    }

    pub fn as_slice(&self) -> &'a [Value] {
        self.slice
    }

    pub fn len(&self) -> usize {
        self.slice.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<Value> {
        self.slice.get(index).copied()
    }

    pub fn iter(&self) -> core::slice::Iter<'a, Value> {
        self.slice.iter()
    }
}

impl core::ops::Index<usize> for GcSlice<'_> {
    type Output = Value;

    fn index(&self, index: usize) -> &Value {
        &self.slice[index]
    }
}
