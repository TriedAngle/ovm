use core::{
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
    EdgeVisitable, GcSlot, Global, HANDLE_BLOCK_SIZE, HeapObject, HeapPtr, HeapRef, NoGc,
    PointerStrength, RawCell, Strong, Tagged, Value, Visitor, Weak,
};

pub struct Handle<'scope, T, R: PointerStrength = Strong> {
    location: NonNull<Value>,
    _phantom: PhantomData<(&'scope (), T, R)>,
}

impl<'s, T, R: PointerStrength> Clone for Handle<'s, T, R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'s, T, R: PointerStrength> Copy for Handle<'s, T, R> {}

impl<'s, T, R: PointerStrength> core::fmt::Debug for Handle<'s, T, R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Handle")
            .field("kind", &core::any::type_name::<R>())
            .field("value", &self.value())
            .finish()
    }
}

impl<'s, T, R: PointerStrength> Handle<'s, T, R> {
    pub fn from_location(location: NonNull<Value>) -> Self {
        Handle {
            location,
            _phantom: PhantomData,
        }
    }

    pub fn value(self) -> Value {
        unsafe { *self.location.as_ptr() }
    }

    pub fn erase(self) -> Handle<'s, Value, R> {
        Handle {
            location: self.location,
            _phantom: PhantomData,
        }
    }
}

impl<'s, T: HeapObject> Handle<'s, T, Strong> {
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

impl<'s, T: HeapObject> Handle<'s, T, Weak> {}

impl<'s, T, R: PointerStrength> From<Handle<'s, T, R>> for Tagged<T> {
    fn from(h: Handle<'s, T, R>) -> Self {
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

    pub fn create_handle<T>(&self, value: Tagged<T>) -> Option<Handle<'_, T>> {
        if value.is_weak_ptr() {
            return None;
        }
        Some(unsafe { self.create_handle_unchecked(value) })
    }

    pub unsafe fn create_handle_unchecked<T>(&self, value: Tagged<T>) -> Handle<'_, T> {
        let slot = unsafe { &*self.data.as_ptr() }.inner().allocate_slot();
        unsafe { *slot = value.erase() };
        Handle::from_location(unsafe { NonNull::new_unchecked(slot) })
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
        self.create_handle(value)
            .expect("weak value cannot be rooted in a handle")
    }
}
