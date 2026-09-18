//! Arena and pool storage backing the shared program IR.
//!
//! The IR is written once by a frontend compiler and read once by the VM's
//! materializer, so it is stored in bulk arrays rather than a graph of
//! individually heap-allocated nodes: [`Arena`] hands out typed index
//! handles, [`Pool`] stores homogeneous runs that [`Span`]s point into.

/// A half-open run of `len` items starting at `start` inside a [`Pool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: u32,
    pub len: u32,
}

impl Span {
    pub const EMPTY: Span = Span { start: 0, len: 0 };

    pub const fn new(start: u32, len: u32) -> Self {
        Self { start, len }
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Index range suitable for slicing a backing slice.
    pub const fn range(self) -> core::ops::Range<usize> {
        self.start as usize..(self.start + self.len) as usize
    }
}

/// Typed index arena: stable `u32` handles, dense storage, no per-item
/// allocation. Handles are not generation-checked (the IR is append-only
/// and never mutated after compilation).
#[derive(Debug)]
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Arena<T> {
    pub const fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
        }
    }

    pub fn insert(&mut self, item: T) -> u32 {
        let index = self.items.len() as u32;
        self.items.push(item);
        index
    }

    pub fn get(&self, index: u32) -> &T {
        &self.items[index as usize]
    }

    pub fn get_mut(&mut self, index: u32) -> &mut T {
        &mut self.items[index as usize]
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.items.iter_mut()
    }
}

/// Homogeneous run storage. Appends return a [`Span`]; reads borrow the
/// backing slice, so no allocation happens on the read path.
#[derive(Debug)]
pub struct Pool<T> {
    items: Vec<T>,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Pool<T> {
    pub const fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
        }
    }

    /// Append a run of items and return the span indexing it.
    pub fn alloc(&mut self, items: &[T]) -> Span
    where
        T: Clone,
    {
        let start = self.items.len() as u32;
        self.items.extend_from_slice(items);
        Span::new(start, items.len() as u32)
    }

    pub fn alloc_one(&mut self, item: T) -> u32 {
        let index = self.items.len() as u32;
        self.items.push(item);
        index
    }

    pub fn get(&self, span: Span) -> &[T] {
        &self.items[span.range()]
    }

    pub fn get_mut(&mut self, span: Span) -> &mut [T] {
        &mut self.items[span.range()]
    }

    pub fn at(&self, index: u32) -> &T {
        &self.items[index as usize]
    }

    pub fn as_slice(&self) -> &[T] {
        &self.items
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
