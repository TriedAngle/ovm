use core::alloc::Layout;

use crate::{
    EdgeVisitable, GcSlot, Handle, HandleScope, Heap, HeapObject, ObjectKind, Smi, Tagged, Value,
    Visitor,
};

/// The sparse-array elements backing store: an open-addressing hash
/// table keyed by array index. FixedArray-shaped, with a three-word
/// header and `[key | value | details]` entries:
///
/// - `key`: a Smi index; `undefined` = never used (probes stop),
///   `the_hole` = deleted (probes continue)
/// - `value`: the element, or the `AccessorPair` for accessor entries
/// - `details`: Smi bit set — kind + attributes (below)
///
/// Capacity is always a power of two (>= 4) with 50% slack; probing is
/// triangular: `first = hash & (cap-1)`, `next = (e + n) & (cap-1)`.
#[repr(C)]
pub struct NumberDictionary {
    pub header: crate::Header,
    pub capacity: GcSlot<Smi>,
    pub nof: GcSlot<Smi>,
    pub nod: GcSlot<Smi>,
    /// bit 0: `requires_slow_elements` (sticky; never densify back).
    /// The remaining bits are reserved.
    pub flags: GcSlot<Smi>,
    pub entries: [GcSlot<Value>; 0],
}

/// key | value | details
pub const ENTRY_WORDS: usize = 3;

pub const DETAILS_ACCESSOR: i64 = 1;
pub const DETAILS_WRITABLE: i64 = 1 << 1;
pub const DETAILS_ENUMERABLE: i64 = 1 << 2;
pub const DETAILS_CONFIGURABLE: i64 = 1 << 3;
/// plain data element: writable, enumerable, configurable
pub const DETAILS_DATA: i64 = DETAILS_WRITABLE | DETAILS_ENUMERABLE | DETAILS_CONFIGURABLE;

const REQUIRES_SLOW_LIMIT: usize = (1 << 29) - 1;

pub const MIN_CAPACITY: usize = 4;

pub struct NumberDictionaryInit {
    pub capacity: usize,
}

impl NumberDictionary {
    pub const ENTRY_SIZE: usize = ENTRY_WORDS;

    pub fn layout_for(capacity: usize) -> Layout {
        let entries =
            Layout::array::<GcSlot<Value>>(capacity * ENTRY_WORDS).expect("entries layout");
        Layout::new::<Self>()
            .extend(entries)
            .expect("dictionary layout")
            .0
    }

    pub fn capacity(&self) -> usize {
        self.capacity.to_smi_unchecked().value() as usize
    }

    pub fn nof(&self) -> usize {
        self.nof.to_smi_unchecked().value() as usize
    }

    pub fn nod(&self) -> usize {
        self.nod.to_smi_unchecked().value() as usize
    }

    pub fn requires_slow_elements(&self) -> bool {
        self.flags.to_smi_unchecked().value() & 1 != 0
    }

    /// Force the sticky never-densify bit (accessor elements).
    pub fn require_slow(&self, heap: &Heap) {
        if !self.requires_slow_elements() {
            self.set_requires_slow_elements(heap);
        }
    }

    fn set_requires_slow_elements(&self, heap: &Heap) {
        let host = self.tagged(heap);
        let v = self.flags.to_smi_unchecked().value() | 1;
        self.flags.set(heap, host, Smi::new(v));
    }

    /// Update the flags after inserting `index`.
    pub fn note_index(&self, heap: &Heap, index: usize) {
        if index > REQUIRES_SLOW_LIMIT && !self.requires_slow_elements() {
            self.set_requires_slow_elements(heap);
        }
    }

    fn entry_ptr(&self, entry: usize) -> *mut GcSlot<Value> {
        debug_assert!(entry < self.capacity());
        unsafe { self.entries.as_ptr().add(entry * ENTRY_WORDS) as *mut GcSlot<Value> }
    }

    fn key_slot(&self, entry: usize) -> &GcSlot<Value> {
        unsafe { &*self.entry_ptr(entry) }
    }

    fn value_slot(&self, entry: usize) -> &GcSlot<Value> {
        unsafe { &*self.entry_ptr(entry).add(1) }
    }

    fn details_slot(&self, entry: usize) -> &GcSlot<Value> {
        unsafe { &*self.entry_ptr(entry).add(2) }
    }

    fn key_word<'a>(&self, heap: &'a Heap, entry: usize) -> Tagged<'a, Value> {
        self.key_slot(entry).get(heap)
    }

    pub fn value_at<'a>(&self, heap: &'a Heap, entry: usize) -> Tagged<'a, Value> {
        self.value_slot(entry).get(heap)
    }

    pub fn details_at(&self, heap: &Heap, entry: usize) -> i64 {
        self.details_slot(entry)
            .get(heap)
            .to_i64()
            .unwrap_or(DETAILS_DATA)
    }

    pub fn is_accessor_at(&self, heap: &Heap, entry: usize) -> bool {
        self.details_at(heap, entry) & DETAILS_ACCESSOR != 0
    }

    /// The live key of an entry, if any.
    pub fn key_at(&self, heap: &Heap, entry: usize) -> Option<usize> {
        let key = self.key_word(heap, entry);
        if key.is_strong_ptr() {
            // undefined (never used) or the_hole (deleted) sentinels
            return None;
        }
        key.to_i64().map(|v| v as usize)
    }

    /// How a stored entry behaves for a plain element access.
    pub fn classify(&self, heap: &Heap, index: usize) -> EntryClass {
        match self.find(heap, index) {
            None => EntryClass::Absent,
            Some(entry) => {
                let details = self.details_at(heap, entry);
                if details & DETAILS_ACCESSOR != 0 {
                    EntryClass::Accessor
                } else if details & DETAILS_WRITABLE != 0 {
                    EntryClass::Data
                } else {
                    EntryClass::ReadOnly
                }
            }
        }
    }

    fn hash(index: usize) -> u32 {
        let h = (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (h >> 32) as u32
    }

    /// Probe for `index`; `None` when absent. The table is never full,
    /// so the walk always terminates at a never-used slot.
    pub fn find(&self, heap: &Heap, index: usize) -> Option<usize> {
        let capacity = self.capacity();
        let mask = capacity - 1;
        let mut entry = (Self::hash(index) as usize) & mask;
        let mut step = 1usize;
        loop {
            let key = self.key_word(heap, entry);
            if key == heap.known().undefined.as_tagged(heap) {
                return None;
            }
            if let Some(k) = key.to_i64().map(|v| v as usize) {
                if k == index {
                    return Some(entry);
                }
            }
            entry = (entry + step) & mask;
            step += 1;
        }
    }

    /// First never-used or deleted slot on the probe path.
    fn find_insertion(&self, heap: &Heap, index: usize) -> usize {
        let capacity = self.capacity();
        let mask = capacity - 1;
        let hole = heap.known().the_hole.as_tagged(heap);
        let undefined = heap.known().undefined.as_tagged(heap);
        let mut entry = (Self::hash(index) as usize) & mask;
        let mut step = 1usize;
        loop {
            let key = self.key_word(heap, entry);
            if key == undefined || key == hole {
                return entry;
            }
            entry = (entry + step) & mask;
            step += 1;
        }
    }

    pub fn compute_capacity(at_least: usize) -> usize {
        let raw = at_least + (at_least >> 1);
        let cap = raw.max(MIN_CAPACITY).next_power_of_two();
        cap
    }

    pub fn has_sufficient_capacity(&self, n: usize) -> bool {
        let nof = self.nof() + n;
        let capacity = self.capacity();
        let nod = self.nod();
        if nof >= capacity || nod > (capacity - nof) / 2 {
            return false;
        }
        let needed_free = nof / 2;
        nof + needed_free <= capacity
    }

    /// Overwrite an existing entry; `false` when `index` is absent.
    pub fn set_in_place(
        &self,
        heap: &Heap,
        index: usize,
        value: Tagged<'_, Value>,
        details: i64,
    ) -> bool {
        match self.find(heap, index) {
            Some(entry) => {
                let host = self.tagged(heap);
                self.value_slot(entry).set(heap, host, value);
                self.details_slot(entry)
                    .set(heap, host, Smi::new(details).into_tagged());
                true
            }
            None => false,
        }
    }

    /// Rehash every live entry into `new` (capacity already set).
    fn rehash_into(&self, heap: &Heap, new: &NumberDictionary) {
        for entry in 0..self.capacity() {
            let Some(index) = self.key_at(heap, entry) else {
                continue;
            };
            let target = new.find_insertion(heap, index);
            let host = new.tagged(heap);
            new.key_slot(target)
                .set(heap, host, Smi::new(index as i64).into_tagged());
            new.value_slot(target)
                .set(heap, host, self.value_at(heap, entry));
            new.details_slot(target).set(
                heap,
                host,
                Smi::new(self.details_at(heap, entry)).into_tagged(),
            );
        }
        let host = new.tagged(heap);
        new.nof.set(heap, host, Smi::new(self.nof() as i64));
        new.nod.set(heap, host, Smi::new(0));
        if self.requires_slow_elements() {
            new.set_requires_slow_elements(heap);
        }
    }

    /// Grow (or same-size de-tombstone) if needed, returning the table
    /// to use: either `dict` itself or a fresh rehashed copy.
    pub fn ensure_capacity<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        dict: &Handle<'s, NumberDictionary>,
    ) -> Handle<'s, NumberDictionary> {
        if dict.as_tagged(heap).as_ref().has_sufficient_capacity(1) {
            return *dict;
        }
        let nof = dict.as_tagged(heap).as_ref().nof();
        let fresh = NumberDictionary::new(heap, scope, nof + 1);
        dict.as_tagged(heap)
            .as_ref()
            .rehash_into(heap, fresh.as_tagged(heap).as_ref());
        fresh
    }

    /// Insert or overwrite `index`. Returns the table to install (either
    /// the original or a grown copy).
    pub fn insert<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        dict: &Handle<'s, NumberDictionary>,
        index: usize,
        value: Tagged<'_, Value>,
        details: i64,
    ) -> Handle<'s, NumberDictionary> {
        let current = *dict;
        if current
            .as_tagged(heap)
            .as_ref()
            .set_in_place(heap, index, value, details)
        {
            return current;
        }
        let table = Self::ensure_capacity(heap, scope, &current);
        {
            let t = table.as_tagged(heap).as_ref();
            let entry = t.find_insertion(heap, index);
            let host = t.tagged(heap);
            let reused_tombstone = t.key_word(heap, entry) == heap.known().the_hole.as_tagged(heap);
            t.key_slot(entry)
                .set(heap, host, Smi::new(index as i64).into_tagged());
            t.value_slot(entry).set(heap, host, value);
            t.details_slot(entry)
                .set(heap, host, Smi::new(details).into_tagged());
            let nof = t.nof() + 1;
            t.nof.set(heap, host, Smi::new(nof as i64));
            if reused_tombstone {
                t.nod.set(heap, host, Smi::new((t.nod() - 1) as i64));
            }
        }
        table.as_tagged(heap).as_ref().note_index(heap, index);
        table
    }

    /// Remove `index` if present (tombstone; no shrink).
    pub fn delete(&self, heap: &Heap, index: usize) -> bool {
        let Some(entry) = self.find(heap, index) else {
            return false;
        };
        let host = self.tagged(heap);
        let hole = heap.known().the_hole.as_tagged(heap).erase();
        self.key_slot(entry).set(heap, host, hole);
        self.value_slot(entry).set(heap, host, hole);
        self.nof.set(heap, host, Smi::new((self.nof() - 1) as i64));
        self.nod.set(heap, host, Smi::new((self.nod() + 1) as i64));
        true
    }

    /// Allocate an empty table sized for `at_least` live entries.
    pub fn new<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        at_least: usize,
    ) -> Handle<'s, NumberDictionary> {
        let capacity = Self::compute_capacity(at_least);
        heap.allocate_handle::<NumberDictionary>(NumberDictionaryInit { capacity }, scope)
    }

    /// Iterate live entries in slot order (NOT index order — callers
    /// that need ascending indices must sort).
    pub fn for_each_entry<'a>(
        &'a self,
        heap: &'a Heap,
        mut f: impl FnMut(usize, Tagged<'a, Value>, i64),
    ) {
        for entry in 0..self.capacity() {
            let Some(index) = self.key_at(heap, entry) else {
                continue;
            };
            f(
                index,
                self.value_at(heap, entry),
                self.details_at(heap, entry),
            );
        }
    }
}

/// What a plain element access faces at some index.
pub enum EntryClass {
    Absent,
    Data,
    ReadOnly,
    Accessor,
}

impl HeapObject for NumberDictionary {
    const KIND: ObjectKind = ObjectKind::NumberDictionary;
    type Init<'a> = NumberDictionaryInit;

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.capacity)
    }

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        let host = self.tagged(heap);
        self.header
            .map
            .init(heap.known().number_dictionary_map.as_tagged(heap));
        self.capacity
            .set(heap, host, Smi::new(config.capacity as i64));
        self.nof.set(heap, host, Smi::new(0));
        self.nod.set(heap, host, Smi::new(0));
        self.flags.set(heap, host, Smi::new(0));
        let undefined = heap.known().undefined.as_tagged(heap).erase();
        for entry in 0..config.capacity {
            self.key_slot(entry).set(heap, host, undefined);
            self.value_slot(entry).set(heap, host, undefined);
            self.details_slot(entry).set(heap, host, undefined);
        }
    }

    fn header(&self) -> &crate::Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.capacity())
    }
}

impl EdgeVisitable for NumberDictionary {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.capacity.as_raw());
        visitor.visit(self.nof.as_raw());
        visitor.visit(self.nod.as_raw());
        visitor.visit(self.flags.as_raw());
        let capacity = self.capacity.to_smi_unchecked().value() as usize;
        for entry in 0..capacity * ENTRY_WORDS {
            visitor.visit(unsafe { &*self.entries.as_ptr().add(entry) }.as_raw());
        }
    }
}
