use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Mutex;

use vm::{
    EdgeVisitable, FixedByteArray, Handle, HandleScope, InternedString, LocalHeap, Visitor,
    WeakGcCell,
};

/// Content hash for interned strings (FNV-1a, masked into smi range).
/// TODO: decide on a hash algorithm
fn hash_bytes(bytes: &[u8]) -> i64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h & ((1 << 62) - 1)) as i64
}

// TODO: weak GC cell is fake kinda, make this better
pub struct StringInterner {
    table: Mutex<HashMap<Box<str>, WeakGcCell<InternedString>>>,
}

impl StringInterner {
    pub fn new() -> Self {
        Self {
            table: Mutex::new(HashMap::new()),
        }
    }

    pub fn intern<'s, L: LocalHeap>(
        &self,
        heap: &mut L,
        scope: &'s HandleScope<'_>,
        s: impl AsRef<str>,
    ) -> Handle<'s, InternedString> {
        let s = s.as_ref();

        {
            let mut table = self.table.lock().unwrap();
            if let Some(entry) = table.get(s) {
                match handle_from_entry(heap, scope, entry) {
                    Some(h) => return h,
                    None => {
                        // dead entry: prune and fall through to re-intern
                        table.remove(s);
                    }
                }
            }
        }

        let backing = heap
            .allocate::<FixedByteArray>(s.as_bytes())
            .into_handle(scope);
        let hash = hash_bytes(s.as_bytes());
        let handle = heap
            .allocate::<InternedString>((backing, hash))
            .into_handle(scope);

        let mut table = self.table.lock().unwrap();
        match table.entry(s.into()) {
            Entry::Occupied(mut e) => match handle_from_entry(heap, scope, e.get()) {
                Some(h) => h,
                // the entry died mid-race => replace it
                None => {
                    e.insert(WeakGcCell::new(handle.get()));
                    handle
                }
            },
            Entry::Vacant(e) => {
                e.insert(WeakGcCell::new(handle.get()));
                handle
            }
        }
    }
}

fn handle_from_entry<'s, L: LocalHeap>(
    heap: &mut L,
    scope: &'s HandleScope<'_>,
    entry: &WeakGcCell<InternedString>,
) -> Option<Handle<'s, InternedString>> {
    heap.no_gc(|nogc, _| entry.upgrade(nogc).map(|r| r.into_handle(scope)))
}

impl EdgeVisitable for StringInterner {
    fn visit_edges(&self, visitor: &mut impl Visitor) {
        for cell in self.table.lock().unwrap().values() {
            visitor.visit(cell.as_raw());
        }
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}
