use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Mutex;

use core::alloc::Layout;

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

        let lb = FixedByteArray::layout_for(s.len());
        let ls = Layout::new::<InternedString>();
        let (total, _) = lb.extend(ls).expect("string layout");

        let handle = heap.allocate_token_enter_nogc(total, |token, nogc, _heap| {
            let backing = token.allocate_ref::<FixedByteArray>(s.as_bytes(), nogc);
            let hash = hash_bytes(s.as_bytes());
            let interned =
                token.allocate_ref::<InternedString>((backing.into_tagged(), hash), nogc);
            interned.into_handle(scope)
        });

        let mut table = self.table.lock().unwrap();
        match table.entry(s.into()) {
            Entry::Occupied(mut e) => match handle_from_entry(heap, scope, e.get()) {
                Some(h) => h,
                // the canonical entry died mid-race: replace it with ours
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
            visitor.visit_weak_slot(cell);
        }
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}
