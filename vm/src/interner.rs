use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Mutex;

use crate::{
    EdgeVisitable, FixedByteArray, Handle, HandleScope, Heap, InternedString, Visitor,
    heap::WeakGcCell, string_content_hash,
};

pub struct StringInterner {
    /// keys are WTF-8: lone surrogates from JS string literals appear as
    /// their 3-byte encoding, so keys are raw bytes, not `str`
    table: Mutex<HashMap<Box<[u8]>, WeakGcCell<InternedString>>>,
}

impl StringInterner {
    pub fn new() -> Self {
        Self {
            table: Mutex::new(HashMap::new()),
        }
    }

    pub fn intern<'s>(
        &self,
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        s: impl AsRef<[u8]>,
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

        let backing = heap.allocate::<FixedByteArray>(s).into_handle(scope);
        let hash = string_content_hash(s);
        let handle = heap
            .allocate::<InternedString>((backing, hash))
            .into_handle(scope);

        let mut table = self.table.lock().unwrap();
        match table.entry(s.into()) {
            Entry::Occupied(mut e) => match handle_from_entry(heap, scope, e.get()) {
                Some(h) => h,
                // the entry died mid-race => replace it
                None => {
                    e.insert(WeakGcCell::new_strong(handle.get()));
                    handle
                }
            },
            Entry::Vacant(e) => {
                e.insert(WeakGcCell::new_strong(handle.get()));
                handle
            }
        }
    }
}

fn handle_from_entry<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    entry: &WeakGcCell<InternedString>,
) -> Option<Handle<'s, InternedString>> {
    heap.no_gc(|nogc| entry.upgrade(nogc).map(|r| r.into_handle(scope)))
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
