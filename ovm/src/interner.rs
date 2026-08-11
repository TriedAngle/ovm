use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Mutex;

use core::alloc::Layout;

use vm::{
    ByteArray, Handle, HandleScope, HeapObject, HeapPtr, InternedString, LocalHeap, Smi,
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
    table: Mutex<HashMap<Box<str>, HeapPtr<InternedString>>>,
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
            let table = self.table.lock().unwrap();
            if let Some(&ptr) = table.get(s) {
                return scope.create_handle_from_ptr(ptr);
            }
        }

        let lb = ByteArray::layout_for(s.len());
        let ls = Layout::new::<InternedString>();
        let (total, _) = lb.extend(ls).expect("string layout");

        let ptr = heap.allocate_token_enter_nogc(total, |token, nogc, heap| {
                let backing = token.allocate_ref::<ByteArray>(lb, nogc);
                backing.init(heap, s.as_bytes());

                let interned = token.allocate_ref::<InternedString>(ls, nogc);
                let inner = interned.string();
                inner
                    .backing
                    .set(heap, inner.erase(), backing.into_tagged());
                inner.hash.set(
                    heap,
                    inner.erase(),
                    Smi::new_unchecked(hash_bytes(s.as_bytes())),
                );
                interned.into_ptr()
            });

        let mut table = self.table.lock().unwrap();
        match table.entry(s.into()) {
            Entry::Occupied(e) => scope.create_handle_from_ptr(*e.get()),
            Entry::Vacant(e) => {
                e.insert(ptr);
                scope.create_handle_from_ptr(ptr)
            }
        }
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}
