use std::collections::HashMap;
use std::sync::Mutex;

use crate::{
    DenseString, EdgeVisitable, Handle, HandleSet, Heap, StringData, Visitor, string_content_hash,
};

use crate::heap::WeakGcCell;

enum InternKey {
    Latin1(Box<[u8]>),
    Utf16(Box<[u16]>),
}

impl InternKey {
    fn from_data(data: StringData<'_>) -> Self {
        match data {
            StringData::Latin1(b) => InternKey::Latin1(b.into()),
            StringData::Utf16(u) if u.iter().all(|&c| c <= 0xFF) => {
                InternKey::Latin1(u.iter().map(|&c| c as u8).collect())
            }
            StringData::Utf16(u) => InternKey::Utf16(u.into()),
        }
    }

    fn data(&self) -> StringData<'_> {
        match self {
            InternKey::Latin1(b) => StringData::Latin1(b),
            InternKey::Utf16(u) => StringData::Utf16(u),
        }
    }
}

type InternTable = HashMap<i64, Vec<(InternKey, WeakGcCell<DenseString>)>>;

pub struct StringInterner {
    table: Mutex<InternTable>,
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
        scope: &'s impl HandleSet,
        data: StringData<'_>,
    ) -> Handle<'s, DenseString> {
        let staged: Result<Handle<'s, DenseString>, (InternKey, i64)> = {
            let hash = string_content_hash(data);
            let mut table = self.table.lock().unwrap();
            match probe_unlocked(heap, scope, &mut table, hash, data) {
                Some(handle) => Ok(handle),
                None => Err((InternKey::from_data(data), hash)),
            }
        };
        match staged {
            Ok(handle) => handle,
            Err((key, hash)) => self.insert_new(heap, scope, key, hash),
        }
    }

    pub fn intern_str<'s>(
        &self,
        heap: &mut Heap,
        scope: &'s impl HandleSet,
        s: &str,
    ) -> Handle<'s, DenseString> {
        if s.is_ascii() {
            return self.intern(heap, scope, StringData::Latin1(s.as_bytes()));
        }
        let units: Vec<u16> = s.encode_utf16().collect();
        self.intern(heap, scope, StringData::Utf16(&units))
    }

    pub fn intern_value<'s>(
        &self,
        heap: &mut Heap,
        scope: &'s impl HandleSet,
        s: &Handle<'_, DenseString>,
    ) -> Handle<'s, DenseString> {
        let staged: Result<Handle<'s, DenseString>, (InternKey, i64)> = {
            let r = s.heap_ref(heap);
            let hash = r.hash(heap);
            let mut table = self.table.lock().unwrap();
            let data = r.data(heap);
            match probe_unlocked(heap, scope, &mut table, hash, data) {
                Some(handle) => Ok(handle),
                None => Err((InternKey::from_data(data), hash)),
            }
        };
        match staged {
            Ok(handle) => handle,
            Err((key, hash)) => self.insert_new(heap, scope, key, hash),
        }
    }

    fn insert_new<'s>(
        &self,
        heap: &mut Heap,
        scope: &'s impl HandleSet,
        key: InternKey,
        hash: i64,
    ) -> Handle<'s, DenseString> {
        let handle = heap
            .allocate::<DenseString>((key.data(), hash))
            .into_handle(scope);
        let mut table = self.table.lock().unwrap();
        let raced = probe_unlocked(heap, scope, &mut table, hash, key.data());
        match raced {
            Some(existing) => existing,
            None => {
                table
                    .entry(hash)
                    .or_default()
                    .push((key, WeakGcCell::new_strong(handle.get())));
                handle
            }
        }
    }
}

fn probe_unlocked<'s>(
    heap: &Heap,
    scope: &'s impl HandleSet,
    table: &mut InternTable,
    hash: i64,
    data: StringData<'_>,
) -> Option<Handle<'s, DenseString>> {
    let bucket = table.get_mut(&hash)?;
    let mut i = 0;
    while i < bucket.len() {
        if bucket[i].0.data().eq(&data) {
            if let Some(r) = bucket[i].1.upgrade(heap) {
                return Some(r.into_handle(scope));
            }
            // dead entry: prune and keep scanning
            bucket.swap_remove(i);
            continue;
        }
        i += 1;
    }
    None
}

impl EdgeVisitable for StringInterner {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        for bucket in self.table.lock().unwrap().values() {
            for (_, cell) in bucket {
                visitor.visit(cell.as_raw());
            }
        }
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}
