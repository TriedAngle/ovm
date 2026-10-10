use std::sync::{Mutex, MutexGuard};

use crate::{
    Cell, CellInit, FixedArray, Handle, HandleScope, Heap, Map, MapKind, MaybeWeak, Object,
    PrototypeInfo, PrototypeInfoInit, Smi, Tagged, Value, WeakFixedArray, WeakFixedArrayInit,
};

pub struct PrototypeRegistry {
    lock: Mutex<()>,
}

impl PrototypeRegistry {
    pub fn new() -> Self {
        Self {
            lock: Mutex::new(()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, ()> {
        self.lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Default for PrototypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Prototype;

impl Prototype {
    pub fn try_get_validity_cell_holder_map<'a>(
        heap: &'a Heap,
        map: Tagged<'a, Map>,
    ) -> Option<Tagged<'a, Map>> {
        if map.as_ref().is_prototype() {
            return Some(map);
        }
        let proto = map.as_ref().prototype.get(heap);
        let object = trackable_prototype(heap, proto)?;
        Some(object.as_ref().map(heap))
    }

    pub fn get_or_create_prototype_chain_validity_cell<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        map: Handle<'s, Map>,
    ) -> Option<Handle<'s, Cell>> {
        // Self-style multiple parents: no single prototype to hang the cell
        // on, so the receiver map owns the cell and every map in its parent
        // closure is registered as a dependent of it.
        if map
            .as_tagged(heap)
            .prototype
            .get(heap)
            .get_as::<FixedArray>(heap)
            .is_some()
        {
            return Self::get_or_create_multi_parent_cell(heap, scope, map);
        }
        let holder = Self::try_get_validity_cell_holder_map(heap, map.as_tagged(heap))?;
        let holder = scope.handle(holder);
        holder.as_tagged(heap).mark_prototype(heap);

        let _guard = heap.prototype_registry().lock();
        lazy_register_prototype_user_locked(heap, scope, holder);

        if let Some(cell) = holder
            .as_tagged(heap)
            .as_ref()
            .published_validity_cell(heap)
            && cell.as_ref().is_valid(heap)
        {
            return Some(cell.as_handle(scope));
        }

        let cell = heap.allocate_handle::<Cell>(
            CellInit {
                value: Smi::new(1).encode(),
            },
            scope,
        );
        holder
            .as_tagged(heap)
            .as_ref()
            .set_validity_cell(heap, cell.as_tagged(heap));
        Some(cell)
    }

    /// Validity cell for a map whose prototype is a Self-style parent list.
    /// The cell is published on the receiver map itself and registered as a
    /// dependent of *every* map in the receiver's transitive parent closure,
    /// so any shape change anywhere in that closure clears it.
    fn get_or_create_multi_parent_cell<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        map: Handle<'s, Map>,
    ) -> Option<Handle<'s, Cell>> {
        if let Some(cell) = map.as_tagged(heap).published_validity_cell(heap)
            && cell.as_ref().is_valid(heap)
        {
            return Some(cell.as_handle(scope));
        }

        let cell = heap.allocate_handle::<Cell>(
            CellInit {
                value: Smi::new(1).encode(),
            },
            scope,
        );

        let mut parents: Vec<Handle<'s, Map>> = Vec::new();
        if !collect_parent_maps(heap, scope, map.as_tagged(heap), &mut parents) {
            return None;
        }
        let mut visited: Vec<usize> = Vec::new();
        let mut i = 0;
        while i < parents.len() {
            let parent = parents[i];
            i += 1;
            let id = parent.as_tagged(heap).raw_addr() as usize;
            if visited.contains(&id) {
                continue;
            }
            visited.push(id);
            parent.as_tagged(heap).mark_prototype(heap);
            let info = Self::get_or_create_prototype_info(heap, scope, parent);
            append_prototype_child(heap, scope, info, cell);
            if !collect_parent_maps(heap, scope, parent.as_tagged(heap), &mut parents) {
                return None;
            }
        }

        map.as_tagged(heap)
            .set_validity_cell(heap, cell.as_tagged(heap));
        Some(cell)
    }

    pub fn ensure_store_transition_validity_cell<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        map: Handle<'s, Map>,
    ) {
        if map
            .as_tagged(heap)
            .as_ref()
            .is_prototype_validity_cell_valid(heap)
        {
            return;
        }
        match Self::get_or_create_prototype_chain_validity_cell(heap, scope, map) {
            Some(cell) => map
                .as_tagged(heap)
                .as_ref()
                .set_validity_cell(heap, cell.as_tagged(heap)),
            None => map
                .as_tagged(heap)
                .as_ref()
                .set_validity_cell_sentinel(heap),
        }
    }

    pub fn get_or_create_prototype_info<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        map: Handle<'s, Map>,
    ) -> Handle<'s, PrototypeInfo> {
        if let Some(info) = map.as_tagged(heap).try_get_prototype_info(heap) {
            return info.as_handle(scope);
        }
        loop {
            let expected = map.as_tagged(heap).prototype_info.load_word(heap);
            if let Some(winner) = map.as_tagged(heap).prototype_info.load(heap) {
                return winner.as_handle(scope);
            }
            let info = heap.allocate_handle::<PrototypeInfo>(PrototypeInfoInit::default(), scope);
            let host = map.as_tagged(heap).erase();
            match map.as_tagged(heap).prototype_info.publish(
                heap,
                host,
                expected,
                info.as_tagged(heap),
            ) {
                Ok(()) => return info,
                Err(_) => continue,
            }
        }
    }

    pub fn lazy_register_prototype_user<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        user: Handle<'s, Map>,
    ) {
        let _guard = heap.prototype_registry().lock();
        lazy_register_prototype_user_locked(heap, scope, user);
    }

    pub fn shape_changed(heap: &Heap, old_map: Tagged<'_, Map>) {
        if !old_map.is_prototype() {
            return;
        }
        invalidate_prototype_chains(heap, old_map);
    }

    #[inline]
    pub fn element_mutated(heap: &Heap, obj: Tagged<'_, Object>) {
        let map = obj.map(heap);
        if !map.kind().is_prototype() {
            return;
        }
        invalidate_prototype_chains(heap, map);
    }

    /// `element_mutated` for callers that already hold the receiver's
    /// decoded map kind: skips the map/kind reload on the hot store path.
    #[inline]
    pub fn element_mutated_kind(heap: &Heap, obj: Tagged<'_, Object>, kind: MapKind) {
        if !kind.is_prototype() {
            return;
        }
        invalidate_prototype_chains(heap, obj.map(heap));
    }

    /// Whether an indexed store into `receiver` (map `map`) may skip the
    /// prototype chain
    pub fn indexed_store_chain_ok<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        receiver: &Handle<'_, Value>,
        map: &Handle<'s, Map>,
        index: usize,
    ) -> bool {
        if map
            .as_tagged(heap)
            .as_ref()
            .published_validity_cell(heap)
            .is_some_and(|cell| cell.as_ref().is_valid(heap))
        {
            return true;
        }
        let holds = { receiver.as_tagged(heap).chain_holds_index_name(heap, index) };
        if holds {
            return false;
        }
        // only cacheable when the chain holds no integer-named
        // descriptors at all: otherwise cleanliness depends on the index
        let any_index = { receiver.as_tagged(heap).chain_holds_any_index_name(heap) };
        if any_index {
            return true;
        }
        Self::ensure_store_transition_validity_cell(heap, scope, *map);
        true
    }
}

fn trackable_prototype<'a>(heap: &'a Heap, value: Tagged<'a, Value>) -> Option<Tagged<'a, Object>> {
    if !value.is_strong_ptr() {
        return None;
    }
    if value.ptr_eq(heap.known().null.as_tagged(heap).erase())
        || value.ptr_eq(heap.known().undefined.as_tagged(heap).erase())
        || value.ptr_eq(heap.known().the_hole.as_tagged(heap).erase())
    {
        return None;
    }
    let object = value.get_as::<Object>(heap)?;
    if !object.as_ref().map(heap).kind().kind().is_js_receiver() {
        return None;
    }
    Some(object)
}

/// Collect the maps of `map`'s direct parents: a single prototype object or
/// every parent in a Self-style parent list. Returns `false` (and leaves
/// `out` partially filled) when a parent cannot be tracked, which makes the
/// whole closure uncacheable.
fn collect_parent_maps<'s>(
    heap: &Heap,
    scope: &'s HandleScope<'_>,
    map: Tagged<'_, Map>,
    out: &mut Vec<Handle<'s, Map>>,
) -> bool {
    let proto = map.prototype.get(heap);
    if let Some(pairs) = proto.get_as::<FixedArray>(heap) {
        let mut i = 1;
        while i < pairs.len() {
            if !push_parent_map(heap, scope, pairs.at(heap, i), out) {
                return false;
            }
            i += 2;
        }
        return true;
    }
    if !proto.is_strong_ptr() {
        return true;
    }
    let known = heap.known();
    if proto.ptr_eq(known.null.as_tagged(heap).erase())
        || proto.ptr_eq(known.undefined.as_tagged(heap).erase())
        || proto.ptr_eq(known.the_hole.as_tagged(heap).erase())
    {
        return true;
    }
    push_parent_map(heap, scope, proto, out)
}

fn push_parent_map<'s>(
    heap: &Heap,
    scope: &'s HandleScope<'_>,
    word: Tagged<'_, Value>,
    out: &mut Vec<Handle<'s, Map>>,
) -> bool {
    if !word.is_strong_ptr() {
        return false;
    }
    let known = heap.known();
    if word.ptr_eq(known.null.as_tagged(heap).erase())
        || word.ptr_eq(known.undefined.as_tagged(heap).erase())
        || word.ptr_eq(known.the_hole.as_tagged(heap).erase())
    {
        return false;
    }
    let Some(obj) = word.as_heap_object() else {
        return false;
    };
    let kind = obj.map(heap).kind();
    if kind.is_proxy() || !kind.kind().is_js_receiver() {
        return false;
    }
    out.push(scope.handle(obj.map(heap)));
    true
}

fn lazy_register_prototype_user_locked<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    user: Handle<'s, Map>,
) {
    let mut current = user;
    loop {
        let proto_word = current.as_tagged(heap).prototype.get(heap);
        let Some(proto) = trackable_prototype(heap, proto_word) else {
            break;
        };
        let proto_map = scope.handle(proto.as_ref().map(heap));
        proto_map.as_tagged(heap).mark_prototype(heap);
        let proto_info = Prototype::get_or_create_prototype_info(heap, scope, proto_map);
        let current_info = Prototype::get_or_create_prototype_info(heap, scope, current);

        let slot = current_info.as_tagged(heap).registry_slot();
        let registered = slot >= 1 && {
            let users = proto_info.as_tagged(heap).prototype_users(heap);
            users.is_some_and(|users| {
                let users = users.as_ref();
                (slot as usize) < users.len()
                    && users
                        .element_slot(slot as usize)
                        .get(heap)
                        .as_strong()
                        .is_some_and(|user| user.ptr_eq(current.as_tagged(heap).erase()))
            })
        };
        if !registered {
            let assigned = append_prototype_child(heap, scope, proto_info, current);
            current_info.as_tagged(heap).set_registry_slot(
                heap,
                current_info.as_tagged(heap).erase(),
                assigned,
            );
        }
        current = proto_map;
    }
}

/// Append `child` (a user map, or a validity cell for multi-parent maps) to
/// `info`'s weak user registry, reusing an empty slot when possible.
fn append_prototype_child<'s, T: 's>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    info: Handle<'s, PrototypeInfo>,
    child: Handle<'s, T>,
) -> i64 {
    enum Plan {
        Init,
        Reuse(usize),
        Grow(usize),
    }

    let plan = match info.as_tagged(heap).prototype_users(heap) {
        None => Plan::Init,
        Some(array) => {
            let array = array.as_ref();
            let len = array.len();
            if len == 0 {
                Plan::Init
            } else {
                let mut empty_slot = array.element_slot(0).get(heap).to_i64().unwrap_or(0);
                if empty_slot == 0 && len > 1 {
                    for i in 1..len {
                        if array.element_slot(i).is_cleared() {
                            mark_slot_empty(heap, array, i);
                        }
                    }
                    empty_slot = array.element_slot(0).get(heap).to_i64().unwrap_or(0);
                }
                if empty_slot > 0 && (empty_slot as usize) < len {
                    Plan::Reuse(empty_slot as usize)
                } else {
                    Plan::Grow(len)
                }
            }
        }
    };

    match plan {
        Plan::Reuse(slot) => {
            let array = info
                .as_tagged(heap)
                .as_ref()
                .prototype_users(heap)
                .expect("planned against an existing registry");
            let array = array.as_ref();
            let next = array.element_slot(slot).get(heap).to_i64().unwrap_or(0);
            array.set_weak(heap, slot, child.as_tagged(heap).erase());
            array.set(heap, 0, Smi::new(next).into_tagged().as_maybe_weak());
            slot as i64
        }
        Plan::Init => {
            let array = heap.allocate_token_enter_heap(
                WeakFixedArray::<Value>::layout_for(2),
                |token, heap| {
                    let values = [
                        Smi::new(0).into_tagged().as_maybe_weak(),
                        child.as_tagged(heap).erase().as_weak(),
                    ];
                    token
                        .allocate::<WeakFixedArray>(WeakFixedArrayInit { values: &values })
                        .as_handle(scope)
                },
            );
            let host = info.as_tagged(heap).erase();
            info.as_tagged(heap)
                .as_ref()
                .prototype_users
                .set(heap, host, array.as_tagged(heap));
            1
        }
        Plan::Grow(len) => {
            let array = heap.allocate_token_enter_heap(
                WeakFixedArray::<Value>::layout_for(len + 1),
                |token, heap| {
                    let old = info
                        .as_tagged(heap)
                        .as_ref()
                        .prototype_users(heap)
                        .expect("planned against an existing registry");
                    let old = old.as_ref();
                    let mut values: Vec<Tagged<'_, MaybeWeak<Value>>> = Vec::with_capacity(len + 1);
                    for i in 0..len {
                        values.push(old.element_slot(i).get(heap));
                    }
                    values.push(child.as_tagged(heap).erase().as_weak());
                    token
                        .allocate::<WeakFixedArray>(WeakFixedArrayInit { values: &values })
                        .as_handle(scope)
                },
            );
            let host = info.as_tagged(heap).erase();
            info.as_tagged(heap)
                .as_ref()
                .prototype_users
                .set(heap, host, array.as_tagged(heap));
            len as i64
        }
    }
}

fn mark_slot_empty(heap: &Heap, array: &WeakFixedArray, index: usize) {
    let previous = array.element_slot(0).get(heap);
    array.set(heap, index, previous);
    array.set(
        heap,
        0,
        Smi::new(index as i64).into_tagged().as_maybe_weak(),
    );
}

fn invalidate_one_prototype_validity_cell(heap: &Heap, map: Tagged<'_, Map>) {
    if let Some(cell) = map.as_ref().published_validity_cell(heap) {
        let cell = cell.as_ref();
        if cell.is_valid(heap) {
            cell.clear();
        }
    }
}

fn invalidate_prototype_chains(heap: &Heap, map: Tagged<'_, Map>) {
    let mut current = Some(map);
    let mut next: Option<Tagged<'_, Map>> = None;
    while let Some(m) = current {
        invalidate_one_prototype_validity_cell(heap, m);

        let Some(info) = m.as_ref().try_get_prototype_info(heap) else {
            return;
        };
        let Some(users) = info.as_ref().prototype_users(heap) else {
            return;
        };
        let users = users.as_ref();
        for i in 1..users.len() {
            let Some(child) = users.element_slot(i).get(heap).as_strong() else {
                continue;
            };
            // Multi-parent maps register their own validity cell (with an
            // already-complete closure) instead of a map to recurse into.
            if let Some(cell) = child.get_as::<Cell>(heap) {
                if cell.is_valid(heap) {
                    cell.clear();
                }
                continue;
            }
            let Some(user_map) = child.get_as::<Map>(heap) else {
                continue;
            };
            if next.is_none() {
                next = Some(user_map);
            } else {
                invalidate_prototype_chains(heap, user_map);
            }
        }
        current = next.take();
    }
}
