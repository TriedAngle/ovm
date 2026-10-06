use std::sync::{Mutex, MutexGuard};

use crate::{
    Cell, CellInit, Handle, HandleScope, Heap, Map, MaybeWeak, Object, PrototypeInfo,
    PrototypeInfoInit, Smi, Tagged, Value, WeakFixedArray, WeakFixedArrayInit,
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
        Some(object.as_ref().map_ref(heap))
    }

    pub fn get_or_create_prototype_chain_validity_cell<'s>(
        heap: &mut Heap,
        scope: &'s HandleScope<'_>,
        map: Handle<'s, Map>,
    ) -> Option<Handle<'s, Cell>> {
        let holder = Self::try_get_validity_cell_holder_map(heap, map.as_tagged(heap))?;
        let holder = scope.handle(holder);
        holder.as_tagged(heap).as_ref().mark_prototype(heap);

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
        if let Some(info) = map.as_tagged(heap).as_ref().try_get_prototype_info(heap) {
            return info.as_handle(scope);
        }
        loop {
            let expected = map.as_tagged(heap).as_ref().prototype_info.load_word(heap);
            if Value::from_bits(expected) != heap.known().the_hole.raw()
                && let Some(winner) = map.as_tagged(heap).as_ref().prototype_info.load(heap)
                && winner.raw().is_strong_ptr()
            {
                return winner.as_handle(scope);
            }
            let info = heap.allocate_handle::<PrototypeInfo>(PrototypeInfoInit::default(), scope);
            let host = map.as_tagged(heap).erase();
            match map.as_tagged(heap).as_ref().prototype_info.publish(
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

    pub fn notify_map_change(heap: &Heap, old_map: Tagged<'_, Map>, _new_map: Tagged<'_, Map>) {
        Self::shape_changed(heap, old_map);
    }

    pub fn shape_changed(heap: &Heap, old_map: Tagged<'_, Map>) {
        if !old_map.as_ref().is_prototype() {
            return;
        }
        invalidate_prototype_chains(heap, old_map);
    }
}

fn trackable_prototype<'a>(heap: &'a Heap, value: Tagged<'a, Value>) -> Option<Tagged<'a, Object>> {
    if !value.raw().is_strong_ptr() {
        return None;
    }
    if value.ptr_eq(heap.known().null.as_tagged(heap).erase())
        || value.ptr_eq(heap.known().undefined.as_tagged(heap).erase())
        || value.ptr_eq(heap.known().the_hole.as_tagged(heap).erase())
    {
        return None;
    }
    let object = value.get_as::<Object>(heap)?;
    if !object.as_ref().map_ref(heap).kind().kind().is_js_receiver() {
        return None;
    }
    Some(object)
}

fn lazy_register_prototype_user_locked<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    user: Handle<'s, Map>,
) {
    let mut current = user;
    loop {
        let proto_word = current.as_tagged(heap).as_ref().prototype.get(heap);
        let Some(proto) = trackable_prototype(heap, proto_word) else {
            break;
        };
        let proto_map = scope.handle(proto.as_ref().map_ref(heap));
        proto_map.as_tagged(heap).as_ref().mark_prototype(heap);
        let proto_info = Prototype::get_or_create_prototype_info(heap, scope, proto_map);
        let current_info = Prototype::get_or_create_prototype_info(heap, scope, current);

        let slot = current_info.as_tagged(heap).as_ref().registry_slot();
        let registered = slot >= 1 && {
            let users = proto_info.as_tagged(heap).as_ref().prototype_users(heap);
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
            let assigned = append_prototype_user(heap, scope, proto_info, current);
            current_info.as_tagged(heap).as_ref().set_registry_slot(
                heap,
                current_info.as_tagged(heap).erase(),
                assigned,
            );
        }
        current = proto_map;
    }
}

fn append_prototype_user<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    info: Handle<'s, PrototypeInfo>,
    user: Handle<'s, Map>,
) -> i64 {
    enum Plan {
        Init,
        Reuse(usize),
        Grow(usize),
    }

    let plan = match info.as_tagged(heap).as_ref().prototype_users(heap) {
        None => Plan::Init,
        Some(array) => {
            let array = array.as_ref();
            let len = array.len();
            if len == 0 {
                Plan::Init
            } else {
                let mut empty_slot = smi_value(array.element_slot(0).get(heap)).unwrap_or(0);
                if empty_slot == 0 && len > 1 {
                    for i in 1..len {
                        if array.element_slot(i).is_cleared() {
                            mark_slot_empty(heap, array, i);
                        }
                    }
                    empty_slot = smi_value(array.element_slot(0).get(heap)).unwrap_or(0);
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
            let next = smi_value(array.element_slot(slot).get(heap)).unwrap_or(0);
            array.set_weak(heap, slot, user.as_tagged(heap).erase());
            array.set(heap, 0, smi_word(next));
            slot as i64
        }
        Plan::Init => {
            let array = heap.allocate_token_enter_heap(
                WeakFixedArray::<Value>::layout_for(2),
                |token, heap| {
                    let values = [smi_word(0), user.as_tagged(heap).erase().as_weak()];
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
                    values.push(user.as_tagged(heap).erase().as_weak());
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

fn smi_value(word: Tagged<'_, MaybeWeak<Value>>) -> Option<i64> {
    Smi::decode(word.raw()).map(|smi| smi.value())
}

fn smi_word(value: i64) -> Tagged<'static, MaybeWeak<Value>> {
    unsafe { Tagged::from_maybe_weak_unchecked(Smi::new(value).encode()) }
}

fn mark_slot_empty(heap: &Heap, array: &WeakFixedArray, index: usize) {
    let previous = array.element_slot(0).get(heap);
    array.set(heap, index, previous);
    array.set(heap, 0, smi_word(index as i64));
}

fn invalidate_one_prototype_validity_cell_internal(heap: &Heap, map: Tagged<'_, Map>) {
    if let Some(cell) = map.as_ref().published_validity_cell(heap) {
        let cell = cell.as_ref();
        if cell.is_valid(heap) {
            cell.clear();
        }
    }
}

fn invalidate_prototype_chains_internal(heap: &Heap, map: Tagged<'_, Map>) {
    let mut current = Some(map);
    let mut next: Option<Tagged<'_, Map>> = None;
    while let Some(m) = current {
        invalidate_one_prototype_validity_cell_internal(heap, m);

        let Some(info) = m.as_ref().try_get_prototype_info(heap) else {
            return;
        };
        let Some(users) = info.as_ref().prototype_users(heap) else {
            return;
        };
        let users = users.as_ref();
        for i in 1..users.len() {
            let Some(user) = users.element_slot(i).get(heap).as_strong() else {
                continue;
            };
            let Some(user_map) = user.get_as::<Map>(heap) else {
                continue;
            };
            if next.is_none() {
                next = Some(user_map);
            } else {
                invalidate_prototype_chains_internal(heap, user_map);
            }
        }
        current = next.take();
    }
}

fn invalidate_prototype_chains(heap: &Heap, map: Tagged<'_, Map>) {
    invalidate_prototype_chains_internal(heap, map);
}
