use mark_sweep::{MarkSweep, MarkSweepConfig};
use vm::ic::{Hit, InlineCache};
use vm::{
    FixedArray, HandleSlice, Heap, Object, PropertyDescriptor, SlotName, Smi, Tagged, Thread, VM,
    Value, new_feedback_vector,
};

fn smi(v: i64) -> Value {
    Smi::new(v).encode()
}

/// Safety helper: these tests only pass words that were loaded or
/// allocated under the very heap borrow they are re-anchored at, with no
/// collection in between.
unsafe fn anchored<'a>(heap: &'a Heap, v: Value) -> Tagged<'a, Value> {
    unsafe { v.assume_valid(heap) }
}

fn raw_name(w: Value) -> Tagged<'static, SlotName> {
    unsafe { Tagged::<Value>::from_value_unchecked(w) }.as_name()
}

fn thread() -> (VM, Thread) {
    let vm = VM::new::<MarkSweep, vm::DefaultInterpreter>(MarkSweepConfig::default()).unwrap();
    let thread = vm.attach();
    (vm, thread)
}

fn null_word(thread: &mut Thread) -> Value {
    let heap = thread.heap();
    heap.known().null.as_tagged(heap).raw()
}

fn object_with(thread: &mut Thread, proto: Value, props: &[(&str, i64)]) -> Value {
    thread.handle_scope(|thread, scope| {
        let map = thread.heap().known().object_initial_map;
        let obj = thread
            .heap()
            .new_object(&scope, map, HandleSlice::EMPTY)
            .as_handle(&scope);
        let proto = scope.handle(unsafe { proto.assume_valid(&*thread.heap()) });
        Object::set_prototype(thread.heap(), &scope, obj, proto).unwrap();
        for (name, v) in props {
            let name = thread.intern(&scope, name);
            let name = scope.handle(name.as_tagged(&*thread.heap()));
            let value = scope.handle(Smi::new(*v));
            Object::add_own_property(
                thread.heap(),
                &scope,
                obj,
                name,
                PropertyDescriptor::data(value),
            )
            .unwrap();
        }
        obj.as_tagged(&*thread.heap()).raw()
    })
}

fn slot_name(thread: &mut Thread, name: &str) -> Tagged<'static, SlotName> {
    thread.handle_scope(|thread, scope| {
        let interned = thread.intern(&scope, name);
        let heap = &*thread.heap();
        raw_name(interned.as_tagged(heap).raw())
    })
}

fn parents_of(thread: &mut Thread, p1: Value, p2: Value) -> Value {
    thread.handle_scope(|thread, scope| {
        let first = thread.intern(&scope, "parent");
        let second = thread.intern(&scope, "mixin");
        let (first, second) = {
            let heap = &*thread.heap();
            (first.as_tagged(heap).raw(), second.as_tagged(heap).raw())
        };
        let arr = thread.heap().allocate_handle::<FixedArray>(
            scope.stage(&[
                unsafe { Tagged::<Value>::from_value_unchecked(first) },
                unsafe { Tagged::from_value_unchecked(p1) },
                unsafe { Tagged::<Value>::from_value_unchecked(second) },
                unsafe { Tagged::from_value_unchecked(p2) },
            ]),
            &scope,
        );
        let heap = &*thread.heap();
        arr.as_tagged(heap).raw()
    })
}

/// A parent-name slot (the pair list exposes each parent under its own
/// name, read-only) is cached as `KIND_PARENT` and reads the parent object
/// out of the map's pair list.
#[test]
fn multi_parent_name_load_returns_the_parent() {
    let (_vm, mut thread) = thread();
    let null = null_word(&mut thread);
    let p1 = object_with(&mut thread, null, &[]);
    let p2 = object_with(&mut thread, null, &[]);
    let protos = parents_of(&mut thread, p1, p2);
    let home = object_with(&mut thread, protos, &[]);
    let name = slot_name(&mut thread, "parent");

    thread.handle_scope(|thread, scope| {
        let vector = new_feedback_vector(thread.heap(), &scope, 2).expect("feedback slots");
        {
            let home_h = scope.handle(
                unsafe { anchored(&*thread.heap(), home) }
                    .as_heap_object()
                    .unwrap(),
            );
            InlineCache::update_load(
                thread.heap(),
                &scope,
                Some(vector),
                0,
                Some(home_h),
                scope.handle(name),
                false,
            );
        }
        {
            let heap = &*thread.heap();
            let hit = InlineCache::try_load(
                heap,
                Some(vector.as_tagged(heap)),
                0,
                unsafe { anchored(heap, home) }.erase(),
            );
            assert!(
                matches!(hit, Some(Hit::Value(v)) if v.raw() == p1),
                "the parent-name slot must return the first parent object"
            );
        }
    });
}

/// A cached miss through a Self-style parent list must be invalidated when
/// a parent gains the property.
#[test]
fn multi_parent_cached_miss_invalidates_when_parent_gains_property() {
    let (_vm, mut thread) = thread();
    let null = null_word(&mut thread);
    let p1 = object_with(&mut thread, null, &[]);
    let p2 = object_with(&mut thread, null, &[]);
    let protos = parents_of(&mut thread, p1, p2);
    let home = object_with(&mut thread, protos, &[]);
    let y = slot_name(&mut thread, "y");

    thread.handle_scope(|thread, scope| {
        let vector = new_feedback_vector(thread.heap(), &scope, 2).expect("feedback slots");
        {
            let home_h = scope.handle(
                unsafe { anchored(&*thread.heap(), home) }
                    .as_heap_object()
                    .unwrap(),
            );
            InlineCache::update_load(
                thread.heap(),
                &scope,
                Some(vector),
                0,
                Some(home_h),
                scope.handle(y),
                true,
            );
        }
        {
            let heap = &*thread.heap();
            let hit = InlineCache::try_load(
                heap,
                Some(vector.as_tagged(heap)),
                0,
                unsafe { anchored(heap, home) }.erase(),
            );
            assert!(matches!(hit, Some(Hit::NotFound)), "the miss is cached");
        }

        {
            let name = thread.intern(&scope, "y");
            let name = scope.handle(name.as_tagged(&*thread.heap()));
            let value = scope.handle(Smi::new(9));
            let p2_h = scope.handle(
                unsafe { anchored(&*thread.heap(), p2) }
                    .as_heap_object()
                    .unwrap(),
            );
            Object::add_own_property(
                thread.heap(),
                &scope,
                p2_h,
                name,
                PropertyDescriptor::data(value),
            )
            .unwrap();
        }

        {
            let heap = &*thread.heap();
            let hit = InlineCache::try_load(
                heap,
                Some(vector.as_tagged(heap)),
                0,
                unsafe { anchored(heap, home) }.erase(),
            );
            assert!(hit.is_none(), "the cached miss must be invalidated");
        }
    });
}

/// A load cached through a Self-style parent list must be invalidated when
/// *any* parent in the closure changes shape, even the higher-priority one
/// that did not hold the property.
#[test]
fn multi_parent_ic_invalidates_on_parent_shape_change() {
    let (_vm, mut thread) = thread();
    let null = null_word(&mut thread);
    let p1 = object_with(&mut thread, null, &[]);
    let p2 = object_with(&mut thread, null, &[("x", 42)]);
    let protos = parents_of(&mut thread, p1, p2);
    let home = object_with(&mut thread, protos, &[]);
    let x = slot_name(&mut thread, "x");

    thread.handle_scope(|thread, scope| {
        let vector = new_feedback_vector(thread.heap(), &scope, 2).expect("feedback slots");

        // Record a load that resolves through the second parent.
        {
            let home_h = scope.handle(
                unsafe { anchored(&*thread.heap(), home) }
                    .as_heap_object()
                    .unwrap(),
            );
            InlineCache::update_load(
                thread.heap(),
                &scope,
                Some(vector),
                0,
                Some(home_h),
                scope.handle(x),
                false,
            );
        }
        {
            let heap = &*thread.heap();
            let hit = InlineCache::try_load(
                heap,
                Some(vector.as_tagged(heap)),
                0,
                unsafe { anchored(heap, home) }.erase(),
            );
            assert!(
                matches!(hit, Some(Hit::Value(v)) if v.raw() == smi(42)),
                "cached load must resolve through the second parent"
            );
        }

        // Give the higher-priority parent the same property: the closure
        // changed even though the receiver's own map did not.
        {
            let name = thread.intern(&scope, "x");
            let name = scope.handle(name.as_tagged(&*thread.heap()));
            let value = scope.handle(Smi::new(7));
            let p1_h = scope.handle(
                unsafe { anchored(&*thread.heap(), p1) }
                    .as_heap_object()
                    .unwrap(),
            );
            Object::add_own_property(
                thread.heap(),
                &scope,
                p1_h,
                name,
                PropertyDescriptor::data(value),
            )
            .unwrap();
        }

        {
            let heap = &*thread.heap();
            let hit = InlineCache::try_load(
                heap,
                Some(vector.as_tagged(heap)),
                0,
                unsafe { anchored(heap, home) }.erase(),
            );
            assert!(
                hit.is_none(),
                "a parent shape change must invalidate the cached handler"
            );
        }

        // Re-record and observe the shadowing value.
        {
            let home_h = scope.handle(
                unsafe { anchored(&*thread.heap(), home) }
                    .as_heap_object()
                    .unwrap(),
            );
            InlineCache::update_load(
                thread.heap(),
                &scope,
                Some(vector),
                0,
                Some(home_h),
                scope.handle(x),
                false,
            );
        }
        {
            let heap = &*thread.heap();
            let hit = InlineCache::try_load(
                heap,
                Some(vector.as_tagged(heap)),
                0,
                unsafe { anchored(heap, home) }.erase(),
            );
            assert!(
                matches!(hit, Some(Hit::Value(v)) if v.raw() == smi(7)),
                "the re-recorded handler must see the shadowing value"
            );
        }
    });
}
