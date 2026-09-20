use core::alloc::Layout;

use crate::{
    EdgeVisitable, GcSlot, Handle, HandleScope, Header, Heap, HeapObject, Map, MaybeWeak,
    MaybeWeakGcSlot, ObjectKind, Smi, Tagged, Value, Visitor, WeakFixedArray,
};


/// `[feedback, feedback_extra]` and encodes its IC state in
/// the base slot's contents:
///
/// - the hole — uninitialized
/// - weak `Map` — monomorphic; the extra slot holds the handler
/// - strong `WeakFixedArray` — polymorphic; flat `[weak map, handler, ...]`
/// - megamorphic symbol — always take the slow path
#[repr(C)]
pub struct FeedbackVector {
    pub header: Header,
    pub length: GcSlot<Smi>,
    pub slots: [MaybeWeakGcSlot; 0],
}

pub struct FeedbackVectorInit {
    /// Initial word of every slot (the uninitialized hole sentinel).
    pub fill: Value,
    pub length: usize,
}

impl FeedbackVector {
    pub fn layout_for(len: usize) -> Layout {
        let slots_layout = Layout::array::<MaybeWeakGcSlot>(len).expect("feedback slots layout");
        Layout::new::<Self>()
            .extend(slots_layout)
            .expect("feedback vector layout")
            .0
    }

    pub fn len(&self) -> usize {
        self.length.to_smi().value() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn data_ptr(&self) -> *mut MaybeWeakGcSlot {
        self.slots.as_ptr() as *mut MaybeWeakGcSlot
    }

    pub fn slot(&self, i: usize) -> &MaybeWeakGcSlot {
        debug_assert!(i < self.len());
        unsafe { &*self.data_ptr().add(i) }
    }

    /// The raw word in slot `i`: a Smi, strong or weak heap reference.
    pub fn inner(&self, i: usize) -> Value {
        self.slot(i).inner()
    }
}

impl HeapObject for FeedbackVector {
    const KIND: ObjectKind = ObjectKind::FeedbackVector;
    type Init<'a> = FeedbackVectorInit;

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.length)
    }

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(heap, host, heap.known().feedback_vector_map.as_tagged(heap));
        self.length.set(heap, host, Smi::new(config.length as i64));
        for i in 0..config.length {
            self.slot(i).as_raw().store_raw(config.fill.to_bits());
        }
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.len())
    }
}

impl EdgeVisitable for FeedbackVector {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        for i in 0..self.len() {
            visitor.visit(self.slot(i).as_raw());
        }
    }
}

impl FeedbackVector {
    /// The `[state, handler]` pair of site `slot`, bounds-checked.
    pub fn site(&self, slot: usize) -> Option<(&MaybeWeakGcSlot, &MaybeWeakGcSlot)> {
        if slot + 1 >= self.len() {
            return None;
        }
        Some((self.slot(slot), self.slot(slot + 1)))
    }

    /// Go monomorphic: `weak map` in the state slot, `handler` in the
    /// handler slot.
    pub fn set_mono(
        &self,
        heap: &Heap,
        slot: usize,
        map: Tagged<'_, Map>,
        handler: Tagged<'_, MaybeWeak<Value>>,
    ) {
        let host = self.erase();
        self.slot(slot).set_weak(heap, host, map.erase());
        Self::store_word(self.slot(slot + 1), heap, host, handler);
    }

    /// Overwrite the handler half of a site's pair.
    pub fn set_handler(&self, heap: &Heap, slot: usize, handler: Tagged<'_, MaybeWeak<Value>>) {
        Self::store_word(self.slot(slot + 1), heap, self.erase(), handler);
    }

    /// Point the state slot at a polymorphic pair array.
    pub fn set_poly(&self, heap: &Heap, slot: usize, pairs: Tagged<'_, WeakFixedArray>) {
        let host = self.erase();
        self.slot(slot).set_strong(heap, host, pairs.erase());
    }

    pub fn set_megamorphic(&self, heap: &Heap, slot: usize) {
        let host = self.erase();
        self.slot(slot).set_strong(
            heap,
            host,
            heap.known().megamorphic_symbol.as_tagged(heap).erase(),
        );
    }

    /// Smis and strong pointers store strong; weak map words store weak
    /// (both take the barrier).
    fn store_word(
        slot: &MaybeWeakGcSlot,
        heap: &Heap,
        host: Value,
        word: Tagged<'_, MaybeWeak<Value>>,
    ) {
        if let Some(strong) = word.strengthen() {
            slot.set_strong(heap, host, strong);
        } else if let Some(live) = word.upgrade() {
            slot.set_weak(heap, host, live);
        }
    }
}

/// Allocate a hole-filled feedback vector; `None` when the function needs no
/// slots.
pub fn new_feedback_vector<'s>(
    heap: &mut Heap,
    scope: &'s HandleScope<'_>,
    slot_count: usize,
) -> Option<Handle<'s, FeedbackVector>> {
    if slot_count == 0 {
        return None;
    }
    // Safety: root-slot read of a permanent well-known object.
    let fill = unsafe { heap.known().the_hole.read_unchecked() };
    Some(heap.allocate_handle::<FeedbackVector>(
        FeedbackVectorInit {
            fill,
            length: slot_count,
        },
        scope,
    ))
}
