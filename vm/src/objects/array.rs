use core::alloc::Layout;

use crate::{
    EdgeVisitable, GcSlot, HandleSlice, Header, Heap, HeapObject, MaybeWeak, MaybeWeakGcSlot,
    ObjectKind, Smi, Tagged, Value, Visitor,
};

#[repr(C)]
pub struct FixedArray<T = Value> {
    pub header: Header,
    pub size: GcSlot<Smi>,
    pub values: [GcSlot<T>; 0],
}

impl<T: 'static> FixedArray<T> {
    pub fn layout_for(len: usize) -> Layout {
        let values_layout = Layout::array::<GcSlot<T>>(len).expect("values layout");
        Layout::new::<Self>()
            .extend(values_layout)
            .expect("array layout")
            .0
    }

    pub fn len(&self) -> usize {
        self.size.to_smi().value() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn data_ptr(&self) -> *mut GcSlot<T> {
        self.values.as_ptr() as *mut GcSlot<T>
    }

    pub fn as_slice(&self) -> &[GcSlot<T>] {
        unsafe { core::slice::from_raw_parts(self.data_ptr(), self.len()) }
    }

    pub fn element_slot(&self, i: usize) -> &GcSlot<T> {
        debug_assert!(i < self.len());
        unsafe { &*self.values.as_ptr().add(i) }
    }

    pub fn at<'a>(&self, heap: &'a Heap, i: usize) -> Tagged<'a, T> {
        self.element_slot(i).get(heap)
    }

    pub fn set<'a>(&self, heap: &Heap, i: usize, v: impl Into<Tagged<'a, T>>)
    where
        T: 'a,
    {
        self.element_slot(i).set(heap, self.erase(), v);
    }
}

impl<T: 'static> HeapObject for FixedArray<T> {
    const KIND: ObjectKind = ObjectKind::FixedArray;
    type Init<'a> = HandleSlice<'a>;

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.len())
    }

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(heap, host, heap.known().array_map.as_tagged(heap));
        self.size.set(heap, host, Smi::new(config.len() as i64));
        // Safety: copying words out of rooted memory during init; no GC
        // can run before the new array is rooted by the caller.
        for (i, v) in config.iter().map(|h| h.as_tagged(heap)).enumerate() {
            self.element_slot(i).as_raw().store_raw(v.raw().to_bits());
        }
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.len())
    }
}

impl<T: 'static> EdgeVisitable for FixedArray<T> {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        let size = self.size.to_smi().value() as usize;
        for i in 0..size {
            visitor.visit(self.element_slot(i).as_raw());
        }
    }
}

#[repr(C)]
pub struct WeakFixedArray<T = Value> {
    pub header: Header,
    pub size: GcSlot<Smi>,
    pub values: [MaybeWeakGcSlot<T>; 0],
}

pub struct WeakFixedArrayInit<'a, T = Value> {
    pub values: &'a [Tagged<'a, MaybeWeak<T>>],
}

impl<T: 'static> WeakFixedArray<T> {
    pub fn layout_for(len: usize) -> Layout {
        let values_layout = Layout::array::<MaybeWeakGcSlot<T>>(len).expect("values layout");
        Layout::new::<Self>()
            .extend(values_layout)
            .expect("weak array layout")
            .0
    }

    pub fn len(&self) -> usize {
        self.size.to_smi().value() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn data_ptr(&self) -> *mut MaybeWeakGcSlot<T> {
        self.values.as_ptr() as *mut MaybeWeakGcSlot<T>
    }

    pub fn as_slice(&self) -> &[MaybeWeakGcSlot<T>] {
        unsafe { core::slice::from_raw_parts(self.data_ptr(), self.len()) }
    }

    pub fn element_slot(&self, i: usize) -> &MaybeWeakGcSlot<T> {
        debug_assert!(i < self.len());
        unsafe { &*self.values.as_ptr().add(i) }
    }

    pub fn get<'a>(&self, heap: &'a Heap, i: usize) -> Tagged<'a, MaybeWeak<T>> {
        self.element_slot(i).get(heap)
    }

    pub fn set_strong<'a>(&self, heap: &Heap, i: usize, v: impl Into<Tagged<'a, T>>)
    where
        T: 'a,
    {
        self.element_slot(i).set_strong(heap, self.erase(), v);
    }

    pub fn set_weak<'a>(&self, heap: &Heap, i: usize, v: impl Into<Tagged<'a, T>>)
    where
        T: 'a,
    {
        self.element_slot(i).set_weak(heap, v);
    }
}

impl<T: 'static> HeapObject for WeakFixedArray<T> {
    const KIND: ObjectKind = ObjectKind::FixedArray;
    type Init<'a> = WeakFixedArrayInit<'a, T>;

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.values.len())
    }

    fn init(&mut self, heap: &Heap, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(heap, host, heap.known().array_map.as_tagged(heap));
        self.size
            .set(heap, host, Smi::new(config.values.len() as i64));
        for (i, v) in config.values.iter().enumerate() {
            self.element_slot(i).as_raw().store_raw(v.raw().to_bits());
        }
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.len())
    }
}

impl<T: 'static> EdgeVisitable for WeakFixedArray<T> {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        let size = self.size.to_smi().value() as usize;
        for i in 0..size {
            visitor.visit(self.element_slot(i).as_raw());
        }
    }
}
