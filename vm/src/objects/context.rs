use core::alloc::Layout;

use crate::{
    EdgeVisitable, FixedArray, GcSlot, Handle, Header, HeapObject, NoGc, ObjectKind, OptionGcSlot,
    Smi, Visitor,
};

// TODO: more info
#[repr(C)]
pub struct ScopeInfo {
    pub header: Header,
    /// parallel to the context's slots
    pub names: GcSlot<FixedArray>,
}

pub struct ScopeInfoInit<'a> {
    pub names: Handle<'a, FixedArray>,
}

impl HeapObject for ScopeInfo {
    const KIND: ObjectKind = ObjectKind::ScopeInfo;
    type Init<'a> = ScopeInfoInit<'a>;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().scope_info_map.as_tagged());
        self.names.set(nogc, host, config.names.as_tagged());
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for ScopeInfo {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.names.as_raw());
    }
}

#[repr(C)]
pub struct Context {
    pub header: Header,
    pub outer: OptionGcSlot<Context>,
    pub slots: GcSlot<FixedArray>,
    pub scope_info: GcSlot<ScopeInfo>,
}

pub struct ContextInit<'a> {
    pub outer: Option<Handle<'a, Context>>,
    pub slots: Handle<'a, FixedArray>,
    pub scope_info: Handle<'a, ScopeInfo>,
}

impl HeapObject for Context {
    const KIND: ObjectKind = ObjectKind::Context;
    type Init<'a> = ContextInit<'a>;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().context_map.as_tagged());
        match config.outer {
            Some(outer) => self.outer.set(nogc, host, outer),
            None => self.outer.clear(nogc.heap()),
        }
        self.slots.set(nogc, host, config.slots.as_tagged());
        self.scope_info
            .set(nogc, host, config.scope_info.as_tagged());
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

impl EdgeVisitable for Context {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
        visitor.visit(self.outer.as_raw());
        visitor.visit(self.slots.as_raw());
        visitor.visit(self.scope_info.as_raw());
    }
}

/// layout `[range_start, range_end, handler_offset]`: a half-open bytecode region
#[repr(C)]
pub struct HandlerEntry {
    pub try_start: GcSlot<Smi>,
    pub try_end: GcSlot<Smi>,
    pub handler_pc: GcSlot<Smi>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct HandlerEntryInit {
    pub try_start: usize,
    pub try_end: usize,
    pub handler_pc: usize,
}

impl HandlerEntryInit {
    pub const fn new(try_start: usize, try_end: usize, handler_pc: usize) -> Self {
        Self {
            try_start,
            try_end,
            handler_pc,
        }
    }
}

#[repr(C)]
pub struct HandlerTable {
    pub header: Header,
    pub size: GcSlot<Smi>,
    pub entries: [HandlerEntry; 0],
}

pub struct HandlerTableInit<'a> {
    pub entries: &'a [HandlerEntryInit],
}

impl HandlerTable {
    pub fn layout_for(entry_count: usize) -> Layout {
        let entries_layout =
            Layout::array::<HandlerEntry>(entry_count).expect("handler table layout");
        Layout::new::<Self>()
            .extend(entries_layout)
            .expect("handler table layout")
            .0
    }

    pub fn len(&self) -> usize {
        self.size.to_smi().value() as usize
    }

    fn entry_ptr(&self) -> *mut HandlerEntry {
        self.entries.as_ptr() as *mut HandlerEntry
    }

    pub fn entry(&self, i: usize) -> HandlerEntryInit {
        debug_assert!(i < self.len());
        let e = unsafe { &*self.entry_ptr().add(i) };
        HandlerEntryInit {
            try_start: e.try_start.to_smi().value() as usize,
            try_end: e.try_end.to_smi().value() as usize,
            handler_pc: e.handler_pc.to_smi().value() as usize,
        }
    }

    pub fn lookup(&self, pc: usize) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for i in 0..self.len() {
            let e = self.entry(i);
            if e.try_start <= pc && pc < e.try_end {
                match best {
                    Some((start, _)) if start >= e.try_start => {}
                    _ => best = Some((e.try_start, e.handler_pc)),
                }
            }
        }
        best.map(|(_, handler_pc)| handler_pc)
    }
}

impl HeapObject for HandlerTable {
    const KIND: ObjectKind = ObjectKind::HandlerTable;
    type Init<'a> = HandlerTableInit<'a>;

    fn layout_for(config: &Self::Init<'_>) -> Layout {
        Self::layout_for(config.entries.len())
    }

    fn init(&mut self, nogc: &NoGc<'_>, config: &Self::Init<'_>) {
        let host = self.erase();
        self.header
            .map
            .set(nogc, host, nogc.known().handler_table_map.as_tagged());
        self.size
            .set(nogc, host, Smi::new(config.entries.len() as i64));
        for (i, e) in config.entries.iter().enumerate() {
            let slot = unsafe { &*self.entry_ptr().add(i) };
            slot.try_start.set(nogc, host, Smi::new(e.try_start as i64));
            slot.try_end.set(nogc, host, Smi::new(e.try_end as i64));
            slot.handler_pc
                .set(nogc, host, Smi::new(e.handler_pc as i64));
        }
    }

    fn header(&self) -> &Header {
        &self.header
    }

    fn layout(&self) -> Layout {
        Self::layout_for(self.len())
    }
}

impl EdgeVisitable for HandlerTable {
    fn visit_edges(&self, visitor: &mut dyn Visitor) {
        visitor.visit(self.header.map.as_raw());
    }
}
