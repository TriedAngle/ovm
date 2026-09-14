pub mod heap;
pub mod value;

pub use heap::{
    AllocError, GcHost, HeapBackend, HeapStats, LocalHeap, RawCell, SharedHeap, Visitor,
};
pub use value::{CLEARED, PTR_BIT, STRONG_PTR, TAG_MASK, TAG_SMI, WEAK_BIT, WEAK_PTR, Word};
