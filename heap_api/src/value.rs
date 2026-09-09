pub type Word = u64;

pub const TAG_MASK: Word = 0b11;
pub const PTR_BIT: Word = 0b01;
pub const WEAK_BIT: Word = 0b10;

pub const TAG_SMI: Word = 0b0;
pub const STRONG_PTR: Word = 0b01;
pub const WEAK_PTR: Word = 0b11;

pub const CLEARED: Word = WEAK_PTR;
