use std::sync::atomic::{AtomicU64, Ordering};

/// Address-indexed atomic bitmap over `[base, base + size)`
/// Addresses must be granule-aligned.
pub struct Bitmap {
    base: usize,
    size: usize,
    shift: u32,
    words: Box<[AtomicU64]>,
}

impl Bitmap {
    pub fn new(base: usize, size: usize, granularity: usize) -> Self {
        assert!(
            granularity.is_power_of_two(),
            "granularity must be a power of two"
        );
        debug_assert_eq!(base % granularity, 0, "base must be granule-aligned");
        let shift = granularity.trailing_zeros();
        let bits = size.div_ceil(granularity);
        let words = (0..bits.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
        Self {
            base,
            size,
            shift,
            words,
        }
    }

    pub fn base(&self) -> usize {
        self.base
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn granularity(&self) -> usize {
        1 << self.shift
    }

    /// Sets the bit at `addr`. Returns `true` if this call set it, `false`
    /// if it was already set.
    pub fn set(&self, addr: usize) -> bool {
        let (word, bit) = self.index(addr);
        let mask = 1u64 << bit;
        self.words[word].fetch_or(mask, Ordering::AcqRel) & mask == 0
    }

    pub fn clear(&self, addr: usize) {
        let (word, bit) = self.index(addr);
        self.words[word].fetch_and(!(1u64 << bit), Ordering::AcqRel);
    }

    pub fn is_set(&self, addr: usize) -> bool {
        let (word, bit) = self.index(addr);
        self.words[word].load(Ordering::Acquire) & (1u64 << bit) != 0
    }

    pub fn clear_all(&self) {
        for word in &self.words {
            word.store(0, Ordering::Release);
        }
    }

    pub fn iter_set(&self) -> SetIter<'_> {
        SetIter {
            bitmap: self,
            word: 0,
            bits: 0,
        }
    }

    fn index(&self, addr: usize) -> (usize, u32) {
        debug_assert!(
            addr >= self.base && addr < self.base + self.size,
            "address {addr:#x} outside bitmap range"
        );
        debug_assert_eq!(
            addr & ((1 << self.shift) - 1),
            0,
            "address {addr:#x} not granule-aligned"
        );
        let bit = (addr - self.base) >> self.shift;
        ((bit >> 6) as usize, (bit & 63) as u32)
    }
}

pub struct SetIter<'a> {
    bitmap: &'a Bitmap,
    word: usize,
    bits: u64,
}

impl Iterator for SetIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if self.bits != 0 {
                let bit = self.bits.trailing_zeros() as usize;
                self.bits &= self.bits - 1;
                let index = ((self.word - 1) << 6) + bit;
                return Some(self.bitmap.base + (index << self.bitmap.shift));
            }
            if self.word >= self.bitmap.words.len() {
                return None;
            }
            self.bits = self.bitmap.words[self.word].load(Ordering::Acquire);
            self.word += 1;
        }
    }
}
