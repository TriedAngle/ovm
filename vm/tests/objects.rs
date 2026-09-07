use vm::{SlotFlags, SlotName, Tagged};

#[test]
fn slot_flags_accessor_bit() {
    assert!(!SlotFlags::VALUE.is_accessor());
    assert!(SlotFlags::ACCESSOR.is_accessor());
    // attribute bits don't disturb the accessor bit
    let f = SlotFlags::VALUE
        .union(SlotFlags::WRITABLE)
        .union(SlotFlags::ENUMERABLE);
    assert!(!f.is_accessor());
}

#[test]
fn slot_flags_attributes() {
    let f = SlotFlags::VALUE
        .union(SlotFlags::WRITABLE)
        .union(SlotFlags::ENUMERABLE);
    assert!(f.is_writable());
    assert!(f.is_enumerable());
    assert!(!f.is_configurable());
}

#[test]
fn slot_name_equality_by_bits() {
    let a = SlotName::from(Tagged::smi(1).unwrap());
    let b = SlotName::from(Tagged::smi(1).unwrap());
    let c = SlotName::from(Tagged::smi(2).unwrap());
    assert_eq!(a, b);
    assert_ne!(a, c);
}
