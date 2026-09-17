use vm::{SlotFlags, SlotName, Smi, Tagged};

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
    let a: Tagged<'_, SlotName> = Tagged::from(Smi::new(1));
    let b: Tagged<'_, SlotName> = Tagged::from(Smi::new(1));
    let c: Tagged<'_, SlotName> = Tagged::from(Smi::new(2));
    assert!(a.ptr_eq(b));
    assert!(!a.ptr_eq(c));
}
