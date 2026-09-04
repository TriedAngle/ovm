use vm::{SlotFlags, SlotKind, SlotName, Tagged};

#[test]
fn slot_flags_kind_decoding() {
    assert_eq!(SlotFlags::VALUE.kind(), SlotKind::Value);
    assert_eq!(SlotFlags::CONST.kind(), SlotKind::Const);
    assert_eq!(SlotFlags::ACCESSOR.kind(), SlotKind::Accessor);
    // attribute bits don't disturb the kind
    let f = SlotFlags::CONST
        .union(SlotFlags::WRITABLE)
        .union(SlotFlags::ENUMERABLE);
    assert_eq!(f.kind(), SlotKind::Const);
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
