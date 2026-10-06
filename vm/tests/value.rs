use core::alloc::Layout;

use vm::{Header, Heap, HeapObject, STRONG_PTR, Smi, Tagged, Value, WEAK_PTR};

/// Stand-in heap object, aligned like a real heap allocation.
#[repr(align(8))]
struct TestObj;

impl HeapObject for TestObj {
    type Init<'a> = ();

    const KIND: vm::ObjectKind = vm::ObjectKind::DenseString;

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, _heap: &Heap, _config: &Self::Init<'_>) {
        unimplemented!("TestObj is boxed, never heap-allocated by these tests")
    }

    fn header(&self) -> &Header {
        unimplemented!("TestObj carries no header; never called by these tests")
    }

    fn layout(&self) -> Layout {
        Layout::new::<Self>()
    }
}

fn alloc_test_obj() -> *mut TestObj {
    Box::into_raw(Box::new(TestObj))
}

unsafe fn free_test_obj(ptr: *mut TestObj) {
    drop(unsafe { Box::from_raw(ptr) });
}

/// The strong pointer word for a stand-in object.
fn strong(raw: *mut TestObj) -> Value {
    Value::from_bits(raw as u64 | STRONG_PTR)
}

mod value {
    use super::*;

    #[test]
    fn bits_roundtrip() {
        for bits in [0, 1, 2, 3, u64::MAX] {
            // Safety: bit-level test; the values are never dereferenced.
            assert_eq!(Value::from_bits(bits).to_bits(), bits);
        }
    }

    #[test]
    fn raw_addr_clears_tag_bits() {
        // Safety: bit-level test; the values are never dereferenced.
        {
            assert_eq!(Value::from_bits(0x1000).raw_addr(), 0x1000);
            assert_eq!(Value::from_bits(0x1001).raw_addr(), 0x1000);
            assert_eq!(Value::from_bits(0x1002).raw_addr(), 0x1000);
            assert_eq!(Value::from_bits(0x1003).raw_addr(), 0x1000);
        }
    }

    #[test]
    fn classification_of_smi() {
        let v = Smi::new(42).encode();
        assert!(v.is_smi());
        assert!(!v.is_ptr());
        assert!(!v.is_strong_ptr());
        assert!(!v.is_weak_ptr());
    }
}

mod smi {
    use super::*;

    #[test]
    fn range_bounds_are_inclusive() {
        assert!(Smi::in_range(Smi::MAX));
        assert!(Smi::in_range(Smi::MIN));
        assert!(!Smi::in_range(Smi::MAX + 1));
        assert!(!Smi::in_range(Smi::MIN - 1));
    }

    #[test]
    fn min_is_negative_of_max_plus_one() {
        // Two's-complement range: exactly one more negative than positive.
        assert_eq!(Smi::MIN, -Smi::MAX - 1);
    }

    #[test]
    fn encode_decode_roundtrip() {
        for n in [0, 1, -1, 42, -42, Smi::MAX, Smi::MIN] {
            let smi = Smi::new(n);
            let decoded = Smi::decode(smi.encode()).unwrap();
            assert_eq!(decoded, smi);
            assert_eq!(decoded.value(), n);
        }
    }

    #[test]
    fn decode_rejects_pointers() {
        // Safety: bit-level test; the values are never dereferenced.
        let (strong, weak) = (
            Value::from_bits(0x1000 | STRONG_PTR),
            Value::from_bits(0x1000 | WEAK_PTR),
        );
        assert_eq!(Smi::decode(strong), None);
        assert_eq!(Smi::decode(weak), None);
    }
}

mod debug {
    use super::*;

    #[test]
    fn value_formats_smi() {
        assert_eq!(format!("{:?}", Smi::new(-7).encode()), "Value(Smi(-7))");
        assert_eq!(format!("{:?}", Smi::new(0).encode()), "Value(Smi(0))");
    }

    #[test]
    fn value_formats_pointers() {
        // Safety: bit-level test; the values are never dereferenced.
        let (strong, weak) = (
            Value::from_bits(0x1000 | STRONG_PTR),
            Value::from_bits(0x1000 | WEAK_PTR),
        );
        assert_eq!(format!("{:?}", strong), "Value(Strong(0x1000))");
        assert_eq!(format!("{:?}", weak), "Value(Weak(0x1000))");
    }

    #[test]
    fn smi_formats_inner_value() {
        assert_eq!(format!("{:?}", Smi::new(42)), "Smi(42)");
    }
}

mod tagged {
    use super::*;

    #[test]
    fn smi_constructor_checks_range() {
        assert!(Tagged::<Smi>::smi(Smi::MAX).is_some());
        assert!(Tagged::<Smi>::smi(Smi::MAX + 1).is_none());
    }

    #[test]
    fn from_smi_roundtrip() {
        let smi = Smi::new(-3);
        let tagged = Tagged::<Smi>::from_smi(smi);

        assert!(tagged.is_smi());
        assert!(!tagged.is_ptr());
        assert_eq!(tagged.to_smi(), Some(smi));
    }

    #[test]
    fn to_smi_rejects_pointers() {
        // Safety: bit-level test; the value is never dereferenced.
        let word = Value::from_bits(0x1000 | STRONG_PTR);
        let tagged = unsafe { Tagged::<Smi>::from_value_unchecked(word) };

        assert_eq!(tagged.to_smi(), None);
    }

    #[test]
    fn strong_word_roundtrip() {
        let raw = alloc_test_obj();
        let tagged = unsafe { Tagged::<TestObj>::from_value_unchecked(strong(raw)) };

        assert!(tagged.is_ptr());
        assert!(tagged.is_strong_ptr());
        assert!(!tagged.is_weak_ptr());
        assert!(!tagged.is_smi());
        assert_eq!(tagged.raw().raw_addr(), raw as u64);

        unsafe { free_test_obj(raw) };
    }

    #[test]
    fn maybe_weak_roundtrip() {
        let raw = alloc_test_obj();
        let strong = unsafe { Tagged::<TestObj>::from_value_unchecked(strong(raw)) };

        let weak = strong.as_weak();
        assert!(weak.raw().is_weak_ptr());
        assert!(!weak.raw().is_strong_ptr());
        assert_eq!(weak.raw().raw_addr(), raw as u64);
        assert_eq!(weak.as_strong().unwrap().raw(), strong.raw());

        let maybe = strong.as_maybe_weak();
        assert!(!maybe.is_cleared());
        assert_eq!(maybe.as_strong().unwrap().raw(), strong.raw());
        assert_eq!(maybe.raw(), strong.raw());

        unsafe { free_test_obj(raw) };
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "weak value in a strong Tagged")]
    fn strong_tagged_rejects_weak_bits() {
        // Safety: bit-level test; the value is never dereferenced.
        let weak = Value::from_bits(0x1000 | WEAK_PTR);
        let _ = unsafe { Tagged::<TestObj>::from_value_unchecked(weak) };
    }

    #[test]
    fn erase_recovers_the_raw_value() {
        let raw = alloc_test_obj();
        let tagged = unsafe { Tagged::<TestObj>::from_value_unchecked(strong(raw)) };

        assert_eq!(tagged.raw(), strong(raw));

        unsafe { free_test_obj(raw) };
    }

    #[test]
    fn erase_and_into_preserve_bits() {
        let tagged = Tagged::<Smi>::smi(7).unwrap();

        let erased: Value = tagged.raw();
        let re_tagged = Tagged::<Value>::try_smi(erased).unwrap();
        assert_eq!(re_tagged.raw().to_bits(), tagged.raw().to_bits());
        assert!(re_tagged.is_smi());
    }

    #[test]
    fn ptr_eq_compares_bits() {
        let a = Tagged::<Smi>::smi(1).unwrap();
        let b = Tagged::<Smi>::smi(1).unwrap();
        let c = Tagged::<Smi>::smi(2).unwrap();

        assert!(a.ptr_eq(b));
        assert!(!a.ptr_eq(c));
    }

    #[test]
    fn debug_formats_type_and_value() {
        let tagged = Tagged::<Smi>::smi(3).unwrap();
        let s = format!("{:?}", tagged);

        assert!(s.starts_with("Tagged("));
        assert!(s.contains("Smi"));
        assert!(s.contains("Value(Smi(3))"));
    }
}
