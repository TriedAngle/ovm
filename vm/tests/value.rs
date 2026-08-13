use core::alloc::Layout;

use vm::{Header, HeapObject, HeapPtr, Smi, Tagged, Value};

/// Stand-in heap object, aligned like a real heap allocation.
#[repr(align(8))]
struct TestObj(u64);

impl HeapObject for TestObj {
    type Init<'a> = ();

    fn layout_for(_config: &Self::Init<'_>) -> Layout {
        Layout::new::<Self>()
    }

    fn init(&mut self, _heap: &impl vm::LocalHeap, _config: &Self::Init<'_>) {
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
    Box::into_raw(Box::new(TestObj(0xDEAD_BEEF)))
}

unsafe fn free_test_obj(ptr: *mut TestObj) {
    drop(unsafe { Box::from_raw(ptr) });
}

mod value {
    use super::*;

    #[test]
    fn bits_roundtrip() {
        for bits in [0, 1, 2, 3, u64::MAX] {
            assert_eq!(Value::from_bits(bits).to_bits(), bits);
        }
    }

    #[test]
    fn raw_addr_clears_tag_bits() {
        assert_eq!(Value::from_bits(0x1000).raw_addr(), 0x1000);
        assert_eq!(Value::from_bits(0x1001).raw_addr(), 0x1000);
        assert_eq!(Value::from_bits(0x1002).raw_addr(), 0x1000);
        assert_eq!(Value::from_bits(0x1003).raw_addr(), 0x1000);
    }

    #[test]
    fn classification_of_smi() {
        let v = Smi::new_unchecked(42).encode();
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
        assert!(Smi::new(Smi::MAX).is_some());
        assert!(Smi::new(Smi::MIN).is_some());
        assert!(Smi::new(Smi::MAX + 1).is_none());
        assert!(Smi::new(Smi::MIN - 1).is_none());
    }

    #[test]
    fn min_is_negative_of_max_plus_one() {
        // Two's-complement range: exactly one more negative than positive.
        assert_eq!(Smi::MIN, -Smi::MAX - 1);
    }

    #[test]
    fn encode_decode_roundtrip() {
        for n in [0, 1, -1, 42, -42, Smi::MAX, Smi::MIN] {
            let smi = Smi::new(n).unwrap();
            let decoded = Smi::decode(smi.encode()).unwrap();
            assert_eq!(decoded, smi);
            assert_eq!(decoded.value(), n);
        }
    }

    #[test]
    fn decode_rejects_pointers() {
        let strong = Value::from_bits(0x1000 | vm::value::STRONG_PTR);
        let weak = Value::from_bits(0x1000 | vm::value::WEAK_PTR);
        assert_eq!(Smi::decode(strong), None);
        assert_eq!(Smi::decode(weak), None);
    }
}

mod heap_ptr {
    use super::*;

    #[test]
    fn strong_pointer_roundtrip() {
        let raw = alloc_test_obj();
        let ptr = unsafe { HeapPtr::new_unchecked(raw) };

        let v = ptr.encode_strong();
        assert!(v.is_strong_ptr());
        assert!(v.is_ptr());
        assert!(!v.is_smi());
        assert!(!v.is_weak_ptr());
        assert_eq!(v.raw_addr(), raw as u64);

        let decoded = HeapPtr::<TestObj>::decode_strong(v).unwrap();
        assert_eq!(decoded.as_ptr(), raw);

        unsafe { free_test_obj(raw) };
    }

    #[test]
    fn weak_pointer_roundtrip() {
        let raw = alloc_test_obj();
        let ptr = unsafe { HeapPtr::new_unchecked(raw) };

        let v = ptr.encode_weak();
        assert!(v.is_weak_ptr());
        assert!(v.is_ptr());
        assert!(!v.is_smi());
        assert!(!v.is_strong_ptr());
        assert_eq!(v.raw_addr(), raw as u64);

        assert!(HeapPtr::<TestObj>::decode_strong(v).is_none());
        let decoded = HeapPtr::<TestObj>::decode(v).unwrap();
        assert_eq!(decoded.as_ptr(), raw);

        unsafe { free_test_obj(raw) };
    }

    #[test]
    fn decode_rejects_smi() {
        let v = Smi::new_unchecked(1).encode();
        assert!(HeapPtr::<TestObj>::decode(v).is_none());
        assert!(HeapPtr::<TestObj>::decode_strong(v).is_none());
    }

    #[test]
    fn as_ref_reads_pointee() {
        let raw = alloc_test_obj();
        let ptr = unsafe { HeapPtr::new_unchecked(raw) };

        let obj = unsafe { ptr.as_ref() };
        assert_eq!(obj.0, 0xDEAD_BEEF);

        unsafe { free_test_obj(raw) };
    }

    #[test]
    fn as_mut_writes_pointee() {
        let raw = alloc_test_obj();
        let ptr = unsafe { HeapPtr::new_unchecked(raw) };

        unsafe { ptr.as_mut() }.0 = 42;
        assert_eq!(unsafe { ptr.as_ref() }.0, 42);

        unsafe { free_test_obj(raw) };
    }
}

mod debug {
    use super::*;

    #[test]
    fn value_formats_smi() {
        assert_eq!(
            format!("{:?}", Smi::new_unchecked(-7).encode()),
            "Value(Smi(-7))"
        );
        assert_eq!(
            format!("{:?}", Smi::new_unchecked(0).encode()),
            "Value(Smi(0))"
        );
    }

    #[test]
    fn value_formats_pointers() {
        let strong = Value::from_bits(0x1000 | vm::value::STRONG_PTR);
        let weak = Value::from_bits(0x1000 | vm::value::WEAK_PTR);
        assert_eq!(format!("{:?}", strong), "Value(Strong(0x1000))");
        assert_eq!(format!("{:?}", weak), "Value(Weak(0x1000))");
    }

    #[test]
    fn smi_formats_inner_value() {
        assert_eq!(format!("{:?}", Smi::new_unchecked(42)), "Smi(42)");
    }

    #[test]
    fn heap_ptr_formats_address() {
        let raw = alloc_test_obj();
        let ptr = unsafe { HeapPtr::<TestObj>::new_unchecked(raw) };

        assert_eq!(format!("{:?}", ptr), format!("HeapPtr({:#x})", raw as u64));

        unsafe { free_test_obj(raw) };
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
        let smi = Smi::new_unchecked(-3);
        let tagged = Tagged::<Smi>::from_smi(smi);

        assert!(tagged.is_smi());
        assert!(!tagged.is_ptr());
        assert_eq!(tagged.to_smi(), Some(smi));
    }

    #[test]
    fn to_smi_rejects_pointers() {
        let raw = alloc_test_obj();
        let ptr = unsafe { HeapPtr::<TestObj>::new_unchecked(raw) };
        let tagged = unsafe { Tagged::<Smi>::from_value_unchecked(ptr.encode_strong()) };

        assert_eq!(tagged.to_smi(), None);

        unsafe { free_test_obj(raw) };
    }

    #[test]
    fn from_ptr_roundtrip() {
        let raw = alloc_test_obj();
        let ptr = unsafe { HeapPtr::<TestObj>::new_unchecked(raw) };
        let tagged = Tagged::from_ptr(ptr);

        assert!(tagged.is_ptr());
        assert!(tagged.is_strong_ptr());
        assert!(!tagged.is_weak_ptr());
        assert!(!tagged.is_smi());
        assert_eq!(tagged.as_ptr().unwrap().as_ptr(), raw);

        unsafe { free_test_obj(raw) };
    }

    #[test]
    fn as_ptr_rejects_smi_and_weak() {
        let smi =
            unsafe { Tagged::<TestObj>::from_value_unchecked(Smi::new_unchecked(1).encode()) };
        assert!(smi.as_ptr().is_none());

        let raw = alloc_test_obj();
        let ptr = unsafe { HeapPtr::<TestObj>::new_unchecked(raw) };
        let weak = unsafe { Tagged::<TestObj>::from_value_unchecked(ptr.encode_weak()) };
        assert!(weak.as_ptr().is_none());

        unsafe { free_test_obj(raw) };
    }

    #[test]
    fn erase_recovers_the_raw_value() {
        let raw = alloc_test_obj();
        let ptr = unsafe { HeapPtr::<TestObj>::new_unchecked(raw) };
        let tagged = Tagged::from_ptr(ptr);

        assert_eq!(tagged.erase(), ptr.encode_strong());

        unsafe { free_test_obj(raw) };
    }

    #[test]
    fn erase_tagged_and_cast_preserve_bits() {
        let tagged = Tagged::<Smi>::smi(7).unwrap();

        let erased = tagged.erase_tagged();
        assert_eq!(erased.erase().to_bits(), tagged.erase().to_bits());

        let cast = unsafe { erased.cast_unchecked::<Smi>() };
        assert_eq!(cast.to_smi().unwrap().value(), 7);
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
