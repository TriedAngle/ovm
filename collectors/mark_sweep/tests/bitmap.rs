use mark_sweep::bitmap::Bitmap;
use std::sync::Arc;

const BASE: usize = 0x1000_0000;

#[test]
fn try_set_exactly_once() {
    let bitmap = Bitmap::new(BASE, 4096, 8);
    assert!(bitmap.try_set(BASE + 8));
    assert!(!bitmap.try_set(BASE + 8));
    assert!(bitmap.is_set(BASE + 8));
    assert!(!bitmap.is_set(BASE + 16));
}

#[test]
fn set_and_clear() {
    let bitmap = Bitmap::new(BASE, 4096, 8);
    bitmap.set(BASE + 16);
    assert!(bitmap.is_set(BASE + 16));
    assert!(!bitmap.try_set(BASE + 16));
    bitmap.clear(BASE + 16);
    assert!(!bitmap.is_set(BASE + 16));
    assert!(bitmap.try_set(BASE + 16));
}

#[test]
fn clear_all_resets_every_bit() {
    let bitmap = Bitmap::new(BASE, 4096, 8);
    bitmap.set(BASE);
    bitmap.set(BASE + 4096 - 8);
    bitmap.clear_all();
    assert!(!bitmap.is_set(BASE));
    assert!(!bitmap.is_set(BASE + 4096 - 8));
    assert_eq!(bitmap.iter_set().count(), 0);
}

#[test]
fn iter_set_yields_ascending_addresses() {
    let bitmap = Bitmap::new(BASE, 4096, 8);
    for offset in [4088, 8, 2048] {
        bitmap.set(BASE + offset);
    }
    let set: Vec<usize> = bitmap.iter_set().collect();
    assert_eq!(set, vec![BASE + 8, BASE + 2048, BASE + 4088]);
}

#[test]
fn independent_granules() {
    let bitmap = Bitmap::new(BASE, 4096, 16);
    assert!(bitmap.try_set(BASE + 32));
    assert!(bitmap.try_set(BASE + 48));
    assert!(!bitmap.try_set(BASE + 32));
    assert!(bitmap.is_set(BASE + 32));
    assert!(!bitmap.is_set(BASE + 64));
    assert_eq!(bitmap.granularity(), 16);
}

#[test]
fn concurrent_try_set_claims_each_bit_once() {
    const BITS: usize = 512;
    let bitmap = Arc::new(Bitmap::new(BASE, BITS * 8, 8));
    let claimed = Arc::new(std::sync::Mutex::new(vec![0usize; BITS]));

    std::thread::scope(|s| {
        for _ in 0..8 {
            let bitmap = Arc::clone(&bitmap);
            let claimed = Arc::clone(&claimed);
            s.spawn(move || {
                for i in 0..BITS {
                    if bitmap.try_set(BASE + i * 8) {
                        claimed.lock().unwrap()[i] += 1;
                    }
                }
            });
        }
    });

    let claimed = Arc::try_unwrap(claimed).unwrap().into_inner().unwrap();
    assert!(claimed.iter().all(|&c| c == 1));
}
