//! ES 21.4 (Date): epoch-millisecond instants. The VM has no time-zone
//! database, so every operation is UTC. `Date.parse`/`toISOString`/
//! `toGMTString`/`toString` are written to round-trip each other exactly.

use vm_core::Handle;
use vm_core::Heap;
use vm_core::Object;
use vm_core::RuntimeContext;
use vm_core::{ContextState, VM};
use vm_core::{Convert, DenseString, HandleSlice, Tagged, Value, VmError};
use vm_core::{raise_runtime, rt_try};

/// Milliseconds in a UTC day.
const MS_PER_DAY: f64 = 86_400_000.0;

const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Milliseconds since the Unix epoch.
fn now_millis() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

// ---- civil math (proleptic Gregorian; exact over the ±1e8-day range) ----

/// Days from 1970-01-01 to (year, month, day).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = ((m + 9) % 12) as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// (year, month, day) from days since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Weekday of a day count (0 = Sunday; 1970-01-01 was a Thursday).
fn week_day(z: i64) -> usize {
    (z + 4).rem_euclid(7) as usize
}

/// A broken-down UTC instant.
struct Broken {
    year: i64,
    month: u32,
    day: u32,
    hour: i64,
    min: i64,
    sec: i64,
    ms: i64,
    weekday: usize,
}

fn break_down(t: f64) -> Option<Broken> {
    if !t.is_finite() {
        return None;
    }
    let days = (t / MS_PER_DAY).floor() as i64;
    let tod = t.rem_euclid(MS_PER_DAY) as i64;
    let (year, month, day) = civil_from_days(days);
    Some(Broken {
        year,
        month,
        day,
        hour: tod / 3_600_000,
        min: (tod / 60_000) % 60,
        sec: (tod / 1_000) % 60,
        ms: tod % 1_000,
        weekday: week_day(days),
    })
}

/// The year in ISO form: 4 digits in 0..=9999, else sign + 6 digits.
fn year_iso(y: i64) -> String {
    if (0..=9999).contains(&y) {
        format!("{y:04}")
    } else if y < 0 {
        format!("-{:06}", y.unsigned_abs())
    } else {
        format!("+{:06}", y.unsigned_abs())
    }
}

fn year_date_string(y: i64) -> String {
    if (0..=9999).contains(&y) {
        format!("{y:04}")
    } else {
        format!("{y}")
    }
}

fn format_iso(b: &Broken) -> String {
    format!(
        "{}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year_iso(b.year),
        b.month,
        b.day,
        b.hour,
        b.min,
        b.sec,
        b.ms
    )
}

fn format_gmt(b: &Broken) -> String {
    format!(
        "{}, {:02} {} {} {:02}:{:02}:{:02} GMT",
        WEEKDAYS[b.weekday],
        b.day,
        MONTHS[(b.month - 1) as usize],
        year_date_string(b.year),
        b.hour,
        b.min,
        b.sec
    )
}

fn format_date_string(b: &Broken) -> String {
    format!(
        "{} {} {:02} {} {:02}:{:02}:{:02} GMT+0000 (Coordinated Universal Time)",
        WEEKDAYS[b.weekday],
        MONTHS[(b.month - 1) as usize],
        b.day,
        year_date_string(b.year),
        b.hour,
        b.min,
        b.sec
    )
}

// ---- parsing ---------------------------------------------------------------

/// A decimal digit run at `i`; returns (value, next index).
fn parse_uint(b: &[u8], i: usize) -> Option<(i64, usize)> {
    let start = i;
    let mut j = i;
    while j < b.len() && b[j].is_ascii_digit() {
        j += 1;
    }
    if j == start {
        return None;
    }
    let s = core::str::from_utf8(&b[start..j]).ok()?;
    Some((s.parse().ok()?, j))
}

/// `YYYY-MM-DDTHH:mm:ss(.sss)(Z|±HH:mm)`; missing tail parts default per
/// ES (date-only forms are UTC). `None` when the input is not ISO-shaped.
fn parse_iso(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut sign = 1i64;
    match b.first() {
        Some(b'+') => i += 1,
        Some(b'-') => {
            sign = -1;
            i += 1;
        }
        Some(c) if c.is_ascii_digit() => {}
        _ => return None,
    }
    let (year, ni) = parse_uint(b, i)?;
    i = ni;
    let (mut month, mut day) = (1u32, 1u32);
    let (mut hour, mut min, mut sec, mut ms) = (0i64, 0i64, 0i64, 0i64);
    let mut has_time = false;
    if b.get(i) == Some(&b'-') {
        let (v, ni) = parse_uint(b, i + 1)?;
        month = u32::try_from(v).ok()?;
        i = ni;
        if b.get(i) == Some(&b'-') {
            let (v, ni) = parse_uint(b, i + 1)?;
            day = u32::try_from(v).ok()?;
            i = ni;
        }
    }
    if matches!(b.get(i), Some(b'T') | Some(b' ')) {
        has_time = true;
        let (v, ni) = parse_uint(b, i + 1)?;
        hour = v;
        i = ni;
        if b.get(i) == Some(&b':') {
            let (v, ni) = parse_uint(b, i + 1)?;
            min = v;
            i = ni;
            if b.get(i) == Some(&b':') {
                let (v, ni) = parse_uint(b, i + 1)?;
                sec = v;
                i = ni;
                if b.get(i) == Some(&b'.') {
                    // 1-3 fractional digits, right-padded to milliseconds
                    let start = i + 1;
                    let (v, ni) = parse_uint(b, start)?;
                    let n = ni - start;
                    if !(1..=3).contains(&n) {
                        return None;
                    }
                    ms = v * 10i64.pow(3 - n as u32);
                    i = ni;
                }
            }
        }
    }
    // zone: Z, or ±HH:mm / ±HHmm / ±HH
    let mut offset_min = 0i64;
    if has_time {
        match b.get(i) {
            Some(b'Z') | None => i += b.get(i).is_some() as usize,
            Some(c @ (b'+' | b'-')) => {
                let zs = if *c == b'+' { 1 } else { -1 };
                let (zh, ni) = parse_uint(b, i + 1)?;
                let mut ni = ni;
                let mut zm = 0;
                if b.get(ni) == Some(&b':') {
                    let (v, n2) = parse_uint(b, ni + 1)?;
                    zm = v;
                    ni = n2;
                } else if b.len() - ni == 2 && b[ni].is_ascii_digit() {
                    // ±HHmm form: parse_uint already consumed all digits
                    zm = zh % 100;
                    offset_min = zh / 100;
                }
                offset_min = offset_min * 60 + zm;
                offset_min *= zs;
                i = ni;
            }
            _ => return None,
        }
    }
    if i != b.len() {
        return None;
    }
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 24 || min > 59 || sec > 59 {
        return None;
    }
    let days = days_from_civil(sign * year, month, day);
    let ms_total =
        days as f64 * MS_PER_DAY + ((hour * 60 + min) * 60 + sec) as f64 * 1000.0 + ms as f64
            - (offset_min as f64 * 60_000.0);
    Some(ms_total)
}

/// The tokenized forms we emit ourselves: `"Thu, 01 Jan 1970 00:00:00
/// GMT"` and `"Thu Jan 01 1970 00:00:00 GMT+0000 (...)"`.
fn parse_loose(s: &str) -> Option<f64> {
    let (mut month, mut day, mut year) = (None, None, None);
    let (mut hour, mut min, mut sec) = (0i64, 0i64, 0i64);
    for tok in s.split(|c: char| c.is_whitespace() || matches!(c, ',' | '(' | ')')) {
        if tok.is_empty() || tok.starts_with("GMT") || tok.starts_with("gmt") {
            continue;
        }
        if month.is_none()
            && let Some(m) = MONTHS.iter().position(|m| m.eq_ignore_ascii_case(tok))
        {
            month = Some(m as u32 + 1);
            continue;
        }
        if tok.contains(':') {
            let mut it = tok.split(':');
            hour = it.next()?.parse().ok()?;
            min = it.next().and_then(|s| s.parse().ok())?;
            sec = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            continue;
        }
        if let Ok(n) = tok.parse::<i64>() {
            // 1-2 digit non-negative token is the day of month, the rest
            // is the year (3+ digits, or signed)
            if tok.len() < 3 && !tok.starts_with(['+', '-']) && day.is_none() {
                day = Some(n);
            } else if year.is_none() {
                year = Some(n);
            }
            continue;
        }
    }
    let (m, d, y) = (month?, day?, year?);
    if !(1..=31).contains(&d) || hour > 24 || min > 59 || sec > 59 {
        return None;
    }
    let days = days_from_civil(y, m, u32::try_from(d).ok()?);
    Some(days as f64 * MS_PER_DAY + ((hour * 60 + min) * 60 + sec) as f64 * 1000.0)
}

/// `Date.parse(string)` (ES 21.4.2.2): the ISO form plus the formats our
/// `toGMTString`/`toString` emit. Unparseable input is NaN.
pub fn date_parse<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let Some(arg) = args.get(1) else {
        return raise_runtime(vm, heap, state, VmError::Arity);
    };
    let text = match Object::to_string(vm, heap, state, arg) {
        Ok(Some(s)) => {
            let word = s.raw();
            // Safety: fresh string word, no allocation since the read.
            unsafe { word.assume_valid(heap) }
                .get_as::<DenseString>()
                .map(|d| d.to_rust_string(heap))
                .unwrap_or_default()
        }
        Ok(None) => return heap.known().exception.as_tagged(heap).erase(),
        Err(err) => return raise_runtime(vm, heap, state, err),
    };
    let ms = parse_iso(&text)
        .or_else(|| parse_loose(&text))
        .unwrap_or(f64::NAN);
    heap.new_number(ms)
}

/// `Date(...)` / `new Date(...)`: no arguments → now; one numeric
/// argument → that many epoch milliseconds; anything richer stays
/// unimplemented.
pub fn date_constructor<'a>(
    nctx: RuntimeContext<'a>,
    new_target: Option<Handle<'_, Value>>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let is_construct = new_target.is_some();
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let ms = match args.get(1) {
        None => now_millis(),
        Some(arg) => {
            let ms = rt_try!(
                vm,
                heap,
                state,
                state.handle_scope(|scope| {
                    let arg = scope.handle(arg.as_tagged(heap));
                    Object::to_numeric(vm, heap, state, arg)
                })
            );
            let Some(ms) = ms else {
                return heap.known().exception.as_tagged(heap).erase();
            };
            ms
        }
    };

    if !is_construct {
        // call form answers the current date-time as a string (ES
        // 21.4.2.2: `Date()` ignores arguments)
        return state.handle_scope(|scope| {
            let text = break_down(now_millis())
                .map(|b| format_date_string(&b))
                .unwrap_or_else(|| "Invalid Date".to_string());
            let s = DenseString::from_utf8(heap, &scope, &text);
            s.as_tagged(heap).erase()
        });
    }

    state.handle_scope(|scope| {
        let map = heap.known().date_instance_map;
        let value = scope.handle(heap.new_number(ms));
        heap.new_object(&scope, map, scope.stage(&[value.as_tagged(heap).erase()]))
            .erase()
    })
}

/// `Date.now()` (ES 21.4.2.2): epoch milliseconds as a Number.
pub fn date_now<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    _args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    nctx.heap.new_number(now_millis())
}

/// `Date.prototype.valueOf` (ES 21.4.4.40): the wrapped epoch
/// milliseconds. Only real Date instances qualify.
pub fn date_value_of<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    let Some(receiver) = args.get(0) else {
        return raise_runtime(vm, heap, state, VmError::Arity);
    };
    date_slot(
        vm,
        heap,
        state,
        // Safety: fresh rooted-slot word, no allocation since the read.
        unsafe { Tagged::<Value>::from_value_unchecked(receiver.raw()) },
    )
}

/// Shared body of the string formatters: break the receiver's time value
/// down and apply `format`. `invalid` is the NaN text ("Invalid Date");
/// `None` asks for a TypeError (toISOString's RangeError approximation).
fn format_with<'a>(
    nctx: RuntimeContext<'a>,
    args: HandleSlice<'_>,
    format: fn(&Broken) -> String,
    invalid: Option<&'static str>,
) -> Tagged<'a, Value> {
    let RuntimeContext {
        vm, heap, state, ..
    } = nctx;
    state.handle_scope(|scope| {
        let Some(receiver) = args.get(0) else {
            return raise_runtime(vm, heap, state, VmError::Arity);
        };
        let exception = heap.known().exception.as_tagged(heap).erase().raw();
        let word = date_slot(
            vm,
            heap,
            state,
            // Safety: fresh rooted-slot word, no allocation since the read.
            unsafe { Tagged::<Value>::from_value_unchecked(receiver.raw()) },
        )
        .raw();
        if word == exception {
            // Safety: old-gen singleton word.
            return unsafe { Tagged::<Value>::from_value_unchecked(word) };
        }
        // Safety: fresh slot word, no allocation since the read.
        let Some(ms) = Convert::as_number(unsafe { Tagged::<Value>::from_value_unchecked(word) })
        else {
            return raise_runtime(vm, heap, state, VmError::Type);
        };
        let text = match break_down(ms) {
            Some(b) => format(&b),
            None => match invalid {
                Some(t) => t.to_string(),
                None => return raise_runtime(vm, heap, state, VmError::Type),
            },
        };
        let s = DenseString::from_utf8(heap, &scope, &text);
        s.as_tagged(heap).erase()
    })
}

/// `Date.prototype.toISOString` (ES 21.4.4.36): NaN is a RangeError
/// (approximated here as a TypeError).
pub fn date_to_iso_string<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    format_with(nctx, args, format_iso, None)
}

/// `Date.prototype.toGMTString` / `toUTCString` (ES 21.4.4.43).
pub fn date_to_gmt_string<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    format_with(nctx, args, format_gmt, Some("Invalid Date"))
}

/// `Date.prototype.toString` (ES 21.4.4.41; UTC-formatted).
pub fn date_to_string<'a>(
    nctx: RuntimeContext<'a>,
    _new_target: Option<Handle<'_, Value>>,
    args: HandleSlice<'_>,
) -> Tagged<'a, Value> {
    format_with(nctx, args, format_date_string, Some("Invalid Date"))
}

/// slots[0] of a Date instance (the milliseconds), or a TypeError.
fn date_slot<'a>(
    vm: &VM,
    heap: &'a mut Heap,
    state: &ContextState,
    receiver: Tagged<'a, Value>,
) -> Tagged<'a, Value> {
    let Some(obj) = receiver.as_heap_object() else {
        return raise_runtime(vm, heap, state, VmError::Type);
    };
    let map = obj.as_ref().header.map.get(heap);
    if map != heap.known().date_instance_map.as_tagged(heap) {
        return raise_runtime(vm, heap, state, VmError::Type);
    }
    obj.as_ref().slots.get(heap).at(heap, 0)
}
