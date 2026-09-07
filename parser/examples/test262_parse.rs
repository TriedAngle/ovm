//! Parse-only test262 harness.
//!
//! Usage: cargo run -p parser --example test262_parse -- <file-or-dir>...
//! With no args, runs a small default selection known to fit the parser subset.
//!
//! Frontmatter handling: files with `negative: { phase: parse|syntax }` are
//! expected to FAIL parsing; `flags: [module]` files are skipped (no import/
//! export support); everything else should parse cleanly.

use std::path::{Path, PathBuf};

use parser::{Parser, Utf8SliceStream};

#[derive(Default)]
struct Stats {
    pass: usize,
    expected_fail: usize,
    skipped_module: usize,
    failed: Vec<(PathBuf, String)>,
    false_positive: Vec<PathBuf>, // negative tests that parsed without error
}

fn frontmatter(src: &str) -> &str {
    let start = src.find("/*---").map(|i| i + 5);
    let end = src.find("---*/");
    match (start, end) {
        (Some(s), Some(e)) if s <= e => &src[s..e],
        _ => "",
    }
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("cannot read dir {}", dir.display());
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "js") {
            out.push(path);
        }
    }
}

fn run_file(path: &Path, stats: &mut Stats) {
    let Ok(src) = std::fs::read_to_string(path) else {
        stats.failed.push((path.to_path_buf(), "not utf-8".into()));
        return;
    };
    let fm = frontmatter(&src);
    if fm.contains("module") && fm.contains("flags") {
        stats.skipped_module += 1;
        return;
    }
    let expect_error =
        fm.contains("negative:") && (fm.contains("phase: parse") || fm.contains("phase: syntax"));

    let mut p = Parser::new(Utf8SliceStream::new(&src));
    match (p.parse_script(), expect_error) {
        (Ok(_), false) => stats.pass += 1,
        (Err(e), true) => {
            let _ = e;
            stats.expected_fail += 1;
        }
        (Ok(_), true) => stats.false_positive.push(path.to_path_buf()),
        (Err(e), false) => stats.failed.push((path.to_path_buf(), format!("{e}"))),
    }
}

fn main() {
    let mut roots: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if roots.is_empty() {
        let home = std::env::var("HOME").expect("HOME not set");
        let base = PathBuf::from(home).join("tools/test262/test/language/expressions");
        roots.push(base.join("addition"));
        roots.push(base.join("array"));
    }

    let mut files = Vec::new();
    for root in &roots {
        if root.is_dir() {
            collect(root, &mut files);
        } else {
            files.push(root.clone());
        }
    }
    files.sort();

    let mut stats = Stats::default();
    for file in &files {
        run_file(file, &mut stats);
    }

    println!("total:          {}", files.len());
    println!("pass:           {}", stats.pass);
    println!("expected fail:  {}", stats.expected_fail);
    println!("skipped module: {}", stats.skipped_module);
    println!("FAILED:         {}", stats.failed.len());
    for (path, err) in stats.failed.iter().take(40) {
        println!("  {}: {}", path.display(), err);
    }
    if stats.failed.len() > 40 {
        println!("  ... and {} more", stats.failed.len() - 40);
    }
    println!("FALSE PASS:     {}", stats.false_positive.len());
    for path in stats.false_positive.iter().take(40) {
        println!("  {}", path.display());
    }

    if !stats.failed.is_empty() || !stats.false_positive.is_empty() {
        std::process::exit(1);
    }
}
