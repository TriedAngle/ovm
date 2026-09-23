//! Frontend benchmark: old hand-written parser vs the oxc pipeline.
//!
//! Stages are measured in isolation (warm inputs, best-of-N), plus
//! end-to-end. Usage: bench_parse <corpus.js> [iterations]

use std::time::Instant;

use oxc_allocator::Allocator;
use oxc_semantic::SemanticBuilder;
use oxc_span::SourceType;

use oxc_parser::{ParseOptions, Parser};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: bench_parse <corpus.js> [iterations]");
    let iterations: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let mb = src.len() as f64 / (1024.0 * 1024.0);

    // --- correctness gate: the frontend must accept the corpus ---
    match js_compiler::compile_js(&src, bytecode::SourceMode::Script) {
        Ok(_) => {}
        Err(e) => panic!("frontend rejected the corpus: {e}"),
    }
    println!("corpus accepted\n");

    // One parse kept alive for the isolated stage loops below.
    let keep_allocator = Allocator::default();
    let keep_ret = Parser::new(&keep_allocator, &src, SourceType::script())
        .with_options(ParseOptions {
            preserve_parens: false,
            allow_return_outside_function: true,
            ..ParseOptions::default()
        })
        .parse();
    assert!(keep_ret.diagnostics.is_empty());
    let keep_semantic = SemanticBuilder::new()
        .with_build_nodes(true)
        .with_check_syntax_error(true)
        .build(&keep_ret.program);
    assert!(keep_semantic.diagnostics.is_empty());
    let keep_facts = js_compiler::analysis::analyze(
        &keep_ret.program,
        &keep_semantic.semantic,
        js_compiler::analysis::Mode::Script,
    );

    let mut parse = Vec::new();
    let mut semantic = Vec::new();
    let mut facts = Vec::new();
    let mut codegen = Vec::new();
    let mut full = Vec::new();

    for _ in 0..iterations {
        // parse only (fresh arena each time, warm page cache)
        let t = Instant::now();
        let allocator = Allocator::default();
        let ret = Parser::new(&allocator, &src, SourceType::script())
            .with_options(ParseOptions {
                preserve_parens: false,
                allow_return_outside_function: true,
                ..ParseOptions::default()
            })
            .parse();
        assert!(ret.diagnostics.is_empty());
        drop(ret);
        drop(allocator);
        parse.push(t.elapsed());

        // semantic only
        let t = Instant::now();
        let semantic_run = SemanticBuilder::new()
            .with_build_nodes(true)
            .with_check_syntax_error(true)
            .build(&keep_ret.program);
        assert!(semantic_run.diagnostics.is_empty());
        semantic.push(t.elapsed());

        // facts only
        let t = Instant::now();
        let facts_run = js_compiler::analysis::analyze(
            &keep_ret.program,
            &keep_semantic.semantic,
            js_compiler::analysis::Mode::Script,
        );
        facts.push(t.elapsed());
        drop(facts_run);

        // codegen only (fresh compiler state each time)
        let t = Instant::now();
        let program_run =
            js_compiler::codegen::generate(keep_semantic.semantic.scoping(), &keep_facts)
                .expect("codegen");
        codegen.push(t.elapsed());
        drop(program_run);

        // end to end
        let t = Instant::now();
        let program = js_compiler::compile_js(&src, bytecode::SourceMode::Script).expect("compile");
        full.push(t.elapsed());
        drop(program);
    }

    let report = |name: &str, samples: &Vec<std::time::Duration>| {
        let mut sorted = samples.clone();
        sorted.sort();
        let best = sorted[0];
        println!(
            "{:<28} best {:>8.1} ms  median {:>8.1} ms  ({:>6.1} MB/s best)",
            name,
            best.as_secs_f64() * 1e3,
            sorted[sorted.len() / 2].as_secs_f64() * 1e3,
            mb / best.as_secs_f64(),
        );
    };

    println!("corpus: {mb:.1} MB, {} iterations\n", iterations);
    println!("--- new frontend (isolated stages) ---");
    report("oxc parse", &parse);
    report("oxc semantic", &semantic);
    report("facts (layout analysis)", &facts);
    report("IR codegen", &codegen);
    println!("--- new frontend (end to end) ---");
    report("parse+semantic+facts+codegen", &full);
}
