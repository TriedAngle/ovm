use parser::{Ast, FunctionId, Node, NodeId, Parser, Resolution, Utf8SliceStream, resolve};

fn resolve_src(src: &str) -> (Ast, parser::Resolved) {
    let mut p = Parser::new(Utf8SliceStream::new(src));
    p.parse_script()
        .unwrap_or_else(|e| panic!("parse error in {src:?}: {e}"));
    let ast = p.into_ast();
    let resolved = resolve(&ast);
    (ast, resolved)
}

/// resolutions of all Identifier nodes with this name, in arena order
fn resolutions_of(ast: &Ast, r: &parser::Resolved, name: &str) -> Vec<Resolution> {
    (0..ast.node_count())
        .map(|i| NodeId(i as u32))
        .filter_map(|id| match ast.node(id) {
            Node::Identifier { sym } if ast.symbol(*sym) == name.as_bytes() => r.resolution(id),
            _ => None,
        })
        .collect()
}

#[test]
fn locals_and_params() {
    let (ast, r) = resolve_src("var x = 1; x; function f(a, b) { return a + b; }");
    assert_eq!(
        resolutions_of(&ast, &r, "x"),
        vec![Resolution::Local {
            reg: 0,
            hole_check: false
        }]
    );
    assert_eq!(
        resolutions_of(&ast, &r, "a"),
        vec![Resolution::Param {
            index: 0,
            hole_check: false
        }]
    );
    assert_eq!(
        resolutions_of(&ast, &r, "b"),
        vec![Resolution::Param {
            index: 1,
            hole_check: false
        }]
    );
    let f = (0..ast.function_count() as u32)
        .map(FunctionId)
        .find(|&id| ast.function(id).name.is_some())
        .unwrap();
    assert_eq!(r.layout(f).register_count, 0);
    assert_eq!(r.layout(f).context_slots, 0);
}

#[test]
fn captured_variables_go_to_context() {
    let (ast, r) = resolve_src("function f() { var x = 1; return () => x; }");
    let f = (0..ast.function_count() as u32)
        .map(FunctionId)
        .find(|&id| ast.function(id).name.is_some())
        .unwrap();
    assert_eq!(
        resolutions_of(&ast, &r, "x"),
        vec![Resolution::Context {
            slot: 0,
            depth: 1, // one function hop from the arrow to f
            hole_check: false
        }]
    );
    assert_eq!(r.layout(f).register_count, 0);
    assert_eq!(r.layout(f).context_slots, 1);
}

#[test]
fn let_const_need_hole_checks_var_does_not() {
    let (ast, r) = resolve_src("let a = 1; a; const b = 2; b; var c = 3; c;");
    assert!(matches!(
        resolutions_of(&ast, &r, "a")[0],
        Resolution::Local {
            hole_check: true,
            ..
        }
    ));
    assert!(matches!(
        resolutions_of(&ast, &r, "b")[0],
        Resolution::Local {
            hole_check: true,
            ..
        }
    ));
    assert!(matches!(
        resolutions_of(&ast, &r, "c")[0],
        Resolution::Local {
            hole_check: false,
            ..
        }
    ));
}

#[test]
fn block_shadowing_resolves_innermost() {
    let (ast, r) = resolve_src("var x = 1; { let x = 2; x; }");
    // the only x *use* is in the block → the let (hole check on)
    assert!(matches!(
        resolutions_of(&ast, &r, "x")[0],
        Resolution::Local {
            hole_check: true,
            ..
        }
    ));
}

#[test]
fn undeclared_is_global() {
    let (ast, r) = resolve_src("y;");
    assert_eq!(
        resolutions_of(&ast, &r, "y"),
        vec![Resolution::GlobalObject]
    );
}

#[test]
fn catch_param_resolves() {
    let (ast, r) = resolve_src("try { may_throw(); } catch (e) { e; }");
    assert!(matches!(
        resolutions_of(&ast, &r, "e")[0],
        Resolution::Local { .. }
    ));
}

#[test]
fn eval_forces_context_allocation() {
    let (ast, r) = resolve_src("function f() { var x = 1; eval('x'); return x; }");
    let f = (0..ast.function_count() as u32)
        .map(FunctionId)
        .find(|&id| ast.function(id).name.is_some())
        .unwrap();
    assert_eq!(r.layout(f).context_slots, 1);
    assert_eq!(r.layout(f).register_count, 0);
    assert!(matches!(
        resolutions_of(&ast, &r, "x")[0],
        Resolution::Context { .. }
    ));
}

#[test]
fn per_iteration_loop_detection() {
    // captured loop variable → per-iteration environment needed
    let (_ast, r) = resolve_src("for (let i = 0; i < 3; i++) { fs.push(() => i); }");
    assert_eq!(r.per_iteration_loops.len(), 1);

    // no capture → no per-iteration environment
    let (_ast, r) = resolve_src("for (let i = 0; i < 3; i++) { g(i); }");
    assert!(r.per_iteration_loops.is_empty());

    // var head is function-scoped → no per-iteration environment
    let (_ast, r) = resolve_src("for (var i = 0; i < 3; i++) { fs.push(() => i); }");
    assert!(r.per_iteration_loops.is_empty());
}
