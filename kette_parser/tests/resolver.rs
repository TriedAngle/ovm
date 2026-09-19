use kette_parser::{
    Ast, Node, NodeId, Parser, Resolution, Resolved, ScopeId, Utf8SliceStream, resolve,
};

fn compile(src: &str) -> (Ast, Resolved) {
    let mut p = Parser::new(Utf8SliceStream::new(src));
    p.parse_script().expect("parse");
    let ast = p.into_ast();
    let resolved = resolve(&ast);
    (ast, resolved)
}

/// Every resolution attached to an `Ident` node whose name is `name`.
fn ident_resolutions(ast: &Ast, resolved: &Resolved, name: &str) -> Vec<Resolution> {
    (0..ast.node_count())
        .filter_map(|i| {
            let id = NodeId(i as u32);
            match ast.node(id) {
                Node::Ident(sym) if ast.symbol(*sym) == name.as_bytes() => resolved.resolution(id),
                _ => None,
            }
        })
        .collect()
}

#[test]
fn top_level_let_is_local() {
    let (ast, res) = compile("let x = 1\nx");
    assert_eq!(
        ident_resolutions(&ast, &res, "x"),
        vec![Resolution::Local {
            scope: ScopeId(0),
            decl: 0
        }]
    );
}

#[test]
fn unbound_name_is_global() {
    let (ast, res) = compile("print(1)");
    assert_eq!(
        ident_resolutions(&ast, &res, "print"),
        vec![Resolution::Global]
    );
}

#[test]
fn block_captures_outer_let() {
    let (ast, res) = compile("let x = 1\n{ x }");
    assert_eq!(
        ident_resolutions(&ast, &res, "x"),
        vec![Resolution::Capture {
            scope: ScopeId(0),
            decl: 0,
            depth: 1
        }]
    );
}

#[test]
fn nested_block_captures_two_levels() {
    let (ast, res) = compile("let x = 1\n{ { x } }");
    assert_eq!(
        ident_resolutions(&ast, &res, "x"),
        vec![Resolution::Capture {
            scope: ScopeId(0),
            decl: 0,
            depth: 2
        }]
    );
}

#[test]
fn parameter_is_local_to_block() {
    let (ast, res) = compile("{ |a| a }");
    assert_eq!(
        ident_resolutions(&ast, &res, "a"),
        vec![Resolution::Local {
            scope: ScopeId(1),
            decl: 0
        }]
    );
    assert_eq!(res.scope(ScopeId(1)).parent, Some(ScopeId(0)));
}

#[test]
fn inner_let_shadows_outer() {
    let (ast, res) = compile("let x = 1\n{ let x = 2\n x }");
    assert_eq!(
        ident_resolutions(&ast, &res, "x"),
        vec![Resolution::Local {
            scope: ScopeId(1),
            decl: 0
        }]
    );
}

#[test]
fn forward_reference_resolves_to_enclosing_scope() {
    // `Vec2` is used before its `let` textually, but the root scope is
    // pre-scanned, so it is captured rather than treated as a global.
    let (ast, res) = compile("let f = { Vec2 }\nlet Vec2 = 1");
    assert_eq!(
        ident_resolutions(&ast, &res, "Vec2"),
        vec![Resolution::Capture {
            scope: ScopeId(0),
            decl: 1,
            depth: 1
        }]
    );
}

#[test]
fn for_loop_variable_is_local_to_body() {
    let (ast, res) = compile("for x in xs { x }");
    assert_eq!(
        ident_resolutions(&ast, &res, "x"),
        vec![Resolution::Local {
            scope: ScopeId(1),
            decl: 0
        }]
    );
    assert_eq!(
        ident_resolutions(&ast, &res, "xs"),
        vec![Resolution::Global]
    );
}

#[test]
fn named_slot_key_is_not_resolved() {
    let (ast, res) = compile("{ bar: 1 }");
    assert!(ident_resolutions(&ast, &res, "bar").is_empty());
}

#[test]
fn element_slot_key_is_a_value() {
    let (ast, res) = compile("let i = 0\n{ [i]: 1 }");
    assert_eq!(
        ident_resolutions(&ast, &res, "i"),
        vec![Resolution::Local {
            scope: ScopeId(0),
            decl: 0
        }]
    );
}

#[test]
fn send_selector_is_not_resolved() {
    let (ast, res) = compile("obj.add(1)");
    assert!(ident_resolutions(&ast, &res, "add").is_empty());
    assert_eq!(
        ident_resolutions(&ast, &res, "obj"),
        vec![Resolution::Global]
    );
}

#[test]
fn match_target_is_not_resolved() {
    let (ast, res) = compile("match x { Circle -> { 1 } }");
    assert!(ident_resolutions(&ast, &res, "Circle").is_empty());
    assert_eq!(ident_resolutions(&ast, &res, "x"), vec![Resolution::Global]);
}

#[test]
fn self_has_no_resolution() {
    let (ast, res) = compile("self");
    for i in 0..ast.node_count() {
        let id = NodeId(i as u32);
        if matches!(ast.node(id), Node::Self_) {
            assert_eq!(res.resolution(id), None);
        }
    }
}

#[test]
fn example_ktt_resolves() {
    let src = include_str!("../../js_parser/example.ktt");
    let mut p = Parser::new(Utf8SliceStream::new(src));
    p.parse_script().expect("parse");
    let ast = p.into_ast();
    let res = resolve(&ast);

    assert!(res.scope_count() > 10);
    // `Vec2` is referenced inside `Vec2Traits.new` before its declaration.
    assert!(
        ident_resolutions(&ast, &res, "Vec2")
            .iter()
            .any(|r| matches!(r, Resolution::Capture { .. }))
    );
    // `print` is never declared: it stays a global.
    assert!(
        ident_resolutions(&ast, &res, "print")
            .iter()
            .all(|r| *r == Resolution::Global)
    );
}
