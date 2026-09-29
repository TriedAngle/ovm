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
fn a_body_object_captures_outer_let() {
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
fn nested_body_objects_capture_two_levels() {
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
fn parameter_is_local_to_the_callable() {
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
fn if_branch_lets_join_the_enclosing_scope() {
    let (ast, res) = compile("if c { let y = 1\n y }");
    assert_eq!(
        ident_resolutions(&ast, &res, "y"),
        vec![Resolution::Local {
            scope: ScopeId(0),
            decl: 0
        }]
    );
}

#[test]
fn branch_let_in_a_let_initializer_counts_first() {
    // codegen stores the branch `t` before the outer `r`: decl order matches
    let (ast, res) = compile("let r = if c { let t = 1\n t } else { 2 }\n t");
    assert_eq!(
        ident_resolutions(&ast, &res, "t"),
        vec![
            Resolution::Local {
                scope: ScopeId(0),
                decl: 0
            };
            2
        ]
    );
    let (_, res2) = compile("let r = if c { let t = 1\n t } else { 2 }\n r");
    let decls = &res2.scope(ScopeId(0)).decls;
    assert_eq!(decls.len(), 2);
}

#[test]
fn closure_slot_values_resolve_in_the_enclosing_scope() {
    // `v` in the slot value is evaluated where the object is created
    let (ast, res) = compile("let v = 1\n{ m: v || v }");
    assert_eq!(
        ident_resolutions(&ast, &res, "v"),
        vec![
            Resolution::Local {
                scope: ScopeId(0),
                decl: 0
            };
            2
        ]
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
fn self_has_no_resolution() {
    let (ast, res) = compile("self");
    for i in 0..ast.node_count() {
        let id = NodeId(i as u32);
        if matches!(ast.node(id), Node::Self_) {
            assert_eq!(res.resolution(id), None);
        }
    }
}
