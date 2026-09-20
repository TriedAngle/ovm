use kette_parser::{Ast, Node, NodeId, Parser, SlotKind, Symbol, Utf8SliceStream};

fn parse(src: &str) -> (Ast, NodeId) {
    let mut p = Parser::new(Utf8SliceStream::new(src));
    let root = p.parse_script().expect("parse");
    (p.into_ast(), root)
}

fn parse_err(src: &str) -> kette_parser::ParseError {
    let mut p = Parser::new(Utf8SliceStream::new(src));
    p.parse_script().expect_err("expected a parse error")
}

fn stmts(ast: &Ast, root: NodeId) -> &[NodeId] {
    match ast.node(root) {
        Node::StmtList { stmts } => ast.list_items(*stmts),
        other => panic!("root is not a StmtList: {other:?}"),
    }
}

fn only_stmt(ast: &Ast, root: NodeId) -> NodeId {
    let s = stmts(ast, root);
    assert_eq!(s.len(), 1, "expected exactly one statement");
    s[0]
}

fn expr_of(ast: &Ast, stmt: NodeId) -> NodeId {
    match ast.node(stmt) {
        Node::ExprStmt { expr } => *expr,
        other => panic!("statement is not an ExprStmt: {other:?}"),
    }
}

fn sym(ast: &Ast, s: Symbol) -> String {
    String::from_utf8(ast.symbol(s).to_vec()).unwrap()
}

// -- literals / objects / blocks --------------------------------------------

#[test]
fn empty_object() {
    let (ast, root) = parse("{}");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    match ast.node(expr) {
        Node::Object { slots } => assert!(ast.list_items(*slots).is_empty()),
        other => panic!("expected object, got {other:?}"),
    }
}

#[test]
fn object_slots() {
    let (ast, root) = parse("{\n parent*: Clonable\n x: 10\n y: 20\n}");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    let slots = match ast.node(expr) {
        Node::Object { slots } => ast.list_items(*slots),
        other => panic!("expected object, got {other:?}"),
    };
    assert_eq!(slots.len(), 3);
    let kinds: Vec<SlotKind> = slots
        .iter()
        .map(|s| match ast.node(*s) {
            Node::Slot { kind, .. } => *kind,
            _ => panic!(),
        })
        .collect();
    assert_eq!(
        kinds,
        vec![SlotKind::Parent, SlotKind::Named, SlotKind::Named]
    );
    let names: Vec<String> = slots
        .iter()
        .map(|s| match ast.node(*s) {
            Node::Slot { key, .. } => match ast.node(*key) {
                Node::Ident(s) => sym(&ast, *s),
                _ => panic!(),
            },
            _ => panic!(),
        })
        .collect();
    assert_eq!(names, vec!["parent", "x", "y"]);
}

#[test]
fn element_slots() {
    let (ast, root) = parse("{ [0]: 10, [1]: 20 }");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    let slots = match ast.node(expr) {
        Node::Object { slots } => ast.list_items(*slots),
        other => panic!("expected object, got {other:?}"),
    };
    assert_eq!(slots.len(), 2);
    for s in slots {
        match ast.node(*s) {
            Node::Slot {
                kind: SlotKind::Element,
                value,
                ..
            } => assert!(matches!(ast.node(*value), Node::Number(_))),
            other => panic!("expected element slot, got {other:?}"),
        }
    }
}

#[test]
fn slot_value_is_full_expression() {
    // binary expression
    let (ast, root) = parse("{ a: 5 + 5 }");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    let slots = match ast.node(expr) {
        Node::Object { slots } => ast.list_items(*slots),
        other => panic!("expected object, got {other:?}"),
    };
    match ast.node(slots[0]) {
        Node::Slot { value, .. } => assert!(matches!(
            ast.node(*value),
            Node::Binary {
                op: kette_parser::BinaryOp::Add,
                ..
            }
        )),
        other => panic!("expected slot, got {other:?}"),
    }

    // block with a `let` and a trailing expression
    let (ast, root) = parse("{ a: { let x = 10; x } }");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    let slots = match ast.node(expr) {
        Node::Object { slots } => ast.list_items(*slots),
        other => panic!("expected object, got {other:?}"),
    };
    match ast.node(slots[0]) {
        Node::Slot { value, .. } => match ast.node(*value) {
            Node::Block { body, .. } => {
                let stmts = match ast.node(*body) {
                    Node::StmtList { stmts } => ast.list_items(*stmts),
                    other => panic!("expected stmt list, got {other:?}"),
                };
                assert_eq!(stmts.len(), 2);
                assert!(matches!(ast.node(stmts[0]), Node::Let { .. }));
            }
            other => panic!("expected block, got {other:?}"),
        },
        other => panic!("expected slot, got {other:?}"),
    }

    // nested object literal
    let (ast, root) = parse("{ a: { b: 1 } }");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    let slots = match ast.node(expr) {
        Node::Object { slots } => ast.list_items(*slots),
        other => panic!("expected object, got {other:?}"),
    };
    match ast.node(slots[0]) {
        Node::Slot { value, .. } => assert!(matches!(ast.node(*value), Node::Object { .. })),
        other => panic!("expected slot, got {other:?}"),
    }

    // control flow as a slot value
    let (ast, root) = parse("{ a: if c { 1 } else { 2 } }");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    let slots = match ast.node(expr) {
        Node::Object { slots } => ast.list_items(*slots),
        other => panic!("expected object, got {other:?}"),
    };
    match ast.node(slots[0]) {
        Node::Slot { value, .. } => assert!(matches!(ast.node(*value), Node::If { .. })),
        other => panic!("expected slot, got {other:?}"),
    }
}

#[test]
fn keyword_slot_name() {
    let (ast, root) = parse("{\n parent*: Clonable\n if: { |t| t }\n}");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    let slots = match ast.node(expr) {
        Node::Object { slots } => ast.list_items(*slots),
        other => panic!("expected object, got {other:?}"),
    };
    assert_eq!(slots.len(), 2);
    match ast.node(slots[1]) {
        Node::Slot { key, .. } => match ast.node(*key) {
            Node::Ident(s) => assert_eq!(sym(&ast, *s), "if"),
            other => panic!("expected ident key, got {other:?}"),
        },
        other => panic!("expected slot, got {other:?}"),
    }
}

#[test]
fn zero_arg_block() {
    let (ast, root) = parse("{ print(1) }");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    match ast.node(expr) {
        Node::Block { params, .. } => assert!(ast.list_items(*params).is_empty()),
        other => panic!("expected block, got {other:?}"),
    }
}

#[test]
fn lambda_block() {
    let (ast, root) = parse("{ |x| x + 1 }");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    match ast.node(expr) {
        Node::Block { params, body } => {
            assert_eq!(ast.list_items(*params).len(), 1);
            let body_stmts = match ast.node(*body) {
                Node::StmtList { stmts } => ast.list_items(*stmts),
                other => panic!("expected stmt list, got {other:?}"),
            };
            assert_eq!(body_stmts.len(), 1);
            assert!(matches!(
                ast.node(expr_of(&ast, body_stmts[0])),
                Node::Binary {
                    op: kette_parser::BinaryOp::Add,
                    ..
                }
            ));
        }
        other => panic!("expected block, got {other:?}"),
    }
}

#[test]
fn nested_object_in_lambda() {
    let (ast, root) = parse("{ |a, b| { first: a, second: b } }");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    let body = match ast.node(expr) {
        Node::Block { body, .. } => *body,
        other => panic!("expected block, got {other:?}"),
    };
    let stmts = match ast.node(body) {
        Node::StmtList { stmts } => ast.list_items(*stmts),
        other => panic!("expected stmt list, got {other:?}"),
    };
    assert!(matches!(
        ast.node(expr_of(&ast, stmts[0])),
        Node::Object { .. }
    ));
}

#[test]
fn array_literal() {
    let (ast, root) = parse("[1, 2, 3]");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    match ast.node(expr) {
        Node::Array { elements } => assert_eq!(ast.list_items(*elements).len(), 3),
        other => panic!("expected array, got {other:?}"),
    }
}

// -- sends / reads / block evaluation ----------------------------------------

#[test]
fn send_with_args() {
    let (ast, root) = parse("obj.add(1)");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    match ast.node(expr) {
        Node::Send { name, args, .. } => {
            assert_eq!(sym(&ast, *name), "add");
            assert_eq!(ast.list_items(*args).len(), 1);
        }
        other => panic!("expected send, got {other:?}"),
    }
}

#[test]
fn bare_call() {
    // `f(x)` is the call shorthand for evaluating the block
    let (ast, root) = parse("f(x)");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    match ast.node(expr) {
        Node::Call { args, .. } => assert_eq!(ast.list_items(*args).len(), 1),
        other => panic!("expected call, got {other:?}"),
    }
}

#[test]
fn explicit_block_evaluation_is_a_send() {
    // `f.value(x)` is the explicit form and stays an ordinary send
    let (ast, root) = parse("f.value(x)");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    match ast.node(expr) {
        Node::Send { name, args, .. } => {
            assert_eq!(sym(&ast, *name), "value");
            assert_eq!(ast.list_items(*args).len(), 1);
        }
        other => panic!("expected value send, got {other:?}"),
    }
}

#[test]
fn named_read_and_element_reads() {
    let (ast, root) = parse("obj.x");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    assert!(matches!(ast.node(expr), Node::Get { .. }));

    let (ast, root) = parse("obj[0]");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    assert!(matches!(ast.node(expr), Node::Index { .. }));

    let (ast, root) = parse("obj.0");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    match ast.node(expr) {
        Node::Index { key, .. } => assert!(matches!(ast.node(*key), Node::Number(_))),
        other => panic!("expected index, got {other:?}"),
    }
}

#[test]
fn block_argument_is_explicit() {
    // blocks are always passed as explicit arguments
    let (ast, root) = parse("arr.each({ |x| print(x) })");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    match ast.node(expr) {
        Node::Send { name, args, .. } => {
            assert_eq!(sym(&ast, *name), "each");
            let args = ast.list_items(*args);
            assert_eq!(args.len(), 1);
            assert!(matches!(ast.node(args[0]), Node::Block { .. }));
        }
        other => panic!("expected send, got {other:?}"),
    }
}

#[test]
fn operator_precedence() {
    let (ast, root) = parse("1 + 2 * 3");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    match ast.node(expr) {
        Node::Binary {
            op: kette_parser::BinaryOp::Add,
            rhs,
            ..
        } => match ast.node(*rhs) {
            Node::Binary {
                op: kette_parser::BinaryOp::Mul,
                ..
            } => {}
            other => panic!("expected `*` on the right, got {other:?}"),
        },
        other => panic!("expected `+` at the root, got {other:?}"),
    }
}

// -- statements / control flow ----------------------------------------------

#[test]
fn let_declaration() {
    let (ast, root) = parse("let x = 1");
    match ast.node(only_stmt(&ast, root)) {
        Node::Let { name, .. } => assert_eq!(sym(&ast, *name), "x"),
        other => panic!("expected let, got {other:?}"),
    }
}

#[test]
fn assignment_targets() {
    let (ast, root) = parse("obj.x = 55");
    assert!(matches!(
        ast.node(expr_of(&ast, only_stmt(&ast, root))),
        Node::Assign { .. }
    ));

    let (ast, root) = parse("arr[3] = 40");
    assert!(matches!(
        ast.node(expr_of(&ast, only_stmt(&ast, root))),
        Node::Assign { .. }
    ));

    let (ast, root) = parse("i = i + 1");
    assert!(matches!(
        ast.node(expr_of(&ast, only_stmt(&ast, root))),
        Node::Assign { .. }
    ));
}

#[test]
fn if_else() {
    let (ast, root) = parse("if c { a } else { b }");
    match ast.node(expr_of(&ast, only_stmt(&ast, root))) {
        Node::If { else_, .. } => assert!(else_.is_some()),
        other => panic!("expected if, got {other:?}"),
    }
}

#[test]
fn if_without_else() {
    let (ast, root) = parse("if c { a }");
    match ast.node(expr_of(&ast, only_stmt(&ast, root))) {
        Node::If { else_, .. } => assert!(else_.is_none()),
        other => panic!("expected if, got {other:?}"),
    }
}

#[test]
fn while_loop() {
    let (ast, root) = parse("while c { i = i + 1 }");
    assert!(matches!(
        ast.node(expr_of(&ast, only_stmt(&ast, root))),
        Node::While { .. }
    ));
}

#[test]
fn for_in_loop() {
    let (ast, root) = parse("for x in xs { print(x) }");
    match ast.node(expr_of(&ast, only_stmt(&ast, root))) {
        Node::ForIn { name, .. } => assert_eq!(sym(&ast, *name), "x"),
        other => panic!("expected for-in, got {other:?}"),
    }
}

#[test]
fn explicit_return() {
    let (ast, root) = parse("return x");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    assert!(matches!(ast.node(expr), Node::Return { .. }));
}

#[test]
fn try_catch_parses() {
    let (ast, root) = parse("try { a } catch e { e }");
    let expr = expr_of(&ast, only_stmt(&ast, root));
    match ast.node(expr) {
        Node::Try { handler, .. } => match ast.node(*handler) {
            Node::Block { params, .. } => {
                assert_eq!(
                    ast.list_items(*params).len(),
                    1,
                    "catch binding is a parameter"
                );
            }
            other => panic!("expected handler block, got {other:?}"),
        },
        other => panic!("expected try, got {other:?}"),
    }
}

#[test]
fn match_arms() {
    let src = "match shape {\n Circle -> { |r| r }\n Rect -> { |w, h| w }\n else -> { 0 }\n}";
    let (ast, root) = parse(src);
    let expr = expr_of(&ast, only_stmt(&ast, root));
    match ast.node(expr) {
        Node::Match { arms, .. } => {
            let arms = ast.list_items(*arms);
            assert_eq!(arms.len(), 3);
            let targets: Vec<bool> = arms
                .iter()
                .map(|a| match ast.node(*a) {
                    Node::MatchArm { target, .. } => target.is_some(),
                    _ => panic!(),
                })
                .collect();
            assert_eq!(targets, vec![true, true, false]);
        }
        other => panic!("expected match, got {other:?}"),
    }
}

// -- newline / ASI -----------------------------------------------------------

#[test]
fn newline_ends_statement() {
    let (ast, root) = parse("a\nb");
    assert_eq!(stmts(&ast, root).len(), 2);
}

#[test]
fn leading_operator_continues() {
    // JS-style: a binary operator at the start of the next line continues
    let (ast, root) = parse("a\n&& b");
    assert_eq!(stmts(&ast, root).len(), 1);
    assert!(matches!(
        ast.node(expr_of(&ast, only_stmt(&ast, root))),
        Node::Binary {
            op: kette_parser::BinaryOp::And,
            ..
        }
    ));
}

#[test]
fn trailing_operator_continues() {
    let (ast, root) = parse("a &&\nb");
    assert_eq!(stmts(&ast, root).len(), 1);
    assert!(matches!(
        ast.node(expr_of(&ast, only_stmt(&ast, root))),
        Node::Binary {
            op: kette_parser::BinaryOp::And,
            ..
        }
    ));
}

#[test]
fn semicolon_forces_end_of_statement() {
    let (ast, root) = parse("a; b");
    assert_eq!(stmts(&ast, root).len(), 2);
}

#[test]
fn postfix_does_not_cross_newline() {
    for src in ["a\n[0]", "a\n(b)", "a\n{ b: 1 }"] {
        let (ast, root) = parse(src);
        assert_eq!(stmts(&ast, root).len(), 2, "{src}");
    }
}

#[test]
fn leading_dot_continues() {
    let (ast, root) = parse("a\n.b");
    assert_eq!(stmts(&ast, root).len(), 1);
    assert!(matches!(
        ast.node(expr_of(&ast, only_stmt(&ast, root))),
        Node::Get { .. }
    ));
}

#[test]
fn object_slot_separators() {
    for src in ["{ a: 1; b: 2 }", "{ a: 1, b: 2 }", "{\n a: 1\n b: 2\n}"] {
        let (ast, root) = parse(src);
        let expr = expr_of(&ast, only_stmt(&ast, root));
        match ast.node(expr) {
            Node::Object { slots } => assert_eq!(ast.list_items(*slots).len(), 2, "{src}"),
            other => panic!("expected object for {src}, got {other:?}"),
        }
    }
}

// -- comments / errors -------------------------------------------------------

#[test]
fn comments_are_trivia() {
    let src = "// line\n/* block /* nested */ still */\nlet x = 1";
    let (ast, root) = parse(src);
    assert!(matches!(ast.node(only_stmt(&ast, root)), Node::Let { .. }));
}

#[test]
fn unterminated_string_is_error() {
    let e = parse_err("\"abc");
    assert!(e.message.contains("unterminated string"));
}

#[test]
fn missing_colon_in_object_is_error() {
    // `{ x 1 }` begins like a block, so it is parsed as statements and fails.
    let e = parse_err("{ x 1 }");
    assert!(!e.message.is_empty());
}

// -- the whole example -------------------------------------------------------

// #[test]
// fn example_ktt_parses() {
//     let src = include_str!("../example.ktt");
//     let (ast, root) = parse(src);
//     let stmts = stmts(&ast, root);
//     assert!(
//         stmts.len() > 20,
//         "expected many top-level statements, got {}",
//         stmts.len()
//     );
//     assert!(ast.node_count() > 100);
// }
