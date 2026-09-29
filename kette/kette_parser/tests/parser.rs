//! AST shape tests for the unified `{ slots | params | body }` parser.

use kette_parser::{Ast, BinaryOp, Node, NodeId, Parser, SlotKind, Utf8SliceStream};

fn parse(src: &str) -> Ast {
    let mut p = Parser::new(Utf8SliceStream::new(src));
    p.parse_script().expect("parse");
    p.into_ast()
}

fn parse_err(src: &str) -> kette_parser::ParseError {
    let mut p = Parser::new(Utf8SliceStream::new(src));
    p.parse_script().expect_err("parse error")
}

/// Every node matching the predicate, in creation order.
fn matching<F: Fn(&Node) -> bool>(ast: &Ast, pred: F) -> Vec<NodeId> {
    (0..ast.node_count() as u32)
        .map(NodeId)
        .filter(|&id| pred(ast.node(id)))
        .collect()
}

/// The single node matching the predicate.
fn the<F: Fn(&Node) -> bool>(ast: &Ast, pred: F) -> NodeId {
    let found = matching(ast, pred);
    assert_eq!(found.len(), 1, "expected exactly one match");
    found[0]
}

fn is_object(node: &Node) -> bool {
    matches!(node, Node::Object(_))
}

fn object<'a>(ast: &'a Ast, id: NodeId) -> &'a kette_parser::ObjectParts {
    match ast.node(id) {
        Node::Object(parts) => parts,
        _ => panic!("not an object"),
    }
}

fn slot_kinds(ast: &Ast, parts: &kette_parser::ObjectParts) -> Vec<SlotKind> {
    ast.list_items(parts.slots)
        .iter()
        .map(|&s| match ast.node(s) {
            Node::Slot { kind, .. } => *kind,
            _ => panic!("not a slot"),
        })
        .collect()
}

// -- literals --------------------------------------------------------------

#[test]
fn empty_object_is_not_callable() {
    let ast = parse("{ }");
    let obj = the(&ast, is_object);
    let parts = object(&ast, obj);
    assert!(ast.list_items(parts.slots).is_empty());
    assert!(parts.params.is_none());
    assert!(ast.object_body_stmts(parts).is_empty());
    assert!(!ast.is_closure(obj));
}

#[test]
fn explicit_empty_params_is_callable() {
    let ast = parse("{ || }");
    let obj = the(&ast, is_object);
    let parts = object(&ast, obj);
    assert!(parts.params.is_some());
    assert!(ast.is_closure(obj));
}

#[test]
fn slots_only_is_not_callable() {
    let ast = parse("{ x: 1, y: 2 }");
    let obj = the(&ast, is_object);
    let parts = object(&ast, obj);
    assert_eq!(
        slot_kinds(&ast, parts),
        vec![SlotKind::Named, SlotKind::Named]
    );
    assert!(parts.params.is_none());
    assert!(!ast.is_closure(obj));
}

#[test]
fn parent_slots() {
    let ast = parse("{ parent*: P, mixin*: M, x: 1 }");
    let obj = the(&ast, is_object);
    assert_eq!(
        slot_kinds(&ast, object(&ast, obj)),
        vec![SlotKind::Parent, SlotKind::Parent, SlotKind::Named]
    );
}

#[test]
fn element_slots() {
    let ast = parse("{ [0]: 10, [1]: 20 }");
    let obj = the(&ast, is_object);
    let parts = object(&ast, obj);
    assert_eq!(
        slot_kinds(&ast, parts),
        vec![SlotKind::Element, SlotKind::Element]
    );
    for (i, &slot) in ast.list_items(parts.slots).iter().enumerate() {
        let Node::Slot { key, .. } = *ast.node(slot) else {
            panic!("slot");
        };
        assert!(matches!(ast.node(key), Node::Number(n) if *n == i as f64));
    }
}

#[test]
fn element_key_is_an_expression() {
    let ast = parse("let i = 0\n{ [i + 1]: 9 }");
    let obj = the(&ast, is_object);
    let Node::Slot { key, .. } = *ast.node(ast.list_items(object(&ast, obj).slots)[0]) else {
        panic!("slot");
    };
    assert!(matches!(
        ast.node(key),
        Node::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
}

#[test]
fn params_make_a_callable() {
    let ast = parse("{ |a, b| a }");
    let obj = the(&ast, is_object);
    let parts = object(&ast, obj);
    assert_eq!(ast.list_items(parts.params.unwrap()).len(), 2);
    assert!(ast.is_closure(obj));
}

#[test]
fn body_statements_make_a_callable() {
    let ast = parse("{ print(1) }");
    let obj = the(&ast, is_object);
    let parts = object(&ast, obj);
    assert!(params_is_none_and_body_len(&ast, parts, 1));
    assert!(ast.is_closure(obj));
}

fn params_is_none_and_body_len(ast: &Ast, parts: &kette_parser::ObjectParts, len: usize) -> bool {
    parts.params.is_none() && ast.object_body_stmts(parts).len() == len
}

#[test]
fn slots_params_and_code_in_one_object() {
    let ast = parse(
        "{\n    count: 0\n    ||\n        self.count = self.count + 1\n        self.count\n}",
    );
    let obj = the(&ast, is_object);
    let parts = object(&ast, obj);
    assert_eq!(slot_kinds(&ast, parts), vec![SlotKind::Named]);
    assert!(
        parts
            .params
            .as_ref()
            .is_some_and(|p| ast.list_items(*p).is_empty())
    );
    assert_eq!(ast.object_body_stmts(parts).len(), 2);
    assert!(ast.is_closure(obj));
}

// -- the slot / statement distinction ---------------------------------------

#[test]
fn a_product_is_a_statement_not_a_parent_slot() {
    let ast = parse("{ x * y }");
    let obj = the(&ast, is_object);
    let parts = object(&ast, obj);
    assert!(ast.list_items(parts.slots).is_empty());
    assert_eq!(ast.object_body_stmts(parts).len(), 1);
    assert!(matches!(
        ast.node(ast.object_body_stmts(parts)[0]),
        Node::ExprStmt { expr: _ }
    ));
    let stmts = ast.object_body_stmts(parts);
    let Node::ExprStmt { expr } = *ast.node(stmts[0]) else {
        panic!()
    };
    assert!(matches!(
        ast.node(expr),
        Node::Binary {
            op: BinaryOp::Mul,
            ..
        }
    ));
}

#[test]
fn star_colon_is_a_parent_slot() {
    let ast = parse("{ parent*: P }");
    let obj = the(&ast, is_object);
    assert_eq!(slot_kinds(&ast, object(&ast, obj)), vec![SlotKind::Parent]);
}

#[test]
fn an_array_statement_stays_a_statement() {
    let ast = parse("{ [1, 2] }");
    let obj = the(&ast, is_object);
    let parts = object(&ast, obj);
    assert!(ast.list_items(parts.slots).is_empty());
    assert_eq!(ast.object_body_stmts(parts).len(), 1);
}

#[test]
fn an_array_literal_after_a_colon_is_an_element_slot() {
    let ast = parse("{ [k]: v }");
    let obj = the(&ast, is_object);
    assert_eq!(slot_kinds(&ast, object(&ast, obj)), vec![SlotKind::Element]);
}

#[test]
fn operators_do_not_continue_across_a_newline() {
    // `count: 0` then the `||` parameter marker on its own line
    let ast = parse("{ count: 0\n||\nself.count }");
    let obj = the(&ast, is_object);
    let parts = object(&ast, obj);
    assert_eq!(slot_kinds(&ast, parts), vec![SlotKind::Named]);
    assert!(parts.params.is_some());
}

#[test]
fn same_line_or_is_a_binary_operator() {
    let ast = parse("{ 0 || 1 }");
    let obj = the(&ast, is_object);
    let parts = object(&ast, obj);
    assert!(ast.list_items(parts.slots).is_empty());
    let stmts = ast.object_body_stmts(parts);
    let Node::ExprStmt { expr } = *ast.node(stmts[0]) else {
        panic!()
    };
    assert!(matches!(
        ast.node(expr),
        Node::Binary {
            op: BinaryOp::Or,
            ..
        }
    ));
}

#[test]
fn value_on_the_next_line_starts_a_new_entry() {
    let ast = parse("{ m: 1\n- 2 }");
    let obj = the(&ast, is_object);
    let parts = object(&ast, obj);
    assert_eq!(slot_kinds(&ast, parts), vec![SlotKind::Named]);
    assert_eq!(ast.object_body_stmts(parts).len(), 1);
    let Node::ExprStmt { expr } = *ast.node(ast.object_body_stmts(parts)[0]) else {
        panic!()
    };
    assert!(matches!(ast.node(expr), Node::Unary { .. }));
}

// -- errors ------------------------------------------------------------------

#[test]
fn params_must_precede_the_body() {
    let err = parse_err("{ f()\n|a| a }");
    assert!(err.message.contains("params must precede"));
}

#[test]
fn slots_must_precede_the_body() {
    let err = parse_err("{ f()\nx: 1 }");
    assert!(err.message.contains("slots must precede"));
}

#[test]
fn a_colon_after_a_full_expression_is_an_error() {
    let err = parse_err("{ x * y : 1 }");
    assert!(err.message.contains("`;` or newline"));
}

#[test]
fn an_element_key_is_a_single_expression() {
    let err = parse_err("{ [a, b]: 1 }");
    assert!(err.message.contains("element slot key"));
}

#[test]
fn while_is_no_longer_a_keyword() {
    // `while` parses as an identifier; the construct is gone
    assert!(
        parse_err("while c { 1 }")
            .message
            .contains("`;` or newline")
    );
    assert!(
        parse_err("match x { 1 }")
            .message
            .contains("`;` or newline")
    );
    assert!(
        parse_err("try { 1 } catch e { 2 }")
            .message
            .contains("`;` or newline")
    );
    assert!(
        parse_err("for x in xs { 1 }")
            .message
            .contains("`;` or newline")
    );
}

// -- if and misc --------------------------------------------------------------

#[test]
fn if_branches_are_statement_lists() {
    let ast = parse("if c { 1 } else { 2 }");
    let if_node = the(&ast, |n| matches!(n, Node::If { .. }));
    let Node::If { then, else_, .. } = *ast.node(if_node) else {
        panic!()
    };
    assert!(matches!(ast.node(then), Node::StmtList { .. }));
    assert!(else_.is_some_and(|e| matches!(ast.node(e), Node::StmtList { .. })));
}

#[test]
fn else_if_chains() {
    let ast = parse("if a { 1 } else if b { 2 } else { 3 }");
    let ifs = matching(&ast, |n| matches!(n, Node::If { .. }));
    assert_eq!(ifs.len(), 2);
}

#[test]
fn call_and_send_and_get() {
    let ast = parse("f(1)\ng.m(2)\nh.x\na[0]\na.0");
    assert_eq!(matching(&ast, |n| matches!(n, Node::Call { .. })).len(), 1);
    assert_eq!(matching(&ast, |n| matches!(n, Node::Send { .. })).len(), 1);
    assert_eq!(matching(&ast, |n| matches!(n, Node::Get { .. })).len(), 1);
    assert_eq!(matching(&ast, |n| matches!(n, Node::Index { .. })).len(), 2);
}

#[test]
fn arrays() {
    let ast = parse("[10, 20, 30]");
    let arr = the(&ast, |n| matches!(n, Node::Array { .. }));
    let Node::Array { elements } = *ast.node(arr) else {
        panic!()
    };
    assert_eq!(ast.list_items(elements).len(), 3);
}
