use parser::{
    Ast, DeclKind, FunctionId, Node, Parser, ScopeId, ScopeKind, Symbol, Utf8SliceStream,
};

fn parse(src: &str) -> Ast {
    let mut p = Parser::new(Utf8SliceStream::new(src));
    p.parse_script()
        .unwrap_or_else(|e| panic!("parse error in {src:?}: {e}"));
    p.into_ast()
}

fn scope_of_body(ast: &Ast, f: FunctionId) -> ScopeId {
    ast.node_scope(ast.function(f).body.unwrap())
        .expect("function body must have a scope")
}

fn decl(ast: &Ast, scope: ScopeId, name: &str) -> Option<DeclKind> {
    ast.scope(scope)
        .decls
        .iter()
        .find(|d| ast.symbol(d.name) == name.as_bytes())
        .map(|d| d.kind)
}

fn decls_in(ast: &Ast, scope: ScopeId) -> Vec<(String, DeclKind)> {
    ast.scope(scope)
        .decls
        .iter()
        .map(|d| {
            (
                String::from_utf8_lossy(ast.symbol(d.name)).into_owned(),
                d.kind,
            )
        })
        .collect()
}

/// collect all scopes with the given kind
fn scopes_of_kind(ast: &Ast, kind: ScopeKind) -> Vec<ScopeId> {
    (0..ast.scope_count())
        .map(|i| ScopeId(i as u32))
        .filter(|&s| ast.scope(s).kind == kind)
        .collect()
}

fn block_scope_containing(ast: &Ast, name: &str) -> ScopeId {
    *scopes_of_kind(ast, ScopeKind::Block)
        .iter()
        .find(|&&s| decl(ast, s, name).is_some())
        .unwrap_or_else(|| panic!("no block scope declares {name}"))
}

#[test]
fn script_scope_is_function_zero() {
    let ast = parse("var x;");
    let scope = scope_of_body(&ast, FunctionId(0));
    assert_eq!(ast.scope(scope).kind, ScopeKind::Script);
    assert_eq!(ast.scope(scope).function, Some(FunctionId(0)));
    assert_eq!(decl(&ast, scope, "x"), Some(DeclKind::Var));
}

#[test]
fn var_hoists_to_function_scope_through_blocks() {
    let ast = parse("function f() { { { var x; } } let y; }");
    // find f: it's the function named "f"
    let f = (0..2)
        .map(FunctionId)
        .find(|&id| ast.function(id).name == Some(ast_symbol(&ast, "f")))
        .unwrap();
    let fscope = scope_of_body(&ast, f);
    assert_eq!(decl(&ast, fscope, "x"), Some(DeclKind::Var)); // hoisted
    assert_eq!(decl(&ast, fscope, "y"), Some(DeclKind::Let)); // body block IS the fn scope
    // x must NOT be in any block scope
    assert!(
        scopes_of_kind(&ast, ScopeKind::Block)
            .iter()
            .all(|&s| decl(&ast, s, "x").is_none())
    );
}

fn ast_symbol(ast: &Ast, s: &str) -> Symbol {
    (0..ast.symbol_count() as u32)
        .map(Symbol)
        .find(|&sym| ast.symbol(sym) == s.as_bytes())
        .unwrap()
}

#[test]
fn params_are_declared_in_function_scope() {
    let ast = parse("function f(a, b) { return a + b; }");
    let f = FunctionId(1); // after top-level
    let scope = scope_of_body(&ast, f);
    let decls = decls_in(&ast, scope);
    assert_eq!(
        decls,
        vec![
            ("a".to_string(), DeclKind::Param),
            ("b".to_string(), DeclKind::Param),
        ]
    );
}

#[test]
fn let_in_block_scope_not_outside() {
    let ast = parse("let outer; { let inner; }");
    let top = scope_of_body(&ast, FunctionId(0));
    assert_eq!(decl(&ast, top, "outer"), Some(DeclKind::Let));
    assert_eq!(decl(&ast, top, "inner"), None);
    let block = block_scope_containing(&ast, "inner");
    assert_eq!(ast.scope(block).parent, Some(top));
}

#[test]
fn for_head_scope_does_not_leak() {
    let ast = parse("for (let i = 0;;) {} let i;");
    let for_scopes = scopes_of_kind(&ast, ScopeKind::For);
    assert_eq!(for_scopes.len(), 1);
    assert_eq!(decl(&ast, for_scopes[0], "i"), Some(DeclKind::Let));
    // and the outer `let i` is a separate declaration in the script scope
    let top = scope_of_body(&ast, FunctionId(0));
    assert_eq!(decl(&ast, top, "i"), Some(DeclKind::Let));
}

#[test]
fn catch_param_in_catch_scope() {
    let ast = parse("try {} catch (e) { let x; }");
    let catch = scopes_of_kind(&ast, ScopeKind::Catch);
    assert_eq!(catch.len(), 1);
    assert_eq!(decl(&ast, catch[0], "e"), Some(DeclKind::CatchParam));
    assert_eq!(decl(&ast, catch[0], "x"), Some(DeclKind::Let));
}

#[test]
fn class_declaration_is_lexical() {
    let ast = parse("class C {}");
    let top = scope_of_body(&ast, FunctionId(0));
    assert_eq!(decl(&ast, top, "C"), Some(DeclKind::Class));
}

#[test]
fn function_decl_var_like_at_function_level_lexical_in_block() {
    let ast = parse("function f() {} function g() { { function h() {} } }");
    let top = scope_of_body(&ast, FunctionId(0));
    assert_eq!(decl(&ast, top, "f"), Some(DeclKind::Function));
    let g = (0..ast.function_count() as u32)
        .map(FunctionId)
        .find(|&id| ast.function(id).name == Some(ast_symbol(&ast, "g")))
        .unwrap();
    let gscope = scope_of_body(&ast, g);
    assert_eq!(decl(&ast, gscope, "h"), None); // h is in the block, not g's scope
    let block = block_scope_containing(&ast, "h");
    assert_eq!(decl(&ast, block, "h"), Some(DeclKind::Function));
}

#[test]
fn strict_directive_reaches_scope() {
    let ast = parse("'use strict'; function f() {}");
    let top = scope_of_body(&ast, FunctionId(0));
    assert!(ast.scope(top).strict);
    // function inherits enclosing strictness
    let f = scope_of_body(&ast, FunctionId(1));
    assert!(ast.scope(f).strict);

    let ast = parse("function f() { 'use strict'; }");
    let f = scope_of_body(&ast, FunctionId(1));
    assert!(ast.scope(f).strict);
    let top = scope_of_body(&ast, FunctionId(0));
    assert!(!ast.scope(top).strict);
}

#[test]
fn contains_function_or_eval_flag() {
    let ast = parse("var x = 1;");
    let top = scope_of_body(&ast, FunctionId(0));
    assert!(!ast.scope(top).contains_function_or_eval);

    let ast = parse("function f() {}");
    let top = scope_of_body(&ast, FunctionId(0));
    assert!(ast.scope(top).contains_function_or_eval);

    let ast = parse("var f = () => 1;");
    let top = scope_of_body(&ast, FunctionId(0));
    assert!(ast.scope(top).contains_function_or_eval);

    let ast = parse("eval('1');");
    let top = scope_of_body(&ast, FunctionId(0));
    assert!(ast.scope(top).calls_eval);
    assert!(ast.scope(top).contains_function_or_eval);

    // not a direct eval: member call
    let ast = parse("obj.eval('1');");
    let top = scope_of_body(&ast, FunctionId(0));
    assert!(!ast.scope(top).calls_eval);
}

#[test]
fn node_scope_links() {
    // every non-function-body block links to a Block scope, for loops to For
    let ast = parse("{ } for (;;) { } while (x) { }");
    let top_body = ast.function(FunctionId(0)).body.unwrap();
    let Node::Block { stmts } = ast.node(top_body) else {
        panic!()
    };
    let stmts = ast.list_items(*stmts);
    assert_eq!(
        ast.scope(ast.node_scope(stmts[0]).unwrap()).kind,
        ScopeKind::Block
    );
    assert_eq!(
        ast.scope(ast.node_scope(stmts[1]).unwrap()).kind,
        ScopeKind::For
    );
}
