use parser::Node::*;
use parser::TokenKind::{self};
use parser::{Ast, FunctionId, Node, NodeId, ParseError, Parser, Symbol, Utf8SliceStream};

fn parse(src: &str) -> Ast {
    let mut p = Parser::new(Utf8SliceStream::new(src));
    p.parse_script()
        .unwrap_or_else(|e| panic!("parse error in {src:?}: {e}"));
    p.into_ast()
}

fn parse_err(src: &str) -> ParseError {
    let mut p = Parser::new(Utf8SliceStream::new(src));
    p.parse_script()
        .expect_err(&format!("expected error for {src:?}"))
}

/// statements of the implicit top-level function
fn stmts(ast: &Ast) -> &[NodeId] {
    let f = ast.function(FunctionId(0));
    match ast.node(f.body.unwrap()) {
        Block { stmts } => ast.list_items(*stmts),
        n => panic!("expected block, got {n:?}"),
    }
}

fn stmt(ast: &Ast, i: usize) -> &Node {
    ast.node(stmts(ast)[i])
}

/// unwrap ExprStmt → inner expression
fn expr(ast: &Ast, i: usize) -> &Node {
    match stmt(ast, i) {
        ExprStmt { expr } => ast.node(*expr),
        n => panic!("expected ExprStmt, got {n:?}"),
    }
}

fn ident_sym(n: &Node) -> Symbol {
    match n {
        Identifier { sym } => *sym,
        n => panic!("expected Identifier, got {n:?}"),
    }
}

fn sym_text(ast: &Ast, sym: Symbol) -> &[u8] {
    ast.symbol(sym)
}

// -- expressions -------------------------------------------------------------

#[test]
fn literals() {
    let ast = parse("1; 'two'; true; false; null;");
    assert!(matches!(expr(&ast, 0), NumberLiteral(1.0)));
    assert!(matches!(expr(&ast, 1), StringLiteral(_)));
    assert!(matches!(expr(&ast, 2), BoolLiteral(true)));
    assert!(matches!(expr(&ast, 3), BoolLiteral(false)));
    assert!(matches!(expr(&ast, 4), NullLiteral));
}

#[test]
fn precedence_mul_before_add() {
    let ast = parse("1 + 2 * 3;");
    let Binary { op, lhs, rhs } = *expr(&ast, 0) else {
        panic!()
    };
    assert_eq!(op, TokenKind::Plus);
    assert!(matches!(ast.node(lhs), NumberLiteral(1.0)));
    let Binary { op, lhs, rhs } = *ast.node(rhs) else {
        panic!()
    };
    assert_eq!(op, TokenKind::Star);
    assert!(matches!(ast.node(lhs), NumberLiteral(2.0)));
    assert!(matches!(ast.node(rhs), NumberLiteral(3.0)));
}

#[test]
fn precedence_parens_override() {
    let ast = parse("(1 + 2) * 3;");
    let Binary { op, lhs, .. } = *expr(&ast, 0) else {
        panic!()
    };
    assert_eq!(op, TokenKind::Star);
    assert!(matches!(
        ast.node(lhs),
        Binary {
            op: TokenKind::Plus,
            ..
        }
    ));
}

#[test]
fn precedence_comparison_over_equality() {
    // 1 < 2 == 3 parses as (1 < 2) == 3
    let ast = parse("1 < 2 == 3;");
    let Binary { op, lhs, .. } = *expr(&ast, 0) else {
        panic!()
    };
    assert_eq!(op, TokenKind::EqEq);
    assert!(matches!(
        ast.node(lhs),
        Binary {
            op: TokenKind::Lt,
            ..
        }
    ));
}

#[test]
fn assignment_is_right_associative() {
    let ast = parse("a = b = c;");
    let Assign { target, value, .. } = *expr(&ast, 0) else {
        panic!()
    };
    assert_eq!(sym_text(&ast, ident_sym(ast.node(target))), b"a");
    let Assign { target, value, .. } = *ast.node(value) else {
        panic!()
    };
    assert_eq!(sym_text(&ast, ident_sym(ast.node(target))), b"b");
    assert_eq!(sym_text(&ast, ident_sym(ast.node(value))), b"c");
}

#[test]
fn exponent_is_right_associative() {
    let ast = parse("2 ** 3 ** 4;");
    let Binary { op, rhs, .. } = *expr(&ast, 0) else {
        panic!()
    };
    assert_eq!(op, TokenKind::StarStar);
    assert!(matches!(
        ast.node(rhs),
        Binary {
            op: TokenKind::StarStar,
            ..
        }
    ));
}

#[test]
fn compound_assignment() {
    let ast = parse("a += 1; a.b **= 2;");
    assert!(matches!(
        expr(&ast, 0),
        Assign {
            op: TokenKind::PlusAssign,
            ..
        }
    ));
    let Assign { op, target, .. } = *expr(&ast, 1) else {
        panic!()
    };
    assert_eq!(op, TokenKind::StarStarAssign);
    assert!(matches!(ast.node(target), Property { .. }));
}

#[test]
fn invalid_assignment_target() {
    parse_err("1 = 2;");
    parse_err("a + b = c;");
    parse_err("(a)++ = 1;");
}

#[test]
fn unary_and_update() {
    let ast = parse("-x; !x; typeof x; ++x; x++; x--;");
    assert!(matches!(
        expr(&ast, 0),
        Unary {
            op: TokenKind::Minus,
            ..
        }
    ));
    assert!(matches!(
        expr(&ast, 1),
        Unary {
            op: TokenKind::Bang,
            ..
        }
    ));
    assert!(matches!(
        expr(&ast, 2),
        Unary {
            op: TokenKind::Typeof,
            ..
        }
    ));
    assert!(matches!(
        expr(&ast, 3),
        Update {
            prefix: true,
            op: TokenKind::PlusPlus,
            ..
        }
    ));
    assert!(matches!(
        expr(&ast, 4),
        Update {
            prefix: false,
            op: TokenKind::PlusPlus,
            ..
        }
    ));
    assert!(matches!(
        expr(&ast, 5),
        Update {
            prefix: false,
            op: TokenKind::MinusMinus,
            ..
        }
    ));
}

#[test]
fn conditional() {
    let ast = parse("a ? b : c;");
    let Conditional { cond, then, else_ } = *expr(&ast, 0) else {
        panic!()
    };
    assert_eq!(sym_text(&ast, ident_sym(ast.node(cond))), b"a");
    assert_eq!(sym_text(&ast, ident_sym(ast.node(then))), b"b");
    assert_eq!(sym_text(&ast, ident_sym(ast.node(else_))), b"c");
}

#[test]
fn member_call_chain() {
    // a.b[c](d).e
    let ast = parse("a.b[c](d).e;");
    let Property {
        object,
        key,
        computed,
    } = *expr(&ast, 0)
    else {
        panic!()
    };
    assert!(!computed);
    assert!(matches!(ast.node(key), StringLiteral(_)));
    let Call { callee, args } = *ast.node(object) else {
        panic!()
    };
    assert_eq!(ast.list_items(args).len(), 1);
    let Property {
        object, computed, ..
    } = *ast.node(callee)
    else {
        panic!()
    };
    assert!(computed);
    assert!(matches!(
        ast.node(object),
        Property {
            computed: false,
            ..
        }
    ));
}

#[test]
fn keyword_property_names() {
    let ast = parse("a.class; a.function;");
    for i in 0..2 {
        let Property { key, .. } = *expr(&ast, i) else {
            panic!()
        };
        assert!(matches!(ast.node(key), StringLiteral(_)));
    }
    let ast = parse("a.class;");
    let Property { key, .. } = *expr(&ast, 0) else {
        panic!()
    };
    let StringLiteral(sym) = *ast.node(key) else {
        panic!()
    };
    assert_eq!(sym_text(&ast, sym), b"class");
}

#[test]
fn array_literal_with_holes() {
    let ast = parse("[1, , 2,];");
    let ArrayLiteral { elements } = *expr(&ast, 0) else {
        panic!()
    };
    let items = ast.list_items(elements);
    assert_eq!(items.len(), 3);
    assert!(matches!(ast.node(items[0]), NumberLiteral(1.0)));
    assert!(matches!(ast.node(items[1]), Hole));
    assert!(matches!(ast.node(items[2]), NumberLiteral(2.0)));
}

#[test]
fn object_literal_forms() {
    let ast = parse("({a: 1, b, 'c': 2, 3: 4, if: 5});");
    let ObjectLiteral { props } = *expr(&ast, 0) else {
        panic!()
    };
    let items = ast.list_items(props);
    assert_eq!(items.len(), 5);
    // shorthand b: value is an Identifier reference
    let ObjectProperty { key, value, .. } = *ast.node(items[1]) else {
        panic!()
    };
    let StringLiteral(k) = *ast.node(key) else {
        panic!()
    };
    assert_eq!(sym_text(&ast, k), b"b");
    let Identifier { sym } = *ast.node(value) else {
        panic!()
    };
    assert_eq!(sym_text(&ast, sym), b"b");
    // number key
    let ObjectProperty { key, .. } = *ast.node(items[3]) else {
        panic!()
    };
    assert!(matches!(ast.node(key), NumberLiteral(3.0)));
    // keyword key
    let ObjectProperty { key, .. } = *ast.node(items[4]) else {
        panic!()
    };
    let StringLiteral(k) = *ast.node(key) else {
        panic!()
    };
    assert_eq!(sym_text(&ast, k), b"if");
}

#[test]
fn braces_are_blocks_in_statement_position() {
    let ast = parse("{}");
    assert!(matches!(stmt(&ast, 0), Block { .. }));
    // `{ a: 1 }` is a block holding a labeled statement, not an object literal
    let ast = parse("{ a: 1 }");
    let Block { stmts } = *stmt(&ast, 0) else {
        panic!()
    };
    let inner = ast.list_items(stmts);
    assert_eq!(inner.len(), 1);
    assert!(matches!(ast.node(inner[0]), Labeled { .. }));
}

// -- ASI -----------------------------------------------------------------------

#[test]
fn asi_inserts_semicolons() {
    let ast = parse("a\nb");
    assert_eq!(stmts(&ast).len(), 2);
    let ast = parse("var x = 1\nvar y = 2");
    assert_eq!(stmts(&ast).len(), 2);
    // same line without `;` is an error
    parse_err("a b");
}

#[test]
fn asi_call_continues_across_newline() {
    // `a\n(b)` is a call, not two statements — ASI only applies when the
    // next token cannot continue the expression
    let ast = parse("a\n(b)");
    assert_eq!(stmts(&ast).len(), 1);
    assert!(matches!(expr(&ast, 0), Call { .. }));
}

#[test]
fn asi_restricted_postfix_update() {
    // x \n ++y  is two statements: `x;` and `++y;`
    let ast = parse("x\n++y");
    assert_eq!(stmts(&ast).len(), 2);
    assert!(matches!(expr(&ast, 0), Identifier { .. }));
    assert!(matches!(expr(&ast, 1), Update { prefix: true, .. }));
    // x++ \n y is postfix
    let ast = parse("x++\ny");
    assert!(matches!(expr(&ast, 0), Update { prefix: false, .. }));
}

#[test]
fn asi_restricted_return() {
    let ast = parse("function f() { return\n5; }");
    let FunctionDecl { function } = *stmt(&ast, 0) else {
        panic!()
    };
    let f = ast.function(function);
    let Block { stmts } = ast.node(f.body.unwrap()) else {
        panic!()
    };
    let body = ast.list_items(*stmts);
    let Return { value } = *ast.node(body[0]) else {
        panic!()
    };
    assert!(value.is_none());
    assert!(matches!(ast.node(body[1]), ExprStmt { .. }));
}

// -- var declarations ------------------------------------------------------------

#[test]
fn var_declarations() {
    let ast = parse("var x = 1, y; let z = 2; const w = 3;");
    let VarDecl { kind, decls } = *stmt(&ast, 0) else {
        panic!()
    };
    assert_eq!(kind, parser::VarKind::Var);
    let decls = ast.list_items(decls);
    assert_eq!(decls.len(), 2);
    let VarDeclarator { target, init } = *ast.node(decls[0]) else {
        panic!()
    };
    let Identifier { sym } = *ast.node(target) else {
        panic!()
    };
    assert_eq!(sym_text(&ast, sym), b"x");
    assert!(init.is_some());
    let VarDeclarator { target, init } = *ast.node(decls[1]) else {
        panic!()
    };
    let Identifier { sym } = *ast.node(target) else {
        panic!()
    };
    assert_eq!(sym_text(&ast, sym), b"y");
    assert!(init.is_none());
    assert!(matches!(
        stmt(&ast, 1),
        VarDecl {
            kind: parser::VarKind::Let,
            ..
        }
    ));
    assert!(matches!(
        stmt(&ast, 2),
        VarDecl {
            kind: parser::VarKind::Const,
            ..
        }
    ));
}

#[test]
fn declaration_errors() {
    parse_err("const x;"); // missing initializer
    parse_err("let x; let x;"); // duplicate lexical
    parse_err("let x; var x;"); // var conflicts with lexical
    parse_err("var x; let x;"); // and vice versa
    parse("var x; var x;"); // var redeclare is fine
    parse("{ let x; } { let x; }"); // different blocks are fine
    parse("{ var x; } var x;"); // var hoists out of blocks
    parse("{ let x; } var x;"); // block is over, no conflict
}

#[test]
fn let_is_contextual() {
    // sloppy mode: let can be an identifier
    let ast = parse("let = 5;");
    assert!(matches!(
        expr(&ast, 0),
        Assign {
            op: TokenKind::Assign,
            ..
        }
    ));
    let ast = parse("let.x = 5;");
    assert!(matches!(expr(&ast, 0), Assign { .. }));
    // but `let x` is a declaration
    let ast = parse("let x = 5;");
    assert!(matches!(
        stmt(&ast, 0),
        VarDecl {
            kind: parser::VarKind::Let,
            ..
        }
    ));
}

// -- control flow ------------------------------------------------------------------

#[test]
fn if_while_for() {
    let ast = parse("if (a) b; else c;");
    let If { cond, then, else_ } = *stmt(&ast, 0) else {
        panic!()
    };
    assert!(matches!(ast.node(cond), Identifier { .. }));
    assert!(else_.is_some());
    let If { else_, .. } = *stmt(&ast, 0) else {
        panic!()
    };
    assert!(else_.is_some());
    let _ = then;

    let ast = parse("if (a) b;");
    let If { else_, .. } = *stmt(&ast, 0) else {
        panic!()
    };
    assert!(else_.is_none());

    let ast = parse("while (a < 10) { a++; }");
    let While { cond, body } = *stmt(&ast, 0) else {
        panic!()
    };
    assert!(matches!(
        ast.node(cond),
        Binary {
            op: TokenKind::Lt,
            ..
        }
    ));
    assert!(matches!(ast.node(body), Block { .. }));

    let ast = parse("for (var i = 0; i < 10; i++) {}");
    let For {
        init,
        cond,
        next,
        body,
    } = *stmt(&ast, 0)
    else {
        panic!()
    };
    assert!(matches!(ast.node(init.unwrap()), VarDecl { .. }));
    assert!(matches!(
        ast.node(cond.unwrap()),
        Binary {
            op: TokenKind::Lt,
            ..
        }
    ));
    assert!(matches!(ast.node(next.unwrap()), Update { .. }));
    assert!(matches!(ast.node(body), Block { .. }));

    let ast = parse("for (;;) {}");
    let For {
        init, cond, next, ..
    } = *stmt(&ast, 0)
    else {
        panic!()
    };
    assert!(init.is_none() && cond.is_none() && next.is_none());
}

#[test]
fn break_continue_need_loop() {
    parse("while (a) { break; continue; }");
    assert!(matches!(stmt(&parse("while(a){break;}"), 0), While { .. }));
    parse_err("break;");
    parse_err("continue;");
    parse_err("function f() { break; }"); // loop depth resets in functions
    // labeled break/continue are supported; target existence is checked in codegen
    parse("outer: while (a) { break outer; }");
    parse("outer: while (a) { continue outer; }");
    let ast = parse("outer: while (a) { break outer; }");
    let Labeled { label, body } = *stmt(&ast, 0) else {
        panic!()
    };
    assert_eq!(sym_text(&ast, label), b"outer");
    assert!(matches!(ast.node(body), While { .. }));
}

// -- functions ----------------------------------------------------------------------

#[test]
fn function_declaration() {
    let ast = parse("function add(a, b) { return a + b; }");
    let FunctionDecl { function } = *stmt(&ast, 0) else {
        panic!()
    };
    let f = ast.function(function);
    assert_eq!(sym_text(&ast, f.name.unwrap()), b"add");
    assert!(f.is_declaration);
    assert_eq!(f.params.len(), 2);
    let Identifier { sym } = *ast.node(f.params[0].target) else {
        panic!()
    };
    assert_eq!(sym_text(&ast, sym), b"a");
    let Block { stmts } = ast.node(f.body.unwrap()) else {
        panic!()
    };
    let Return { value } = *ast.node(ast.list_items(*stmts)[0]) else {
        panic!()
    };
    assert!(matches!(
        ast.node(value.unwrap()),
        Binary {
            op: TokenKind::Plus,
            ..
        }
    ));
}

#[test]
fn function_expression_and_trailing_comma() {
    let ast = parse("var f = function(a,) { return a; };");
    let VarDecl { decls, .. } = *stmt(&ast, 0) else {
        panic!()
    };
    let VarDeclarator { init, .. } = *ast.node(ast.list_items(decls)[0]) else {
        panic!()
    };
    let FunctionExpr { function } = *ast.node(init.unwrap()) else {
        panic!()
    };
    assert!(ast.function(function).name.is_none());
    assert_eq!(ast.function(function).params.len(), 1);
}

#[test]
fn nested_functions_get_distinct_literal_ids() {
    let ast = parse("function outer() { function inner() {} return inner; }");
    let FunctionDecl { function } = *stmt(&ast, 0) else {
        panic!()
    };
    let outer = ast.function(function);
    assert_eq!(outer.literal_id, 1); // top-level script is 0
    let Block { stmts } = ast.node(outer.body.unwrap()) else {
        panic!()
    };
    let FunctionDecl { function } = *ast.node(ast.list_items(*stmts)[0]) else {
        panic!()
    };
    let inner = ast.function(function);
    assert_eq!(inner.literal_id, 2);
    assert_ne!(function, FunctionId(0));
}

#[test]
fn function_param_conflicts() {
    parse_err("function f(a) { let a; }"); // param vs let
    parse("function f(a) { var a; }"); // param vs var is fine
}

#[test]
fn anonymous_function_declaration_is_an_error() {
    parse_err("function () {}");
    parse_err("function f(");
    parse_err("function f(a, b { }");
}

// -- strict mode ---------------------------------------------------------------------

#[test]
fn directive_prologue_sets_strict() {
    let ast = parse("'use strict'; 1;");
    assert!(ast.function(FunctionId(0)).strict);

    let ast = parse("1; 'use strict';");
    assert!(!ast.function(FunctionId(0)).strict);

    let ast = parse("\"use strict\";");
    assert!(ast.function(FunctionId(0)).strict);

    // not a bare literal statement → not a directive
    let ast = parse("'use strict' + x;");
    assert!(!ast.function(FunctionId(0)).strict);

    // nested function has its own prologue
    let ast = parse("function f() { 'use strict'; }");
    assert!(!ast.function(FunctionId(0)).strict);
    let FunctionDecl { function } = *stmt(&ast, 0) else {
        panic!()
    };
    assert!(ast.function(function).strict);
}

// -- errors ----------------------------------------------------------------------------

#[test]
fn syntax_errors() {
    parse_err("(");
    parse_err("1 +");
    parse_err("var = 1;");
    parse_err("if (a) ");
    parse_err("while a) {}");
    let _ = parse(""); // empty script is fine
    let e = parse_err("1 +");
    assert!(e.span.start <= e.span.end);
}

#[test]
fn top_level_return_is_allowed() {
    // the script body is an implicit function
    parse("return;");
    parse("if (x) return 1;");
}

#[test]
fn spans_cover_source() {
    let ast = parse("var x = 1;");
    let VarDecl { .. } = *stmt(&ast, 0) else {
        panic!()
    };
    assert_eq!(ast.span(stmts(&ast)[0]), parser::Span::new(0, 9));
    let f = ast.function(FunctionId(0));
    assert_eq!(f.span, parser::Span::new(0, 10));
}

// -- try/catch/throw ---------------------------------------------------------------

#[test]
fn new_expressions() {
    let ast = parse("new f;");
    assert!(matches!(expr(&ast, 0), New { args: None, .. }));

    let ast = parse("new f(1, 2);");
    let New { args, .. } = *expr(&ast, 0) else {
        panic!()
    };
    assert_eq!(ast.list_items(args.unwrap()).len(), 2);

    // callee binds member tails but not parens: new (a.b)(1)
    let ast = parse("new a.b(1);");
    let New { callee, .. } = *expr(&ast, 0) else {
        panic!()
    };
    assert!(matches!(ast.node(callee), Property { .. }));

    // (new a()).b
    let ast = parse("new a().b;");
    let Property { object, .. } = *expr(&ast, 0) else {
        panic!()
    };
    assert!(matches!(ast.node(object), New { args: Some(_), .. }));

    // nested new: `new new f()()` = new (new f()) ()
    let ast = parse("new new f()();");
    let New { callee, args } = *expr(&ast, 0) else {
        panic!()
    };
    assert!(args.is_some());
    assert!(matches!(ast.node(callee), New { args: Some(_), .. }));
}

#[test]
fn try_catch_forms() {
    let ast = parse("try { a(); } catch (e) { b(e); }");
    let TryCatch {
        catch_param,
        catch_block,
        finally_block,
        ..
    } = *stmt(&ast, 0)
    else {
        panic!()
    };
    let Identifier { sym } = *ast.node(catch_param.unwrap()) else {
        panic!()
    };
    assert_eq!(sym_text(&ast, sym), b"e");
    assert!(catch_block.is_some());
    assert!(finally_block.is_none());

    let ast = parse("try {} finally {}");
    let TryCatch {
        catch_block,
        finally_block,
        ..
    } = *stmt(&ast, 0)
    else {
        panic!()
    };
    assert!(catch_block.is_none());
    assert!(finally_block.is_some());

    // catch without binding (ES2019)
    let ast = parse("try {} catch {}");
    let TryCatch {
        catch_param,
        catch_block,
        ..
    } = *stmt(&ast, 0)
    else {
        panic!()
    };
    assert!(catch_param.is_none());
    assert!(catch_block.is_some());

    // catch param scopes the block: no conflict with outer let
    parse("let e; try {} catch (e) { let x = e; }");
    // but duplicates inside the same catch scope are caught
    parse_err("try {} catch (e) { let e; }");
}

#[test]
fn throw_statement() {
    let ast = parse("throw err;");
    let Throw { .. } = *stmt(&ast, 0) else {
        panic!()
    };
    // restricted production: newline before the argument is an error
    parse_err("throw\n1;");
    parse_err("try {}");
}

// -- bigint, spread, accessors, methods, computed keys ----------------------------

#[test]
fn bigint_literals() {
    let ast = parse("1n; 9007199254740993n;");
    let BigIntLiteral(sym) = *expr(&ast, 0) else {
        panic!()
    };
    assert_eq!(sym_text(&ast, sym), b"1");
    let BigIntLiteral(sym) = *expr(&ast, 1) else {
        panic!()
    };
    assert_eq!(sym_text(&ast, sym), b"9007199254740993");
    // floats/exponents with n are not bigint
    parse_err("1.5n;");
    parse_err("1e3n;");
}

#[test]
fn spread_in_array_call_object() {
    let ast = parse("[...a, 1];");
    let ArrayLiteral { elements } = *expr(&ast, 0) else {
        panic!()
    };
    let items = ast.list_items(elements);
    assert_eq!(items.len(), 2);
    assert!(matches!(ast.node(items[0]), Spread { .. }));
    assert!(matches!(ast.node(items[1]), NumberLiteral(1.0)));

    let ast = parse("f(...a, 1);");
    let Call { args, .. } = *expr(&ast, 0) else {
        panic!()
    };
    let items = ast.list_items(args);
    assert!(matches!(ast.node(items[0]), Spread { .. }));
    assert!(matches!(ast.node(items[1]), NumberLiteral(1.0)));

    let ast = parse("({...o, a: 1});");
    let ObjectLiteral { props } = *expr(&ast, 0) else {
        panic!()
    };
    let items = ast.list_items(props);
    assert!(matches!(ast.node(items[0]), Spread { .. }));
    assert!(matches!(ast.node(items[1]), ObjectProperty { .. }));
}

#[test]
fn getters_setters_methods() {
    let ast = parse("({ get x() { return 1; }, set x(v) {}, m(a) { return a; } });");
    let ObjectLiteral { props } = *expr(&ast, 0) else {
        panic!()
    };
    let items = ast.list_items(props);
    assert_eq!(items.len(), 3);

    let ObjectProperty { kind, value, .. } = *ast.node(items[0]) else {
        panic!()
    };
    assert_eq!(kind, parser::PropKind::Get);
    let FunctionExpr { function } = *ast.node(value) else {
        panic!()
    };
    assert_eq!(ast.function(function).params.len(), 0);

    let ObjectProperty { kind, value, .. } = *ast.node(items[1]) else {
        panic!()
    };
    assert_eq!(kind, parser::PropKind::Set);
    let FunctionExpr { function } = *ast.node(value) else {
        panic!()
    };
    assert_eq!(ast.function(function).params.len(), 1);

    let ObjectProperty { kind, .. } = *ast.node(items[2]) else {
        panic!()
    };
    assert_eq!(kind, parser::PropKind::Method);

    // accessor arity is an early error
    parse_err("({ get x(a) {} });");
    parse_err("({ set x() {} });");
    parse_err("({ set x(a, b) {} });");

    // `get`/`set` as normal properties still work
    parse("({ get: 1, set: 2 });");
    parse("({ get, set });");
    let ast = parse("({ get() {} });");
    let ObjectLiteral { props } = *expr(&ast, 0) else {
        panic!()
    };
    let ObjectProperty { kind, .. } = *ast.node(ast.list_items(props)[0]) else {
        panic!()
    };
    assert_eq!(kind, parser::PropKind::Method); // method NAMED get
}

#[test]
fn computed_property_keys() {
    let ast = parse("({ [k]: 1, [m]() {}, [g]: 2 });");
    let ObjectLiteral { props } = *expr(&ast, 0) else {
        panic!()
    };
    let items = ast.list_items(props);
    let ObjectProperty { computed, kind, .. } = *ast.node(items[0]) else {
        panic!()
    };
    assert!(computed);
    assert_eq!(kind, parser::PropKind::Init);
    let ObjectProperty { computed, kind, .. } = *ast.node(items[1]) else {
        panic!()
    };
    assert!(computed);
    assert_eq!(kind, parser::PropKind::Method);
    // shorthand with computed key is an error
    parse_err("({ [k] });");
}

// -- classes -----------------------------------------------------------------------

#[test]
fn class_declaration() {
    let ast = parse("class Point extends Base { constructor(x, y) { this.x = x; } }");
    let ClassDecl { class } = *stmt(&ast, 0) else {
        panic!()
    };
    let c = ast.class(class);
    assert_eq!(sym_text(&ast, c.name.unwrap()), b"Point");
    assert!(c.superclass.is_some());
    assert_eq!(c.members.len(), 1);
    assert!(c.members[0].is_constructor);
    assert_eq!(c.members[0].kind, parser::PropKind::Method);
    let FunctionExpr { function } = *ast.node(c.members[0].value) else {
        panic!()
    };
    assert_eq!(ast.function(function).params.len(), 2);
    assert_eq!(
        ast.function(function).kind,
        parser::FunctionKind::DerivedClassConstructor
    );
    // class bodies are always strict
    assert!(ast.function(function).strict);
}

#[test]
fn class_expression_and_members() {
    let ast =
        parse("var C = class { static make() {} get v() { return 1; } set v(x) {} ['m']() {} };");
    let VarDecl { decls, .. } = *stmt(&ast, 0) else {
        panic!()
    };
    let VarDeclarator { init, .. } = *ast.node(ast.list_items(decls)[0]) else {
        panic!()
    };
    let ClassExpr { class } = *ast.node(init.unwrap()) else {
        panic!()
    };
    let c = ast.class(class);
    assert!(c.name.is_none());
    assert!(c.superclass.is_none());
    assert_eq!(c.members.len(), 4);
    assert!(c.members[0].is_static && !c.members[0].computed);
    assert_eq!(c.members[1].kind, parser::PropKind::Get);
    assert_eq!(c.members[2].kind, parser::PropKind::Set);
    assert!(c.members[3].computed);
    let expected = [
        parser::FunctionKind::Method,
        parser::FunctionKind::Getter,
        parser::FunctionKind::Setter,
        parser::FunctionKind::Method,
    ];
    for (member, expected) in c.members.iter().zip(expected) {
        let FunctionExpr { function } = *ast.node(member.value) else {
            panic!()
        };
        assert_eq!(ast.function(function).kind, expected);
    }
}

#[test]
fn class_early_errors() {
    parse_err("class C { constructor() {} constructor() {} }"); // duplicate
    parse_err("class C { get constructor() {} }");
    parse_err("class C { static prototype() {} }"); // ES 15.7.1
    parse_err("class C { static { } }"); // static blocks not supported yet
    parse_err("class { }"); // declaration needs a name
    parse_err("class C { get x(a) {} }");
    parse_err("class C { set x() {} }");
    // field early errors (ES 15.7.1)
    parse_err("class C { constructor = 1; }"); // field named constructor
    parse_err("class C { static prototype = 1; }"); // static field named prototype
    parse_err("class C { #x; #x; }"); // duplicate private name
    parse_err("class C { m() { this.#y; } }"); // undeclared private name
    parse_err("this.#x;"); // private name outside a class
    // fields now parse
    parse("class C { x = 1; static y = 2; #z = 3; [w] = 4; static [v] = 5; }");
    // `static` as a member name still works
    parse("class C { static() {} }");
    // a static member named "constructor" is an ordinary method
    parse("class C { static constructor() {} }");
    // class declarations are lexical
    parse_err("class C {} class C {}");
    parse_err("class C {} var C;");
}

// -- generators & arrows ---------------------------------------------------------

#[test]
fn generator_functions() {
    let ast = parse("function* g() {} var h = function*() {};");
    let FunctionDecl { function } = *stmt(&ast, 0) else {
        panic!()
    };
    assert!(ast.function(function).kind.is_generator());
    assert!(!ast.function(function).kind.is_arrow());
    let VarDecl { decls, .. } = *stmt(&ast, 1) else {
        panic!()
    };
    let VarDeclarator { init, .. } = *ast.node(ast.list_items(decls)[0]) else {
        panic!()
    };
    let FunctionExpr { function } = *ast.node(init.unwrap()) else {
        panic!()
    };
    assert!(ast.function(function).kind.is_generator());

    // plain functions are unaffected
    let ast = parse("function f() {}");
    let FunctionDecl { function } = *stmt(&ast, 0) else {
        panic!()
    };
    assert!(!ast.function(function).kind.is_generator());
}

#[test]
fn arrow_functions() {
    // single ident param, expression body gets an implicit return
    let ast = parse("x => x + 1;");
    let FunctionExpr { function } = *expr(&ast, 0) else {
        panic!()
    };
    let f = ast.function(function);
    assert!(f.kind.is_arrow());
    assert_eq!(f.params.len(), 1);
    let Block { stmts } = ast.node(f.body.unwrap()) else {
        panic!()
    };
    let Return { value } = *ast.node(ast.list_items(*stmts)[0]) else {
        panic!()
    };
    assert!(matches!(
        ast.node(value.unwrap()),
        Binary {
            op: TokenKind::Plus,
            ..
        }
    ));

    // parenthesized params
    let ast = parse("(a, b) => a * b;");
    let FunctionExpr { function } = *expr(&ast, 0) else {
        panic!()
    };
    assert_eq!(ast.function(function).params.len(), 2);

    // empty params, block body
    let ast = parse("() => { return 42; };");
    let FunctionExpr { function } = *expr(&ast, 0) else {
        panic!()
    };
    assert!(ast.function(function).params.is_empty());

    // trailing comma
    parse("(a, b,) => 0;");

    // arrows nest (right-associative bodies)
    let ast = parse("x => y => x;");
    let FunctionExpr { function } = *expr(&ast, 0) else {
        panic!()
    };
    let Block { stmts } = ast.node(ast.function(function).body.unwrap()) else {
        panic!()
    };
    let Return { value } = *ast.node(ast.list_items(*stmts)[0]) else {
        panic!()
    };
    assert!(matches!(ast.node(value.unwrap()), FunctionExpr { .. }));

    // parens without `=>` are still expressions
    let ast = parse("(a + b) * c;");
    assert!(matches!(
        expr(&ast, 0),
        Binary {
            op: TokenKind::Star,
            ..
        }
    ));

    // newline before => is an error
    parse_err("x\n=> 1;");
    parse_err("(a, b)\n=> 1;");
}
