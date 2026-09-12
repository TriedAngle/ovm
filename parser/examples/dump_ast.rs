//! Dump the AST of a JS file as an indented tree.
//! Usage: cargo run -p parser --example dump_ast -- <file.js>  (or -e "code")

use parser::{Ast, Node, NodeId, NodeList, Parser, Symbol, Utf8SliceStream};

fn name_of(ast: &Ast, s: Symbol) -> String {
    String::from_utf8_lossy(ast.symbol(s)).into_owned()
}

fn dump(ast: &Ast, id: NodeId, indent: usize) {
    let pad = "  ".repeat(indent);
    let span = ast.span(id);
    let at = format!("@{}..{}", span.start, span.end);
    match *ast.node(id) {
        Node::NumberLiteral(n) => println!("{pad}Number({n}) {at}"),
        Node::StringLiteral(s) => println!("{pad}String({:?}) {at}", name_of(ast, s)),
        Node::BigIntLiteral(s) => println!("{pad}BigInt({}) {at}", name_of(ast, s)),
        Node::BoolLiteral(b) => println!("{pad}Bool({b}) {at}"),
        Node::NullLiteral => println!("{pad}Null {at}"),
        Node::Identifier { sym } => println!("{pad}Ident({}) {at}", name_of(ast, sym)),
        Node::This => println!("{pad}This {at}"),
        Node::Unary { op, expr } => {
            println!("{pad}Unary({op:?}) {at}");
            dump(ast, expr, indent + 1);
        }
        Node::Update { op, prefix, target } => {
            println!("{pad}Update({op:?} prefix={prefix}) {at}");
            dump(ast, target, indent + 1);
        }
        Node::Binary { op, lhs, rhs } => {
            println!("{pad}Binary({op:?}) {at}");
            dump(ast, lhs, indent + 1);
            dump(ast, rhs, indent + 1);
        }
        Node::Assign { op, target, value } => {
            println!("{pad}Assign({op:?}) {at}");
            dump(ast, target, indent + 1);
            dump(ast, value, indent + 1);
        }
        Node::Conditional { cond, then, else_ } => {
            println!("{pad}Conditional {at}");
            dump(ast, cond, indent + 1);
            dump(ast, then, indent + 1);
            dump(ast, else_, indent + 1);
        }
        Node::Call { callee, args } => {
            println!("{pad}Call {at}");
            dump(ast, callee, indent + 1);
            dump_list(ast, args, indent + 1);
        }
        Node::New { callee, args } => {
            println!("{pad}New {at}");
            dump(ast, callee, indent + 1);
            if let Some(args) = args {
                dump_list(ast, args, indent + 1);
            }
        }
        Node::Property {
            object,
            key,
            computed,
        } => {
            println!("{pad}Property(computed={computed}) {at}");
            dump(ast, object, indent + 1);
            dump(ast, key, indent + 1);
        }
        Node::ArrayLiteral { elements } => {
            println!("{pad}Array {at}");
            dump_list(ast, elements, indent + 1);
        }
        Node::Hole => println!("{pad}Hole {at}"),
        Node::Spread { expr } => {
            println!("{pad}Spread {at}");
            dump(ast, expr, indent + 1);
        }
        Node::ObjectLiteral { props } => {
            println!("{pad}Object {at}");
            dump_list(ast, props, indent + 1);
        }
        Node::ObjectProperty {
            key,
            value,
            kind,
            computed,
        } => {
            println!("{pad}Prop({kind:?} computed={computed}) {at}");
            dump(ast, key, indent + 1);
            dump(ast, value, indent + 1);
        }
        Node::FunctionExpr { function } => {
            let f = ast.function(function);
            println!(
                "{pad}FunctionExpr({:?} lit_id={}) {at}",
                f.name.map(|s| name_of(ast, s)),
                f.literal_id
            );
            dump(ast, f.body.unwrap(), indent + 1);
        }
        Node::ExprStmt { expr } => {
            println!("{pad}ExprStmt {at}");
            dump(ast, expr, indent + 1);
        }
        Node::VarDecl { kind, decls } => {
            println!("{pad}VarDecl({kind:?}) {at}");
            dump_list(ast, decls, indent + 1);
        }
        Node::VarDeclarator { name, init } => {
            println!("{pad}Declarator({}) {at}", name_of(ast, name));
            if let Some(init) = init {
                dump(ast, init, indent + 1);
            }
        }
        Node::Block { stmts } => {
            println!("{pad}Block {at}");
            dump_list(ast, stmts, indent + 1);
        }
        Node::If { cond, then, else_ } => {
            println!("{pad}If {at}");
            dump(ast, cond, indent + 1);
            dump(ast, then, indent + 1);
            if let Some(e) = else_ {
                dump(ast, e, indent + 1);
            }
        }
        Node::While { cond, body } => {
            println!("{pad}While {at}");
            dump(ast, cond, indent + 1);
            dump(ast, body, indent + 1);
        }
        Node::For {
            init,
            cond,
            next,
            body,
        } => {
            println!("{pad}For {at}");
            for part in [init, cond, next].into_iter().flatten() {
                dump(ast, part, indent + 1);
            }
            dump(ast, body, indent + 1);
        }
        Node::Return { value } => {
            println!("{pad}Return {at}");
            if let Some(v) = value {
                dump(ast, v, indent + 1);
            }
        }
        Node::Break { label } => println!("{pad}Break({label:?}) {at}"),
        Node::Continue { label } => println!("{pad}Continue({label:?}) {at}"),
        Node::Throw { expr } => {
            println!("{pad}Throw {at}");
            dump(ast, expr, indent + 1);
        }
        Node::TryCatch {
            try_block,
            catch_param,
            catch_block,
            finally_block,
        } => {
            println!("{pad}Try(catch={catch_param:?}) {at}");
            dump(ast, try_block, indent + 1);
            if let Some(c) = catch_block {
                dump(ast, c, indent + 1);
            }
            if let Some(f) = finally_block {
                dump(ast, f, indent + 1);
            }
        }
        Node::FunctionDecl { function } => {
            let f = ast.function(function);
            println!(
                "{pad}FunctionDecl({:?} lit_id={} strict={}) {at}",
                f.name.map(|s| name_of(ast, s)),
                f.literal_id,
                f.strict
            );
            dump(ast, f.body.unwrap(), indent + 1);
        }
        Node::Switch { disc, cases } => {
            println!("{pad}Switch {at}");
            dump(ast, disc, indent + 1);
            for &case in ast.list_items(cases) {
                dump(ast, case, indent + 1);
            }
        }
        Node::Labeled { label, body } => {
            println!("{pad}Labeled({}) {at}", name_of(ast, label));
            dump(ast, body, indent + 1);
        }
        Node::SwitchCase { test, stmts } => {
            println!("{pad}Case(default={}) {at}", test.is_none());
            if let Some(t) = test {
                dump(ast, t, indent + 1);
            }
            for &s in ast.list_items(stmts) {
                dump(ast, s, indent + 1);
            }
        }
        Node::ClassDecl { class } => dump_class(ast, class, "ClassDecl", pad, at, indent),
        Node::ClassExpr { class } => dump_class(ast, class, "ClassExpr", pad, at, indent),
        Node::SuperProperty {
            key,
            computed,
            is_static,
        } => {
            println!("{pad}SuperProperty(computed={computed} static={is_static}) {at}");
            dump(ast, key, indent + 1);
        }
        Node::SuperCall { args } => {
            println!("{pad}SuperCall {at}");
            dump_list(ast, args, indent + 1);
        }
        Node::NewTarget => println!("{pad}NewTarget {at}"),
        Node::Empty => println!("{pad}Empty {at}"),
    }
}

fn dump_class(ast: &Ast, id: parser::ClassId, tag: &str, pad: String, at: String, indent: usize) {
    let c = ast.class(id);
    println!(
        "{pad}{tag}({:?} members={}) {at}",
        c.name.map(|s| name_of(ast, s)),
        c.members.len()
    );
    if let Some(sup) = c.superclass {
        dump(ast, sup, indent + 1);
    }
    for m in &c.members {
        println!(
            "{pad}  Member({:?} static={} ctor={} computed={})",
            m.kind, m.is_static, m.is_constructor, m.computed
        );
        dump(ast, m.key, indent + 2);
        dump(ast, m.value, indent + 2);
    }
}

fn dump_list(ast: &Ast, list: NodeList, indent: usize) {
    for item in ast.list_items(list) {
        dump(ast, *item, indent);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let src = if args.first().map(String::as_str) == Some("-e") {
        args[1].clone()
    } else {
        std::fs::read_to_string(&args[0]).expect("read file")
    };
    let mut p = Parser::new(Utf8SliceStream::new(&src));
    let top = p.parse_script().expect("parse error");
    let ast = p.into_ast();
    let f = ast.function(top);
    println!("Script(lit_id={} strict={})", f.literal_id, f.strict);
    dump(&ast, f.body.unwrap(), 1);
}
