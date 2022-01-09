extern crate nom;
use std::collections::{btree_map, BTreeMap};

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric0, digit1},
    combinator::{eof, map, recognize},
    multi::separated_list0,
    sequence::{delimited, pair, preceded, terminated},
    Finish, IResult,
};

use super::csp::{BoolExpr, BoolVar, Domain, IntExpr, IntVar, Stmt};

#[derive(PartialEq, Eq, Debug)]
enum SyntaxTree<'a> {
    Ident(&'a str),
    Int(i32),
    Node(Vec<SyntaxTree<'a>>),
}

impl<'a> SyntaxTree<'a> {
    fn as_op_name(&self) -> &'a str {
        match self {
            &SyntaxTree::Ident(s) => s,
            _ => panic!("op name expected"),
        }
    }

    fn as_ident(&self) -> &'a str {
        match self {
            &SyntaxTree::Ident(s) => s,
            _ => panic!("identifier expected"),
        }
    }

    fn as_int(&self) -> i32 {
        match self {
            &SyntaxTree::Int(n) => n,
            _ => panic!("int expected"),
        }
    }

    fn as_usize(&self) -> usize {
        let n = self.as_int();
        assert!(n >= 0);
        n as usize
    }
}

fn parse_to_tree(input: &str) -> Result<SyntaxTree, nom::error::Error<&str>> {
    fn rec_parser(input: &str) -> IResult<&str, SyntaxTree> {
        let ident_or_op = recognize(pair(alpha1, alphanumeric0));
        let op = alt((
            tag("&&"),
            tag("||"),
            tag("^"),
            tag("=>"),
            tag("=="),
            tag("="),
            tag("!="),
            tag("!"),
            tag("<="),
            tag("<"),
            tag(">="),
            tag(">"),
            tag("+"),
            tag("-"),
        ));
        alt((
            delimited(
                tag("("),
                map(separated_list0(tag(" "), rec_parser), SyntaxTree::Node),
                tag(")"),
            ),
            map(tag("graph-active-vertices-connected"), SyntaxTree::Ident),
            map(ident_or_op, SyntaxTree::Ident),
            map(digit1, |s: &str| SyntaxTree::Int(s.parse::<i32>().unwrap())), // TODO
            map(preceded(tag("-"), digit1), |s: &str| {
                SyntaxTree::Int(-s.parse::<i32>().unwrap())
            }), // TODO
            map(op, SyntaxTree::Ident),
        ))(input)
    }

    terminated(rec_parser, eof)(input).finish().map(|(i, o)| {
        assert!(i.is_empty());
        o
    })
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum Var {
    Bool(BoolVar),
    Int(IntVar),
}

#[derive(PartialEq, Eq, Debug)]
pub enum ParseResult<'a> {
    BoolVarDecl(&'a str),
    IntVarDecl(&'a str, Domain),
    Stmt(Stmt),
}

pub struct VarMap(BTreeMap<String, Var>);

impl VarMap {
    pub fn new() -> VarMap {
        VarMap(BTreeMap::new())
    }

    pub fn add_bool_var(&mut self, name: &str, var: BoolVar) {
        assert!(self.0.insert(String::from(name), Var::Bool(var)).is_none());
    }

    pub fn add_int_var(&mut self, name: &str, var: IntVar) {
        assert!(self.0.insert(String::from(name), Var::Int(var)).is_none());
    }

    pub fn get_var(&self, name: &str) -> Option<Var> {
        self.0.get(name).copied()
    }

    pub fn iter<'a>(&'a self) -> btree_map::Iter<'a, String, Var> {
        self.0.iter()
    }
}

pub fn parse<'a, 'b>(var_map: &'a VarMap, input: &'b str) -> ParseResult<'b> {
    // TODO: return error info
    let tree = parse_to_tree(input).unwrap();
    let child = match &tree {
        SyntaxTree::Node(child) => child,
        _ => return ParseResult::Stmt(Stmt::Expr(parse_bool_expr(var_map, &tree))),
    };
    assert!(child.len() >= 1);
    let op_name = child[0].as_op_name();

    if op_name == "bool" {
        assert_eq!(child.len(), 2);
        let var_name = child[1].as_ident();
        ParseResult::BoolVarDecl(var_name)
    } else if op_name == "int" {
        assert_eq!(child.len(), 4);
        let var_name = child[1].as_ident();
        let low = child[2].as_int();
        let high = child[3].as_int();
        ParseResult::IntVarDecl(var_name, Domain::range(low, high))
    } else if op_name == "alldifferent" {
        let exprs = child[1..]
            .iter()
            .map(|c| parse_int_expr(var_map, c))
            .collect::<Vec<_>>();
        ParseResult::Stmt(Stmt::AllDifferent(exprs))
    } else if op_name == "graph-active-vertices-connected" {
        let num_vertices = child[1].as_usize();
        let num_edges = child[2].as_usize();
        let num_vertices = num_vertices as usize;
        let num_edges = num_edges as usize;
        assert_eq!(child.len(), 3 + num_vertices + num_edges * 2);

        let vertices = (0..num_vertices)
            .map(|i| parse_bool_expr(var_map, &child[i + 3]))
            .collect::<Vec<_>>();
        let edges = (0..num_edges)
            .map(|i| {
                (
                    child[i * 2 + 3 + num_vertices].as_usize(),
                    child[i * 2 + 4 + num_vertices].as_usize(),
                )
            })
            .collect::<Vec<_>>();
        ParseResult::Stmt(Stmt::ActiveVerticesConnected(vertices, edges))
    } else {
        ParseResult::Stmt(Stmt::Expr(parse_bool_expr(var_map, &tree)))
    }
}

fn parse_bool_expr(var_map: &VarMap, tree: &SyntaxTree) -> BoolExpr {
    match tree {
        &SyntaxTree::Ident(id) => {
            if id == "true" {
                return BoolExpr::Const(true);
            } else if id == "false" {
                return BoolExpr::Const(false);
            }

            let var = var_map.get_var(id).unwrap();
            match var {
                Var::Bool(b) => b.expr(),
                Var::Int(_) => panic!("int var is given while bool expr is expected"),
            }
        }
        &SyntaxTree::Int(_) => panic!("int constant is given while bool expr is expected"),
        SyntaxTree::Node(child) => {
            let op_name = child[0].as_op_name();
            if op_name == "not" || op_name == "!" {
                assert_eq!(child.len(), 2);
                !parse_bool_expr(var_map, &child[1])
            } else if op_name == "and" || op_name == "&&" {
                BoolExpr::And(
                    child[1..]
                        .iter()
                        .map(|t| Box::new(parse_bool_expr(var_map, t)))
                        .collect(),
                )
            } else if op_name == "or" || op_name == "||" {
                BoolExpr::Or(
                    child[1..]
                        .iter()
                        .map(|t| Box::new(parse_bool_expr(var_map, t)))
                        .collect(),
                )
            } else if op_name == "xor" || op_name == "^" {
                assert_eq!(child.len(), 3);
                parse_bool_expr(var_map, &child[1]) ^ parse_bool_expr(var_map, &child[2])
            } else if op_name == "iff" {
                assert_eq!(child.len(), 3);
                parse_bool_expr(var_map, &child[1]).iff(parse_bool_expr(var_map, &child[2]))
            } else if op_name == "imp" || op_name == "=>" {
                assert_eq!(child.len(), 3);
                parse_bool_expr(var_map, &child[1]).imp(parse_bool_expr(var_map, &child[2]))
            } else if op_name == "=" || op_name == "==" || op_name == "eq" {
                assert_eq!(child.len(), 3);
                parse_int_expr(var_map, &child[1]).eq(parse_int_expr(var_map, &child[2]))
            } else if op_name == "!=" || op_name == "ne" {
                assert_eq!(child.len(), 3);
                parse_int_expr(var_map, &child[1]).ne(parse_int_expr(var_map, &child[2]))
            } else if op_name == "<=" || op_name == "le" {
                assert_eq!(child.len(), 3);
                parse_int_expr(var_map, &child[1]).le(parse_int_expr(var_map, &child[2]))
            } else if op_name == "<" || op_name == "lt" {
                assert_eq!(child.len(), 3);
                parse_int_expr(var_map, &child[1]).lt(parse_int_expr(var_map, &child[2]))
            } else if op_name == ">=" || op_name == "ge" {
                assert_eq!(child.len(), 3);
                parse_int_expr(var_map, &child[1]).ge(parse_int_expr(var_map, &child[2]))
            } else if op_name == ">" || op_name == "gt" {
                assert_eq!(child.len(), 3);
                parse_int_expr(var_map, &child[1]).gt(parse_int_expr(var_map, &child[2]))
            } else {
                panic!("unknown operator: {}", op_name);
            }
        }
    }
}

fn parse_int_expr(var_map: &VarMap, tree: &SyntaxTree) -> IntExpr {
    match tree {
        &SyntaxTree::Ident(id) => {
            if id == "true" || id == "false" {
                panic!("bool constant is given while int expr is expected");
            }

            let var = var_map.get_var(id).unwrap();
            match var {
                Var::Bool(_) => panic!("int var is given while bool expr is expected"),
                Var::Int(i) => i.expr(),
            }
        }
        &SyntaxTree::Int(n) => IntExpr::Const(n),
        SyntaxTree::Node(child) => {
            let op_name = child[0].as_op_name();
            if op_name == "+" || op_name == "add" {
                IntExpr::Linear(
                    child[1..]
                        .iter()
                        .map(|t| (Box::new(parse_int_expr(var_map, t)), 1))
                        .collect(),
                )
            } else if op_name == "-" || op_name == "sub" {
                assert!(child.len() == 2 || child.len() == 3);
                if child.len() == 2 {
                    IntExpr::Linear(vec![(Box::new(parse_int_expr(var_map, &child[1])), -1)])
                } else {
                    parse_int_expr(var_map, &child[1]) - parse_int_expr(var_map, &child[2])
                }
            } else if op_name == "if" {
                assert_eq!(child.len(), 4);
                parse_bool_expr(var_map, &child[1]).ite(
                    parse_int_expr(var_map, &child[2]),
                    parse_int_expr(var_map, &child[3]),
                )
            } else {
                panic!("unknown operator: {}", op_name);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::integration::IntegratedSolver;
    use super::*;

    #[test]
    fn test_parser_syntax_tree() {
        assert_eq!(parse_to_tree("xyz"), Result::Ok(SyntaxTree::Ident("xyz")));
        assert_eq!(parse_to_tree("12345"), Result::Ok(SyntaxTree::Int(12345)));
        assert_eq!(parse_to_tree("-12345"), Result::Ok(SyntaxTree::Int(-12345)));
        assert!(parse_to_tree("1a").is_err());
        assert_eq!(
            parse_to_tree("(x 1 2)"),
            Result::Ok(SyntaxTree::Node(vec![
                SyntaxTree::Ident("x"),
                SyntaxTree::Int(1),
                SyntaxTree::Int(2)
            ]))
        );
        assert!(parse_to_tree("(x 1 2 ()").is_err());
        assert_eq!(
            parse_to_tree("(x 1 2 (y -42))"),
            Result::Ok(SyntaxTree::Node(vec![
                SyntaxTree::Ident("x"),
                SyntaxTree::Int(1),
                SyntaxTree::Int(2),
                SyntaxTree::Node(vec![SyntaxTree::Ident("y"), SyntaxTree::Int(-42)])
            ]))
        );
    }

    #[test]
    fn test_parser_parser() {
        let mut var_map = VarMap::new();
        let mut solver = IntegratedSolver::new();

        let result = parse(&var_map, "(bool foo)");
        assert_eq!(result, ParseResult::BoolVarDecl("foo"));
        let foo = solver.new_bool_var();
        var_map.add_bool_var("foo", foo);

        let result = parse(&var_map, "(bool bar)");
        assert_eq!(result, ParseResult::BoolVarDecl("bar"));
        let bar = solver.new_bool_var();
        var_map.add_bool_var("bar", bar);

        let result = parse(&var_map, "(|| (xor foo bar) bar)");
        assert_eq!(
            result,
            ParseResult::Stmt(Stmt::Expr((foo.expr() ^ bar.expr()) | bar.expr()))
        );

        let result = parse(&var_map, "foo");
        assert_eq!(result, ParseResult::Stmt(Stmt::Expr(foo.expr())));
    }
}
