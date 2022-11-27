use crate::util::ConvertMapIndex;
use std::io::Write;
use std::ops::{Add, BitAnd, BitOr, BitXor, Mul, Not, Sub};

use crate::arithmetic::CmpOp;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct BoolVar(usize);

impl BoolVar {
    pub fn new(id: usize) -> BoolVar {
        BoolVar(id)
    }

    pub fn expr(self) -> BoolExpr {
        BoolExpr::Var(self)
    }
}

impl ConvertMapIndex for BoolVar {
    fn to_index(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct IntVar(usize);

impl IntVar {
    pub fn new(id: usize) -> IntVar {
        IntVar(id)
    }

    pub fn expr(self) -> IntExpr {
        IntExpr::Var(self)
    }
}

impl ConvertMapIndex for IntVar {
    fn to_index(&self) -> usize {
        self.0
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Stmt {
    Expr(BoolExpr),
    AllDifferent(Vec<IntExpr>),
    ActiveVerticesConnected(Vec<BoolExpr>, Vec<(usize, usize)>),
    Circuit(Vec<IntVar>),
    ExtensionSupports(Vec<IntVar>, Vec<Vec<Option<i32>>>),
}

impl Stmt {
    pub fn pretty_print<W: Write>(&self, out: &mut W) -> std::io::Result<()> {
        match self {
            Stmt::Expr(e) => e.pretty_print(out)?,
            Stmt::AllDifferent(exprs) => {
                write!(out, "(alldifferent")?;
                for expr in exprs {
                    write!(out, " ")?;
                    expr.pretty_print(out)?;
                }
                write!(out, ")")?;
            }
            Stmt::ActiveVerticesConnected(exprs, edges) => {
                write!(out, "(active-vertices-connected")?;
                for (i, expr) in exprs.iter().enumerate() {
                    write!(out, " {}:", i)?;
                    expr.pretty_print(out)?;
                }
                write!(out, " graph=[")?;
                let mut is_first = true;
                for &(u, v) in edges {
                    if !is_first {
                        write!(out, " ")?;
                    } else {
                        is_first = false;
                    }
                    write!(out, "{}--{}", u, v)?;
                }
                write!(out, "])")?;
            }
            Stmt::Circuit(vars) => {
                write!(out, "(circuit")?;
                for v in vars {
                    write!(out, " <i{}>", v.0)?;
                }
                write!(out, ")")?;
            }
            Stmt::ExtensionSupports(_, _) => todo!(),
        }
        Ok(())
    }
}
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum BoolExpr {
    Const(bool),
    Var(BoolVar),
    NVar(super::norm_csp::BoolVar),
    And(Vec<Box<BoolExpr>>),
    Or(Vec<Box<BoolExpr>>),
    Not(Box<BoolExpr>),
    Xor(Box<BoolExpr>, Box<BoolExpr>),
    Iff(Box<BoolExpr>, Box<BoolExpr>),
    Imp(Box<BoolExpr>, Box<BoolExpr>),
    Cmp(CmpOp, Box<IntExpr>, Box<IntExpr>),
}

impl BoolExpr {
    pub fn imp(self, rhs: BoolExpr) -> BoolExpr {
        BoolExpr::Imp(Box::new(self), Box::new(rhs))
    }

    pub fn iff(self, rhs: BoolExpr) -> BoolExpr {
        BoolExpr::Iff(Box::new(self), Box::new(rhs))
    }

    pub fn ite(self, t: IntExpr, f: IntExpr) -> IntExpr {
        IntExpr::If(Box::new(self), Box::new(t), Box::new(f))
    }

    pub fn is_const(&self) -> Option<bool> {
        match self {
            &BoolExpr::Const(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_var(&self) -> Option<BoolVar> {
        match self {
            &BoolExpr::Var(v) => Some(v),
            _ => None,
        }
    }

    pub fn pretty_print<W: Write>(&self, out: &mut W) -> std::io::Result<()> {
        match self {
            &BoolExpr::Const(b) => write!(out, "{}", b)?,
            &BoolExpr::Var(v) => write!(out, "<b{}>", v.0)?,
            &BoolExpr::NVar(v) => write!(out, "<nb{}>", v.id())?,
            BoolExpr::And(exprs) => {
                write!(out, "(&&")?;
                for expr in exprs {
                    write!(out, " ")?;
                    expr.pretty_print(out)?;
                }
                write!(out, ")")?;
            }
            BoolExpr::Or(exprs) => {
                write!(out, "(||")?;
                for expr in exprs {
                    write!(out, " ")?;
                    expr.pretty_print(out)?;
                }
                write!(out, ")")?;
            }
            BoolExpr::Not(expr) => {
                write!(out, "(! ")?;
                expr.pretty_print(out)?;
                write!(out, ")")?;
            }
            BoolExpr::Xor(e1, e2) => {
                write!(out, "(xor ")?;
                e1.pretty_print(out)?;
                write!(out, " ")?;
                e2.pretty_print(out)?;
                write!(out, ")")?;
            }
            BoolExpr::Iff(e1, e2) => {
                write!(out, "(iff ")?;
                e1.pretty_print(out)?;
                write!(out, " ")?;
                e2.pretty_print(out)?;
                write!(out, ")")?;
            }
            BoolExpr::Imp(e1, e2) => {
                write!(out, "(=> ")?;
                e1.pretty_print(out)?;
                write!(out, " ")?;
                e2.pretty_print(out)?;
                write!(out, ")")?;
            }
            BoolExpr::Cmp(op, e1, e2) => {
                write!(out, "({} ", op)?;
                e1.pretty_print(out)?;
                write!(out, " ")?;
                e2.pretty_print(out)?;
                write!(out, ")")?;
            }
        }
        Ok(())
    }
}

impl BitAnd<BoolExpr> for BoolExpr {
    type Output = BoolExpr;

    fn bitand(self, rhs: BoolExpr) -> Self::Output {
        BoolExpr::And(vec![Box::new(self), Box::new(rhs)])
    }
}

impl BitOr<BoolExpr> for BoolExpr {
    type Output = BoolExpr;

    fn bitor(self, rhs: BoolExpr) -> Self::Output {
        BoolExpr::Or(vec![Box::new(self), Box::new(rhs)])
    }
}

impl BitXor<BoolExpr> for BoolExpr {
    type Output = BoolExpr;

    fn bitxor(self, rhs: BoolExpr) -> Self::Output {
        BoolExpr::Xor(Box::new(self), Box::new(rhs))
    }
}

impl Not for BoolExpr {
    type Output = BoolExpr;

    fn not(self) -> Self::Output {
        BoolExpr::Not(Box::new(self))
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum IntExpr {
    Const(i32),
    Var(IntVar),
    NVar(super::norm_csp::IntVar),
    Linear(Vec<(Box<IntExpr>, i32)>),
    If(Box<BoolExpr>, Box<IntExpr>, Box<IntExpr>),
    Abs(Box<IntExpr>),
    Mul(Box<IntExpr>, Box<IntExpr>),
}

impl IntExpr {
    pub fn eq(self, rhs: IntExpr) -> BoolExpr {
        BoolExpr::Cmp(CmpOp::Eq, Box::new(self), Box::new(rhs))
    }

    pub fn ne(self, rhs: IntExpr) -> BoolExpr {
        BoolExpr::Cmp(CmpOp::Ne, Box::new(self), Box::new(rhs))
    }

    pub fn le(self, rhs: IntExpr) -> BoolExpr {
        BoolExpr::Cmp(CmpOp::Le, Box::new(self), Box::new(rhs))
    }

    pub fn lt(self, rhs: IntExpr) -> BoolExpr {
        BoolExpr::Cmp(CmpOp::Lt, Box::new(self), Box::new(rhs))
    }

    pub fn ge(self, rhs: IntExpr) -> BoolExpr {
        BoolExpr::Cmp(CmpOp::Ge, Box::new(self), Box::new(rhs))
    }

    pub fn gt(self, rhs: IntExpr) -> BoolExpr {
        BoolExpr::Cmp(CmpOp::Gt, Box::new(self), Box::new(rhs))
    }

    pub fn abs(self) -> IntExpr {
        IntExpr::Abs(Box::new(self))
    }

    pub fn pretty_print<W: Write>(&self, out: &mut W) -> std::io::Result<()> {
        match self {
            &IntExpr::Const(c) => write!(out, "{}", c)?,
            &IntExpr::Var(v) => write!(out, "<i{}>", v.0)?,
            &IntExpr::NVar(v) => write!(out, "<ni{}>", v.id())?,
            IntExpr::Linear(terms) => {
                write!(out, "(")?;
                let mut is_first = true;
                for (expr, coef) in terms {
                    if !is_first {
                        write!(out, "+")?;
                    } else {
                        is_first = false;
                    }
                    expr.pretty_print(out)?;
                    write!(out, "*{}", coef)?;
                }
                write!(out, ")")?;
            }
            IntExpr::If(cond, t, f) => {
                write!(out, "(if ")?;
                cond.pretty_print(out)?;
                write!(out, " ")?;
                t.pretty_print(out)?;
                write!(out, " ")?;
                f.pretty_print(out)?;
                write!(out, ")")?;
            }
            IntExpr::Abs(x) => {
                write!(out, "(abs ")?;
                x.pretty_print(out)?;
                write!(out, ")")?;
            }
            IntExpr::Mul(x, y) => {
                write!(out, "(mul ")?;
                x.pretty_print(out)?;
                write!(out, " ")?;
                y.pretty_print(out)?;
                write!(out, ")")?;
            }
        }
        Ok(())
    }
}

impl Add<IntExpr> for IntExpr {
    type Output = IntExpr;

    fn add(self, rhs: IntExpr) -> IntExpr {
        IntExpr::Linear(vec![(Box::new(self), 1), (Box::new(rhs), 1)])
    }
}

impl Sub<IntExpr> for IntExpr {
    type Output = IntExpr;

    fn sub(self, rhs: IntExpr) -> IntExpr {
        IntExpr::Linear(vec![(Box::new(self), 1), (Box::new(rhs), -1)])
    }
}

impl Mul<i32> for IntExpr {
    type Output = IntExpr;

    fn mul(self, rhs: i32) -> IntExpr {
        IntExpr::Linear(vec![(Box::new(self), rhs)])
    }
}

impl Mul<IntExpr> for IntExpr {
    type Output = IntExpr;

    fn mul(self, rhs: IntExpr) -> IntExpr {
        IntExpr::Mul(Box::new(self), Box::new(rhs))
    }
}
