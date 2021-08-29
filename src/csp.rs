use std::ops::{Add, BitAnd, BitOr, BitXor, Mul, Not, Sub};

use super::CmpOp;

#[derive(Clone, Copy)]
pub struct Domain {
    low: i32,
    high: i32,
}

impl Domain {
    pub fn range(low: i32, high: i32) -> Domain {
        Domain { low, high }
    }

    pub fn enumerate(&self) -> Vec<i32> {
        (self.low..=self.high).into_iter().collect::<Vec<_>>()
    }
}

impl Add<Domain> for Domain {
    type Output = Domain;

    fn add(self, rhs: Domain) -> Domain {
        Domain::range(
            self.low.checked_add(rhs.low).unwrap(),
            self.high.checked_add(rhs.high).unwrap(),
        )
    }
}

impl Mul<i32> for Domain {
    type Output = Domain;

    fn mul(self, rhs: i32) -> Domain {
        if rhs == 0 {
            Domain::range(0, 0)
        } else if rhs > 0 {
            Domain::range(
                self.low.checked_mul(rhs).unwrap(),
                self.high.checked_mul(rhs).unwrap(),
            )
        } else {
            Domain::range(
                self.high.checked_mul(rhs).unwrap(),
                self.low.checked_mul(rhs).unwrap(),
            )
        }
    }
}

impl BitOr<Domain> for Domain {
    type Output = Domain;

    fn bitor(self, rhs: Domain) -> Domain {
        Domain::range(self.low.min(rhs.low), self.high.max(rhs.high))
    }
}

pub(super) struct BoolVarData {
    possibility_mask: u8,
}

impl BoolVarData {
    fn new() -> BoolVarData {
        BoolVarData {
            possibility_mask: 3,
        }
    }

    fn is_feasible(&self, b: bool) -> bool {
        (self.possibility_mask & (if b { 2 } else { 1 })) != 0
    }

    fn is_unsatisfiable(&self) -> bool {
        self.possibility_mask == 0
    }

    fn set_infeasible(&mut self, b: bool) -> bool {
        let res = self.is_feasible(b);
        self.possibility_mask &= if b { 1 } else { 2 };
        res
    }
}

pub(super) struct IntVarData {
    pub(super) domain: Domain,
}

impl IntVarData {
    fn new(domain: Domain) -> IntVarData {
        IntVarData {
            domain: domain.clone(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct BoolVar(pub(super) usize);

impl BoolVar {
    pub fn expr(self) -> BoolExpr {
        BoolExpr::Var(self)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct IntVar(pub(super) usize);

impl IntVar {
    pub fn expr(self) -> IntExpr {
        IntExpr::Var(self)
    }
}

#[derive(Clone)]
pub enum Stmt {
    Expr(BoolExpr),
    AllDifferent(Vec<IntExpr>),
    // TODO: graph constraints go here
}

#[derive(Clone)]
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

    pub(super) fn decompose_neg(self) -> Box<BoolExpr> {
        match self {
            BoolExpr::Not(x) => x,
            _ => panic!(),
        }
    }

    pub(super) fn decompose_or(self) -> Vec<Box<BoolExpr>> {
        match self {
            BoolExpr::Or(x) => x,
            _ => panic!(),
        }
    }

    pub(super) fn decompose_binary_or(self) -> (Box<BoolExpr>, Box<BoolExpr>) {
        match self {
            BoolExpr::Or(mut x) => {
                assert_eq!(x.len(), 2);
                let b = x.remove(1);
                let a = x.remove(0);
                (a, b)
            }
            _ => panic!(),
        }
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

#[derive(Clone)]
pub enum IntExpr {
    Const(i32),
    Var(IntVar),
    NVar(super::norm_csp::IntVar),
    Linear(Vec<(Box<IntExpr>, i32)>),
    If(Box<BoolExpr>, Box<IntExpr>, Box<IntExpr>),
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

pub(super) struct CSPVars {
    pub(super) bool_var: Vec<BoolVarData>,
    pub(super) int_var: Vec<IntVarData>,
}

pub struct CSP {
    pub(super) vars: CSPVars,
    pub(super) constraints: Vec<Stmt>,
}

impl CSP {
    pub fn new() -> CSP {
        CSP {
            vars: CSPVars {
                bool_var: vec![],
                int_var: vec![],
            },
            constraints: vec![],
        }
    }

    pub fn new_bool_var(&mut self) -> BoolVar {
        let id = self.vars.bool_var.len();
        self.vars.bool_var.push(BoolVarData::new());
        BoolVar(id)
    }

    pub fn new_int_var(&mut self, domain: Domain) -> IntVar {
        let id = self.vars.int_var.len();
        self.vars.int_var.push(IntVarData::new(domain));
        IntVar(id)
    }

    pub fn add_constraint(&mut self, stmt: Stmt) {
        self.constraints.push(stmt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hoge() {
        let mut csp = CSP::new();
        let v = csp.new_int_var(Domain { low: 5, high: 10 });
        let e = IntExpr::Var(v) + IntExpr::Var(v);
    }
}
