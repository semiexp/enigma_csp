// Normalized CSP

use std::collections::BTreeMap;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use super::csp::Domain;
use super::CmpOp;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct BoolVar(pub(super) usize);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct IntVar(pub(super) usize);

pub struct BoolLit {
    pub(super) var: BoolVar,
    pub(super) negated: bool,
}

impl BoolLit {
    pub fn new(var: BoolVar, negated: bool) -> BoolLit {
        BoolLit { var, negated }
    }
}

#[derive(Clone, Debug)]
pub struct LinearSum {
    pub(super) term: BTreeMap<IntVar, i32>,
    pub(super) constant: i32,
}

impl LinearSum {
    pub fn new() -> LinearSum {
        LinearSum {
            term: BTreeMap::new(),
            constant: 0,
        }
    }

    pub fn constant(v: i32) -> LinearSum {
        LinearSum {
            term: BTreeMap::new(),
            constant: v,
        }
    }

    pub fn singleton(var: IntVar) -> LinearSum {
        let mut ret = LinearSum::new();
        ret.add_coef(var, 1);
        ret
    }

    pub fn add_constant(&mut self, v: i32) {
        self.constant = self.constant.checked_add(v).unwrap();
    }

    pub fn add_coef(&mut self, var: IntVar, coef: i32) {
        if coef == 0 {
            return;
        }
        let new_coef = match self.term.get(&var) {
            Some(&e) => e.checked_add(coef).unwrap(),
            _ => coef,
        };
        if new_coef == 0 {
            self.term.remove(&var);
        } else {
            self.term.insert(var, new_coef);
        }
    }

    pub fn terms(&self) -> Vec<(IntVar, i32)> {
        self.term.iter().map(|(v, c)| (*v, *c)).collect()
    }
}

impl AddAssign<LinearSum> for LinearSum {
    fn add_assign(&mut self, rhs: LinearSum) {
        for (&key, &value) in rhs.term.iter() {
            self.add_coef(key, value);
        }
        self.add_constant(rhs.constant);
    }
}

impl Add<LinearSum> for LinearSum {
    type Output = LinearSum;

    fn add(self, rhs: LinearSum) -> LinearSum {
        let mut ret = self;
        ret += rhs;
        ret
    }
}

impl SubAssign<LinearSum> for LinearSum {
    fn sub_assign(&mut self, rhs: LinearSum) {
        for (&key, &value) in rhs.term.iter() {
            self.add_coef(key, -value);
        }
        self.add_constant(-rhs.constant);
    }
}

impl Sub<LinearSum> for LinearSum {
    type Output = LinearSum;

    fn sub(self, rhs: LinearSum) -> LinearSum {
        let mut ret = self;
        ret -= rhs;
        ret
    }
}

impl MulAssign<i32> for LinearSum {
    fn mul_assign(&mut self, rhs: i32) {
        if rhs == 0 {
            *self = LinearSum::new();
        }
        self.constant = self.constant.checked_mul(rhs).unwrap();
        for (_, value) in self.term.iter_mut() {
            *value = value.checked_mul(rhs).unwrap();
        }
    }
}

impl Mul<i32> for LinearSum {
    type Output = LinearSum;

    fn mul(self, rhs: i32) -> LinearSum {
        let mut ret = self;
        ret *= rhs;
        ret
    }
}

/// Literal stating (`sum` `op` 0) where `op` is one of comparison operators (like `>=`).
pub struct LinearLit {
    pub(super) sum: LinearSum,
    pub(super) op: CmpOp,
}

impl LinearLit {
    pub fn new(sum: LinearSum, op: CmpOp) -> LinearLit {
        LinearLit { sum, op }
    }
}

pub struct Constraint {
    pub(super) bool_lit: Vec<BoolLit>,
    pub(super) linear_lit: Vec<LinearLit>,
}

impl Constraint {
    pub fn new() -> Constraint {
        Constraint {
            bool_lit: vec![],
            linear_lit: vec![],
        }
    }

    pub fn add_bool(&mut self, lit: BoolLit) {
        self.bool_lit.push(lit);
    }

    pub fn add_linear(&mut self, lit: LinearLit) {
        self.linear_lit.push(lit);
    }
}

pub(super) struct NormCSPVars {
    // TODO: remove `pub(super)`
    pub(super) num_bool_var: usize,
    pub(super) int_var: Vec<super::csp::Domain>,
}

pub struct NormCSP {
    pub(super) vars: NormCSPVars,
    pub(super) constraints: Vec<Constraint>,
    pub(super) num_encoded_vars: usize,
}

impl NormCSP {
    pub fn new() -> NormCSP {
        NormCSP {
            vars: NormCSPVars {
                num_bool_var: 0,
                int_var: vec![],
            },
            constraints: vec![],
            num_encoded_vars: 0,
        }
    }

    pub fn new_bool_var(&mut self) -> BoolVar {
        let id = self.vars.num_bool_var;
        self.vars.num_bool_var += 1;
        BoolVar(id)
    }

    pub fn new_int_var(&mut self, domain: super::csp::Domain) -> IntVar {
        let id = self.vars.int_var.len();
        self.vars.int_var.push(domain);
        IntVar(id)
    }

    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    pub(super) fn get_domain_linear_sum(&self, linear_sum: &LinearSum) -> Domain {
        let mut ret = Domain::range(linear_sum.constant, linear_sum.constant);

        for (var, coef) in &linear_sum.term {
            ret = ret + self.vars.int_var[var.0] * *coef;
        }

        ret
    }
}
