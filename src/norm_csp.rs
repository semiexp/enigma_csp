// Normalized CSP

use std::collections::{btree_map, BTreeMap};
use std::ops::{Add, AddAssign, BitAnd, BitOr, Mul, MulAssign, Sub, SubAssign};

use super::csp::Domain;
use super::CmpOp;
use crate::util::{div_ceil, div_floor, ConvertMapIndex, UpdateStatus};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct BoolVar(usize);

impl ConvertMapIndex for BoolVar {
    fn to_index(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct IntVar(usize);

impl ConvertMapIndex for IntVar {
    fn to_index(&self) -> usize {
        self.0
    }
}

#[derive(Debug)]
pub struct BoolLit {
    pub(super) var: BoolVar,
    pub(super) negated: bool,
}

impl BoolLit {
    pub fn new(var: BoolVar, negated: bool) -> BoolLit {
        BoolLit { var, negated }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Range {
    low: i32,
    high: i32,
}

impl Range {
    pub fn new(low: i32, high: i32) -> Range {
        Range { low, high }
    }

    pub fn empty() -> Range {
        Range {
            low: i32::max_value(),
            high: i32::min_value(),
        }
    }

    pub fn any() -> Range {
        Range {
            low: i32::min_value(),
            high: i32::max_value(),
        }
    }

    pub fn at_least(c: i32) -> Range {
        Range {
            low: c,
            high: i32::max_value(),
        }
    }

    pub fn at_most(c: i32) -> Range {
        Range {
            low: i32::min_value(),
            high: c,
        }
    }

    pub fn constant(c: i32) -> Range {
        Range { low: c, high: c }
    }

    pub fn is_empty(&self) -> bool {
        self.low > self.high
    }
}

impl Add<Range> for Range {
    type Output = Range;

    fn add(self, rhs: Range) -> Self::Output {
        if self.is_empty() || rhs.is_empty() {
            Range::empty()
        } else {
            Range::new(
                self.low.checked_add(rhs.low).unwrap(),
                self.high.checked_add(rhs.high).unwrap(),
            )
        }
    }
}

impl Mul<i32> for Range {
    type Output = Range;

    fn mul(self, rhs: i32) -> Self::Output {
        if self.is_empty() {
            Range::empty()
        } else if rhs >= 0 {
            Range::new(
                self.low.checked_mul(rhs).unwrap(),
                self.high.checked_mul(rhs).unwrap(),
            )
        } else {
            Range::new(
                self.high.checked_mul(rhs).unwrap(),
                self.low.checked_mul(rhs).unwrap(),
            )
        }
    }
}

impl BitAnd<Range> for Range {
    type Output = Range;

    fn bitand(self, rhs: Range) -> Self::Output {
        Range::new(self.low.max(rhs.low), self.high.min(rhs.high))
    }
}

impl BitOr<Range> for Range {
    type Output = Range;

    fn bitor(self, rhs: Range) -> Self::Output {
        Range::new(self.low.min(rhs.low), self.high.max(rhs.high))
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

    pub fn iter(&self) -> btree_map::Iter<IntVar, i32> {
        self.term.iter()
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
#[derive(Clone, Debug)]
pub struct LinearLit {
    pub(super) sum: LinearSum,
    pub(super) op: CmpOp,
}

impl LinearLit {
    pub fn new(sum: LinearSum, op: CmpOp) -> LinearLit {
        LinearLit { sum, op }
    }
}

#[derive(Debug)]
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
    num_bool_var: usize,
    pub(super) int_var: Vec<super::csp::Domain>,
}

impl NormCSPVars {
    pub(super) fn bool_vars_iter(&self) -> impl Iterator<Item = BoolVar> {
        (0..self.num_bool_var).map(|i| BoolVar(i))
    }

    pub(super) fn int_vars_iter(&self) -> impl Iterator<Item = IntVar> {
        (0..self.int_var.len()).map(|i| IntVar(i))
    }

    pub(super) fn int_var(&self, var: IntVar) -> &super::csp::Domain {
        &self.int_var[var.0]
    }

    pub(super) fn new_int_var(&mut self, domain: super::csp::Domain) -> IntVar {
        let id = self.int_var.len();
        self.int_var.push(domain);
        IntVar(id)
    }

    pub(super) fn get_domain_linear_sum(&self, linear_sum: &LinearSum) -> Domain {
        let mut ret = Domain::range(linear_sum.constant, linear_sum.constant);

        for (var, coef) in &linear_sum.term {
            ret = ret + self.int_var(*var).clone() * *coef;
        }

        ret
    }

    fn refine_var(&self, op: CmpOp, sum: &LinearSum, target: IntVar) -> Range {
        if op == CmpOp::Ne {
            return Range::any();
        }
        if op == CmpOp::Eq {
            return self.refine_var(CmpOp::Ge, sum, target)
                & self.refine_var(CmpOp::Le, sum, target);
        }

        let mut target_coef = None;
        let mut range_other = Range::constant(sum.constant);
        for (&v, &c) in &sum.term {
            if v == target {
                target_coef = Some(c);
            } else {
                let dom = self.int_var(v);
                range_other = range_other + Range::new(dom.lower_bound(), dom.upper_bound()) * c;
            }
        }

        let mut target_coef = target_coef.unwrap();
        assert_ne!(target_coef, 0);

        // Normalize `op` to `CmpOp::Ge` to reduce case analyses
        match op {
            CmpOp::Ge => (),
            CmpOp::Gt => range_other = range_other + Range::constant(-1),
            CmpOp::Le => {
                range_other = range_other * -1;
                target_coef = target_coef.checked_mul(-1).unwrap();
            }
            CmpOp::Lt => {
                range_other = range_other * -1 + Range::constant(-1);
                target_coef = target_coef.checked_mul(-1).unwrap();
            }
            CmpOp::Eq | CmpOp::Ne => unreachable!(),
        }

        if target_coef > 0 {
            let lb = div_ceil(-range_other.high, target_coef);
            Range::at_least(lb)
        } else {
            let ub = div_floor(range_other.high, -target_coef);
            Range::at_most(ub)
        }
    }

    fn refine_domain(&mut self, constraint: &Constraint) -> UpdateStatus {
        if !constraint.bool_lit.is_empty() {
            return UpdateStatus::NotUpdated;
        }

        let mut occurrence = BTreeMap::<IntVar, usize>::new();
        for linear_lit in &constraint.linear_lit {
            for (v, _) in linear_lit.sum.iter() {
                let n = occurrence.get(v).copied().unwrap_or(0);
                occurrence.insert(*v, n + 1);
            }
        }

        let mut status = UpdateStatus::NotUpdated;
        let common_vars = occurrence
            .iter()
            .filter_map(|(&v, &occ)| {
                if occ == constraint.linear_lit.len() {
                    Some(v)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        for v in common_vars {
            let mut range = Range::empty();
            for linear_lit in &constraint.linear_lit {
                range = range | self.refine_var(linear_lit.op, &linear_lit.sum, v);
            }

            let domain = &mut self.int_var[v.0];
            status |= domain.refine_lower_bound(range.low);
            status |= domain.refine_upper_bound(range.high);
        }
        status
    }
}

pub enum ExtraConstraint {
    ActiveVerticesConnected(Vec<BoolLit>, Vec<(usize, usize)>),
}

pub struct NormCSP {
    pub(super) vars: NormCSPVars,
    pub(super) constraints: Vec<Constraint>,
    pub(super) extra_constraints: Vec<ExtraConstraint>,
    pub(super) num_encoded_vars: usize,
    inconsistent: bool,
}

impl NormCSP {
    pub fn new() -> NormCSP {
        NormCSP {
            vars: NormCSPVars {
                num_bool_var: 0,
                int_var: vec![],
            },
            constraints: vec![],
            extra_constraints: vec![],
            num_encoded_vars: 0,
            inconsistent: false,
        }
    }

    pub fn new_bool_var(&mut self) -> BoolVar {
        let id = self.vars.num_bool_var;
        self.vars.num_bool_var += 1;
        BoolVar(id)
    }

    pub fn new_int_var(&mut self, domain: super::csp::Domain) -> IntVar {
        self.vars.new_int_var(domain)
    }

    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    pub fn add_extra_constraint(&mut self, constraint: ExtraConstraint) {
        self.extra_constraints.push(constraint);
    }

    pub fn bool_vars_iter(&self) -> impl Iterator<Item = BoolVar> {
        self.vars.bool_vars_iter()
    }

    pub fn int_vars_iter(&self) -> impl Iterator<Item = IntVar> {
        self.vars.int_vars_iter()
    }

    pub fn unencoded_int_vars(&self) -> impl Iterator<Item = IntVar> {
        (self.num_encoded_vars..self.vars.int_var.len()).map(|x| IntVar(x))
    }

    pub(super) fn get_domain_linear_sum(&self, linear_sum: &LinearSum) -> Domain {
        self.vars.get_domain_linear_sum(linear_sum)
    }

    pub fn is_inconsistent(&self) -> bool {
        self.inconsistent
    }

    pub fn refine_domain(&mut self) {
        loop {
            let mut update_status = UpdateStatus::NotUpdated;

            for constraint in &self.constraints {
                update_status |= self.vars.refine_domain(constraint);
            }

            match update_status {
                UpdateStatus::NotUpdated => break,
                UpdateStatus::Updated => (),
                UpdateStatus::Unsatisfiable => {
                    self.inconsistent = true;
                    return;
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct Assignment {
    bool_val: BTreeMap<BoolVar, bool>,
    int_val: BTreeMap<IntVar, i32>,
}

impl Assignment {
    pub fn new() -> Assignment {
        Assignment {
            bool_val: BTreeMap::new(),
            int_val: BTreeMap::new(),
        }
    }

    pub fn set_bool(&mut self, var: BoolVar, val: bool) {
        self.bool_val.insert(var, val);
    }

    pub fn set_int(&mut self, var: IntVar, val: i32) {
        self.int_val.insert(var, val);
    }

    pub fn eval_constraint(&self, constr: &Constraint) -> bool {
        for l in &constr.bool_lit {
            if self.bool_val.get(&l.var).unwrap() ^ l.negated {
                return true;
            }
        }
        for l in &constr.linear_lit {
            let sum = &l.sum;
            let mut v = sum.constant;
            for (var, coef) in &sum.term {
                v = v
                    .checked_add(self.int_val.get(var).unwrap().checked_mul(*coef).unwrap())
                    .unwrap();
            }

            if match l.op {
                CmpOp::Eq => v == 0,
                CmpOp::Ne => v != 0,
                CmpOp::Le => v <= 0,
                CmpOp::Lt => v < 0,
                CmpOp::Ge => v >= 0,
                CmpOp::Gt => v > 0,
            } {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn construct_linear_sum(terms: &[(IntVar, i32)], constant: i32) -> LinearSum {
        let mut ret = LinearSum::constant(constant);
        for &(v, c) in terms {
            ret.add_coef(v, c);
        }
        ret
    }

    #[test]
    fn test_norm_csp_refinement1() {
        let mut norm_csp = NormCSP::new();

        let a = norm_csp.new_int_var(Domain::range(0, 100));
        let b = norm_csp.new_int_var(Domain::range(0, 90));
        let c = norm_csp.new_int_var(Domain::range(0, 80));

        let mut constraint = Constraint::new();
        constraint.add_linear(LinearLit::new(
            construct_linear_sum(&[(a, 2), (b, 3), (c, 4)], -70),
            CmpOp::Le,
        ));

        assert_eq!(
            norm_csp.vars.refine_domain(&constraint),
            UpdateStatus::Updated
        );
        assert_eq!(norm_csp.vars.int_var(a).lower_bound(), 0);
        assert_eq!(norm_csp.vars.int_var(a).upper_bound(), 35);
        assert_eq!(norm_csp.vars.int_var(b).lower_bound(), 0);
        assert_eq!(norm_csp.vars.int_var(b).upper_bound(), 23);
        assert_eq!(norm_csp.vars.int_var(c).lower_bound(), 0);
        assert_eq!(norm_csp.vars.int_var(c).upper_bound(), 17);
    }

    #[test]
    fn test_norm_csp_refinement2() {
        let mut norm_csp = NormCSP::new();

        let a = norm_csp.new_int_var(Domain::range(0, 100));
        let b = norm_csp.new_int_var(Domain::range(-10, 90));

        let mut constraint = Constraint::new();
        constraint.add_linear(LinearLit::new(
            construct_linear_sum(&[(a, 2), (b, -3)], -71),
            CmpOp::Ge,
        ));

        assert_eq!(
            norm_csp.vars.refine_domain(&constraint),
            UpdateStatus::Updated
        );
        assert_eq!(norm_csp.vars.int_var(a).lower_bound(), 21);
        assert_eq!(norm_csp.vars.int_var(a).upper_bound(), 100);
        assert_eq!(norm_csp.vars.int_var(b).lower_bound(), -10);
        assert_eq!(norm_csp.vars.int_var(b).upper_bound(), 43);
    }
}
