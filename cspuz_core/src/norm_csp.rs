// Normalized CSP

use std::collections::BTreeMap;
use std::ops::Not;

use super::domain::Domain;
use crate::arithmetic::{CheckedInt, CmpOp, Range};
use crate::custom_constraints::PropagatorGenerator;
use crate::util::{ConvertMapIndex, UpdateStatus};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct BoolVar(usize);

impl BoolVar {
    pub fn id(&self) -> usize {
        self.0
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
    pub fn id(&self) -> usize {
        self.0
    }
}

impl ConvertMapIndex for IntVar {
    fn to_index(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoolLit {
    pub(super) var: BoolVar,
    pub(super) negated: bool,
}

impl BoolLit {
    pub fn new(var: BoolVar, negated: bool) -> BoolLit {
        BoolLit { var, negated }
    }

    pub fn negate_if(self, cond: bool) -> BoolLit {
        BoolLit {
            var: self.var,
            negated: self.negated ^ cond,
        }
    }
}

impl Not for BoolLit {
    type Output = BoolLit;

    fn not(self) -> Self::Output {
        BoolLit {
            var: self.var,
            negated: !self.negated,
        }
    }
}

pub type LinearSum = super::arithmetic::LinearSum<IntVar>;

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

    pub fn pretty_print<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        write!(out, "[")?;
        let mut is_first = true;
        for lit in &self.bool_lit {
            if !is_first {
                write!(out, " ")?;
            } else {
                is_first = false;
            }
            if lit.negated {
                write!(out, "!")?;
            }
            write!(out, "<nb{}>", lit.var.id())?;
        }

        for lit in &self.linear_lit {
            if !is_first {
                write!(out, " ")?;
            } else {
                is_first = false;
            }

            for (i, (var, coef)) in lit.sum.iter().enumerate() {
                if i > 0 {
                    write!(out, "+")?;
                }
                write!(out, "<ni{}>*{}", var.id(), coef.get())?;
            }

            write!(out, "+{}{}0", lit.sum.constant.get(), lit.op)?;
        }
        write!(out, "]")?;
        Ok(())
    }
}

pub(super) enum IntVarRepresentation {
    Domain(super::domain::Domain),
    Binary(BoolLit, CheckedInt, CheckedInt), // condition, false, true (order encoding)
}

impl IntVarRepresentation {
    #[allow(unused)]
    pub(super) fn enumerate(&self) -> Vec<CheckedInt> {
        match self {
            IntVarRepresentation::Domain(domain) => domain.enumerate(),
            &IntVarRepresentation::Binary(_, f, t) => {
                if f < t {
                    vec![f, t]
                } else if f > t {
                    vec![t, f]
                } else {
                    vec![f]
                }
            }
        }
    }

    #[allow(dead_code)]
    pub(super) fn is_domain(&self) -> bool {
        match self {
            IntVarRepresentation::Domain(_) => true,
            _ => false,
        }
    }

    #[allow(dead_code)]
    fn as_domain(&self) -> &super::domain::Domain {
        match self {
            IntVarRepresentation::Domain(domain) => domain,
            _ => panic!(),
        }
    }

    pub(super) fn lower_bound_checked(&self) -> CheckedInt {
        match self {
            IntVarRepresentation::Domain(domain) => domain.lower_bound_checked(),
            IntVarRepresentation::Binary(_, t, f) => (*t).min(*f),
        }
    }

    pub(super) fn upper_bound_checked(&self) -> CheckedInt {
        match self {
            IntVarRepresentation::Domain(domain) => domain.upper_bound_checked(),
            IntVarRepresentation::Binary(_, t, f) => (*t).max(*f),
        }
    }
}

pub(super) struct NormCSPVars {
    num_bool_var: usize,
    int_var: Vec<IntVarRepresentation>,
}

impl NormCSPVars {
    #[allow(dead_code)]
    pub(super) fn new() -> NormCSPVars {
        NormCSPVars {
            num_bool_var: 0,
            int_var: vec![],
        }
    }

    pub(super) fn bool_vars_iter(&self) -> impl Iterator<Item = BoolVar> {
        (0..self.num_bool_var).map(|i| BoolVar(i))
    }

    pub(super) fn int_vars_iter(&self) -> impl Iterator<Item = IntVar> {
        (0..self.int_var.len()).map(|i| IntVar(i))
    }

    pub(super) fn int_var(&self, var: IntVar) -> &IntVarRepresentation {
        &self.int_var[var.0]
    }

    pub(super) fn num_int_vars(&self) -> usize {
        self.int_var.len()
    }

    pub(super) fn new_int_var(&mut self, domain: IntVarRepresentation) -> IntVar {
        let id = self.int_var.len();
        self.int_var.push(domain);
        IntVar(id)
    }

    pub(super) fn get_domain_linear_sum(&self, linear_sum: &LinearSum) -> Domain {
        let mut ret = Domain::range_from_checked(linear_sum.constant, linear_sum.constant);

        for (var, coef) in &linear_sum.term {
            match self.int_var(*var) {
                IntVarRepresentation::Domain(dom) => ret = ret + dom.clone() * *coef,
                IntVarRepresentation::Binary(_, t, f) => {
                    let dom = Domain::range(t.min(f).get(), t.max(f).get());
                    ret = ret + dom * *coef;
                }
            }
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
                let repr = self.int_var(v);
                range_other = range_other
                    + Range::new(repr.lower_bound_checked(), repr.upper_bound_checked()) * c;
            }
        }

        let mut target_coef = target_coef.unwrap();
        assert_ne!(target_coef, 0);

        // Normalize `op` to `CmpOp::Ge` to reduce case analyses
        match op {
            CmpOp::Ge => (),
            CmpOp::Gt => range_other = range_other + Range::constant(CheckedInt::new(-1)),
            CmpOp::Le => {
                range_other = range_other * CheckedInt::new(-1);
                target_coef = -target_coef;
            }
            CmpOp::Lt => {
                range_other =
                    range_other * CheckedInt::new(-1) + Range::constant(CheckedInt::new(-1));
                target_coef = -target_coef;
            }
            CmpOp::Eq | CmpOp::Ne => unreachable!(),
        }

        if range_other.is_empty() {
            Range::empty()
        } else if target_coef > 0 {
            let lb = (-range_other.high).div_ceil(target_coef);
            Range::at_least(lb)
        } else {
            let ub = range_other.high.div_floor(-target_coef);
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

            if let IntVarRepresentation::Domain(domain) = &mut self.int_var[v.0] {
                status |= domain.refine_lower_bound(range.low);
                status |= domain.refine_upper_bound(range.high);
            }
        }
        status
    }
}

pub enum ExtraConstraint {
    ActiveVerticesConnected(Vec<BoolLit>, Vec<(usize, usize)>),
    Mul(IntVar, IntVar, IntVar),
    ExtensionSupports(Vec<IntVar>, Vec<Vec<Option<CheckedInt>>>),
    GraphDivision(Vec<Option<IntVar>>, Vec<(usize, usize)>, Vec<BoolLit>),
    CustomConstraint(Vec<BoolLit>, Box<dyn PropagatorGenerator>),
}

pub struct NormCSP {
    pub(super) vars: NormCSPVars,
    pub(super) constraints: Vec<Constraint>,
    pub(super) extra_constraints: Vec<ExtraConstraint>,
    pub(super) num_encoded_vars: usize,
    inconsistent: bool,
    pub(super) prenormalize_vars: Vec<BoolVar>,
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
            prenormalize_vars: vec![],
        }
    }

    pub fn new_bool_var(&mut self) -> BoolVar {
        let id = self.vars.num_bool_var;
        self.vars.num_bool_var += 1;
        BoolVar(id)
    }

    pub fn new_int_var(&mut self, domain: super::domain::Domain) -> IntVar {
        self.vars.new_int_var(IntVarRepresentation::Domain(domain))
    }

    pub fn add_prenormalize_var(&mut self, var: BoolVar) {
        self.prenormalize_vars.push(var);
    }

    pub fn new_binary_int_var(
        &mut self,
        cond: BoolLit,
        val_true: CheckedInt,
        val_false: CheckedInt,
    ) -> IntVar {
        if val_true < val_false {
            self.vars
                .new_int_var(IntVarRepresentation::Binary(!cond, val_true, val_false))
        } else {
            self.vars
                .new_int_var(IntVarRepresentation::Binary(cond, val_false, val_true))
        }
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
    int_val: BTreeMap<IntVar, CheckedInt>,
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
        self.int_val.insert(var, CheckedInt::new(val));
    }

    pub fn get_bool(&self, var: BoolVar) -> Option<bool> {
        self.bool_val.get(&var).copied()
    }

    pub fn get_int(&self, var: IntVar) -> Option<CheckedInt> {
        self.int_val.get(&var).copied()
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
                v = v + self.int_val.get(var).copied().unwrap() * *coef;
            }

            if l.op.compare(v, CheckedInt::new(0)) {
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
        let mut ret = LinearSum::constant(CheckedInt::new(constant));
        for &(v, c) in terms {
            ret.add_coef(v, CheckedInt::new(c));
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
        assert_eq!(
            norm_csp.vars.int_var(a).as_domain().lower_bound_checked(),
            0
        );
        assert_eq!(
            norm_csp.vars.int_var(a).as_domain().upper_bound_checked(),
            35
        );
        assert_eq!(
            norm_csp.vars.int_var(b).as_domain().lower_bound_checked(),
            0
        );
        assert_eq!(
            norm_csp.vars.int_var(b).as_domain().upper_bound_checked(),
            23
        );
        assert_eq!(
            norm_csp.vars.int_var(c).as_domain().lower_bound_checked(),
            0
        );
        assert_eq!(
            norm_csp.vars.int_var(c).as_domain().upper_bound_checked(),
            17
        );
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
        assert_eq!(
            norm_csp.vars.int_var(a).as_domain().lower_bound_checked(),
            CheckedInt::new(21)
        );
        assert_eq!(
            norm_csp.vars.int_var(a).as_domain().upper_bound_checked(),
            CheckedInt::new(100)
        );
        assert_eq!(
            norm_csp.vars.int_var(b).as_domain().lower_bound_checked(),
            CheckedInt::new(-10)
        );
        assert_eq!(
            norm_csp.vars.int_var(b).as_domain().upper_bound_checked(),
            CheckedInt::new(43)
        );
    }

    #[test]
    fn test_norm_csp_refinement3() {
        let mut norm_csp = NormCSP::new();

        let a = norm_csp.new_int_var(Domain::range(1, 2));
        let b = norm_csp.new_int_var(Domain::range(3, 4));
        let c = norm_csp.new_int_var(Domain::range(7, 8));

        let mut constraint1 = Constraint::new();
        constraint1.add_linear(LinearLit::new(
            construct_linear_sum(&[(a, 1), (b, 1), (c, -1)], 0),
            CmpOp::Ge,
        ));

        assert_eq!(
            norm_csp.vars.refine_domain(&constraint1),
            UpdateStatus::Unsatisfiable
        );

        let mut constraint2 = Constraint::new();
        constraint2.add_linear(LinearLit::new(
            construct_linear_sum(&[(a, 1), (b, 1), (c, -1)], -10),
            CmpOp::Le,
        ));

        norm_csp.vars.refine_domain(&constraint1);
    }
}
