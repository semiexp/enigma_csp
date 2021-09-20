use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::norm_csp::{
    BoolLit, BoolVar, Constraint, ExtraConstraint, IntVar, LinearLit, LinearSum, NormCSP,
    NormCSPVars,
};
use super::sat::{Lit, SATModel, VarArray, SAT};
use super::CmpOp;
use crate::util::{div_ceil, ConvertMap};

/// Order encoding of an integer variable with domain of `domain`.
/// `vars[i]` is the logical variable representing (the value of this int variable) >= `domain[i+1]`.
pub struct OrderEncoding {
    domain: Vec<i32>,
    vars: VarArray,
}

pub struct EncodeMap {
    bool_map: ConvertMap<BoolVar, Lit>, // mapped to Lit rather than Var so that further optimization can be done
    int_map: ConvertMap<IntVar, OrderEncoding>,
}

impl EncodeMap {
    pub fn new() -> EncodeMap {
        EncodeMap {
            bool_map: ConvertMap::new(),
            int_map: ConvertMap::new(),
        }
    }

    fn convert_bool_var(&mut self, _norm_vars: &NormCSPVars, sat: &mut SAT, var: BoolVar) -> Lit {
        match self.bool_map[var] {
            Some(x) => x,
            None => {
                let ret = sat.new_var().as_lit(false);
                self.bool_map[var] = Some(ret);
                ret
            }
        }
    }

    fn convert_bool_lit(&mut self, norm_vars: &NormCSPVars, sat: &mut SAT, lit: BoolLit) -> Lit {
        let var_lit = self.convert_bool_var(norm_vars, sat, lit.var);
        if lit.negated {
            !var_lit
        } else {
            var_lit
        }
    }

    fn convert_int_var(&mut self, norm_vars: &NormCSPVars, sat: &mut SAT, var: IntVar) {
        // Currently, only order encoding is supported

        if self.int_map[var].is_none() {
            let domain = norm_vars.int_var(var).enumerate();
            assert_ne!(domain.len(), 0);
            let vars = sat.new_vars((domain.len() - 1) as i32);
            for i in 1..vars.len() {
                // vars[i] implies vars[i - 1]
                sat.add_clause(vec![vars.at(i).as_lit(true), vars.at(i - 1).as_lit(false)]);
            }

            self.int_map[var] = Some(OrderEncoding { domain, vars });
        }
    }

    pub fn get_bool_var(&self, var: BoolVar) -> Option<Lit> {
        self.bool_map[var]
    }

    pub fn get_int_value(&self, model: &SATModel, var: IntVar) -> Option<i32> {
        if self.int_map[var].is_none() {
            return None;
        }
        let encoding = self.int_map[var].as_ref().unwrap();

        // Find the number of true value in `encoding.vars`
        let mut left = 0;
        let mut right = encoding.vars.len();
        while left < right {
            let mid = (left + right + 1) / 2;
            if model.assignment(encoding.vars.at(mid - 1)) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        Some(encoding.domain[left as usize])
    }
}

struct EncoderEnv<'a, 'b, 'c> {
    norm_vars: &'a mut NormCSPVars,
    sat: &'b mut SAT,
    map: &'c mut EncodeMap,
}

impl<'a, 'b, 'c> EncoderEnv<'a, 'b, 'c> {
    fn convert_bool_lit(&mut self, lit: BoolLit) -> Lit {
        self.map.convert_bool_lit(self.norm_vars, self.sat, lit)
    }
}

pub fn encode(norm: &mut NormCSP, sat: &mut SAT, map: &mut EncodeMap) {
    for var in norm.unencoded_int_vars() {
        map.convert_int_var(&mut norm.vars, sat, var);
    }
    norm.num_encoded_vars = norm.vars.int_var.len();

    let mut env = EncoderEnv {
        norm_vars: &mut norm.vars,
        sat,
        map,
    };

    let constrs = std::mem::replace(&mut norm.constraints, vec![]);
    for constr in constrs {
        encode_constraint(&mut env, constr);
    }

    let extra_constrs = std::mem::replace(&mut norm.extra_constraints, vec![]);
    for constr in extra_constrs {
        match constr {
            ExtraConstraint::ActiveVerticesConnected(vertices, edges) => {
                let lits = vertices
                    .into_iter()
                    .map(|l| env.convert_bool_lit(l))
                    .collect::<Vec<_>>();
                env.sat.add_active_vertices_connected(lits, edges);
            }
        }
    }
}

fn encode_constraint(env: &mut EncoderEnv, constr: Constraint) {
    if constr.linear_lit.len() == 0 {
        let mut clause = vec![];
        for lit in constr.bool_lit {
            clause.push(env.convert_bool_lit(lit));
        }

        env.sat.add_clause(clause);
        return;
    }
    let mut linear_lits = vec![];

    for mut linear_lit in constr.linear_lit {
        match linear_lit.op {
            CmpOp::Eq => {
                linear_lits.push(linear_lit);
            }
            CmpOp::Ne => {
                {
                    let mut linear_lit = linear_lit.clone();
                    linear_lit.sum *= -1;
                    linear_lit.sum.add_constant(-1);
                    linear_lits.push(LinearLit::new(linear_lit.sum, CmpOp::Ge));
                }
                {
                    linear_lit.sum.add_constant(-1);
                    linear_lits.push(LinearLit::new(linear_lit.sum, CmpOp::Ge));
                }
            }
            CmpOp::Le => {
                linear_lit.sum *= -1;
                linear_lits.push(LinearLit::new(linear_lit.sum, CmpOp::Ge));
            }
            CmpOp::Lt => {
                linear_lit.sum *= -1;
                linear_lit.sum.add_constant(-1);
                linear_lits.push(LinearLit::new(linear_lit.sum, CmpOp::Ge));
            }
            CmpOp::Ge => {
                linear_lits.push(LinearLit::new(linear_lit.sum, CmpOp::Ge));
            }
            CmpOp::Gt => {
                linear_lit.sum.add_constant(-1);
                linear_lits.push(LinearLit::new(linear_lit.sum, CmpOp::Ge));
            }
        }
    }

    fn encode(env: &mut EncoderEnv, mut linear: LinearLit, bool_lit: &Vec<Lit>) {
        match linear.op {
            CmpOp::Ge => {
                encode_linear_ge_with_simplification(env, &linear.sum, bool_lit);
            }
            CmpOp::Eq => {
                // TODO: don't create the same aux vars twice
                encode_linear_ge_with_simplification(env, &linear.sum, bool_lit);
                linear.sum *= -1;
                encode_linear_ge_with_simplification(env, &linear.sum, bool_lit);
            }
            _ => unimplemented!(),
        }
    }

    let mut bool_lit = constr
        .bool_lit
        .into_iter()
        .map(|lit| env.convert_bool_lit(lit))
        .collect::<Vec<_>>();

    if linear_lits.len() == 1 {
        encode(env, linear_lits.remove(0), &bool_lit);
    } else {
        for linear_lit in linear_lits {
            let aux = env.sat.new_var();
            bool_lit.push(aux.as_lit(false));
            encode(env, linear_lit, &vec![aux.as_lit(true)]);
        }
        env.sat.add_clause(bool_lit);
    }
}

enum ExtendedLit {
    True,
    False,
    Lit(Lit),
}

/// Helper struct for encoding linear constraints on variables represented in order encoding.
/// With this struct, all coefficients can be virtually treated as positive.
struct LinearInfoForOrderEncoding<'a> {
    coef: Vec<i32>,
    encoding: Vec<&'a OrderEncoding>,
}

impl<'a> LinearInfoForOrderEncoding<'a> {
    pub fn new(coef: Vec<i32>, encoding: Vec<&'a OrderEncoding>) -> LinearInfoForOrderEncoding<'a> {
        LinearInfoForOrderEncoding { coef, encoding }
    }

    fn len(&self) -> usize {
        self.coef.len()
    }

    /// Coefficient of the i-th variable (after normalizing negative coefficients)
    fn coef(&self, i: usize) -> i32 {
        self.coef[i].abs()
    }

    fn domain_size(&self, i: usize) -> usize {
        self.encoding[i].domain.len()
    }

    /// j-th smallest domain value for the i-th variable (after normalizing negative coefficients)
    fn domain(&self, i: usize, j: usize) -> i32 {
        if self.coef[i] > 0 {
            self.encoding[i].domain[j]
        } else {
            -self.encoding[i].domain[self.encoding[i].domain.len() - 1 - j]
        }
    }

    /// The literal asserting that (the value of the i-th variable) is at least `domain(i, j)`.
    fn at_least(&self, i: usize, j: usize) -> Lit {
        assert!(0 < j && j < self.encoding[i].domain.len());
        if self.coef[i] > 0 {
            self.encoding[i].vars.at((j - 1) as i32).as_lit(false)
        } else {
            self.encoding[i]
                .vars
                .at((self.encoding[i].domain.len() - 1 - j) as i32)
                .as_lit(true)
        }
    }

    /// The literal asserting (x >= val) under the assumption that x is in the domain of the i-th variable.
    fn at_least_val(&self, i: usize, val: i32) -> ExtendedLit {
        let dom_size = self.domain_size(i);

        if val <= self.domain(i, 0) {
            ExtendedLit::True
        } else if val > self.domain(i, dom_size - 1) {
            ExtendedLit::False
        } else {
            // compute the largest j such that val <= domain[j]
            let mut left = 0;
            let mut right = dom_size - 1;

            while left < right {
                let mid = (left + right) / 2;
                if val <= self.domain(i, mid) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }

            ExtendedLit::Lit(self.at_least(i, left))
        }
    }
}

const DOM_PRODUCT_THRESHOLD: usize = 1000; // TODO: make this parameter configurable

fn encode_linear_ge_with_simplification(
    env: &mut EncoderEnv,
    sum: &LinearSum,
    bool_lit: &Vec<Lit>,
) {
    let mut heap = BinaryHeap::new();
    for (&var, &coef) in &sum.term {
        let dom_size = env.map.int_map[var].as_ref().unwrap().domain.len();
        heap.push(Reverse((dom_size, var, coef)));
    }

    let mut pending: Vec<(usize, IntVar, i32)> = vec![];
    let mut dom_product = 1usize;
    while let Some(&Reverse(top)) = heap.peek() {
        let (dom_size, _, _) = top;
        if dom_product * dom_size >= DOM_PRODUCT_THRESHOLD && pending.len() >= 2 && heap.len() >= 2
        {
            // Introduce auxiliary variable which aggregates current pending terms
            let mut aux_sum = LinearSum::new();
            for &(_, var, coef) in &pending {
                aux_sum.add_coef(var, coef);
            }
            let aux_dom = env.norm_vars.get_domain_linear_sum(&aux_sum);
            let aux_var = env.norm_vars.new_int_var(aux_dom);
            env.map
                .convert_int_var(&mut env.norm_vars, &mut env.sat, aux_var);

            // aux_sum >= aux_var
            aux_sum.add_coef(aux_var, -1);
            if pending.len() <= 3 {
                // TODO: make this parameter configurable
                encode_linear_ge_direct(env, &aux_sum, &vec![]);
            } else {
                encode_linear_ge(env, &aux_sum, &vec![]);
            }

            pending.clear();
            let dom_size = env.map.int_map[aux_var].as_ref().unwrap().domain.len();
            pending.push((dom_size, aux_var, 1));
            dom_product = dom_size;

            continue;
        }
        dom_product *= dom_size;
        pending.push(top);
        heap.pop();
    }

    let mut sum = LinearSum::constant(sum.constant);
    for &(_, var, coef) in &pending {
        sum.add_coef(var, coef);
    }
    encode_linear_ge(env, &sum, bool_lit);
}

fn encode_linear_ge_direct(env: &mut EncoderEnv, sum: &LinearSum, bool_lit: &Vec<Lit>) {
    assert!(bool_lit.is_empty()); // TODO

    let mut coef = vec![];
    let mut encoding = vec![];
    for (v, c) in sum.terms() {
        assert_ne!(c, 0);
        coef.push(c);
        encoding.push(env.map.int_map[v].as_ref().unwrap());
    }
    let info = LinearInfoForOrderEncoding::new(coef, encoding);

    let mut lits = vec![];
    let mut domain = vec![];
    let mut coefs = vec![];
    let constant = sum.constant;

    for i in 0..info.len() {
        let mut lits_r = vec![];
        let mut domain_r = vec![];
        for j in 0..info.domain_size(i) {
            if j > 0 {
                lits_r.push(info.at_least(i, j));
            }
            domain_r.push(info.domain(i, j));
        }
        lits.push(lits_r);
        domain.push(domain_r);
        coefs.push(info.coef(i));
    }

    env.sat
        .add_order_encoding_linear(lits, domain, coefs, constant);
}

fn encode_linear_ge(env: &mut EncoderEnv, sum: &LinearSum, bool_lit: &Vec<Lit>) {
    // TODO: better ordering of variables
    let mut coef = vec![];
    let mut encoding = vec![];
    for (v, c) in sum.terms() {
        assert_ne!(c, 0);
        coef.push(c);
        encoding.push(env.map.int_map[v].as_ref().unwrap());
    }
    let info = LinearInfoForOrderEncoding::new(coef, encoding);

    fn encode_sub(
        info: &LinearInfoForOrderEncoding,
        sat: &mut SAT,
        clause: &mut Vec<Lit>,
        idx: usize,
        constant: i32,
    ) {
        if idx == info.len() {
            if constant < 0 {
                sat.add_clause(clause.clone());
            }
            return;
        }
        if idx + 1 == info.len() {
            let a = info.coef(idx);
            // a * x >= -constant
            // x >= ceil(-constant / a)
            let threshold = div_ceil(-constant, a);
            match info.at_least_val(idx, threshold) {
                ExtendedLit::True => (),
                ExtendedLit::False => sat.add_clause(clause.clone()),
                ExtendedLit::Lit(lit) => {
                    clause.push(lit);
                    sat.add_clause(clause.clone());
                    clause.pop();
                }
            }
            return;
        }
        let domain_size = info.domain_size(idx);
        for j in 0..domain_size {
            let new_constant = constant
                .checked_add(info.domain(idx, j).checked_mul(info.coef(idx)).unwrap())
                .unwrap();
            if j + 1 < domain_size {
                clause.push(info.at_least(idx, j + 1));
            }
            encode_sub(info, sat, clause, idx + 1, new_constant);
            if j + 1 < domain_size {
                clause.pop();
            }
        }
    }

    let mut clause = bool_lit.clone();
    encode_sub(&info, &mut env.sat, &mut clause, 0, sum.constant);
}
