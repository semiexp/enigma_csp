use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};

use super::config::Config;
use super::norm_csp::{
    BoolLit, BoolVar, Constraint, ExtraConstraint, IntVar, IntVarRepresentation, LinearLit,
    LinearSum, NormCSP, NormCSPVars,
};
use super::sat::{Lit, SATModel, SAT};
use super::CmpOp;
use crate::arithmetic::{CheckedInt, Range};
use crate::util::ConvertMap;

/// Order encoding of an integer variable with domain of `domain`.
/// `vars[i]` is the logical variable representing (the value of this int variable) >= `domain[i+1]`.
struct OrderEncoding {
    domain: Vec<CheckedInt>,
    lits: Vec<Lit>,
}

impl OrderEncoding {
    fn range(&self) -> Range {
        if self.domain.is_empty() {
            Range::empty()
        } else {
            Range::new(self.domain[0], self.domain[self.domain.len() - 1])
        }
    }
}

struct DirectEncoding {
    domain: Vec<CheckedInt>,
    lits: Vec<Lit>,
}

impl DirectEncoding {
    fn range(&self) -> Range {
        if self.domain.is_empty() {
            Range::empty()
        } else {
            Range::new(self.domain[0], self.domain[self.domain.len() - 1])
        }
    }
}

enum Encoding {
    OrderEncoding(OrderEncoding),
    DirectEncoding(DirectEncoding),
}

impl Encoding {
    fn as_order_encoding(&self) -> &OrderEncoding {
        match self {
            Encoding::OrderEncoding(e) => e,
            _ => panic!(),
        }
    }

    fn as_direct_encoding(&self) -> &DirectEncoding {
        match self {
            Encoding::DirectEncoding(e) => e,
            _ => panic!(),
        }
    }

    fn is_direct_encoding(&self) -> bool {
        match self {
            Encoding::DirectEncoding(_) => true,
            _ => false,
        }
    }

    fn range(&self) -> Range {
        match self {
            Encoding::OrderEncoding(encoding) => encoding.range(),
            Encoding::DirectEncoding(encoding) => encoding.range(),
        }
    }
}

pub struct EncodeMap {
    bool_map: ConvertMap<BoolVar, Lit>, // mapped to Lit rather than Var so that further optimization can be done
    int_map: ConvertMap<IntVar, Encoding>,
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

    fn convert_int_var_order_encoding(
        &mut self,
        norm_vars: &NormCSPVars,
        sat: &mut SAT,
        var: IntVar,
    ) {
        if self.int_map[var].is_none() {
            match norm_vars.int_var(var) {
                IntVarRepresentation::Domain(domain) => {
                    let domain = domain.enumerate();
                    assert_ne!(domain.len(), 0);
                    let lits = sat.new_vars_as_lits(domain.len() - 1);
                    for i in 1..lits.len() {
                        // vars[i] implies vars[i - 1]
                        sat.add_clause(vec![!lits[i], lits[i - 1]]);
                    }

                    self.int_map[var] =
                        Some(Encoding::OrderEncoding(OrderEncoding { domain, lits }));
                }
                &IntVarRepresentation::Binary(cond, t, f) => {
                    let domain;
                    let lits;
                    if f <= t {
                        domain = vec![f, t];
                        lits = vec![self.convert_bool_var(norm_vars, sat, cond)];
                    } else {
                        domain = vec![t, f];
                        lits = vec![!self.convert_bool_var(norm_vars, sat, cond)];
                    }
                    self.int_map[var] =
                        Some(Encoding::OrderEncoding(OrderEncoding { domain, lits }));
                }
            }
        }
    }

    fn convert_int_var_direct_encoding(
        &mut self,
        norm_vars: &NormCSPVars,
        sat: &mut SAT,
        var: IntVar,
    ) {
        if self.int_map[var].is_none() {
            let domain = norm_vars.int_var(var).enumerate();
            assert_ne!(domain.len(), 0);
            let lits = sat.new_vars_as_lits(domain.len());
            sat.add_clause(lits.clone());
            for i in 1..lits.len() {
                for j in 0..i {
                    sat.add_clause(vec![!lits[i], !lits[j]]);
                }
            }

            self.int_map[var] = Some(Encoding::DirectEncoding(DirectEncoding { domain, lits }));
        }
    }

    pub fn get_bool_var(&self, var: BoolVar) -> Option<Lit> {
        self.bool_map[var]
    }

    pub fn get_bool_lit(&self, lit: BoolLit) -> Option<Lit> {
        self.bool_map[lit.var].map(|l| if lit.negated { !l } else { l })
    }

    pub(crate) fn get_int_value_checked(
        &self,
        model: &SATModel,
        var: IntVar,
    ) -> Option<CheckedInt> {
        if self.int_map[var].is_none() {
            return None;
        }
        let encoding = self.int_map[var].as_ref().unwrap();

        match encoding {
            Encoding::OrderEncoding(encoding) => {
                // Find the number of true value in `encoding.vars`
                let mut left = 0;
                let mut right = encoding.lits.len();
                while left < right {
                    let mid = (left + right + 1) / 2;
                    if model.assignment_lit(encoding.lits[mid - 1]) {
                        left = mid;
                    } else {
                        right = mid - 1;
                    }
                }
                Some(encoding.domain[left as usize])
            }
            Encoding::DirectEncoding(encoding) => {
                let mut ret = None;
                for i in 0..encoding.lits.len() {
                    if model.assignment_lit(encoding.lits[i]) {
                        assert!(
                            ret.is_none(),
                            "multiple indicator bits are set for a direct-encoded variable"
                        );
                        ret = Some(encoding.domain[i as usize]);
                    }
                }
                assert!(
                    ret.is_some(),
                    "no indicator bits are set for a direct-encoded variable"
                );
                ret
            }
        }
    }

    pub fn get_int_value(&self, model: &SATModel, var: IntVar) -> Option<i32> {
        self.get_int_value_checked(model, var).map(CheckedInt::get)
    }
}

struct EncoderEnv<'a, 'b, 'c, 'd> {
    norm_vars: &'a mut NormCSPVars,
    sat: &'b mut SAT,
    map: &'c mut EncodeMap,
    config: &'d Config,
}

impl<'a, 'b, 'c, 'd> EncoderEnv<'a, 'b, 'c, 'd> {
    fn convert_bool_lit(&mut self, lit: BoolLit) -> Lit {
        self.map.convert_bool_lit(self.norm_vars, self.sat, lit)
    }
}

pub fn encode(norm: &mut NormCSP, sat: &mut SAT, map: &mut EncodeMap, config: &Config) {
    let mut direct_encoding_vars = BTreeSet::<IntVar>::new();
    if config.use_direct_encoding {
        for var in norm.unencoded_int_vars() {
            if norm.vars.int_var(var).is_domain() {
                direct_encoding_vars.insert(var);
            }
        }
        for constr in &norm.constraints {
            for lit in &constr.linear_lit {
                let is_simple =
                    (lit.op == CmpOp::Eq || lit.op == CmpOp::Ne) && lit.sum.terms().len() <= 1;
                if !is_simple {
                    for (v, _) in lit.sum.iter() {
                        direct_encoding_vars.remove(v);
                    }
                }
            }
        }
    }
    for var in norm.unencoded_int_vars() {
        if direct_encoding_vars.contains(&var) {
            map.convert_int_var_direct_encoding(&mut norm.vars, sat, var);
        } else {
            map.convert_int_var_order_encoding(&mut norm.vars, sat, var);
        }
    }

    let mut env = EncoderEnv {
        norm_vars: &mut norm.vars,
        sat,
        map,
        config,
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
    norm.num_encoded_vars = norm.vars.int_var.len();
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
    let mut bool_lits_for_direct_encoding = vec![];

    for mut linear_lit in constr.linear_lit {
        // use direct encoding if applicable
        let use_simple_direct_encoding;
        {
            let terms = linear_lit.sum.terms();
            use_simple_direct_encoding = terms.len() == 1
                && env.map.int_map[terms[0].0]
                    .as_ref()
                    .unwrap()
                    .is_direct_encoding();
        }
        if use_simple_direct_encoding {
            let encoded = encode_simple_linear_direct_encoding(env, &linear_lit);
            if let Some(encoded) = encoded {
                bool_lits_for_direct_encoding.extend(encoded);
            } else {
                return;
            }
            continue;
        }

        // Unsatisfiable literals should be removed.
        // Otherwise, panic may happen (test_integration_csp_optimization3)
        let mut range = Range::constant(linear_lit.sum.constant);
        for (var, coef) in linear_lit.sum.terms() {
            let encoding = env.map.int_map[var].as_ref().unwrap();
            let var_range = encoding.range();
            range = range + var_range * coef;
        }
        match linear_lit.op {
            CmpOp::Eq => {
                if range.low > 0 || range.high < 0 {
                    continue;
                }
            }
            CmpOp::Ne => {
                if range.low == 0 && range.high == 0 {
                    continue;
                }
            }
            CmpOp::Le => {
                if range.low > 0 {
                    continue;
                }
            }
            CmpOp::Lt => {
                if range.low >= 0 {
                    continue;
                }
            }
            CmpOp::Ge => {
                if range.high < 0 {
                    continue;
                }
            }
            CmpOp::Gt => {
                if range.high <= 0 {
                    continue;
                }
            }
        }

        match linear_lit.op {
            CmpOp::Eq => {
                linear_lits.push(linear_lit);
            }
            CmpOp::Ne => {
                {
                    let mut linear_lit = linear_lit.clone();
                    linear_lit.sum *= -1;
                    linear_lit.sum += -1;
                    linear_lits.push(LinearLit::new(linear_lit.sum, CmpOp::Ge));
                }
                {
                    linear_lit.sum += -1;
                    linear_lits.push(LinearLit::new(linear_lit.sum, CmpOp::Ge));
                }
            }
            CmpOp::Le => {
                linear_lit.sum *= -1;
                linear_lits.push(LinearLit::new(linear_lit.sum, CmpOp::Ge));
            }
            CmpOp::Lt => {
                linear_lit.sum *= -1;
                linear_lit.sum += -1;
                linear_lits.push(LinearLit::new(linear_lit.sum, CmpOp::Ge));
            }
            CmpOp::Ge => {
                linear_lits.push(LinearLit::new(linear_lit.sum, CmpOp::Ge));
            }
            CmpOp::Gt => {
                linear_lit.sum += -1;
                linear_lits.push(LinearLit::new(linear_lit.sum, CmpOp::Ge));
            }
        }
    }

    fn encode(env: &mut EncoderEnv, linear: LinearLit, bool_lit: &Vec<Lit>) {
        // TODO: shared auxiliary variables may deteriorate the performance
        if linear.op == CmpOp::Eq {
            let mut linear = linear;
            linear.op = CmpOp::Ge;
            encode(env, linear.clone(), bool_lit);
            linear.sum *= -1;
            encode(env, linear, bool_lit);
            return;
        }

        let (mut decomposed, extra) = decompose_linear_lit(env, &linear);

        for mut linear in extra {
            match linear.op {
                CmpOp::Ge => {
                    encode_linear_ge_order_encoding(env, &linear.sum, &vec![]);
                }
                CmpOp::Eq => {
                    encode_linear_ge_order_encoding(env, &linear.sum, &vec![]);
                    linear.sum *= -1;
                    encode_linear_ge_order_encoding(env, &linear.sum, &vec![]);
                }
                _ => unreachable!(),
            }
        }
        match decomposed.op {
            CmpOp::Ge => {
                encode_linear_ge_order_encoding(env, &decomposed.sum, bool_lit);
            }
            CmpOp::Eq => {
                encode_linear_ge_order_encoding(env, &decomposed.sum, bool_lit);
                decomposed.sum *= -1;
                encode_linear_ge_order_encoding(env, &decomposed.sum, bool_lit);
            }
            _ => unreachable!(),
        }
    }

    let mut bool_lit = constr
        .bool_lit
        .into_iter()
        .map(|lit| env.convert_bool_lit(lit))
        .collect::<Vec<_>>();
    bool_lit.extend(bool_lits_for_direct_encoding);

    if linear_lits.len() == 1 {
        encode(env, linear_lits.remove(0), &bool_lit);
    } else {
        for linear_lit in linear_lits {
            if linear_lit.op == CmpOp::Ge && linear_lit.sum.len() == 1 {
                // v * coef + constant >= 0
                let constant = linear_lit.sum.constant;
                let (&v, &coef) = linear_lit.sum.iter().next().unwrap();

                let encoding = env.map.int_map[v].as_ref().unwrap().as_order_encoding();
                if coef > 0 {
                    let lb = (-constant).div_ceil(coef);
                    if encoding.domain[0] >= lb {
                        // already satisfied
                        return;
                    } else if encoding.domain[encoding.domain.len() - 1] < lb {
                        // skipped (unsatisfiable literal)
                        continue;
                    }
                    for i in 1..encoding.domain.len() {
                        if encoding.domain[i] >= lb {
                            bool_lit.push(encoding.lits[i - 1]);
                            break;
                        }
                    }
                } else {
                    let ub = constant.div_floor(-coef);
                    // v <= ub iff !(v >= ub + 1)
                    let nlb = ub + CheckedInt::new(1);
                    if encoding.domain[0] >= nlb {
                        continue;
                    } else if encoding.domain[encoding.domain.len() - 1] < nlb {
                        return;
                    }
                    for i in 1..encoding.domain.len() {
                        if encoding.domain[i] >= nlb {
                            bool_lit.push(!encoding.lits[i - 1]);
                            break;
                        }
                    }
                }
                continue;
            }

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
    coef: Vec<CheckedInt>,
    encoding: Vec<&'a OrderEncoding>,
}

impl<'a> LinearInfoForOrderEncoding<'a> {
    pub fn new(
        coef: Vec<CheckedInt>,
        encoding: Vec<&'a OrderEncoding>,
    ) -> LinearInfoForOrderEncoding<'a> {
        LinearInfoForOrderEncoding { coef, encoding }
    }

    fn len(&self) -> usize {
        self.coef.len()
    }

    /// Coefficient of the i-th variable (after normalizing negative coefficients)
    fn coef(&self, i: usize) -> CheckedInt {
        self.coef[i].abs()
    }

    fn domain_size(&self, i: usize) -> usize {
        self.encoding[i].domain.len()
    }

    /// j-th smallest domain value for the i-th variable (after normalizing negative coefficients)
    fn domain(&self, i: usize, j: usize) -> CheckedInt {
        if self.coef[i] > 0 {
            self.encoding[i].domain[j]
        } else {
            -self.encoding[i].domain[self.encoding[i].domain.len() - 1 - j]
        }
    }

    fn domain_min(&self, i: usize) -> CheckedInt {
        self.domain(i, 0)
    }

    fn domain_max(&self, i: usize) -> CheckedInt {
        self.domain(i, self.domain_size(i) - 1)
    }

    /// The literal asserting that (the value of the i-th variable) is at least `domain(i, j)`.
    fn at_least(&self, i: usize, j: usize) -> Lit {
        assert!(0 < j && j < self.encoding[i].domain.len());
        if self.coef[i] > 0 {
            self.encoding[i].lits[j - 1]
        } else {
            !self.encoding[i].lits[self.encoding[i].domain.len() - 1 - j]
        }
    }

    /// The literal asserting (x >= val) under the assumption that x is in the domain of the i-th variable.
    fn at_least_val(&self, i: usize, val: CheckedInt) -> ExtendedLit {
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

fn decompose_linear_lit(env: &mut EncoderEnv, lit: &LinearLit) -> (LinearLit, Vec<LinearLit>) {
    assert!(lit.op == CmpOp::Ge || lit.op == CmpOp::Eq);

    let mut heap = BinaryHeap::new();
    for (&var, &coef) in &lit.sum.term {
        let dom_size = env.map.int_map[var]
            .as_ref()
            .unwrap()
            .as_order_encoding()
            .domain
            .len();
        heap.push(Reverse((dom_size, var, coef)));
    }

    let mut ret = vec![];

    let mut pending: Vec<(usize, IntVar, CheckedInt)> = vec![];
    let mut dom_product = 1usize;
    while let Some(&Reverse(top)) = heap.peek() {
        let (dom_size, _, _) = top;
        if dom_product * dom_size >= env.config.domain_product_threshold
            && pending.len() >= 2
            && heap.len() >= 2
        {
            // Introduce auxiliary variable which aggregates current pending terms
            let mut aux_sum = LinearSum::new();
            for &(_, var, coef) in &pending {
                aux_sum.add_coef(var, coef);
            }
            let mut aux_dom = env.norm_vars.get_domain_linear_sum(&aux_sum);

            let mut rem_sum = LinearSum::new();
            for &Reverse((_, var, coef)) in &heap {
                rem_sum.add_coef(var, coef);
            }
            let rem_dom = env.norm_vars.get_domain_linear_sum(&rem_sum);
            aux_dom.refine_upper_bound(-(lit.sum.constant + rem_dom.lower_bound_checked()));
            aux_dom.refine_lower_bound(-(lit.sum.constant + rem_dom.upper_bound_checked()));

            let aux_var = env
                .norm_vars
                .new_int_var(IntVarRepresentation::Domain(aux_dom));
            env.map
                .convert_int_var_order_encoding(&mut env.norm_vars, &mut env.sat, aux_var);

            // aux_sum >= aux_var
            aux_sum.add_coef(aux_var, CheckedInt::new(-1));
            ret.push(LinearLit::new(aux_sum, lit.op));

            pending.clear();
            let dom_size = env.map.int_map[aux_var]
                .as_ref()
                .unwrap()
                .as_order_encoding()
                .domain
                .len();
            heap.push(Reverse((dom_size, aux_var, CheckedInt::new(1))));
            dom_product = 1;
            continue;
        }
        dom_product *= dom_size;
        pending.push(top);
        heap.pop();
    }

    let mut sum = LinearSum::constant(lit.sum.constant);
    for &(_, var, coef) in &pending {
        sum.add_coef(var, coef);
    }
    (LinearLit::new(sum, lit.op), ret)
}

fn encode_linear_ge_order_encoding(env: &mut EncoderEnv, sum: &LinearSum, bool_lit: &Vec<Lit>) {
    if bool_lit.is_empty() && sum.len() <= env.config.native_linear_encoding_terms {
        encode_linear_ge_order_encoding_native(env, sum, bool_lit);
    } else {
        encode_linear_ge_order_encoding_literals(env, sum, bool_lit);
    }
}

fn encode_linear_ge_order_encoding_native(
    env: &mut EncoderEnv,
    sum: &LinearSum,
    bool_lit: &Vec<Lit>,
) {
    assert!(bool_lit.is_empty()); // TODO

    let mut coef = vec![];
    let mut encoding = vec![];
    for (v, c) in sum.terms() {
        assert_ne!(c, 0);
        coef.push(c);
        encoding.push(env.map.int_map[v].as_ref().unwrap().as_order_encoding());
    }
    let info = LinearInfoForOrderEncoding::new(coef, encoding);

    let mut lits = vec![];
    let mut domain = vec![];
    let mut coefs = vec![];
    let constant = sum.constant.get();

    for i in 0..info.len() {
        let mut lits_r = vec![];
        let mut domain_r = vec![];
        for j in 0..info.domain_size(i) {
            if j > 0 {
                lits_r.push(info.at_least(i, j));
            }
            domain_r.push(info.domain(i, j).get());
        }
        lits.push(lits_r);
        domain.push(domain_r);
        coefs.push(info.coef(i).get());
    }

    env.sat
        .add_order_encoding_linear(lits, domain, coefs, constant);
}

fn encode_linear_ge_order_encoding_literals(
    env: &mut EncoderEnv,
    sum: &LinearSum,
    bool_lit: &Vec<Lit>,
) {
    let mut coef = vec![];
    let mut encoding = vec![];
    for (v, c) in sum.terms() {
        assert_ne!(c, 0);
        coef.push(c);
        encoding.push(env.map.int_map[v].as_ref().unwrap().as_order_encoding());
    }
    let info = LinearInfoForOrderEncoding::new(coef, encoding);

    fn encode_sub(
        info: &LinearInfoForOrderEncoding,
        sat: &mut SAT,
        clause: &mut Vec<Lit>,
        idx: usize,
        constant: CheckedInt,
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
            let threshold = (-constant).div_ceil(a);
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
        let mut min_possible = constant;
        let mut max_possible = constant;
        for i in idx..info.len() {
            min_possible += info.domain_min(i) * info.coef(i);
            max_possible += info.domain_max(i) * info.coef(i);
        }
        if min_possible >= 0 {
            return;
        }
        if max_possible < 0 {
            sat.add_clause(clause.clone());
            return;
        }

        let domain_size = info.domain_size(idx);
        for j in 0..domain_size {
            let new_constant = constant + info.domain(idx, j) * info.coef(idx);
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

// Return Some(clause) where `clause` encodes `lit` (the truth value of `clause` is equal to that of `lit`),
// or None when `lit` always holds.
fn encode_simple_linear_direct_encoding(env: &mut EncoderEnv, lit: &LinearLit) -> Option<Vec<Lit>> {
    let op = lit.op;
    let terms = lit.sum.terms();
    assert_eq!(terms.len(), 1);
    let (var, coef) = terms[0];

    let encoding = env.map.int_map[var].as_ref().unwrap().as_direct_encoding();
    let mut oks = vec![];
    for i in 0..encoding.domain.len() {
        let lhs = encoding.domain[i] * coef + lit.sum.constant;
        let isok = match op {
            CmpOp::Eq => lhs == 0,
            CmpOp::Ne => lhs != 0,
            CmpOp::Le => lhs <= 0,
            CmpOp::Lt => lhs < 0,
            CmpOp::Ge => lhs >= 0,
            CmpOp::Gt => lhs > 0,
        };
        if isok {
            oks.push(encoding.lits[i]);
        }
    }

    if oks.len() == encoding.domain.len() {
        None
    } else {
        Some(oks)
    }
}
