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

struct Encoding {
    order_encoding: Option<OrderEncoding>,
    direct_encoding: Option<DirectEncoding>,
}

impl Encoding {
    fn order_encoding(enc: OrderEncoding) -> Encoding {
        Encoding {
            order_encoding: Some(enc),
            direct_encoding: None,
        }
    }

    fn direct_encoding(enc: DirectEncoding) -> Encoding {
        Encoding {
            order_encoding: None,
            direct_encoding: Some(enc),
        }
    }

    fn as_order_encoding(&self) -> &OrderEncoding {
        self.order_encoding.as_ref().unwrap()
    }

    fn as_direct_encoding(&self) -> &DirectEncoding {
        self.direct_encoding.as_ref().unwrap()
    }

    fn is_direct_encoding(&self) -> bool {
        assert!(self.order_encoding.is_some() || self.direct_encoding.is_some());
        return self.direct_encoding.is_some();
    }

    fn range(&self) -> Range {
        if let Some(order_encoding) = &self.order_encoding {
            order_encoding.range()
        } else if let Some(direct_encoding) = &self.direct_encoding {
            direct_encoding.range()
        } else {
            panic!();
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
                        Some(Encoding::order_encoding(OrderEncoding { domain, lits }));
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
                        Some(Encoding::order_encoding(OrderEncoding { domain, lits }));
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
            match norm_vars.int_var(var) {
                IntVarRepresentation::Domain(domain) => {
                    let domain = domain.enumerate();
                    assert_ne!(domain.len(), 0);
                    let lits = sat.new_vars_as_lits(domain.len());
                    sat.add_clause(lits.clone());
                    for i in 1..lits.len() {
                        for j in 0..i {
                            sat.add_clause(vec![!lits[i], !lits[j]]);
                        }
                    }

                    self.int_map[var] =
                        Some(Encoding::direct_encoding(DirectEncoding { domain, lits }));
                }
                &IntVarRepresentation::Binary(cond, t, f) => {
                    let c = self.convert_bool_var(norm_vars, sat, cond);
                    let domain;
                    let lits;
                    if f <= t {
                        domain = vec![f, t];
                        lits = vec![!c, c];
                    } else {
                        domain = vec![t, f];
                        lits = vec![c, !c];
                    }
                    self.int_map[var] =
                        Some(Encoding::direct_encoding(DirectEncoding { domain, lits }));
                }
            }
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

        if let Some(encoding) = &encoding.order_encoding {
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
        } else if let Some(encoding) = &encoding.direct_encoding {
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
        } else {
            panic!();
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

fn is_unsatisfiable_linear(env: &EncoderEnv, linear_lit: &LinearLit) -> bool {
    let mut range = Range::constant(linear_lit.sum.constant);
    for (var, coef) in linear_lit.sum.terms() {
        let encoding = env.map.int_map[var].as_ref().unwrap();
        let var_range = encoding.range();
        range = range + var_range * coef;
    }
    match linear_lit.op {
        CmpOp::Eq => range.low > 0 || range.high < 0,
        CmpOp::Ne => range.low == 0 && range.high == 0,
        CmpOp::Le => range.low > 0,
        CmpOp::Lt => range.low >= 0,
        CmpOp::Ge => range.high < 0,
        CmpOp::Gt => range.high <= 0,
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
        if is_unsatisfiable_linear(env, &linear_lit) {
            continue;
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
                    encode_linear_ge_order_encoding(env, &linear.sum, bool_lit);
                }
                CmpOp::Eq => {
                    encode_linear_ge_order_encoding(env, &linear.sum, bool_lit);
                    linear.sum *= -1;
                    encode_linear_ge_order_encoding(env, &linear.sum, bool_lit);
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
/// With this struct, all coefficients can be virtually treated as 1.
struct LinearInfoForOrderEncoding<'a> {
    coef: CheckedInt,
    encoding: &'a OrderEncoding,
}

impl<'a> LinearInfoForOrderEncoding<'a> {
    pub fn new(coef: CheckedInt, encoding: &'a OrderEncoding) -> LinearInfoForOrderEncoding<'a> {
        LinearInfoForOrderEncoding { coef, encoding }
    }

    fn domain_size(&self) -> usize {
        self.encoding.domain.len()
    }

    /// j-th smallest domain value after normalizing negative coefficients
    fn domain(&self, j: usize) -> CheckedInt {
        if self.coef > 0 {
            self.encoding.domain[j] * self.coef
        } else {
            self.encoding.domain[self.encoding.domain.len() - 1 - j] * self.coef
        }
    }

    #[allow(unused)]
    fn domain_min(&self) -> CheckedInt {
        self.domain(0)
    }

    fn domain_max(&self) -> CheckedInt {
        self.domain(self.domain_size() - 1)
    }

    /// The literal asserting that (the value) is at least `domain(i, j)`.
    fn at_least(&self, j: usize) -> Lit {
        assert!(0 < j && j < self.encoding.domain.len());
        if self.coef > 0 {
            self.encoding.lits[j - 1]
        } else {
            !self.encoding.lits[self.encoding.domain.len() - 1 - j]
        }
    }

    /// The literal asserting (x >= val) under the assumption that x is in the domain.
    fn at_least_val(&self, val: CheckedInt) -> ExtendedLit {
        let dom_size = self.domain_size();

        if val <= self.domain(0) {
            ExtendedLit::True
        } else if val > self.domain(dom_size - 1) {
            ExtendedLit::False
        } else {
            // compute the largest j such that val <= domain[j]
            let mut left = 0;
            let mut right = dom_size - 1;

            while left < right {
                let mid = (left + right) / 2;
                if val <= self.domain(mid) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }

            ExtendedLit::Lit(self.at_least(left))
        }
    }
}

struct LinearInfoForDirectEncoding<'a> {
    coef: CheckedInt,
    encoding: &'a DirectEncoding,
}

impl<'a> LinearInfoForDirectEncoding<'a> {
    pub fn new(coef: CheckedInt, encoding: &'a DirectEncoding) -> LinearInfoForDirectEncoding<'a> {
        LinearInfoForDirectEncoding { coef, encoding }
    }

    fn domain_size(&self) -> usize {
        self.encoding.domain.len()
    }

    fn domain(&self, j: usize) -> CheckedInt {
        if self.coef > 0 {
            self.encoding.domain[j] * self.coef
        } else {
            self.encoding.domain[self.encoding.domain.len() - 1 - j] * self.coef
        }
    }

    #[allow(unused)]
    fn domain_min(&self) -> CheckedInt {
        self.domain(0)
    }

    fn domain_max(&self) -> CheckedInt {
        self.domain(self.domain_size() - 1)
    }

    // The literal asserting that (the value) equals `domain(j)`.
    fn equals(&self, j: usize) -> Lit {
        if self.coef > 0 {
            self.encoding.lits[j]
        } else {
            self.encoding.lits[self.domain_size() - 1 - j]
        }
    }
}

enum LinearInfo<'a> {
    Order(LinearInfoForOrderEncoding<'a>),
    Direct(LinearInfoForDirectEncoding<'a>),
}

fn decompose_linear_lit(env: &mut EncoderEnv, lit: &LinearLit) -> (LinearLit, Vec<LinearLit>) {
    assert!(lit.op == CmpOp::Ge || lit.op == CmpOp::Eq);

    let mut heap = BinaryHeap::new();
    for (&var, &coef) in &lit.sum.term {
        let encoding = env.map.int_map[var].as_ref().unwrap();
        let dom_size = if let Some(order_encoding) = &encoding.order_encoding {
            order_encoding.domain.len()
        } else if let Some(direct_encoding) = &encoding.direct_encoding {
            direct_encoding.domain.len()
        } else {
            panic!();
        };
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
    let mut support_order_encoding_all = true;
    for (var, _) in sum.terms() {
        if env.map.int_map[var]
            .as_ref()
            .unwrap()
            .order_encoding
            .is_none()
        {
            support_order_encoding_all = false;
        }
    }
    if support_order_encoding_all
        && bool_lit.is_empty()
        && sum.len() <= env.config.native_linear_encoding_terms
    {
        encode_linear_ge_order_encoding_native(env, sum, bool_lit);
    } else {
        // encode_linear_ge_order_encoding_literals(env, sum, bool_lit);
        let clauses = encode_linear_gt_mixed(env, sum);
        for mut clause in clauses {
            let mut c = bool_lit.clone();
            c.append(&mut clause);
            env.sat.add_clause(c);
        }
    }
}

fn encode_linear_ge_order_encoding_native(
    env: &mut EncoderEnv,
    sum: &LinearSum,
    bool_lit: &Vec<Lit>,
) {
    assert!(bool_lit.is_empty()); // TODO

    let mut info = vec![];
    for (v, c) in sum.terms() {
        assert_ne!(c, 0);
        info.push(LinearInfoForOrderEncoding::new(
            c,
            env.map.int_map[v].as_ref().unwrap().as_order_encoding(),
        ));
    }

    let mut lits = vec![];
    let mut domain = vec![];
    let mut coefs = vec![];
    let constant = sum.constant.get();

    for i in 0..info.len() {
        let mut lits_r = vec![];
        let mut domain_r = vec![];
        for j in 0..info[i].domain_size() {
            if j > 0 {
                lits_r.push(info[i].at_least(j));
            }
            domain_r.push(info[i].domain(j).get());
        }
        lits.push(lits_r);
        domain.push(domain_r);
        coefs.push(1);
    }

    env.sat
        .add_order_encoding_linear(lits, domain, coefs, constant);
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

fn encode_linear_gt_mixed(env: &EncoderEnv, sum: &LinearSum) -> Vec<Vec<Lit>> {
    let mut info = vec![];
    for (var, coef) in sum.terms() {
        let encoding = env.map.int_map[var].as_ref().unwrap();

        if let Some(order_encoding) = &encoding.order_encoding {
            // Prefer order encoding
            info.push(LinearInfo::Order(LinearInfoForOrderEncoding::new(
                coef,
                order_encoding,
            )));
        } else if let Some(direct_encoding) = &encoding.direct_encoding {
            info.push(LinearInfo::Direct(LinearInfoForDirectEncoding::new(
                coef,
                direct_encoding,
            )));
        }
    }

    fn encode_sub(
        info: &[LinearInfo],
        clause: &mut Vec<Lit>,
        idx: usize,
        upper_bound: CheckedInt,
        min_relax_on_erasure: Option<CheckedInt>,
        clauses_buf: &mut Vec<Vec<Lit>>,
    ) {
        if upper_bound < 0 {
            if let Some(min_relax_on_erasure) = min_relax_on_erasure {
                if upper_bound + min_relax_on_erasure < 0 {
                    return;
                }
            }
            clauses_buf.push(clause.clone());
            return;
        }
        if idx == info.len() {
            return;
        }

        match &info[idx] {
            LinearInfo::Order(order_encoding) => {
                if idx + 1 == info.len() {
                    match order_encoding.at_least_val(-(upper_bound - order_encoding.domain_max()))
                    {
                        ExtendedLit::True => (),
                        ExtendedLit::False => panic!(),
                        ExtendedLit::Lit(lit) => {
                            clause.push(lit);
                            clauses_buf.push(clause.clone());
                            clause.pop();
                        }
                    }
                    return;
                }
                let ub_for_this_term = order_encoding.domain_max();

                for i in 0..(order_encoding.domain_size() - 1) {
                    // assume (value) <= domain[i]
                    let value = order_encoding.domain(i);
                    let next_ub = upper_bound - ub_for_this_term + value;
                    // let next_min_relax = min_relax_on_erasure.unwrap_or(CheckedInt::max_value()).min(order_encoding.domain(i + 1) - value);
                    clause.push(order_encoding.at_least(i + 1));
                    encode_sub(info, clause, idx + 1, next_ub, None, clauses_buf);
                    clause.pop();
                }

                encode_sub(
                    info,
                    clause,
                    idx + 1,
                    upper_bound,
                    min_relax_on_erasure,
                    clauses_buf,
                );
            }
            LinearInfo::Direct(direct_encoding) => {
                let ub_for_this_term = direct_encoding.domain_max();

                for i in 0..(direct_encoding.domain_size() - 1) {
                    let value = direct_encoding.domain(i);
                    let next_ub = upper_bound - ub_for_this_term + value;
                    let next_min_relax = min_relax_on_erasure
                        .unwrap_or(CheckedInt::max_value())
                        .min(ub_for_this_term - value);
                    clause.push(!direct_encoding.equals(i));
                    encode_sub(
                        info,
                        clause,
                        idx + 1,
                        next_ub,
                        Some(next_min_relax),
                        clauses_buf,
                    );
                    clause.pop();
                }

                encode_sub(
                    info,
                    clause,
                    idx + 1,
                    upper_bound,
                    min_relax_on_erasure,
                    clauses_buf,
                );
            }
        }
    }

    let mut upper_bound = sum.constant;
    for linear in &info {
        upper_bound += match linear {
            LinearInfo::Order(order_encoding) => order_encoding.domain_max(),
            LinearInfo::Direct(direct_encoding) => direct_encoding.domain_max(),
        };
    }

    let mut clauses_buf: Vec<Vec<Lit>> = vec![];
    encode_sub(&info, &mut vec![], 0, upper_bound, None, &mut clauses_buf);

    clauses_buf
}
