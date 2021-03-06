use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
use std::ops::Index;

use super::config::Config;
use super::norm_csp::{
    BoolLit, BoolVar, Constraint, ExtraConstraint, IntVar, IntVarRepresentation, LinearLit,
    LinearSum, NormCSP, NormCSPVars,
};
use super::sat::{Lit, SATModel, SAT};
use crate::arithmetic::{CheckedInt, CmpOp, Range};
use crate::util::ConvertMap;

struct ClauseSet {
    data: Vec<Lit>,
    indices: Vec<usize>,
}

impl ClauseSet {
    fn new() -> ClauseSet {
        ClauseSet {
            data: vec![],
            indices: vec![0],
        }
    }

    fn len(&self) -> usize {
        self.indices.len() - 1
    }

    fn push(&mut self, clause: &[Lit]) {
        self.indices.push(self.data.len() + clause.len());
        for &l in clause {
            self.data.push(l);
        }
    }

    fn append(&mut self, mut other: ClauseSet) {
        let offset = self.data.len();
        self.data.append(&mut other.data);
        for i in 1..other.indices.len() {
            self.indices.push(other.indices[i] + offset);
        }
    }
}

impl Index<usize> for ClauseSet {
    type Output = [Lit];

    fn index(&self, index: usize) -> &Self::Output {
        let start = self.indices[index];
        let end = self.indices[index + 1];
        &self.data[start..end]
    }
}

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
                        sat.add_clause(&vec![!lits[i], lits[i - 1]]);
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
                    sat.add_clause(&lits);
                    for i in 1..lits.len() {
                        for j in 0..i {
                            sat.add_clause(&vec![!lits[i], !lits[j]]);
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
                // TODO: use direct encoding for more complex cases
                let is_simple = (lit.op == CmpOp::Eq || lit.op == CmpOp::Ne) && lit.sum.len() <= 2;
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
    for (&var, &coef) in linear_lit.sum.iter() {
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
    let mut bool_lits = constr
        .bool_lit
        .into_iter()
        .map(|lit| env.convert_bool_lit(lit))
        .collect::<Vec<_>>();
    if constr.linear_lit.len() == 0 {
        env.sat.add_clause(&bool_lits);
        return;
    }

    let mut simplified_linears: Vec<Vec<LinearLit>> = vec![];
    for linear_lit in constr.linear_lit {
        if is_unsatisfiable_linear(env, &linear_lit) {
            continue;
        }

        match suggest_encoder(env, &linear_lit) {
            EncoderKind::MixedGe => {
                if linear_lit.op == CmpOp::Ne {
                    // `ne` is decomposed to a disjunction of 2 linear literals and handled separately
                    simplified_linears.push(decompose_linear_lit(
                        env,
                        &LinearLit::new(linear_lit.sum.clone() * (-1) + (-1), CmpOp::Ge),
                    ));
                    simplified_linears.push(decompose_linear_lit(
                        env,
                        &LinearLit::new(linear_lit.sum.clone() + (-1), CmpOp::Ge),
                    ));
                } else {
                    let simplified_sums = match linear_lit.op {
                        CmpOp::Eq => {
                            vec![linear_lit.sum.clone(), linear_lit.sum.clone() * -1]
                        }
                        CmpOp::Ne => unreachable!(),
                        CmpOp::Le => vec![linear_lit.sum * -1],
                        CmpOp::Lt => vec![linear_lit.sum * -1 + (-1)],
                        CmpOp::Ge => vec![linear_lit.sum],
                        CmpOp::Gt => vec![linear_lit.sum + (-1)],
                    };
                    let mut decomposed = vec![];
                    for sum in simplified_sums {
                        decomposed.append(&mut decompose_linear_lit(
                            env,
                            &LinearLit::new(sum, CmpOp::Ge),
                        ));
                    }
                    simplified_linears.push(decomposed);
                }
            }
            EncoderKind::DirectSimple => {
                simplified_linears.push(vec![linear_lit]);
            }
            EncoderKind::DirectEqNe => {
                assert!(linear_lit.op == CmpOp::Eq || linear_lit.op == CmpOp::Ne);
                simplified_linears.push(decompose_linear_lit(env, &linear_lit));
            }
        }
    }

    if simplified_linears.len() == 0 {
        env.sat.add_clause(&bool_lits);
        return;
    }

    if simplified_linears.len() == 1 && bool_lits.len() == 0 {
        // native encoding may be applicable
        let linears = simplified_linears.remove(0);
        for linear_lit in linears {
            match suggest_encoder(env, &linear_lit) {
                EncoderKind::MixedGe => {
                    assert_eq!(linear_lit.op, CmpOp::Ge);
                    if is_ge_order_encoding_native_applicable(env, &linear_lit.sum) {
                        encode_linear_ge_order_encoding_native(env, &linear_lit.sum);
                    } else {
                        let encoded = encode_linear_ge_mixed(env, &linear_lit.sum);
                        for i in 0..encoded.len() {
                            env.sat.add_clause(&encoded[i]);
                        }
                    }
                }
                EncoderKind::DirectSimple => {
                    let encoded = encode_simple_linear_direct_encoding(env, &linear_lit);
                    if let Some(encoded) = encoded {
                        env.sat.add_clause(&encoded);
                    }
                }
                EncoderKind::DirectEqNe => {
                    assert!(linear_lit.op == CmpOp::Eq || linear_lit.op == CmpOp::Ne);
                    let encoded = if linear_lit.op == CmpOp::Eq {
                        encode_linear_eq_direct(env, &linear_lit.sum)
                    } else {
                        encode_linear_ne_direct(env, &linear_lit.sum)
                    };
                    for i in 0..encoded.len() {
                        env.sat.add_clause(&encoded[i]);
                    }
                }
            }
        }
        return;
    }

    // Vec<Lit>: a clause
    // ClauseSet: list clauses whose conjunction is equivalent to a linear literal
    // Vec<ClauseSet>: the above for each linear literal
    let mut encoded_lits: Vec<ClauseSet> = vec![];
    for linear_lits in simplified_linears {
        let mut encoded_conjunction: ClauseSet = ClauseSet::new();
        for linear_lit in linear_lits {
            match suggest_encoder(env, &linear_lit) {
                EncoderKind::MixedGe => {
                    let encoded = encode_linear_ge_mixed(env, &linear_lit.sum);
                    encoded_conjunction.append(encoded);
                }
                EncoderKind::DirectSimple => {
                    let encoded = encode_simple_linear_direct_encoding(env, &linear_lit);
                    if let Some(encoded) = encoded {
                        encoded_conjunction.push(&encoded);
                    }
                }
                EncoderKind::DirectEqNe => {
                    assert!(linear_lit.op == CmpOp::Eq || linear_lit.op == CmpOp::Ne);
                    let encoded = if linear_lit.op == CmpOp::Eq {
                        encode_linear_eq_direct(env, &linear_lit.sum)
                    } else {
                        encode_linear_ne_direct(env, &linear_lit.sum)
                    };
                    encoded_conjunction.append(encoded);
                }
            }
        }

        if encoded_conjunction.len() == 0 {
            // This constraint always holds
            return;
        }
        if encoded_conjunction.len() == 1 {
            bool_lits.extend_from_slice(&encoded_conjunction[0]);
            continue;
        }
        encoded_lits.push(encoded_conjunction);
    }

    if encoded_lits.len() == 0 {
        env.sat.add_clause(&bool_lits);
    } else if encoded_lits.len() == 1 {
        // TODO: a channeling literal may be needed if `bool_lits` contains too many literals
        let clauses = encoded_lits.remove(0);
        let mut buffer = vec![];
        for i in 0..clauses.len() {
            buffer.clear();
            buffer.extend_from_slice(&clauses[i]);
            buffer.extend_from_slice(&bool_lits);
            env.sat.add_clause(&buffer);
        }
    } else {
        let mut channeling_lits = vec![];
        if encoded_lits.len() == 2 && bool_lits.len() == 0 {
            let v = env.sat.new_var();
            channeling_lits.push(v.as_lit(false));
            channeling_lits.push(v.as_lit(true));
        } else {
            for _ in 0..encoded_lits.len() {
                let v = env.sat.new_var();
                channeling_lits.push(v.as_lit(true));
                bool_lits.push(v.as_lit(false));
            }
            env.sat.add_clause(&bool_lits);
        }
        for (i, clauses) in encoded_lits.into_iter().enumerate() {
            let channeling_lit = channeling_lits[i];
            let mut buffer = vec![];
            for i in 0..clauses.len() {
                buffer.clear();
                buffer.extend_from_slice(&clauses[i]);
                buffer.push(channeling_lit);
                env.sat.add_clause(&buffer);
            }
        }
    }
}

enum EncoderKind {
    MixedGe,
    DirectSimple,
    DirectEqNe,
}

fn suggest_encoder(env: &EncoderEnv, linear_lit: &LinearLit) -> EncoderKind {
    if linear_lit.sum.len() == 1
        && env.map.int_map[*linear_lit.sum.iter().next().unwrap().0]
            .as_ref()
            .unwrap()
            .is_direct_encoding()
    {
        return EncoderKind::DirectSimple;
    }
    let is_all_direct_encoded = linear_lit
        .sum
        .iter()
        .all(|(&v, _)| env.map.int_map[v].as_ref().unwrap().is_direct_encoding());
    if (linear_lit.op == CmpOp::Eq || linear_lit.op == CmpOp::Ne) && is_all_direct_encoded {
        return EncoderKind::DirectEqNe;
    }
    EncoderKind::MixedGe
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

    /// The literal asserting (x == val), or `None` if `val` is not in the domain.
    fn equals_val(&self, val: CheckedInt) -> Option<Lit> {
        let mut left = 0;
        let mut right = self.domain_size() - 1;

        while left < right {
            let mid = (left + right) / 2;
            if val <= self.domain(mid) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        if self.domain(left) == val {
            Some(self.equals(left))
        } else {
            None
        }
    }
}

enum LinearInfo<'a> {
    Order(LinearInfoForOrderEncoding<'a>),
    Direct(LinearInfoForDirectEncoding<'a>),
}

fn decompose_linear_lit(env: &mut EncoderEnv, lit: &LinearLit) -> Vec<LinearLit> {
    assert!(lit.op == CmpOp::Ge || lit.op == CmpOp::Eq || lit.op == CmpOp::Ne);
    let op_for_aux_lits = if lit.op == CmpOp::Ge {
        CmpOp::Ge
    } else {
        CmpOp::Eq
    };

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
            ret.push(LinearLit::new(aux_sum, op_for_aux_lits));

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
    ret.push(LinearLit::new(sum, lit.op));
    ret
}

fn is_ge_order_encoding_native_applicable(env: &EncoderEnv, sum: &LinearSum) -> bool {
    for (&var, _) in sum.iter() {
        if env.map.int_map[var]
            .as_ref()
            .unwrap()
            .order_encoding
            .is_none()
        {
            return false;
        }
    }
    if sum.len() > env.config.native_linear_encoding_terms {
        return false;
    }
    let mut domain_product = 1usize;
    for (&var, _) in sum.iter() {
        domain_product *= env.map.int_map[var]
            .as_ref()
            .unwrap()
            .as_order_encoding()
            .domain
            .len();
    }
    domain_product >= env.config.native_linear_encoding_domain_product_threshold
}

fn encode_linear_ge_order_encoding_native(env: &mut EncoderEnv, sum: &LinearSum) {
    let mut info = vec![];
    for (&v, &c) in sum.iter() {
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
    assert_eq!(lit.sum.len(), 1);
    let (&var, &coef) = lit.sum.iter().next().unwrap();

    let encoding = env.map.int_map[var].as_ref().unwrap().as_direct_encoding();
    let mut oks = vec![];
    let mut ngs = vec![];
    for i in 0..encoding.domain.len() {
        let lhs = encoding.domain[i] * coef + lit.sum.constant;
        if op.compare(lhs, CheckedInt::new(0)) {
            oks.push(encoding.lits[i]);
        } else {
            ngs.push(!encoding.lits[i]);
        }
    }

    if oks.len() == encoding.domain.len() {
        None
    } else if ngs.len() == 1 {
        Some(ngs)
    } else {
        Some(oks)
    }
}

fn encode_linear_ge_mixed(env: &EncoderEnv, sum: &LinearSum) -> ClauseSet {
    let mut info = vec![];
    for (&var, &coef) in sum.iter() {
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
        clauses_buf: &mut ClauseSet,
    ) {
        if upper_bound < 0 {
            if let Some(min_relax_on_erasure) = min_relax_on_erasure {
                if upper_bound + min_relax_on_erasure < 0 {
                    return;
                }
            }
            clauses_buf.push(&clause);
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
                            clauses_buf.push(&clause);
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

    let mut clauses_buf = ClauseSet::new();
    encode_sub(&info, &mut vec![], 0, upper_bound, None, &mut clauses_buf);

    clauses_buf
}

fn encode_linear_eq_direct_two_terms(
    info: &[LinearInfoForDirectEncoding],
    constant: CheckedInt,
) -> ClauseSet {
    assert_eq!(info.len(), 2);

    let mut ret = ClauseSet::new();

    for u in 0..2 {
        let v = u ^ 1;

        for i in 0..info[u].domain_size() {
            let mut clause = vec![!info[u].equals(i)];
            clause.extend(info[v].equals_val(-constant - info[u].domain(i)));
            ret.push(&clause);
        }
    }

    ret
}

fn encode_linear_eq_direct(env: &EncoderEnv, sum: &LinearSum) -> ClauseSet {
    let mut info = vec![];
    for (&var, &coef) in sum.iter() {
        let encoding = env.map.int_map[var].as_ref().unwrap();

        let direct_encoding = encoding.as_direct_encoding();
        info.push(LinearInfoForDirectEncoding::new(coef, direct_encoding));
    }
    info.sort_by(|encoding1, encoding2| {
        encoding1
            .encoding
            .lits
            .len()
            .cmp(&encoding2.encoding.lits.len())
    });

    if info.len() == 2 {
        return encode_linear_eq_direct_two_terms(&info, sum.constant);
    }

    fn encode_sub(
        info: &[LinearInfoForDirectEncoding],
        clause: &mut Vec<Lit>,
        idx: usize,
        lower_bound: CheckedInt,
        upper_bound: CheckedInt,
        min_relax_for_lb: Option<CheckedInt>,
        min_relax_for_ub: Option<CheckedInt>,
        clauses_buf: &mut ClauseSet,
    ) {
        if lower_bound > 0 || upper_bound < 0 {
            let mut cannot_prune = true;
            if lower_bound > 0
                && min_relax_for_lb
                    .map(|m| lower_bound - m <= 0)
                    .unwrap_or(true)
            {
                cannot_prune = true;
            }
            if upper_bound < 0
                && min_relax_for_ub
                    .map(|m| upper_bound + m >= 0)
                    .unwrap_or(true)
            {
                cannot_prune = true;
            }
            if cannot_prune {
                clauses_buf.push(&clause);
            }
            return;
        }
        if idx == info.len() {
            return;
        }
        if idx == info.len() - 1 {
            let direct_encoding = &info[idx];
            let lb_for_this_term = direct_encoding.domain_min();
            let ub_for_this_term = direct_encoding.domain_max();

            let prev_lb = lower_bound - lb_for_this_term;
            let prev_ub = upper_bound - ub_for_this_term;

            let mut possible_cand = vec![];

            for i in 0..direct_encoding.domain_size() {
                let value = direct_encoding.domain(i);

                if prev_ub + value < 0 || 0 < prev_lb + value {
                    continue;
                }
                possible_cand.push(direct_encoding.equals(i));
            }

            if possible_cand.len() == direct_encoding.domain_size() {
                return;
            }
            let n_possible_cand = possible_cand.len();
            clause.append(&mut possible_cand);
            clauses_buf.push(&clause);
            clause.truncate(clause.len() - n_possible_cand);
            return;
        }

        let direct_encoding = &info[idx];
        let lb_for_this_term = direct_encoding.domain_min();
        let ub_for_this_term = direct_encoding.domain_max();

        for i in 0..direct_encoding.domain_size() {
            let value = direct_encoding.domain(i);
            let next_lb = lower_bound - lb_for_this_term + value;
            let next_ub = upper_bound - ub_for_this_term + value;
            let next_min_relax_for_lb = Some(
                min_relax_for_lb
                    .unwrap_or(CheckedInt::max_value())
                    .min(value - lb_for_this_term),
            );
            let next_min_relax_for_ub = Some(
                min_relax_for_ub
                    .unwrap_or(CheckedInt::max_value())
                    .min(ub_for_this_term - value),
            );
            clause.push(!direct_encoding.equals(i));
            encode_sub(
                info,
                clause,
                idx + 1,
                next_lb,
                next_ub,
                next_min_relax_for_lb,
                next_min_relax_for_ub,
                clauses_buf,
            );
            clause.pop();
        }

        encode_sub(
            info,
            clause,
            idx + 1,
            lower_bound,
            upper_bound,
            min_relax_for_lb,
            min_relax_for_ub,
            clauses_buf,
        );
    }

    let mut lower_bound = sum.constant;
    let mut upper_bound = sum.constant;
    for direct_encoding in &info {
        lower_bound += direct_encoding.domain_min();
        upper_bound += direct_encoding.domain_max();
    }

    let mut clauses_buf = ClauseSet::new();
    encode_sub(
        &info,
        &mut vec![],
        0,
        lower_bound,
        upper_bound,
        None,
        None,
        &mut clauses_buf,
    );

    clauses_buf
}

fn encode_linear_ne_direct(env: &EncoderEnv, sum: &LinearSum) -> ClauseSet {
    let mut info = vec![];
    for (&var, &coef) in sum.iter() {
        let encoding = env.map.int_map[var].as_ref().unwrap();

        let direct_encoding = encoding.as_direct_encoding();
        info.push(LinearInfoForDirectEncoding::new(coef, direct_encoding));
    }

    fn encode_sub(
        info: &[LinearInfoForDirectEncoding],
        clause: &mut Vec<Lit>,
        idx: usize,
        lower_bound: CheckedInt,
        upper_bound: CheckedInt,
        clauses_buf: &mut ClauseSet,
    ) {
        if lower_bound > 0 || upper_bound < 0 {
            return;
        }
        if idx == info.len() {
            assert_eq!(lower_bound, upper_bound);
            if lower_bound == 0 {
                clauses_buf.push(&clause);
            }
            return;
        }
        if idx == info.len() - 1 {
            let direct_encoding = &info[idx];
            let lb_for_this_term = direct_encoding.domain_min();
            let ub_for_this_term = direct_encoding.domain_max();

            assert_eq!(
                lower_bound - lb_for_this_term,
                upper_bound - ub_for_this_term
            );
            let prev_val = lower_bound - lb_for_this_term;

            let mut forbidden = None;
            for i in 0..direct_encoding.domain_size() {
                let value = direct_encoding.domain(i);

                if prev_val + value == 0 {
                    assert!(forbidden.is_none());
                    forbidden = Some(direct_encoding.equals(i));
                }
            }

            if let Some(forbidden) = forbidden {
                clause.push(!forbidden);
                clauses_buf.push(&clause);
                clause.pop();
            }
            return;
        }

        let direct_encoding = &info[idx];
        let lb_for_this_term = direct_encoding.domain_min();
        let ub_for_this_term = direct_encoding.domain_max();

        for i in 0..direct_encoding.domain_size() {
            let value = direct_encoding.domain(i);
            let next_lb = lower_bound - lb_for_this_term + value;
            let next_ub = upper_bound - ub_for_this_term + value;
            clause.push(!direct_encoding.equals(i));
            encode_sub(info, clause, idx + 1, next_lb, next_ub, clauses_buf);
            clause.pop();
        }
    }

    let mut lower_bound = sum.constant;
    let mut upper_bound = sum.constant;
    for direct_encoding in &info {
        lower_bound += direct_encoding.domain_min();
        upper_bound += direct_encoding.domain_max();
    }

    let mut clauses_buf = ClauseSet::new();
    encode_sub(
        &info,
        &mut vec![],
        0,
        lower_bound,
        upper_bound,
        &mut clauses_buf,
    );

    clauses_buf
}

// TODO: add tests for ClauseSet
#[cfg(test)]
mod tests {
    use super::super::{
        config::Config, csp::Domain, norm_csp::IntVarRepresentation, norm_csp::NormCSPVars,
        sat::SAT,
    };
    use super::*;

    struct EncoderTester {
        norm_vars: NormCSPVars,
        sat: SAT,
        map: EncodeMap,
        config: Config,
    }

    impl EncoderTester {
        fn new() -> EncoderTester {
            EncoderTester {
                norm_vars: NormCSPVars::new(),
                sat: SAT::new(),
                map: EncodeMap::new(),
                config: Config::default(),
            }
        }

        fn env(&mut self) -> EncoderEnv {
            EncoderEnv {
                norm_vars: &mut self.norm_vars,
                sat: &mut self.sat,
                map: &mut self.map,
                config: &self.config,
            }
        }

        fn add_clause(&mut self, clause: &[Lit]) {
            self.sat.add_clause(clause);
        }

        fn add_clause_set(&mut self, clause_set: ClauseSet) {
            for i in 0..clause_set.len() {
                self.sat.add_clause(&clause_set[i]);
            }
        }

        fn add_int_var(&mut self, domain: Domain, is_direct_encoding: bool) -> IntVar {
            let v = self
                .norm_vars
                .new_int_var(IntVarRepresentation::Domain(domain));

            if is_direct_encoding {
                self.map
                    .convert_int_var_direct_encoding(&self.norm_vars, &mut self.sat, v);
            } else {
                self.map
                    .convert_int_var_order_encoding(&self.norm_vars, &mut self.sat, v);
            }

            v
        }

        fn enumerate_valid_assignments_by_sat(&mut self) -> Vec<Vec<CheckedInt>> {
            let sat = &mut self.sat;
            let map = &self.map;
            let norm_vars = &self.norm_vars;

            let int_vars = norm_vars.int_vars_iter().collect::<Vec<_>>();
            let sat_vars = sat.all_vars();

            let mut ret = vec![];
            while let Some(model) = sat.solve() {
                let values = int_vars
                    .iter()
                    .map(|&v| map.get_int_value_checked(&model, v).unwrap())
                    .collect::<Vec<_>>();
                ret.push(values);

                let refutation_clause = sat_vars
                    .iter()
                    .map(|&v| v.as_lit(model.assignment(v)))
                    .collect::<Vec<_>>();
                sat.add_clause(&refutation_clause);
            }

            ret
        }

        fn enumerate_valid_assignments_by_literals(
            &self,
            lits: &[LinearLit],
        ) -> Vec<Vec<CheckedInt>> {
            let int_vars = self.norm_vars.int_vars_iter().collect::<Vec<_>>();
            let domains = int_vars
                .iter()
                .map(|&v| match self.norm_vars.int_var(v) {
                    IntVarRepresentation::Domain(domain) => domain.enumerate(),
                    IntVarRepresentation::Binary(_, t, f) => vec![*t, *f],
                })
                .collect::<Vec<_>>();

            let all_assignments = crate::util::product_multi(&domains);
            let valid_assignments = all_assignments
                .into_iter()
                .filter(|assignment| {
                    for lit in lits {
                        let sum = &lit.sum;
                        let mut value = sum.constant;
                        for (&var, &coef) in sum.iter() {
                            let idx = int_vars.iter().position(|&v| v == var).unwrap();
                            value += assignment[idx] * coef;
                        }
                        if !lit.op.compare(value, CheckedInt::new(0)) {
                            return false;
                        }
                    }
                    true
                })
                .collect();
            valid_assignments
        }

        fn run_check(mut self, lits: &[LinearLit]) {
            let mut result_by_literals = self.enumerate_valid_assignments_by_literals(lits);
            result_by_literals.sort();
            let mut result_by_sat = self.enumerate_valid_assignments_by_sat();
            result_by_sat.sort();

            assert_eq!(result_by_literals, result_by_sat);
        }
    }

    fn linear_sum(terms: &[(IntVar, i32)], constant: i32) -> LinearSum {
        let mut ret = LinearSum::constant(CheckedInt::new(constant));
        for &(var, coef) in terms {
            ret.add_coef(var, CheckedInt::new(coef));
        }
        ret
    }

    #[test]
    fn test_encode_simple_linear_direct_encoding() {
        for op in [
            CmpOp::Eq,
            CmpOp::Ne,
            CmpOp::Le,
            CmpOp::Lt,
            CmpOp::Ge,
            CmpOp::Gt,
        ] {
            let mut tester = EncoderTester::new();

            let x = tester.add_int_var(Domain::range(-2, 5), true);
            let lits = [LinearLit::new(linear_sum(&[(x, 1)], 1), op)];
            {
                let clause = encode_simple_linear_direct_encoding(&mut tester.env(), &lits[0]);
                if let Some(clause) = clause {
                    tester.add_clause(&clause);
                }
            }
            tester.run_check(&lits);
        }
    }

    #[test]
    fn test_encode_linear_eq_direct_two_terms() {
        let mut tester = EncoderTester::new();

        let x = tester.add_int_var(Domain::range(0, 5), true);
        let y = tester.add_int_var(Domain::range(2, 6), true);

        let lits = [LinearLit::new(linear_sum(&[(x, 2), (y, -1)], 1), CmpOp::Eq)];
        {
            let clause_set = encode_linear_eq_direct(&tester.env(), &lits[0].sum);
            tester.add_clause_set(clause_set);
        }
        tester.run_check(&lits);
    }

    #[test]
    fn test_encode_linear_eq_direct() {
        let mut tester = EncoderTester::new();

        let x = tester.add_int_var(Domain::range(0, 5), true);
        let y = tester.add_int_var(Domain::range(2, 6), true);
        let z = tester.add_int_var(Domain::range(-1, 4), true);

        let lits = [LinearLit::new(
            linear_sum(&[(x, 1), (y, -1), (z, 2)], -1),
            CmpOp::Eq,
        )];
        {
            let clause_set = encode_linear_eq_direct(&tester.env(), &lits[0].sum);
            tester.add_clause_set(clause_set);
        }
        tester.run_check(&lits);
    }

    #[test]
    fn test_encode_linear_ne_direct() {
        let mut tester = EncoderTester::new();

        let x = tester.add_int_var(Domain::range(0, 5), true);
        let y = tester.add_int_var(Domain::range(2, 6), true);
        let z = tester.add_int_var(Domain::range(-1, 4), true);

        let lits = [LinearLit::new(
            linear_sum(&[(x, 1), (y, -1), (z, 2)], -1),
            CmpOp::Ne,
        )];
        {
            let clause_set = encode_linear_ne_direct(&tester.env(), &lits[0].sum);
            tester.add_clause_set(clause_set);
        }
        tester.run_check(&lits);
    }

    #[test]
    fn test_encode_linear_ge_mixed() {
        for mask in 0..8 {
            let mut tester = EncoderTester::new();

            let x = tester.add_int_var(Domain::range(0, 5), (mask & 4) != 0);
            let y = tester.add_int_var(Domain::range(2, 6), (mask & 2) != 0);
            let z = tester.add_int_var(Domain::range(-1, 4), (mask & 1) != 0);

            let lits = [LinearLit::new(
                linear_sum(&[(x, 3), (y, -4), (z, 2)], -1),
                CmpOp::Ge,
            )];
            {
                let clause_set = encode_linear_ge_mixed(&tester.env(), &lits[0].sum);
                tester.add_clause_set(clause_set);
            }
            tester.run_check(&lits);
        }
    }

    #[test]
    fn test_encode_linear_ge_order_encoding_native() {
        let mut tester = EncoderTester::new();

        let x = tester.add_int_var(Domain::range(0, 5), false);
        let y = tester.add_int_var(Domain::range(2, 6), false);
        let z = tester.add_int_var(Domain::range(-1, 4), false);

        let lits = [LinearLit::new(
            linear_sum(&[(x, 3), (y, -4), (z, 2)], -1),
            CmpOp::Ge,
        )];
        encode_linear_ge_order_encoding_native(&mut tester.env(), &lits[0].sum);

        tester.run_check(&lits);
    }
}
