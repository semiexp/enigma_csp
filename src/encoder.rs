use super::norm_csp::{
    BoolLit, BoolVar, Constraint, IntVar, LinearLit, LinearSum, NormCSP, NormCSPVars,
};
use super::sat::{Lit, SATModel, VarArray, SAT};
use super::CmpOp;

/// Order encoding of an integer variable with domain of `domain`.
/// `vars[i]` is the logical variable representing (the value of this int variable) >= `domain[i+1]`.
pub struct OrderEncoding {
    domain: Vec<i32>,
    vars: VarArray,
}

pub struct EncodeMap {
    bool_map: Vec<Option<Lit>>, // mapped to Lit rather than Var so that further optimization can be done
    int_map: Vec<Option<OrderEncoding>>,
}

impl EncodeMap {
    pub fn new() -> EncodeMap {
        EncodeMap {
            bool_map: vec![],
            int_map: vec![],
        }
    }

    fn convert_bool_var(&mut self, _norm_vars: &NormCSPVars, sat: &mut SAT, var: BoolVar) -> Lit {
        let id = var.0;

        while self.bool_map.len() <= id {
            self.bool_map.push(None);
        }

        match self.bool_map[id] {
            Some(x) => x,
            None => {
                let ret = sat.new_var().as_lit(false);
                self.bool_map[id] = Some(ret);
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
        let id = var.0;

        while self.int_map.len() <= id {
            self.int_map.push(None);
        }

        if self.int_map[id].is_none() {
            let domain = norm_vars.int_var[id].enumerate();
            assert_ne!(domain.len(), 0);
            let vars = sat.new_vars((domain.len() - 1) as i32);
            for i in 1..vars.len() {
                // vars[i] implies vars[i - 1]
                sat.add_clause(vec![vars.at(i).as_lit(true), vars.at(i - 1).as_lit(false)]);
            }

            self.int_map[id] = Some(OrderEncoding { domain, vars });
        }
    }

    pub fn get_bool_var(&self, var: BoolVar) -> Option<Lit> {
        if var.0 < self.bool_map.len() {
            self.bool_map[var.0]
        } else {
            None
        }
    }

    pub fn get_int_value(&self, model: &SATModel, var: IntVar) -> Option<i32> {
        if var.0 >= self.int_map.len() || self.int_map[var.0].is_none() {
            return None;
        }
        let encoding = self.int_map[var.0].as_ref().unwrap();

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
    for i in norm.num_encoded_vars..norm.vars.int_var.len() {
        map.convert_int_var(&mut norm.vars, sat, IntVar(i));
    }
    norm.num_encoded_vars = norm.vars.int_var.len();

    let mut env = EncoderEnv {
        norm_vars: &mut norm.vars,
        sat,
        map,
    };

    let mut constrs = vec![];
    std::mem::swap(&mut constrs, &mut norm.constraints);

    for constr in constrs {
        encode_constraint(&mut env, constr);
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
                encode_linear_ge(env, &linear.sum, bool_lit);
            }
            CmpOp::Eq => {
                encode_linear_ge(env, &linear.sum, bool_lit);
                linear.sum *= -1;
                encode_linear_ge(env, &linear.sum, bool_lit);
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
}

fn encode_linear_ge(env: &mut EncoderEnv, sum: &LinearSum, bool_lit: &Vec<Lit>) {
    // TODO: better ordering of variables
    let mut coef = vec![];
    let mut encoding = vec![];
    for (v, c) in sum.terms() {
        assert_ne!(c, 0);
        coef.push(c);
        encoding.push(env.map.int_map[v.0].as_ref().unwrap());
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
