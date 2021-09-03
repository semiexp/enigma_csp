use super::norm_csp::{BoolLit, BoolVar, Constraint, NormCSP, NormCSPVars};
use super::sat::{Lit, SATModel, Var, VarArray, SAT};

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

    fn convert_bool_var(&mut self, norm_vars: &NormCSPVars, sat: &mut SAT, var: BoolVar) -> Lit {
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

    pub fn get_bool_var(&self, var: BoolVar) -> Option<Lit> {
        self.bool_map[var.0]
    }

    pub fn get_int_value<'a, 'b>(&'a self, model: SATModel<'b>) -> i32 {
        todo!();
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
    todo!();
}
