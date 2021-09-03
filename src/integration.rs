use super::csp::{BoolExpr, BoolVar, Domain, IntVar, Stmt, CSP};
use super::encoder::{encode, EncodeMap};
use super::norm_csp::NormCSP;
use super::normalizer::{normalize, NormalizeMap};
use super::sat::{SATModel, SAT};

pub struct IntegratedSolver {
    csp: CSP,
    normalize_map: NormalizeMap,
    norm: NormCSP,
    encode_map: EncodeMap,
    sat: SAT,
}

impl IntegratedSolver {
    pub fn new() -> IntegratedSolver {
        IntegratedSolver {
            csp: CSP::new(),
            normalize_map: NormalizeMap::new(),
            norm: NormCSP::new(),
            encode_map: EncodeMap::new(),
            sat: SAT::new(),
        }
    }

    pub fn new_bool_var(&mut self) -> BoolVar {
        self.csp.new_bool_var()
    }

    pub fn new_int_var(&mut self, domain: Domain) -> IntVar {
        self.csp.new_int_var(domain)
    }

    pub fn add_constraint(&mut self, stmt: Stmt) {
        self.csp.add_constraint(stmt)
    }

    pub fn add_expr(&mut self, expr: BoolExpr) {
        self.add_constraint(Stmt::Expr(expr))
    }

    pub fn solve<'a>(&'a mut self) -> Option<Model<'a>> {
        normalize(&mut self.csp, &mut self.norm, &mut self.normalize_map);
        encode(&mut self.norm, &mut self.sat, &mut self.encode_map);

        match self.sat.solve() {
            Some(model) => Some(Model {
                normalize_map: &self.normalize_map,
                encode_map: &self.encode_map,
                model,
            }),
            None => None,
        }
    }
}

pub struct Model<'a> {
    normalize_map: &'a NormalizeMap,
    encode_map: &'a EncodeMap,
    model: SATModel<'a>,
}

impl<'a> Model<'a> {
    pub fn get_bool(&self, var: BoolVar) -> bool {
        let norm_var = self.normalize_map.get_bool_var(var).unwrap(); // TODO: remove `unwrap`
        let sat_lit = self.encode_map.get_bool_var(norm_var).unwrap(); // TODO: remove `unwrap`
        self.model.assignment(sat_lit.var()) ^ sat_lit.is_negated()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_simple_logic1() {
        let mut solver = IntegratedSolver::new();

        let x = solver.new_bool_var();
        let y = solver.new_bool_var();
        solver.add_expr(x.expr() | y.expr());
        solver.add_expr(x.expr() | !y.expr());
        solver.add_expr(!x.expr() | !y.expr());

        let model = solver.solve();
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.get_bool(x), true);
        assert_eq!(model.get_bool(y), false);
    }

    #[test]
    fn test_integration_simple_logic2() {
        let mut solver = IntegratedSolver::new();

        let x = solver.new_bool_var();
        let y = solver.new_bool_var();
        solver.add_expr(x.expr() ^ y.expr());
        solver.add_expr(x.expr().iff(y.expr()));

        let model = solver.solve();
        assert!(model.is_none());
    }

    #[test]
    fn test_integration_simple_logic3() {
        let mut solver = IntegratedSolver::new();

        let v = solver.new_bool_var();
        let w = solver.new_bool_var();
        let x = solver.new_bool_var();
        let y = solver.new_bool_var();
        let z = solver.new_bool_var();
        solver.add_expr(v.expr() ^ w.expr());
        solver.add_expr(w.expr() ^ x.expr());
        solver.add_expr(x.expr() ^ y.expr());
        solver.add_expr(y.expr() ^ z.expr());
        solver.add_expr(z.expr() | v.expr());

        let model = solver.solve();
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.get_bool(v), true);
        assert_eq!(model.get_bool(w), false);
        assert_eq!(model.get_bool(x), true);
        assert_eq!(model.get_bool(y), false);
        assert_eq!(model.get_bool(z), true);
    }

    #[test]
    fn test_integration_simple_logic4() {
        let mut solver = IntegratedSolver::new();

        let v = solver.new_bool_var();
        let w = solver.new_bool_var();
        let x = solver.new_bool_var();
        let y = solver.new_bool_var();
        let z = solver.new_bool_var();
        solver.add_expr(v.expr() ^ w.expr());
        solver.add_expr(w.expr() ^ x.expr());
        solver.add_expr(x.expr() ^ y.expr());
        solver.add_expr(y.expr() ^ z.expr());
        solver.add_expr(z.expr() ^ v.expr());

        let model = solver.solve();
        assert!(model.is_none());
    }
}
