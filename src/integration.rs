use crate::arithmetic::CheckedInt;

use super::config::Config;
use super::csp::{
    Assignment, BoolExpr, BoolVar, BoolVarStatus, Domain, IntExpr, IntVar, IntVarStatus, Stmt, CSP,
};
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
    already_used: bool,
    config: Config,
}

impl IntegratedSolver {
    pub fn new() -> IntegratedSolver {
        IntegratedSolver {
            csp: CSP::new(),
            normalize_map: NormalizeMap::new(),
            norm: NormCSP::new(),
            encode_map: EncodeMap::new(),
            sat: SAT::new(),
            already_used: false,
            config: Config::default(),
        }
    }

    pub fn set_config(&mut self, config: Config) {
        self.config = config;
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
        let is_first = !self.already_used;
        self.already_used = true;

        if self.config.use_constant_folding {
            self.csp
                .optimize(is_first && self.config.use_constant_propagation);
        }

        if self.csp.is_inconsistent() {
            return None;
        }

        normalize(&mut self.csp, &mut self.norm, &mut self.normalize_map);

        if is_first && self.config.use_norm_domain_refinement {
            self.norm.refine_domain();
        }
        if self.norm.is_inconsistent() {
            return None;
        }

        encode(
            &mut self.norm,
            &mut self.sat,
            &mut self.encode_map,
            &self.config,
        );

        match self.sat.solve() {
            Some(model) => Some(Model {
                csp: &self.csp,
                normalize_map: &self.normalize_map,
                norm_csp: &self.norm,
                encode_map: &self.encode_map,
                model,
            }),
            None => None,
        }
    }

    /// Enumerate all the valid assignments of the CSP problem.
    /// Since this function may modify the problem instance, this consumes `self` to avoid further operations.
    pub fn enumerate_valid_assignments(mut self) -> Vec<Assignment> {
        let mut bool_vars = vec![];
        for v in self.csp.vars.bool_vars_iter() {
            bool_vars.push(v);
        }
        let mut int_vars = vec![];
        for v in self.csp.vars.int_vars_iter() {
            int_vars.push(v);
        }

        let mut ret = vec![];
        loop {
            let refutation_expr;

            match self.solve() {
                Some(model) => {
                    let mut refutation = vec![];
                    let mut assignment = Assignment::new();
                    for &var in &bool_vars {
                        let val = model.get_bool(var);
                        assignment.set_bool(var, val);
                        if val {
                            refutation.push(Box::new(!var.expr()));
                        } else {
                            refutation.push(Box::new(var.expr()));
                        }
                    }
                    for &var in &int_vars {
                        let val = model.get_int(var);
                        assignment.set_int(var, val);
                        refutation.push(Box::new(var.expr().ne(IntExpr::Const(val))));
                    }
                    refutation_expr = BoolExpr::Or(refutation);
                    ret.push(assignment);
                }
                None => break,
            }

            self.add_expr(refutation_expr);
        }
        ret
    }

    pub fn decide_irrefutable_facts(
        mut self,
        bool_vars: &[BoolVar],
        int_vars: &[IntVar],
    ) -> Option<Assignment> {
        let mut assignment = Assignment::new();
        match self.solve() {
            Some(model) => {
                for &var in bool_vars {
                    assignment.set_bool(var, model.get_bool(var));
                }
                for &var in int_vars {
                    assignment.set_int(var, model.get_int(var));
                }
            }
            None => return None,
        }
        loop {
            let mut refutation = vec![];
            for (&v, &b) in assignment.bool_iter() {
                refutation.push(Box::new(if b { !v.expr() } else { v.expr() }));
            }
            for (&v, &i) in assignment.int_iter() {
                refutation.push(Box::new(v.expr().ne(IntExpr::Const(i))));
            }
            self.add_expr(BoolExpr::Or(refutation));

            match self.solve() {
                Some(model) => {
                    let bool_erased = assignment
                        .bool_iter()
                        .filter_map(|(&v, &b)| {
                            if model.get_bool(v) != b {
                                Some(v)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    let int_erased = assignment
                        .int_iter()
                        .filter_map(|(&v, &i)| if model.get_int(v) != i { Some(v) } else { None })
                        .collect::<Vec<_>>();

                    bool_erased
                        .iter()
                        .for_each(|&v| assert!(assignment.remove_bool(v).is_some()));
                    int_erased
                        .iter()
                        .for_each(|&v| assert!(assignment.remove_int(v).is_some()));
                }
                None => break,
            }
        }

        Some(assignment)
    }
}

pub struct Model<'a> {
    csp: &'a CSP,
    normalize_map: &'a NormalizeMap,
    norm_csp: &'a NormCSP,
    encode_map: &'a EncodeMap,
    model: SATModel<'a>,
}

impl<'a> Model<'a> {
    pub fn get_bool(&self, var: BoolVar) -> bool {
        match self.normalize_map.get_bool_var(var) {
            Some(norm_var) => {
                self.encode_map
                    .get_bool_var(norm_var)
                    .map(|sat_lit| self.model.assignment(sat_lit.var()) ^ sat_lit.is_negated())
                    .unwrap_or(false) // unused variable optimization
            }
            None => {
                let var_data = self.csp.get_bool_var_status(var);
                match var_data {
                    BoolVarStatus::Infeasible => panic!(),
                    BoolVarStatus::Fixed(v) => v,
                    BoolVarStatus::Unfixed => false, // unused variable optimization
                }
            }
        }
    }

    pub fn get_int(&self, var: IntVar) -> i32 {
        self.get_int_checked(var).get()
    }

    fn get_int_checked(&self, var: IntVar) -> CheckedInt {
        match self.normalize_map.get_int_var(var) {
            Some(norm_var) => {
                self.encode_map
                    .get_int_value_checked(&self.model, norm_var)
                    .unwrap_or(self.norm_csp.vars.int_var(norm_var).lower_bound_checked())
                // unused variable optimization
            }
            None => {
                let var_data = self.csp.get_int_var_status(var);
                match var_data {
                    IntVarStatus::Infeasible => panic!(),
                    IntVarStatus::Fixed(v) => v,
                    IntVarStatus::Unfixed(v) => v,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::csp;
    use super::*;
    use crate::util;

    struct IntegrationTester {
        original_constr: Vec<Stmt>,
        solver: IntegratedSolver,
        bool_vars: Vec<BoolVar>,
        int_vars: Vec<(IntVar, Domain)>,
    }

    impl IntegrationTester {
        fn new() -> IntegrationTester {
            IntegrationTester {
                original_constr: vec![],
                solver: IntegratedSolver::new(),
                bool_vars: vec![],
                int_vars: vec![],
            }
        }

        fn new_bool_var(&mut self) -> BoolVar {
            let ret = self.solver.new_bool_var();
            self.bool_vars.push(ret);
            ret
        }

        fn new_int_var(&mut self, domain: Domain) -> IntVar {
            let ret = self.solver.new_int_var(domain.clone());
            self.int_vars.push((ret, domain));
            ret
        }

        fn add_expr(&mut self, expr: BoolExpr) {
            self.add_constraint(Stmt::Expr(expr));
        }

        fn add_constraint(&mut self, stmt: Stmt) {
            self.original_constr.push(stmt.clone());
            self.solver.add_constraint(stmt);
        }

        fn check(self) {
            let mut bool_domains = vec![];
            for _ in &self.bool_vars {
                bool_domains.push(vec![false, true]);
            }
            let mut int_domains = vec![];
            for (_, domain) in &self.int_vars {
                int_domains.push(domain.enumerate());
            }

            let mut n_assignment_expected = 0;
            for (vb, vi) in util::product_binary(
                &util::product_multi(&bool_domains),
                &util::product_multi(&int_domains),
            ) {
                let mut assignment = csp::Assignment::new();
                for i in 0..self.bool_vars.len() {
                    assignment.set_bool(self.bool_vars[i], vb[i]);
                }
                for i in 0..self.int_vars.len() {
                    assignment.set_int(self.int_vars[i].0, vi[i].get());
                }
                let is_sat_csp = self.is_satisfied_csp(&assignment);
                if is_sat_csp {
                    n_assignment_expected += 1;
                }
            }

            let n_assignment = self.solver.enumerate_valid_assignments().len();
            assert_eq!(n_assignment, n_assignment_expected);
        }

        fn is_satisfied_csp(&self, assignment: &csp::Assignment) -> bool {
            for stmt in &self.original_constr {
                match stmt {
                    Stmt::Expr(e) => {
                        if !assignment.eval_bool_expr(e) {
                            return false;
                        }
                    }
                    Stmt::AllDifferent(_) => todo!(),
                    Stmt::ActiveVerticesConnected(_, _) => todo!(),
                }
            }
            true
        }
    }

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

    #[test]
    fn test_integration_simple_linear1() {
        let mut solver = IntegratedSolver::new();

        let a = solver.new_int_var(Domain::range(0, 2));
        let b = solver.new_int_var(Domain::range(0, 2));
        solver.add_expr((a.expr() + b.expr()).ge(IntExpr::Const(3)));
        solver.add_expr(a.expr().gt(b.expr()));

        let model = solver.solve();
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.get_int(a), 2);
        assert_eq!(model.get_int(b), 1);
    }

    #[test]
    fn test_integration_simple_linear2() {
        let mut solver = IntegratedSolver::new();

        let a = solver.new_int_var(Domain::range(1, 4));
        let b = solver.new_int_var(Domain::range(1, 4));
        let c = solver.new_int_var(Domain::range(1, 4));
        solver.add_expr((a.expr() + b.expr() + c.expr()).ge(IntExpr::Const(9)));
        solver.add_expr(a.expr().gt(b.expr()));
        solver.add_expr(b.expr().gt(c.expr()));

        let model = solver.solve();
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.get_int(a), 4);
        assert_eq!(model.get_int(b), 3);
        assert_eq!(model.get_int(c), 2);
    }

    #[test]
    fn test_integration_simple_linear3() {
        let mut solver = IntegratedSolver::new();

        let a = solver.new_int_var(Domain::range(3, 4));
        let b = solver.new_int_var(Domain::range(1, 2));
        let c = solver.new_int_var(Domain::range(1, 2));
        solver.add_expr(a.expr().ne(b.expr() + c.expr()));
        solver.add_expr(b.expr().gt(c.expr()));

        let model = solver.solve();
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.get_int(a), 4);
        assert_eq!(model.get_int(b), 2);
        assert_eq!(model.get_int(c), 1);
    }

    #[test]
    fn test_integration_simple_linear4() {
        let mut solver = IntegratedSolver::new();

        let a = solver.new_int_var(Domain::range(1, 2));
        let b = solver.new_int_var(Domain::range(1, 2));
        let c = solver.new_int_var(Domain::range(1, 2));
        solver.add_expr(a.expr().ne(b.expr()));
        solver.add_expr(b.expr().ne(c.expr()));
        solver.add_expr(c.expr().ne(a.expr()));

        let model = solver.solve();
        assert!(model.is_none());
    }

    #[test]
    fn test_integration_simple_linear5() {
        let mut solver = IntegratedSolver::new();

        let a = solver.new_int_var(Domain::range(1, 2));
        let b = solver.new_int_var(Domain::range(1, 2));
        let c = solver.new_int_var(Domain::range(1, 2));
        solver.add_expr(a.expr().ne(b.expr()));
        solver.add_expr(b.expr().ne(c.expr()));
        solver.add_expr(c.expr().ne(a.expr()) | (a.expr() + c.expr()).eq(b.expr()));

        let model = solver.solve();
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.get_int(a), 1);
        assert_eq!(model.get_int(b), 2);
        assert_eq!(model.get_int(c), 1);
    }

    #[test]
    fn test_integration_unused_bool() {
        let mut solver = IntegratedSolver::new();

        let x = solver.new_bool_var();
        let y = solver.new_bool_var();
        let z = solver.new_bool_var();
        solver.add_expr(y.expr() | z.expr());

        let model = solver.solve();
        assert!(model.is_some());
        let model = model.unwrap();
        let _ = model.get_bool(x);
        let _ = model.get_bool(y);
        let _ = model.get_bool(z);
    }

    #[test]
    fn test_integration_unused_int() {
        let mut solver = IntegratedSolver::new();

        let a = solver.new_int_var(Domain::range(0, 2));
        let b = solver.new_int_var(Domain::range(0, 2));
        let c = solver.new_int_var(Domain::range(0, 2));
        solver.add_expr(a.expr().gt(b.expr()));

        let model = solver.solve();
        assert!(model.is_some());
        let model = model.unwrap();
        let _ = model.get_int(a);
        let _ = model.get_int(b);
        let _ = model.get_int(c);
    }

    #[test]
    fn test_integration_solve_twice() {
        let mut solver = IntegratedSolver::new();

        let x = solver.new_bool_var();
        let y = solver.new_bool_var();
        solver.add_expr((x.expr() ^ BoolExpr::Const(false)) | (y.expr() ^ BoolExpr::Const(true)));

        {
            let model = solver.solve();
            assert!(model.is_some());
        }

        solver.add_expr(x.expr() ^ y.expr());
        {
            let model = solver.solve();
            assert!(model.is_some());
        }
    }

    #[test]
    fn test_integration_csp_optimization1() {
        let mut solver = IntegratedSolver::new();

        let x = solver.new_bool_var();
        let y = solver.new_bool_var();
        solver.add_expr(x.expr() & !y.expr());

        let res = solver.solve();
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res.get_bool(x), true);
        assert_eq!(res.get_bool(y), false);
    }

    #[test]
    fn test_integration_csp_optimization2() {
        let mut solver = IntegratedSolver::new();

        let x = solver.new_bool_var();
        solver.add_expr(x.expr() | x.expr());
        solver.add_expr(!x.expr());

        let res = solver.solve();
        assert!(res.is_none());
    }

    #[test]
    fn test_integration_irrefutable_logic1() {
        let mut solver = IntegratedSolver::new();

        let x = solver.new_bool_var();
        let y = solver.new_bool_var();
        let z = solver.new_bool_var();
        solver.add_expr(x.expr() | y.expr());
        solver.add_expr(y.expr() | z.expr());
        solver.add_expr(!(x.expr() & z.expr()));

        let res = solver.decide_irrefutable_facts(&[x, y, z], &[]);
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res.get_bool(x), None);
        assert_eq!(res.get_bool(y), Some(true));
        assert_eq!(res.get_bool(z), None);
    }

    #[test]
    fn test_integration_irrefutable_complex1() {
        let mut solver = IntegratedSolver::new();

        let x = solver.new_bool_var();
        let a = solver.new_int_var(Domain::range(0, 2));
        let b = solver.new_int_var(Domain::range(0, 2));
        solver.add_expr(x.expr().ite(a.expr(), b.expr()).eq(a.expr()));
        solver.add_expr(a.expr().ne(b.expr()));

        let res = solver.decide_irrefutable_facts(&[x], &[a, b]);
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res.get_bool(x), Some(true));
        assert_eq!(res.get_int(a), None);
        assert_eq!(res.get_int(b), None);
    }

    #[test]
    fn test_integration_irrefutable_complex2() {
        let mut solver = IntegratedSolver::new();

        let x = solver.new_bool_var();
        let a = solver.new_int_var(Domain::range(0, 2));
        let b = solver.new_int_var(Domain::range(0, 2));
        let c = solver.new_int_var(Domain::range(0, 2));
        solver.add_expr(
            x.expr()
                .ite(a.expr(), b.expr())
                .lt(c.expr() - IntExpr::Const(1)),
        );
        solver.add_expr(a.expr().ne(b.expr()));

        let res = solver.decide_irrefutable_facts(&[x], &[a, b, c]);
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res.get_bool(x), None);
        assert_eq!(res.get_int(a), None);
        assert_eq!(res.get_int(b), None);
        assert_eq!(res.get_int(c), Some(2));
    }

    #[test]
    fn test_integration_irrefutable_many_terms() {
        let mut solver = IntegratedSolver::new();

        let mut ivars = vec![];
        for _ in 0..30 {
            ivars.push(solver.new_int_var(Domain::range(0, 1)));
        }
        solver.add_expr(
            IntExpr::Linear(ivars.iter().map(|v| (Box::new(v.expr()), 1)).collect())
                .ge(IntExpr::Const(10)),
        );

        let x = solver.new_bool_var();
        solver.add_expr(
            IntExpr::Linear(ivars.iter().map(|v| (Box::new(v.expr()), 1)).collect())
                .ge(IntExpr::Const(9))
                .iff(x.expr()),
        );

        let res = solver.decide_irrefutable_facts(&[x], &ivars);
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res.get_bool(x), Some(true));
        assert_eq!(res.get_int(ivars[0]), None);
    }

    #[test]
    fn test_integration_exhaustive_bool1() {
        let mut tester = IntegrationTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        let z = tester.new_bool_var();
        let w = tester.new_bool_var();
        tester.add_expr(x.expr().imp(y.expr() ^ z.expr()));
        tester.add_expr(y.expr().imp(z.expr().iff(w.expr())));

        tester.check();
    }

    #[test]
    fn test_integration_exhaustive_linear1() {
        let mut tester = IntegrationTester::new();

        let a = tester.new_int_var(Domain::range(0, 2));
        let b = tester.new_int_var(Domain::range(0, 2));
        let c = tester.new_int_var(Domain::range(0, 2));
        tester.add_expr((a.expr() + b.expr() + c.expr()).ge(IntExpr::Const(3)));

        tester.check();
    }

    #[test]
    fn test_integration_exhaustive_linear2() {
        let mut tester = IntegrationTester::new();

        let a = tester.new_int_var(Domain::range(0, 3));
        let b = tester.new_int_var(Domain::range(0, 3));
        let c = tester.new_int_var(Domain::range(0, 3));
        let d = tester.new_int_var(Domain::range(0, 3));
        tester.add_expr((a.expr() + b.expr() + c.expr()).ge(IntExpr::Const(5)));
        tester.add_expr((b.expr() + c.expr() + d.expr()).le(IntExpr::Const(5)));

        tester.check();
    }

    #[test]
    fn test_integration_exhaustive_linear3() {
        let mut tester = IntegrationTester::new();

        let a = tester.new_int_var(Domain::range(0, 4));
        let b = tester.new_int_var(Domain::range(0, 4));
        let c = tester.new_int_var(Domain::range(0, 4));
        let d = tester.new_int_var(Domain::range(0, 4));
        tester.add_expr((a.expr() * 2 - b.expr() + c.expr() * 3 + d.expr()).ge(IntExpr::Const(10)));
        tester.add_expr(
            (a.expr() + b.expr() * 4 - c.expr() * 2 - d.expr() * 3).le(IntExpr::Const(2)),
        );

        tester.check();
    }

    #[test]
    fn test_integration_exhaustive_complex1() {
        let mut tester = IntegrationTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        let z = tester.new_bool_var();
        let a = tester.new_int_var(Domain::range(0, 3));
        let b = tester.new_int_var(Domain::range(0, 3));
        let c = tester.new_int_var(Domain::range(0, 3));
        tester.add_expr(
            x.expr()
                .ite(a.expr(), b.expr() + c.expr())
                .eq(a.expr() - b.expr()),
        );
        tester.add_expr(a.expr().ge(y.expr().ite(b.expr(), c.expr())) ^ z.expr());

        tester.check();
    }

    #[test]
    fn test_integration_exhaustive_complex2() {
        let mut tester = IntegrationTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        let z = tester.new_bool_var();
        let a = tester.new_int_var(Domain::range(0, 3));
        let b = tester.new_int_var(Domain::range(0, 3));
        let c = tester.new_int_var(Domain::range(0, 3));
        tester.add_expr(
            x.expr()
                .ite(a.expr(), b.expr() + c.expr())
                .eq(a.expr() - b.expr()),
        );
        tester.add_expr(a.expr().ge(y.expr().ite(b.expr(), c.expr())) ^ z.expr());
        tester.add_expr(x.expr());

        tester.check();
    }

    #[test]
    fn test_integration_exhaustive_complex3() {
        let mut tester = IntegrationTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        let z = tester.new_bool_var();
        let a = tester.new_int_var(Domain::range(0, 3));
        let b = tester.new_int_var(Domain::range(0, 3));
        let c = tester.new_int_var(Domain::range(0, 3));
        tester.add_expr(x.expr() | (a.expr().ge(IntExpr::Const(2))));
        tester.add_expr(
            y.expr() | (b.expr().eq(IntExpr::Const(2))) | (c.expr().ne(IntExpr::Const(1))),
        );
        tester.add_expr(
            (z.expr().ite(IntExpr::Const(1), IntExpr::Const(2)) + b.expr()).ge(a.expr() + c.expr()),
        );

        tester.check();
    }

    #[test]
    fn test_integration_exhaustive_enumerative1() {
        let mut tester = IntegrationTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        let z = tester.new_bool_var();
        let a = tester.new_int_var(Domain::range(0, 2));
        let b = tester.new_int_var(Domain::range(0, 3));
        let c = tester.new_int_var(Domain::range(0, 3));
        tester.add_expr(
            x.expr().ite(IntExpr::Const(3), b.expr() + c.expr()).eq(a
                .expr()
                .ne(IntExpr::Const(2))
                .ite(IntExpr::Const(1), IntExpr::Const(3))
                - b.expr()),
        );
        tester.add_expr(
            a.expr()
                .ne(IntExpr::Const(0))
                .ite(IntExpr::Const(2), IntExpr::Const(1))
                .ge(y.expr().ite(b.expr(), c.expr()))
                ^ z.expr(),
        );
        tester.add_expr(x.expr() ^ a.expr().eq(IntExpr::Const(1)));

        tester.check();
    }

    #[test]
    fn test_integration_exhaustive_enumerative2() {
        let mut tester = IntegrationTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        let z = tester.new_bool_var();
        let a = tester.new_int_var(Domain::range(0, 2));
        let b = tester.new_int_var(Domain::range(0, 3));
        let c = tester.new_int_var(Domain::range(0, 3));
        tester.add_expr(x.expr().iff(a.expr().eq(IntExpr::Const(0))));
        tester.add_expr(y.expr().iff(b.expr().ne(IntExpr::Const(1))));
        tester.add_expr(z.expr().iff(c.expr().eq(IntExpr::Const(2))));
        tester.check();
    }

    #[test]
    fn test_integration_exhaustive_binary1() {
        let mut tester = IntegrationTester::new();

        let x = tester.new_bool_var();
        let a = tester.new_int_var(Domain::range(0, 3));
        tester.add_expr(
            x.expr()
                .ite(IntExpr::Const(2), IntExpr::Const(3))
                .ge(a.expr()),
        );

        tester.check();
    }

    #[test]
    fn test_integration_exhaustive_binary2() {
        let mut tester = IntegrationTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        let z = tester.new_bool_var();
        let a = tester.new_int_var(Domain::range(0, 3));
        let b = tester.new_int_var(Domain::range(0, 3));
        let c = tester.new_int_var(Domain::range(0, 3));
        tester.add_expr(
            (x.expr().ite(IntExpr::Const(1), IntExpr::Const(2))
                + y.expr().ite(IntExpr::Const(2), IntExpr::Const(1)))
            .ge(a.expr() + b.expr() * 2 - c.expr()),
        );
        tester
            .add_expr((a.expr() + z.expr().ite(IntExpr::Const(1), IntExpr::Const(0))).le(c.expr()));
        tester.add_expr(x.expr() | z.expr());

        tester.check();
    }

    #[test]
    fn test_integration_exhaustive_binary3() {
        let mut tester = IntegrationTester::new();

        let x = tester.new_bool_var();
        tester.add_expr(
            x.expr()
                .ite(IntExpr::Const(1), IntExpr::Const(0))
                .eq(IntExpr::Const(1)),
        );

        tester.check();
    }
}
