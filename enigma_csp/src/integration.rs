use crate::arithmetic::CheckedInt;

use super::config::Config;
use super::csp::{
    Assignment, BoolExpr, BoolVar, BoolVarStatus, IntExpr, IntVar, IntVarStatus, Stmt, CSP,
};
use super::encoder::{encode, EncodeMap};
use super::norm_csp::NormCSP;
use super::normalizer::{normalize, NormalizeMap};
use super::sat::{SATModel, SAT};
use crate::domain::Domain;
use std::cell::Cell;

#[derive(Clone, Debug)]
pub struct PerfStats {
    time_normalize: Cell<f64>,
    time_encode: Cell<f64>,
    time_sat_solver: Cell<f64>,
    decisions: Cell<u64>,
    propagations: Cell<u64>,
    conflicts: Cell<u64>,
}

impl PerfStats {
    pub fn new() -> PerfStats {
        PerfStats {
            time_normalize: Cell::new(0.0f64),
            time_encode: Cell::new(0.0f64),
            time_sat_solver: Cell::new(0.0f64),
            decisions: Cell::new(0u64),
            propagations: Cell::new(0u64),
            conflicts: Cell::new(0u64),
        }
    }

    pub fn time_normalize(&self) -> f64 {
        self.time_normalize.get()
    }

    pub fn time_encode(&self) -> f64 {
        self.time_encode.get()
    }

    pub fn time_sat_solver(&self) -> f64 {
        self.time_sat_solver.get()
    }

    pub fn decisions(&self) -> u64 {
        self.decisions.get()
    }

    pub fn propagations(&self) -> u64 {
        self.propagations.get()
    }

    pub fn conflicts(&self) -> u64 {
        self.conflicts.get()
    }
}

pub struct IntegratedSolver<'a> {
    csp: CSP,
    normalize_map: NormalizeMap,
    norm: NormCSP,
    encode_map: EncodeMap,
    sat: SAT,
    already_used: bool,
    config: Config,
    perf_stats: Option<&'a PerfStats>,
}

impl<'a> IntegratedSolver<'a> {
    pub fn new() -> IntegratedSolver<'a> {
        IntegratedSolver {
            csp: CSP::new(),
            normalize_map: NormalizeMap::new(),
            norm: NormCSP::new(),
            encode_map: EncodeMap::new(),
            sat: SAT::new(),
            already_used: false,
            config: Config::default(),
            perf_stats: None,
        }
    }

    pub fn set_config(&mut self, config: Config) {
        self.config = config;
        self.sat.set_rnd_init_act(self.config.glucose_rnd_init_act);
        self.sat
            .set_dump_analysis_info(self.config.dump_analysis_info);
        if let Some(seed) = self.config.glucose_random_seed {
            self.sat.set_seed(seed);
        }
    }

    pub fn new_bool_var(&mut self) -> BoolVar {
        self.csp.new_bool_var()
    }

    pub fn new_int_var(&mut self, domain: Domain) -> IntVar {
        self.csp.new_int_var(domain)
    }

    pub fn new_int_var_from_list(&mut self, domain_list: Vec<i32>) -> IntVar {
        let domain_list = domain_list
            .into_iter()
            .map(CheckedInt::new)
            .collect::<Vec<_>>();
        self.csp.new_int_var_from_list(domain_list)
    }

    pub fn add_constraint(&mut self, stmt: Stmt) {
        self.csp.add_constraint(stmt)
    }

    pub fn add_expr(&mut self, expr: BoolExpr) {
        self.add_constraint(Stmt::Expr(expr))
    }

    pub fn encode(&mut self) -> bool {
        let is_first = !self.already_used;
        self.already_used = true;

        if self.config.use_constant_folding {
            self.csp.optimize(
                is_first && self.config.use_constant_propagation,
                self.config.verbose,
            );
        }

        if self.csp.is_inconsistent() {
            return false;
        }

        let start = std::time::Instant::now();
        normalize(
            &mut self.csp,
            &mut self.norm,
            &mut self.normalize_map,
            &self.config,
        );
        if let Some(perf_stats) = self.perf_stats {
            perf_stats
                .time_normalize
                .set(perf_stats.time_normalize() + start.elapsed().as_secs_f64());
        }

        if is_first && self.config.use_norm_domain_refinement {
            self.norm.refine_domain();
        }
        if self.norm.is_inconsistent() {
            return false;
        }

        let start = std::time::Instant::now();
        encode(
            &mut self.norm,
            &mut self.sat,
            &mut self.encode_map,
            &self.config,
        );
        if let Some(perf_stats) = self.perf_stats {
            perf_stats
                .time_encode
                .set(perf_stats.time_encode() + start.elapsed().as_secs_f64());
        }
        let solver_stats = self.sat.stats();
        if let Some(perf_stats) = self.perf_stats {
            if let Some(decisions) = solver_stats.decisions {
                perf_stats.decisions.set(decisions);
            }
            if let Some(propagations) = solver_stats.propagations {
                perf_stats.propagations.set(propagations);
            }
            if let Some(conflicts) = solver_stats.conflicts {
                perf_stats.conflicts.set(conflicts);
            }
        }
        true
    }

    pub fn solve<'b>(&'b mut self) -> Option<Model<'b>> {
        if !self.encode() {
            return None;
        }
        let start = std::time::Instant::now();
        let solver_result = if self.sat.solve_without_model() {
            Some(unsafe { self.sat.model() })
        } else {
            None
        };
        if let Some(perf_stats) = self.perf_stats {
            perf_stats
                .time_sat_solver
                .set(perf_stats.time_sat_solver() + start.elapsed().as_secs_f64());
        }
        let solver_stats = self.sat.stats();
        if let Some(perf_stats) = self.perf_stats {
            if let Some(decisions) = solver_stats.decisions {
                perf_stats.decisions.set(decisions);
            }
            if let Some(propagations) = solver_stats.propagations {
                perf_stats.propagations.set(propagations);
            }
            if let Some(conflicts) = solver_stats.conflicts {
                perf_stats.conflicts.set(conflicts);
            }
        }

        match solver_result {
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
    pub fn enumerate_valid_assignments(self) -> Vec<Assignment> {
        let mut bool_vars = vec![];
        for v in self.csp.vars.bool_vars_iter() {
            bool_vars.push(v);
        }
        let mut int_vars = vec![];
        for v in self.csp.vars.int_vars_iter() {
            int_vars.push(v);
        }

        self.answer_iter(&bool_vars, &int_vars).collect()
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

    pub fn answer_iter(self, bool_vars: &[BoolVar], int_vars: &[IntVar]) -> AnswerIterator<'a> {
        AnswerIterator {
            solver: self,
            key_bool: bool_vars.iter().cloned().collect(),
            key_int: int_vars.iter().cloned().collect(),
        }
    }

    pub fn set_perf_stats<'b: 'a>(&mut self, perf_stats: &'b PerfStats) {
        self.perf_stats = Some(perf_stats);
    }

    pub fn perf_stats(&self) -> Option<PerfStats> {
        self.perf_stats.as_deref().cloned()
    }
}

pub struct AnswerIterator<'a> {
    solver: IntegratedSolver<'a>,
    key_bool: Vec<BoolVar>,
    key_int: Vec<IntVar>,
}

impl<'a> Iterator for AnswerIterator<'a> {
    type Item = Assignment;

    fn next(&mut self) -> Option<Assignment> {
        let model = self.solver.solve();
        if let Some(model) = &model {
            let mut ret = Assignment::new();
            let mut refutation = vec![];
            for &var in &self.key_bool {
                let b = model.get_bool(var);
                ret.set_bool(var, b);
                refutation.push(Box::new(if b { !var.expr() } else { var.expr() }));
            }
            for &var in &self.key_int {
                let n = model.get_int(var);
                ret.set_int(var, n);
                refutation.push(Box::new(var.expr().ne(IntExpr::Const(n))));
            }
            self.solver.add_expr(BoolExpr::Or(refutation));

            Some(ret)
        } else {
            None
        }
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
            Some(norm_lit) => {
                self.encode_map
                    .get_bool_lit(norm_lit)
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
    use crate::test_util;

    enum DomainOrList {
        Domain(Domain),
        DomainList(Vec<CheckedInt>),
    }

    struct IntegrationTester<'a> {
        original_constr: Vec<Stmt>,
        solver: IntegratedSolver<'a>,
        bool_vars: Vec<BoolVar>,
        int_vars: Vec<(IntVar, DomainOrList)>,
    }

    impl<'a> IntegrationTester<'a> {
        fn new() -> IntegrationTester<'a> {
            IntegrationTester {
                original_constr: vec![],
                solver: IntegratedSolver::new(),
                bool_vars: vec![],
                int_vars: vec![],
            }
        }

        fn set_config(&mut self, config: Config) {
            self.solver.set_config(config);
        }

        fn new_bool_var(&mut self) -> BoolVar {
            let ret = self.solver.new_bool_var();
            self.bool_vars.push(ret);
            ret
        }

        fn new_int_var(&mut self, domain: Domain) -> IntVar {
            let ret = self.solver.new_int_var(domain.clone());
            self.int_vars.push((ret, DomainOrList::Domain(domain)));
            ret
        }

        fn new_int_var_from_list(&mut self, domain_list: Vec<i32>) -> IntVar {
            let l = domain_list
                .iter()
                .map(|&x| CheckedInt::new(x))
                .collect::<Vec<_>>();
            let ret = self.solver.new_int_var_from_list(domain_list);
            self.int_vars.push((ret, DomainOrList::DomainList(l)));
            ret
        }

        fn add_expr(&mut self, expr: BoolExpr) {
            self.add_constraint(Stmt::Expr(expr));
        }

        fn add_constraint(&mut self, stmt: Stmt) {
            self.original_constr.push(stmt.clone());
            self.solver.add_constraint(stmt);
        }

        fn check_expect(self, n_assignment_expected: usize) {
            let n_assignment = self.solver.enumerate_valid_assignments().len();
            assert_eq!(n_assignment, n_assignment_expected);
        }

        fn check(self) {
            let mut bool_domains = vec![];
            for _ in &self.bool_vars {
                bool_domains.push(vec![false, true]);
            }
            let mut int_domains = vec![];
            for (_, domain) in &self.int_vars {
                int_domains.push(match domain {
                    DomainOrList::Domain(dom) => dom.enumerate(),
                    DomainOrList::DomainList(list) => list.clone(),
                });
            }

            let mut n_assignment_expected = 0;
            for (vb, vi) in test_util::product_binary(
                &test_util::product_multi(&bool_domains),
                &test_util::product_multi(&int_domains),
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

            self.check_expect(n_assignment_expected);
        }

        fn is_satisfied_csp(&self, assignment: &csp::Assignment) -> bool {
            for stmt in &self.original_constr {
                match stmt {
                    Stmt::Expr(e) => {
                        if !assignment.eval_bool_expr(e) {
                            return false;
                        }
                    }
                    Stmt::AllDifferent(exprs) => {
                        let values = exprs
                            .iter()
                            .map(|e| assignment.eval_int_expr(e))
                            .collect::<Vec<_>>();
                        for i in 0..values.len() {
                            for j in (i + 1)..values.len() {
                                if values[i] == values[j] {
                                    return false;
                                }
                            }
                        }
                    }
                    Stmt::ActiveVerticesConnected(_, _) => todo!(),
                    Stmt::Circuit(values) => {
                        let values = values
                            .iter()
                            .map(|&v| assignment.get_int(v).unwrap())
                            .collect::<Vec<_>>();
                        let n = values.len();
                        if values.iter().any(|&x| x < 0 || x >= n as i32) {
                            return false;
                        }
                        let values = values.iter().map(|&x| x as usize).collect::<Vec<_>>();

                        let mut cyc_size = 0;
                        for i in 0..n {
                            if values[i] != i {
                                cyc_size += 1;
                            }
                        }

                        let mut visited = vec![false; n];
                        for i in 0..n {
                            if values[i] != i {
                                let mut size = 0;
                                let mut p = i;
                                while !visited[p] {
                                    if values[p] == p {
                                        return false;
                                    }
                                    size += 1;
                                    visited[p] = true;
                                    p = values[p];
                                }
                                if p != i {
                                    return false;
                                }
                                if size != cyc_size {
                                    return false;
                                }
                                break;
                            }
                        }
                    }
                    Stmt::ExtensionSupports(vars, supports) => {
                        let values = vars
                            .iter()
                            .map(|&v| assignment.get_int(v).unwrap())
                            .collect::<Vec<_>>();
                        let mut isok = false;
                        for support in supports {
                            let mut flg = true;
                            for i in 0..values.len() {
                                if let Some(n) = support[i] {
                                    if values[i] != n {
                                        flg = false;
                                    }
                                }
                            }
                            if flg {
                                isok = true;
                            }
                        }
                        if !isok {
                            return false;
                        }
                    }
                    Stmt::GraphDivision(sizes, edges, edges_lit) => {
                        let sizes = sizes
                            .iter()
                            .map(|v| v.map(|v| assignment.get_int(v).unwrap()))
                            .collect::<Vec<_>>();
                        let edge_disconnected = edges_lit
                            .iter()
                            .map(|e| assignment.eval_bool_expr(e))
                            .collect::<Vec<_>>();

                        if !test_util::check_graph_division(&sizes, edges, &edge_disconnected) {
                            return false;
                        }
                    }
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
    fn test_integration_simple_alldifferent() {
        let mut solver = IntegratedSolver::new();

        let a = solver.new_int_var(Domain::range(1, 2));
        let b = solver.new_int_var(Domain::range(1, 2));
        let c = solver.new_int_var(Domain::range(1, 2));
        solver.add_constraint(Stmt::AllDifferent(vec![a.expr(), b.expr(), c.expr()]));

        let model = solver.solve();
        assert!(model.is_none());
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
    fn test_integration_unused_then_used_bool() {
        let mut solver = IntegratedSolver::new();

        let x = solver.new_bool_var();
        let y = solver.new_bool_var();
        let z = solver.new_bool_var();
        solver.add_expr(y.expr() | z.expr());

        {
            let model = solver.solve();
            assert!(model.is_some());
            let model = model.unwrap();
            let _ = model.get_bool(x);
            let _ = model.get_bool(y);
            let _ = model.get_bool(z);
        }

        solver.add_expr(!x.expr());
        {
            let model = solver.solve();
            assert!(model.is_some());
            let model = model.unwrap();
            assert_eq!(model.get_bool(x), false);
        }

        solver.add_expr(x.expr() | !(y.expr() | z.expr()));
        let model = solver.solve();
        assert!(model.is_none());
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
    fn test_integration_bool_lit_after_decomposition() {
        let mut config = Config::default();
        config.domain_product_threshold = 1;
        let mut solver = IntegratedSolver::new();
        solver.set_config(config);

        let x = solver.new_bool_var();

        let a = solver.new_int_var(Domain::range(0, 5));
        let b = solver.new_int_var(Domain::range(0, 5));
        let c = solver.new_int_var(Domain::range(0, 5));
        let d = solver.new_int_var(Domain::range(0, 5));
        let e = solver.new_int_var(Domain::range(0, 5));

        solver.add_expr(
            x.expr()
                .imp((a.expr() + b.expr() + c.expr() + d.expr()).le(e.expr())),
        );
        solver.add_expr(x.expr().imp(a.expr().ge(IntExpr::Const(4))));
        solver.add_expr(x.expr().imp(b.expr().ge(IntExpr::Const(4))));
        solver.add_expr((!x.expr()).imp(a.expr().ge(IntExpr::Const(4))));
        solver.add_expr((!x.expr()).imp(b.expr().ge(IntExpr::Const(4))));
        solver.add_expr((!x.expr()).imp(c.expr().ge(IntExpr::Const(4))));
        solver.add_expr((!x.expr()).imp(d.expr().ge(IntExpr::Const(4))));
        solver.add_expr((!x.expr()).imp(e.expr().ge(IntExpr::Const(4))));

        let res = solver.solve();
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res.get_bool(x), false);
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
    fn test_integration_csp_optimization3() {
        let mut solver = IntegratedSolver::new();

        let mut vars = vec![];
        let mut coefs = vec![];
        for _ in 0..15 {
            let v = solver.new_bool_var();
            vars.push(v);
            coefs.push((
                Box::new(v.expr().ite(IntExpr::Const(1), IntExpr::Const(0))),
                1,
            ));
        }
        solver.add_expr(vars[0].expr());
        solver.add_expr(IntExpr::Linear(coefs).eq(IntExpr::Const(0)));

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
    fn test_integration_irrefutable_alldifferent() {
        let mut solver = IntegratedSolver::new();

        let a = solver.new_int_var(Domain::range(1, 3));
        let b = solver.new_int_var(Domain::range(1, 3));
        let c = solver.new_int_var(Domain::range(1, 3));
        let d = solver.new_int_var(Domain::range(1, 4));
        solver.add_constraint(Stmt::AllDifferent(vec![
            a.expr(),
            b.expr(),
            c.expr(),
            d.expr(),
        ]));

        let res = solver.decide_irrefutable_facts(&[], &[a, b, c, d]);
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res.get_int(a), None);
        assert_eq!(res.get_int(b), None);
        assert_eq!(res.get_int(c), None);
        assert_eq!(res.get_int(d), Some(4));
    }

    #[test]
    fn test_integration_solver_iterator() {
        let mut solver = IntegratedSolver::new();

        let a = solver.new_int_var(Domain::range(1, 3));
        let b = solver.new_int_var(Domain::range(1, 3));
        let x = solver.new_bool_var();

        solver
            .add_expr((a.expr() + b.expr()).ge(x.expr().ite(IntExpr::Const(3), IntExpr::Const(4))));

        let mut n_ans = 0;
        for _ in solver.answer_iter(&[x], &[a, b]) {
            n_ans += 1;
        }
        assert_eq!(n_ans, 14);
    }

    #[test]
    fn test_integration_perf_stats() {
        let perf_stats = PerfStats::new();
        let mut solver = IntegratedSolver::new();
        solver.set_perf_stats(&perf_stats);

        let a = solver.new_int_var(Domain::range(0, 5));
        let b = solver.new_int_var(Domain::range(0, 5));
        solver.add_expr((a.expr() + b.expr()).ge(IntExpr::Const(4)));
        solver.add_expr((a.expr() - b.expr()).le(IntExpr::Const(2)));

        let mut propagations_prev = 0;
        let mut n_ans = 0;
        for _ in solver.answer_iter(&[], &[a, b]) {
            assert!(propagations_prev < perf_stats.propagations());
            propagations_prev = perf_stats.propagations();
            n_ans += 1;
        }
        assert_eq!(n_ans, 21);
    }

    #[test]
    fn test_integration_seed() {
        let mut propagations = vec![];
        for i in 1..=10 {
            let perf_stats = PerfStats::new();
            let mut solver = IntegratedSolver::new();
            solver.set_perf_stats(&perf_stats);
            solver.sat.set_rnd_init_act(true);
            solver.sat.set_seed(i as f64 / 10.0);

            let a = solver.new_int_var(Domain::range(0, 5));
            let b = solver.new_int_var(Domain::range(0, 5));
            solver.add_expr((a.expr() + b.expr()).ge(IntExpr::Const(4)));
            solver.add_expr((a.expr() - b.expr()).le(IntExpr::Const(2)));

            let mut propagations_prev = 0;
            let mut n_ans = 0;
            for _ in solver.answer_iter(&[], &[a, b]) {
                assert!(propagations_prev < perf_stats.propagations());
                propagations_prev = perf_stats.propagations();
                n_ans += 1;
            }
            assert_eq!(n_ans, 21);
            propagations.push(perf_stats.propagations());
        }
        assert!(propagations.iter().any(|&p| p != propagations[0]));
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

    #[cfg(feature = "csp-extra-constraints")]
    #[test]
    fn test_integration_exhaustive_mul1() {
        let mut tester = IntegrationTester::new();

        let a = tester.new_int_var(Domain::range(-5, 5));
        let b = tester.new_int_var(Domain::range(-4, 4));
        let c = tester.new_int_var(Domain::range(-4, 4));
        let d = tester.new_int_var(Domain::range(-4, 4));
        tester.add_expr((a.expr() * b.expr()).eq(c.expr() * d.expr() + IntExpr::Const(1)));

        tester.check();
    }

    #[cfg(feature = "csp-extra-constraints")]
    #[test]
    fn test_integration_exhaustive_mul2() {
        let mut tester = IntegrationTester::new();
        let mut config = Config::default();
        config.force_use_log_encoding = true;
        tester.set_config(config);

        let a = tester.new_int_var(Domain::range(1, 100));
        let b = tester.new_int_var(Domain::range(1, 100));
        let c = tester.new_int_var(Domain::range(1, 100));
        tester.add_expr((a.expr() * a.expr() + b.expr() * b.expr()).eq(c.expr() * c.expr()));

        tester.check_expect(104);
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
    fn test_integration_exhaustive_complex4() {
        let mut tester = IntegrationTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        let z = tester.new_bool_var();
        let a = tester.new_int_var(Domain::range(-3, 3));
        let b = tester.new_int_var(Domain::range(0, 3));
        let c = tester.new_int_var(Domain::range(0, 3));
        tester.add_expr(x.expr() | (a.expr().ge(IntExpr::Const(2))));
        tester.add_expr(
            y.expr() | (b.expr().eq(IntExpr::Const(2))) | (c.expr().ne(IntExpr::Const(1))),
        );
        tester.add_expr(
            (z.expr().ite(IntExpr::Const(1), IntExpr::Const(2)) + b.expr())
                .ge(a.expr().abs() + c.expr()),
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
    fn test_integration_exhaustive_enumerative3() {
        let mut tester = IntegrationTester::new();

        let a = tester.new_int_var(Domain::range(0, 1));
        let b = tester.new_int_var(Domain::range(0, 1));
        let c = tester.new_int_var(Domain::range(0, 1));
        tester.add_expr(a.expr().eq(b.expr()) | b.expr().eq(c.expr()));
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

    #[test]
    fn test_integration_exhaustive_binary4() {
        let mut tester = IntegrationTester::new();
        let mut config = Config::default();
        config.direct_encoding_for_binary_vars = true;
        tester.set_config(config);

        let x = tester.new_bool_var();
        tester.add_expr(
            x.expr()
                .ite(IntExpr::Const(1), IntExpr::Const(0))
                .eq(IntExpr::Const(1))
                | x.expr(),
        );

        tester.check();
    }

    #[test]
    fn test_integration_exhaustive_alldifferent() {
        let mut tester = IntegrationTester::new();

        let a = tester.new_int_var(Domain::range(0, 3));
        let b = tester.new_int_var(Domain::range(0, 3));
        let c = tester.new_int_var(Domain::range(1, 4));
        tester.add_constraint(Stmt::AllDifferent(vec![a.expr(), b.expr(), c.expr()]));
        tester.add_expr((a.expr() + b.expr()).ge(c.expr()));

        tester.check();
    }

    #[test]
    fn test_integration_domain_list1() {
        let mut tester = IntegrationTester::new();

        let a = tester.new_int_var_from_list(vec![0, 1, 2, 3, 5]);
        let b = tester.new_int_var_from_list(vec![0, 1, 3, 4, 5]);
        let c = tester.new_int_var_from_list(vec![0, 2, 3, 4, 5]);
        tester.add_constraint(Stmt::AllDifferent(vec![a.expr(), b.expr(), c.expr()]));
        tester.add_expr((a.expr() + b.expr()).ge(c.expr()));

        tester.check();
    }

    #[cfg(feature = "csp-extra-constraints")]
    #[test]
    fn test_integration_exhaustive_circuit1() {
        let mut tester = IntegrationTester::new();

        let a = tester.new_int_var(Domain::range(0, 5));
        let b = tester.new_int_var(Domain::range(-1, 3));
        let c = tester.new_int_var(Domain::range(0, 3));
        let d = tester.new_int_var(Domain::range(1, 3));

        tester.add_constraint(Stmt::Circuit(vec![a, b, c, d]));

        tester.check();
    }

    #[cfg(feature = "csp-extra-constraints")]
    #[test]
    fn test_integration_exhaustive_circuit2() {
        let mut tester = IntegrationTester::new();

        let a = tester.new_int_var_from_list(vec![1, 2, 3, 4]);
        let b = tester.new_int_var_from_list(vec![0, 2, 3, 4]);
        let c = tester.new_int_var_from_list(vec![0, 1, 4]);
        let d = tester.new_int_var_from_list(vec![0, 2, 4]);
        let e = tester.new_int_var_from_list(vec![0, 1, 2, 3]);

        tester.add_constraint(Stmt::Circuit(vec![a, b, c, d, e]));

        tester.check();
    }

    #[cfg(feature = "csp-extra-constraints")]
    #[test]
    fn test_integration_exhaustive_extension_supports1() {
        for use_native in [false, true] {
            let mut config = Config::default();
            config.use_native_extension_supports = use_native;
            let mut tester = IntegrationTester::new();
            tester.set_config(config);

            let a = tester.new_int_var_from_list(vec![0, 2, 3, 4]);
            let b = tester.new_int_var(Domain::range(0, 4));
            let c = tester.new_int_var(Domain::range(0, 4));
            let d = tester.new_int_var(Domain::range(1, 4));

            tester.add_constraint(Stmt::ExtensionSupports(
                vec![a, b, c],
                vec![
                    vec![Some(0), Some(0), Some(1)],
                    vec![Some(0), Some(2), Some(1)],
                    vec![Some(0), Some(2), Some(3)],
                    vec![Some(0), Some(3), Some(4)],
                    vec![Some(1), Some(2), Some(4)],
                    vec![Some(2), Some(1), Some(1)],
                    vec![Some(2), Some(2), Some(2)],
                    vec![Some(3), Some(3), Some(2)],
                    vec![Some(4), Some(4), Some(0)],
                ],
            ));
            tester.add_constraint(Stmt::ExtensionSupports(
                vec![b, c, d],
                vec![
                    vec![Some(0), None, None],
                    vec![None, Some(1), None],
                    vec![Some(2), None, Some(2)],
                    vec![None, Some(3), Some(4)],
                ],
            ));

            tester.check();
        }
    }

    #[test]
    fn test_integration_graph_division1() {
        let mut tester = IntegrationTester::new();

        let mut vars = vec![];
        for _ in 0..12 {
            vars.push(tester.new_bool_var().expr());
        }

        let a = tester.new_int_var(Domain::range(2, 3));
        let b = tester.new_int_var(Domain::range(4, 5));

        tester.add_constraint(Stmt::GraphDivision(
            vec![Some(a), None, None, None, None, None, None, None, Some(b)],
            vec![
                (0, 1),
                (1, 2),
                (3, 4),
                (4, 5),
                (6, 7),
                (7, 8),
                (0, 3),
                (1, 4),
                (2, 5),
                (3, 6),
                (4, 7),
                (5, 8),
            ],
            vars,
        ));

        tester.check();
    }
}
