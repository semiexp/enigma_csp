use super::csp::{BoolExpr, BoolVar, CSPVars, Domain, IntExpr, IntVar, Stmt, CSP};
use super::norm_csp::BoolVar as NBoolVar;
use super::norm_csp::IntVar as NIntVar;
use super::norm_csp::{BoolLit, Constraint, LinearLit, LinearSum, NormCSP};

use super::CmpOp;

pub struct NormalizeMap {
    bool_map: Vec<Option<NBoolVar>>,
    int_map: Vec<Option<NIntVar>>,
}

impl NormalizeMap {
    pub fn new() -> NormalizeMap {
        NormalizeMap {
            bool_map: vec![],
            int_map: vec![],
        }
    }

    fn convert_bool_var(
        &mut self,
        csp_vars: &CSPVars,
        norm: &mut NormCSP,
        var: BoolVar,
    ) -> NBoolVar {
        let id = var.0;

        while self.bool_map.len() <= id {
            self.bool_map.push(None);
        }

        match self.bool_map[id] {
            Some(x) => x,
            None => {
                let ret = norm.new_bool_var();
                self.bool_map[id] = Some(ret);
                ret
            }
        }
    }

    fn convert_int_var(&mut self, csp_vars: &CSPVars, norm: &mut NormCSP, var: IntVar) -> NIntVar {
        let id = var.0;

        while self.int_map.len() <= id {
            self.int_map.push(None);
        }

        match self.int_map[id] {
            Some(x) => x,
            None => {
                let ret = norm.new_int_var(csp_vars.int_var[id].domain.clone());
                self.int_map[id] = Some(ret);
                ret
            }
        }
    }

    pub fn get_bool_var(&self, var: BoolVar) -> Option<NBoolVar> {
        self.bool_map[var.0]
    }

    pub fn get_int_var(&self, var: IntVar) -> Option<NIntVar> {
        self.int_map[var.0]
    }
}

struct NormalizerEnv<'a, 'b, 'c> {
    csp_vars: &'a mut CSPVars,
    norm: &'b mut NormCSP,
    map: &'c mut NormalizeMap,
}

impl<'a, 'b, 'c> NormalizerEnv<'a, 'b, 'c> {
    fn convert_bool_var(&mut self, var: BoolVar) -> NBoolVar {
        self.map.convert_bool_var(self.csp_vars, self.norm, var)
    }

    fn convert_int_var(&mut self, var: IntVar) -> NIntVar {
        self.map.convert_int_var(self.csp_vars, self.norm, var)
    }
}

/// Normalize constraints in `csp`. Existing constraints in `csp` are cleared.
pub fn normalize(csp: &mut CSP, norm: &mut NormCSP, map: &mut NormalizeMap) {
    let mut env = NormalizerEnv {
        csp_vars: &mut csp.vars,
        norm,
        map,
    };

    let mut stmts = vec![];
    std::mem::swap(&mut stmts, &mut csp.constraints);

    for stmt in stmts {
        normalize_stmt(&mut env, stmt);
    }
}

fn normalize_stmt(env: &mut NormalizerEnv, stmt: Stmt) {
    match stmt {
        Stmt::Expr(mut expr) => {
            // TODO: apply Tseitin transformation to avoid the exponential explosion of constraints caused by Iff/Xor
            let constraints = normalize_bool_expr(env, &expr, false);
            for c in constraints {
                env.norm.add_constraint(c);
            }
        }
        Stmt::AllDifferent(exprs) => {
            todo!();
        }
    }
}

/// Apply Tseitin transformation for `expr` to avoid the exponential explosion of constraints caused by Iff/Xor.
fn tseitin_transformation_bool(
    env: &mut NormalizerEnv,
    extra: &mut Vec<BoolExpr>,
    expr: &mut BoolExpr,
    transform: bool,
) {
    match expr {
        BoolExpr::Const(_) | BoolExpr::Var(_) | BoolExpr::NVar(_) => (),
        BoolExpr::And(es) | BoolExpr::Or(es) => {
            for e in es {
                tseitin_transformation_bool(env, extra, e, transform);
            }
        }
        BoolExpr::Xor(e1, e2) | BoolExpr::Iff(e1, e2) => {
            if transform {
                let v1 = env.norm.new_bool_var();
                let v2 = env.norm.new_bool_var();

                let mut f1 = BoolExpr::NVar(v1);
                std::mem::swap(e1.as_mut(), &mut f1);
                let mut f2 = BoolExpr::NVar(v2);
                std::mem::swap(e2.as_mut(), &mut f2);

                tseitin_transformation_bool(env, extra, &mut f1, true);
                extra.push(BoolExpr::Iff(Box::new(f1), Box::new(BoolExpr::NVar(v1))));

                tseitin_transformation_bool(env, extra, &mut f2, true);
                extra.push(BoolExpr::Iff(Box::new(f2), Box::new(BoolExpr::NVar(v2))));
            } else {
                tseitin_transformation_bool(env, extra, e1, true);
                tseitin_transformation_bool(env, extra, e2, true);
            }
        }
        BoolExpr::Not(e) => tseitin_transformation_bool(env, extra, e, transform),
        BoolExpr::Imp(e1, e2) => {
            tseitin_transformation_bool(env, extra, e1, transform);
            tseitin_transformation_bool(env, extra, e2, transform);
        }
        BoolExpr::Cmp(_, e1, e2) => todo!(),
    }
}

/// Normalize `expr` into a set of `Constraint`s. If `neg` is `true`, not(`expr`) is normalized instead.
fn normalize_bool_expr(env: &mut NormalizerEnv, expr: &BoolExpr, neg: bool) -> Vec<Constraint> {
    match (expr, neg) {
        (&BoolExpr::Const(b), neg) => {
            if b ^ neg {
                vec![]
            } else {
                vec![Constraint::new()]
            }
        }
        (&BoolExpr::Var(v), neg) => {
            let nv = env.convert_bool_var(v);
            let mut constraint = Constraint::new();
            constraint.add_bool(BoolLit::new(nv, neg));
            vec![constraint]
        }
        (&BoolExpr::NVar(v), neg) => {
            let mut constraint = Constraint::new();
            constraint.add_bool(BoolLit::new(v, neg));
            vec![constraint]
        }
        (BoolExpr::And(es), false) | (BoolExpr::Or(es), true) => normalize_conjunction(
            es.into_iter()
                .map(|e| normalize_bool_expr(env, e, neg))
                .collect(),
        ),
        (BoolExpr::And(es), true) | (BoolExpr::Or(es), false) => {
            let constrs = es
                .into_iter()
                .map(|e| normalize_bool_expr(env, e, neg))
                .collect();
            normalize_disjunction(env, constrs)
        }
        (BoolExpr::Not(e), neg) => normalize_bool_expr(env, e, !neg),
        (BoolExpr::Xor(e1, e2), false) | (BoolExpr::Iff(e1, e2), true) => {
            let sub1 = vec![
                normalize_bool_expr(env, e1, false),
                normalize_bool_expr(env, e2, false),
            ];
            let sub2 = vec![
                normalize_bool_expr(env, e1, true),
                normalize_bool_expr(env, e2, true),
            ];
            normalize_conjunction(vec![
                normalize_disjunction(env, sub1),
                normalize_disjunction(env, sub2),
            ])
        }
        (BoolExpr::Xor(e1, e2), true) | (BoolExpr::Iff(e1, e2), false) => {
            let sub1 = vec![
                normalize_bool_expr(env, e1, false),
                normalize_bool_expr(env, e2, true),
            ];
            let sub2 = vec![
                normalize_bool_expr(env, e1, true),
                normalize_bool_expr(env, e2, false),
            ];
            normalize_conjunction(vec![
                normalize_disjunction(env, sub1),
                normalize_disjunction(env, sub2),
            ])
        }
        (BoolExpr::Imp(e1, e2), false) => {
            let constrs = vec![
                normalize_bool_expr(env, e1, true),
                normalize_bool_expr(env, e2, false),
            ];
            normalize_disjunction(env, constrs)
        }
        (BoolExpr::Imp(e1, e2), true) => {
            let constrs = vec![
                normalize_bool_expr(env, e1, false),
                normalize_bool_expr(env, e2, true),
            ];
            normalize_conjunction(constrs)
        }
        (BoolExpr::Cmp(op, e1, e2), _) => {
            let op = if neg {
                match op {
                    CmpOp::Eq => CmpOp::Ne,
                    CmpOp::Ne => CmpOp::Eq,
                    CmpOp::Le => CmpOp::Gt,
                    CmpOp::Lt => CmpOp::Ge,
                    CmpOp::Ge => CmpOp::Lt,
                    CmpOp::Gt => CmpOp::Le,
                }
            } else {
                *op
            };

            let v1 = normalize_int_expr(env, e1);
            let v2 = normalize_int_expr(env, e2);

            let mut constraint = Constraint::new();
            constraint.add_linear(LinearLit::new(v1 - v2, op));
            return vec![constraint];
        }
    }
}

fn normalize_conjunction(constrs: Vec<Vec<Constraint>>) -> Vec<Constraint> {
    let mut ret = vec![];
    for constr in constrs {
        ret.extend(constr);
    }
    ret
}

fn normalize_disjunction(
    env: &mut NormalizerEnv,
    constrs: Vec<Vec<Constraint>>,
) -> Vec<Constraint> {
    let mut constrs = constrs;
    if constrs.len() == 0 {
        vec![]
    } else if constrs.len() == 1 {
        constrs.remove(0)
    } else {
        let mut ret = vec![];
        let mut aux = Constraint::new();

        for mut constr in constrs {
            if constr.len() == 0 {
                continue;
            } else if constr.len() == 1 {
                let c = constr.remove(0);
                aux.bool_lit.extend(c.bool_lit);
                aux.linear_lit.extend(c.linear_lit);
            } else {
                let v = env.norm.new_bool_var();
                aux.add_bool(BoolLit::new(v, false));
                for mut con in constr {
                    con.add_bool(BoolLit::new(v, true));
                    ret.push(con);
                }
            }
        }

        ret.push(aux);
        ret
    }
}

fn normalize_int_expr(env: &mut NormalizerEnv, expr: &IntExpr) -> LinearSum {
    match expr {
        &IntExpr::Const(c) => LinearSum::constant(c),
        &IntExpr::Var(v) => {
            let nv = env.convert_int_var(v);
            LinearSum::singleton(nv)
        }
        &IntExpr::NVar(v) => LinearSum::singleton(v),
        IntExpr::Linear(es) => {
            let mut ret = LinearSum::new();
            for (e, coef) in es {
                ret += normalize_int_expr(env, e) * *coef;
            }
            ret
        }
        IntExpr::If(c, t, f) => {
            let t = normalize_int_expr(env, t);
            let f = normalize_int_expr(env, f);
            let dom = env.norm.get_domain_linear_sum(&t) | env.norm.get_domain_linear_sum(&f);
            let v = env.norm.new_int_var(dom);

            let mut constr1 = normalize_bool_expr(env, c, false);
            {
                let mut c = Constraint::new();
                c.add_linear(LinearLit::new(t - LinearSum::singleton(v), CmpOp::Eq));
                constr1.push(c);
            }

            let mut constr2 = normalize_bool_expr(env, c, true);
            {
                let mut c = Constraint::new();
                c.add_linear(LinearLit::new(f - LinearSum::singleton(v), CmpOp::Eq));
                constr2.push(c);
            }

            for c in normalize_disjunction(env, vec![constr1, constr2]) {
                env.norm.add_constraint(c);
            }

            LinearSum::singleton(v)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;

    #[derive(Debug)]
    struct Assignment {
        bool_val: BTreeMap<BoolVar, bool>,
        int_val: BTreeMap<IntVar, i32>,
    }

    fn eval_bool_expr(assignment: &Assignment, expr: &BoolExpr) -> bool {
        match expr {
            BoolExpr::Const(b) => *b,
            BoolExpr::Var(v) => *(assignment.bool_val.get(v).unwrap()),
            &BoolExpr::NVar(_) => panic!(),
            BoolExpr::And(es) => {
                for e in es {
                    if !eval_bool_expr(assignment, e) {
                        return false;
                    }
                }
                true
            }
            BoolExpr::Or(es) => {
                for e in es {
                    if eval_bool_expr(assignment, e) {
                        return true;
                    }
                }
                false
            }
            BoolExpr::Not(e) => !eval_bool_expr(assignment, e),
            BoolExpr::Xor(e1, e2) => {
                eval_bool_expr(assignment, e1) ^ eval_bool_expr(assignment, e2)
            }
            BoolExpr::Iff(e1, e2) => {
                eval_bool_expr(assignment, e1) == eval_bool_expr(assignment, e2)
            }
            BoolExpr::Imp(e1, e2) => {
                !eval_bool_expr(assignment, e1) || eval_bool_expr(assignment, e2)
            }
            BoolExpr::Cmp(op, e1, e2) => {
                let v1 = eval_int_expr(assignment, e1);
                let v2 = eval_int_expr(assignment, e2);
                match *op {
                    CmpOp::Eq => v1 == v2,
                    CmpOp::Ne => v1 != v2,
                    CmpOp::Le => v1 <= v2,
                    CmpOp::Lt => v1 < v2,
                    CmpOp::Ge => v1 >= v2,
                    CmpOp::Gt => v1 > v2,
                }
            }
        }
    }

    fn eval_int_expr(assignment: &Assignment, expr: &IntExpr) -> i32 {
        match expr {
            IntExpr::Const(c) => *c,
            IntExpr::Var(v) => *(assignment.int_val.get(v).unwrap()),
            &IntExpr::NVar(_) => panic!(),
            IntExpr::Linear(es) => {
                let mut ret = 0i32;
                for (e, c) in es {
                    ret = ret
                        .checked_add(eval_int_expr(assignment, e).checked_mul(*c).unwrap())
                        .unwrap();
                }
                ret
            }
            IntExpr::If(c, t, f) => eval_int_expr(
                assignment,
                if eval_bool_expr(assignment, c) { t } else { f },
            ),
        }
    }

    struct NAssignment {
        bool_val: BTreeMap<NBoolVar, bool>,
        int_val: BTreeMap<NIntVar, i32>,
    }

    fn eval_constraint(constr: &Constraint, assignment: &NAssignment) -> bool {
        for l in &constr.bool_lit {
            if assignment.bool_val.get(&l.var).unwrap() ^ l.negated {
                return true;
            }
        }
        for l in &constr.linear_lit {
            let sum = &l.sum;
            let mut v = sum.constant;
            for (var, coef) in &sum.term {
                v = v
                    .checked_add(
                        assignment
                            .int_val
                            .get(var)
                            .unwrap()
                            .checked_mul(*coef)
                            .unwrap(),
                    )
                    .unwrap();
            }

            if match l.op {
                CmpOp::Eq => v == 0,
                CmpOp::Ne => v != 0,
                CmpOp::Le => v <= 0,
                CmpOp::Lt => v < 0,
                CmpOp::Ge => v >= 0,
                CmpOp::Gt => v > 0,
            } {
                return true;
            }
        }
        false
    }

    struct NormalizerTester {
        original_constr: Vec<Stmt>,
        csp: CSP,
        norm: NormCSP,
        bool_vars: Vec<BoolVar>,
        int_vars: Vec<(IntVar, Domain)>,
        unfixed_bool_vars: Vec<NBoolVar>,
        unfixed_int_vars: Vec<NIntVar>,
        map: NormalizeMap,
    }

    impl NormalizerTester {
        fn new() -> NormalizerTester {
            NormalizerTester {
                original_constr: vec![],
                csp: CSP::new(),
                norm: NormCSP::new(),
                bool_vars: vec![],
                int_vars: vec![],
                unfixed_bool_vars: vec![],
                unfixed_int_vars: vec![],
                map: NormalizeMap::new(),
            }
        }

        fn new_bool_var(&mut self) -> BoolVar {
            let mut ret = self.csp.new_bool_var();
            self.bool_vars.push(ret);
            ret
        }

        fn new_int_var(&mut self, domain: Domain) -> IntVar {
            let mut ret = self.csp.new_int_var(domain.clone());
            self.int_vars.push((ret, domain));
            ret
        }

        fn add_expr(&mut self, expr: BoolExpr) {
            self.add_constraint(Stmt::Expr(expr));
        }

        fn add_constraint(&mut self, stmt: Stmt) {
            self.original_constr.push(stmt.clone());
            self.csp.add_constraint(stmt);
        }

        fn check(&mut self) {
            normalize(&mut self.csp, &mut self.norm, &mut self.map);

            let mut unfixed_bool_vars = BTreeSet::<NBoolVar>::new();
            for i in 0..self.norm.vars.num_bool_var {
                unfixed_bool_vars.insert(NBoolVar(i));
            }

            for v in &self.bool_vars {
                match self.map.get_bool_var(*v) {
                    Some(v) => {
                        assert!(unfixed_bool_vars.remove(&v));
                    }
                    None => {
                        // TODO: corresponding NBoolVar may be absent once optimizations are introduced
                        panic!();
                    }
                }
            }
            self.unfixed_bool_vars = unfixed_bool_vars.into_iter().collect::<Vec<_>>();

            let mut unfixed_int_vars = BTreeSet::<NIntVar>::new();
            for i in 0..self.norm.vars.int_var.len() {
                unfixed_int_vars.insert(NIntVar(i));
            }

            for (v, _) in &self.int_vars {
                match self.map.get_int_var(*v) {
                    Some(v) => {
                        assert!(unfixed_int_vars.remove(&v));
                    }
                    None => {
                        // TODO: corresponding NBoolVar may be absent once optimizations are introduced
                        panic!();
                    }
                }
            }
            self.unfixed_int_vars = unfixed_int_vars.into_iter().collect::<Vec<_>>();

            let mut assignment = Assignment {
                bool_val: BTreeMap::new(),
                int_val: BTreeMap::new(),
            };
            self.exhaustive_test_bool_var(0, &mut assignment);
        }

        fn exhaustive_test_bool_var(&self, idx: usize, assignment: &mut Assignment) {
            if idx == self.bool_vars.len() {
                self.exhaustive_test_int_var(0, assignment);
                return;
            }
            let var = self.bool_vars[idx];
            assignment.bool_val.insert(var, false);
            self.exhaustive_test_bool_var(idx + 1, assignment);
            assignment.bool_val.insert(var, true);
            self.exhaustive_test_bool_var(idx + 1, assignment);
        }

        fn exhaustive_test_int_var(&self, idx: usize, assignment: &mut Assignment) {
            if idx == self.int_vars.len() {
                let is_sat_csp = self.is_satisfied_csp(assignment);
                println!("{:?}: {}", assignment, is_sat_csp);

                let mut n_assignment = NAssignment {
                    bool_val: BTreeMap::new(),
                    int_val: BTreeMap::new(),
                };
                for (var, v) in &assignment.bool_val {
                    n_assignment
                        .bool_val
                        .insert(self.map.get_bool_var(*var).unwrap(), *v);
                }
                for (var, v) in &assignment.int_val {
                    n_assignment
                        .int_val
                        .insert(self.map.get_int_var(*var).unwrap(), *v);
                }

                let is_sat_norm = self.has_satisfying_assignment_bool(0, &mut n_assignment);

                assert_eq!(is_sat_csp, is_sat_norm, "assignment: {:?}", assignment);
                return;
            }
            let var = self.int_vars[idx].0;
            for i in self.int_vars[idx].1.enumerate() {
                assignment.int_val.insert(var, i);
                self.exhaustive_test_int_var(idx + 1, assignment);
            }
        }

        fn has_satisfying_assignment_bool(&self, idx: usize, assignment: &mut NAssignment) -> bool {
            if idx == self.unfixed_bool_vars.len() {
                return self.has_satisfying_assignment_int(0, assignment);
            }
            let var = self.unfixed_bool_vars[idx];
            assignment.bool_val.insert(var, false);
            if self.has_satisfying_assignment_bool(idx + 1, assignment) {
                return true;
            }
            assignment.bool_val.insert(var, true);
            if self.has_satisfying_assignment_bool(idx + 1, assignment) {
                return true;
            }
            false
        }

        fn has_satisfying_assignment_int(&self, idx: usize, assignment: &mut NAssignment) -> bool {
            if idx == self.unfixed_int_vars.len() {
                return self.is_satisfied_norm(assignment);
            }
            let var = self.unfixed_int_vars[idx];
            let dom = self.norm.vars.int_var[var.0];
            for i in dom.enumerate() {
                assignment.int_val.insert(var, i);
                if self.has_satisfying_assignment_int(idx + 1, assignment) {
                    return true;
                }
            }
            false
        }

        fn is_satisfied_csp(&self, assignment: &Assignment) -> bool {
            for stmt in &self.original_constr {
                match stmt {
                    Stmt::Expr(e) => {
                        if !eval_bool_expr(assignment, e) {
                            return false;
                        }
                    }
                    Stmt::AllDifferent(_) => todo!(),
                }
            }
            true
        }

        fn is_satisfied_norm(&self, assignment: &NAssignment) -> bool {
            for constr in &self.norm.constraints {
                if !eval_constraint(constr, assignment) {
                    return false;
                }
            }
            true
        }
    }

    #[test]
    fn test_normalization_empty() {
        let mut tester = NormalizerTester::new();

        tester.check();
    }

    #[test]
    fn test_normalization_and() {
        let mut tester = NormalizerTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        tester.add_expr(x.expr() & y.expr());

        tester.check();
    }

    #[test]
    fn test_normalization_or() {
        let mut tester = NormalizerTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        tester.add_expr(x.expr() | y.expr());

        tester.check();
    }

    #[test]
    fn test_normalization_not() {
        let mut tester = NormalizerTester::new();

        let x = tester.new_bool_var();
        tester.add_expr(!x.expr());

        tester.check();
    }

    #[test]
    fn test_normalization_imp() {
        let mut tester = NormalizerTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        tester.add_expr(x.expr().imp(y.expr()));

        tester.check();
    }

    #[test]
    fn test_normalization_xor() {
        let mut tester = NormalizerTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        tester.add_expr(x.expr() ^ y.expr());

        tester.check();
    }

    #[test]
    fn test_normalization_iff() {
        let mut tester = NormalizerTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        tester.add_expr(x.expr().iff(y.expr()));

        tester.check();
    }

    #[test]
    fn test_normalization_logic_complex1() {
        let mut tester = NormalizerTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        let z = tester.new_bool_var();
        tester.add_expr(
            (x.expr() & y.expr() & !z.expr()) | (!x.expr() & !y.expr()) | !(y.expr() & z.expr()),
        );

        tester.check();
    }

    #[test]
    fn test_normalization_logic_complex2() {
        let mut tester = NormalizerTester::new();

        let x = tester.new_bool_var();
        let y = tester.new_bool_var();
        let z = tester.new_bool_var();
        let v = tester.new_bool_var();
        let w = tester.new_bool_var();
        tester.add_expr(
            x.expr().iff((y.expr() & w.expr()).imp(z.expr())) ^ (z.expr() | (v.expr() ^ w.expr())),
        );

        tester.check();
    }

    #[test]
    fn test_normalization_numeral() {
        let mut tester = NormalizerTester::new();

        let a = tester.new_int_var(Domain::range(0, 2));
        let b = tester.new_int_var(Domain::range(0, 2));
        tester.add_expr(a.expr().ge(b.expr()));

        tester.check();
    }

    #[test]
    fn test_normalization_linear_1() {
        let mut tester = NormalizerTester::new();

        let a = tester.new_int_var(Domain::range(0, 2));
        let b = tester.new_int_var(Domain::range(0, 2));
        tester.add_expr((a.expr() + b.expr()).ge(a.expr() * 2 - b.expr()));

        tester.check();
    }

    #[test]
    fn test_normalization_linear_2() {
        let mut tester = NormalizerTester::new();

        let a = tester.new_int_var(Domain::range(0, 2));
        let b = tester.new_int_var(Domain::range(0, 2));
        tester.add_expr((a.expr() + b.expr()).ge(IntExpr::Const(3)));

        tester.check();
    }
    #[test]
    fn test_normalization_if() {
        let mut tester = NormalizerTester::new();

        let x = tester.new_bool_var();
        let a = tester.new_int_var(Domain::range(0, 2));
        let b = tester.new_int_var(Domain::range(0, 2));
        let c = tester.new_int_var(Domain::range(0, 2));
        tester.add_expr(
            x.expr()
                .ite(a.expr(), b.expr() + c.expr())
                .le(a.expr() - b.expr()),
        );

        tester.check();
    }
}
