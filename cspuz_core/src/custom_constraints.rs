use crate::backend::glucose::{CustomPropagator, SolverManipulator};
use crate::sat::Lit;

/// To make the CSP solver more flexible, we provide a way to define custom constraints (propagators).
/// Currently, custom constraints are supported only in the Glucose backend.
/// We provide two ways to define custom constraints:
/// - `PropagatorGenerator` trait (more flexible but hard to implement)
/// - `SimpleCustomConstraint` trait (less flexible but easy to implement)
///
/// In custom constraints, we abstract a constraint as a predicate on N boolean values, and the constraint requires that the predicate is satisfied.
/// These N boolean values are represented as `BoolExpr`s, which are passed to `Stmt::CustomConstraint` along with the custom constraint itself.
/// If the constraint needs access to the concrete definitions of these values, you can explicitly pass the values to the custom constraint.
///
/// # `PropagatorGenerator` trait
/// You can implement `PropagatorGenerator` trait to define a custom constraint.
/// Its `generate` method takes a `Vec<Lit>` object, which represents the boolean values (inputs of the constraint),
/// and returns a `Box<dyn CustomPropagator>`, which is a propagator object that can be used in the Glucose backend.
///
/// # `SimpleCustomConstraint` trait
/// `SimpleCustomConstraint` provides a simpler way to define custom constraints.
///
/// Before the constraint is used as a propagator in the SAT solver, `initialize_sat` is called to initialize the constraint.
/// It takes the number of input values as an argument.
/// `notify`, `find_inconsistency`, and `undo` will be called only after `initialize_sat` is called.
///
/// When at least one value is decided, the constraint is notified by calling `notify`. `notify` receives the index and the value (true / false) of the decided value.
/// Several values may correspond to the same `Lit` in the SAT solver. In this case, `notify` is called for each value.
///
/// Also, the decision of values can be cancelled. In this case, the constraint is notified by calling `undo`.
/// When `undo` is called, the decision to be cancelled is always the last decision whic has not been undone yet.
/// In other words, suppore there is a stack of decisions. `notify` pushes a new decision to the stack,
/// and `undo` pops the last decision from the stack.
///
/// Sometimes, the constraint is asked to find an inconsistency. If the constraint finds an inconsistency,
/// it should return a list of tuples (index, value) that yields the inconsistency. This list represents the values that cannot be satisfied simultaneously.
/// If no inconsistency is found, `find_inconsistency` should return `None`.
///
/// `find_inconsistency` should not necessarily find an inconsistency even when the constraint can be proven to be inconsistent from the current state.
/// However, if all values are decided and the constraint is inconsistent, the constraint should report the inconsistency.
/// For example, suppose there is a constraint requiring that at most one input value can be true.
/// For this constraint, we can state that the constraint is inconsistent immediately when two values are decided to be true.
/// Even though, it is permissive that the constraint does not find the inconsistency until all values are decided.
///
/// TODO: support int values

pub trait PropagatorGenerator {
    fn generate<'a>(self: Box<Self>, proxy_map: Vec<Lit>) -> Box<dyn CustomPropagator + 'a>
    where
        Self: 'a;
}

pub trait SimpleCustomConstraint {
    fn initialize_sat(&mut self, num_inputs: usize);
    fn notify(&mut self, index: usize, value: bool);
    fn find_inconsistency(&mut self) -> Option<Vec<(usize, bool)>>;
    fn undo(&mut self);

    fn lazy_propagation(&self) -> bool {
        false
    }
}

impl<T: SimpleCustomConstraint> PropagatorGenerator for T {
    fn generate<'a>(self: Box<Self>, proxy_map: Vec<Lit>) -> Box<dyn CustomPropagator + 'a>
    where
        Self: 'a,
    {
        Box::new(CustomConstraintWrapperForGlucose::new(*self, proxy_map))
    }
}

pub(crate) struct CustomConstraintWrapperForGlucose<T: SimpleCustomConstraint> {
    constraint: T,
    inputs: Vec<Lit>,
    all_lits: Vec<(Lit, usize, bool)>,
    reason: Option<Vec<Lit>>,
}

impl<T: SimpleCustomConstraint> CustomConstraintWrapperForGlucose<T> {
    pub(crate) fn new(constraint: T, inputs: Vec<Lit>) -> Self {
        let mut all_lits = vec![];
        for (idx, lit) in inputs.iter().enumerate() {
            all_lits.push((*lit, idx, true));
            all_lits.push((!*lit, idx, false));
        }
        all_lits.sort();

        CustomConstraintWrapperForGlucose {
            constraint,
            inputs,
            all_lits,
            reason: None,
        }
    }
}

unsafe impl<T: SimpleCustomConstraint> CustomPropagator for CustomConstraintWrapperForGlucose<T> {
    fn initialize(&mut self, mut solver: SolverManipulator) -> bool {
        for i in 0..self.all_lits.len() {
            if i == 0 || self.all_lits[i].0 != self.all_lits[i - 1].0 {
                unsafe {
                    solver.add_watch(self.all_lits[i].0);
                }
            }
        }

        for (lit, i, value) in &self.all_lits {
            if let Some(true) = unsafe { solver.value(*lit) } {
                self.constraint.notify(*i, *value);
            }
        }

        self.constraint.find_inconsistency().is_none()
    }

    fn propagate(
        &mut self,
        _solver: SolverManipulator,
        p: Lit,
        num_pending_propations: i32,
    ) -> bool {
        let mut idx = self.all_lits.partition_point(|(lit, _, _)| *lit < p);
        while idx < self.all_lits.len() && self.all_lits[idx].0 == p {
            let (_, i, value) = self.all_lits[idx];
            self.constraint.notify(i, value);
            idx += 1;
        }

        if self.constraint.lazy_propagation() && num_pending_propations > 0 {
            return true;
        }

        if let Some(inconsistency) = self.constraint.find_inconsistency() {
            let mut lits = vec![];
            for &(idx, value) in &inconsistency {
                let lit = if value {
                    self.inputs[idx]
                } else {
                    !self.inputs[idx]
                };
                let cur_value = unsafe { _solver.value(lit) };
                match cur_value {
                    Some(true) => (),
                    Some(false) => panic!(
                        "the value of the input {} in the reason is not {}",
                        idx, value
                    ),
                    None => panic!("the input {} in the reason is not decided", idx),
                }
                lits.push(lit);
            }
            lits.sort();
            lits.dedup();

            // TODO: it is unclear whether we need to keep the original order
            /*
            let mut lits_dedup = vec![];
            let mut lits_set = std::collections::HashSet::new();
            for lit in lits {
                if lits_set.insert(lit.0) {
                    lits_dedup.push(lit);
                }
            }
            lits = lits_dedup;
            */

            let mut has_current_level: bool = false;
            for &lit in &lits {
                if unsafe { _solver.is_current_level(lit) } {
                    has_current_level = true;
                    break;
                }
            }
            assert!(
                has_current_level,
                "the reason must contain a lit in the current level"
            );

            self.reason = Some(lits);
            false
        } else {
            true
        }
    }

    fn calc_reason(
        &mut self,
        _solver: SolverManipulator,
        p: Option<Lit>,
        extra: Option<Lit>,
    ) -> Vec<Lit> {
        assert!(self.reason.is_some());
        assert!(p.is_none());
        let mut reason = self.reason.take().unwrap();
        reason.extend(extra);
        // TODO: check invariant
        reason
    }

    fn undo(&mut self, _solver: SolverManipulator, p: Lit) {
        let mut idx = self.all_lits.partition_point(|(lit, _, _)| *lit < p);
        while idx < self.all_lits.len() && self.all_lits[idx].0 == p {
            self.constraint.undo();
            idx += 1;
        }
        self.reason = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csp_repr::Stmt;
    use crate::integration::IntegratedSolver;

    #[derive(PartialEq, Eq)]
    enum FaultKind {
        UnrelatedLit,
        WrongValue,
        TooLateIncosistencyDetection,
    }

    struct AtMost {
        k: usize,
        decision_stack: Vec<(usize, bool)>,
    }

    impl SimpleCustomConstraint for AtMost {
        fn initialize_sat(&mut self, _num_inputs: usize) {}

        fn notify(&mut self, index: usize, value: bool) {
            self.decision_stack.push((index, value));
        }

        fn find_inconsistency(&mut self) -> Option<Vec<(usize, bool)>> {
            let mut n_true = 0;
            for &(_, value) in &self.decision_stack {
                if value {
                    n_true += 1;
                }
            }

            if n_true > self.k {
                Some(self.decision_stack.clone())
            } else {
                None
            }
        }

        fn undo(&mut self) {
            assert!(!self.decision_stack.is_empty());
            self.decision_stack.pop();
        }
    }

    struct AtMostWithFault {
        k: usize,
        n: usize,
        fault: FaultKind,
        decision_stack: Vec<(usize, bool)>,
    }

    impl SimpleCustomConstraint for AtMostWithFault {
        fn initialize_sat(&mut self, num_inputs: usize) {
            assert_eq!(self.n, num_inputs);
        }

        fn notify(&mut self, index: usize, value: bool) {
            self.decision_stack.push((index, value));
        }

        fn find_inconsistency(&mut self) -> Option<Vec<(usize, bool)>> {
            let mut n_true = 0;
            for &(_, value) in &self.decision_stack {
                if value {
                    n_true += 1;
                }
            }

            if n_true <= self.k {
                return None;
            }

            match self.fault {
                FaultKind::UnrelatedLit => {
                    let mut ret = self.decision_stack.clone();
                    let mut has_zero = false;
                    for &(idx, _) in &self.decision_stack {
                        if idx == 0 {
                            has_zero = true;
                        }
                    }
                    if !has_zero {
                        ret.push((0, true));
                    }
                    Some(ret)
                }
                FaultKind::WrongValue => {
                    let mut ret = self.decision_stack.clone();
                    ret[0].1 = !ret[0].1;
                    Some(ret)
                }
                FaultKind::TooLateIncosistencyDetection => {
                    if self.decision_stack.len() < self.n && n_true <= self.k + 1 {
                        return None;
                    }

                    let mut ret = vec![];
                    let mut n_true = 0;
                    for &(idx, val) in &self.decision_stack {
                        if val {
                            ret.push((idx, val));
                            n_true += 1;
                        }
                        if n_true > self.k {
                            break;
                        }
                    }
                    Some(ret)
                }
            }
        }

        fn undo(&mut self) {
            assert!(!self.decision_stack.is_empty());
            self.decision_stack.pop();
        }
    }

    #[test]
    fn test_custom_constraints_atmost() {
        for n in [2, 6, 10, 20, 30] {
            let mut solver = IntegratedSolver::new();

            let mut vars = vec![];
            let mut vars_expr = vec![];
            for _ in 0..n {
                let var = solver.new_bool_var();
                vars.push(var);
                vars_expr.push(var.expr());
            }

            let at_most = AtMost {
                k: 2,
                decision_stack: vec![],
            };
            solver.add_constraint(Stmt::CustomConstraint(vars_expr, Box::new(at_most)));

            let iter = solver.answer_iter(&vars, &[]);
            assert_eq!(iter.count(), 1 + n + n * (n - 1) / 2);
        }
    }

    #[test]
    fn test_custom_constraints_atmost_same_lit() {
        for n in [2, 6, 10, 20, 30] {
            let mut solver = IntegratedSolver::new();

            let mut vars = vec![];
            let mut vars_expr = vec![];
            for _ in 0..n {
                let var = solver.new_bool_var();
                vars.push(var);
                vars_expr.push(var.expr());
            }
            vars_expr.push(vars_expr[0].clone());

            let at_most = AtMost {
                k: 2,
                decision_stack: vec![],
            };
            solver.add_constraint(Stmt::CustomConstraint(vars_expr, Box::new(at_most)));

            let iter = solver.answer_iter(&vars, &[]);
            assert_eq!(iter.count(), 1 + n + (n - 1) * (n - 2) / 2);
        }
    }

    #[test]
    fn test_custom_constraints_atmost_opposite_lit() {
        for n in [2, 6, 10, 20] {
            let mut solver = IntegratedSolver::new();

            let mut vars = vec![];
            let mut vars_expr = vec![];
            for _ in 0..n {
                let var = solver.new_bool_var();
                vars.push(var);
                vars_expr.push(var.expr());
            }

            let v = solver.new_bool_var();
            vars.push(v);
            vars_expr.push(!v.expr());
            vars_expr.push(v.expr());

            let at_most = AtMost {
                k: 3,
                decision_stack: vec![],
            };
            solver.add_constraint(Stmt::CustomConstraint(vars_expr, Box::new(at_most)));

            let iter = solver.answer_iter(&vars, &[]);
            assert_eq!(iter.count(), 2 * (1 + n + n * (n - 1) / 2));
        }
    }

    #[test]
    fn test_custom_constraints_atmost_random_sign() {
        for n in [2, 6, 10, 20, 30] {
            let mut solver = IntegratedSolver::new();

            let mut vars = vec![];
            let mut vars_expr = vec![];
            for i in 0..n {
                let var = solver.new_bool_var();
                vars.push(var);
                if i % 2 == 0 {
                    vars_expr.push(var.expr());
                } else {
                    vars_expr.push(!var.expr());
                }
            }

            let at_most = AtMost {
                k: 2,
                decision_stack: vec![],
            };
            solver.add_constraint(Stmt::CustomConstraint(vars_expr, Box::new(at_most)));

            let iter = solver.answer_iter(&vars, &[]);
            assert_eq!(iter.count(), 1 + n + n * (n - 1) / 2);
        }
    }

    #[test]
    #[should_panic(expected = "in the reason is not decided")]
    fn test_custom_constraints_atmost_unrelated_lit() {
        let n = 20;
        let mut solver = IntegratedSolver::new();

        let mut vars = vec![];
        let mut vars_expr = vec![];
        for _ in 0..n {
            let var = solver.new_bool_var();
            vars.push(var);
            vars_expr.push(var.expr());
        }

        let at_most = AtMostWithFault {
            k: 2,
            n,
            fault: FaultKind::UnrelatedLit,
            decision_stack: vec![],
        };
        solver.add_constraint(Stmt::CustomConstraint(vars_expr, Box::new(at_most)));

        let _ = solver.enumerate_valid_assignments();
    }

    #[test]
    #[should_panic(expected = "in the reason is not")]
    fn test_custom_constraints_atmost_wrong_value() {
        let n = 20;
        let mut solver = IntegratedSolver::new();

        let mut vars = vec![];
        let mut vars_expr = vec![];
        for _ in 0..n {
            let var = solver.new_bool_var();
            vars.push(var);
            vars_expr.push(var.expr());
        }

        let at_most = AtMostWithFault {
            k: 2,
            n,
            fault: FaultKind::WrongValue,
            decision_stack: vec![],
        };
        solver.add_constraint(Stmt::CustomConstraint(vars_expr, Box::new(at_most)));

        let _ = solver.enumerate_valid_assignments();
    }

    #[test]
    #[should_panic(expected = "the reason must contain a lit in the current level")]
    fn test_custom_constraints_atmost_too_late_inconsistency_detection() {
        let n = 20;
        let mut solver = IntegratedSolver::new();

        let mut vars = vec![];
        let mut vars_expr = vec![];
        for _ in 0..n {
            let var = solver.new_bool_var();
            vars.push(var);
            vars_expr.push(var.expr());
        }

        let at_most = AtMostWithFault {
            k: 2,
            n,
            fault: FaultKind::TooLateIncosistencyDetection,
            decision_stack: vec![],
        };
        solver.add_constraint(Stmt::CustomConstraint(vars_expr, Box::new(at_most)));

        let _ = solver.enumerate_valid_assignments();
    }
}
