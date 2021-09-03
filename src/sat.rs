use std::ops::Not;

use crate::glucose::Lit as GlucoseLit;
use crate::glucose::Model as GlucoseModel;
use crate::glucose::Solver as GlucoseSolver;
use crate::glucose::Var as GlucoseVar;

#[derive(Clone, Copy)]
pub struct Var(GlucoseVar);

impl Var {
    pub fn as_lit(self, negated: bool) -> Lit {
        Lit(GlucoseLit::new(self.0, negated))
    }
}

#[derive(Clone, Copy)]
pub struct Lit(GlucoseLit);

impl Lit {
    pub fn var(self) -> Var {
        Var(self.0.var())
    }

    pub fn is_negated(self) -> bool {
        self.0.is_negated()
    }
}

impl Not for Lit {
    type Output = Lit;

    fn not(self) -> Self::Output {
        Lit(!self.0)
    }
}

pub struct VarArray {
    vars: Vec<Var>,
}

impl VarArray {
    pub fn len(&self) -> i32 {
        self.vars.len() as i32
    }

    pub fn at(&self, idx: i32) -> Var {
        assert!(0 <= idx && idx < self.len());
        self.vars[idx as usize]
    }
}

/// Adapter to SAT solver.
/// To support other SAT solver without changing previous stages, we introduce an adapter instead of
/// using `glucose::Solver` directly from the encoder.
pub struct SAT {
    solver: GlucoseSolver,
}

impl SAT {
    pub fn new() -> SAT {
        SAT {
            solver: GlucoseSolver::new(),
        }
    }

    pub fn new_var(&mut self) -> Var {
        Var(self.solver.new_var())
    }

    pub fn new_vars(&mut self, count: i32) -> VarArray {
        let mut vars = vec![];
        for i in 0..count {
            vars.push(self.new_var());
        }
        VarArray { vars }
    }

    pub fn add_clause(&mut self, clause: Vec<Lit>) {
        let mut c = vec![];
        for l in clause {
            c.push(l.0);
        }
        self.solver.add_clause(&c);
    }

    pub fn solve<'a>(&'a mut self) -> Option<SATModel<'a>> {
        self.solver.solve().map(|model| SATModel { model })
    }
}

pub struct SATModel<'a> {
    model: GlucoseModel<'a>,
}

impl<'a> SATModel<'a> {
    pub fn assignment(&self, var: Var) -> bool {
        self.model.assignment(var.0)
    }
}
