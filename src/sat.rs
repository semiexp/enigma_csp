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

#[repr(C)]
#[derive(Clone, Copy, Debug)]
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

    pub fn new_vars(&mut self, count: usize) -> Vec<Var> {
        let mut vars = vec![];
        for _ in 0..count {
            vars.push(self.new_var());
        }
        vars
    }

    pub fn new_vars_as_lits(&mut self, count: usize) -> Vec<Lit> {
        let vars = self.new_vars(count);
        vars.iter().map(|v| v.as_lit(false)).collect()
    }

    pub fn add_clause(&mut self, clause: &[Lit]) {
        self.solver
            .add_clause(unsafe { std::mem::transmute::<&[Lit], &[GlucoseLit]>(clause) });
    }

    pub fn add_order_encoding_linear(
        &mut self,
        lits: Vec<Vec<Lit>>,
        domain: Vec<Vec<i32>>,
        coefs: Vec<i32>,
        constant: i32,
    ) -> bool {
        let lits = lits
            .iter()
            .map(|x| x.iter().map(|l| l.0).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        self.solver
            .add_order_encoding_linear(&lits, &domain, &coefs, constant)
    }

    pub fn add_active_vertices_connected(
        &mut self,
        lits: Vec<Lit>,
        edges: Vec<(usize, usize)>,
    ) -> bool {
        let lits = lits.iter().map(|x| x.0).collect::<Vec<_>>();
        self.solver.add_active_vertices_connected(&lits, &edges)
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

    pub fn assignment_lit(&self, lit: Lit) -> bool {
        self.model.assignment(lit.var().0) ^ lit.is_negated()
    }
}
