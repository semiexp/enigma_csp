use std::ops::{Drop, Not};

#[repr(C)]
struct Opaque {
    _private: [u8; 0],
}

extern "C" {
    fn Glucose_CreateSolver() -> *mut Opaque;
    fn Glucose_DestroySolver(solver: *mut Opaque);
    fn Glucose_NewVar(solver: *mut Opaque) -> i32;
    fn Glucose_AddClause(solver: *mut Opaque, lits: *const Lit, n_lits: i32) -> i32;
    fn Glucose_Solve(solver: *mut Opaque) -> i32;
    fn Glucose_NumVar(solver: *mut Opaque) -> i32;
    fn Glucose_GetModelValueVar(solver: *mut Opaque, var: i32) -> i32;
    fn Glucose_AddOrderEncodingLinear(
        solver: *mut Opaque,
        n_terms: i32,
        domain_size: *const i32,
        lits: *const Lit,
        domain: *const i32,
        coefs: *const i32,
        constant: i32,
    ) -> i32;
    fn Glucose_AddActiveVerticesConnected(
        solver: *mut Opaque,
        n_vertices: i32,
        lits: *const Lit,
        n_edges: i32,
        edges: *const i32,
    ) -> i32;
}

#[derive(Clone, Copy)]
pub struct Var(i32);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Lit(i32);

impl Lit {
    pub fn new(var: Var, negated: bool) -> Lit {
        Lit(var.0 * 2 + if negated { 1 } else { 0 })
    }

    pub fn var(self) -> Var {
        Var(self.0 / 2)
    }

    pub fn is_negated(self) -> bool {
        self.0 % 2 == 1
    }
}

impl Not for Lit {
    type Output = Lit;

    fn not(self) -> Self::Output {
        Lit(self.0 ^ 1)
    }
}

pub struct Solver {
    ptr: *mut Opaque,
}

const NUM_VAR_MAX: i32 = 0x3fffffff;

impl Solver {
    pub fn new() -> Solver {
        Solver {
            ptr: unsafe { Glucose_CreateSolver() },
        }
    }

    pub fn new_var(&mut self) -> Var {
        let var_id = unsafe { Glucose_NewVar(self.ptr) };
        assert!(0 <= var_id && var_id <= NUM_VAR_MAX);
        Var(var_id)
    }

    pub fn num_var(&self) -> i32 {
        unsafe { Glucose_NumVar(self.ptr) }
    }

    pub fn all_vars(&self) -> Vec<Var> {
        (0..self.num_var()).map(|i| Var(i)).collect()
    }

    pub fn add_clause(&mut self, clause: &[Lit]) -> bool {
        assert!(clause.len() <= i32::max_value() as usize);
        let res = unsafe { Glucose_AddClause(self.ptr, clause.as_ptr(), clause.len() as i32) };
        res != 0
    }

    pub fn add_order_encoding_linear(
        &mut self,
        lits: &[Vec<Lit>],
        domain: &[Vec<i32>],
        coefs: &[i32],
        constant: i32,
    ) -> bool {
        assert!(lits.len() <= i32::max_value() as usize);
        assert_eq!(lits.len(), domain.len());
        assert_eq!(lits.len(), coefs.len());
        let n_terms = lits.len() as i32;
        let domain_size = domain.iter().map(|x| x.len() as i32).collect::<Vec<_>>();
        for i in 0..lits.len() {
            assert!(domain[i].len() <= i32::max_value() as usize);
            assert_eq!(lits[i].len() + 1, domain[i].len());
        }
        let lits_flat = lits.iter().flatten().copied().collect::<Vec<_>>();
        let domain_flat = domain.iter().flatten().copied().collect::<Vec<_>>();
        let res = unsafe {
            Glucose_AddOrderEncodingLinear(
                self.ptr,
                n_terms,
                domain_size.as_ptr(),
                lits_flat.as_ptr(),
                domain_flat.as_ptr(),
                coefs.as_ptr(),
                constant,
            )
        };
        res != 0
    }

    pub fn add_active_vertices_connected(
        &mut self,
        lits: &[Lit],
        edges: &[(usize, usize)],
    ) -> bool {
        assert!(lits.len() <= i32::max_value() as usize);
        assert!(edges.len() <= i32::max_value() as usize);

        let mut edges_flat = vec![];
        for &(u, v) in edges {
            assert!(u < lits.len());
            assert!(v < lits.len());
            edges_flat.push(u as i32);
            edges_flat.push(v as i32);
        }

        let res = unsafe {
            Glucose_AddActiveVerticesConnected(
                self.ptr,
                lits.len() as i32,
                lits.as_ptr(),
                edges.len() as i32,
                edges_flat.as_ptr(),
            )
        };
        res != 0
    }

    pub fn solve<'a>(&'a mut self) -> Option<Model<'a>> {
        let res = unsafe { Glucose_Solve(self.ptr) };
        if res != 0 {
            Some(Model { solver: self })
        } else {
            None
        }
    }
}

impl Drop for Solver {
    fn drop(&mut self) {
        unsafe {
            Glucose_DestroySolver(self.ptr);
        }
    }
}

#[derive(Clone, Copy)]
pub struct Model<'a> {
    solver: &'a Solver,
}

impl<'a> Model<'a> {
    pub fn assignment(&self, var: Var) -> bool {
        assert!(0 <= var.0 && var.0 < self.solver.num_var());
        unsafe { Glucose_GetModelValueVar(self.solver.ptr, var.0) != 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver() {
        let mut solver = Solver::new();
        let x = solver.new_var();
        let y = solver.new_var();

        assert!(solver.add_clause(&[Lit::new(x, false), Lit::new(y, false)]));
        assert!(solver.add_clause(&[Lit::new(x, false), Lit::new(y, true)]));
        assert!(solver.add_clause(&[Lit::new(x, true), Lit::new(y, false)]));
        assert!(solver.add_clause(&[Lit::new(x, true), Lit::new(y, true)]));
        assert!(solver.solve().is_none());
    }

    #[test]
    fn test_solver2() {
        let mut solver = Solver::new();
        let x = solver.new_var();
        let y = solver.new_var();

        assert!(solver.add_clause(&[Lit::new(x, false), Lit::new(y, false)]));
        assert!(solver.add_clause(&[Lit::new(x, false), Lit::new(y, true)]));
        assert!(solver.add_clause(&[Lit::new(x, true), Lit::new(y, true)]));
        {
            match solver.solve() {
                Some(model) => {
                    assert!(model.assignment(x));
                    assert!(!model.assignment(y));
                }
                None => panic!(),
            }
        }
    }
}
