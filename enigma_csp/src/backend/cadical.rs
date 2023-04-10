use std::ops::Drop;

use crate::sat::{Lit, Var};

#[repr(C)]
struct Opaque {
    _private: [u8; 0],
}

extern "C" {
    fn CaDiCaL_CreateSolver() -> *mut Opaque;
    fn CaDiCaL_DestroySolver(solver: *mut Opaque);
    fn CaDiCaL_AddClause(solver: *mut Opaque, lits: *const i32, n_lits: i32);
    fn CaDiCaL_Solve(solver: *mut Opaque) -> i32;
    fn CaDiCaL_GetModelValueVar(solver: *mut Opaque, var: i32) -> i32;
    fn CaDiCaL_AddActiveVerticesConnected(
        solver: *mut Opaque,
        n_vertices: i32,
        lits: *const i32,
        n_edges: i32,
        edges: *const i32,
    ) -> i32;
}

pub struct Solver {
    ptr: *mut Opaque,
    num_var: i32,
}

const NUM_VAR_MAX: i32 = 0x3fffffff;

impl Solver {
    pub fn new() -> Solver {
        Solver {
            ptr: unsafe { CaDiCaL_CreateSolver() },
            num_var: 0,
        }
    }

    pub fn new_var(&mut self) -> Var {
        assert!(self.num_var < NUM_VAR_MAX);
        let var_id = self.num_var;
        self.num_var += 1;
        Var(var_id)
    }

    pub fn num_var(&self) -> i32 {
        self.num_var
    }

    pub fn all_vars(&self) -> Vec<Var> {
        (0..self.num_var()).map(|i| Var(i)).collect()
    }

    pub fn add_clause(&mut self, clause: &[Lit]) {
        assert!(clause.len() <= i32::max_value() as usize);
        let clause = unsafe { std::mem::transmute::<_, &[i32]>(clause) };
        for &c in clause {
            assert!(0 <= c && c < 2 * self.num_var);
        }
        unsafe { CaDiCaL_AddClause(self.ptr, clause.as_ptr(), clause.len() as i32) };
    }

    pub fn add_active_vertices_connected(
        &mut self,
        lits: &[Lit],
        edges: &[(usize, usize)],
    ) -> bool {
        assert!(lits.len() <= i32::max_value() as usize);
        assert!(edges.len() <= i32::max_value() as usize);

        let lits = unsafe { std::mem::transmute::<_, &[i32]>(lits) };
        for &l in lits {
            assert!(0 <= l && l < 2 * self.num_var);
        }

        let mut edges_flat = vec![];
        for &(u, v) in edges {
            assert!(u < lits.len());
            assert!(v < lits.len());
            edges_flat.push(u as i32);
            edges_flat.push(v as i32);
        }

        let res = unsafe {
            CaDiCaL_AddActiveVerticesConnected(
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
        if self.solve_without_model() {
            Some(unsafe { self.model() })
        } else {
            None
        }
    }

    pub fn solve_without_model(&mut self) -> bool {
        let res = unsafe { CaDiCaL_Solve(self.ptr) };
        res != 0
    }

    pub(crate) unsafe fn model<'a>(&'a self) -> Model<'a> {
        Model { solver: self }
    }
}

impl Drop for Solver {
    fn drop(&mut self) {
        unsafe {
            CaDiCaL_DestroySolver(self.ptr);
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
        unsafe { CaDiCaL_GetModelValueVar(self.solver.ptr, var.0) != 0 }
    }
}
