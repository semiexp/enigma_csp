use std::ffi::{c_void, CString};
use std::ops::Drop;
use std::os::raw::c_char;

use crate::sat::{Lit, OrderEncodingLinearMode, Var};

#[repr(C)]
struct Opaque {
    _private: [u8; 0],
}

extern "C" {
    fn Glucose_CreateSolver() -> *mut Opaque;
    fn Glucose_DestroySolver(solver: *mut Opaque);
    fn Glucose_NewVar(solver: *mut Opaque) -> i32;
    fn Glucose_NewNamedVar(solver: *mut Opaque, name: *const c_char) -> i32;
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
    fn Glucose_AddDirectEncodingExtensionSupports(
        solver: *mut Opaque,
        n_vars: i32,
        domain_size: *const i32,
        lits: *const Lit,
        n_supports: i32,
        supports: *const i32,
    ) -> i32;
    fn Glucose_AddGraphDivision(
        solver: *mut Opaque,
        n_vertices: i32,
        dom_sizes: *const i32,
        domains: *const i32,
        dom_lits: *const Lit,
        n_edges: i32,
        edges: *const i32,
        edge_lits: *const Lit,
    ) -> i32;
    fn Glucose_SolverStats_decisions(solver: *mut Opaque) -> u64;
    fn Glucose_SolverStats_propagations(solver: *mut Opaque) -> u64;
    fn Glucose_SolverStats_conflicts(solver: *mut Opaque) -> u64;
    fn Glucose_Set_random_seed(solver: *mut Opaque, random_seed: f64);
    fn Glucose_Set_rnd_init_act(solver: *mut Opaque, rnd_init_act: i32);
    fn Glucose_Set_dump_analysis_info(solver: *mut Opaque, value: i32);
}

pub struct Solver {
    ptr: *mut Opaque,
    custom_constraints: Vec<Box<Box<dyn CustomConstraint>>>,
    order_encoding_linear_mode: OrderEncodingLinearMode,
}

const NUM_VAR_MAX: i32 = 0x3fffffff;

impl Solver {
    pub fn new() -> Solver {
        Solver {
            ptr: unsafe { Glucose_CreateSolver() },
            custom_constraints: vec![],
            order_encoding_linear_mode: OrderEncodingLinearMode::Cpp,
        }
    }

    pub fn new_var(&mut self) -> Var {
        let var_id = unsafe { Glucose_NewVar(self.ptr) };
        assert!(0 <= var_id && var_id <= NUM_VAR_MAX);
        Var(var_id)
    }

    pub fn new_named_var(&mut self, name: &str) -> Var {
        let c_string = CString::new(name).unwrap();
        let var_id = unsafe { Glucose_NewNamedVar(self.ptr, c_string.as_ptr()) };
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

        for i in 0..lits.len() {
            assert!(domain[i].len() <= i32::max_value() as usize);
            assert_eq!(lits[i].len() + 1, domain[i].len());
        }

        match self.order_encoding_linear_mode {
            OrderEncodingLinearMode::Cpp => {
                let n_terms = lits.len() as i32;
                let domain_size = domain.iter().map(|x| x.len() as i32).collect::<Vec<_>>();
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
            OrderEncodingLinearMode::Rust | OrderEncodingLinearMode::RustOptimized => {
                let mut terms = vec![];
                for i in 0..lits.len() {
                    terms.push(LinearTerm {
                        lits: lits[i].clone(),
                        domain: domain[i].clone(),
                        coef: coefs[i],
                    });
                }
                let optimized =
                    self.order_encoding_linear_mode == OrderEncodingLinearMode::RustOptimized;
                self.add_custom_constraint(Box::new(OrderEncodingLinear::new(
                    terms, constant, optimized,
                )))
            }
        }
    }

    pub fn set_order_encoding_linear_mode(&mut self, mode: OrderEncodingLinearMode) {
        self.order_encoding_linear_mode = mode;
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

    pub fn add_direct_encoding_extension_supports(
        &mut self,
        vars: &[Vec<Lit>],
        supports: &[Vec<Option<usize>>],
    ) -> bool {
        let mut len_total = 0;
        for v in vars {
            len_total += v.len();
        }
        assert!(len_total <= i32::max_value() as usize);
        assert!(vars.len() * supports.len() <= i32::max_value() as usize);

        let domain_size = vars.iter().map(|v| v.len() as i32).collect::<Vec<_>>();
        let vars_flat = vars.iter().flatten().copied().collect::<Vec<_>>();
        let mut supports_flat = vec![];
        for support in supports {
            assert_eq!(vars.len(), support.len());
            for i in 0..vars.len() {
                if let Some(n) = support[i] {
                    assert!(n < vars[i].len());
                    supports_flat.push(n as i32);
                } else {
                    supports_flat.push(-1);
                }
            }
        }

        let res = unsafe {
            Glucose_AddDirectEncodingExtensionSupports(
                self.ptr,
                vars.len() as i32,
                domain_size.as_ptr(),
                vars_flat.as_ptr(),
                supports.len() as i32,
                supports_flat.as_ptr(),
            )
        };
        res != 0
    }

    pub fn add_graph_division(
        &mut self,
        domains: &[Vec<i32>],
        dom_lits: &[Vec<Lit>],
        edges: &[(usize, usize)],
        edge_lits: &[Lit],
    ) -> bool {
        assert_eq!(domains.len(), dom_lits.len());
        assert_eq!(edges.len(), edge_lits.len());

        let mut dom_sizes = vec![];
        let mut dom_lits_flat = vec![];
        let mut domains_flat = vec![];

        for i in 0..domains.len() {
            dom_sizes.push(domains[i].len() as i32);
            if !domains[i].is_empty() {
                assert_eq!(domains[i].len(), dom_lits[i].len() + 1);
                for &v in &domains[i] {
                    domains_flat.push(v);
                }
                for &l in &dom_lits[i] {
                    dom_lits_flat.push(l);
                }
            }
        }

        let mut edges_flat = vec![];
        for &(u, v) in edges {
            edges_flat.push(u as i32);
            edges_flat.push(v as i32);
        }

        let res = unsafe {
            Glucose_AddGraphDivision(
                self.ptr,
                domains.len() as i32,
                dom_sizes.as_ptr(),
                domains_flat.as_ptr(),
                dom_lits_flat.as_ptr(),
                edges.len() as i32,
                edges_flat.as_ptr(),
                edge_lits.as_ptr(),
            )
        };
        res != 0
    }

    pub fn add_custom_constraint(&mut self, constraint: Box<dyn CustomConstraint>) -> bool {
        self.custom_constraints.push(Box::new(constraint));
        let c: &Box<dyn CustomConstraint> =
            &self.custom_constraints[self.custom_constraints.len() - 1];
        let c = unsafe { std::mem::transmute::<_, *mut c_void>(c) };
        let res = unsafe { Glucose_AddRustExtraConstraint(self.ptr, c) };
        res != 0
    }

    pub fn set_seed(&mut self, seed: f64) {
        unsafe {
            Glucose_Set_random_seed(self.ptr, seed);
        }
    }

    pub fn set_rnd_init_act(&mut self, rnd_init_act: bool) {
        unsafe {
            Glucose_Set_rnd_init_act(self.ptr, if rnd_init_act { 1 } else { 0 });
        }
    }

    pub fn set_dump_analysis_info(&mut self, dump_analysis_info: bool) {
        unsafe { Glucose_Set_dump_analysis_info(self.ptr, if dump_analysis_info { 1 } else { 0 }) }
    }

    pub fn solve<'a>(&'a mut self) -> Option<Model<'a>> {
        if self.solve_without_model() {
            Some(unsafe { self.model() })
        } else {
            None
        }
    }

    pub fn solve_without_model(&mut self) -> bool {
        let res = unsafe { Glucose_Solve(self.ptr) };
        res != 0
    }

    pub(crate) unsafe fn model<'a>(&'a self) -> Model<'a> {
        Model { solver: self }
    }

    pub fn stats_decisions(&self) -> u64 {
        unsafe { Glucose_SolverStats_decisions(self.ptr) }
    }

    pub fn stats_propagations(&self) -> u64 {
        unsafe { Glucose_SolverStats_propagations(self.ptr) }
    }

    pub fn stats_conflicts(&self) -> u64 {
        unsafe { Glucose_SolverStats_conflicts(self.ptr) }
    }
}

impl Drop for Solver {
    fn drop(&mut self) {
        unsafe {
            Glucose_DestroySolver(self.ptr);
        }
    }
}

const LIT_UNDEF: Lit = Lit(-2);

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

// Interface for implementing custom constraints in Rust

extern "C" {
    fn Glucose_AddRustExtraConstraint(solver: *mut Opaque, trait_object: *mut c_void) -> i32;
    fn Glucose_CustomConstraintCopyReason(reason_vec: *mut c_void, n_lits: i32, lits: *const Lit);
    fn Glucose_SolverValue(solver: *mut Opaque, lit: Lit) -> i32;
    fn Glucose_SolverAddWatch(solver: *mut Opaque, lit: Lit, wrapper_object: *mut c_void);
    fn Glucose_SolverEnqueue(solver: *mut Opaque, lit: Lit, wrapper_object: *mut c_void) -> i32;
}

#[derive(Clone, Copy)]
pub struct SolverManipulator {
    ptr: *mut Opaque,
    wrapper_object: Option<*mut c_void>,
}

impl SolverManipulator {
    pub unsafe fn value(&self, lit: Lit) -> Option<bool> {
        let v = Glucose_SolverValue(self.ptr, lit);
        match v {
            0 => Some(true),
            1 => Some(false),
            2 | 3 => None,
            _ => unreachable!(),
        }
    }

    pub unsafe fn add_watch(&mut self, lit: Lit) {
        assert!(self.wrapper_object.is_some());
        Glucose_SolverAddWatch(self.ptr, lit, self.wrapper_object.unwrap());
    }

    pub unsafe fn enqueue(&mut self, lit: Lit) -> bool {
        assert!(self.wrapper_object.is_some());
        Glucose_SolverEnqueue(self.ptr, lit, self.wrapper_object.unwrap()) != 0
    }
}

pub unsafe trait CustomConstraint {
    fn initialize(&mut self, solver: SolverManipulator) -> bool;
    fn propagate(&mut self, solver: SolverManipulator, p: Lit) -> bool;
    fn calc_reason(
        &mut self,
        solver: SolverManipulator,
        p: Option<Lit>,
        extra: Option<Lit>,
    ) -> Vec<Lit>;
    fn undo(&mut self, solver: SolverManipulator, p: Lit);
}

#[no_mangle]
extern "C" fn Glucose_CallCustomConstraintInitialize(
    solver: *mut Opaque,
    wrapper_object: *mut c_void,
    trait_object: *mut c_void,
) -> i32 {
    let trait_object =
        unsafe { std::mem::transmute::<_, &mut Box<dyn CustomConstraint>>(trait_object) };
    let res = trait_object.initialize(SolverManipulator {
        ptr: solver,
        wrapper_object: Some(wrapper_object),
    });
    if res {
        1
    } else {
        0
    }
}

#[no_mangle]
extern "C" fn Glucose_CallCustomConstraintPropagate(
    solver: *mut Opaque,
    wrapper_object: *mut c_void,
    trait_object: *mut c_void,
    p: Lit,
) -> i32 {
    let trait_object =
        unsafe { std::mem::transmute::<_, &mut Box<dyn CustomConstraint>>(trait_object) };
    let res = trait_object.propagate(
        SolverManipulator {
            ptr: solver,
            wrapper_object: Some(wrapper_object),
        },
        p,
    );
    if res {
        1
    } else {
        0
    }
}

#[no_mangle]
extern "C" fn Glucose_CallCustomConstraintCalcReason(
    solver: *mut Opaque,
    trait_object: *mut c_void,
    p: Lit,
    extra: Lit,
    out_reason: *mut c_void,
) {
    let trait_object =
        unsafe { std::mem::transmute::<_, &mut Box<dyn CustomConstraint>>(trait_object) };
    let res = trait_object.calc_reason(
        SolverManipulator {
            ptr: solver,
            wrapper_object: None,
        },
        if p != LIT_UNDEF { Some(p) } else { None },
        if extra != LIT_UNDEF {
            Some(extra)
        } else {
            None
        },
    );
    unsafe {
        Glucose_CustomConstraintCopyReason(out_reason, res.len() as i32, res.as_ptr());
    }
}

#[no_mangle]
extern "C" fn Glucose_CallCustomConstraintUndo(
    solver: *mut Opaque,
    trait_object: *mut c_void,
    p: Lit,
) {
    let trait_object =
        unsafe { std::mem::transmute::<_, &mut Box<dyn CustomConstraint>>(trait_object) };
    trait_object.undo(
        SolverManipulator {
            ptr: solver,
            wrapper_object: None,
        },
        p,
    );
}

struct LinearTerm {
    lits: Vec<Lit>,
    domain: Vec<i32>,
    coef: i32,
}

impl LinearTerm {
    fn normalize(&mut self) {
        if self.coef < 0 {
            self.lits.reverse();
            for lit in &mut self.lits {
                *lit = !*lit;
            }
            self.domain.reverse();
        }
        for d in &mut self.domain {
            *d *= self.coef;
        }
        self.coef = 1;
    }
}

struct OrderEncodingLinear {
    terms: Vec<LinearTerm>,
    lits: Vec<(Lit, usize, usize)>,
    ub_index: Vec<usize>,
    undo_list: Vec<Option<(usize, usize)>>,
    active_lits: Vec<Lit>,
    total_ub: i32,
    use_optimize: bool,
}

impl OrderEncodingLinear {
    fn new(mut terms: Vec<LinearTerm>, constant: i32, use_optimize: bool) -> OrderEncodingLinear {
        for term in &mut terms {
            assert!(term.coef != 0);
            if term.coef != 1 {
                term.normalize();
            }
        }
        let ub_index = terms.iter().map(|x| x.lits.len()).collect();
        let total_ub = constant
            + terms
                .iter()
                .map(|x| x.domain[x.domain.len() - 1])
                .sum::<i32>();
        let mut lits = vec![];
        for i in 0..terms.len() {
            for j in 0..terms[i].lits.len() {
                lits.push((terms[i].lits[j], i, j));
            }
        }
        lits.sort();
        OrderEncodingLinear {
            terms,
            lits,
            ub_index,
            undo_list: vec![],
            active_lits: vec![],
            total_ub,
            use_optimize,
        }
    }
}

unsafe impl CustomConstraint for OrderEncodingLinear {
    fn initialize(&mut self, mut solver: SolverManipulator) -> bool {
        let mut unique_watchers = vec![];
        for &(lit, _, _) in &self.lits {
            unique_watchers.push(!lit);
        }
        unique_watchers.sort();
        unique_watchers.dedup();

        for &lit in &unique_watchers {
            unsafe {
                solver.add_watch(lit);
            }
        }

        for lit in unique_watchers {
            let val = unsafe { solver.value(lit) };
            if val == Some(true) {
                if !self.propagate(solver, lit) {
                    return false;
                }
            }
        }

        if self.total_ub < 0 {
            return false;
        }

        true
    }

    fn propagate(&mut self, mut solver: SolverManipulator, p: Lit) -> bool {
        self.active_lits.push(p);
        self.undo_list.push(None);

        let mut idx = self.lits.partition_point(|&(lit, _, _)| lit < !p);
        while idx < self.lits.len() && self.lits[idx].0 == !p {
            let (_, i, j) = self.lits[idx];
            idx += 1;
            if self.ub_index[i] <= j {
                continue;
            }
            self.undo_list.push(Some((i, self.ub_index[i])));
            self.total_ub -= self.terms[i].domain[self.ub_index[i]] - self.terms[i].domain[j];
            self.ub_index[i] = j;

            if self.total_ub < 0 {
                return false;
            }
        }

        for i in 0..self.terms.len() {
            let ubi = self.ub_index[i];
            if ubi == 0 {
                continue;
            }
            if self.total_ub - (self.terms[i].domain[ubi] - self.terms[i].domain[0]) >= 0 {
                continue;
            }

            let threshold = self.terms[i].domain[ubi] - self.total_ub;
            let left = self.terms[i].domain.partition_point(|&x| x < threshold) - 1;
            if !unsafe { solver.enqueue(self.terms[i].lits[left]) } {
                return false;
            }
        }

        true
    }

    fn calc_reason(
        &mut self,
        _: SolverManipulator,
        p: Option<Lit>,
        extra: Option<Lit>,
    ) -> Vec<Lit> {
        let mut p_idx: Option<usize> = None;
        if self.use_optimize {
            if let Some(p) = p {
                let idx = self.lits.partition_point(|&(lit, _, _)| lit < p);
                assert!(self.lits[idx].0 == p);
                if idx + 1 == self.lits.len() || self.lits[idx + 1].0 != p {
                    p_idx = Some(self.lits[idx].1);
                }
            }
        }
        let mut ret = vec![];
        for i in 0..self.terms.len() {
            if Some(i) == p_idx {
                continue;
            }
            if self.ub_index[i] < self.terms[i].lits.len() {
                ret.push(!self.terms[i].lits[self.ub_index[i]]);
            }
        }
        ret.extend(extra);
        ret
    }

    fn undo(&mut self, _: SolverManipulator, _: Lit) {
        while let Some((i, j)) = self.undo_list.pop().unwrap() {
            self.total_ub += self.terms[i].domain[j] - self.terms[i].domain[self.ub_index[i]];
            self.ub_index[i] = j;
        }
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

    struct Xor {
        vars: Vec<Var>,
        values: Vec<Option<bool>>,
        parity: bool,
        n_undecided: usize,
    }

    impl Xor {
        fn new(vars: Vec<Var>, parity: bool) -> Xor {
            // This implementation is just for testing.
            // For practical use, we need to add deduplication of literals.
            let size = vars.len();
            Xor {
                vars,
                values: vec![None; size],
                parity,
                n_undecided: size,
            }
        }

        fn var_index(&self, var: Var) -> Option<usize> {
            for i in 0..self.vars.len() {
                if self.vars[i] == var {
                    return Some(i);
                }
            }
            None
        }
    }

    unsafe impl CustomConstraint for Xor {
        fn initialize(&mut self, mut solver: SolverManipulator) -> bool {
            for &var in &self.vars {
                unsafe {
                    solver.add_watch(var.as_lit(false));
                    solver.add_watch(var.as_lit(true));
                }
            }

            for var in self.vars.clone() {
                // TODO: this `clone` is just for silencing the borrow checker
                if let Some(v) = unsafe { solver.value(var.as_lit(false)) } {
                    let lit_sign = var.as_lit(!v);
                    if !self.propagate(solver, lit_sign) {
                        return false;
                    }
                }
            }
            true
        }

        fn propagate(&mut self, mut solver: SolverManipulator, p: Lit) -> bool {
            let s = !p.is_negated();
            let v = p.var();

            let idx = self.var_index(v).unwrap();
            assert!(self.values[idx].is_none());
            self.values[idx] = Some(s);
            self.parity ^= s;
            assert_ne!(self.n_undecided, 0);
            self.n_undecided -= 1;

            if self.n_undecided == 0 {
                if self.parity {
                    return false;
                }
            } else if self.n_undecided == 1 {
                for i in 0..self.vars.len() {
                    if self.values[i].is_none() {
                        if !unsafe { solver.enqueue(self.vars[i].as_lit(!self.parity)) } {
                            return false;
                        }
                    }
                }
            }

            true
        }

        fn calc_reason(
            &mut self,
            _: SolverManipulator,
            p: Option<Lit>,
            extra: Option<Lit>,
        ) -> Vec<Lit> {
            let mut ret = vec![];
            for i in 0..self.vars.len() {
                if let Some(v) = self.values[i] {
                    ret.push(self.vars[i].as_lit(!v));
                } else if p.is_none() && extra.is_some() {
                    ret.push(self.vars[i].as_lit(self.parity));
                }
            }
            assert_eq!(if p.is_some() { 1 } else { 0 } + ret.len(), self.vars.len());
            ret
        }

        fn undo(&mut self, _: SolverManipulator, p: Lit) {
            let v = p.var();

            let idx = self.var_index(v).unwrap();
            assert!(self.values[idx] == Some(!p.is_negated()));
            self.parity ^= self.values[idx].unwrap();
            self.values[idx] = None;
            self.n_undecided += 1;
        }
    }

    #[test]
    fn test_custom_constraint_xor1() {
        let mut solver = Solver::new();
        let x = solver.new_var();
        let y = solver.new_var();

        assert!(solver.add_clause(&[Lit::new(x, false), Lit::new(y, true)]));
        assert!(solver.add_custom_constraint(Box::new(Xor::new(vec![x, y], true))));
        let model = solver.solve().unwrap();
        assert!(model.assignment(x));
        assert!(!model.assignment(y));
    }

    #[test]
    fn test_custom_constraint_xor2() {
        let mut solver = Solver::new();
        let mut vars = vec![];
        for _ in 0..10 {
            vars.push(solver.new_var());
        }

        assert!(solver.add_custom_constraint(Box::new(Xor::new(
            vec![vars[0], vars[1], vars[2], vars[3], vars[4]],
            false
        ))));
        assert!(solver.add_custom_constraint(Box::new(Xor::new(
            vec![vars[3], vars[4], vars[5], vars[6], vars[7]],
            true
        ))));
        assert!(solver.add_custom_constraint(Box::new(Xor::new(
            vec![vars[7], vars[8], vars[9], vars[0], vars[1]],
            true
        ))));

        let mut n_assignments = 0;
        loop {
            match solver.solve() {
                Some(model) => {
                    n_assignments += 1;
                    let mut new_clause = vec![];
                    for &v in &vars {
                        new_clause.push(v.as_lit(model.assignment(v)));
                    }
                    solver.add_clause(&new_clause);
                }
                None => break,
            }
        }

        assert_eq!(n_assignments, 128);
    }

    #[test]
    fn test_custom_constraint_xor3() {
        let mut solver = Solver::new();
        let mut vars = vec![];
        for _ in 0..10 {
            vars.push(solver.new_var());
        }

        assert!(solver.add_custom_constraint(Box::new(Xor::new(
            vec![vars[0], vars[1], vars[2], vars[3], vars[4]],
            false
        ))));
        assert!(solver.add_custom_constraint(Box::new(Xor::new(
            vec![vars[3], vars[4], vars[5], vars[6], vars[7]],
            true
        ))));
        assert!(solver.add_custom_constraint(Box::new(Xor::new(
            vec![vars[7], vars[8], vars[9], vars[0], vars[1]],
            true
        ))));
        assert!(solver.add_custom_constraint(Box::new(Xor::new(
            vec![vars[2], vars[5], vars[6], vars[8], vars[9]],
            true
        ))));
        assert!(solver.solve().is_none());
    }
}
