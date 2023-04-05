use std::io::{Read, Write};
use std::ops::Not;
use std::process::{Command, Stdio};

#[derive(Clone, Copy, Debug)]
pub struct Var(pub(crate) i32);

impl Var {
    pub fn as_lit(&self, negated: bool) -> Lit {
        Lit::new(*self, negated)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Lit(pub(crate) i32);

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
    num_vars: i32,
    clauses: Vec<Vec<Lit>>,
    model: Vec<bool>,
}

impl Solver {
    pub fn new() -> Solver {
        Solver {
            num_vars: 0,
            clauses: vec![],
            model: vec![],
        }
    }

    pub fn new_var(&mut self) -> Var {
        let ret = Var(self.num_vars);
        self.num_vars += 1;
        ret
    }

    pub fn num_var(&self) -> i32 {
        self.num_vars
    }

    pub fn all_vars(&self) -> Vec<Var> {
        (0..self.num_var()).map(|i| Var(i)).collect()
    }

    pub fn add_clause(&mut self, clause: &[Lit]) -> bool {
        self.clauses.push(clause.to_owned());
        true
    }

    pub fn solve<'a>(&'a mut self) -> Option<Model<'a>> {
        if self.solve_without_model() {
            Some(unsafe { self.model() })
        } else {
            None
        }
    }

    pub fn solve_without_model(&mut self) -> bool {
        let mut description = String::new();
        description.push_str(&format!("p cnf {} {}\n", self.num_vars, self.clauses.len()));
        for clause in &self.clauses {
            for l in clause {
                let n = (l.var().0 + 1) * if l.is_negated() { -1 } else { 1 };
                description.push_str(&(n.to_string()));
                description.push(' ');
            }
            description.push_str("0\n");
        }

        let solver_command = std::env::var("ENIGMA_CSP_EXTERNAL_SOLVER").unwrap();
        let process = Command::new(solver_command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();

        process
            .stdin
            .unwrap()
            .write_all(description.as_bytes())
            .unwrap();

        let mut output = String::new();
        process.stdout.unwrap().read_to_string(&mut output).unwrap();

        let mut is_sat: Option<bool> = None;
        let mut model = vec![false; self.num_vars as usize];

        for line in output.split(&['\r', '\n']) {
            if line.starts_with("s ") {
                if line.starts_with("s UNSAT") {
                    is_sat = Some(false);
                } else if line.starts_with("s SAT") {
                    is_sat = Some(true);
                } else {
                    panic!();
                }
            } else if line.starts_with("v ") {
                for toks in line.split(' ').skip(1) {
                    let n = toks.parse::<i32>().unwrap();
                    if n == 0 {
                        break;
                    }
                    if n > 0 {
                        model[(n - 1) as usize] = true;
                    }
                }
            }
        }

        if is_sat == Some(true) {
            self.model = model;
        }

        is_sat.unwrap()
    }

    pub(crate) unsafe fn model<'a>(&'a self) -> Model<'a> {
        Model { solver: self }
    }
}

pub struct Model<'a> {
    solver: &'a Solver,
}

impl<'a> Model<'a> {
    pub fn assignment(&self, var: Var) -> bool {
        self.solver.model[var.0 as usize]
    }
}
