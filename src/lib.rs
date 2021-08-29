pub mod csp;
pub mod glucose;
pub mod norm_csp;
pub mod normalizer;

#[derive(Clone, Copy)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Lt,
    Ge,
    Gt,
}
