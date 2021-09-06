pub mod csp;
pub mod encoder;
pub mod glucose;
pub mod integration;
pub mod norm_csp;
pub mod normalizer;
pub mod sat;

#[derive(Clone, Copy, Hash)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Lt,
    Ge,
    Gt,
}
