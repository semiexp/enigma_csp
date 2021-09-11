pub mod csp;
pub mod encoder;
pub mod glucose;
pub mod integration;
pub mod norm_csp;
pub mod normalizer;
pub mod parser;
pub mod sat;
mod util;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Lt,
    Ge,
    Gt,
}
