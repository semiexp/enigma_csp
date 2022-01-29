pub mod arithmetic;
pub mod config;
pub mod csp;
mod csp_repr;
pub mod csugar_cli;

pub mod encoder;
pub mod glucose;
pub mod integration;
pub mod norm_csp;
pub mod normalizer;
pub mod parser;

#[cfg(not(target_arch = "wasm32"))]
mod pyo3_binding;

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

#[cfg(not(target_arch = "wasm32"))]
pub use pyo3_binding::enigma_csp;

#[cfg(target_arch = "wasm32")]
mod wasm;
