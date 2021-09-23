use pyo3::prelude::*;

use crate::csugar_cli::csugar_cli;

#[pyfunction]
fn solver(input: String) -> String {
    let mut bytes = input.as_bytes();
    let res = csugar_cli(&mut bytes);
    res
}

#[pymodule]
pub fn enigma_csp(_py: Python, m: &PyModule) -> PyResult<()> {
    // TODO: addition of `solver` is temporarily disabled for rust-analyzer running on Windows
    #[cfg(target_os = "linux")]
    m.add_function(wrap_pyfunction!(solver, m)?)?;

    Ok(())
}
