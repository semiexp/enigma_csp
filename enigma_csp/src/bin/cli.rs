extern crate enigma_csp;

#[allow(unused)]
use std::io;

#[cfg(feature = "parser")]
fn main() {
    let stdin = io::stdin();
    let mut lock = stdin.lock();
    let config = enigma_csp::config::Config::parse_from_args();
    let (res, _) = enigma_csp::csugar_cli::csugar_cli(&mut lock, config);
    print!("{}", res);
}

#[cfg(not(feature = "parser"))]
fn main() {
    panic!("parser feature not enabled");
}
