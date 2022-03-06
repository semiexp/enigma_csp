extern crate enigma_csp;

use std::io;

fn main() {
    let stdin = io::stdin();
    let mut lock = stdin.lock();
    let config = enigma_csp::config::Config::parse_from_args();
    let res = enigma_csp::csugar_cli::csugar_cli(&mut lock, config);
    print!("{}", res);
}
