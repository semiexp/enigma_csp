# EnigmaCSP
CSP solver for puzzles

# Build

To build enigma_csp, you need to setup [Rust](https://www.rust-lang.org/) and a C++ compiler first.

Then, clone this repository including submodules:

```
git clone --recursive https://github.com/semiexp/enigma_csp.git
```

and you can build enigma_csp by

```
cargo build --release
```

This will produce two binaries in `target/release/`:

- `cli`: a CLI interface compatible with [Sugar](https://cspsat.gitlab.io/sugar/) and [csugar](https://github.com/semiexp/csugar).
- `libenigma_csp.so`: a Python binding which can be directly called from [cspuz](https://github.com/semiexp/cspuz). To use this, you will have to make a symlink of name `enigma_csp.so` and add the directory in which `enigma_csp.so` exists to `PYTHONPATH`.

If you are running enigma_csp on Mac, please follow the instruction in [PyO3 user guide](https://pyo3.rs/v0.15.1/building_and_distribution.html#macos).
