# cspuz_core
CSP solver for puzzles

# Build

To build cspuz_core, you need to setup [Rust](https://www.rust-lang.org/) and a C++ compiler first.

Then, clone this repository including submodules:

```
git clone --recursive https://github.com/semiexp/cspuz_core.git
```

and you can build cspuz_core by

```
cargo build --release
```

This will produce two binaries in `target/release/`:

- `cli`: a CLI interface compatible with [Sugar](https://cspsat.gitlab.io/sugar/) and [csugar](https://github.com/semiexp/csugar).
- `libcspuz_core.so`: a Python binding which can be directly called from [cspuz](https://github.com/semiexp/cspuz). To use this, you will have to make a symlink of name `cspuz_core.so` and add the directory in which `enigma_csp.so` exists to `PYTHONPATH`.

If you are running cspuz_core on Mac, please follow the instruction in [PyO3 user guide](https://pyo3.rs/v0.15.1/building_and_distribution.html#macos).
