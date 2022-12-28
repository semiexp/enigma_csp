use std::env;

fn main() {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let build_target = vec![
        "lib/glucose/core/Solver.cc",
        "lib/glucose/utils/Options.cc",
        "lib/glucose/constraints/Graph.cc",
        "lib/glucose/constraints/OrderEncodingLinear.cc",
    ];

    #[cfg(feature = "csp-extra-constraints")]
    let build_target = build_target
        .into_iter()
        .chain(["lib/glucose/constraints/DirectEncodingExtension.cc"])
        .collect::<Vec<_>>();

    #[cfg(not(feature = "csp-extra-constraints"))]
    let puzzle_solver_minimal_flag = "-DPUZZLE_SOLVER_MINIMAL=1";
    #[cfg(feature = "csp-extra-constraints")]
    let puzzle_solver_minimal_flag = "-DPUZZLE_SOLVER_MINIMAL=0";

    if arch == "wasm32" {
        // TODO: enigma_csp can be built for wasm32-unknown-emscripten target, but the produced
        // library is hard to use (only a .wasm file is generated without any glue .js files).
        cc::Build::new()
            .cpp(true)
            .file("lib/glucose_bridge.cpp")
            .files(&build_target)
            .include("lib/glucose")
            .flag("-std=c++17")
            .flag("-DGLUCOSE_FIX_OPTIONS")
            .flag("-DGLUCOSE_UNUSE_STDIO")
            .flag(puzzle_solver_minimal_flag)
            .warnings(false)
            .compile("calc");
    } else {
        cc::Build::new()
            .cpp(true)
            .file("lib/glucose_bridge.cpp")
            .files(&build_target)
            .include("lib/glucose")
            .flag("-std=c++17")
            .flag(puzzle_solver_minimal_flag)
            .warnings(false)
            .compile("calc");
    }
    println!("cargo:rerun-if-changed=lib/glucose_bridge.cpp");
}
