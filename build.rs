use std::env;

fn main() {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    if arch == "wasm32" {
        // TODO: enigma_csp can be built for wasm32-unknown-emscripten target, but the produced
        // library is hard to use (only a .wasm file is generated without any glue .js files).
        cc::Build::new()
            .cpp(true)
            .file("lib/glucose_bridge.cpp")
            .files(&[
                "lib/glucose/core/Solver.cc",
                "lib/glucose/utils/Options.cc",
                "lib/glucose/utils/System.cc",
                "lib/glucose/constraints/Graph.cc",
                "lib/glucose/constraints/OrderEncodingLinear.cc",
            ])
            .include("lib/glucose")
            .flag("-std=c++17")
            .warnings(false)
            .compile("calc");
        println!("cargo:rustc-link-arg=--no-entry");
    } else {
        cc::Build::new()
            .cpp(true)
            .file("lib/glucose_bridge.cpp")
            .files(&[
                "lib/glucose/core/Solver.cc",
                "lib/glucose/utils/Options.cc",
                "lib/glucose/utils/System.cc",
                "lib/glucose/constraints/Graph.cc",
                "lib/glucose/constraints/OrderEncodingLinear.cc",
            ])
            .include("lib/glucose")
            .flag("-std=c++17")
            .warnings(false)
            .compile("calc");
    }
    println!("cargo:rerun-if-changed=lib/glucose_bridge.cpp");
}
