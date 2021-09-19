#[cfg(target_os = "linux")]
fn main() {
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
        //        .cpp_link_stdlib("c++")
        .include("lib/glucose")
        .flag("-std=c++17")
        .warnings(false)
        .compile("calc");
    println!("cargo:rerun-if-changed=lib/glucose_bridge.cpp");
}

#[cfg(not(target_os = "linux"))]
fn main() {
    // TODO: build on non-linux target is temporarily disabled for rust-analyzer running on Windows
}
