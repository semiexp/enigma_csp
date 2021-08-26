fn main() {
    cc::Build::new()
        .cpp(true)
        .file("lib/glucose_bridge.cpp")
        .files(&[
            "lib/glucose/core/Solver.cc",
            "lib/glucose/utils/Options.cc",
            "lib/glucose/utils/System.cc",
        ])
        //        .cpp_link_stdlib("c++")
        .include("lib/glucose")
        .flag("-std=c++17")
        .compile("calc");
    println!("cargo:rerun-if-changed=lib/glucose_bridge.cpp");
}
