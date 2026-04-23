fn main() {
    // tango-bench requires rustc to export symbols for dynamic linking from
    // benchmark binaries so the `compare` mode can load the baseline binary.
    println!("cargo:rustc-link-arg-benches=-rdynamic");
    println!("cargo:rerun-if-changed=build.rs");
}
