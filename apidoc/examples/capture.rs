//! Render an isolated API capture without replacing committed expectations.
fn main() {
    let mut args = std::env::args().skip(1);
    let workspace = args.next().expect("workspace directory required");
    let output = args.next().expect("output directory required");
    assert!(
        args.next().is_none(),
        "expected workspace and output directories"
    );
    zenutils_apidoc::ApiDoc::new()
        .workspace_dir(&workspace)
        .out_dir(&output)
        .run();
}
