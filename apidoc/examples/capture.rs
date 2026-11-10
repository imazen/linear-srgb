//! Render an isolated API capture without replacing committed expectations.
fn main() {
    let mut args = std::env::args().skip(1);
    let workspace = args.next().expect("workspace directory required");
    let output = args.next().expect("output directory required");
    assert!(
        args.next().is_none(),
        "expected workspace and output directories"
    );
    std::env::var("ZEN_API_DOC_TARGET")
        .expect("select the documented target with ZEN_API_DOC_TARGET");
    zenutils_apidoc::ApiDoc::new()
        .workspace_dir(&workspace)
        .out_dir(&output)
        .run();
}
