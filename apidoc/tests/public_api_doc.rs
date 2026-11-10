//! Public-API surface snapshots for the PARENT package (docs/public-api/).
//! Shared implementation + format docs: the `zenutils-apidoc` crate.
#[test]
fn public_api_surface_docs_are_current() {
    let target = std::env::var("ZEN_API_DOC_TARGET").expect(
        "select a target with ZEN_API_DOC_TARGET, or run just api-doc-check for all targets",
    );
    assert!(
        [
            "x86_64-unknown-linux-gnu",
            "aarch64-unknown-linux-gnu",
            "wasm32-wasip1"
        ]
        .contains(&target.as_str()),
        "unsupported snapshot target: {target}"
    );
    eprintln!("Checking public API surface for {target}");
    zenutils_apidoc::ApiDoc::new()
        .workspace_dir("..")
        .out_dir(&format!("docs/public-api/{target}"))
        .run();
}
