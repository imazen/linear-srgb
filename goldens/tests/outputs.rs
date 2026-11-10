//! A baseline from main, independent of the PR's implementation and parity tests.
#![forbid(unsafe_code)]

use linear_srgb_goldens as goldens;

macro_rules! fixtures {
    ($profile:literal) => {
        [
            (
                "scalar.txt",
                include_str!(concat!("../fixtures/", $profile, "/scalar.txt")),
            ),
            (
                "slices.txt",
                include_str!(concat!("../fixtures/", $profile, "/slices.txt")),
            ),
            (
                "rgba.txt",
                include_str!(concat!("../fixtures/", $profile, "/rgba.txt")),
            ),
            (
                "integers.txt",
                include_str!(concat!("../fixtures/", $profile, "/integers.txt")),
            ),
            (
                "u8_decode.txt",
                include_str!(concat!("../fixtures/", $profile, "/u8_decode.txt")),
            ),
            (
                "tokens.txt",
                include_str!(concat!("../fixtures/", $profile, "/tokens.txt")),
            ),
        ]
    };
}

#[test]
fn numerical_outputs_match_main() {
    let expected = match goldens::profile() {
        "x86_64" => fixtures!("x86_64"),
        "aarch64" => fixtures!("aarch64"),
        "wasm32-default" => fixtures!("wasm32-default"),
        "wasm32-precise" => fixtures!("wasm32-precise"),
        profile => panic!("missing committed golden profile {profile}"),
    };
    let actual = goldens::capture();
    #[cfg(all(target_arch = "x86_64", feature = "v4"))]
    let expected = {
        let mut expected = expected.to_vec();
        expected.push((
            "tokens-v4.txt",
            include_str!("../fixtures/x86_64/tokens-v4.txt"),
        ));
        expected
    };
    assert_eq!(expected.len(), actual.len(), "golden group set changed");
    let mut errors = Vec::new();
    for (name, expected) in expected {
        errors.extend(goldens::compare(name, expected, &actual[name]));
    }
    assert!(
        errors.is_empty(),
        "numerical drift from main ({} chunks/cases):\n{}",
        errors.len(),
        errors.join("\n")
    );
}
