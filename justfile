# linear-srgb development recipes

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the standalone apidoc/ package, so plain
# `cargo test` never compiles its dependency tree or runs rustdoc.
fmt:
    cargo fmt --all
    ZEN_API_DOC_TOOLCHAIN=nightly-2026-07-13 cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    ZEN_API_DOC_TOOLCHAIN=nightly-2026-07-13 cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check ZEN_API_DOC_TOOLCHAIN=nightly-2026-07-13 cargo test --manifest-path apidoc/Cargo.toml

# Validate the owner-requested archmage beta constraint update.
beta-check:
    cargo fmt --check
    cargo clippy --locked --lib -- -D warnings
    cargo test --locked --lib

beta-library-check:
    cargo test --locked --all-features --lib

# Precision experiments; these do not change the library's encode policy.
precision-u8:
    cargo run --release --example u8_encode_accuracy -- --exhaustive

precision-u8-bench:
    cargo bench --bench u8_encode_options

precision-parity-check:
    cargo test --features transfer --test remaining_parity --test u8_position

# Emit standalone candidates and measure their linked executable/section sizes.
precision-u8-size output_dir:
    cargo run --release --example u8_encode_accuracy -- --emit-size {{output_dir}}
    bash scripts/measure-encode-size.sh {{output_dir}}

precision-release-check:
    cargo test --all-targets --all-features
    cargo test --doc --all-features
    cargo +1.98.1 semver-checks --baseline-version 0.6.12

precision-audit output_dir:
    bash tools/precision-review/run.sh {{output_dir}} ../garb

precision-inventory workspace output_json:
    python3 tools/precision-review/inventory.py {{workspace}} {{output_json}}

precision-native-summary before after output_tsv:
    python3 tools/precision-review/summarize.py {{before}} {{after}} {{output_tsv}}

portable-math-check:
    PORTABLE_MATH_REQUIRE_NATIVE=1 cargo test --test portable_math
    cargo check --no-default-features
    cargo run --release --example portable_math_accuracy

portable-math-bench:
    cargo bench --bench portable_math
