# linear-srgb development recipes

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the standalone apidoc/ package, so plain
# `cargo test` never compiles its dependency tree or runs rustdoc.
fmt:
    cargo fmt --all
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Validate the owner-requested archmage beta constraint update.
beta-check:
    cargo fmt --check
    cargo clippy --locked --lib -- -D warnings
    cargo test --locked --lib

beta-library-check:
    cargo test --locked --all-features --lib

# Verify CPU-feature entry points, compatibility, docs and feature combinations.
verify-target-features:
    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test --all-features --release
    cargo test --doc --all-features
    cargo test --no-default-features --features std,transfer --test target_features
    cargo check --no-default-features

# Type-check ARM and execute the portable SIMD entry points under WASI.
verify-target-features-portable:
    cargo check --all-features --tests --target aarch64-unknown-linux-gnu
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run --env LINEAR_SRGB_REQUIRE_TIERS=Wasm128Token" CARGO_TARGET_WASM32_WASIP1_RUSTFLAGS="-C target-feature=+simd128" cargo test --all-features --test target_features --test target_features_dispatch --target wasm32-wasip1

# Run after regenerating snapshots; includes aliases collapsed by the renderer.
check-target-features-surface:
    python3 scripts/check_target_features_api.py

# Require both x86 SIMD tiers to execute, rather than accept unavailable hardware.
verify-target-features-native-v4:
    LINEAR_SRGB_REQUIRE_TIERS=X64V3Token,X64V4Token cargo test --all-features --test target_features

# Regenerate the missing width/tier entry points from their shared operation list.
generate-portable-target-features:
    python3 scripts/generate_portable_target_features.py

# Caller-owned runtime dispatch with token-free x8/x16 library boundaries.
verify-wide-dispatch:
    cargo test --all-features --release --test target_features_dispatch

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

precision-policy output_dir:
    python3 tools/precision-review/generate.py {{output_dir}} ../garb
    POLICIES=1 cargo run --release --manifest-path {{output_dir}}/Cargo.toml --bin oracle

# Print portable/native array and tail fingerprints for an ownership refactor.
ownership-fingerprint:
    cargo run --release --all-features --example entry_point_fingerprint

# Capture the existing dispatcher stubs without changing committed ASM snapshots.
ownership-codegen:
    cargo rustc --release --all-features --example asm-stub -- --emit=asm
