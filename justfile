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
