# linear-srgb development recipes

# Match the public-api parser and preserve correct const-generic auto traits.
api_doc_toolchain := env_var_or_default("ZEN_API_DOC_TOOLCHAIN", "nightly-2026-07-13")
api_doc_targets := "x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu wasm32-wasip1"

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the standalone apidoc/ package, so plain
# `cargo test` never compiles its dependency tree or runs rustdoc.
fmt:
    cargo fmt --all
    just api-doc

# Regenerate the public-API surface snapshots only
api-doc:
    for target in {{api_doc_targets}}; do ZEN_API_DOC_TARGET="$target" ZEN_API_DOC_TOOLCHAIN={{api_doc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml -- --nocapture || exit $?; done
    python3 scripts/check_target_features_api.py
    python3 scripts/test_target_features_api.py

# Verify the committed snapshots are current
api-doc-check:
    for target in {{api_doc_targets}}; do ZEN_API_DOC_TARGET="$target" ZEN_API_DOC_TOOLCHAIN={{api_doc_toolchain}} ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml -- --nocapture || exit $?; done
    python3 scripts/check_target_features_api.py
    python3 scripts/test_target_features_api.py

# Record real expanded SIMD modules, with pinned Rust and macro dependencies.
expand-snapshots:
    python3 scripts/expand_snapshots.py

expand-snapshots-check:
    python3 scripts/expand_snapshots.py --check

# Validate the owner-requested archmage beta constraint update.
beta-check:
    cargo fmt --check
    cargo clippy --locked --lib -- -D warnings
    cargo test --locked --lib

beta-library-check:
    cargo test --locked --all-features --lib

# Verify CPU-feature entry points, compatibility, docs and feature combinations.
verify-target-features:
    python3 scripts/check_target_feature_ownership.py
    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test --all-features --release
    cargo test --doc --all-features
    cargo test --no-default-features --features std,transfer --test target_features
    cargo check --no-default-features

# Type-check ARM and execute the portable SIMD entry points under WASI.
verify-target-features-portable:
    cargo check --all-features --tests --target aarch64-unknown-linux-gnu
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run --env LINEAR_SRGB_REQUIRE_TIERS=Wasm128Token" CARGO_TARGET_WASM32_WASIP1_RUSTFLAGS="-C target-feature=+simd128" cargo test --all-features --lib --test target_features --test target_features_dispatch --target wasm32-wasip1

# Run after regenerating snapshots; includes aliases collapsed by the renderer.
check-target-features-surface:
    python3 scripts/check_target_features_api.py
    python3 scripts/test_target_features_api.py

# Require both x86 SIMD tiers to execute, rather than accept unavailable hardware.
verify-target-features-native-v4:
    LINEAR_SRGB_REQUIRE_TIERS=X64V3Token,X64V4Token cargo test --all-features --lib target_features::compatibility_tests

# Regenerate private internal dispatch helpers; public replacements stay explicit.
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
    LINEAR_SRGB_REQUIRE_TIERS=X64V3Token cargo test --release --all-features --lib target_features::fingerprint::entry_point_fingerprint -- --nocapture

# Capture the existing dispatcher stubs without changing committed ASM snapshots.
ownership-codegen:
    cargo rustc --release --all-features --example asm-stub -- --emit=asm

# Enforce dependency direction without relying on public-signature snapshots.
check-target-feature-ownership:
    python3 scripts/check_target_feature_ownership.py

# Compare captured probe output against the pre-refactor numerical fingerprints.
ownership-fingerprint-check output:
    python3 scripts/check_entry_point_fingerprints.py {{output}}

# Isolated API render: compare original/current without replacing expectations.
ownership-api workspace output_dir target="x86_64-unknown-linux-gnu":
    ZEN_API_DOC_TARGET={{target}} ZEN_API_DOC_TOOLCHAIN={{api_doc_toolchain}} ZEN_API_DOC=regen cargo run --manifest-path apidoc/Cargo.toml --example capture -- {{workspace}} {{output_dir}}

ownership-asm-compare before after:
    python3 scripts/compare_entry_point_asm.py {{before}} {{after}} --require-identical

# Render assembly to an isolated review directory; preserve committed expectations.
ownership-asm output_dir:
    bash scripts/dump-asm.sh "" {{output_dir}}

# Default WASM policy must compose through the scalar and SIMD entry families too.
verify-target-features-wasm-default:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run --env LINEAR_SRGB_REQUIRE_TIERS=Wasm128Token" CARGO_TARGET_WASM32_WASIP1_RUSTFLAGS="-C target-feature=+simd128" cargo test --no-default-features --features std,transfer --release --lib --test target_features --test target_features_dispatch --test tier_parity --test remaining_parity --target wasm32-wasip1

# Full ownership regression matrix, including both WASM rounding policies.
verify-ownership-regressions:
    just verify-target-features
    cargo clippy --no-default-features --features transfer -- -D warnings
    cargo check --all-features --tests --target aarch64-unknown-linux-gnu
    just verify-target-features-wasm-default
    CARGO_TARGET_WASM32_WASIP1_RUSTFLAGS="-C target-feature=+simd128" cargo clippy --no-default-features --features std,transfer --target wasm32-wasip1 --lib --test target_features --test target_features_dispatch --test tier_parity --test remaining_parity -- -D warnings
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run --env LINEAR_SRGB_REQUIRE_TIERS=Wasm128Token" CARGO_TARGET_WASM32_WASIP1_RUSTFLAGS="-C target-feature=+simd128" cargo test --all-features --release --target wasm32-wasip1 --lib --test target_features --test target_features_dispatch --test tier_parity --test remaining_parity

# Capture the portable fingerprints with the frozen precise-portable policy.
ownership-fingerprint-wasm:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run" CARGO_TARGET_WASM32_WASIP1_RUSTFLAGS="-C target-feature=+simd128" cargo test --release --all-features --target wasm32-wasip1 --lib target_features::fingerprint::entry_point_fingerprint -- --nocapture

# Exercise relaxed SIMD with an explicitly fusing or nonfusing Wasmtime engine.
probe-relaxed-wasm fma="true" features="std,transfer":
    CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run -W relaxed-simd=true -C cranelift-has-fma={{fma}}" CARGO_TARGET_WASM32_WASIP1_RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" cargo test --no-default-features --features "{{features}}" --release --target wasm32-wasip1 --lib --test target_features --test target_features_dispatch --test remaining_parity --test tier_parity -- --nocapture
