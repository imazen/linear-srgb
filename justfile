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

# Check immutable output chunks, captured from main before PR #30 merges.
numerical-goldens-check:
    env -u CARGO_ENCODED_RUSTFLAGS RUSTFLAGS="" RUSTUP_TOOLCHAIN=1.99.1 CARGO_TARGET_DIR=target/goldens cargo test --locked --release --manifest-path goldens/Cargo.toml

numerical-goldens-capture new_directory:
    env -u CARGO_ENCODED_RUSTFLAGS RUSTFLAGS="" RUSTUP_TOOLCHAIN=1.99.1 CARGO_TARGET_DIR=target/goldens cargo run --locked --release --manifest-path goldens/Cargo.toml --example capture -- {{new_directory}}

numerical-goldens-wasm features="":
    env -u CARGO_ENCODED_RUSTFLAGS RUSTFLAGS="-C target-feature=+simd128" RUSTUP_TOOLCHAIN=1.99.1 CARGO_TARGET_DIR=target/goldens CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run" cargo test --locked --release --manifest-path goldens/Cargo.toml --no-default-features --features "{{features}}" --target wasm32-wasip1 -- --nocapture

# This explicit mode fails when the caller lacks AVX-512; it never skips it.
numerical-goldens-v4:
    env -u CARGO_ENCODED_RUSTFLAGS RUSTFLAGS="" RUSTUP_TOOLCHAIN=1.99.1 CARGO_TARGET_DIR=target/goldens cargo test --locked --release --manifest-path goldens/Cargo.toml --features v4

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
