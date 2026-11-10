# Entry-point ownership baseline

Source: PR #30 `f36f932c74db8b9cdd4f6687eed36459a30c085b`, with the
`entry_point_fingerprint` example added before changing library code.
Host: i270 (Core Ultra 7 270K Plus), rustc 1.99.1; native V3 available,
V4 unavailable. WASM: wasmtime, SIMD128, all crate features.
The root lock resolved archmage/magetypes 0.9.31.

Commands: `just ownership-fingerprint`, `just ownership-codegen`, and
`CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run"
CARGO_TARGET_WASM32_WASIP1_RUSTFLAGS="-C target-feature=+simd128"
cargo run --release --all-features --example entry_point_fingerprint --target wasm32-wasip1`.

Each hash folds output bits over the probe's fixed array grid, five gamma
exponents and slice lengths 0 through 65. NaN payload/sign are normalized;
finite bits and signed zero remain distinct. Widths have different chunk
padding, so compare each named row with its own pre-refactor value.
These are numerical fingerprints, not throughput measurements or exhaustive
floating-point proofs. The transfer rows cover all eight transfer functions.

```text
wasm x4 scalar 348a68c0ef3afc2d
wasm x8 scalar 189a94216e6bf13f
wasm x16 scalar e2f702e2ba31eb8e
wasm x4 scalar_transfer 7cb3c2c4d0a8381b
wasm x8 scalar_transfer 583a6c212cb86b5e
wasm x16 scalar_transfer 7b77f9068d50a12e
wasm x4 v3 348a68c0ef3afc2d
wasm x8 v3 189a94216e6bf13f
wasm x16 v3 e2f702e2ba31eb8e
wasm x4 v3_transfer 7cb3c2c4d0a8381b
wasm x8 v3_transfer 583a6c212cb86b5e
wasm x16 v3_transfer 7b77f9068d50a12e
wasm x4 scalar 348a68c0ef3afc2d
wasm x8 scalar 189a94216e6bf13f
wasm x16 scalar e2f702e2ba31eb8e
wasm x4 scalar_transfer 7cb3c2c4d0a8381b
wasm x8 scalar_transfer 583a6c212cb86b5e
wasm x16 scalar_transfer 7b77f9068d50a12e
wasm x4 wasm128 196ca68ba31db809
wasm x8 wasm128 17f4931a7e87e837
wasm x16 wasm128 8aecfbc2140aa5be
wasm x4 wasm128_transfer 3dd42b31e7300b7e
wasm x8 wasm128_transfer 3eccada05eb692e7
wasm x16 wasm128_transfer 7879ec24f877b577
```
