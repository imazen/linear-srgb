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
native x4 scalar 348a68c0ef3afc2d
native x8 scalar 189a94216e6bf13f
native x16 scalar e2f702e2ba31eb8e
native x4 scalar_transfer 7cb3c2c4d0a8381b
native x8 scalar_transfer 583a6c212cb86b5e
native x16 scalar_transfer 7b77f9068d50a12e
native x4 v3 348a68c0ef3afc2d
native x8 v3 189a94216e6bf13f
native x16 v3 e2f702e2ba31eb8e
native x4 v3_transfer 7cb3c2c4d0a8381b
native x8 v3_transfer 583a6c212cb86b5e
native x16 v3_transfer 7b77f9068d50a12e
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
native x4 v4 348a68c0ef3afc2d
native x8 v4 189a94216e6bf13f
native x16 v4 e2f702e2ba31eb8e
native x4 v4_transfer 7cb3c2c4d0a8381b
native x8 v4_transfer 583a6c212cb86b5e
native x16 v4_transfer 7b77f9068d50a12e
```

## Refactor validation

The final native V3/scalar and WASM captures match all 24 original rows above.
Prebuilt original/current probes also ran on r7900x without a remote build;
all 18 available-tier rows match there, including the six V4 rows added above.
The required-V3/V4 compatibility tests pass. These V4 rows were captured from
the unchanged original before comparing the refactor.
The all-feature release suite, doctests, required-V3 compatibility tests,
no-AVX512 compatibility/parity tests and WASM compatibility/dispatch/tier/row
parity tests pass. Strict all-target/all-feature and no_std Clippy pass;
ARM64 source and tests type-check. Validation used both the manifest's
resolved 0.9.31 dependencies and its published 0.9.31-beta.2 floor during
this work; the final full regression run used beta.2.

The supported, feature and internal API renders are byte-identical to the
unchanged original PR when rendered with the same local tooling. The local
`just api-doc-check` separately fails on the original PR too: its nightly
renderer reports negative auto-traits for the two const-generic LUT types,
where the committed snapshot reports three auto-trait-complete types.
The committed API expectation is unchanged; no function signature differs.

Assembly captures use separate Cargo target directories for the unchanged
original and current sources. Sharing an output directory overwrote example
assembly outputs, so the initial shared-directory comparison was marked invalid
and rerun with isolated outputs. All 24 original instruction shapes match
the committed expectations with the isolated captures.

The reviewed refactor changes six ARM instruction shapes and two WASM gamma
shapes. ARM static opcode counts: sRGB decode 225 to 224; encode 234 to 232;
extended decode 267 to 266; extended encode 278 to 276; gamma decode 377 to
375; gamma encode 383 to 381. WASM gamma counts remain 644 and 646, with a
local-variable setup reorder. The other 16 shapes are unchanged. These are
static instruction counts, not runtime speed measurements. The owner approved updating these eight assembly expectation pairs on
2026-11-10. Other snapshots and numerical expectations are unchanged.

No runtime throughput or clean compilation speed improvement is claimed.
The full beta.2 regression command's resource wrapper reported:

```text
run-heavy: done rc=0 49s | peak-RSS 0.41GiB | min-avail 24717MiB | peak-load 2.31
```
