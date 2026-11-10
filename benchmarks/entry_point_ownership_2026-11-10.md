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

## Initial refactor validation

This section records the captures before the later public-surface reduction.
The reduction below intentionally changes public signatures while preserving
the numerical captures.

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

The refactor is `d79656ce`; the pre-refactor probe is `3c498ae8` on the
unchanged numerical source at `f36f932c`. Supporting assembly captures,
probe binaries and validation logs were preserved on **i270** at
`/mnt/v/output/linear-srgb/api-ownership-2026-11-10/`; `SHA256SUMS` records
their hashes. This path was checked on i270, not dev. It is local validation
output, not a canonical training-data store. V4 execution used the identical
prebuilt binaries on r7900x, without a build on that host.


## Default-WASM rounding follow-up

The original PR also fails its existing caller-owned PQ dispatch assertion and
u16 row-split assertion when `precise-portable` is off. Reproduced on unchanged
`3c498ae8` with `--no-default-features --features std,transfer` under WASI
SIMD128: `target_features_dispatch` fails at its PQ array equality, and
`remaining_parity` fails at `decode width=1`. Fused scalar math and relaxed
SIMD math take different rounding paths.

Fix `e24d12ee` uses the existing relaxed generic kernels for default-WASM
scalar HDR entries and pads u16 tails through their vector-block calculation.
Scalar f32 slice entries also follow their relaxed array kernels on this
configuration. Native and `precise-portable` keep the previous evaluation.
No numerical assertions or expected outputs were weakened. CI now exercises
the existing WASM dispatch, compatibility, tier and row-parity integration tests
in addition to the library tests.

`just verify-ownership-regressions` runs the native suite, strict native and
WASM lint checks, ARM64 type-check and both WASM numerical policies. WASM
linting covers the library and the four exercised integration-test targets;
native-only benchmark harnesses cannot compile for WASM, and unrelated
architecture-specific test targets have pre-existing WASM unused-item warnings.
The full native all-target lint remains strict.

After this fix, `just ownership-fingerprint` and
`just ownership-fingerprint-wasm` still match all 24 original native/WASM
fingerprint rows. A fresh assembly capture matches all 24 approved opcode
expectations, and all three API renders remain byte-identical to the unchanged
original PR. These probes use `--all-features`, including `precise-portable`;
default-WASM correctness is checked by the existing exact dispatch and row
assertions rather than by those frozen fingerprints.

The follow-up assembly capture lives in `asm-followup/` within the artifact
directory above. The original default-WASM failures, successful regression
logs and fresh fingerprint logs are preserved beside it with updated hashes.

The final follow-up regression command reported:

```text
run-heavy: done rc=0 28s | peak-RSS 0.14GiB | min-avail 24923MiB | peak-load 2.89
```


## Narrow public replacements (2026-11-10)

The user requested direct x4/x8 token replacements only, with no direct public
replacement for tokens::x16. The default-feature x86 snapshot's target_features
summary falls from 123 to 34 items (32 functions and two width modules).
Across the captured default/optional feature sets, the API guard requires exact
equality with 64 token-free x4/x8 signatures: 28 x4 and 36 x8. The 28 legacy
x16 signatures remain under the deprecated token module, with private code
behind them. No numerical implementation or expected bits changed for this
reduction.

The original compatibility and wide-dispatch assertions moved intact into
private unit-test modules; consumer integration tests exercise the public
replacements. The frozen probe moved from an example into
`src/target_features/fingerprint.rs`; `just ownership-fingerprint` and its
WASM counterpart still require the original captured hashes. Recipes that
exercise private compatibility/dispatch now include library unit tests.

The API renderer's existing auto-trait discrepancy is avoided by using the
installed nightly-2026-07-13 with the same zenutils-apidoc 0.1.1. It renders all
three original auto-trait-complete types, and `just api-doc-check` now uses that
toolchain by default. Only the authorized API removal appears in the public
snapshot change; no auto-trait guarantee was relaxed.

The visibility reduction is `0c5c8bde`; the exact-set API guard, deprecation
notice and narrowed snapshots are `320dcd6b`. Compatibility assertions moved
in `511752ec`. The fresh ARM/WASM capture in `asm-narrow/` matches all 24
approved opcode expectations. Strict native and selected WASM lint, the native
release suite/doctests, ARM64 checks, strict-SIMD WASM tests under both arithmetic
policies and all 24 native/WASM fingerprints pass. The narrowed API snapshot
check also passes. The unrelated relaxed-SIMD fusing-engine parity finding
remains open.

```text
run-heavy: done rc=0 83s | peak-RSS 0.50GiB | min-avail 24757MiB | peak-load 2.42
```

This combined command included numerical regression tests, frozen probes and
the compatible-nightly API capture. A follow-up removed an unused test-only
macro import on portable targets; its WASM test compilation and API/ASM checks
passed in the later capture command.

The narrowed API captures and verification logs are preserved under
`narrow-api-review/` in the i270 artifact directory above; its manifest now
includes those files and `asm-narrow/`.


## Target captures and attune cleanup (2026-11-10)

Independent x86-64, ARM64 and WASM captures now verify their own replacement
sets (64, 28 and 28 respectively). The three snapshot jobs and mutation guard
checks run in API CI. The shared renderer adds cross-target rustdoc invocation,
target-specific JSON lookup and architecture labels resolved from rustc;
its dev-only git pin is documented in the snapshot guide.

Direct feature contexts use attune. Scalar functions are ordinary inlined
functions; unused scalar slice variants were removed rather than suppressing
dead-code warnings. The existing fusion selection is centralized in
`USE_FUSED_MATH`, including scalar evaluation and u16 tails. Generated target
wrappers and scalar implementations no longer repeat WASM/precision cfgs.

The native release suite and doctests, strict native and selected WASM lint,
ARM64 checks, strict-SIMD WASM under both policies, all three API captures,
all 24 native/WASM frozen hashes and all 24 approved opcode expectations pass.
The fresh assembly is preserved under `asm-attune/` in the artifact directory
above. The combined verification resource wrapper reported:

```text
run-heavy: done rc=0 98s | peak-RSS 0.49GiB | min-avail 23716MiB | peak-load 3.73
```

The four relaxed-SIMD engine/policy commands were also rerun. Their outcomes
remain three passes and the known default-policy/fusing-engine dispatch
assertion failure; this cleanup did not change the arithmetic contract or
hide that failing test. Logs are preserved under `target-and-attune-review/`
with updated hashes in the same i270 artifact directory.
