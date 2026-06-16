# Changelog

## [Unreleased]

### Added

- Versioned public-API surface snapshot at `docs/public-api/linear-srgb.txt`, regenerated on every `cargo test` by `tests/public_api_doc.rs` (`ZEN_API_DOC=check` verifies in CI, `=off` skips); `justfile` recipes `fmt` / `api-doc` / `api-doc-check`. Replaces the manual `api-snapshots/{default,all}-features.txt` diff gate in `api-surface.yml` with the same strictness (item counts carried over exactly: 173 default / 250 all-features) plus local auto-regen — intentional API changes now land the snapshot in the same commit as the code. The `semver-checks` job is unchanged. Dev-only — not part of the published package.

### Changed

- **Packaging: exclude `asm-snapshots/`, `api-snapshots/`, `scripts/`, `benchmarks/`, `docs/`, `tests/`, `benches/`, `perf.md`, `.gitignore` from published crate.** Reduces package from ~1.1 MB to 579 KiB (126 KiB compressed). `src/data/*.bin` lookup tables remain included (load-bearing `include_bytes!`). Removed `perf.md` from git (historical scratch, superseded by README).

- **Benchmarks: added `s2l_scalar_fast` / `l2s_scalar_fast` arms** to the
  `dispatch_overhead` group measuring the C0-continuous fast-poly scalar
  (`default::srgb_to_linear` / `linear_to_srgb`) per element. Used to falsify
  the proposed small-slice SIMD/scalar crossover fix on Neoverse-N1: the only
  correctness-safe scalar (the f64-intermediate rational poly) is slower than
  the SIMD slice at every size except N=8 (a sub-noise 0.6–1.6 ns gap), and
  LLVM already hoists the coefficient splats out of the chunk loop, so neither
  candidate fix produces a measurable win. No `src/` change. Full data:
  `benchmarks/arm_neoverse_n1_baseline_2026-05-31.md` §6.

### Fixed

- docs(readme): state f32 linear normalization range `[0, 1]` + u8 encoder
  rounding/round-trip exactness — found by insulated-developer test. The
  README documented the u16 encode precision but left the f32 linear range
  implied and the u8 encode behavior unstated; now the Type-conversions
  section notes the `[0.0, 1.0]` normalization (u8/u16 LUT decode + f32
  single-value API) and the u8 encode (rounds to nearest, `u8 → linear → u8`
  exact within ±1 level, unlike the bit-exact `linear_to_srgb_u16` path).

- **Slice tail used a different transfer curve than the SIMD body.** The
  remainder loops in `srgb_to_linear_slice` / `linear_to_srgb_slice` (and
  their RGBA variants) converted the trailing up-to-15 elements of any
  slice whose length isn't a multiple of 16 with the `powf`-based IEC
  curve (`scalar::srgb_to_linear` / `linear_to_srgb`), while the SIMD
  chunk loop used the C0-continuous rational polynomial. Different
  transfer function, different piecewise threshold — a cross-path pixel
  divergence on unaligned slices. The tail now calls
  `scalar::srgb_to_linear_fast` / `scalar::linear_to_srgb_fast` (the same
  polynomial), so the whole slice is on one curve. Also a measured
  2.3–2.7× speedup on the tail on Neoverse-N1 (e.g. a 15-element slice:
  188.8 → 69.5 ns/call); aligned-size throughput is unchanged. Benchmark
  provenance: `benchmarks/arm_neoverse_n1_baseline_2026-05-31.md`.

## [0.6.12] - 2026-04-25

Internally a major refactor of `src/simd.rs` (issue #23) that **does not
change the public API surface in any breaking way** — `cargo semver-checks`
reports 196/196 pass against 0.6.11. The only API diff vs 0.6.11 is
additive: 4 new `pub fn`s in `tokens::x8` (listed below). Plus several
SIMD perf improvements on existing public functions and the polynomial-
coefficient refit that landed between 0.6.10 and 0.6.11 but never made
the changelog.

> **Note on `linear_srgb::tf`:** the module remains `pub` for compatibility
> with 0.6.11 callers that import through it. New code should prefer
> `linear_srgb::default::*` — every scalar function in `tf` is also
> re-exported there with the same name. The `tf` path is documented as a
> backward-compat path; a future major release may make it `pub(crate)`.

### Added

- **Public API surface gate** via `cargo-public-api` snapshots in
  `api-snapshots/` (default + all-features) plus a `cargo semver-checks`
  job in `.github/workflows/api-surface.yml`. Locks the `pub` surface
  so internal refactors cannot introduce unintentional additions or
  removals without an explicit snapshot update (a48e84b).

- **Per-target ASM snapshot CI gate** in `.github/workflows/asm-snapshots.yml`
  + `scripts/dump-asm.sh` + `examples/asm-stub.rs`. Cross-compiles to
  `aarch64-unknown-linux-gnu` (NEON) and `wasm32-unknown-unknown`
  (SIMD128), dumps codegen for each public dispatcher via `cargo asm`,
  and `git diff --exit-code`s against committed baselines (0595cd9, bb64deb).

- **Cross-tier f32 RGBA consistency tests** (`tests/rgba_premultiply_consistency.rs`).
  Drives each tier permutation via `archmage::testing::for_each_token_permutation`
  and asserts all tiers produce equivalent f32 output within
  architecture-appropriate tolerance (FMA vs separate mul+add accounts
  for ~1 ULP at unit magnitude; bounds calibrated per family). Closes
  the gap where u8 RGBA cross-tier tests existed but f32 RGBA didn't,
  letting sub-LSB f32 drift escape detection (b131556).

- **Tango paired-benchmark regression gate** in `tango/`. Compares the
  local WIP against the published `=0.6.11` crate via `tango-bench`,
  interleaving both versions in the same process for paired statistics.
  Six TF slice dispatchers × four input distributions × three sizes
  (67329f3, b573290).

- **Alpha-preserving RGBA variants for BT.709 / PQ / HLG slices:**
  `bt709_to_linear_rgba_slice`, `linear_to_bt709_rgba_slice`,
  `pq_to_linear_rgba_slice`, `linear_to_pq_rgba_slice`,
  `hlg_to_linear_rgba_slice`, `linear_to_hlg_rgba_slice`. Applies the TF
  to every RGB lane while leaving alpha bit-identical (e4685e8).

- **`tokens::x8` u16 polynomial rites** are now `pub`:
  `srgb_u16_to_linear_v3`, `srgb_u16_to_linear_scalar`,
  `linear_to_srgb_u16_v3`, `linear_to_srgb_u16_scalar`. Lets downstream
  pipelines invoke u16↔linear inside their own `#[arcane]` blocks
  without going through the slice dispatcher (7ab2e61, closes #20).
  Closes #18 (HLG regression on small slices).

- **Fitter script `scripts/fit_srgb_fast.py`** committed with the inputs
  used to produce the current rational-polynomial coefficients (degrees,
  domain, weights, restarts). Reproducible from a clean checkout:
  `python scripts/fit_srgb_fast.py` (04f0e1f).

### Changed

- **Internal dedup of `src/simd.rs` via archmage 0.9.22 magetypes flags**
  (issue #23). simd.rs went from 4153 → 3472 lines (-681, -16%). Public
  API unchanged.
  - Pattern 3 — `define(f32x16)`: 18 manual
    `type f32x16 = g_f32x16<Token>;` boilerplate sites collapsed into the
    new `define(...)` flag inside the existing `#[magetypes(...)]`
    attributes (0f8f944).
  - Pattern 1 — `rite, define`: 8 hand-written `#[rite]` helpers
    (`srgb_to_linear_mt`/`_x16`, `linear_to_srgb_mt`/`_x16`,
    `gamma_to_linear_mt`/`_x16_2x8`, `linear_to_gamma_mt`/`_x16_2x8`)
    collapsed into 4 unified `#[magetypes(rite, define(f32x16), ...)]`
    functions. The `pow_midp` polyfill on `f32x16<X64V4Token>` eliminates
    the `token.v3()` 2×x8 split helpers (82433fd, eb25e7b cross-arch fix).
  - Pattern 2 — family-level magetypes: 8 dispatcher families collapsed
    from 5 hand-written tier dispatchers each (V3, V4, NEON, WASM, scalar)
    into single `#[magetypes(...)]` bodies. NEON / WASM outer loop changes
    from 4-wide → 16-wide via `f32x16` polyfill = 4× f32x4; same per-pixel
    SIMD ops, expected unrolling difference verified by ASM snapshots
    (1b2f936).

- **`linear_to_srgb_u8_slice` / `_rgba` / `linear_to_srgb_u16_slice` / `_rgba`
  now SIMD-dispatched.** Previously scalar loops that serialized on
  `cvtss2si` → LUT load or on the per-pixel polynomial. Now dispatch
  through `incant!` over `[v4, v3, neon, wasm128, scalar]`, evaluating
  the polynomial / LUT index 4/8/16-wide per chunk. On Ryzen 7950X,
  N=10000 elements (9a2dcc3):
  - `linear_to_srgb_u8_slice`: 10.5µs → 2.9µs (**3.6×**)
  - `linear_to_srgb_u16_slice`: 117µs → 2.3µs (**~51×**)
  - `linear_to_srgb_u8_rgba_slice`: 8.3µs → 3.6µs (**2.3×**)
  - `linear_to_srgb_u16_rgba_slice`: 86µs → 5.6µs (**~15×**)

  u8 paths use the 4096-entry LUT with SIMD-computed indices (bit-exact
  across tiers). u16 paths evaluate the rational polynomial in SIMD then
  quantize — cross-tier output may differ by ±1 u16 LSB at polynomial
  boundaries. Alpha lanes in the RGBA variants remain bit-exact.

- **Base 4/4 scalar rational polynomial coefficients refit** via polyfit
  (Sanathanan-Koerner + Levenberg-Marquardt with Nielsen damping, 8 restarts,
  f32 ULP local search). Exhaustive sweep over all 1.07B f32 values in
  `[0, 1]` (ebbfc9c):
  - `srgb_to_linear` fast: 11 → **8** ULP max (−27%)
  - `linear_to_srgb` fast: 14 → **10** ULP max (−29%)
  - `fast vs precise linear_to_srgb`: 16 → **12** ULP max (−25%)
  - `fast vs precise srgb_to_linear`: 12 → 12 ULP max (unchanged)

- **`archmage` 0.9.19 → 0.9.22 / `magetypes` 0.9.21 → 0.9.22** to access
  the new `magetypes(rite, define(...))` flags used by the issue #23
  refactor (b052f87).

- **Refactor commits collapsing per-tier slice wrappers via `#[magetypes]`**
  (no behavior change, structural cleanup) — gamma slice tier wrappers
  (b399c18), tokens/x4.rs core helpers (736b5de), sRGB extended-range
  slice tier wrappers (ca95391), BT.709/PQ/HLG slice tier wrappers
  (ce9b0ce).

### Fixed

- **BT.709 / PQ / HLG slice APIs now SIMD-dispatched.** The public
  `bt709_to_linear_slice`, `linear_to_bt709_slice`, `pq_to_linear_slice`,
  `linear_to_pq_slice`, `hlg_to_linear_slice`, and `linear_to_hlg_slice`
  functions were scalar loops with only `#[autoversion]`, so HDR/video
  callers never reached the AVX-512 / AVX2+FMA / NEON / WASM SIMD128
  rites in `tokens::{x4,x8,x16}` that already existed. They now dispatch
  through `incant!` over `[v4, v3, neon, wasm128, scalar]` like
  `srgb_to_linear_slice`. Closes #10 (54e03b4).

- **Alpha threshold for unpremultiply** moved from a stricter cutoff to
  the spec'd 1/1024 (74a42a8 — actually shipped in 0.6.11 but not noted
  there).

### Tradeoffs (honest)

- **Roundtrip absolute error grew** in a narrow region from the polynomial
  refit: fwd 4.17e-7 → 6.56e-7 (+57%), inverse 1.01e-6 → 1.49e-6 (+47%).
  Still well under 1 u16 step (1.53e-5). No u16 roundtrip regression: 0
  values round-trip to a different u16, same as before.
- **L2S piecewise threshold gap widened** 1 → 3 ULP (under the 4 ULP test
  tolerance, but the margin is narrower).
- **30,273 of 65,536 u16 inputs** produce different f32 linear values vs
  0.6.10. Any caller with baked test-vector hashes will see failures.
  Maximum absolute change: 4.99e-7 (well below u16 LSB).
- **NEON / WASM outer-loop unrolling** changed from 4-wide to 16-wide via
  `f32x16` polyfill. Same per-pixel SIMD instruction sequences; ASM-diff
  CI gate verified equivalence (no `panic_bounds_check`, no scalar fallback,
  same `fmla`/`fmin`/`fmax`/`fcmgt`/`bsl` ops on aarch64; same `f32x4.*`
  on wasm32). Bench verification on aarch64 / wasm32 runtime lives in CI.

## 0.6.10

Also published as 0.7.0 (unnecessarily bumped — no API was broken).

### Fixed

- **`srgb_to_linear_extended` / `linear_to_srgb_extended` now use sign-preserving
  semantics** per CSS Color 4: `sign(v) * f(|v|)`. Previously, negatives passed
  through the linear segment only, giving wrong results for out-of-gamut values
  from 3×3 gamut matrix conversions (e.g., -0.5 mapped to -0.039 instead of
  -0.214). The old behavior was a bug.

### Added

- **`srgb_to_linear_extended_slice` / `linear_to_srgb_extended_slice`** — SIMD
  extended-range conversion for cross-gamut pipelines (P3→sRGB, BT.2020→sRGB).
  Uses 6/6 rational polynomials fitted to wider domains via abs+sign, dispatching
  to AVX2+FMA (8-wide), NEON (4-wide), WASM SIMD128 (4-wide), or scalar.

- **6/6 extended-range polynomial coefficients** in `rational_poly.rs`:
  - S2L fitted to [0, 8]: 8 ULP max in [0,1] via SIMD FMA, u8-safe to 8×, u16-safe to ~4.2×
  - L2S fitted on √x to [0, 64]: 8 ULP max in [0,1], u16-safe across full domain

- Token rites: `srgb_to_linear_extended_v3` / `linear_to_srgb_extended_v3` in
  `tokens::x4` (via `#[magetypes]` generics) and `tokens::x8`.

- `extended_simd_doc_accuracy_claims` test: exhaustive ~1B-value sweep via SIMD
  dispatch, pinning all README/doc accuracy claims to measured values.

### Changed

- `archmage` / `magetypes` updated to 0.9.19.

### Dependencies

- `archmage`: 0.9.15 → 0.9.19
- `magetypes`: 0.9.15 → 0.9.19

## 0.6.9

No public API changes.

### Changed

- **Const LUT tables replaced with binary blobs.** The three embedded lookup
  tables (21 KB total) are now stored as raw little-endian bytes loaded via
  `include_bytes!` + `bytemuck::cast_slice`, replacing 4,581 lines of float/u8
  literals. A `#[repr(C, align(4))]` wrapper guarantees f32 alignment; LLVM
  optimizes the cast to a direct static address load (zero runtime cost,
  verified via `cargo asm`). Bit-exact with the previous const arrays.

### Added

- `compile_error!` on big-endian targets — the binary LUT blobs are
  little-endian and would silently produce wrong values without byte-swapping.
- Three new tests verifying binary blobs match runtime-computed tables
  bit-for-bit.

## Upcoming breaking changes

- **`avx512` will be removed from default features in a future 0.x release.**
  The AVX-512 code paths (16-wide f32, `tokens::x16`, `X64V4Token` dispatch)
  add ~175ms to cold compile time. Most consumers don't benefit — the AVX2
  (8-wide) path is already fast and available on far more hardware. If you use
  `tokens::x16` or rely on AVX-512 dispatch, add `features = ["avx512"]`
  explicitly to avoid breakage when the default changes. The `incant!` dispatch
  in slice functions automatically falls through to the AVX2 tier when AVX-512
  is not compiled in, so most users won't notice any difference.

## 0.6.5

u16 LUT overhaul: zero binary bloat, sqrt-indexed encode, two-tier encode API.

### Breaking (behavioral)

- **`linear_to_srgb_u16()` is now polynomial (was uniform LUT).** Perfect
  roundtrip but ~10× slower than the previous LUT path. Callers who need
  the old speed should switch to `linear_to_srgb_u16_fast()`.
- **u16 LUTs are now lazily initialized** via `OnceLock` instead of compiled
  into the binary. First call to any u16 function pays ~200µs init.

### Added

- **`linear_to_srgb_u16_fast()`** — sqrt-indexed LUT encode. 10× faster than
  polynomial, max ±1 u16 roundtrip error, 94.2% exact. The sqrt indexing
  concentrates resolution where the sRGB curve is steepest (near black).
- **`linear_to_srgb_u16_slice_fast()`**, **`linear_to_srgb_u16_rgba_slice_fast()`**
  — slice variants of the fast encode path.
- Systematic length-variant tests for all public slice functions at 12 element
  counts (1–100) exercising scalar, AVX2, and AVX-512 SIMD boundaries.
- CI: code coverage via `cargo-llvm-cov` → Codecov.
- README: CI, codecov, and MSRV badges.
- Examples: `roundtrip_matrix`, `encode_lut_strategies`, `encode_perf`,
  `lut_init_time`, `u16_roundtrip_audit`.

### Changed

- **Deleted 1.8MB `const_luts_u16.rs`** (70,570 lines). u16 LUTs are now
  generated at runtime via `OnceLock` using SIMD-accelerated slice functions
  in L1-sized chunks. ~200µs init, zero binary bloat, only allocated if u16
  API is called.
- Decode LUT (`srgb_u16_to_linear`) unchanged in behavior — still a direct
  65536-entry lookup, now lazily initialized instead of const.
- `no_std`: u16 functions fall back to rational polynomial (no LUT, no heap).
- Updated all doc comments referencing "const LUT" or "f64 powf".
- Tightened u16 roundtrip test tolerance.

### Accuracy

| Path | vs f64 reference | Roundtrip |
|------|-----------------|-----------|
| `srgb_u16_to_linear` (LUT decode) | ≤12 ULP | exact input |
| `linear_to_srgb_u16` (polynomial) | ≤14 ULP | 100% exact |
| `linear_to_srgb_u16_fast` (sqrt LUT) | ≤14 ULP | 94.2% exact, max ±1 |

### Closed

- **Issue #3**: u16 LUT design resolved — OnceLock lazy init, sqrt-indexed
  encode, no feature gate needed.

## 0.6.4

SIMD premultiply fused into single-pass, PQ/HLG slice operations added.

### Deprecated

- `gamma_to_linear_premultiply_rgba_slice` — use `srgb_to_linear_premultiply_rgba_slice` instead.
- `unpremultiply_linear_to_gamma_rgba_slice` — use `unpremultiply_linear_to_srgb_rgba_slice` instead.

These gamma-based premultiply functions are retained for backward compatibility
but will be removed in a future release.

### Changed

- **`srgb_to_linear_premultiply_rgba_slice` is now truly single-pass SIMD.**
  Previously two passes (sRGB→linear, then premultiply loop). Now fused into
  one memory traversal with dedicated AVX-512 (4 px), AVX2+FMA (2 px), and
  scalar tiers.
- **`unpremultiply_linear_to_srgb_rgba_slice` is now truly single-pass SIMD.**
  Previously unpremultiplied in a scalar loop then called
  `linear_to_srgb_rgba_slice`. Now fused with the same three-tier dispatch.
- All `incant!` dispatch calls now include `scalar` in tier lists, fixing
  deprecation warnings from archmage 0.9.12.
- `gamma_to_linear_premultiply_rgba_slice` and
  `unpremultiply_linear_to_gamma_rgba_slice` retained for backwards
  compatibility with 0.6.x.

### Added

- **`default::hlg_to_linear_slice`** — HLG signal f32 → linear in-place.
  Requires `transfer` feature.
- **`default::linear_to_hlg_slice`** — linear → HLG signal in-place.
  Requires `transfer` feature.
- **`default::pq_to_linear_slice`** — PQ (ST 2084) signal f32 → linear
  in-place. Requires `transfer` feature.
- **`default::linear_to_pq_slice`** — linear → PQ signal in-place.
  Requires `transfer` feature.
- CI: MSRV verification job via `cargo hack check --rust-version`.

### Removed

- 24 internal per-tier test wrappers (`tokens/x8.rs`, `tokens/x16.rs`,
  `lut.rs`). Redundant with public dispatch-layer tests in `simd.rs`.

### Dependencies

- `archmage`: 0.9.5 → 0.9.12
- `magetypes`: 0.9.5 → 0.9.12

### Tests

200+ tests passing. Net reduction of ~24 test functions from
internal tier-wrapper cleanup; public API coverage unchanged.

## 0.6.3

### Added

- **Fused premultiply/unpremultiply SIMD** — `srgb_to_linear_premultiply_rgba_slice` and
  `unpremultiply_linear_to_srgb_rgba_slice` now run in a single SIMD pass (conversion + alpha
  multiply in one memory traversal). AVX-512 (16-wide), AVX2 (8-wide), and scalar fallback.
- **u8 premultiply round-trips** — `srgb_u8_to_linear_premultiply_rgba_slice` converts u8 sRGB
  straight-alpha directly to linear premultiplied f32. `unpremultiply_linear_to_srgb_u8_rgba_slice`
  converts back.
- **Custom gamma premultiply/unpremultiply** — `gamma_to_linear_premultiply_rgba_slice` and
  `unpremultiply_linear_to_gamma_rgba_slice` for arbitrary gamma (2.2, 1.8, etc.) with the same
  fused single-pass SIMD treatment.
- Benchmark suite for RGBA premultiply approaches (`rgba_approach`).

### Changed

- Updated `archmage` and `magetypes` dependencies to 0.9.4.

## 0.6.2

### Changed

- `unsafe_simd` feature is now a no-op — all paths use safe Rust. Feature kept for backward
  compatibility; will be removed in 0.7.
- Removed all `unsafe` code. The crate is now `#![forbid(unsafe_code)]`.

### Fixed

- Eliminated sRGB piecewise discontinuity in fast/SIMD paths.
- Eliminated monotonicity violations in scalar rational polynomial.
- Regenerated all LUTs with C0-continuous constants.
- Two-range PQ EOTF for sub-U16 roundtrip accuracy (with `transfer` feature).

## 0.6.1

### Fixed

- `no_std` compatibility: added missing `num_traits::Float` imports.
- Feature-combination CI coverage.

## 0.6.0

Initial public release with rational polynomial sRGB conversion, SIMD dispatch via archmage,
LUT-based u8/u16 paths, precise powf paths, extended-range support, and transfer functions
(BT.709, PQ, HLG).
