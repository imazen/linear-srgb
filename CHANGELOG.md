# Changelog

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
