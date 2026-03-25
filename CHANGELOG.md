# Changelog

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
