# linear-srgb

Fast, SIMD-accelerated sRGB↔linear conversion for image processing pipelines.

Handles `f32`, `f64`, `u8`, and `u16` data. Supports in-place RGBA with alpha
preservation, fused premultiply/unpremultiply, custom gamma, and extended range.
`no_std` compatible.

[![Crates.io](https://img.shields.io/crates/v/linear-srgb?style=for-the-badge)](https://crates.io/crates/linear-srgb)
[![docs.rs](https://img.shields.io/docsrs/linear-srgb?style=for-the-badge)](https://docs.rs/linear-srgb)
[![CI](https://img.shields.io/github/actions/workflow/status/imazen/linear-srgb/ci.yml?branch=main&style=for-the-badge&label=CI)](https://github.com/imazen/linear-srgb/actions/workflows/ci.yml)
[![codecov](https://img.shields.io/codecov/c/github/imazen/linear-srgb?style=for-the-badge)](https://codecov.io/gh/imazen/linear-srgb)
[![License](https://img.shields.io/crates/l/linear-srgb?style=for-the-badge)](https://github.com/imazen/linear-srgb#license)
[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=for-the-badge)](https://github.com/imazen/linear-srgb)

```toml
[dependencies]
linear-srgb = "0.6"

# Optional: BT.709, PQ (HDR10), and HLG transfer functions
linear-srgb = { version = "0.6", features = ["transfer"] }
```

## Quick Start

Use the slice functions. They're SIMD-accelerated (AVX-512, AVX2, SSE4.1, NEON,
WASM SIMD128) with automatic runtime CPU dispatch — typically 4–16x faster than
scalar loops.

```rust
use linear_srgb::default::*;

// f32 slices (in-place, SIMD-accelerated)
let mut values = vec![0.5f32; 10000];
srgb_to_linear_slice(&mut values);
linear_to_srgb_slice(&mut values);
```

For RGBA data, use the `_rgba_` variants — they convert only the RGB channels
and leave alpha untouched. This matters: alpha is linear by definition, so
applying the sRGB transfer function to it is a bug.

```rust
use linear_srgb::default::*;

// RGBA f32 — alpha channel preserved, only RGB converted
let mut rgba = vec![0.5f32, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
srgb_to_linear_rgba_slice(&mut rgba);
assert_eq!(rgba[3], 0.75); // alpha untouched
```

### Type conversions

Convert directly between integer sRGB and linear f32 without intermediate steps.

```rust
use linear_srgb::default::*;

// u8 sRGB → linear f32 (LUT-based, extremely fast)
let srgb_bytes: Vec<u8> = vec![128u8; 1024];
let mut linear = vec![0.0f32; 1024];
srgb_u8_to_linear_slice(&srgb_bytes, &mut linear);

// linear f32 → sRGB u8 (SIMD-accelerated)
let mut srgb_out = vec![0u8; 1024];
linear_to_srgb_u8_slice(&linear, &mut srgb_out);

// RGBA u8 → linear f32 (alpha passed through as a/255, not sRGB-decoded)
let rgba_bytes = vec![128u8, 128, 128, 200, 64, 64, 64, 128];
let mut rgba_linear = vec![0.0f32; 8];
srgb_u8_to_linear_rgba_slice(&rgba_bytes, &mut rgba_linear);

// u16 support too — decode via 65536-entry LUT (30-40× faster than polynomial)
let mut u16_linear = vec![0.0f32; 256];
let srgb_u16: Vec<u16> = (0..256).map(|i| (i * 256) as u16).collect();
srgb_u16_to_linear_slice(&srgb_u16, &mut u16_linear);
```

### u16 encode: exact vs fast

Two paths for linear f32 → sRGB u16, depending on whether you need perfect
roundtrip or maximum throughput:

```rust
use linear_srgb::default::*;

let linear = srgb_u16_to_linear(32768); // LUT decode (always fast, exact)

// Exact roundtrip (polynomial, ~89 Mops/s)
let exact = linear_to_srgb_u16(linear);

// Fast encode (sqrt-indexed LUT, ~609 Mops/s, max ±1 level)
let fast = linear_to_srgb_u16_fast(linear);
```

Slice variants: `linear_to_srgb_u16_slice` / `linear_to_srgb_u16_slice_fast`,
`linear_to_srgb_u16_rgba_slice` / `linear_to_srgb_u16_rgba_slice_fast`.

### Premultiplied alpha (fused, single-pass)

Convert between sRGB straight-alpha and linear premultiplied alpha in one SIMD
pass — no intermediate buffer, no second memory traversal.

```rust
use linear_srgb::default::*;

// sRGB straight → linear premultiplied (f32 in-place)
let mut rgba = vec![0.8f32, 0.5, 0.2, 0.75, 1.0, 1.0, 1.0, 1.0];
srgb_to_linear_premultiply_rgba_slice(&mut rgba);

// linear premultiplied → sRGB straight (f32 in-place)
unpremultiply_linear_to_srgb_rgba_slice(&mut rgba);

// Also available as u8→f32 and f32→u8:
// srgb_u8_to_linear_premultiply_rgba_slice(&srgb_bytes, &mut linear_premul);
// unpremultiply_linear_to_srgb_u8_rgba_slice(&linear_premul, &mut srgb_out);
```

### Single values

When you only need one value at a time (not a batch):

```rust
use linear_srgb::default::*;

// f32 — rational polynomial (≤14 ULP max, perfectly monotonic)
let linear = srgb_to_linear(0.5f32);
let srgb = linear_to_srgb(linear);

// u8 — LUT-based, zero math
let linear = srgb_u8_to_linear(128u8);
let srgb_byte = linear_to_srgb_u8(linear);

// u16 — LUT-based
let linear = srgb_u16_to_linear(32768u16);
let srgb_u16 = linear_to_srgb_u16(linear);
```

### Precise (powf) conversions

For maximum accuracy or extended-range (HDR/ICC) workflows:

```rust
use linear_srgb::precise::*;

// f32 — exact powf, C0-continuous (6 ULP max)
let linear = srgb_to_linear(0.5f32);
let srgb = linear_to_srgb(0.214f32);

// f64 high-precision
let linear = srgb_to_linear_f64(0.5f64);

// Extended range — no clamping, for cross-gamut / scRGB pipelines
let linear = srgb_to_linear_extended(-0.1);
let srgb = linear_to_srgb_extended(1.5);
```

### Custom gamma

For pure power-law gamma (no linear toe segment) — gamma 2.2, 1.8, etc.:

```rust
use linear_srgb::default::*;

let linear = gamma_to_linear(0.5f32, 2.2);
let encoded = linear_to_gamma(linear, 2.2);

// SIMD-accelerated slices
let mut values = vec![0.5f32; 1000];
gamma_to_linear_slice(&mut values, 2.2);

// Fused premultiply/unpremultiply also available:
// gamma_to_linear_premultiply_rgba_slice(&mut rgba, 2.2);
// unpremultiply_linear_to_gamma_rgba_slice(&mut rgba, 2.2);
```

### HDR transfer functions (`transfer` feature)

BT.709, PQ (ST 2084 / HDR10), and HLG (ARIB STD-B67) — scalar and SIMD.

```toml
linear-srgb = { version = "0.6", features = ["transfer"] }
```

```rust,ignore
use linear_srgb::default::*;

let linear = pq_to_linear(0.5);       // PQ (HDR10) → linear
let pq = linear_to_pq(linear);

let linear = hlg_to_linear(0.5);      // HLG → linear
let linear = bt709_to_linear(0.5);    // BT.709 → linear
```

### LUT for custom bit depths

```rust
use linear_srgb::lut::{LinearTable16, EncodingTable16, lut_interp_linear_float};

// 16-bit linearization (65536 entries)
let lut = LinearTable16::new();
let linear = lut.lookup(32768);

// Interpolated encoding
let encode_lut = EncodingTable16::new();
let srgb = lut_interp_linear_float(0.5, encode_lut.as_slice());
```

## API Summary

| Data | Function |
|------|----------|
| `&mut [f32]` | `srgb_to_linear_slice` / `linear_to_srgb_slice` |
| RGBA `&mut [f32]` | `srgb_to_linear_rgba_slice` / `linear_to_srgb_rgba_slice` |
| RGBA f32 premultiply | `srgb_to_linear_premultiply_rgba_slice` / `unpremultiply_linear_to_srgb_rgba_slice` |
| `&[u8]` ↔ `&mut [f32]` | `srgb_u8_to_linear_slice` / `linear_to_srgb_u8_slice` |
| RGBA `&[u8]` ↔ `&mut [f32]` | `srgb_u8_to_linear_rgba_slice` / `linear_to_srgb_u8_rgba_slice` |
| RGBA u8↔f32 premultiply | `srgb_u8_to_linear_premultiply_rgba_slice` / `unpremultiply_linear_to_srgb_u8_rgba_slice` |
| `&[u16]` ↔ `&mut [f32]` | `srgb_u16_to_linear_slice` / `linear_to_srgb_u16_slice` |
| Custom gamma `&mut [f32]` | `gamma_to_linear_slice` / `linear_to_gamma_slice` |
| Custom gamma RGBA premul | `gamma_to_linear_premultiply_rgba_slice` / `unpremultiply_linear_to_gamma_rgba_slice` |
| Single f32 | `srgb_to_linear` / `linear_to_srgb` |
| Single u8 | `srgb_u8_to_linear` / `linear_to_srgb_u8` |
| Single u16 | `srgb_u16_to_linear` / `linear_to_srgb_u16` |
| Exact powf f32/f64 | `precise::srgb_to_linear` / `precise::linear_to_srgb` |
| Extended range f32 | `precise::srgb_to_linear_extended` / `precise::linear_to_srgb_extended` |

All functions live in `linear_srgb::default` unless noted.

## Accuracy

### Transfer function constants

All code paths use C0-continuous constants derived from the
[moxcms](https://github.com/niclasberg/moxcms) reference implementation. These
adjust the IEC 61966-2-1 offset from 0.055 to 0.055011 and the threshold from
0.04045 to 0.03929, making the piecewise transfer function mathematically
continuous (~2.3e-9 gap eliminated).

At u8 precision the two constant sets produce identical values. At u16, the max
difference is ~1 LSB near the threshold. See [docs/iec.md](docs/iec.md) for a
detailed comparison.

For interop with software that uses the original IEC textbook constants, enable
the `iec` feature for `linear_srgb::iec::srgb_to_linear` /
`linear_srgb::iec::linear_to_srgb`.

### Accuracy summary (exhaustive f32 sweep)

| Path | Max ULP | Avg ULP | Monotonic |
|------|---------|---------|-----------|
| `default` s→l (rational poly) | 11 | ~0.5 | yes |
| `default` l→s (rational poly) | 14 | ~0.4 | yes |
| `precise` s→l (powf) | 6 | ~0.1 | yes |
| `precise` l→s (powf) | 3 | ~0.1 | yes |

**What does 14 ULP mean in practice?** 1 ULP (unit in the last place) is the
spacing between adjacent f32 values at a given magnitude. At 0.5 that's ~6e-8,
so 14 ULP ≈ 8e-7 — about 6 decimal digits of precision. At 0.01 it's ~1e-8.
For any 8-bit or 16-bit output, this error is invisible — it's thousands of
times smaller than one output level.

Reference: C0-continuous f64 powf. The scalar rational polynomial evaluates in
f64 intermediate precision, guaranteeing perfect monotonicity (zero reversals
across all ~1B f32 values in [0, 1]). SIMD paths use f32 evaluation for
throughput and are also monotonic within each segment.

## Feature Flags

- **`std`** (default): Required for runtime SIMD dispatch
- **`avx512`** (default): AVX-512 code paths (16-wide f32)
- **`transfer`**: BT.709, PQ, HLG transfer functions (scalar + SIMD)
- **`iec`**: IEC 61966-2-1 textbook sRGB functions for legacy interop
- **`alt`**: Alternative/experimental implementations for benchmarking

```toml
# no_std (requires alloc for LUT generation)
linear-srgb = { version = "0.6", default-features = false }
```

## Module Organization

- **`default`** — Recommended API. Rational polynomial for f32, LUT for integers, SIMD for slices.
- **`precise`** — Exact `powf()` conversions with C0-continuous constants (not IEC textbook). f32/f64, extended range.
- **`lut`** — Lookup tables for custom bit depths (10-bit, 12-bit, 16-bit).
- **`tf`** — Transfer functions: BT.709, PQ, HLG. Requires `transfer` feature.
- **`iec`** — IEC 61966-2-1 textbook constants for legacy interop. Requires `iec` feature.
- **`tokens`** — Inlineable `#[rite]` functions for embedding in SIMD pipelines (see below).

## Embedding in SIMD Pipelines (`tokens` module)

If you're writing your own SIMD code with [archmage](https://crates.io/crates/archmage),
the `tokens` module provides `#[rite]` functions that inline directly into your
`#[arcane]` functions — zero dispatch overhead.

```rust,ignore
use linear_srgb::tokens::x8;
use archmage::arcane;

#[arcane]
fn my_pipeline(token: X64V3Token, data: &mut [f32]) {
    // x8::srgb_to_linear_v3 is #[rite] — inlines into your function
    // Available widths: x4 (SSE/NEON/WASM), x8 (AVX2), x16 (AVX-512)
}
```

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with Claude (Anthropic). All code has been reviewed and benchmarked, but verify critical paths for your use case.
