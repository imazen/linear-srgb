# linear-srgb

Fast linear↔sRGB color space conversion with runtime CPU dispatch.

[![Crates.io](https://img.shields.io/crates/v/linear-srgb.svg)](https://crates.io/crates/linear-srgb)
[![Docs.rs](https://docs.rs/linear-srgb/badge.svg)](https://docs.rs/linear-srgb)
[![License](https://img.shields.io/crates/l/linear-srgb.svg)](LICENSE)

## Quick Start

```rust
use linear_srgb::default::*;

// Single values (rational polynomial — fast, ≤14 ULP, perfectly monotonic)
let linear = srgb_to_linear(0.5f32);
let srgb = linear_to_srgb(linear);

// Slices (SIMD-accelerated)
let mut values = vec![0.5f32; 10000];
srgb_to_linear_slice(&mut values);
linear_to_srgb_slice(&mut values);

// u8 ↔ f32 (image processing)
let linear = srgb_u8_to_linear(128);
let srgb_byte = linear_to_srgb_u8(linear);
```

## Which Function Should I Use?

| Your situation | Use this |
|----------------|----------|
| One f32 value (fast) | `default::srgb_to_linear(x)` / `default::linear_to_srgb(x)` |
| One f32 value (exact) | `precise::srgb_to_linear(x)` / `precise::linear_to_srgb(x)` |
| One u8 value | `default::srgb_u8_to_linear(x)` (LUT, fastest) |
| `&mut [f32]` slice | `default::srgb_to_linear_slice()` / `default::linear_to_srgb_slice()` |
| RGBA `&mut [f32]` (keep alpha) | `default::srgb_to_linear_rgba_slice()` / `default::linear_to_srgb_rgba_slice()` |
| RGBA sRGB → linear premultiplied | `default::srgb_to_linear_premultiply_rgba_slice()` |
| RGBA linear premultiplied → sRGB | `default::unpremultiply_linear_to_srgb_rgba_slice()` |
| `&[u8]` → `&mut [f32]` | `default::srgb_u8_to_linear_slice()` |
| RGBA `&[u8]` → `&mut [f32]` | `default::srgb_u8_to_linear_rgba_slice()` / `linear_to_srgb_u8_rgba_slice()` |
| RGBA u8 sRGB → linear premul f32 | `default::srgb_u8_to_linear_premultiply_rgba_slice()` |
| RGBA f32 linear premul → sRGB u8 | `default::unpremultiply_linear_to_srgb_u8_rgba_slice()` |
| `&[u16]` ↔ `&mut [f32]` | `default::srgb_u16_to_linear_slice()` / `default::linear_to_srgb_u16_slice()` |
| `&[f32]` → `&mut [u8]` | `default::linear_to_srgb_u8_slice()` |
| Inside `#[arcane]` fn | `tokens::x8::srgb_to_linear_v3()` (inlines, no dispatch) |

## API Reference

### Single Values

```rust
use linear_srgb::default::*;

// f32 conversions — rational polynomial (≤14 ULP max, perfectly monotonic)
let linear = srgb_to_linear(0.5f32);
let srgb = linear_to_srgb(0.214f32);

// u8 conversions (LUT-based, zero math)
let linear = srgb_u8_to_linear(128u8);
let srgb_byte = linear_to_srgb_u8(0.214f32);

// u16 conversions (LUT-based)
let linear = srgb_u16_to_linear(32768u16);
let srgb_u16 = linear_to_srgb_u16(0.214f32);
```

### Precise (powf) Conversions

Uses C0-continuous constants that eliminate the IEC spec's piecewise discontinuity.
See the [Accuracy](#accuracy) section for details on how these differ from the IEC textbook values.

```rust
use linear_srgb::precise::*;

// f32 — exact powf, C0-continuous (6 ULP max)
let linear = srgb_to_linear(0.5f32);
let srgb = linear_to_srgb(0.214f32);

// f64 high-precision
let linear = srgb_to_linear_f64(0.5f64);

// Extended range (HDR/ICC — no clamping)
use linear_srgb::precise::{srgb_to_linear_extended, linear_to_srgb_extended};
let linear = srgb_to_linear_extended(-0.1);
let srgb = linear_to_srgb_extended(1.5);
```

### Slice Processing (Recommended for Batches)

```rust
use linear_srgb::default::*;

// In-place f32 conversion (SIMD-accelerated)
let mut values = vec![0.5f32; 10000];
srgb_to_linear_slice(&mut values);
linear_to_srgb_slice(&mut values);

// RGBA slices — alpha channel is preserved, only RGB converted
let mut rgba = vec![0.5f32, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
srgb_to_linear_rgba_slice(&mut rgba);
assert_eq!(rgba[3], 0.75); // alpha untouched

// u8 → f32 (LUT-based, extremely fast)
let srgb_bytes: Vec<u8> = (0..=255).collect();
let mut linear = vec![0.0f32; 256];
srgb_u8_to_linear_slice(&srgb_bytes, &mut linear);

// RGBA u8 → f32 (alpha passed through as a/255, not sRGB-decoded)
let rgba_bytes = vec![128u8, 128, 128, 200, 64, 64, 64, 128];
let mut rgba_linear = vec![0.0f32; 8];
srgb_u8_to_linear_rgba_slice(&rgba_bytes, &mut rgba_linear);

// f32 → u8 (SIMD-accelerated)
let linear_values: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
let mut srgb_bytes = vec![0u8; 256];
linear_to_srgb_u8_slice(&linear_values, &mut srgb_bytes);
```

### Premultiplied Alpha (Fused, Single-Pass)

Convert between sRGB straight-alpha and linear premultiplied alpha in one
SIMD pass — no intermediate buffer, no second memory traversal.

```rust
use linear_srgb::default::*;

// sRGB straight → linear premultiplied (f32 in-place)
let mut rgba = vec![0.8f32, 0.5, 0.2, 0.75, 1.0, 1.0, 1.0, 1.0];
srgb_to_linear_premultiply_rgba_slice(&mut rgba);

// linear premultiplied → sRGB straight (f32 in-place)
unpremultiply_linear_to_srgb_rgba_slice(&mut rgba);

// u8 sRGB → linear premultiplied f32
let srgb_bytes = vec![200u8, 128, 64, 192, 255, 255, 255, 255];
let mut linear_premul = vec![0.0f32; 8];
srgb_u8_to_linear_premultiply_rgba_slice(&srgb_bytes, &mut linear_premul);

// linear premultiplied f32 → sRGB u8
let mut srgb_out = vec![0u8; 8];
unpremultiply_linear_to_srgb_u8_rgba_slice(&linear_premul, &mut srgb_out);
```

### Custom Gamma (Non-sRGB)

For pure power-law gamma without the sRGB linear segment:

```rust
use linear_srgb::default::*;

// gamma 2.2 (common in legacy workflows)
let linear = gamma_to_linear(0.5f32, 2.2);
let encoded = linear_to_gamma(linear, 2.2);

// Also available for slices
let mut values = vec![0.5f32; 1000];
gamma_to_linear_slice(&mut values, 2.2);
```

### LUT for Custom Bit Depths

```rust
use linear_srgb::lut::{LinearTable16, EncodingTable16, lut_interp_linear_float};

// 16-bit linearization (65536 entries)
let lut = LinearTable16::new();
let linear = lut.lookup(32768);

// Interpolated encoding
let encode_lut = EncodingTable16::new();
let srgb = lut_interp_linear_float(0.5, encode_lut.as_slice());
```

### Advanced: Token-Based `#[rite]` Functions

For zero-overhead SIMD when embedding inside your own `#[arcane]` code:

```rust,ignore
use linear_srgb::tokens::x8;
use archmage::arcane;

#[arcane]
fn my_pipeline(token: X64V3Token, data: &mut [f32]) {
    // x8::srgb_to_linear_v3 is #[rite] — inlines into your function
    // Available widths: x4 (SSE/NEON/WASM), x8 (AVX2), x16 (AVX-512)
}
```

## Module Organization

- **`default`** — Recommended API. Rational polynomial for f32, LUT for integers, SIMD for slices.
- **`precise`** — Exact `powf()` conversions with C0-continuous constants (not IEC textbook). f32/f64, extended range.
- **`tokens`** — Inlineable `#[rite]` functions for x4/x8/x16 widths. For use inside `#[arcane]` code.
- **`lut`** — Lookup tables for custom bit depths.
- **`tf`** — Transfer functions: BT.709, PQ, HLG (feature-gated behind `transfer`).
- **`iec`** — IEC 61966-2-1 textbook constants for legacy interop (feature-gated).

## Feature Flags

```toml
[dependencies]
linear-srgb = "0.6"  # std enabled by default

# no_std (requires alloc for LUT generation)
linear-srgb = { version = "0.6", default-features = false }

# HDR transfer functions (BT.709, PQ, HLG)
linear-srgb = { version = "0.6", features = ["transfer"] }
```

- **`std`** (default): Required for runtime SIMD dispatch
- **`transfer`**: BT.709, PQ, HLG transfer functions
- **`iec`**: IEC 61966-2-1 textbook sRGB functions for legacy interop
- **`alt`**: Alternative/experimental implementations for benchmarking

## Accuracy

### Transfer function constants

All code paths use C0-continuous constants derived from the [moxcms](https://github.com/niclasberg/moxcms) reference implementation. These adjust the IEC 61966-2-1 offset from 0.055 to 0.055011 and the threshold from 0.04045 to 0.03929, making the piecewise transfer function mathematically continuous (~2.3e-9 gap eliminated).

At u8 precision the two constant sets produce identical values. At u16, the max difference is ~1 LSB near the threshold. See [docs/iec.md](docs/iec.md) for a detailed comparison.

For interop with software that uses the original IEC textbook constants, enable the `iec` feature for `linear_srgb::iec::srgb_to_linear` / `linear_srgb::iec::linear_to_srgb`.

### Accuracy summary (exhaustive f32 sweep)

| Path | Max ULP | Avg ULP | Monotonic |
|------|---------|---------|-----------|
| `default` s→l (rational poly) | 11 | ~0.5 | yes |
| `default` l→s (rational poly) | 14 | ~0.4 | yes |
| `precise` s→l (powf) | 6 | ~0.1 | yes |
| `precise` l→s (powf) | 3 | ~0.1 | yes |

Reference: C0-continuous f64 powf. The scalar rational polynomial evaluates in f64 intermediate precision, guaranteeing perfect monotonicity (zero reversals across all ~1B f32 values in [0, 1]). SIMD paths use f32 evaluation for throughput and are also monotonic within each segment.

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with Claude (Anthropic). All code has been reviewed and benchmarked, but verify critical paths for your use case.
