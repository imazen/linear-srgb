# linear-srgb

Fast linear↔sRGB color space conversion with runtime CPU dispatch.

[![Crates.io](https://img.shields.io/crates/v/linear-srgb.svg)](https://crates.io/crates/linear-srgb)
[![Docs.rs](https://docs.rs/linear-srgb/badge.svg)](https://docs.rs/linear-srgb)
[![License](https://img.shields.io/crates/l/linear-srgb.svg)](LICENSE)

## Quick Start

```rust
use linear_srgb::default::*;

// Single values (rational polynomial — fast, ~8 ULP max)
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
| `&[u8]` → `&mut [f32]` | `default::srgb_u8_to_linear_slice()` |
| `&[f32]` → `&mut [u8]` | `default::linear_to_srgb_u8_slice()` |
| Inside `#[arcane]` fn | `tokens::x8::srgb_to_linear_v3()` (inlines, no dispatch) |

## API Reference

### Single Values

```rust
use linear_srgb::default::*;

// f32 conversions — rational polynomial (~110 ULP max near threshold, <8 ULP elsewhere)
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

```rust
use linear_srgb::precise::*;

// f32 — exact powf, C0-continuous (6 ULP max vs moxcms reference)
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

// u8 → f32 (LUT-based, extremely fast)
let srgb_bytes: Vec<u8> = (0..=255).collect();
let mut linear = vec![0.0f32; 256];
srgb_u8_to_linear_slice(&srgb_bytes, &mut linear);

// f32 → u8 (SIMD-accelerated)
let linear_values: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
let mut srgb_bytes = vec![0u8; 256];
linear_to_srgb_u8_slice(&linear_values, &mut srgb_bytes);
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
    // Available widths: x4 (NEON/WASM), x8 (AVX2), x16 (AVX-512)
}
```

## Module Organization

- **`default`** — Recommended API. Rational polynomial for f32, LUT for integers, SIMD for slices.
- **`precise`** — Exact `powf()` conversions. C0-continuous, f32/f64, extended range.
- **`tokens`** — Inlineable `#[rite]` functions for x4/x8/x16 widths. For use inside `#[arcane]` code.
- **`lut`** — Lookup tables for custom bit depths.
- **`tf`** — Transfer functions: BT.709, PQ, HLG (feature-gated behind `transfer`).

## Feature Flags

```toml
[dependencies]
linear-srgb = "0.5"  # std enabled by default

# no_std (requires alloc for LUT generation)
linear-srgb = { version = "0.5", default-features = false }

# HDR transfer functions (BT.709, PQ, HLG)
linear-srgb = { version = "0.5", features = ["transfer"] }
```

- **`std`** (default): Required for runtime SIMD dispatch
- **`transfer`**: BT.709, PQ, HLG transfer functions
- **`alt`**: Alternative/experimental implementations for benchmarking

## Accuracy

### Transfer function constants

The IEC 61966-2-1 sRGB spec defines a piecewise transfer function with a linear segment and a power curve. The textbook constants (threshold 0.04045 / 0.0031308, offset 0.055) create a tiny discontinuity at the boundary — the two segments don't quite meet (~2.3e-9 in f64).

This crate uses two constant sets, each chosen for correctness in its context:

| Code path | Constants | Threshold (gamma) | Why |
|-----------|-----------|-------------------|-----|
| `default` (rational poly) | IEC textbook | 0.04045 | Polynomial was fitted to the IEC power curve |
| `precise` (powf) | moxcms C0 | 0.039293... | Eliminates the discontinuity |
| SIMD / `tokens` | IEC textbook | 0.04045 | Same rational polynomial as `default` |
| LUT tables | IEC textbook | 0.04045 | Identical to moxcms at u8/u16 precision |

The rational polynomial (from libjxl) approximates `((x+0.055)/1.055)^2.4` — the IEC power segment. Using the IEC threshold gives 110 ULP max error. Switching to the moxcms threshold would push values into the linear segment that should be evaluated by the polynomial, causing 3100+ ULP errors. The IEC threshold is optimal for this approximation.

The `precise` path uses moxcms C0-continuous constants (derived from the moxcms reference implementation) because they make the piecewise function mathematically continuous. With `powf()` computing the exact power curve, this distinction actually matters.

### Accuracy summary (exhaustive f32 sweep)

| Path | Reference | Max error | Avg error |
|------|-----------|-----------|-----------|
| `default` s→l | IEC f64 | 110 ULP | 0.55 ULP |
| `default` l→s | IEC f64 | 31 ULP | 0.37 ULP |
| `precise` s→l | moxcms f64 | 6 ULP | 0.11 ULP |
| `precise` l→s | moxcms f64 | 3 ULP | 0.10 ULP |

The `default` path's worst case (110 ULP for s→l) occurs at the piecewise threshold where the linear segment meets the rational polynomial. Away from the threshold, typical error is <8 ULP.

### Practical impact

The two constant sets produce identical results at u8 precision (the threshold falls between u8 values 10 and 11). At u16 precision, the maximum difference is ~1 LSB near the threshold. The difference only becomes measurable with raw f32 values in the narrow threshold region (0.039–0.041 gamma-space).

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with Claude (Anthropic). All code has been reviewed and benchmarked, but verify critical paths for your use case.
