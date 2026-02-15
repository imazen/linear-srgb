# linear-srgb

Fast linear↔sRGB color space conversion with runtime CPU dispatch.

[![Crates.io](https://img.shields.io/crates/v/linear-srgb.svg)](https://crates.io/crates/linear-srgb)
[![Docs.rs](https://docs.rs/linear-srgb/badge.svg)](https://docs.rs/linear-srgb)
[![License](https://img.shields.io/crates/l/linear-srgb.svg)](LICENSE)

## Quick Start

```rust
use linear_srgb::default::*;

// Single values
let linear = srgb_to_linear(0.5f32);
let srgb = linear_to_srgb(linear);

// Fast polynomial (~4x faster than powf, 294 ULP max near black, &lt;4 ULP in upper half)
let linear = srgb_to_linear_fast(0.5f32);
let srgb = linear_to_srgb_fast(linear);

// Slices (SIMD-accelerated, polynomial)
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
| One f32 value (exact) | `srgb_to_linear(x)` / `linear_to_srgb(x)` |
| One f32 value (fast) | `srgb_to_linear_fast(x)` / `linear_to_srgb_fast(x)` |
| One u8 value | `srgb_u8_to_linear(x)` (LUT, fastest) |
| `&mut [f32]` slice | `srgb_to_linear_slice()` / `linear_to_srgb_slice()` |
| `&[u8]` → `&mut [f32]` | `srgb_u8_to_linear_slice()` |
| `&[f32]` → `&mut [u8]` | `linear_to_srgb_u8_slice()` |
| Inside `#[arcane]` | `default::inline::*` (no dispatch) |
| Standalone x8 call | `srgb_to_linear_x8()` (has dispatch, that's fine) |

## API Reference

### Single Values

```rust
use linear_srgb::default::*;

// f32 conversions — powf (exact reference)
let linear = srgb_to_linear(0.5f32);
let srgb = linear_to_srgb(0.214f32);

// f32 conversions — polynomial (~4x faster, 294 ULP max near black, &lt;4 ULP in upper half)
let linear = srgb_to_linear_fast(0.5f32);
let srgb = linear_to_srgb_fast(0.214f32);

// f64 high-precision
let linear = srgb_to_linear_f64(0.5f64);

// u8 conversions (LUT-based)
let linear = srgb_u8_to_linear(128u8);           // u8 → f32
let srgb_byte = linear_to_srgb_u8(0.214f32);     // f32 → u8

// u16 conversions (LUT-based)
let linear = srgb_u16_to_linear(32768u16);        // u16 → f32
let srgb_u16 = linear_to_srgb_u16(0.214f32);     // f32 → u16
```

### Slice Processing (Recommended for Batches)

```rust
use linear_srgb::default::*;

// In-place f32 conversion (SIMD-accelerated)
let mut values = vec![0.5f32; 10000];
srgb_to_linear_slice(&mut values);  // Modifies in-place
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

### Extended Range (HDR / Wide Gamut)

The standard functions clamp to \[0, 1\]. For cross-gamut pipelines (Rec. 2020 → sRGB, scRGB, HDR):

```rust
use linear_srgb::scalar::{srgb_to_linear_extended, linear_to_srgb_extended};

let linear = srgb_to_linear_extended(-0.1);  // Preserves negatives
let srgb = linear_to_srgb_extended(1.5);     // Preserves >1.0
```

See crate docs for when clamped vs extended is appropriate.

### LUT for Custom Bit Depths

```rust
use linear_srgb::lut::{LinearTable16, EncodingTable16, lut_interp_linear_float};

// 16-bit linearization (65536 entries)
let lut = LinearTable16::new();
let linear = lut.lookup(32768);  // Direct lookup

// Interpolated encoding
let encode_lut = EncodingTable16::new();
let srgb = lut_interp_linear_float(0.5, encode_lut.as_slice());
```

## Advanced: Token-Based Dispatch (`mage` feature)

For zero-overhead SIMD when you control the dispatch point:

```rust,ignore
use linear_srgb::mage;

// Obtain a token once, pass to all calls
mage::srgb_to_linear_slice(&mut values);  // Uses archmage incant! internally
```

## Advanced: Inlineable `#[rite]` Functions (`rites` feature)

For embedding inside your own `#[arcane]` code with no dispatch overhead:

```rust,ignore
use linear_srgb::rites::x8;
use archmage::arcane;

#[arcane]
fn my_pipeline(token: Desktop64, data: &mut [f32]) {
    // x8::srgb_to_linear_v3 is #[rite] — inlines into your function
    // Available widths: x4 (NEON/WASM), x8 (AVX2), x16 (AVX-512)
}
```

## Module Organization

- **`default`** — Recommended API. Re-exports optimal implementations.
- **`default::inline`** — Dispatch-free `wide::f32x8` variants for use inside your own SIMD code.
- **`simd`** — Full SIMD API with `_dispatch` and `_inline` variants.
- **`scalar`** — Single-value functions. Includes `_fast` (polynomial) and `_extended` (unclamped) variants.
- **`lut`** — Lookup tables for custom bit depths.
- **`mage`** — Token-based dispatch via archmage (feature-gated).
- **`rites`** — Inlineable `#[rite]` functions for x4/x8/x16 widths (feature-gated).

## Feature Flags

```toml
[dependencies]
linear-srgb = "0.5"  # std enabled by default

# no_std (requires alloc for LUT generation)
linear-srgb = { version = "0.5", default-features = false }

# Token-based dispatch (zero overhead)
linear-srgb = { version = "0.5", features = ["mage"] }

# Inlineable rites for embedding in #[arcane] code
linear-srgb = { version = "0.5", features = ["rites"] }
```

- **`std`** (default): Required for runtime SIMD dispatch
- **`mage`**: Token-based API using archmage
- **`rites`**: Inlineable `#[rite]` functions for x4/x8/x16
- **`alt`**: Alternative/experimental implementations for benchmarking
- **`unsafe_simd`**: Union-based bit manipulation, unchecked indexing

## Accuracy

Implements IEC 61966-2-1:1999 sRGB transfer functions with:
- C0-continuous piecewise function (no discontinuity at threshold)
- Constants derived from moxcms reference implementation
- Scalar `powf`: exact to f32/f64 precision
- Polynomial (`_fast`, SIMD): 294 ULP max near threshold, 2-3 ULP in upper half (exhaustive f32 sweep)
- f32 roundtrip: ~1e-5 accuracy
- f64 roundtrip: ~1e-10 accuracy

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with Claude (Anthropic). All code has been reviewed and benchmarked, but verify critical paths for your use case.
