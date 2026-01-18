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

// Slices (SIMD-accelerated)
let mut values = vec![0.5f32; 10000];
srgb_to_linear_slice(&mut values);
linear_to_srgb_slice(&mut values);

// u8 ↔ f32 (image processing)
let linear = srgb_u8_to_linear(128);
let srgb_byte = linear_to_srgb_u8(linear);
```

## Which Function Should I Use?

```
                    ┌─────────────────────────┐
                    │ How many values?        │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
         ┌────────┐        ┌────────┐        ┌────────────┐
         │ One    │        │ Slice  │        │ Building   │
         │ value  │        │ [f32]  │        │ own SIMD?  │
         └───┬────┘        └───┬────┘        └─────┬──────┘
             │                 │                   │
             ▼                 ▼                   ▼
    ┌─────────────────┐  ┌──────────────┐   ┌─────────────────┐
    │ srgb_to_linear  │  │ *_slice()    │   │ Inside your own │
    │ linear_to_srgb  │  │              │   │ #[multiversed]? │
    │ srgb_u8_to_     │  │ Dispatch once│   └────────┬────────┘
    │   linear (LUT)  │  │ loop is fast │            │
    └─────────────────┘  └──────────────┘     ┌──────┴──────┐
                                              ▼             ▼
                                           ┌─────┐      ┌─────┐
                                           │ Yes │      │ No  │
                                           └──┬──┘      └──┬──┘
                                              │            │
                                              ▼            ▼
                                    ┌──────────────┐  ┌──────────────┐
                                    │ default::    │  │ *_x8() or    │
                                    │ inline::*    │  │ *_x8_slice() │
                                    │              │  │              │
                                    │ No dispatch, │  │ Has dispatch │
                                    │ #[inline]    │  │ (that's fine)│
                                    └──────────────┘  └──────────────┘
```

**Quick reference:**

| Your situation | Use this |
|----------------|----------|
| One f32 value | `srgb_to_linear(x)` / `linear_to_srgb(x)` |
| One u8 value | `srgb_u8_to_linear(x)` (LUT, 20x faster than scalar) |
| `&mut [f32]` slice | `srgb_to_linear_slice()` / `linear_to_srgb_slice()` |
| `&[u8]` → `&mut [f32]` | `srgb_u8_to_linear_slice()` |
| `&[f32]` → `&mut [u8]` | `linear_to_srgb_u8_slice()` |
| `&mut [f32x8]` slice | `linear_to_srgb_x8_slice()` (dispatch once) |
| Inside `#[multiversed]` | `default::inline::*` (no dispatch) |
| Standalone x8 call | `linear_to_srgb_x8()` (has dispatch, that's fine) |

## Performance Guide

This crate is carefully tuned for maximum throughput. The `default` module exposes the fastest implementation for each conversion type, chosen based on extensive benchmarking.

### Why Each Default Was Chosen

| Conversion | Default Implementation | Why |
|------------|----------------------|-----|
| **u8 → f32** | LUT direct lookup | 3-4 Gelem/s. 256-entry table fits in L1 cache. Beats both scalar (170 Melem/s) and SIMD. |
| **u16 → f32** | LUT direct lookup | 450-820 Melem/s. 2.5-16x faster than scalar powf. |
| **f32 → f32 (sRGB→linear)** | Scalar powf | 1.5-1.7 Gelem/s. *Counterintuitively*, hardware transcendentals beat SIMD polynomial approximation by 4x. |
| **f32 → f32 (linear→sRGB)** | SIMD with dispatch | 440-480 Melem/s. ~2x faster than scalar for this direction. |
| **f32 → u8** | SIMD with dispatch | 270-275 Melem/s. ~1.8x faster than scalar. |
| **f32 → u16** | Scalar powf | 145-200 Melem/s. Beats LUT interpolation due to interpolation overhead. |

### Dispatch Overhead

The `_dispatch` variants use runtime CPU feature detection (AVX2, SSE4.1, NEON, etc.) via `multiversed`. This adds ~1-3ns per call, which is fully amortized even at 8 elements.

**Bottom line:** Always use the slice functions for batches. The dispatch cost is negligible.

## API Reference

### Single Values

```rust
use linear_srgb::default::*;

// f32 conversions (scalar - fast for individual values)
let linear = srgb_to_linear(0.5f32);
let srgb = linear_to_srgb(0.214f32);

// f64 high-precision
let linear = srgb_to_linear_f64(0.5f64);

// u8 conversions (LUT-based)
let linear = srgb_u8_to_linear(128u8);           // u8 → f32
let srgb_byte = linear_to_srgb_u8(0.214f32);     // f32 → u8
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

### x8 SIMD Functions

For processing exactly 8 values with explicit SIMD:

```rust
use linear_srgb::default::*;
use wide::f32x8;

// With CPU dispatch (recommended for standalone use)
let srgb = f32x8::from([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
let linear = srgb_to_linear_x8(srgb);  // Uses _dispatch internally

// u8 array → f32x8
let srgb_bytes = [0u8, 32, 64, 96, 128, 160, 192, 255];
let linear = srgb_u8_to_linear_x8(srgb_bytes);
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
let linear = lut.lookup(32768);  // Direct lookup

// Interpolated encoding
let encode_lut = EncodingTable16::new();
let srgb = lut_interp_linear_float(0.5, encode_lut.as_slice());
```

## Advanced: Using `default::inline` with `#[multiversed]`

If you're building your own SIMD-accelerated function with `multiversed`, use `default::inline::*` to avoid nested dispatch overhead:

```rust
use linear_srgb::default::inline::*;  // Clean names, no _inline suffix
use multiversed::multiversed;
use wide::f32x8;

#[multiversed]  // Your function handles dispatch
#[inline]
pub fn process_pixels(data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ]);

        // No dispatch here - your #[multiversed] already handled it
        let linear = srgb_to_linear_x8(v);
        let processed = linear * f32x8::splat(1.5);  // Your processing
        let result = linear_to_srgb_x8(processed);

        let arr: [f32; 8] = result.into();
        chunk.copy_from_slice(&arr);
    }
}
```

**Why this matters:**
- `default::*` x8 functions: Include CPU feature detection (~1-3ns overhead per call)
- `default::inline::*`: Pure SIMD code, `#[inline(always)]`, zero overhead

If you call dispatched functions inside a loop within your own `#[multiversed]` function, you pay dispatch cost per iteration. Use `default::inline::*` to avoid this.

## Benchmark Results

Measured on AMD Ryzen / Intel with AVX2. Results show median time per element.

### sRGB → Linear (Linearization)

| Input | Output | Method | Throughput | Notes |
|-------|--------|--------|------------|-------|
| u8 | f32 | LUT8 direct | **3.0-4.3 Gelem/s** | Fastest. Used by default. |
| u8 | f32 | Scalar powf | 170-180 Melem/s | 20x slower than LUT |
| u16 | f32 | LUT16 direct | **450-820 Melem/s** | 2.5-16x faster than scalar |
| f32 | f32 | Scalar powf | **1.5-1.7 Gelem/s** | Fastest. Hardware transcendentals win. |
| f32 | f32 | SIMD dispatch | 275-435 Melem/s | 4x slower than scalar! |

### Linear → sRGB (Encoding)

| Input | Output | Method | Throughput | Notes |
|-------|--------|--------|------------|-------|
| f32 | f32 | SIMD dispatch | **440-480 Melem/s** | Fastest. Used by default. |
| f32 | f32 | Scalar powf | 190-200 Melem/s | 2.4x slower |
| f32 | u8 | SIMD dispatch | **270-310 Melem/s** | Fastest. Used by default. |
| f32 | u8 | Scalar powf | 145-160 Melem/s | 1.8x slower |
| f32 | u8 | LUT12 interp | 125-135 Melem/s | Slowest due to interp overhead |
| f32 | u16 | Scalar powf | **145-200 Melem/s** | Fastest. Beats LUT interp. |
| f32 | u16 | LUT16 interp | 120-130 Melem/s | Interpolation overhead |

### Dispatch Overhead

At small sizes (8-64 elements), dispatch overhead is measurable but acceptable:

| Size | Slice dispatch once | x8 dispatch per chunk | x8 inline (no dispatch) |
|------|--------------------|-----------------------|-------------------------|
| 8 | 27.5 ns | 31.0 ns | 28.2 ns |
| 64 | 144 ns | 165 ns | 151 ns |
| 1024 | 2116 ns | 2487 ns | 2377 ns |

**Conclusion:** Slice functions (dispatch once) have essentially no overhead vs inline at practical sizes.

## Module Organization

- **`default`** - Recommended API. Re-exports optimal implementations.
- **`default::inline`** - Dispatch-free variants for use inside `#[multiversed]`.
- **`simd`** - Full SIMD API with `_dispatch` and `_inline` variants.
- **`scalar`** - Single-value functions. Use for individual conversions.
- **`lut`** - Lookup tables for custom bit depths.

## Deprecated Functions

These functions are marked `#[deprecated]` because faster alternatives exist. They remain available for benchmarking and compatibility.

| Deprecated | Speed vs Alternative | Use Instead |
|------------|---------------------|-------------|
| `scalar::srgb_u8_to_linear` | 20x slower | `simd::srgb_u8_to_linear` (LUT) |
| `simd::srgb_to_linear_x8*` | 4x slower | `scalar::srgb_to_linear` in a loop |
| `SrgbConverter::linear_to_srgb_u8` | 2x slower | `simd::linear_to_srgb_u8_slice` |
| `SrgbConverter::batch_linear_to_srgb` | 2x slower | `simd::linear_to_srgb_u8_slice` |

**Why SIMD srgb_to_linear is slower:** The sRGB→linear direction uses `powf(2.4)`. Hardware scalar transcendentals beat our SIMD polynomial approximation by 4x. The inverse direction (`powf(1/2.4)`) is different enough that SIMD wins there.

## Feature Flags

```toml
[dependencies]
linear-srgb = "0.2"  # std enabled by default

# no_std (requires alloc for LUT generation)
linear-srgb = { version = "0.2", default-features = false }

# Enable unsafe optimizations
linear-srgb = { version = "0.2", features = ["unsafe_simd"] }
```

- **`std`** (default): Required for runtime SIMD dispatch
- **`unsafe_simd`**: Union-based bit manipulation, unchecked indexing

## Accuracy

Implements IEC 61966-2-1:1999 sRGB transfer functions with:
- C0-continuous piecewise function (no discontinuity at threshold)
- Constants derived from moxcms reference implementation
- f32: ~1e-5 roundtrip accuracy
- f64: ~1e-10 roundtrip accuracy

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with Claude (Anthropic). All code has been reviewed and benchmarked, but verify critical paths for your use case.
