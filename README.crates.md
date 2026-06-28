<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# linear-srgb

Fast, SIMD-accelerated sRGB↔linear conversion for image processing pipelines.
Pure Rust, `#![forbid(unsafe_code)]`, `no_std`-compatible.

Handles `f32`, `f64`, `u8`, and `u16` data. Supports in-place RGBA with alpha
preservation, fused premultiply/unpremultiply, custom gamma, and extended range.

## Quick start

```toml
[dependencies]
linear-srgb = "0.6"

# Optional: BT.709, PQ (HDR10), and HLG transfer functions
linear-srgb = { version = "0.6", features = ["transfer"] }
```

```rust
use linear_srgb::default::*;

// f32 slices (in-place, SIMD-accelerated)
let mut values = vec![0.5f32; 10000];
srgb_to_linear_slice(&mut values);
linear_to_srgb_slice(&mut values);
```

Slice functions are SIMD-accelerated (AVX-512, AVX2, SSE4.1, NEON, WASM SIMD128)
with automatic runtime CPU dispatch — typically 4–16× faster than scalar loops.

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

**Length contract.** The two-slice converters (those taking a separate `input`
and `output`, e.g. `srgb_u8_to_linear_slice`) require `input.len() == output.len()`
and **panic** otherwise — size the destination to exactly the source. The in-place
single-slice converters (e.g. `srgb_to_linear_slice(&mut values)`) have no such
constraint. The `_rgba_` variants treat the buffer as tightly-packed RGBA
(4 components per pixel, so the length should be a multiple of 4): R/G/B are
sRGB-decoded and the alpha component is passed through (as `a/255` for the u8
variants), not sRGB-decoded.

**Linear range is normalized `[0.0, 1.0]`.** The u8/u16 LUT decode paths map the
integer black point to `0.0` and the integer white point to `1.0` (u8: `0 → 0.0`,
`255 → 1.0`; u16: `0 → 0.0`, `65535 → 1.0`), so the linear f32 output lands in
`[0.0, 1.0]`. The single-value f32 API (`srgb_to_linear` / `linear_to_srgb`)
expects and returns the same normalized `[0, 1]` range and clamps inputs to it.
For values outside `[0, 1]` (HDR, cross-gamut matrix output), use the
extended-range functions below instead.

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

**u8 encode** (`linear_to_srgb_u8` / `linear_to_srgb_u8_slice`) rounds to the
nearest level (LUT index `linear * 4095 + 0.5`) and is accurate to within ±1 u8
level vs exact computation. The `u8 → linear → u8` round-trip is exact to within
**±1 level** (not bit-exact 0 like the `linear_to_srgb_u16` polynomial path) —
the LUT-quantized encode can land one level off near segment boundaries. For most
8-bit image work this is invisible; if you need a guaranteed-exact round-trip,
stay in u16.

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

// f32 — rational polynomial (≤10 ULP max, perfectly monotonic)
let linear = srgb_to_linear(0.5f32);
let srgb = linear_to_srgb(linear);

// u8 — LUT-based, zero math
let linear = srgb_u8_to_linear(128u8);
let srgb_byte = linear_to_srgb_u8(linear);

// u16 — LUT-based
let linear = srgb_u16_to_linear(32768u16);
let srgb_u16 = linear_to_srgb_u16(linear);
```

### Extended range (cross-gamut, HDR)

For values outside [0, 1] from gamut matrix conversions (P3→sRGB, BT.2020→sRGB):

```rust
use linear_srgb::default::*;

// SIMD-accelerated, sign-preserving (CSS Color 4)
// Uses 6/6 rational polynomials — no powf, pure SIMD
let mut values = vec![-0.1f32, 0.0, 0.5, 1.0, 1.5];
srgb_to_linear_extended_slice(&mut values);
linear_to_srgb_extended_slice(&mut values);
```

The extended slice functions use purpose-fitted 6/6 rational polynomials with
wider domains than the clamped path's 4/4. The S2L polynomial covers
|encoded| ≤ 8 (u8-safe) / ≤ ~4.2 (u16-safe). The L2S polynomial covers
|linear| ≤ 64 at u16 precision.

For exact `powf`-based extended range (scalar, any range):

```rust
use linear_srgb::precise::*;

let linear = srgb_to_linear_extended(-0.1);
let srgb = linear_to_srgb_extended(1.5);
```

### Precise (powf) conversions

For maximum accuracy:

```rust
use linear_srgb::precise::*;

// f32 — exact powf, C0-continuous (6 ULP max)
let linear = srgb_to_linear(0.5f32);
let srgb = linear_to_srgb(0.214f32);

// f64 high-precision
let linear = srgb_to_linear_f64(0.5f64);
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

Build your own tables for 8/10/12-bit depths (`LinearTable8`/`10`/`12`,
`EncodeTable8`/`12`):

```rust
use linear_srgb::lut::{LinearTable12, EncodeTable12, lut_interp_linear_float};

// 12-bit linearization (4096 entries)
let lut = LinearTable12::new();
let linear = lut.lookup(2048);

// Interpolated encoding
let encode_lut = EncodeTable12::new();
let srgb = lut_interp_linear_float(0.5, encode_lut.as_slice());
```

For 16-bit, skip the const tables: `LinearTable16` / `EncodeTable16` are
deprecated (each adds ~1s of const-eval compile time). Use
`default::srgb_u16_to_linear` and `default::linear_to_srgb_u16` instead —
they share a SIMD-generated `OnceLock` LUT.

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
| Extended range `&mut [f32]` | `srgb_to_linear_extended_slice` / `linear_to_srgb_extended_slice` |
| Custom gamma `&mut [f32]` | `gamma_to_linear_slice` / `linear_to_gamma_slice` |
| Custom gamma RGBA premul | `gamma_to_linear_premultiply_rgba_slice` / `unpremultiply_linear_to_gamma_rgba_slice` |
| Single f32 | `srgb_to_linear` / `linear_to_srgb` |
| Single u8 | `srgb_u8_to_linear` / `linear_to_srgb_u8` |
| Single u16 | `srgb_u16_to_linear` / `linear_to_srgb_u16` |
| Exact powf f32/f64 | `precise::srgb_to_linear` / `precise::linear_to_srgb` |
| Extended range (scalar) | `precise::srgb_to_linear_extended` / `precise::linear_to_srgb_extended` |

All functions live in `linear_srgb::default` unless noted.

## Accuracy

### Transfer function constants

All code paths use C0-continuous constants derived from the
[moxcms](https://github.com/niclasberg/moxcms) reference implementation. These
adjust the IEC 61966-2-1 offset from 0.055 to 0.055011 and the threshold from
0.04045 to 0.03929, making the piecewise transfer function mathematically
continuous (~2.3e-9 gap eliminated).

At u8 precision the two constant sets produce identical values. At u16, the max
difference is ~1 LSB near the threshold. See
[docs/iec.md](https://github.com/imazen/linear-srgb/blob/main/docs/iec.md) for a
detailed comparison.

For interop with software that uses the original IEC textbook constants, enable
the `iec` feature for `linear_srgb::iec::srgb_to_linear` /
`linear_srgb::iec::linear_to_srgb`.

### Accuracy summary (exhaustive f32 sweep)

Exhaustive f32 sweep (all ~1B values in [0, 1]) against f64 reference.
"SIMD" rows measured via the actual dispatched SIMD path (f32 FMA evaluation).
"Scalar" rows use f64 intermediate precision.

| Path | Max ULP | Avg ULP | Monotonic | Fitted domain |
|------|---------|---------|-----------|---------------|
| `default` s→l (4/4 scalar) | 8 | 0.18 | yes | [0, 1] |
| `default` l→s (4/4 scalar) | 10 | 0.32 | yes | [0, 1] |
| `default` s→l (4/4 SIMD) | 4 | 0.09 | yes | [0, 1] |
| `default` l→s (4/4 SIMD) | 5 | 0.10 | yes | [0, 1] |
| `extended_slice` s→l (6/6 SIMD) | 8* | 0.12 | yes | [0, 8] |
| `extended_slice` l→s (6/6 SIMD) | 8* | 0.17 | yes | [0, 64] |
| `precise` s→l (powf) | 6 | 0.1 | yes | unbounded |
| `precise` l→s (powf) | 3 | 0.1 | yes | unbounded |

\*The 6/6 extended polynomials use larger coefficients to cover a wider domain,
which costs ~2 ULP vs the clamped 4/4 in a narrow band near the piecewise
threshold (0.04–0.05). Affects < 0.1% of values; avg ULP is comparable.

**What does 10 ULP mean in practice?** 1 ULP (unit in the last place) is the
spacing between adjacent f32 values at a given magnitude. At 0.5 that's ~6e-8,
so 10 ULP ≈ 6e-7 — about 6 decimal digits of precision. At 0.01 it's ~1e-8.
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

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · **linear-srgb** · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
