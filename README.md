# linear-srgb [![CI](https://img.shields.io/github/actions/workflow/status/imazen/linear-srgb/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/linear-srgb/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/linear-srgb?style=flat-square)](https://crates.io/crates/linear-srgb) [![lib.rs](https://img.shields.io/crates/v/linear-srgb?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/linear-srgb) [![docs.rs](https://img.shields.io/docsrs/linear-srgb?style=flat-square)](https://docs.rs/linear-srgb) [![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/linear-srgb?style=flat-square)](#license) [![codecov](https://img.shields.io/codecov/c/github/imazen/linear-srgb?style=flat-square)](https://codecov.io/gh/imazen/linear-srgb)

Fast, SIMD-accelerated sRGB↔linear conversion for image processing pipelines.
Pure Rust, `#![forbid(unsafe_code)]`, `no_std`-compatible.

Handles `f32`, `f64`, `u8`, and `u16` data. Supports in-place RGBA with alpha
preservation, fused premultiply/unpremultiply, custom gamma, and extended range.

## Quick start

```toml
[dependencies]
linear-srgb = "0.6.13"

# Optional: BT.709, PQ (HDR10), and HLG transfer functions
linear-srgb = { version = "0.6.13", features = ["transfer"] }
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

**u8 encode** (`linear_to_srgb_u8` / `linear_to_srgb_u8_slice`) quantizes the
linear input to a 4096-entry table. It can differ by one code from correctly
rounded encoding: 64,806 of 4,194,304 uniform inputs and 84,213 of 1,048,576
dark log-spaced inputs in the [precision review](https://github.com/imazen/linear-srgb/blob/main/docs/precision.md).
All 256 u8 decode→encode round trips are exact with the current constants;
filtered values between those codes still encounter table quantization error.

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

For `powf`-based extended range (scalar, any range):

```rust
use linear_srgb::precise::*;

let linear = srgb_to_linear_extended(-0.1);
let srgb = linear_to_srgb_extended(1.5);
```

### Precise (powf) conversions

For maximum accuracy:

```rust
use linear_srgb::precise::*;

// f32 — platform powf, C0-continuous (not correctly rounded)
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
linear-srgb = { version = "0.6.13", features = ["transfer"] }
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
| Extended range RGBA `&mut [f32]` | `srgb_to_linear_extended_rgba_slice` / `linear_to_srgb_extended_rgba_slice` |
| Custom gamma `&mut [f32]` | `gamma_to_linear_slice` / `linear_to_gamma_slice` |
| Custom gamma RGBA `&mut [f32]` | `gamma_to_linear_rgba_slice` / `linear_to_gamma_rgba_slice` |
| Custom gamma RGBA premul | `gamma_to_linear_premultiply_rgba_slice` / `unpremultiply_linear_to_gamma_rgba_slice` |
| Single f32 | `srgb_to_linear` / `linear_to_srgb` |
| Single u8 | `srgb_u8_to_linear` / `linear_to_srgb_u8` |
| Single u16 | `srgb_u16_to_linear` / `linear_to_srgb_u16` |
| Platform powf f32/f64 | `precise::srgb_to_linear` / `precise::linear_to_srgb` |
| Extended range (scalar) | `precise::srgb_to_linear_extended` / `precise::linear_to_srgb_extended` |

All functions live in `linear_srgb::default` unless noted.

## Accuracy

### Transfer function constants

All code paths use C0-continuous constants derived from the
[moxcms](https://github.com/awxkee/moxcms/blob/master/src/gamma.rs) reference implementation. These
adjust the IEC 61966-2-1 offset from 0.055 to 0.055011 and the threshold from
0.04045 to 0.03929, making the piecewise transfer function mathematically
continuous (~2.3e-9 gap eliminated).

The decoded f32 values differ at 244 of the 256 u8 codes: code 12 differs
by 2.28e-4 relative. The change affects the power branch throughout its range,
not just the junction. Re-encoding each set of decoded u8 values to u8 preserves
all 256 codes; that round trip does not establish f32 equivalence. See
[docs/iec.md](https://github.com/imazen/linear-srgb/blob/main/docs/iec.md) for a
detailed comparison.

For interop with software that uses the original IEC textbook constants, enable
the `iec` feature for `linear_srgb::iec::srgb_to_linear` /
`linear_srgb::iec::linear_to_srgb`.

### Measured accuracy

Against the crate's C0 f64 formula rounded to f32, a 5,242,880-point grid
(4,194,304 uniform plus 1,048,576 dark logarithmic inputs) gives:

| Path | Sampled max ULP |
|---|---:|
| `default` s→l scalar, f64 polynomial intermediates | 8 |
| `default` l→s scalar, f64 polynomial intermediates | 10 |
| `default` s→l dispatched slice, f32 fused evaluation | 11 |
| `default` l→s dispatched slice, f32 fused evaluation | 12 |
| `precise` s→l, platform powf | 6 |
| `precise` l→s, platform powf | 3 |

These are measured grid maxima, not all-input bounds. The earlier SIMD
4/5-ULP summary does not describe the current implementation. Scalar and
slice APIs use different evaluation precision; the existing monotonicity
and round-trip regression tests remain in place. One ULP at 0.5 is 2^-24;
even a small float error can change a quantized code at a rounding boundary.

See [precision contracts](docs/precision.md) and
[measured alternatives](benchmarks/u8_encode_2026-11-05.md) for LUT errors,
curve policy, row/tier portability and exact-rounding prototype costs.

## Feature Flags

- **`std`** (default): Required for runtime SIMD dispatch
- **`avx512`** (default): AVX-512 code paths (16-wide f32)
- **`transfer`**: BT.709, PQ, HLG transfer functions (scalar + SIMD)
- **`iec`**: IEC 61966-2-1 textbook sRGB functions for legacy interop
- **`alt`**: Alternative/experimental implementations for benchmarking

```toml
# no_std (requires alloc for LUT generation)
linear-srgb = { version = "0.6.13", default-features = false }
```

## Module Organization

- **`default`** — Recommended API. Rational polynomial for f32, LUT for integers, SIMD for slices.
- **`precise`** — `powf()`-based conversions with C0-continuous constants (not IEC textbook). f32/f64, extended range.
- **`lut`** — Lookup tables for custom bit depths (10-bit, 12-bit, 16-bit).
- **`tf`** — Transfer functions: BT.709, PQ, HLG. Requires `transfer` feature.
- **`iec`** — IEC 61966-2-1 textbook constants for legacy interop. Requires `iec` feature.
- **`target_features`** — Array/slice conversions inside matching CPU-feature functions (see below).
- **`tokens`** — Deprecated token-taking compatibility API.

## Composing inside CPU-feature functions (`target_features`)

Source transition: [PR #30](https://github.com/imazen/linear-srgb/pull/30); this API is not in crates.io 0.6.12.
See the [migration and outstanding-work inventory](https://github.com/imazen/linear-srgb/blob/api/target-features/docs/SIMD_TRANSITION.md).

For ordinary calls, use `linear_srgb::default`: it selects CPU features at runtime.
Use `target_features` when your loop already runs inside a matching
`#[target_feature]` function. The signatures use arrays and slices, so callers
can use a different version of archmage or magetypes.

```rust
# #[cfg(target_arch = "x86_64")]
# {
use archmage::{arcane, SimdToken, X64V3Token};
use linear_srgb::target_features::x8;

#[arcane]
fn my_pipeline(_token: X64V3Token, rows: &mut [[f32; 8]]) {
    for row in rows {
        *row = x8::srgb_to_linear_v3(*row); // no token crosses this call
    }
}

if let Some(token) = X64V3Token::summon() {
    my_pipeline(token, &mut [[0.5; 8]]);
}
# }
```

The width modules are `x4`, `x8`, and `x16`; suffixes specify the required
feature context: `_v3`, `_v4`, `_neon`, `_wasm128`, or `_scalar`.
**On native targets, these are safe calls inside matching feature functions.**
An ordinary caller gets E0133: a runtime feature check alone does not provide
the compiler context. Put the call in a feature function as above; ordinary
callers should use `default` instead of adding an `unsafe` block. WASM validates
SIMD support when loading the module. See the
[`target_features` documentation](https://docs.rs/linear-srgb/latest/linear_srgb/target_features/)
for the complete feature requirements and contracts.

The old `tokens` paths remain available with deprecation warnings. Migrate by
changing `tokens` to `target_features` and removing the token argument inside
the matching feature function. Conversion behavior and length contracts stay
the same. Token reexports and token-taking operation APIs are queued for removal
in the next version. The f32 families support caller-owned `incant!` dispatch
at x8 and x16 as well as x4; see the module documentation for an example.

<!-- crates.io:skip-start -->
## Benchmarks

The benches are interleaved [zenbench](https://github.com/imazen/zenbench) A/B
comparisons of linear-srgb's own strategies — scalar `powf` vs rational
polynomial vs LUT vs SIMD slice, plus an in-file imageflow-style reference
baseline. No third-party crate is timed. Inputs are built into RAM before the
timed region (no file I/O), output is consumed so it isn't optimized away, and
builds use runtime SIMD dispatch — **no `-C target-cpu=native`**, which is what
ships.

```bash
git clone https://github.com/imazen/linear-srgb && cd linear-srgb
cargo bench --bench benchmarks                    # throughput by type / strategy
cargo bench --features transfer --bench tf_bench  # BT.709 / PQ / HLG dispatch
cargo bench --bench rgba_approach                 # RGBA alpha-handling strategies
```

Full methodology, host details, and pinned-commit reproduction:
**[benchmarks/README.md](https://github.com/imazen/linear-srgb/blob/main/benchmarks/README.md)**.
Committed result files (the numbers live here — not reproduced inline):

| File | What it covers |
|------|----------------|
| [`arm_neoverse_n1_baseline_2026-05-31.md`](https://github.com/imazen/linear-srgb/blob/main/benchmarks/arm_neoverse_n1_baseline_2026-05-31.md) | ARM Neoverse-N1 throughput baseline, slice-tail curve fix, x86 cross-check |
| [`tf_dispatch_2026-04-23.md`](https://github.com/imazen/linear-srgb/blob/main/benchmarks/tf_dispatch_2026-04-23.md) | BT.709 / PQ / HLG `incant!` SIMD dispatch A/B (issue #10) |
<!-- crates.io:skip-end -->

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with AI assistance. Not all code has been manually reviewed. Verify
critical paths for your use case.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenjxl-decoder] · [jxl-encoder] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenrav1e] · [rav1d-safe] · [zenravif] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · **linear-srgb** · [garb] · [zenyuv] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] · [zenanalyze-api] |
| Test corpora | [codec-corpus] · [imazen-26] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter] · [zenutils]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zenextras
[zenpdf]: https://github.com/imazen/zenextras
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenravif]: https://github.com/imazen/cavif-rs
[zenavif-parse]: https://github.com/imazen/zenavif
[zenavif-serialize]: https://github.com/imazen/zenavif
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenpipe
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[garb]: https://github.com/imazen/garb
[zenyuv]: https://github.com/imazen/zenjpeg
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zenpipe
[zenlayout]: https://github.com/imazen/zenpipe
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
[zenanalyze-api]: https://github.com/imazen/zenanalyze
[codec-corpus]: https://github.com/imazen/codec-corpus
[imazen-26]: https://github.com/imazen/imazen-26
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[zenutils]: https://github.com/imazen/zenutils
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
