# Precision review, November 2026

For the scalar and SIMD math dependencies, see the
[transcendental inventory](transcendentals.md).

The default u8 encoder remains approximate. The u16 polynomial is also not a
correctly rounded encoder for arbitrary f32 input. Exact integer round trips
do not establish either property.

## Reference checks

Measured against f64 formula evaluations with the crate's continuous sRGB
constants on x86-64 and ARM Neoverse-N1. The grid has 4,194,304 uniform linear
values and 1,048,576 dark log-spaced values. Integer domains are exhaustive.
See the [encode experiments](../benchmarks/u8_encode_2026-11-05.md).

| Function / family | Result | Precision choice |
|---|---|---|
| `linear_to_srgb_u8`, slice and RGBA RGB | 64,806 / 4,194,304 uniform and 84,213 / 1,048,576 dark inputs differ by one code | 12-bit index into a 4 KiB table |
| `linear_to_srgb_u16_slice` | 44,579 / 5,242,880 differ by one code | f32 rational polynomial and quantization |
| `srgb_u8_to_linear` | 0 / 256 differ from f64 rounded to f32 | 256-entry f32 table |
| `srgb_u16_to_linear` | 59,538 / 65,536 differ, maximum 9 ULP | Reciprocal normalization and polynomial table generation |
| u8 / u16 decode→encode | 0 / 256 and 0 / 65,536 differences | Round-trip property only |
| 8/10/12/14-bit codes through canonical u16 and linear | Every code returns | Does not test values between codes |

`precise` calls platform `powf`; it does not promise correctly rounded math
or portable libm bits. Scalar `default` f32 conversion evaluates its
polynomial with f64 intermediates; slice kernels use f32 fused evaluation.
These are distinct precision contracts. Generic `lut` tables/interpolation
add index/interpolation error. `*_u16_fast` explicitly permits one-code
round-trip error from its sqrt-indexed table.

## Row and tier fixes

Compared with 0.6.12, plain slices and direct token slice hooks process
tails with the same arithmetic as complete vectors. HDR RGBA paths preserve
alpha around the plain kernels, including the corrected near-black PQ
formula. u16 decode tails use their slice's f32 polynomial; RGBA u16 encoding
shares the plain encoder. AVX-512 PQ hooks use the shared corrected kernel.

Trailing f32 samples, scalar-tier output, near-black PQ output, and some u16
RGBA samples change. Consumers with output digests must review and regenerate
them after the dependency update. No digest fixtures were relaxed.

Native tier parity applies to finite inputs. Extended paths can produce
different NaN signs/payloads on scalar and SIMD. Strict WASM needs
`precise-portable` for fused arithmetic; relaxed SIMD depends on the engine.
Libm-based scalar APIs are outside the slice parity contract.

## Other precision policies

- Integer output saturates. Extended f32 functions preserve signs and permit
  out-of-range values; clamped functions do not.
- Unpremultiplication zeros RGB at alpha ≤ `UNPREMUL_ALPHA_THRESHOLD`
  (1/1024). This deliberately differs from division for every positive alpha.
- Continuous and textbook sRGB differ across the power branch; see
  [the constant comparison](iec.md). ICC transforms must honor profile TRCs.
- Continuous BT.709 constants differ from rounded broadcast constants. HLG
  inverse OETF is not a full display EOTF including an OOTF.
- Changing polynomial degree, FMA or normalization changes bits even when
  integer round trips pass. Preserve reference-port math in metric and codec
  crates instead of substituting an approximate curve.

## Recommendation

Use an accurately rounded integer encoder as the future default, with
explicitly named fast alternatives. The table-plus-threshold prototype
matches the f64 oracle at all 1,065,353,217 nonnegative f32 values through
one, on both measured architectures. A wider uniform table alone does not
solve correct rounding. This review does not change the encode default.

Keep the curve for this patch; evaluate textbook sRGB as the next default
with an explicit continuous compatibility policy. Leave matrices, luma,
display modeling and reference-exact transcendentals with their numerical
owners. Unused workspace table types are API-review candidates, not
authorization to remove public functions.

Float LUT interpolation now evaluates `lower + fraction * (upper - lower)`
with one fused operation. The earlier two-product form lost near-black
precision and differed by target FMA availability (63,274 / 160,800 sampled
`SrgbConverter` results differed across x86/ARM, up to 3,226,306 ULP near zero).
The revision fixes cancellation; `precise`/IEC powf paths still inherit platform
libm rounding and are not a cross-platform bit-exact API. NaN sign/payload
and out-of-domain transcendental results are not portable contracts.

The 5,242,880-point uniform/dark grid measures default scalar decode/encode
maxima of 8/10 ULP and dispatched slice maxima of 11/12 ULP against C0
f64→f32. The `precise` paths measure 6/3 ULP; the converter's interpolated
encode reaches 4367 ULP (absolute error 1.6270163367089474e-5), and the
sqrt-indexed u16 fast table differs at 1,177,899 samples, max one code.
These grid results replace the unsupported README SIMD 4/5-ULP summary.

Clamping is API-specific: ordinary sRGB/gamma functions and integer outputs
clamp, while BT.709 preserves its linear negative branch and HDR transfer
functions do not apply a common upper-unit clamp. Their valid transfer domain
and display-light normalization must be chosen by the caller.

Extended sRGB fits measure 8/7 ULP maxima over the same unit-range grid.
BT.709 decode/encode measure scalar maxima 69/30 ULP and slice maxima 69/29;
HLG 4/6 ULP at both; PQ scalar 955/61 ULP and slice 1662/61 ULP. PQ decode's
slice maximum relative error is 1.0241151287345248e-4 on this grid, while its
maximum absolute normalized error is 0.33144092779103906 u16-code units.
See [full float measurements](../benchmarks/precision_f32_2026-11-05.md)
for domains, counts and separate relative/absolute errors. The near-black
formula fallback remains approximate f32 arithmetic, not exact rounding.

Replaying the same dense oracle against0.6.12 confirms that the measured
sRGB polynomial maxima already applied to the release. The released PQ
vector decode reaches369.12907035951235 normalized u16-code units of
absolute error; the corrected near-black path measures0.33144092779103906.
The existing79-code historical note understated the denser sampled error.
