# Precision review, November 2026

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
