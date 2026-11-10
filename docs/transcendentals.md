# Transcendental inventory for portable math

Inspected 2026-11-10: main `24a876779772d8fddeb08b179354df05b1541c5c`
and [PR #30](https://github.com/imazen/linear-srgb/pull/30), head
`f36f932c74db8b9cdd4f6687eed36459a30c085b` (`api/target-features`).
The PR is open. This is a source inventory for planned archmage portable
math, not a numerical validation or a dependency/API migration.

The production requirements are **power, base-2 logarithm and base-2
exponential**, with square root as an additional algebraic primitive.
All SIMD math here is f32 at widths 4, 8 and 16; scalar formula APIs also
need f64 power. PR #30 retains the same math implementations and adds
calling surfaces that reach them.

## Runtime implementation families

| Operation | Implementation | Callers and precision |
| --- | --- | --- |
| `powf(x, y)` | [`src/scalar.rs`](../src/scalar.rs): ten call sites | C0 sRGB encode/decode in f32/f64, sign-preserving extended sRGB in f32, and custom gamma in f32/f64. Re-exported through `precise` and, for custom gamma, `default`. |
| `powf(x, y)` | [`src/iec.rs`](../src/iec.rs): four call sites | Textbook IEC sRGB encode/decode in f32/f64; requires `iec`. |
| `powf(x, y)` | [`src/tf/pq.rs`](../src/tf/pq.rs): four call sites | Scalar formula fallback below encoded signal `0.02` or linear light `1e-7`; requires `transfer`. Exponents are `1/M2`, `1/M1`, `M1`, `M2`, with `M1 = 0.1593017578125`, `M2 = 78.84375`. |
| Approximate `pow(x, y) = exp2(y * log2(x))` | `pow_for!` generates `pow_x4`, `pow_x8`, `pow_x16` in [`src/tf/fast_math.rs`](../src/tf/fast_math.rs); scalar counterpart `pow`/`log2`/`exp2` in [`src/lanes.rs`](../src/lanes.rs) | Custom-gamma arrays/slices, including RGBA and premultiplication, and near-black PQ vector/slice paths. Uses `POW_LOG2`, `POW_EXP2`, `POW_SQRT1_2`; the private midp approximation family. Exponent is one scalar broadcast across the vector. |
| Approximate `log2(x)` | `fast_log2f` and `fast_log2f_x4/x8/x16` in `src/tf/fast_math.rs` | Rational `LOG2_P/LOG2_Q` family, used by BT.709 power and HLG encode. Scalar lane implementations reuse `fast_log2f`. Requires `transfer` for those curve callers. |
| Approximate `exp2(x)` | `fast_pow2f` and `fast_pow2f_x4/x8/x16` in `src/tf/fast_math.rs`; `lanes::pow2` | Rational `POW2_NUM/POW2_DEN` family, used by BT.709 power and HLG decode. `fast_pow2f` means base-2 exponential, not squaring. |
| Approximate `pow(x, y)` | `fast_powf` and `fast_powf_x4/x8/x16` compose the preceding rational log2/exp2 | BT.709 encode/decode, exponents `0.45` and `1/0.45`. Distinct coefficients and evaluation from the midp `pow_x*` family. |
| `sqrt(x)` | Scalar and magetypes vector methods in `src/rational_poly.rs`, `src/scalar.rs`, `src/lanes.rs`, `src/simd.rs`, `src/tf/{srgb,pq,hlg}.rs`, `src/tokens/{x4,x8,x16}.rs` | sRGB encode polynomial input, extended sRGB encode, u16 fast LUT indexing, low HLG encode, and PQ encode's fourth root implemented as `sqrt().sqrt()`. These are f32 operations, including the sqrt before the scalar f64-intermediate polynomial. |

There are 18 direct production `powf` source expressions outside `alt`:
12 f32 and 6 f64. This counts expressions before macro expansion and excludes
comments, doctests and unit tests. Reexports and LUT constructors can reach
these expressions without containing a direct `powf` call themselves.

Generic runtime `LinearizationTable<N>::new` and `EncodingTable<N>::new`
in [`src/lut.rs`](../src/lut.rs) call the f64 scalar formula APIs. The
specialized u16 LUT generation in [`src/u16_lut.rs`](../src/u16_lut.rs)
uses the rational polynomial instead. The embedded tables in
[`src/const_luts.rs`](../src/const_luts.rs) perform no runtime transcendental
evaluation.

## PR #30

The new `target_features::{x4,x8,x16}` array/slice functions forward to
existing token kernels or compose x4 kernels for missing widths/tiers.
Scalar x4 gamma uses the generic vector core with `ScalarToken`; scalar
x4 BT.709/PQ/HLG calls `lanes`. There is no new transcendental algorithm
in the generated wrappers. The PR makes the scalar fast-math helpers
`pub(crate)` and documents the paired private copies.

Thus portable replacements need to cover scalar and x4/x8/x16 execution,
including V3, V4, NEON and WASM128. f64 SIMD transcendentals are not used
by this crate at the inspected revisions.

## Compatibility requirements for replacement

- Keep the two approximation families distinct until replacement behavior
  has been verified. An identically named `pow`, `log2` or `exp2` is not
  evidence of matching bits.
- `Mla::mla` selects fused evaluation on native targets and with
  `precise-portable`; default WASM selects relaxed evaluation. Scalar
  lane copies explicitly use `mul_add`. Preserve the evaluation policy
  alongside coefficients and operation order.
- The midp exp2 implementation returns zero below exponent `-126` and
  infinity at/above `128`; it does not generate the full exp2 subnormal
  range. Its scalar/vector implementations also contain log-domain and
  rounding handling. The rational scalar `fast_pow2f` clamps its argument
  to `[-126, 128]`; vector variants and `lanes::pow2` do not apply that
  clamp. These are different current behaviors, not interchangeable
  general-purpose libm contracts.
- SIMD transfer curves evaluate branches that are later masked out,
  including log2 of nonpositive arguments. A replacement must allow
  those lanes to be evaluated without panicking. Keep curve-level clamps,
  seam handling and alpha preservation in linear-srgb.
- `precise` and IEC scalar APIs currently use platform `powf` with `std`
  and `num_traits::Float` backed by libm without `std`. A midp replacement
  does not satisfy their existing precision contract by itself. Scalar
  libm APIs are outside the slice bit-parity contract.
- Existing numerical gates include [`tests/tier_parity.rs`](../tests/tier_parity.rs),
  [`tests/remaining_parity.rs`](../tests/remaining_parity.rs),
  [`tests/brute_force.rs`](../tests/brute_force.rs),
  [`tests/simd_consistency.rs`](../tests/simd_consistency.rs) and, on PR #30,
  `tests/target_features.rs` and `tests/target_features_dispatch.rs`.
  See [precision contracts](precision.md). This inventory did not rerun them.

## Other math and development-only uses

`floor`, `ceil`, ties-to-even rounding, integer conversion, fused multiply-add
and bit manipulation support these implementations. They are not
transcendentals, but removing `num-traits`/libm requires accounting for
them too, including LUT interpolation and `src/mlaf.rs`.

The compile-time `SRGB_U8_TO_LINEAR_LUT` in `src/scalar.rs` implements
`base^2.4` with multiplication and a Newton fifth-root iteration. It
makes no runtime `powf` call; replacing it requires const-capable math
and a separate check of the generated table's bits.

The `alt` feature has a third approximation family:
`safe_impl::{fastlog2, fastpow2}` and `fastpow` in
[`src/alt/imageflow.rs`](../src/alt/imageflow.rs). It also has two direct
f32 `powf` expressions for sRGB decoding and LUT generation.
[`src/alt/accuracy.rs`](../src/alt/accuracy.rs) adds four textbook reference
`powf` expressions (two f32, two f64). Preserve these comparison baselines
when evaluating a replacement.

Unit/integration tests, examples, benches and fitting/review scripts use
additional reference math and sampling operations. For example,
`src/tf/mod.rs` tests use f64 `powf`, `exp`, `ln`, `log2`, `exp2` and
`sqrt`; test grids use integer `powi`; Python fitting scripts use NumPy
power, log/exp and trigonometric sampling. These are not extra shipped
runtime requirements. Keep independent reference math independent when
checking portable implementations.

No runtime `sin`, `cos`, `tan`, inverse/hyperbolic trig, `cbrt`, `hypot`,
`log10`, `expm1` or `log1p` calls occur in `src/` outside tests.
HLG's natural log/exp are already expressed through scaled log2/exp2.
