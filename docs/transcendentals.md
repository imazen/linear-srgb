# Transcendental inventory for portable math

## Upstream source available after this inventory (2026-11-10)

The current archmage repository contains a standalone
[Magemath crate](https://github.com/imazen/archmage/tree/main/magemath) with
configurable **f32** power, exponential and logarithm implementations. Its
[power contract](https://github.com/imazen/archmage/blob/main/docs/POWF.md)
defines bit-exact approximation graphs, bounded domains and matching typed
vector/scalar-tail calls. MIDP is an approximation contract, not a substitute
for the existing precise/IEC accuracy requirements. A binary64 graph used
internally for f32 HIGHP is not an exposed general f64 power replacement.

Integrating it here still requires matching each curve's domain, evaluating
speculative inactive lanes safely, preserving scalar/tail arithmetic, and
checking the existing accuracy gates and immutable main goldens. ISA-specific
NaN/min/max/zero behavior also needs pointwise investigation; shared power
alone does not establish complete transfer-function portability.

For bit-exact arithmetic, native FMA and engine-proven relaxed-WASM fusion
have hardware paths; strict/nonfusing WASM needs exact software fallback.
The [upstream performance reference](https://github.com/imazen/archmage/blob/main/docs/POWF_REFERENCE.md)
does not provide a qualified strict/relaxed-WASM timing result. This inventory
has no measured linear-srgb integration slowdown or implementation-time budget.
The source API remains subject to its upstream review; this note does not
change linear-srgb's shipped dependency resolution.

## Original source inventory

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

## External math calls

The following inventory counts production source expressions outside `alt`,
unit tests and doctests, across optional features and target configurations.
It counts expressions, not runtime invocations or expanded SIMD variants.
The math inventory is the same on PR #30.

| Scalar method | Types / source expressions | Callers | Backend without `std` |
| --- | --- | --- | --- |
| `powf` | f32: 12; f64: 6 | `scalar.rs`, `iec.rs`, `tf/pq.rs`; generic LUT constructors reach the f64 scalar APIs | `num_traits::Float` → `libm::powf` / `libm::pow` |
| `sqrt` | f32: 10 | `lanes.rs`: 5; `rational_poly.rs`: 1; `scalar.rs`: 1; `tf/pq.rs`: 2; `tf/hlg.rs`: 1 | `Float` → `libm::sqrtf` |
| `floor` | f32: 4 | `lanes.rs`: 2; `tf/fast_math.rs`: 1; `lut.rs`: 1 | `Float` → `libm::floorf` |
| `ceil` | f32: 1 | `lut.rs`: 1 | `Float` → `libm::ceilf` |
| `mul_add` | f32: 28; f64: 8 | `lanes.rs`: 19 f32; `tf/fast_math.rs`: 8 f32; `lut.rs`: 1 f32; `rational_poly.rs`: 8 f64 | `Float` → `libm::fmaf` / `libm::fma` |
| `MulAdd::mul_add` | 1 generic expression, used for f32/f64 | `mlaf.rs:30`, reached through `fmla` in scalar/IEC encode; compiled only with the file's hardware-feature condition | `num_traits::MulAdd` → `Float::mul_add` → `libm::fmaf` / `libm::fma` |

With `std`, the ordinary scalar methods above select Rust's inherent float
methods; num-traits' `Float` implementation also forwards to those methods.
`MulAdd` forwards through `Float` even with `std`. Inspection of local
Rust 1.99.1 sources confirms the std math path uses compiler intrinsics;
this is not a promise of a hardware instruction or the same implementation
on every target. In particular, platform `powf` is not a portable numerical
reference. No performance measurement was made in this inventory.

The generic `mlaf` wrapper has another difference: when its compile-time
hardware condition is false it evaluates `acc + a * b`, without requesting
fused rounding. This is separate from direct scalar `mul_add`, which always
requests fused rounding, and from the vector `Mla` policy below. Default
scalar sRGB therefore still has external math dependencies even though its
power curve is a polynomial: f32 sqrt on encode and eight f64 FMA expressions
in the shared evaluator.

The external magetypes operations used by production kernels are:

- f32x4/x8/x16 `sqrt` (23 source expressions), `floor` (3 expressions),
  and macro-generated `round` and `to_i32_round` (1 expression each).
- `mul_add_fused` and `mul_add_relaxed`, selected through the local `Mla`
  trait. There is one source expression for each in `mla_for!`.
- `abs`, `min`, `max`, numeric `from_i32_t`, truncating `to_i32`, and
  bitcasts, shifts, comparisons and blends supporting the kernels.

These are vector-library calls, including when instantiated with
`ScalarToken`; they do not resolve through the local `num_traits::Float`
imports. In inspected magetypes 0.9.31-beta.2 source, scalar vector sqrt,
floor and ties-to-even rounding call its own `nostd_math` helpers. Scalar
fused FMA calls its `soft_fma` hardware/software implementation. Native
vector backends implement the primitives with target operations; composed
wide backends delegate to narrower backends. Thus replacing local scalar
libm calls alone does not unify the scalar and vector implementations.
There are **no calls to external magetypes `pow_midp`, `log2_midp` or
`exp2_midp`** in these revisions: those approximations are private local
copies described above.

Other scalar float methods in production are `abs`, `signum`, `max`,
`clamp`, `is_nan`, `to_bits` and `from_bits`, plus numeric `From` and `as`
conversions. They use core operations, rather than the std-only math methods
or num-traits' libm forwarding. Float `clamp` includes f64 alpha quantization;
`max` includes scalar BT.709/HLG/PQ. Integer `min`, `div_ceil`, wrapping
arithmetic and array/index operations are also present, but are not float
math dependencies. `core::f32::consts::{LN_2, LOG2_E}` supplies constants,
not calls. Archmage supplies dispatch/macros here; bytemuck supplies casts.

The std-only math dependency names to replace or deliberately retain are
therefore **f32/f64 power, f32 sqrt, f32 floor/ceil, and f32/f64 fused FMA**,
including the generic `MulAdd` wrapper. A shared implementation needs a
declared scalar/vector policy, not platform `powf` as its bitwise oracle.
Tests/reference tools additionally use external `powi`, `exp`, `ln`,
`log2` and `exp2`; keep their oracle role separate from production migration.

Backend forwarding was checked against downloaded num-traits 0.2.19 and
magetypes 0.9.31-beta.2 sources via `cargo read`, not the Cargo registry source
cache. This checkout has no root Cargo.lock; the manifest allows later
compatible versions. These dependency observations describe those inspected
versions, not an asserted current dependency resolution.

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
  needs validation against their existing accuracy gates; current platform
  outputs are not a portable bitwise contract. Scalar libm APIs are outside
  the slice bit-parity contract. A shared scalar/vector implementation can
  establish a consistent policy without copying platform-dependent bits.
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
