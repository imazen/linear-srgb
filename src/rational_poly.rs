//! Rational polynomial sRGB transfer function approximation.
//!
//! Single source of truth for all polynomial constants used across scalar,
//! SIMD, magetypes rites (x4/x8/x16), and mage implementations.
//!
//! Evaluates P(x)/Q(x) via Horner's method: 4 `mul_add` + 1 `div` per
//! direction. Much faster and more accurate than Chebyshev polynomials.
//!
//! Uses C0-continuous thresholds (moxcms) that eliminate the ~2.3e-9
//! discontinuity present in the IEC 61966-2-1 textbook constants. The
//! polynomials are fitted to the C0-continuous curve with a boundary
//! constraint ensuring exact continuity at the piecewise threshold.
//!
//! # Accuracy (exhaustive f32 sweep)
//!
//! | Direction | Max ULP (power segment) | Max ULP (boundary) | Avg ULP |
//! |---|---|---|---|
//! | sRGB → linear | 11 | 1 | ~0.5 |
//! | linear → sRGB | 14 | 0 | ~0.4 |
//!
//! The scalar evaluator uses f64 intermediate precision, which guarantees
//! perfect monotonicity (zero reversals across all ~1B f32 values in \[0, 1\]).
//!
//! Roundtrip error is sub-U16 (< 1/65535) across the full [0, 1] range.

#[allow(unused_imports)]
use num_traits::Float; // provides mul_add/sqrt via libm in no_std

// =============================================================================
// Coefficients (lowest-degree-first: p[0] + p[1]*x + p[2]*x^2 + ...)
// Fitted to the C0-continuous (moxcms) sRGB curve with boundary constraints.
// =============================================================================

/// sRGB EOTF (encoded → linear) numerator coefficients.
pub(crate) const S2L_P: [f32; 5] = [
    1.724_942_4e-2,
    8.335_514_7e-1,
    1.326_215_8e1,
    7.033_073_4e1,
    8.387_046e1,
];

/// sRGB EOTF (encoded → linear) denominator coefficients.
pub(crate) const S2L_Q: [f32; 5] = [2.066_183e1, 9.917_607e1, 5.466_011e1, -7.183_806, 1.0];

/// sRGB inverse EOTF (linear → encoded) numerator coefficients.
/// Evaluated on `sqrt(linear)`.
pub(crate) const L2S_P: [f32; 5] = [
    -1.513_885e-2,
    1.167_372_8e-1,
    1.257_921_2e1,
    5.259_309_8e1,
    2.852_907_6e1,
];

/// sRGB inverse EOTF (linear → encoded) denominator coefficients.
/// Evaluated on `sqrt(linear)`.
pub(crate) const L2S_Q: [f32; 5] = [2.943_901_4e-1, 9.779_103, 4.726_487_7e1, 3.546_463_8e1, 1.0];

// =============================================================================
// Extended-range 6/6 coefficients — fitted to wider domains for abs+sign path
// =============================================================================

/// Extended sRGB EOTF numerator (degree 6, fitted to [threshold, 8]).
/// 5 ULP max in [0,1], u16-safe to |encoded| ≤ ~4.2 (SIMD FMA), u8-safe to 8.0.
#[allow(clippy::excessive_precision)]
pub(crate) const EXT_S2L_P: [f32; 7] = [
    1.802_136_5e1,
    9.110_411_4e2,
    1.570_602_1e4,
    1.020_638_2e5,
    2.199_931_2e5,
    1.338_269_2e5,
    1.706_519_4e4,
];

/// Extended sRGB EOTF denominator (degree 6, fitted to [threshold, 8]).
#[allow(clippy::excessive_precision)]
pub(crate) const EXT_S2L_Q: [f32; 7] = [
    2.159_401_7e4,
    1.508_555_1e5,
    2.303_299_0e5,
    8.239_410_8e4,
    4.473_249_1e3,
    -6.359_000_1e1,
    1.0,
];

/// Extended sRGB inverse EOTF numerator (degree 6, fitted on sqrt to [threshold, 64]).
/// 5 ULP max in [0,1], u16-safe across full [0, 64] domain.
/// Coefficients from polyfit (SK init + LM + f32 ULP grid search).
#[allow(clippy::excessive_precision)]
pub(crate) const EXT_L2S_P: [f32; 7] = [
    -1.025_467_4,
    -3.075_361_5e-1,
    1.027_286e3,
    7.093_665e3,
    1.006_868_9e4,
    3.230_716e3,
    1.769_130_4e2,
];

/// Extended sRGB inverse EOTF denominator (degree 6, fitted on sqrt to [threshold, 64]).
#[allow(clippy::excessive_precision)]
pub(crate) const EXT_L2S_Q: [f32; 7] = [
    1.977_460_5e1,
    8.308_271e2,
    6.024_792_5e3,
    1.024_407_5e4,
    4.157_534e3,
    3.179_324_6e2,
    1.0,
];

// =============================================================================
// C0-continuous thresholds (moxcms) — exact continuity at the piecewise join
// =============================================================================

/// sRGB linearization threshold in gamma domain (C0-continuous).
/// Values below this use the linear segment `v / 12.92`.
pub(crate) const SRGB_THRESHOLD: f32 = 0.039_293_37;

/// sRGB linearization threshold in linear domain (C0-continuous).
/// Values below this use the linear segment `v * 12.92`.
pub(crate) const LINEAR_THRESHOLD: f32 = 0.003_041_282_6;

/// Scale factor for the linear segment (1 / 12.92).
pub(crate) const LINEAR_SCALE: f32 = 1.0 / 12.92;

/// Scale factor for the inverse linear segment.
pub(crate) const TWELVE_92: f32 = 12.92;

// =============================================================================
// Scalar evaluator
// =============================================================================

/// Evaluate a degree-4 rational polynomial P(x)/Q(x) using Horner's method.
///
/// Evaluates in f64 to eliminate monotonicity violations from f32 rounding.
/// The f64→f32 cast (round-to-nearest-even) preserves monotonicity because
/// the f64 result has far more precision than needed for f32 output.
///
/// Coefficients are lowest-degree-first: `p[0] + p[1]*x + p[2]*x^2 + ...`
#[inline(always)]
fn eval_rational_poly_5(x: f32, p: [f32; 5], q: [f32; 5]) -> f32 {
    let x = x as f64;

    let yp = (p[4] as f64).mul_add(x, p[3] as f64);
    let yp = yp.mul_add(x, p[2] as f64);
    let yp = yp.mul_add(x, p[1] as f64);
    let yp = yp.mul_add(x, p[0] as f64);

    let yq = (q[4] as f64).mul_add(x, q[3] as f64);
    let yq = yq.mul_add(x, q[2] as f64);
    let yq = yq.mul_add(x, q[1] as f64);
    let yq = yq.mul_add(x, q[0] as f64);

    (yp / yq) as f32
}

/// Convert sRGB gamma-encoded value to linear light using a rational polynomial (f32).
///
/// Replaces `powf()` with a 5/5 rational polynomial (Horner's method).
/// Max error: ~8 ULP in the power segment, 1 ULP at the piecewise threshold.
/// Uses C0-continuous (moxcms) thresholds for exact continuity.
///
/// **Clamps** inputs to \[0, 1\]. For exact `powf()`, see [`crate::precise::srgb_to_linear`].
#[inline]
pub fn srgb_to_linear_fast(gamma: f32) -> f32 {
    if gamma < 0.0 {
        return 0.0;
    }
    if gamma >= 1.0 {
        return 1.0;
    }
    if gamma <= SRGB_THRESHOLD {
        return gamma * LINEAR_SCALE;
    }
    eval_rational_poly_5(gamma, S2L_P, S2L_Q)
}

/// Convert linear light value to sRGB gamma-encoded using a rational polynomial (f32).
///
/// Uses sqrt + 5/5 rational polynomial (Horner's method).
/// Max error: ~8 ULP in the power segment, 0 ULP at the piecewise threshold.
/// Uses C0-continuous (moxcms) thresholds for exact continuity.
///
/// **Clamps** inputs to \[0, 1\]. For exact `powf()`, see [`crate::precise::linear_to_srgb`].
#[inline]
pub fn linear_to_srgb_fast(linear: f32) -> f32 {
    if linear < 0.0 {
        return 0.0;
    }
    if linear >= 1.0 {
        return 1.0;
    }
    if linear <= LINEAR_THRESHOLD {
        return linear * TWELVE_92;
    }
    let s = linear.sqrt();
    eval_rational_poly_5(s, L2S_P, L2S_Q)
}
