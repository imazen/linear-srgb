//! Rational polynomial sRGB transfer function approximation.
//!
//! Single source of truth for all polynomial constants used across scalar,
//! SIMD, magetypes rites (x4/x8/x16), and mage implementations.
//!
//! Coefficients from libjxl (BSD-3-Clause). Evaluates P(x)/Q(x) via Horner's
//! method: 4 `mul_add` + 1 `div` per direction. Much faster and more accurate
//! than the degree-11/15 Chebyshev polynomials they replace.
//!
//! # Accuracy (exhaustive f32 sweep)
//!
//! | Direction | Max ULP (power segment) | Max ULP (overall) | Avg ULP |
//! |---|---|---|---|
//! | sRGB → linear | ~8 | 110 (at threshold) | 0.55 |
//! | linear → sRGB | ~8 | 31 (at threshold) | 0.37 |
//!
//! The overall max occurs at the piecewise threshold where the linear segment
//! meets the polynomial. Away from the threshold, error is <8 ULP.

// =============================================================================
// Coefficients (lowest-degree-first: p[0] + p[1]*x + p[2]*x^2 + ...)
// =============================================================================

/// sRGB EOTF (encoded → linear) numerator coefficients.
pub(crate) const S2L_P: [f32; 5] = [
    2.200_248_3e-4,
    1.043_637_6e-2,
    1.624_820_4e-1,
    7.961_565e-1,
    8.210_153e-1,
];

/// sRGB EOTF (encoded → linear) denominator coefficients.
pub(crate) const S2L_Q: [f32; 5] = [
    2.631_847e-1,
    1.076_976_5,
    4.987_528_3e-1,
    -5.512_498_3e-2,
    6.521_209e-3,
];

/// sRGB inverse EOTF (linear → encoded) numerator coefficients.
/// Evaluated on `sqrt(linear)`.
pub(crate) const L2S_P: [f32; 5] = [
    -5.135_152_6e-4,
    5.287_254_7e-3,
    3.903_843e-1,
    1.474_205_3,
    7.352_63e-1,
];

/// sRGB inverse EOTF (linear → encoded) denominator coefficients.
/// Evaluated on `sqrt(linear)`.
pub(crate) const L2S_Q: [f32; 5] = [
    1.004_519_6e-2,
    3.036_675_5e-1,
    1.340_817,
    9.258_482e-1,
    2.424_867_8e-2,
];

// =============================================================================
// IEC 61966-2-1 thresholds (matching the curve the polynomial approximates)
// =============================================================================

/// sRGB linearization threshold in gamma domain (IEC 61966-2-1).
/// Values below this use the linear segment `v / 12.92`.
pub(crate) const SRGB_THRESHOLD: f32 = 0.04045;

/// sRGB linearization threshold in linear domain (IEC 61966-2-1).
/// Values below this use the linear segment `v * 12.92`.
pub(crate) const LINEAR_THRESHOLD: f32 = 0.003_130_8;

/// Scale factor for the linear segment (1 / 12.92).
pub(crate) const LINEAR_SCALE: f32 = 1.0 / 12.92;

/// Scale factor for the inverse linear segment.
pub(crate) const TWELVE_92: f32 = 12.92;

// =============================================================================
// Scalar evaluator
// =============================================================================

/// Evaluate a degree-4 rational polynomial P(x)/Q(x) using Horner's method.
///
/// Coefficients are lowest-degree-first: `p[0] + p[1]*x + p[2]*x^2 + ...`
#[inline(always)]
fn eval_rational_poly_5(x: f32, p: [f32; 5], q: [f32; 5]) -> f32 {
    let yp = p[4].mul_add(x, p[3]);
    let yp = yp.mul_add(x, p[2]);
    let yp = yp.mul_add(x, p[1]);
    let yp = yp.mul_add(x, p[0]);

    let yq = q[4].mul_add(x, q[3]);
    let yq = yq.mul_add(x, q[2]);
    let yq = yq.mul_add(x, q[1]);
    let yq = yq.mul_add(x, q[0]);

    yp / yq
}

/// Convert sRGB gamma-encoded value to linear light using a rational polynomial (f32).
///
/// Replaces `powf()` with a 5/5 rational polynomial (Horner's method).
/// Max error: 110 ULP at the piecewise threshold, <8 ULP elsewhere.
/// Uses IEC 61966-2-1 thresholds (the polynomial was fitted to the IEC power curve).
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
/// Max error: 31 ULP at the piecewise threshold, <8 ULP elsewhere.
/// Uses IEC 61966-2-1 thresholds (the polynomial was fitted to the IEC power curve).
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
