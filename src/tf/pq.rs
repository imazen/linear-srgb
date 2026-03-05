//! PQ (SMPTE ST 2084) transfer function (scalar + generic SIMD).

use super::fast_math;

// =============================================================================
// Scalar
// =============================================================================

/// PQ EOTF: signal → linear. Rational polynomial, max error ~7e-7.
///
/// Zero `powf()` calls for signal > 0.02 — uses `x + x*x` input transformation
/// and rational polynomial. Very small signal values use exact formula.
/// Coefficients from libjxl.
#[inline(always)]
pub fn pq_to_linear(v: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    if v < 0.02 {
        return pq_to_linear_exact(v);
    }
    let a = v;
    let x = a + a * a;
    fast_math::eval_rational_poly(x, PQ_EOTF_P, PQ_EOTF_Q)
}

/// PQ inverse EOTF: linear → signal. Rational polynomial on x^(1/4), max error ~3e-6.
///
/// Two-range approximation. Very small values (v < 1e-4) use exact powf.
/// Coefficients from libjxl.
#[inline(always)]
pub fn linear_to_pq(v: f32) -> f32 {
    if v <= 0.0 {
        return 0.0;
    }
    if v < 1e-4 {
        return pq_from_linear_exact(v);
    }
    let a = v.sqrt().sqrt();
    if a < 0.1 {
        fast_math::eval_rational_poly(a, PQ_INV_P_SMALL, PQ_INV_Q_SMALL)
    } else {
        fast_math::eval_rational_poly(a, PQ_INV_P_LARGE, PQ_INV_Q_LARGE)
    }
}

#[inline(always)]
fn pq_to_linear_exact(v: f32) -> f32 {
    const M1: f32 = 0.1593017578125;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.8515625;
    const C3: f32 = 18.6875;
    let vp = v.powf(1.0 / M2);
    let num = (vp - C1).max(0.0);
    let den = C2 - C3 * vp;
    if den <= 0.0 {
        return 1.0;
    }
    (num / den).powf(1.0 / M1)
}

#[inline(always)]
fn pq_from_linear_exact(v: f32) -> f32 {
    const M1: f32 = 0.1593017578125;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.8515625;
    const C3: f32 = 18.6875;
    let vp = v.powf(M1);
    let num = C1 + C2 * vp;
    let den = 1.0 + C3 * vp;
    (num / den).powf(M2)
}

// =============================================================================
// Coefficients
// =============================================================================

pub(crate) const PQ_EOTF_P: [f32; 5] = [
    2.6297566e-4,
    -6.235531e-3,
    7.386023e-1,
    2.6455317,
    5.500349e-1,
];
pub(crate) const PQ_EOTF_Q: [f32; 5] = [
    4.213501e2,
    -4.2873682e2,
    1.7436467e2,
    -3.3907887e1,
    2.6771877,
];

pub(crate) const PQ_INV_P_LARGE: [f32; 5] =
    [1.351392e-2, -1.095778, 5.522776e1, 1.492516e2, 4.838434e1];
pub(crate) const PQ_INV_Q_LARGE: [f32; 5] =
    [1.012416, 2.016708e1, 9.26371e1, 1.120607e2, 2.590418e1];

pub(crate) const PQ_INV_P_SMALL: [f32; 5] = [
    9.863406e-6,
    3.881234e-1,
    1.352821e2,
    6.889862e4,
    -2.864824e5,
];
pub(crate) const PQ_INV_Q_SMALL: [f32; 5] =
    [3.371868e1, 1.477719e3, 1.608477e4, -4.389884e4, -2.072546e5];

// =============================================================================
// Generic SIMD — x4
// =============================================================================

use magetypes::simd::backends::F32x4Convert;
use magetypes::simd::generic::f32x4;

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn pq_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let zero = f32x4::zero(t);
    let a = v.max(zero);
    let x = a.mul_add(a, a);
    let result = fast_math::eval_rational_poly_x4(t, x, PQ_EOTF_P, PQ_EOTF_Q);
    let mask = v.simd_gt(zero);
    result & mask
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn linear_to_pq_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let zero = f32x4::zero(t);
    let a = v.max(zero);
    let a4 = a.sqrt().sqrt();

    let threshold = f32x4::splat(t, 0.1);
    let large = fast_math::eval_rational_poly_x4(t, a4, PQ_INV_P_LARGE, PQ_INV_Q_LARGE);
    let small = fast_math::eval_rational_poly_x4(t, a4, PQ_INV_P_SMALL, PQ_INV_Q_SMALL);

    let mask = a4.simd_lt(threshold);
    let result = f32x4::blend(mask, small, large);

    let pos_mask = v.simd_gt(zero);
    result & pos_mask
}

// =============================================================================
// Generic SIMD — x8
// =============================================================================

use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

#[inline(always)]
pub(crate) fn pq_to_linear_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let zero = f32x8::zero(t);
    let a = v.max(zero);
    let x = a.mul_add(a, a);
    let result = fast_math::eval_rational_poly_x8(t, x, PQ_EOTF_P, PQ_EOTF_Q);
    let mask = v.simd_gt(zero);
    result & mask
}

#[inline(always)]
pub(crate) fn linear_to_pq_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let zero = f32x8::zero(t);
    let a = v.max(zero);
    let a4 = a.sqrt().sqrt();

    let threshold = f32x8::splat(t, 0.1);
    let large = fast_math::eval_rational_poly_x8(t, a4, PQ_INV_P_LARGE, PQ_INV_Q_LARGE);
    let small = fast_math::eval_rational_poly_x8(t, a4, PQ_INV_P_SMALL, PQ_INV_Q_SMALL);

    let mask = a4.simd_lt(threshold);
    let result = f32x8::blend(mask, small, large);

    let pos_mask = v.simd_gt(zero);
    result & pos_mask
}
