//! PQ (SMPTE ST 2084) transfer function (scalar + generic SIMD).

use super::polynomial;
use crate::tf::polynomial::Mla;
#[allow(unused_imports)]
use num_traits::Float; // provides powf/sqrt via libm in no_std

// =============================================================================
// Scalar
// =============================================================================

/// PQ EOTF for normalized signal values. Uses the ST 2084 formula in the dark
/// toe and a two-segment rational approximation over the remainder of the range.
#[inline]
pub fn pq_to_linear(v: f32) -> f32 {
    if v.is_nan() || v <= 0.0 || v >= 1.0 || v < EOTF_POLY_FLOOR {
        return pq_to_linear_precise(v);
    }
    let x = v.mul_add(v, v);
    if v < 0.25 {
        polynomial::eval_rational_poly(x, PQ_EOTF_P_SMALL, PQ_EOTF_Q_SMALL)
    } else {
        polynomial::eval_rational_poly(x, PQ_EOTF_P_LARGE, PQ_EOTF_Q_LARGE)
    }
}

/// PQ inverse EOTF for normalized linear light. Uses the ST 2084 formula for
/// values below 1e-4 and a rational approximation above that threshold.
#[inline]
pub fn linear_to_pq(v: f32) -> f32 {
    if v.is_nan() || v <= 0.0 || v >= 1.0 || v < OETF_POLY_FLOOR {
        return linear_to_pq_precise(v);
    }
    let a = v.sqrt().sqrt();
    if a < 0.1 {
        polynomial::eval_rational_poly(a, PQ_INV_P_SMALL, PQ_INV_Q_SMALL)
    } else {
        polynomial::eval_rational_poly(a, PQ_INV_P_LARGE, PQ_INV_Q_LARGE)
    }
}

const M1: f64 = 2610.0 / 16384.0;
const M2: f64 = 2523.0 / 32.0;
const C1: f64 = 3424.0 / 4096.0;
const C2: f64 = 2413.0 / 128.0;
const C3: f64 = 2392.0 / 128.0;

const EOTF_POLY_FLOOR: f32 = 0.02;
const OETF_POLY_FLOOR: f32 = 1e-4;

#[inline]
fn pq_to_linear_precise(v: f32) -> f32 {
    if v.is_nan() {
        return v;
    }
    if v <= 0.0 {
        return 0.0;
    }
    if v >= 1.0 {
        return 1.0;
    }
    let vp = f64::from(v).powf(1.0 / M2);
    (((vp - C1).max(0.0) / (C2 - C3 * vp)).powf(1.0 / M1)) as f32
}

#[inline]
fn linear_to_pq_precise(v: f32) -> f32 {
    if v.is_nan() {
        return v;
    }
    if v <= 0.0 {
        return 0.0;
    }
    if v >= 1.0 {
        return 1.0;
    }
    let vp = f64::from(v).powf(M1);
    ((C1 + C2 * vp) / (1.0 + C3 * vp)).powf(M2) as f32
}

/// Convert normalized ST 2084 signals to linear light normalized to 10,000 nits.
/// Uses the reference formula with f64 intermediates throughout the range.
/// Inputs clip to [0, 1], NaNs propagate, and black and white are exact.
pub fn pq_to_linear_precise_slice(values: &mut [f32]) {
    for value in values {
        *value = pq_to_linear_precise(*value);
    }
}

/// Convert linear light normalized to 10,000 nits to normalized ST 2084 signals.
/// Uses the reference formula with f64 intermediates throughout the range.
/// Inputs clip to [0, 1], NaNs propagate, and zero encodes to exact zero.
pub fn linear_to_pq_precise_slice(values: &mut [f32]) {
    for value in values {
        *value = linear_to_pq_precise(*value);
    }
}

// =============================================================================
// Coefficients
// =============================================================================

// Two-range EOTF: split at signal = 0.25 for sub-U16-step roundtrip accuracy.
// Fitted with relative-error weighting (minimax in signal-space roundtrip).
pub(crate) const PQ_EOTF_P_SMALL: [f32; 5] = [
    4.36033203e-10,
    -1.53651077e-07,
    2.06864196e-06,
    2.80304711e-03,
    1.47809948e-02,
];
pub(crate) const PQ_EOTF_Q_SMALL: [f32; 5] = [
    -9.89955022e-03,
    1.89911767e+00,
    -1.34828020e+00,
    -7.10547502e-01,
    1.00000000e+00,
];
pub(crate) const PQ_EOTF_P_LARGE: [f32; 5] = [
    5.43046166e-04,
    -6.31017431e-03,
    2.90480734e-01,
    9.80363032e-01,
    2.22819483e-01,
];
pub(crate) const PQ_EOTF_Q_LARGE: [f32; 5] = [
    1.58688561e+02,
    -1.60764060e+02,
    6.51370372e+01,
    -1.26438391e+01,
    1.00000000e+00,
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

use magetypes::simd::{backends::*, generic::*};

macro_rules! pq_for {
    ($decode:ident, $encode:ident, $float:ident, $backend:ident, $rational:ident) => {
        #[inline(always)]
        pub(crate) fn $decode<T: $backend>(t: T, v: $float<T>) -> $float<T> {
            let floor = $float::splat_t(t, EOTF_POLY_FLOOR);
            let one = $float::splat_t(t, 1.0);
            // Keep the small fit above its pole even on lanes replaced below.
            let a = v.max(floor).min(one);
            let x = a.mla(a, a);
            let large = polynomial::$rational(t, x, PQ_EOTF_P_LARGE, PQ_EOTF_Q_LARGE);
            let small = polynomial::$rational(t, x, PQ_EOTF_P_SMALL, PQ_EOTF_Q_SMALL);
            let result = $float::blend(a.simd_lt($float::splat_t(t, 0.25)), small, large);
            let fallback = v.simd_lt(floor) | v.simd_ge(one) | v.simd_ne(v);
            if fallback.bitcast_to_i32().any_true() {
                let mut result = result.to_array();
                for (source, output) in v.to_array().into_iter().zip(&mut result) {
                    if source.is_nan() || !(EOTF_POLY_FLOOR..1.0).contains(&source) {
                        *output = pq_to_linear_precise(source);
                    }
                }
                $float::from_array_t(t, result)
            } else {
                result
            }
        }

        #[inline(always)]
        pub(crate) fn $encode<T: $backend>(t: T, v: $float<T>) -> $float<T> {
            let floor = $float::splat_t(t, OETF_POLY_FLOOR);
            let one = $float::splat_t(t, 1.0);
            let a = v.max(floor).min(one);
            let a4 = a.sqrt().sqrt();
            let large = polynomial::$rational(t, a4, PQ_INV_P_LARGE, PQ_INV_Q_LARGE);
            let small = polynomial::$rational(t, a4, PQ_INV_P_SMALL, PQ_INV_Q_SMALL);
            let result = $float::blend(a4.simd_lt($float::splat_t(t, 0.1)), small, large);
            let fallback = v.simd_lt(floor) | v.simd_ge(one) | v.simd_ne(v);
            if fallback.bitcast_to_i32().any_true() {
                let mut result = result.to_array();
                for (source, output) in v.to_array().into_iter().zip(&mut result) {
                    if source.is_nan() || !(OETF_POLY_FLOOR..1.0).contains(&source) {
                        *output = linear_to_pq_precise(source);
                    }
                }
                $float::from_array_t(t, result)
            } else {
                result
            }
        }
    };
}

pq_for!(
    pq_to_linear_x4,
    linear_to_pq_x4,
    f32x4,
    F32x4Convert,
    eval_rational_poly_x4
);
pq_for!(
    pq_to_linear_x8,
    linear_to_pq_x8,
    f32x8,
    F32x8Convert,
    eval_rational_poly_x8
);
pq_for!(
    pq_to_linear_x16,
    linear_to_pq_x16,
    f32x16,
    F32x16Convert,
    eval_rational_poly_x16
);
