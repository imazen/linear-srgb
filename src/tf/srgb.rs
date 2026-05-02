//! sRGB transfer function — generic SIMD wrappers.
//!
//! Scalar sRGB delegates to `crate::rational_poly` (C0-continuous rational polynomial).
//! SIMD versions use the same polynomial coefficients inline.

use crate::rational_poly;

use magetypes::simd::backends::F32x4Convert;
use magetypes::simd::generic::f32x4;

#[allow(dead_code)]
#[inline(always)]
pub fn srgb_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let one = f32x4::splat(t, 1.0);
    let threshold = f32x4::splat(t, rational_poly::SRGB_THRESHOLD);
    let inv_12_92 = f32x4::splat(t, rational_poly::LINEAR_SCALE);

    let linear = v * inv_12_92;
    let poly =
        super::fast_math::eval_rational_poly_x4(t, v, rational_poly::S2L_P, rational_poly::S2L_Q)
            .min(one);

    let mask = v.simd_le(threshold);
    f32x4::blend(mask, linear, poly)
}

#[allow(dead_code)]
#[inline(always)]
pub fn linear_to_srgb_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let one = f32x4::splat(t, 1.0);
    let threshold = f32x4::splat(t, rational_poly::LINEAR_THRESHOLD);
    let scale = f32x4::splat(t, rational_poly::TWELVE_92);

    let linear = v * scale;
    let s = v.sqrt();
    let poly =
        super::fast_math::eval_rational_poly_x4(t, s, rational_poly::L2S_P, rational_poly::L2S_Q)
            .min(one);

    let mask = v.simd_le(threshold);
    f32x4::blend(mask, linear, poly)
}

use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

#[inline(always)]
pub fn srgb_to_linear_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let one = f32x8::splat(t, 1.0);
    let threshold = f32x8::splat(t, rational_poly::SRGB_THRESHOLD);
    let inv_12_92 = f32x8::splat(t, rational_poly::LINEAR_SCALE);

    let linear = v * inv_12_92;
    let poly =
        super::fast_math::eval_rational_poly_x8(t, v, rational_poly::S2L_P, rational_poly::S2L_Q)
            .min(one);

    let mask = v.simd_le(threshold);
    f32x8::blend(mask, linear, poly)
}

#[inline(always)]
pub fn linear_to_srgb_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let one = f32x8::splat(t, 1.0);
    let threshold = f32x8::splat(t, rational_poly::LINEAR_THRESHOLD);
    let scale = f32x8::splat(t, rational_poly::TWELVE_92);

    let linear = v * scale;
    let s = v.sqrt();
    let poly =
        super::fast_math::eval_rational_poly_x8(t, s, rational_poly::L2S_P, rational_poly::L2S_Q)
            .min(one);

    let mask = v.simd_le(threshold);
    f32x8::blend(mask, linear, poly)
}

// =============================================================================
// Clamped sRGB — x16
// =============================================================================

use magetypes::simd::backends::F32x16Convert;
use magetypes::simd::generic::f32x16;

#[inline(always)]
pub fn srgb_to_linear_x16<T: F32x16Convert>(t: T, v: f32x16<T>) -> f32x16<T> {
    let one = f32x16::splat(t, 1.0);
    let threshold = f32x16::splat(t, rational_poly::SRGB_THRESHOLD);
    let inv_12_92 = f32x16::splat(t, rational_poly::LINEAR_SCALE);

    let linear = v * inv_12_92;
    let poly =
        super::fast_math::eval_rational_poly_x16(t, v, rational_poly::S2L_P, rational_poly::S2L_Q)
            .min(one);

    let mask = v.simd_le(threshold);
    f32x16::blend(mask, linear, poly)
}

#[inline(always)]
pub fn linear_to_srgb_x16<T: F32x16Convert>(t: T, v: f32x16<T>) -> f32x16<T> {
    let one = f32x16::splat(t, 1.0);
    let threshold = f32x16::splat(t, rational_poly::LINEAR_THRESHOLD);
    let scale = f32x16::splat(t, rational_poly::TWELVE_92);

    let linear = v * scale;
    let s = v.sqrt();
    let poly =
        super::fast_math::eval_rational_poly_x16(t, s, rational_poly::L2S_P, rational_poly::L2S_Q)
            .min(one);

    let mask = v.simd_le(threshold);
    f32x16::blend(mask, linear, poly)
}

// =============================================================================
// Extended-range (sign-preserving, 6/6 rational polynomial) — x16
// =============================================================================

use magetypes::simd::backends::F32x16Backend;

/// Extended-range sRGB→linear, 16-wide. `sign(v) * eotf(|v|)` per CSS Color 4.
#[inline(always)]
pub fn srgb_to_linear_extended_x16<T: F32x16Backend>(t: T, v: f32x16<T>) -> f32x16<T> {
    use rational_poly::{EXT_S2L_P as P, EXT_S2L_Q as Q};
    let zero = f32x16::zero(t);
    let neg_mask = v.simd_lt(zero);
    let abs_v = v.abs();

    let linear_result = abs_v * f32x16::splat(t, rational_poly::LINEAR_SCALE);

    let x = abs_v;
    let yp = f32x16::splat(t, P[6]).mul_add(x, f32x16::splat(t, P[5]));
    let yp = yp.mul_add(x, f32x16::splat(t, P[4]));
    let yp = yp.mul_add(x, f32x16::splat(t, P[3]));
    let yp = yp.mul_add(x, f32x16::splat(t, P[2]));
    let yp = yp.mul_add(x, f32x16::splat(t, P[1]));
    let yp = yp.mul_add(x, f32x16::splat(t, P[0]));

    let yq = f32x16::splat(t, Q[6]).mul_add(x, f32x16::splat(t, Q[5]));
    let yq = yq.mul_add(x, f32x16::splat(t, Q[4]));
    let yq = yq.mul_add(x, f32x16::splat(t, Q[3]));
    let yq = yq.mul_add(x, f32x16::splat(t, Q[2]));
    let yq = yq.mul_add(x, f32x16::splat(t, Q[1]));
    let yq = yq.mul_add(x, f32x16::splat(t, Q[0]));

    let power_result = yp / yq;

    let thresh_mask = abs_v.simd_lt(f32x16::splat(t, rational_poly::SRGB_THRESHOLD));
    let result = f32x16::blend(thresh_mask, linear_result, power_result);
    f32x16::blend(neg_mask, -result, result)
}

/// Extended-range linear→sRGB, 16-wide. `sign(v) * oetf(|v|)` per CSS Color 4.
#[inline(always)]
pub fn linear_to_srgb_extended_x16<T: F32x16Backend>(t: T, v: f32x16<T>) -> f32x16<T> {
    use rational_poly::{EXT_L2S_P as P, EXT_L2S_Q as Q};
    let zero = f32x16::zero(t);
    let neg_mask = v.simd_lt(zero);
    let abs_v = v.abs();

    let linear_result = abs_v * f32x16::splat(t, rational_poly::TWELVE_92);

    let x = abs_v.sqrt();
    let yp = f32x16::splat(t, P[6]).mul_add(x, f32x16::splat(t, P[5]));
    let yp = yp.mul_add(x, f32x16::splat(t, P[4]));
    let yp = yp.mul_add(x, f32x16::splat(t, P[3]));
    let yp = yp.mul_add(x, f32x16::splat(t, P[2]));
    let yp = yp.mul_add(x, f32x16::splat(t, P[1]));
    let yp = yp.mul_add(x, f32x16::splat(t, P[0]));

    let yq = f32x16::splat(t, Q[6]).mul_add(x, f32x16::splat(t, Q[5]));
    let yq = yq.mul_add(x, f32x16::splat(t, Q[4]));
    let yq = yq.mul_add(x, f32x16::splat(t, Q[3]));
    let yq = yq.mul_add(x, f32x16::splat(t, Q[2]));
    let yq = yq.mul_add(x, f32x16::splat(t, Q[1]));
    let yq = yq.mul_add(x, f32x16::splat(t, Q[0]));

    let power_result = yp / yq;

    let thresh_mask = abs_v.simd_lt(f32x16::splat(t, rational_poly::LINEAR_THRESHOLD));
    let result = f32x16::blend(thresh_mask, linear_result, power_result);
    f32x16::blend(neg_mask, -result, result)
}
