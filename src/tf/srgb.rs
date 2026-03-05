//! sRGB transfer function — generic SIMD wrappers.
//!
//! Scalar sRGB delegates to `crate::rational_poly` (libjxl rational polynomial).
//! SIMD versions use the same polynomial coefficients inline.

use crate::rational_poly;

use magetypes::simd::backends::F32x4Convert;
use magetypes::simd::generic::f32x4;

#[inline(always)]
pub(crate) fn srgb_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let threshold = f32x4::splat(t, rational_poly::SRGB_THRESHOLD);
    let inv_12_92 = f32x4::splat(t, rational_poly::LINEAR_SCALE);

    let linear = v * inv_12_92;
    let poly =
        super::fast_math::eval_rational_poly_x4(t, v, rational_poly::S2L_P, rational_poly::S2L_Q);

    let mask = v.simd_le(threshold);
    f32x4::blend(mask, linear, poly)
}

#[inline(always)]
pub(crate) fn linear_to_srgb_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let threshold = f32x4::splat(t, rational_poly::LINEAR_THRESHOLD);
    let scale = f32x4::splat(t, rational_poly::TWELVE_92);

    let linear = v * scale;
    let s = v.sqrt();
    let poly =
        super::fast_math::eval_rational_poly_x4(t, s, rational_poly::L2S_P, rational_poly::L2S_Q);

    let mask = v.simd_le(threshold);
    f32x4::blend(mask, linear, poly)
}

use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

#[inline(always)]
pub(crate) fn srgb_to_linear_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let threshold = f32x8::splat(t, rational_poly::SRGB_THRESHOLD);
    let inv_12_92 = f32x8::splat(t, rational_poly::LINEAR_SCALE);

    let linear = v * inv_12_92;
    let poly =
        super::fast_math::eval_rational_poly_x8(t, v, rational_poly::S2L_P, rational_poly::S2L_Q);

    let mask = v.simd_le(threshold);
    f32x8::blend(mask, linear, poly)
}

#[inline(always)]
pub(crate) fn linear_to_srgb_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let threshold = f32x8::splat(t, rational_poly::LINEAR_THRESHOLD);
    let scale = f32x8::splat(t, rational_poly::TWELVE_92);

    let linear = v * scale;
    let s = v.sqrt();
    let poly =
        super::fast_math::eval_rational_poly_x8(t, s, rational_poly::L2S_P, rational_poly::L2S_Q);

    let mask = v.simd_le(threshold);
    f32x8::blend(mask, linear, poly)
}
