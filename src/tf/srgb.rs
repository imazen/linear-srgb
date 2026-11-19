//! sRGB transfer function — generic SIMD wrappers.
//!
//! Scalar sRGB delegates to `crate::rational_poly` (C0-continuous rational polynomial).
//! SIMD versions use the same polynomial coefficients inline.

use magetypes::simd::backends::F32x4Convert;
use magetypes::simd::generic::f32x4;

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn srgb_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    f32x4::from_array_t(
        t,
        crate::kernels::float::x4::srgb_to_linear(t, v.to_array()),
    )
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn linear_to_srgb_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    f32x4::from_array_t(
        t,
        crate::kernels::float::x4::linear_to_srgb(t, v.to_array()),
    )
}

use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

#[inline(always)]
pub(crate) fn srgb_to_linear_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    f32x8::from_array_t(
        t,
        crate::kernels::float::x8::srgb_to_linear(t, v.to_array()),
    )
}

#[inline(always)]
pub(crate) fn linear_to_srgb_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    f32x8::from_array_t(
        t,
        crate::kernels::float::x8::linear_to_srgb(t, v.to_array()),
    )
}
