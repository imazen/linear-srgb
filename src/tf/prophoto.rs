//! ROMM RGB / ProPhoto transfer function per ISO 22028-2: pure power
//! gamma `g=1.8` with a linear toe near black (slope `c=1/16`, encoded
//! break `d=1/32`).
//!
//! The pure-gamma form (no toe) violates ISO 22028-2; use this module for
//! ROMM/ProPhoto, and reserve [`crate::scalar::gamma_to_linear`] at
//! `gamma=1.8` only for legacy gamma-1.8 data without the linear segment.
//!
//! Scalar + generic SIMD (x4 / x8) following the BT.709 template.

use super::fast_math;

/// ROMM/ProPhoto encoding gamma (per ISO 22028-2).
const ROMM_GAMMA: f32 = 1.8;
/// Encoded-side linear-toe break: `1/32 = 0.03125`. Toe extends to `encoded < d`.
const ROMM_ENCODED_BREAK: f32 = 1.0 / 32.0;
/// Linear-side break: `(1/32)^1.8 = 1/512 = 0.001953125` (since `32^1.8 = 512`).
/// At this point the inverse curve switches from `linear * 16` to `linear^(1/g)`.
const ROMM_LINEAR_BREAK: f32 = 1.0 / 512.0;
/// Toe slope `c = 1/16` (encoded → linear in the toe).
const ROMM_TOE_SLOPE: f32 = 1.0 / 16.0;
/// Inverse toe slope `1/c = 16` (linear → encoded in the toe).
const ROMM_INV_TOE_SLOPE: f32 = 16.0;
const ROMM_INV_GAMMA: f32 = 1.0 / ROMM_GAMMA;

/// ROMM/ProPhoto EOTF: encoded → linear. Uses fast_powf, max error ~3e-5.
#[inline(always)]
pub fn prophoto_to_linear(v: f32) -> f32 {
    if v < ROMM_ENCODED_BREAK {
        v * ROMM_TOE_SLOPE
    } else {
        fast_math::fast_powf(v, ROMM_GAMMA)
    }
}

/// ROMM/ProPhoto inverse EOTF: linear → encoded. Uses fast_powf, max error ~3e-5.
#[inline(always)]
pub fn linear_to_prophoto(v: f32) -> f32 {
    if v < ROMM_LINEAR_BREAK {
        v * ROMM_INV_TOE_SLOPE
    } else {
        fast_math::fast_powf(v, ROMM_INV_GAMMA)
    }
}

// =============================================================================
// Generic SIMD — x4
// =============================================================================

use magetypes::simd::backends::F32x4Convert;
use magetypes::simd::generic::f32x4;

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn prophoto_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let threshold = f32x4::splat(t, ROMM_ENCODED_BREAK);
    let toe_slope = f32x4::splat(t, ROMM_TOE_SLOPE);

    let linear = v * toe_slope;

    let safe = v.max(f32x4::splat(t, f32::MIN_POSITIVE));
    let power = fast_math::fast_powf_x4(t, safe, ROMM_GAMMA);

    let mask = v.simd_lt(threshold);
    f32x4::blend(mask, linear, power)
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn linear_to_prophoto_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let threshold = f32x4::splat(t, ROMM_LINEAR_BREAK);
    let inv_toe_slope = f32x4::splat(t, ROMM_INV_TOE_SLOPE);

    let linear = v * inv_toe_slope;

    let safe = v.max(f32x4::splat(t, f32::MIN_POSITIVE));
    let power = fast_math::fast_powf_x4(t, safe, ROMM_INV_GAMMA);

    let mask = v.simd_lt(threshold);
    f32x4::blend(mask, linear, power)
}

// =============================================================================
// Generic SIMD — x8
// =============================================================================

use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

#[inline(always)]
pub(crate) fn prophoto_to_linear_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let threshold = f32x8::splat(t, ROMM_ENCODED_BREAK);
    let toe_slope = f32x8::splat(t, ROMM_TOE_SLOPE);

    let linear = v * toe_slope;

    let safe = v.max(f32x8::splat(t, f32::MIN_POSITIVE));
    let power = fast_math::fast_powf_x8(t, safe, ROMM_GAMMA);

    let mask = v.simd_lt(threshold);
    f32x8::blend(mask, linear, power)
}

#[inline(always)]
pub(crate) fn linear_to_prophoto_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let threshold = f32x8::splat(t, ROMM_LINEAR_BREAK);
    let inv_toe_slope = f32x8::splat(t, ROMM_INV_TOE_SLOPE);

    let linear = v * inv_toe_slope;

    let safe = v.max(f32x8::splat(t, f32::MIN_POSITIVE));
    let power = fast_math::fast_powf_x8(t, safe, ROMM_INV_GAMMA);

    let mask = v.simd_lt(threshold);
    f32x8::blend(mask, linear, power)
}
