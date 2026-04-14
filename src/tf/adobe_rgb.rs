//! Adobe RGB 1998 transfer function as encoded in real-world ICC profiles
//! (ICC `parametricCurveType funcType=3`): pure power gamma `g=2.19921875`
//! with a linear toe near black (slope `c=1/32`, encoded break `d=0.05568`).
//!
//! This is *not* what the Adobe RGB 1998 encoding spec (§4.3.4.2) defines —
//! that is pure gamma with no toe, accessible via [`crate::scalar::gamma_to_linear`]
//! at `gamma=2.19921875`. The toe form is the CMS interop convention used by
//! lcms2, saucecontrol's Compact-ICC profiles, and effectively every Adobe RGB
//! ICC profile in the wild. Use this module when you need byte-exact round-trip
//! against such profiles.
//!
//! Scalar + generic SIMD (x4 / x8) following the BT.709 template.

use super::fast_math;

/// Adobe RGB encoding gamma (= 563/256).
const ADOBE_GAMMA: f32 = 2.19921875;
/// Encoded-side linear-toe break: `c * (encoded threshold) = (encoded threshold)^g`.
/// Per ICC paraType-3, the toe extends to `encoded < d`. Profiles in the wild
/// quantize `d` to s15.16 fixed-point as `0x0000_0E40`; the f32 equivalent is
/// `0.05568`. (The exact mathematical break for `c=1/32`, `g=2.19921875` is
/// `d ≈ 0.05568...`.)
const ADOBE_ENCODED_BREAK: f32 = 0.05568;
/// Linear-side break = `ADOBE_ENCODED_BREAK / 32`, i.e. the linear value at
/// which the inverse curve switches from `linear * 32` to `linear^(1/g)`.
const ADOBE_LINEAR_BREAK: f32 = ADOBE_ENCODED_BREAK / 32.0;
/// Toe slope `c = 1/32` (encoded → linear in the toe).
const ADOBE_TOE_SLOPE: f32 = 1.0 / 32.0;
/// Inverse toe slope `1/c = 32` (linear → encoded in the toe).
const ADOBE_INV_TOE_SLOPE: f32 = 32.0;
const ADOBE_INV_GAMMA: f32 = 1.0 / ADOBE_GAMMA;

/// Adobe RGB EOTF: encoded → linear. Uses fast_powf, max error ~3e-5.
#[inline(always)]
pub fn adobe_rgb_to_linear(v: f32) -> f32 {
    if v < ADOBE_ENCODED_BREAK {
        v * ADOBE_TOE_SLOPE
    } else {
        fast_math::fast_powf(v, ADOBE_GAMMA)
    }
}

/// Adobe RGB inverse EOTF: linear → encoded. Uses fast_powf, max error ~3e-5.
#[inline(always)]
pub fn linear_to_adobe_rgb(v: f32) -> f32 {
    if v < ADOBE_LINEAR_BREAK {
        v * ADOBE_INV_TOE_SLOPE
    } else {
        fast_math::fast_powf(v, ADOBE_INV_GAMMA)
    }
}

// =============================================================================
// Generic SIMD — x4
// =============================================================================

use magetypes::simd::backends::F32x4Convert;
use magetypes::simd::generic::f32x4;

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn adobe_rgb_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let threshold = f32x4::splat(t, ADOBE_ENCODED_BREAK);
    let toe_slope = f32x4::splat(t, ADOBE_TOE_SLOPE);

    let linear = v * toe_slope;

    let safe = v.max(f32x4::splat(t, f32::MIN_POSITIVE));
    let power = fast_math::fast_powf_x4(t, safe, ADOBE_GAMMA);

    let mask = v.simd_lt(threshold);
    f32x4::blend(mask, linear, power)
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn linear_to_adobe_rgb_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let threshold = f32x4::splat(t, ADOBE_LINEAR_BREAK);
    let inv_toe_slope = f32x4::splat(t, ADOBE_INV_TOE_SLOPE);

    let linear = v * inv_toe_slope;

    let safe = v.max(f32x4::splat(t, f32::MIN_POSITIVE));
    let power = fast_math::fast_powf_x4(t, safe, ADOBE_INV_GAMMA);

    let mask = v.simd_lt(threshold);
    f32x4::blend(mask, linear, power)
}

// =============================================================================
// Generic SIMD — x8
// =============================================================================

use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

#[inline(always)]
pub(crate) fn adobe_rgb_to_linear_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let threshold = f32x8::splat(t, ADOBE_ENCODED_BREAK);
    let toe_slope = f32x8::splat(t, ADOBE_TOE_SLOPE);

    let linear = v * toe_slope;

    let safe = v.max(f32x8::splat(t, f32::MIN_POSITIVE));
    let power = fast_math::fast_powf_x8(t, safe, ADOBE_GAMMA);

    let mask = v.simd_lt(threshold);
    f32x8::blend(mask, linear, power)
}

#[inline(always)]
pub(crate) fn linear_to_adobe_rgb_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let threshold = f32x8::splat(t, ADOBE_LINEAR_BREAK);
    let inv_toe_slope = f32x8::splat(t, ADOBE_INV_TOE_SLOPE);

    let linear = v * inv_toe_slope;

    let safe = v.max(f32x8::splat(t, f32::MIN_POSITIVE));
    let power = fast_math::fast_powf_x8(t, safe, ADOBE_INV_GAMMA);

    let mask = v.simd_lt(threshold);
    f32x8::blend(mask, linear, power)
}
