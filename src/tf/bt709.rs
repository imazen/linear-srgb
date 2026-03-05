//! BT.709 transfer function (scalar + generic SIMD).

use super::fast_math;

const BT709_ALPHA: f32 = 0.09929682680944;
const BT709_BETA: f32 = 0.018053968510807;

/// BT.709 EOTF: encoded → linear. Uses fast_powf, max error ~3e-6.
#[inline(always)]
pub fn bt709_to_linear(v: f32) -> f32 {
    if v < 4.5 * BT709_BETA {
        v / 4.5
    } else {
        fast_math::fast_powf((v + BT709_ALPHA) / (1.0 + BT709_ALPHA), 1.0 / 0.45)
    }
}

/// BT.709 inverse EOTF: linear → encoded. Uses fast_powf, max error ~3e-5.
#[inline(always)]
pub fn linear_to_bt709(v: f32) -> f32 {
    if v < BT709_BETA {
        4.5 * v
    } else {
        (1.0 + BT709_ALPHA) * fast_math::fast_powf(v, 0.45) - BT709_ALPHA
    }
}

// =============================================================================
// Generic SIMD — x4
// =============================================================================

use magetypes::simd::backends::F32x4Convert;
use magetypes::simd::generic::f32x4;

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn bt709_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let threshold = f32x4::splat(t, 4.5 * BT709_BETA);
    let inv_4_5 = f32x4::splat(t, 1.0 / 4.5);
    let alpha = f32x4::splat(t, BT709_ALPHA);
    let one_plus_alpha = f32x4::splat(t, 1.0 + BT709_ALPHA);

    let linear = v * inv_4_5;

    let normalized = (v + alpha) / one_plus_alpha;
    let safe = normalized.max(f32x4::splat(t, f32::MIN_POSITIVE));
    let power = fast_math::fast_powf_x4(t, safe, 1.0 / 0.45);

    let mask = v.simd_lt(threshold);
    f32x4::blend(mask, linear, power)
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn linear_to_bt709_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let threshold = f32x4::splat(t, BT709_BETA);
    let scale_4_5 = f32x4::splat(t, 4.5);
    let one_plus_alpha = f32x4::splat(t, 1.0 + BT709_ALPHA);
    let alpha = f32x4::splat(t, BT709_ALPHA);

    let linear = v * scale_4_5;

    let safe = v.max(f32x4::splat(t, f32::MIN_POSITIVE));
    let power = fast_math::fast_powf_x4(t, safe, 0.45);
    let power = one_plus_alpha.mul_add(power, -alpha);

    let mask = v.simd_lt(threshold);
    f32x4::blend(mask, linear, power)
}

// =============================================================================
// Generic SIMD — x8
// =============================================================================

use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

#[inline(always)]
pub(crate) fn bt709_to_linear_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let threshold = f32x8::splat(t, 4.5 * BT709_BETA);
    let inv_4_5 = f32x8::splat(t, 1.0 / 4.5);
    let alpha = f32x8::splat(t, BT709_ALPHA);
    let one_plus_alpha = f32x8::splat(t, 1.0 + BT709_ALPHA);

    let linear = v * inv_4_5;

    let normalized = (v + alpha) / one_plus_alpha;
    let safe = normalized.max(f32x8::splat(t, f32::MIN_POSITIVE));
    let power = fast_math::fast_powf_x8(t, safe, 1.0 / 0.45);

    let mask = v.simd_lt(threshold);
    f32x8::blend(mask, linear, power)
}

#[inline(always)]
pub(crate) fn linear_to_bt709_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let threshold = f32x8::splat(t, BT709_BETA);
    let scale_4_5 = f32x8::splat(t, 4.5);
    let one_plus_alpha = f32x8::splat(t, 1.0 + BT709_ALPHA);
    let alpha = f32x8::splat(t, BT709_ALPHA);

    let linear = v * scale_4_5;

    let safe = v.max(f32x8::splat(t, f32::MIN_POSITIVE));
    let power = fast_math::fast_powf_x8(t, safe, 0.45);
    let power = one_plus_alpha.mul_add(power, -alpha);

    let mask = v.simd_lt(threshold);
    f32x8::blend(mask, linear, power)
}
