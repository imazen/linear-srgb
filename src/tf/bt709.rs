//! BT.709 transfer function (scalar + generic SIMD).

use super::math;
use crate::tf::polynomial::Mla;
use magemath::SimdMath;
#[allow(unused_imports)]
use num_traits::Float;

pub(crate) const BT709_ALPHA: f32 = 0.09929682680944;
pub(crate) const BT709_BETA: f32 = 0.018053968510807;

/// BT.709 EOTF: encoded → linear. Uses the configured Magemath power profile.
#[inline(always)]
pub fn bt709_to_linear(v: f32) -> f32 {
    if v < 4.5 * BT709_BETA {
        v * (1.0 / 4.5)
    } else {
        math::fixed_pow::<{ math::BT709_DECODE_POWER }>(
            (v + BT709_ALPHA) / (1.0 + BT709_ALPHA),
            1.0 / 0.45,
        )
        .max(BT709_BETA)
    }
}

/// BT.709 inverse EOTF: linear → encoded. Uses the configured Magemath power profile.
#[inline(always)]
pub fn linear_to_bt709(v: f32) -> f32 {
    if v < BT709_BETA {
        4.5 * v
    } else {
        (1.0 + BT709_ALPHA).mul_add(
            math::fixed_pow::<{ math::BT709_ENCODE_POWER }>(v, 0.45),
            -BT709_ALPHA,
        )
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
    let threshold = f32x4::splat_t(t, 4.5 * BT709_BETA);
    let inv_4_5 = f32x4::splat_t(t, 1.0 / 4.5);
    let alpha = f32x4::splat_t(t, BT709_ALPHA);
    let one_plus_alpha = f32x4::splat_t(t, 1.0 + BT709_ALPHA);

    let linear = v * inv_4_5;

    let normalized = (v + alpha) / one_plus_alpha;
    let mask = v.simd_lt(threshold);
    let input = f32x4::blend(mask, f32x4::splat_t(t, 1.0), normalized);
    let Ok(power) = input.try_powf_const::<{ math::BT709_DECODE_POWER }>() else {
        return f32x4::from_array_t(t, v.to_array().map(bt709_to_linear));
    };

    // The curves meet only to within rounding: keep the power branch above the linear one.
    let power = power.max(f32x4::splat_t(t, (4.5 * BT709_BETA) * (1.0 / 4.5)));
    f32x4::blend(mask, linear, power)
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn linear_to_bt709_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let threshold = f32x4::splat_t(t, BT709_BETA);
    let scale_4_5 = f32x4::splat_t(t, 4.5);
    let one_plus_alpha = f32x4::splat_t(t, 1.0 + BT709_ALPHA);
    let alpha = f32x4::splat_t(t, BT709_ALPHA);

    let linear = v * scale_4_5;

    let mask = v.simd_lt(threshold);
    let input = f32x4::blend(mask, f32x4::splat_t(t, 1.0), v);
    let Ok(power) = input.try_powf_const::<{ math::BT709_ENCODE_POWER }>() else {
        return f32x4::from_array_t(t, v.to_array().map(linear_to_bt709));
    };
    let power = one_plus_alpha.mla(power, -alpha);

    f32x4::blend(mask, linear, power)
}

// =============================================================================
// Generic SIMD — x8
// =============================================================================

use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

#[inline(always)]
pub(crate) fn bt709_to_linear_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let threshold = f32x8::splat_t(t, 4.5 * BT709_BETA);
    let inv_4_5 = f32x8::splat_t(t, 1.0 / 4.5);
    let alpha = f32x8::splat_t(t, BT709_ALPHA);
    let one_plus_alpha = f32x8::splat_t(t, 1.0 + BT709_ALPHA);

    let linear = v * inv_4_5;

    let normalized = (v + alpha) / one_plus_alpha;
    let mask = v.simd_lt(threshold);
    let input = f32x8::blend(mask, f32x8::splat_t(t, 1.0), normalized);
    let Ok(power) = input.try_powf_const::<{ math::BT709_DECODE_POWER }>() else {
        return f32x8::from_array_t(t, v.to_array().map(bt709_to_linear));
    };

    // The curves meet only to within rounding: keep the power branch above the linear one.
    let power = power.max(f32x8::splat_t(t, (4.5 * BT709_BETA) * (1.0 / 4.5)));
    f32x8::blend(mask, linear, power)
}

#[inline(always)]
pub(crate) fn linear_to_bt709_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let threshold = f32x8::splat_t(t, BT709_BETA);
    let scale_4_5 = f32x8::splat_t(t, 4.5);
    let one_plus_alpha = f32x8::splat_t(t, 1.0 + BT709_ALPHA);
    let alpha = f32x8::splat_t(t, BT709_ALPHA);

    let linear = v * scale_4_5;

    // See linear_to_bt709_x4: no MIN_POSITIVE clamp needed.
    let mask = v.simd_lt(threshold);
    let input = f32x8::blend(mask, f32x8::splat_t(t, 1.0), v);
    let Ok(power) = input.try_powf_const::<{ math::BT709_ENCODE_POWER }>() else {
        return f32x8::from_array_t(t, v.to_array().map(linear_to_bt709));
    };
    let power = one_plus_alpha.mla(power, -alpha);

    f32x8::blend(mask, linear, power)
}

// =============================================================================
// Generic SIMD — x16
// =============================================================================

use magetypes::simd::backends::F32x16Convert;
use magetypes::simd::generic::f32x16;

#[inline(always)]
pub(crate) fn bt709_to_linear_x16<T: F32x16Convert>(t: T, v: f32x16<T>) -> f32x16<T> {
    let threshold = f32x16::splat_t(t, 4.5 * BT709_BETA);
    let inv_4_5 = f32x16::splat_t(t, 1.0 / 4.5);
    let alpha = f32x16::splat_t(t, BT709_ALPHA);
    let one_plus_alpha = f32x16::splat_t(t, 1.0 + BT709_ALPHA);

    let linear = v * inv_4_5;

    let normalized = (v + alpha) / one_plus_alpha;
    let mask = v.simd_lt(threshold);
    let input = f32x16::blend(mask, f32x16::splat_t(t, 1.0), normalized);
    let Ok(power) = input.try_powf_const::<{ math::BT709_DECODE_POWER }>() else {
        return f32x16::from_array_t(t, v.to_array().map(bt709_to_linear));
    };

    // The curves meet only to within rounding: keep the power branch above the linear one.
    let power = power.max(f32x16::splat_t(t, (4.5 * BT709_BETA) * (1.0 / 4.5)));
    f32x16::blend(mask, linear, power)
}

#[inline(always)]
pub(crate) fn linear_to_bt709_x16<T: F32x16Convert>(t: T, v: f32x16<T>) -> f32x16<T> {
    let threshold = f32x16::splat_t(t, BT709_BETA);
    let scale_4_5 = f32x16::splat_t(t, 4.5);
    let one_plus_alpha = f32x16::splat_t(t, 1.0 + BT709_ALPHA);
    let alpha = f32x16::splat_t(t, BT709_ALPHA);

    let linear = v * scale_4_5;

    // See linear_to_bt709_x4: no MIN_POSITIVE clamp needed.
    let mask = v.simd_lt(threshold);
    let input = f32x16::blend(mask, f32x16::splat_t(t, 1.0), v);
    let Ok(power) = input.try_powf_const::<{ math::BT709_ENCODE_POWER }>() else {
        return f32x16::from_array_t(t, v.to_array().map(linear_to_bt709));
    };
    let power = one_plus_alpha.mla(power, -alpha);

    f32x16::blend(mask, linear, power)
}
