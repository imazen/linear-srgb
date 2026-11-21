//! BT.709 transfer function (scalar + generic SIMD).

use super::math;
use super::polynomial::{Mla, mla_scalar};
use magemath::SimdMath;

pub(crate) const BT709_ALPHA: f32 = 0.09929682680944;
pub(crate) const BT709_BETA: f32 = 0.018053968510807;
const BT709_DECODE_MIN: f32 = (4.5 * BT709_BETA) * (1.0 / 4.5);

/// BT.709 EOTF: encoded → linear. Uses the configured Magemath power profile.
#[inline]
pub fn bt709_to_linear(v: f32) -> f32 {
    archmage::attuned!(bt709_to_linear_scalar(v))
}

#[archmage::attune(_*_t, define(f32x4))]
fn bt709_to_linear_scalar(token: Token, v: f32) -> f32 {
    bt709_to_linear_with_token(token, v)
}

#[inline(always)]
pub(crate) fn bt709_to_linear_with_token<T: F32x4Convert>(token: T, v: f32) -> f32 {
    if v.is_nan() {
        v
    } else if v < 4.5 * BT709_BETA {
        v * (1.0 / 4.5)
    } else {
        math::fixed_pow::<{ math::BT709_DECODE_POWER }, _>(
            token,
            (v + BT709_ALPHA) / (1.0 + BT709_ALPHA),
            1.0 / 0.45,
        )
        .max(BT709_DECODE_MIN)
    }
}

/// BT.709 inverse EOTF: linear → encoded. Uses the configured Magemath power profile.
#[inline]
pub fn linear_to_bt709(v: f32) -> f32 {
    archmage::attuned!(linear_to_bt709_scalar(v))
}

#[archmage::attune(_*_t, define(f32x4))]
fn linear_to_bt709_scalar(token: Token, v: f32) -> f32 {
    linear_to_bt709_with_token(token, v)
}

#[inline(always)]
pub(crate) fn linear_to_bt709_with_token<T: F32x4Convert>(token: T, v: f32) -> f32 {
    if v.is_nan() {
        v
    } else if v < BT709_BETA {
        4.5 * v
    } else {
        mla_scalar(
            token,
            1.0 + BT709_ALPHA,
            math::fixed_pow::<{ math::BT709_ENCODE_POWER }, _>(token, v, 0.45),
            -BT709_ALPHA,
        )
    }
}

// =============================================================================
// Generic SIMD — x4 / x8 / x16
// =============================================================================

use magetypes::simd::{backends::*, generic::*};

#[cold]
#[inline(never)]
fn bt709_to_linear_fallback<T: F32x4Convert, const N: usize>(t: T, v: [f32; N]) -> [f32; N] {
    v.map(|x| bt709_to_linear_with_token(t, x))
}

#[cold]
#[inline(never)]
fn linear_to_bt709_fallback<T: F32x4Convert, const N: usize>(t: T, v: [f32; N]) -> [f32; N] {
    v.map(|x| linear_to_bt709_with_token(t, x))
}

macro_rules! bt709_for {
    (
        $try_decode:ident,
        $decode:ident,
        $decode_slice:ident,
        $try_encode:ident,
        $encode:ident,
        $encode_slice:ident,
        $float:ident,
        $width:literal,
        $($bound:tt)+
    ) => {
        #[inline(always)]
        pub(crate) fn $try_decode<T: $($bound)+>(t: T, v: $float<T>) -> Option<$float<T>> {
            let threshold = $float::splat_t(t, 4.5 * BT709_BETA);
            let inv_4_5 = $float::splat_t(t, 1.0 / 4.5);
            let alpha = $float::splat_t(t, BT709_ALPHA);
            let one_plus_alpha = $float::splat_t(t, 1.0 + BT709_ALPHA);

            let linear = v * inv_4_5;
            let normalized = (v + alpha) / one_plus_alpha;
            let mask = v.simd_lt(threshold);
            let input = $float::blend(mask, $float::splat_t(t, 1.0), normalized);
            let Ok(power) = input.try_powf_const::<{ math::BT709_DECODE_POWER }>() else {
                return None;
            };

            // The curves meet only to within rounding: keep the power branch above the linear one.
            // Neither `power` nor `BT709_DECODE_MIN` can be NaN here (`try_powf_const` rejected NaN).
            let power = power.max_fast($float::splat_t(t, BT709_DECODE_MIN));
            Some($float::blend(mask, linear, power))
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $decode<T: $($bound)+>(t: T, v: $float<T>) -> $float<T> {
            match $try_decode(t, v) {
                Some(out) => out,
                None => $float::from_array_t(t, bt709_to_linear_fallback(t, v.to_array())),
            }
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $decode_slice<T: $($bound)+>(t: T, values: &mut [f32]) {
            crate::kernels::map_slice_try_inline!(
                values,
                $width,
                t,
                |chunk| $try_decode(t, $float::from_array_t(t, chunk)).map(|r| r.to_array()),
                bt709_to_linear_fallback::<T, $width>
            );
        }

        #[inline(always)]
        pub(crate) fn $try_encode<T: $($bound)+>(t: T, v: $float<T>) -> Option<$float<T>> {
            let threshold = $float::splat_t(t, BT709_BETA);
            let scale_4_5 = $float::splat_t(t, 4.5);
            let one_plus_alpha = $float::splat_t(t, 1.0 + BT709_ALPHA);
            let alpha = $float::splat_t(t, BT709_ALPHA);

            let linear = v * scale_4_5;
            let mask = v.simd_lt(threshold);
            let input = $float::blend(mask, $float::splat_t(t, 1.0), v);
            let Ok(power) = input.try_powf_const::<{ math::BT709_ENCODE_POWER }>() else {
                return None;
            };
            let power = one_plus_alpha.mla(power, -alpha);
            Some($float::blend(mask, linear, power))
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $encode<T: $($bound)+>(t: T, v: $float<T>) -> $float<T> {
            match $try_encode(t, v) {
                Some(out) => out,
                None => $float::from_array_t(t, linear_to_bt709_fallback(t, v.to_array())),
            }
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $encode_slice<T: $($bound)+>(t: T, values: &mut [f32]) {
            crate::kernels::map_slice_try_inline!(
                values,
                $width,
                t,
                |chunk| $try_encode(t, $float::from_array_t(t, chunk)).map(|r| r.to_array()),
                linear_to_bt709_fallback::<T, $width>
            );
        }
    };
}

bt709_for!(
    try_bt709_to_linear_x4,
    bt709_to_linear_x4,
    bt709_to_linear_slice_x4,
    try_linear_to_bt709_x4,
    linear_to_bt709_x4,
    linear_to_bt709_slice_x4,
    f32x4,
    4,
    F32x4Convert
);
bt709_for!(
    try_bt709_to_linear_x8,
    bt709_to_linear_x8,
    bt709_to_linear_slice_x8,
    try_linear_to_bt709_x8,
    linear_to_bt709_x8,
    linear_to_bt709_slice_x8,
    f32x8,
    8,
    F32x8Convert + F32x4Convert
);
bt709_for!(
    try_bt709_to_linear_x16,
    bt709_to_linear_x16,
    bt709_to_linear_slice_x16,
    try_linear_to_bt709_x16,
    linear_to_bt709_x16,
    linear_to_bt709_slice_x16,
    f32x16,
    16,
    F32x16Convert + F32x4Convert
);
