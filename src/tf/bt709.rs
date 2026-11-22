//! BT.709 transfer function (scalar + generic SIMD with const-generic rail ablation).

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
// Generic SIMD — x4 / x8 / x16 (const-generic RAILS)
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
        $decode_cfg:ident,
        $decode:ident,
        $decode_slice_cfg:ident,
        $decode_slice:ident,
        $try_encode:ident,
        $encode_cfg:ident,
        $encode:ident,
        $encode_slice_cfg:ident,
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
            // Lanes < threshold (including <= 0, subnormals, and -inf) are blended to 1.0;
            // remaining finite non-NaN lanes have `input >= 0.164232 > 2^-16`. Thus a single
            // ordered `<= 65_536.0` check suffices to reject both `> 65_536.0` and `NaN`
            // while eliding subnormal flushing and lower-bound checks.
            let input = $float::blend(mask, $float::splat_t(t, 1.0), normalized);
            if !input
                .simd_le($float::splat_t(t, 65_536.0))
                .bitcast_to_i32()
                .all_true()
            {
                return None;
            }
            let power = input
                .powf_const::<{ math::BT709_DECODE_FAST }>()
                .max_fast($float::splat_t(t, BT709_DECODE_MIN));
            Some($float::blend(mask, linear, power))
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $decode_cfg<T: $($bound)+, const RAILS: bool>(
            t: T,
            v: $float<T>,
        ) -> $float<T> {
            if !RAILS {
                let threshold = $float::splat_t(t, 4.5 * BT709_BETA);
                let inv_4_5 = $float::splat_t(t, 1.0 / 4.5);
                let alpha = $float::splat_t(t, BT709_ALPHA);
                let one_plus_alpha = $float::splat_t(t, 1.0 + BT709_ALPHA);
                let linear = v * inv_4_5;
                let normalized = (v + alpha) / one_plus_alpha;
                let mask = v.simd_lt(threshold);
                let input = $float::blend(mask, $float::splat_t(t, 1.0), normalized);
                let power = input
                    .powf_const::<{ math::BT709_DECODE_FAST }>()
                    .max_fast($float::splat_t(t, BT709_DECODE_MIN));
                return $float::blend(mask, linear, power);
            }
            match $try_decode(t, v) {
                Some(out) => out,
                None => $float::from_array_t(t, bt709_to_linear_fallback(t, v.to_array())),
            }
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $decode<T: $($bound)+>(t: T, v: $float<T>) -> $float<T> {
            $decode_cfg::<T, true>(t, v)
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $decode_slice_cfg<T: $($bound)+, const RAILS: bool>(
            t: T,
            values: &mut [f32],
        ) {
            if !RAILS {
                fn tail<T: $($bound)+>(t: T, chunk: [f32; $width]) -> [f32; $width] {
                    $decode_cfg::<T, false>(t, $float::from_array_t(t, chunk)).to_array()
                }
                crate::kernels::map_slice_inline!(
                    values,
                    $width,
                    t,
                    |chunk| $decode_cfg::<T, false>(t, $float::from_array_t(t, chunk)).to_array(),
                    tail::<T>
                );
                return;
            }
            fn slow<T: $($bound)+>(t: T, chunk: [f32; $width]) -> [f32; $width] {
                $decode(t, $float::from_array_t(t, chunk)).to_array()
            }
            crate::kernels::map_slice_try_inline!(
                values,
                $width,
                t,
                |chunk| $try_decode(t, $float::from_array_t(t, chunk)).map(|r| r.to_array()),
                slow::<T>
            );
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $decode_slice<T: $($bound)+>(t: T, values: &mut [f32]) {
            $decode_slice_cfg::<T, true>(t, values);
        }

        #[inline(always)]
        pub(crate) fn $try_encode<T: $($bound)+>(t: T, v: $float<T>) -> Option<$float<T>> {
            let threshold = $float::splat_t(t, BT709_BETA);
            let scale_4_5 = $float::splat_t(t, 4.5);
            let one_plus_alpha = $float::splat_t(t, 1.0 + BT709_ALPHA);
            let alpha = $float::splat_t(t, BT709_ALPHA);

            let linear = v * scale_4_5;
            let mask = v.simd_lt(threshold);
            // Lanes < BT709_BETA (0.018054) are blended to 1.0; remaining finite non-NaN
            // lanes are >= 0.018054 > 2^-16. Ordered `<= 65_536.0` rejects both `> 65_536.0`
            // and `NaN` in one check.
            let input = $float::blend(mask, $float::splat_t(t, 1.0), v);
            if !input
                .simd_le($float::splat_t(t, 65_536.0))
                .bitcast_to_i32()
                .all_true()
            {
                return None;
            }
            let power = input.powf_const::<{ math::BT709_ENCODE_FAST }>();
            let power = one_plus_alpha.mla(power, -alpha);
            Some($float::blend(mask, linear, power))
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $encode_cfg<T: $($bound)+, const RAILS: bool>(
            t: T,
            v: $float<T>,
        ) -> $float<T> {
            if !RAILS {
                let threshold = $float::splat_t(t, BT709_BETA);
                let scale_4_5 = $float::splat_t(t, 4.5);
                let one_plus_alpha = $float::splat_t(t, 1.0 + BT709_ALPHA);
                let alpha = $float::splat_t(t, BT709_ALPHA);
                let linear = v * scale_4_5;
                let mask = v.simd_lt(threshold);
                let input = $float::blend(mask, $float::splat_t(t, 1.0), v);
                let power = input.powf_const::<{ math::BT709_ENCODE_FAST }>();
                let power = one_plus_alpha.mla(power, -alpha);
                return $float::blend(mask, linear, power);
            }
            match $try_encode(t, v) {
                Some(out) => out,
                None => $float::from_array_t(t, linear_to_bt709_fallback(t, v.to_array())),
            }
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $encode<T: $($bound)+>(t: T, v: $float<T>) -> $float<T> {
            $encode_cfg::<T, true>(t, v)
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $encode_slice_cfg<T: $($bound)+, const RAILS: bool>(
            t: T,
            values: &mut [f32],
        ) {
            if !RAILS {
                fn tail<T: $($bound)+>(t: T, chunk: [f32; $width]) -> [f32; $width] {
                    $encode_cfg::<T, false>(t, $float::from_array_t(t, chunk)).to_array()
                }
                crate::kernels::map_slice_inline!(
                    values,
                    $width,
                    t,
                    |chunk| $encode_cfg::<T, false>(t, $float::from_array_t(t, chunk)).to_array(),
                    tail::<T>
                );
                return;
            }
            fn slow<T: $($bound)+>(t: T, chunk: [f32; $width]) -> [f32; $width] {
                $encode(t, $float::from_array_t(t, chunk)).to_array()
            }
            crate::kernels::map_slice_try_inline!(
                values,
                $width,
                t,
                |chunk| $try_encode(t, $float::from_array_t(t, chunk)).map(|r| r.to_array()),
                slow::<T>
            );
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $encode_slice<T: $($bound)+>(t: T, values: &mut [f32]) {
            $encode_slice_cfg::<T, true>(t, values);
        }
    };
}

bt709_for!(
    try_bt709_to_linear_x4,
    bt709_to_linear_x4_cfg,
    bt709_to_linear_x4,
    bt709_to_linear_slice_x4_cfg,
    bt709_to_linear_slice_x4,
    try_linear_to_bt709_x4,
    linear_to_bt709_x4_cfg,
    linear_to_bt709_x4,
    linear_to_bt709_slice_x4_cfg,
    linear_to_bt709_slice_x4,
    f32x4,
    4,
    F32x4Convert
);
bt709_for!(
    try_bt709_to_linear_x8,
    bt709_to_linear_x8_cfg,
    bt709_to_linear_x8,
    bt709_to_linear_slice_x8_cfg,
    bt709_to_linear_slice_x8,
    try_linear_to_bt709_x8,
    linear_to_bt709_x8_cfg,
    linear_to_bt709_x8,
    linear_to_bt709_slice_x8_cfg,
    linear_to_bt709_slice_x8,
    f32x8,
    8,
    F32x8Convert + F32x4Convert
);
bt709_for!(
    try_bt709_to_linear_x16,
    bt709_to_linear_x16_cfg,
    bt709_to_linear_x16,
    bt709_to_linear_slice_x16_cfg,
    bt709_to_linear_slice_x16,
    try_linear_to_bt709_x16,
    linear_to_bt709_x16_cfg,
    linear_to_bt709_x16,
    linear_to_bt709_slice_x16_cfg,
    linear_to_bt709_slice_x16,
    f32x16,
    16,
    F32x16Convert + F32x4Convert
);
