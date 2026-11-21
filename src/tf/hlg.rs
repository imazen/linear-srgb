//! HLG (ARIB STD-B67) transfer function (scalar + generic SIMD).

use super::math;
use super::polynomial::{Mla, floor_zero, mla_scalar};
use magemath::SimdMath;

// =============================================================================
// Scalar
// =============================================================================

const HLG_A: f32 = 0.17883277;
pub(crate) const HLG_B: f32 = 0.28466892; // 1 - 4 * A
pub(crate) const HLG_C: f32 = 0.55991073; // 0.5 - A * ln(4 * A)
pub(crate) const HLG_A_INV_LOG2E: f32 = HLG_A * core::f32::consts::LN_2; // A * ln(2) for log2 conversion
pub(crate) const HLG_INV_A_LOG2E: f32 = core::f32::consts::LOG2_E / HLG_A; // log2(e) / A for exp2 conversion

/// HLG inverse OETF: signal → scene linear. Uses Magemath for the exponential segment.
#[inline]
pub fn hlg_to_linear(v: f32) -> f32 {
    archmage::attuned!(hlg_to_linear_scalar(v))
}

#[archmage::attune(_*_t, define(f32x4))]
fn hlg_to_linear_scalar(token: Token, v: f32) -> f32 {
    hlg_to_linear_with_token(token, v)
}

#[inline(always)]
pub(crate) fn hlg_to_linear_with_token<T: F32x4Convert>(token: T, v: f32) -> f32 {
    if v <= 0.0 || v.is_nan() {
        0.0
    } else if v <= 0.5 {
        (v * v) * (1.0 / 3.0)
    } else {
        (math::exp2_with_token(token, (v - HLG_C) * HLG_INV_A_LOG2E) + HLG_B) * (1.0 / 12.0)
    }
}

/// HLG OETF: scene linear → signal. Uses Magemath for the logarithmic segment.
#[inline]
pub fn linear_to_hlg(v: f32) -> f32 {
    archmage::attuned!(linear_to_hlg_scalar(v))
}

#[archmage::attune(_*_t, define(f32x4))]
fn linear_to_hlg_scalar(token: Token, v: f32) -> f32 {
    linear_to_hlg_with_token(token, v)
}

#[inline(always)]
pub(crate) fn linear_to_hlg_with_token<T: F32x4Convert>(token: T, v: f32) -> f32 {
    if v <= 0.0 || v.is_nan() {
        0.0
    } else if v <= 1.0 / 12.0 {
        super::polynomial::fast_sqrt_f32(3.0 * v)
    } else {
        mla_scalar(
            token,
            HLG_A_INV_LOG2E,
            math::log2_with_token(token, 12.0 * v - HLG_B),
            HLG_C,
        )
        .max(0.5)
    }
}

// =============================================================================
// Generic SIMD — x4 / x8 / x16
// =============================================================================

use magetypes::simd::{backends::*, generic::*};

#[cold]
#[inline(never)]
fn hlg_to_linear_fallback<T: F32x4Convert, const N: usize>(t: T, v: [f32; N]) -> [f32; N] {
    v.map(|x| hlg_to_linear_with_token(t, x))
}

#[cold]
#[inline(never)]
fn linear_to_hlg_fallback<T: F32x4Convert, const N: usize>(t: T, v: [f32; N]) -> [f32; N] {
    v.map(|x| linear_to_hlg_with_token(t, x))
}

macro_rules! hlg_for {
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
            let zero = $float::zero_t(t);
            let half = $float::splat_t(t, 0.5);
            let third = $float::splat_t(t, 1.0 / 3.0);
            let inv_12 = $float::splat_t(t, 1.0 / 12.0);
            let hlg_b = $float::splat_t(t, HLG_B);
            let hlg_c = $float::splat_t(t, HLG_C);
            let hlg_inv_a_log2e = $float::splat_t(t, HLG_INV_A_LOG2E);

            // nonpositive and unordered inputs become +0 ⇒ low = 0; the a<=0.5 mask then
            // already selects `low` for those lanes, so no separate positive mask is
            // needed (saves a cmp + mask materialize on AVX-512).
            let a = floor_zero!(v, zero);
            let low = (a * a) * third;

            let mask = a.simd_le(half);
            let exp_arg = $float::blend(mask, zero, (a - hlg_c) * hlg_inv_a_log2e);
            let Ok(exp_val) = exp_arg.try_exp2_with::<{ math::HLG_EXP }>() else {
                return None;
            };
            let high = (exp_val + hlg_b) * inv_12;

            Some($float::blend(mask, low, high))
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $decode<T: $($bound)+>(t: T, v: $float<T>) -> $float<T> {
            match $try_decode(t, v) {
                Some(out) => out,
                None => $float::from_array_t(t, hlg_to_linear_fallback(t, v.to_array())),
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
                hlg_to_linear_fallback::<T, $width>
            );
        }

        #[inline(always)]
        pub(crate) fn $try_encode<T: $($bound)+>(t: T, v: $float<T>) -> Option<$float<T>> {
            let zero = $float::zero_t(t);
            let threshold = $float::splat_t(t, 1.0 / 12.0);
            let three = $float::splat_t(t, 3.0);
            let twelve = $float::splat_t(t, 12.0);
            let hlg_a_ln2 = $float::splat_t(t, HLG_A_INV_LOG2E);
            let hlg_b = $float::splat_t(t, HLG_B);
            let hlg_c = $float::splat_t(t, HLG_C);

            // nonpositive and unordered inputs become +0 ⇒ low = sqrt(0) = 0; the
            // a<=1/12 mask then selects `low` (=0) for those lanes. Blending masked
            // lanes to 1.0 keeps the log2 input strictly positive normal (>= 0.715331)
            // for all finite non-overflowing lanes, eliding subnormal normalization.
            let a = floor_zero!(v, zero);
            let low = (three * a).sqrt();

            let mask = a.simd_le(threshold);
            let arg = $float::blend(mask, $float::splat_t(t, 1.0), twelve * a - hlg_b);
            let Ok(log2_val) = arg.try_log2_with::<{ math::HLG_LOG }>() else {
                return None;
            };
            let high = hlg_a_ln2.mla(log2_val, hlg_c).max_fast($float::splat_t(t, 0.5));

            Some($float::blend(mask, low, high))
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $encode<T: $($bound)+>(t: T, v: $float<T>) -> $float<T> {
            match $try_encode(t, v) {
                Some(out) => out,
                None => $float::from_array_t(t, linear_to_hlg_fallback(t, v.to_array())),
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
                linear_to_hlg_fallback::<T, $width>
            );
        }
    };
}

hlg_for!(
    try_hlg_to_linear_x4,
    hlg_to_linear_x4,
    hlg_to_linear_slice_x4,
    try_linear_to_hlg_x4,
    linear_to_hlg_x4,
    linear_to_hlg_slice_x4,
    f32x4,
    4,
    F32x4Convert
);
hlg_for!(
    try_hlg_to_linear_x8,
    hlg_to_linear_x8,
    hlg_to_linear_slice_x8,
    try_linear_to_hlg_x8,
    linear_to_hlg_x8,
    linear_to_hlg_slice_x8,
    f32x8,
    8,
    F32x8Convert + F32x4Convert
);
hlg_for!(
    try_hlg_to_linear_x16,
    hlg_to_linear_x16,
    hlg_to_linear_slice_x16,
    try_linear_to_hlg_x16,
    linear_to_hlg_x16,
    linear_to_hlg_slice_x16,
    f32x16,
    16,
    F32x16Convert + F32x4Convert
);
