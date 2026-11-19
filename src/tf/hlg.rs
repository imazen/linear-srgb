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
        magetypes::scalar::sqrt_f32(3.0 * v)
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
// Generic SIMD — x4
// =============================================================================

use magetypes::simd::backends::F32x4Convert;
use magetypes::simd::generic::f32x4;

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn hlg_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let zero = f32x4::zero_t(t);
    let half = f32x4::splat_t(t, 0.5);
    let third = f32x4::splat_t(t, 1.0 / 3.0);
    let inv_12 = f32x4::splat_t(t, 1.0 / 12.0);
    let hlg_b = f32x4::splat_t(t, HLG_B);
    let hlg_c = f32x4::splat_t(t, HLG_C);
    let hlg_inv_a_log2e = f32x4::splat_t(t, HLG_INV_A_LOG2E);

    // nonpositive and unordered inputs become +0 ⇒ low = 0; the a<=0.5 mask then
    // already selects `low` for those lanes, so no separate positive mask is
    // needed (saves a cmp + mask materialize on AVX-512).
    let a = floor_zero!(v, zero);
    let low = (a * a) * third;

    let exp_arg = (a - hlg_c) * hlg_inv_a_log2e;
    let exp_val = exp_arg.exp2::<{ math::EXP }>();
    let high = (exp_val + hlg_b) * inv_12;

    let mask = a.simd_le(half);
    f32x4::blend(mask, low, high)
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn linear_to_hlg_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let zero = f32x4::zero_t(t);
    let threshold = f32x4::splat_t(t, 1.0 / 12.0);
    let three = f32x4::splat_t(t, 3.0);
    let twelve = f32x4::splat_t(t, 12.0);
    let hlg_a_ln2 = f32x4::splat_t(t, HLG_A_INV_LOG2E);
    let hlg_b = f32x4::splat_t(t, HLG_B);
    let hlg_c = f32x4::splat_t(t, HLG_C);

    // nonpositive and unordered inputs become +0 ⇒ low = sqrt(0) = 0; the
    // a<=1/12 mask then selects `low` (=0) for those lanes. The `high` path
    // is always discarded when a<=1/12, so no log-argument clamp is needed
    // (garbage from fast_log2f on a small or non-positive arg is masked out).
    let a = floor_zero!(v, zero);
    let low = (three * a).sqrt();

    let arg = twelve * a - hlg_b;
    let log2_val = arg.log2::<{ math::LOG }>();
    let high = hlg_a_ln2.mla(log2_val, hlg_c).max(f32x4::splat_t(t, 0.5));

    let mask = a.simd_le(threshold);
    f32x4::blend(mask, low, high)
}

// =============================================================================
// Generic SIMD — x8
// =============================================================================

use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

#[inline(always)]
pub(crate) fn hlg_to_linear_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let zero = f32x8::zero_t(t);
    let half = f32x8::splat_t(t, 0.5);
    let third = f32x8::splat_t(t, 1.0 / 3.0);
    let inv_12 = f32x8::splat_t(t, 1.0 / 12.0);
    let hlg_b = f32x8::splat_t(t, HLG_B);
    let hlg_c = f32x8::splat_t(t, HLG_C);
    let hlg_inv_a_log2e = f32x8::splat_t(t, HLG_INV_A_LOG2E);

    // See comments on hlg_to_linear_x4.
    let a = floor_zero!(v, zero);
    let low = (a * a) * third;

    let exp_arg = (a - hlg_c) * hlg_inv_a_log2e;
    let exp_val = exp_arg.exp2::<{ math::EXP }>();
    let high = (exp_val + hlg_b) * inv_12;

    let mask = a.simd_le(half);
    f32x8::blend(mask, low, high)
}

#[inline(always)]
pub(crate) fn linear_to_hlg_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let zero = f32x8::zero_t(t);
    let threshold = f32x8::splat_t(t, 1.0 / 12.0);
    let three = f32x8::splat_t(t, 3.0);
    let twelve = f32x8::splat_t(t, 12.0);
    let hlg_a_ln2 = f32x8::splat_t(t, HLG_A_INV_LOG2E);
    let hlg_b = f32x8::splat_t(t, HLG_B);
    let hlg_c = f32x8::splat_t(t, HLG_C);

    // See comments on linear_to_hlg_x4.
    let a = floor_zero!(v, zero);
    let low = (three * a).sqrt();

    let arg = twelve * a - hlg_b;
    let log2_val = arg.log2::<{ math::LOG }>();
    let high = hlg_a_ln2.mla(log2_val, hlg_c).max(f32x8::splat_t(t, 0.5));

    let mask = a.simd_le(threshold);
    f32x8::blend(mask, low, high)
}

// =============================================================================
// Generic SIMD — x16
// =============================================================================

use magetypes::simd::backends::F32x16Convert;
use magetypes::simd::generic::f32x16;

#[inline(always)]
pub(crate) fn hlg_to_linear_x16<T: F32x16Convert>(t: T, v: f32x16<T>) -> f32x16<T> {
    let zero = f32x16::zero_t(t);
    let half = f32x16::splat_t(t, 0.5);
    let third = f32x16::splat_t(t, 1.0 / 3.0);
    let inv_12 = f32x16::splat_t(t, 1.0 / 12.0);
    let hlg_b = f32x16::splat_t(t, HLG_B);
    let hlg_c = f32x16::splat_t(t, HLG_C);
    let hlg_inv_a_log2e = f32x16::splat_t(t, HLG_INV_A_LOG2E);

    // See comments on hlg_to_linear_x4.
    let a = floor_zero!(v, zero);
    let low = (a * a) * third;

    let exp_arg = (a - hlg_c) * hlg_inv_a_log2e;
    let exp_val = exp_arg.exp2::<{ math::EXP }>();
    let high = (exp_val + hlg_b) * inv_12;

    let mask = a.simd_le(half);
    f32x16::blend(mask, low, high)
}

#[inline(always)]
pub(crate) fn linear_to_hlg_x16<T: F32x16Convert>(t: T, v: f32x16<T>) -> f32x16<T> {
    let zero = f32x16::zero_t(t);
    let threshold = f32x16::splat_t(t, 1.0 / 12.0);
    let three = f32x16::splat_t(t, 3.0);
    let twelve = f32x16::splat_t(t, 12.0);
    let hlg_a_ln2 = f32x16::splat_t(t, HLG_A_INV_LOG2E);
    let hlg_b = f32x16::splat_t(t, HLG_B);
    let hlg_c = f32x16::splat_t(t, HLG_C);
    // See comments on linear_to_hlg_x4.
    let a = floor_zero!(v, zero);
    let low = (three * a).sqrt();

    let arg = twelve * a - hlg_b;
    let log2_val = arg.log2::<{ math::LOG }>();
    let high = hlg_a_ln2.mla(log2_val, hlg_c).max(f32x16::splat_t(t, 0.5));

    let mask = a.simd_le(threshold);
    f32x16::blend(mask, low, high)
}
