//! HLG (ARIB STD-B67) transfer function (scalar + generic SIMD).

use super::fast_math;
#[allow(unused_imports)]
use num_traits::Float; // provides sqrt via libm in no_std

// =============================================================================
// Scalar
// =============================================================================

const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 0.28466892; // 1 - 4 * A
const HLG_C: f32 = 0.55991073; // 0.5 - A * ln(4 * A)
const HLG_A_INV_LOG2E: f32 = HLG_A * core::f32::consts::LN_2; // A * ln(2) for log2 conversion
const HLG_INV_A_LOG2E: f32 = core::f32::consts::LOG2_E / HLG_A; // log2(e) / A for exp2 conversion

/// HLG inverse OETF: signal → scene linear. Uses fast_pow2f for exp().
#[inline(always)]
pub fn hlg_to_linear(v: f32) -> f32 {
    if v <= 0.0 {
        0.0
    } else if v <= 0.5 {
        (v * v) / 3.0
    } else {
        (fast_math::fast_pow2f((v - HLG_C) * HLG_INV_A_LOG2E) + HLG_B) / 12.0
    }
}

/// HLG OETF: scene linear → signal. Uses fast_log2f for ln().
#[inline(always)]
pub fn linear_to_hlg(v: f32) -> f32 {
    if v <= 0.0 {
        0.0
    } else if v <= 1.0 / 12.0 {
        (3.0 * v).sqrt()
    } else {
        HLG_A_INV_LOG2E * fast_math::fast_log2f(12.0 * v - HLG_B) + HLG_C
    }
}

// =============================================================================
// Generic SIMD — x4
// =============================================================================

use magetypes::simd::backends::F32x4Convert;
use magetypes::simd::generic::f32x4;

#[allow(dead_code)]
#[inline(always)]
pub fn hlg_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let zero = f32x4::zero(t);
    let half = f32x4::splat(t, 0.5);
    let third = f32x4::splat(t, 1.0 / 3.0);
    let inv_12 = f32x4::splat(t, 1.0 / 12.0);
    let hlg_b = f32x4::splat(t, HLG_B);
    let hlg_c = f32x4::splat(t, HLG_C);
    let hlg_inv_a_log2e = f32x4::splat(t, HLG_INV_A_LOG2E);

    // v <= 0 is absorbed by a = max(v, 0) = 0 ⇒ low = 0; the a<=0.5 mask then
    // already selects `low` for those lanes, so no separate positive mask is
    // needed (saves a cmp + mask materialize on AVX-512).
    let a = v.max(zero);
    let low = (a * a) * third;

    let exp_arg = (a - hlg_c) * hlg_inv_a_log2e;
    let exp_val = fast_math::fast_pow2f_x4(t, exp_arg);
    let high = (exp_val + hlg_b) * inv_12;

    let mask = a.simd_le(half);
    f32x4::blend(mask, low, high)
}

#[allow(dead_code)]
#[inline(always)]
pub fn linear_to_hlg_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let zero = f32x4::zero(t);
    let threshold = f32x4::splat(t, 1.0 / 12.0);
    let three = f32x4::splat(t, 3.0);
    let twelve = f32x4::splat(t, 12.0);
    let hlg_a_ln2 = f32x4::splat(t, HLG_A_INV_LOG2E);
    let hlg_b = f32x4::splat(t, HLG_B);
    let hlg_c = f32x4::splat(t, HLG_C);

    // v <= 0 is absorbed by a = max(v, 0) = 0 ⇒ low = sqrt(0) = 0; the
    // a<=1/12 mask then selects `low` (=0) for those lanes. The `high` path
    // is always discarded when a<=1/12, so no log-argument clamp is needed
    // (garbage from fast_log2f on a small or non-positive arg is masked out).
    let a = v.max(zero);
    let low = (three * a).sqrt();

    let arg = twelve * a - hlg_b;
    let log2_val = fast_math::fast_log2f_x4(t, arg);
    let high = hlg_a_ln2.mul_add(log2_val, hlg_c);

    let mask = a.simd_le(threshold);
    f32x4::blend(mask, low, high)
}

// =============================================================================
// Generic SIMD — x8
// =============================================================================

use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

#[inline(always)]
pub fn hlg_to_linear_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let zero = f32x8::zero(t);
    let half = f32x8::splat(t, 0.5);
    let third = f32x8::splat(t, 1.0 / 3.0);
    let inv_12 = f32x8::splat(t, 1.0 / 12.0);
    let hlg_b = f32x8::splat(t, HLG_B);
    let hlg_c = f32x8::splat(t, HLG_C);
    let hlg_inv_a_log2e = f32x8::splat(t, HLG_INV_A_LOG2E);

    // See comments on hlg_to_linear_x4.
    let a = v.max(zero);
    let low = (a * a) * third;

    let exp_arg = (a - hlg_c) * hlg_inv_a_log2e;
    let exp_val = fast_math::fast_pow2f_x8(t, exp_arg);
    let high = (exp_val + hlg_b) * inv_12;

    let mask = a.simd_le(half);
    f32x8::blend(mask, low, high)
}

#[inline(always)]
pub fn linear_to_hlg_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let zero = f32x8::zero(t);
    let threshold = f32x8::splat(t, 1.0 / 12.0);
    let three = f32x8::splat(t, 3.0);
    let twelve = f32x8::splat(t, 12.0);
    let hlg_a_ln2 = f32x8::splat(t, HLG_A_INV_LOG2E);
    let hlg_b = f32x8::splat(t, HLG_B);
    let hlg_c = f32x8::splat(t, HLG_C);

    // See comments on linear_to_hlg_x4.
    let a = v.max(zero);
    let low = (three * a).sqrt();

    let arg = twelve * a - hlg_b;
    let log2_val = fast_math::fast_log2f_x8(t, arg);
    let high = hlg_a_ln2.mul_add(log2_val, hlg_c);

    let mask = a.simd_le(threshold);
    f32x8::blend(mask, low, high)
}

// =============================================================================
// Generic SIMD — x16
// =============================================================================

use magetypes::simd::backends::F32x16Convert;
use magetypes::simd::generic::f32x16;

#[inline(always)]
pub fn hlg_to_linear_x16<T: F32x16Convert>(t: T, v: f32x16<T>) -> f32x16<T> {
    let zero = f32x16::zero(t);
    let half = f32x16::splat(t, 0.5);
    let third = f32x16::splat(t, 1.0 / 3.0);
    let inv_12 = f32x16::splat(t, 1.0 / 12.0);
    let hlg_b = f32x16::splat(t, HLG_B);
    let hlg_c = f32x16::splat(t, HLG_C);
    let hlg_inv_a_log2e = f32x16::splat(t, HLG_INV_A_LOG2E);

    // See comments on hlg_to_linear_x4.
    let a = v.max(zero);
    let low = (a * a) * third;

    let exp_arg = (a - hlg_c) * hlg_inv_a_log2e;
    let exp_val = fast_math::fast_pow2f_x16(t, exp_arg);
    let high = (exp_val + hlg_b) * inv_12;

    let mask = a.simd_le(half);
    f32x16::blend(mask, low, high)
}

#[inline(always)]
pub fn linear_to_hlg_x16<T: F32x16Convert>(t: T, v: f32x16<T>) -> f32x16<T> {
    let zero = f32x16::zero(t);
    let threshold = f32x16::splat(t, 1.0 / 12.0);
    let three = f32x16::splat(t, 3.0);
    let twelve = f32x16::splat(t, 12.0);
    let hlg_a_ln2 = f32x16::splat(t, HLG_A_INV_LOG2E);
    let hlg_b = f32x16::splat(t, HLG_B);
    let hlg_c = f32x16::splat(t, HLG_C);
    // See comments on linear_to_hlg_x4.
    let a = v.max(zero);
    let low = (three * a).sqrt();

    let arg = twelve * a - hlg_b;
    let log2_val = fast_math::fast_log2f_x16(t, arg);
    let high = hlg_a_ln2.mul_add(log2_val, hlg_c);

    let mask = a.simd_le(threshold);
    f32x16::blend(mask, low, high)
}
