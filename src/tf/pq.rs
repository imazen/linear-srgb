//! PQ (SMPTE ST 2084) transfer function (scalar + generic SIMD).

use super::math;
use super::polynomial;
use super::polynomial::{Mla, fast_sqrt_f32, mla_scalar};
use magemath::SimdMath;
use magemath::log1pf::Log1pf;
use magemath::powf::Powf;
#[allow(unused_imports)]
use num_traits::Float; // provides f64::powf via libm in no_std for precise reference slices

// =============================================================================
// Scalar
// =============================================================================

/// PQ EOTF for normalized signal values. Uses a cancellation-free `f32`
/// `magemath` formula in the dark toe (`v < 0.02`) and a two-segment rational
/// approximation over the remainder of the range.
#[inline]
pub fn pq_to_linear(v: f32) -> f32 {
    archmage::attuned!(pq_to_linear_scalar(v))
}

#[archmage::attune(_*_t, define(f32x4))]
fn pq_to_linear_scalar(token: Token, v: f32) -> f32 {
    pq_to_linear_with_token(token, v)
}

#[inline(always)]
pub(crate) fn pq_to_linear_with_token<T: F32x4Convert>(token: T, v: f32) -> f32 {
    if v.is_nan() {
        return v;
    }
    if v <= EOTF_DARK_FLOOR {
        return 0.0;
    }
    if v >= 1.0 {
        return 1.0;
    }
    if v < EOTF_POLY_FLOOR {
        return pq_to_linear_dark_toe_scalar(token, v);
    }
    let x = mla_scalar(token, v, v, v);
    if v < 0.25 {
        polynomial::eval_rational_poly(token, x, PQ_EOTF_P_SMALL, PQ_EOTF_Q_SMALL)
    } else {
        polynomial::eval_rational_poly(token, x, PQ_EOTF_P_LARGE, PQ_EOTF_Q_LARGE)
    }
}

/// PQ inverse EOTF for normalized linear light. Uses `magemath` `f32` powers
/// below `1e-7` and a two-segment rational approximation above that threshold.
#[inline]
pub fn linear_to_pq(v: f32) -> f32 {
    archmage::attuned!(linear_to_pq_scalar(v))
}

#[archmage::attune(_*_t, define(f32x4))]
fn linear_to_pq_scalar(token: Token, v: f32) -> f32 {
    linear_to_pq_with_token(token, v)
}

#[inline(always)]
pub(crate) fn linear_to_pq_with_token<T: F32x4Convert>(token: T, v: f32) -> f32 {
    if v.is_nan() {
        return v;
    }
    if v <= 0.0 {
        return 0.0;
    }
    if v >= 1.0 {
        return 1.0;
    }
    if v < OETF_POLY_FLOOR {
        return linear_to_pq_dark_toe_scalar(token, v);
    }
    let a = fast_sqrt_f32(fast_sqrt_f32(v));
    if a < 0.1 {
        polynomial::eval_rational_poly(token, a, PQ_INV_P_SMALL, PQ_INV_Q_SMALL)
    } else {
        polynomial::eval_rational_poly(token, a, PQ_INV_P_LARGE, PQ_INV_Q_LARGE)
    }
}

const M1: f64 = 2610.0 / 16384.0;
const M2: f64 = 2523.0 / 32.0;
const C1: f64 = 3424.0 / 4096.0;
const C2: f64 = 2413.0 / 128.0;
const C3: f64 = 2392.0 / 128.0;

const INV_M2_F32: f32 = 32.0 / 2523.0;
const C1_F32: f32 = 3424.0 / 4096.0;
const C2_F32: f32 = 2413.0 / 128.0;
const C3_F32: f32 = 2392.0 / 128.0;
/// Double-word `V0 = C1^M2 ≈ 7.309559025812299e-7` for Sterbenz-exact `(v - V0) / V0`:
/// `dm1 = fma(v - V0_HI_F32, INV_V0_F32, DM1_CORR_F32)` preserves full `f32` precision
/// down to `EOTF_DARK_FLOOR`.
const V0_HI_F32: f32 = f32::from_bits(0x3544_36e8);
const INV_V0_F32: f32 = 1_368_071.625;
const DM1_CORR_F32: f32 = f32::from_bits(0xb298_9668);

const EOTF_POLY_FLOOR: f32 = 0.02;
const OETF_POLY_FLOOR: f32 = 1e-7;
/// Signal threshold below which ST 2084 linear light is `< 3.5e-37` (`< 3.5e-33` nits),
/// flushing cleanly to `0.0` so active dark-toe lanes keep `dm1 >= 4.7e-4` and
/// `base >= 1.5e-6` inside `PQ_EOTF_LOG1P` and `PQ_EOTF_POW`.
const EOTF_DARK_FLOOR: f32 = 7.313e-7;

/// Cancellation-free `f32` ST 2084 EOTF dark toe for `v in (EOTF_DARK_FLOOR, 0.02)`:
/// `dm1 = fma(v - V0_HI, 1/V0, -V0_LO/V0)`, `w = log1p(dm1) / M2 in (0, 0.1296)`,
/// `num = C1 * expm1(w)`, `base = num / (C2 - C3 * (C1 + num))`, `base^(1/M1)`.
#[inline(always)]
fn pq_to_linear_dark_toe_scalar<T: F32x4Convert>(token: T, v: f32) -> f32 {
    let dm1 = mla_scalar(token, v - V0_HI_F32, INV_V0_F32, DM1_CORR_F32);
    let w = Log1pf::<{ math::PQ_EOTF_LOG1P }>::scalar_with_token(token, dm1)
        .expect("infallible PQ EOTF log1p")
        * INV_M2_F32;
    let mut p = 1.0_f32 / 720.0;
    p = mla_scalar(token, p, w, 1.0 / 120.0);
    p = mla_scalar(token, p, w, 1.0 / 24.0);
    p = mla_scalar(token, p, w, 1.0 / 6.0);
    p = mla_scalar(token, p, w, 0.5);
    p = mla_scalar(token, p, w, 1.0);
    let em1 = p * w;
    let num = C1_F32 * em1;
    let vp = C1_F32 + num;
    let den = mla_scalar(token, -C3_F32, vp, C2_F32);
    let base = num / den;
    Powf::<{ math::PQ_EOTF_POW }>::scalar_const_with_token(token, base)
        .expect("infallible PQ EOTF pow")
}

/// Pure `f32` ST 2084 OETF dark toe for `v in (0, 1e-7)`.
#[inline(always)]
fn linear_to_pq_dark_toe_scalar<T: F32x4Convert>(token: T, v: f32) -> f32 {
    let vp = Powf::<{ math::PQ_OETF_POW1 }>::scalar_const_with_token(token, v)
        .expect("infallible PQ OETF pow1");
    let num = mla_scalar(token, vp, C2_F32, C1_F32);
    let den = mla_scalar(token, vp, C3_F32, 1.0);
    let ratio = num / den;
    Powf::<{ math::PQ_OETF_POW2 }>::scalar_const_with_token(token, ratio)
        .expect("infallible PQ OETF pow2")
}

#[inline]
fn pq_to_linear_precise(v: f32) -> f32 {
    if v.is_nan() {
        return v;
    }
    if v <= 0.0 {
        return 0.0;
    }
    if v >= 1.0 {
        return 1.0;
    }
    let vp = f64::from(v).powf(1.0 / M2);
    (((vp - C1).max(0.0) / (C2 - C3 * vp)).powf(1.0 / M1)) as f32
}

#[inline]
fn linear_to_pq_precise(v: f32) -> f32 {
    if v.is_nan() {
        return v;
    }
    if v <= 0.0 {
        return 0.0;
    }
    if v >= 1.0 {
        return 1.0;
    }
    let vp = f64::from(v).powf(M1);
    ((C1 + C2 * vp) / (1.0 + C3 * vp)).powf(M2) as f32
}

/// Convert normalized ST 2084 signals to linear light normalized to 10,000 nits.
/// Uses the reference formula with f64 intermediates throughout the range.
/// Inputs clip to [0, 1], NaNs propagate, and black and white are exact.
pub fn pq_to_linear_precise_slice(values: &mut [f32]) {
    for value in values {
        *value = pq_to_linear_precise(*value);
    }
}

/// Convert linear light normalized to 10,000 nits to normalized ST 2084 signals.
/// Uses the reference formula with f64 intermediates throughout the range.
/// Inputs clip to [0, 1], NaNs propagate, and zero encodes to exact zero.
pub fn linear_to_pq_precise_slice(values: &mut [f32]) {
    for value in values {
        *value = linear_to_pq_precise(*value);
    }
}

// =============================================================================
// Coefficients
// =============================================================================

// Two-range EOTF: split at signal = 0.25 for sub-U16-step roundtrip accuracy.
// Fitted with relative-error weighting (minimax in signal-space roundtrip).
pub(crate) const PQ_EOTF_P_SMALL: [f32; 5] = [
    4.36033203e-10,
    -1.53651077e-07,
    2.06864196e-06,
    2.80304711e-03,
    1.47809948e-02,
];
pub(crate) const PQ_EOTF_Q_SMALL: [f32; 5] = [
    -9.89955022e-03,
    1.89911767e+00,
    -1.34828020e+00,
    -7.10547502e-01,
    1.00000000e+00,
];
pub(crate) const PQ_EOTF_P_LARGE: [f32; 5] = [
    5.43046166e-04,
    -6.31017431e-03,
    2.90480734e-01,
    9.80363032e-01,
    2.22819483e-01,
];
pub(crate) const PQ_EOTF_Q_LARGE: [f32; 5] = [
    1.58688561e+02,
    -1.60764060e+02,
    6.51370372e+01,
    -1.26438391e+01,
    1.00000000e+00,
];

pub(crate) const PQ_INV_P_LARGE: [f32; 5] =
    [1.351392e-2, -1.095778, 5.522776e1, 1.492516e2, 4.838434e1];
pub(crate) const PQ_INV_Q_LARGE: [f32; 5] =
    [1.012416, 2.016708e1, 9.26371e1, 1.120607e2, 2.590418e1];

pub(crate) const PQ_INV_P_SMALL: [f32; 5] = [
    9.863406e-6,
    3.881234e-1,
    1.352821e2,
    6.889862e4,
    -2.864824e5,
];
pub(crate) const PQ_INV_Q_SMALL: [f32; 5] =
    [3.371868e1, 1.477719e3, 1.608477e4, -4.389884e4, -2.072546e5];

use magetypes::simd::{backends::*, generic::*};

// =============================================================================
// Cold pure-f32 SIMD dark-toe patch helpers (token-dispatched, zero f64 / libm)
// =============================================================================

macro_rules! pq_patch_body {
    ($decode_body:ident, $encode_body:ident, $float:ident, $width:literal, $($bound:tt)+) => {
        #[allow(dead_code)]
        #[inline(always)]
        fn $decode_body<T: $($bound)+>(t: T, source: &[f32; $width], result: &mut [f32; $width]) {
            let v = $float::load_t(t, source);
            let zero = $float::zero_t(t);
            let dark_floor = $float::splat_t(t, EOTF_DARK_FLOOR);
            let poly_floor = $float::splat_t(t, EOTF_POLY_FLOOR);
            let is_dark = v.simd_gt(dark_floor) & v.simd_lt(poly_floor);
            let v_safe = $float::blend(is_dark, v, $float::splat_t(t, 0.01));

            let dm1 = (v_safe - $float::splat_t(t, V0_HI_F32)).mla(
                $float::splat_t(t, INV_V0_F32),
                $float::splat_t(t, DM1_CORR_F32),
            );
            let w = dm1.log1p::<{ math::PQ_EOTF_LOG1P }>() * $float::splat_t(t, INV_M2_F32);
            let mut p = $float::splat_t(t, 1.0 / 720.0);
            p = p.mla(w, $float::splat_t(t, 1.0 / 120.0));
            p = p.mla(w, $float::splat_t(t, 1.0 / 24.0));
            p = p.mla(w, $float::splat_t(t, 1.0 / 6.0));
            p = p.mla(w, $float::splat_t(t, 0.5));
            p = p.mla(w, $float::splat_t(t, 1.0));
            let em1 = p * w;
            let c1 = $float::splat_t(t, C1_F32);
            let num = c1 * em1;
            let vp = c1 + num;
            let den = $float::splat_t(t, -C3_F32).mla(vp, $float::splat_t(t, C2_F32));
            let base = num / den;
            let dark_y = base.powf_const::<{ math::PQ_EOTF_POW }>();

            let out_v = $float::load_t(t, result);
            let patched = $float::blend(is_dark, dark_y, out_v);
            let patched = $float::blend(v.simd_gt(zero) & v.simd_le(dark_floor), zero, patched);
            let patched = $float::blend(v.simd_ne(v), v, patched);
            patched.store(result);
        }

        #[allow(dead_code)]
        #[inline(always)]
        fn $encode_body<T: $($bound)+>(t: T, source: &[f32; $width], result: &mut [f32; $width]) {
            let v = $float::load_t(t, source);
            let zero = $float::zero_t(t);
            let one = $float::splat_t(t, 1.0);
            let is_dark = v.simd_gt(zero) & v.simd_lt($float::splat_t(t, OETF_POLY_FLOOR));
            let v_safe = $float::blend(is_dark, v, $float::splat_t(t, 1e-8));

            let vp = v_safe.powf_const::<{ math::PQ_OETF_POW1 }>();
            let num = vp.mla($float::splat_t(t, C2_F32), $float::splat_t(t, C1_F32));
            let den = vp.mla($float::splat_t(t, C3_F32), one);
            let ratio = num / den;
            let dark_y = ratio.powf_const::<{ math::PQ_OETF_POW2 }>();

            let out_v = $float::load_t(t, result);
            let patched = $float::blend(is_dark, dark_y, out_v);
            let patched = $float::blend(v.simd_ne(v), v, patched);
            patched.store(result);
        }
    };
}

pq_patch_body!(
    pq_to_linear_patch_x4_body,
    linear_to_pq_patch_x4_body,
    f32x4,
    4,
    F32x4Convert
);
pq_patch_body!(
    pq_to_linear_patch_x8_body,
    linear_to_pq_patch_x8_body,
    f32x8,
    8,
    F32x8Convert + F32x4Convert
);
pq_patch_body!(
    pq_to_linear_patch_x16_body,
    linear_to_pq_patch_x16_body,
    f32x16,
    16,
    F32x16Convert + F32x4Convert
);

#[cold]
#[inline(never)]
fn pq_to_linear_patch_x4_scalar(source: &[f32; 4], result: &mut [f32; 4]) {
    pq_to_linear_patch_x4_body(archmage::ScalarToken, source, result);
}

#[cold]
#[inline(never)]
fn linear_to_pq_patch_x4_scalar(source: &[f32; 4], result: &mut [f32; 4]) {
    linear_to_pq_patch_x4_body(archmage::ScalarToken, source, result);
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[cold]
#[inline(never)]
#[archmage::arcane]
fn pq_to_linear_patch_x4_v3(token: archmage::X64V3Token, source: &[f32; 4], result: &mut [f32; 4]) {
    pq_to_linear_patch_x4_body(token, source, result);
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[cold]
#[inline(never)]
#[archmage::arcane]
fn linear_to_pq_patch_x4_v3(token: archmage::X64V3Token, source: &[f32; 4], result: &mut [f32; 4]) {
    linear_to_pq_patch_x4_body(token, source, result);
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[cold]
#[inline(never)]
#[archmage::arcane]
fn pq_to_linear_patch_x8_v3(token: archmage::X64V3Token, source: &[f32; 8], result: &mut [f32; 8]) {
    pq_to_linear_patch_x8_body(token, source, result);
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[cold]
#[inline(never)]
#[archmage::arcane]
fn linear_to_pq_patch_x8_v3(token: archmage::X64V3Token, source: &[f32; 8], result: &mut [f32; 8]) {
    linear_to_pq_patch_x8_body(token, source, result);
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[cold]
#[inline(never)]
#[archmage::arcane]
fn pq_to_linear_patch_x4_v4(token: archmage::X64V4Token, source: &[f32; 4], result: &mut [f32; 4]) {
    pq_to_linear_patch_x4_body(token, source, result);
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[cold]
#[inline(never)]
#[archmage::arcane]
fn linear_to_pq_patch_x4_v4(token: archmage::X64V4Token, source: &[f32; 4], result: &mut [f32; 4]) {
    linear_to_pq_patch_x4_body(token, source, result);
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[cold]
#[inline(never)]
#[archmage::arcane]
fn pq_to_linear_patch_x8_v4(token: archmage::X64V4Token, source: &[f32; 8], result: &mut [f32; 8]) {
    pq_to_linear_patch_x8_body(token, source, result);
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[cold]
#[inline(never)]
#[archmage::arcane]
fn linear_to_pq_patch_x8_v4(token: archmage::X64V4Token, source: &[f32; 8], result: &mut [f32; 8]) {
    linear_to_pq_patch_x8_body(token, source, result);
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[cold]
#[inline(never)]
#[archmage::arcane]
fn pq_to_linear_patch_x16_v4(
    token: archmage::X64V4Token,
    source: &[f32; 16],
    result: &mut [f32; 16],
) {
    pq_to_linear_patch_x16_body(token, source, result);
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
#[cold]
#[inline(never)]
#[archmage::arcane]
fn linear_to_pq_patch_x16_v4(
    token: archmage::X64V4Token,
    source: &[f32; 16],
    result: &mut [f32; 16],
) {
    linear_to_pq_patch_x16_body(token, source, result);
}

#[cfg(target_arch = "aarch64")]
#[cold]
#[inline(never)]
#[archmage::arcane]
fn pq_to_linear_patch_x4_neon(
    token: archmage::NeonToken,
    source: &[f32; 4],
    result: &mut [f32; 4],
) {
    pq_to_linear_patch_x4_body(token, source, result);
}

#[cfg(target_arch = "aarch64")]
#[cold]
#[inline(never)]
#[archmage::arcane]
fn linear_to_pq_patch_x4_neon(
    token: archmage::NeonToken,
    source: &[f32; 4],
    result: &mut [f32; 4],
) {
    linear_to_pq_patch_x4_body(token, source, result);
}

#[cfg(target_arch = "wasm32")]
#[cold]
#[inline(never)]
#[archmage::arcane]
fn pq_to_linear_patch_x4_wasm128(
    token: archmage::Wasm128Token,
    source: &[f32; 4],
    result: &mut [f32; 4],
) {
    pq_to_linear_patch_x4_body(token, source, result);
}

#[cfg(target_arch = "wasm32")]
#[cold]
#[inline(never)]
#[archmage::arcane]
fn linear_to_pq_patch_x4_wasm128(
    token: archmage::Wasm128Token,
    source: &[f32; 4],
    result: &mut [f32; 4],
) {
    linear_to_pq_patch_x4_body(token, source, result);
}

#[inline(always)]
fn patch_by_4<const N: usize>(
    source: &[f32; N],
    result: &mut [f32; N],
    mut f: impl FnMut(&[f32; 4], &mut [f32; 4]),
) {
    for (s, r) in source
        .as_chunks::<4>()
        .0
        .iter()
        .zip(result.as_chunks_mut::<4>().0.iter_mut())
    {
        f(s, r);
    }
}

pub(crate) trait PqDarkToeToken: F32x4Convert {
    fn patch_decode_x4(self, source: &[f32; 4], result: &mut [f32; 4]);
    fn patch_encode_x4(self, source: &[f32; 4], result: &mut [f32; 4]);
    fn patch_decode_x8(self, source: &[f32; 8], result: &mut [f32; 8]);
    fn patch_encode_x8(self, source: &[f32; 8], result: &mut [f32; 8]);
    fn patch_decode_x16(self, source: &[f32; 16], result: &mut [f32; 16]);
    fn patch_encode_x16(self, source: &[f32; 16], result: &mut [f32; 16]);
}

impl PqDarkToeToken for archmage::ScalarToken {
    #[inline(always)]
    fn patch_decode_x4(self, source: &[f32; 4], result: &mut [f32; 4]) {
        pq_to_linear_patch_x4_scalar(source, result);
    }
    #[inline(always)]
    fn patch_encode_x4(self, source: &[f32; 4], result: &mut [f32; 4]) {
        linear_to_pq_patch_x4_scalar(source, result);
    }
    #[inline(always)]
    fn patch_decode_x8(self, source: &[f32; 8], result: &mut [f32; 8]) {
        patch_by_4(source, result, pq_to_linear_patch_x4_scalar);
    }
    #[inline(always)]
    fn patch_encode_x8(self, source: &[f32; 8], result: &mut [f32; 8]) {
        patch_by_4(source, result, linear_to_pq_patch_x4_scalar);
    }
    #[inline(always)]
    fn patch_decode_x16(self, source: &[f32; 16], result: &mut [f32; 16]) {
        patch_by_4(source, result, pq_to_linear_patch_x4_scalar);
    }
    #[inline(always)]
    fn patch_encode_x16(self, source: &[f32; 16], result: &mut [f32; 16]) {
        patch_by_4(source, result, linear_to_pq_patch_x4_scalar);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
impl PqDarkToeToken for archmage::X64V3Token {
    #[inline(always)]
    fn patch_decode_x4(self, source: &[f32; 4], result: &mut [f32; 4]) {
        pq_to_linear_patch_x4_v3(self, source, result);
    }
    #[inline(always)]
    fn patch_encode_x4(self, source: &[f32; 4], result: &mut [f32; 4]) {
        linear_to_pq_patch_x4_v3(self, source, result);
    }
    #[inline(always)]
    fn patch_decode_x8(self, source: &[f32; 8], result: &mut [f32; 8]) {
        pq_to_linear_patch_x8_v3(self, source, result);
    }
    #[inline(always)]
    fn patch_encode_x8(self, source: &[f32; 8], result: &mut [f32; 8]) {
        linear_to_pq_patch_x8_v3(self, source, result);
    }
    #[inline(always)]
    fn patch_decode_x16(self, source: &[f32; 16], result: &mut [f32; 16]) {
        for (s, r) in source
            .as_chunks::<8>()
            .0
            .iter()
            .zip(result.as_chunks_mut::<8>().0.iter_mut())
        {
            pq_to_linear_patch_x8_v3(self, s, r);
        }
    }
    #[inline(always)]
    fn patch_encode_x16(self, source: &[f32; 16], result: &mut [f32; 16]) {
        for (s, r) in source
            .as_chunks::<8>()
            .0
            .iter()
            .zip(result.as_chunks_mut::<8>().0.iter_mut())
        {
            linear_to_pq_patch_x8_v3(self, s, r);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
impl PqDarkToeToken for archmage::X64V4Token {
    #[inline(always)]
    fn patch_decode_x4(self, source: &[f32; 4], result: &mut [f32; 4]) {
        pq_to_linear_patch_x4_v4(self, source, result);
    }
    #[inline(always)]
    fn patch_encode_x4(self, source: &[f32; 4], result: &mut [f32; 4]) {
        linear_to_pq_patch_x4_v4(self, source, result);
    }
    #[inline(always)]
    fn patch_decode_x8(self, source: &[f32; 8], result: &mut [f32; 8]) {
        pq_to_linear_patch_x8_v4(self, source, result);
    }
    #[inline(always)]
    fn patch_encode_x8(self, source: &[f32; 8], result: &mut [f32; 8]) {
        linear_to_pq_patch_x8_v4(self, source, result);
    }
    #[inline(always)]
    fn patch_decode_x16(self, source: &[f32; 16], result: &mut [f32; 16]) {
        pq_to_linear_patch_x16_v4(self, source, result);
    }
    #[inline(always)]
    fn patch_encode_x16(self, source: &[f32; 16], result: &mut [f32; 16]) {
        linear_to_pq_patch_x16_v4(self, source, result);
    }
}

#[cfg(target_arch = "aarch64")]
impl PqDarkToeToken for archmage::NeonToken {
    #[inline(always)]
    fn patch_decode_x4(self, source: &[f32; 4], result: &mut [f32; 4]) {
        pq_to_linear_patch_x4_neon(self, source, result);
    }
    #[inline(always)]
    fn patch_encode_x4(self, source: &[f32; 4], result: &mut [f32; 4]) {
        linear_to_pq_patch_x4_neon(self, source, result);
    }
    #[inline(always)]
    fn patch_decode_x8(self, source: &[f32; 8], result: &mut [f32; 8]) {
        patch_by_4(source, result, |s, r| {
            pq_to_linear_patch_x4_neon(self, s, r)
        });
    }
    #[inline(always)]
    fn patch_encode_x8(self, source: &[f32; 8], result: &mut [f32; 8]) {
        patch_by_4(source, result, |s, r| {
            linear_to_pq_patch_x4_neon(self, s, r)
        });
    }
    #[inline(always)]
    fn patch_decode_x16(self, source: &[f32; 16], result: &mut [f32; 16]) {
        patch_by_4(source, result, |s, r| {
            pq_to_linear_patch_x4_neon(self, s, r)
        });
    }
    #[inline(always)]
    fn patch_encode_x16(self, source: &[f32; 16], result: &mut [f32; 16]) {
        patch_by_4(source, result, |s, r| {
            linear_to_pq_patch_x4_neon(self, s, r)
        });
    }
}

#[cfg(target_arch = "wasm32")]
impl PqDarkToeToken for archmage::Wasm128Token {
    #[inline(always)]
    fn patch_decode_x4(self, source: &[f32; 4], result: &mut [f32; 4]) {
        pq_to_linear_patch_x4_wasm128(self, source, result);
    }
    #[inline(always)]
    fn patch_encode_x4(self, source: &[f32; 4], result: &mut [f32; 4]) {
        linear_to_pq_patch_x4_wasm128(self, source, result);
    }
    #[inline(always)]
    fn patch_decode_x8(self, source: &[f32; 8], result: &mut [f32; 8]) {
        patch_by_4(source, result, |s, r| {
            pq_to_linear_patch_x4_wasm128(self, s, r)
        });
    }
    #[inline(always)]
    fn patch_encode_x8(self, source: &[f32; 8], result: &mut [f32; 8]) {
        patch_by_4(source, result, |s, r| {
            linear_to_pq_patch_x4_wasm128(self, s, r)
        });
    }
    #[inline(always)]
    fn patch_decode_x16(self, source: &[f32; 16], result: &mut [f32; 16]) {
        patch_by_4(source, result, |s, r| {
            pq_to_linear_patch_x4_wasm128(self, s, r)
        });
    }
    #[inline(always)]
    fn patch_encode_x16(self, source: &[f32; 16], result: &mut [f32; 16]) {
        patch_by_4(source, result, |s, r| {
            linear_to_pq_patch_x4_wasm128(self, s, r)
        });
    }
}

macro_rules! pq_for {
    (
        $decode_poly:ident,
        $try_decode:ident,
        $decode_cfg:ident,
        $decode:ident,
        $decode_slice_cfg:ident,
        $decode_slice:ident,
        $encode_poly:ident,
        $try_encode:ident,
        $encode_cfg:ident,
        $encode:ident,
        $encode_slice_cfg:ident,
        $encode_slice:ident,
        $patch_decode:ident,
        $patch_encode:ident,
        $float:ident,
        $blend_poly:ident,
        $width:literal,
        $($bound:tt)+
    ) => {
        #[inline(always)]
        fn $decode_poly<T: $($bound)+>(t: T, v: $float<T>) -> ($float<T>, $float<T>) {
            let floor = $float::splat_t(t, EOTF_POLY_FLOOR);
            let one = $float::splat_t(t, 1.0);
            // Keep the small fit above its pole even on lanes replaced below.
            let a = if cfg!(target_arch = "aarch64") {
                $float::blend(v.simd_gt(floor), v, floor).min_fast(one)
            } else {
                const NON_NAN_RHS: u32 = magetypes::numerics::ExtremaOptions::new()
                    .non_nan_rhs()
                    .bits();
                v.max_with::<NON_NAN_RHS>(floor).min_fast(one)
            };
            let x = a.mla(a, a);
            let small = a.simd_lt($float::splat_t(t, 0.25));
            let p = polynomial::$blend_poly(t, x, small, PQ_EOTF_P_SMALL, PQ_EOTF_P_LARGE);
            let q = polynomial::$blend_poly(t, x, small, PQ_EOTF_Q_SMALL, PQ_EOTF_Q_LARGE);
            let zero = $float::zero_t(t);
            let positive = v.simd_gt(zero);
            let result = $float::blend(v.simd_ge(one), one, (p / q) & positive);
            let fallback = (positive & v.simd_lt(floor)) | v.simd_ne(v);
            (fallback, result)
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $try_decode<T: $($bound)+>(t: T, v: $float<T>) -> Option<$float<T>> {
            let (fallback, result) = $decode_poly(t, v);
            if fallback.bitcast_to_i32().any_true() {
                None
            } else {
                Some(result)
            }
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $decode_cfg<T: $($bound)+, const RAILS: bool>(t: T, v: $float<T>) -> $float<T> {
            if !RAILS {
                let x = v.mla(v, v);
                let small = v.simd_lt($float::splat_t(t, 0.25));
                let p = polynomial::$blend_poly(t, x, small, PQ_EOTF_P_SMALL, PQ_EOTF_P_LARGE);
                let q = polynomial::$blend_poly(t, x, small, PQ_EOTF_Q_SMALL, PQ_EOTF_Q_LARGE);
                return p / q;
            }
            let (fallback, result) = $decode_poly(t, v);
            if fallback.bitcast_to_i32().any_true() {
                let src = v.to_array();
                let mut out = result.to_array();
                t.$patch_decode(&src, &mut out);
                $float::from_array_t(t, out)
            } else {
                result
            }
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $decode<T: $($bound)+>(t: T, v: $float<T>) -> $float<T> {
            $decode_cfg::<T, true>(t, v)
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $decode_slice_cfg<T: $($bound)+, const RAILS: bool>(t: T, values: &mut [f32]) {
            if !RAILS {
                #[cfg(not(target_arch = "x86_64"))]
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
            // Keep the already-computed SIMD polynomial result; store directly from register on
            // the hot path (no stack bounce / sret) and patch by pointer only when a lane falls
            // into the dark toe or NaN.
            let (chunks, remainder) = values.as_chunks_mut::<$width>();
            for slot in chunks {
                let (fallback, result) = $decode_poly(t, $float::load_t(t, slot));
                if fallback.bitcast_to_i32().any_true() {
                    let mut out = result.to_array();
                    t.$patch_decode(slot, &mut out);
                    *slot = out;
                } else {
                    result.store(slot);
                }
            }
            if !remainder.is_empty() {
                #[cfg(target_arch = "x86_64")]
                {
                    let mut block = [0.0; $width];
                    block[..remainder.len()].copy_from_slice(remainder);
                    let (fallback, result) = $decode_poly(t, $float::load_t(t, &block));
                    let mut out = result.to_array();
                    if fallback.bitcast_to_i32().any_true() {
                        t.$patch_decode(&block, &mut out);
                    }
                    remainder.copy_from_slice(&out[..remainder.len()]);
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    fn tail<T: $($bound)+>(t: T, chunk: [f32; $width]) -> [f32; $width] {
                        $decode(t, $float::from_array_t(t, chunk)).to_array()
                    }
                    crate::kernels::finish_tail_cold(remainder, t, tail::<T>);
                }
            }
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $decode_slice<T: $($bound)+>(t: T, values: &mut [f32]) {
            $decode_slice_cfg::<T, true>(t, values);
        }

        #[inline(always)]
        fn $encode_poly<T: $($bound)+>(t: T, v: $float<T>) -> ($float<T>, $float<T>) {
            let floor = $float::splat_t(t, OETF_POLY_FLOOR);
            let one = $float::splat_t(t, 1.0);
            let a = if cfg!(target_arch = "aarch64") {
                $float::blend(v.simd_gt(floor), v, floor).min_fast(one)
            } else {
                const NON_NAN_RHS: u32 = magetypes::numerics::ExtremaOptions::new()
                    .non_nan_rhs()
                    .bits();
                v.max_with::<NON_NAN_RHS>(floor).min_fast(one)
            };
            let a4 = a.fast_sqrt().fast_sqrt();
            let small = a4.simd_lt($float::splat_t(t, 0.1));
            let p = polynomial::$blend_poly(t, a4, small, PQ_INV_P_SMALL, PQ_INV_P_LARGE);
            let q = polynomial::$blend_poly(t, a4, small, PQ_INV_Q_SMALL, PQ_INV_Q_LARGE);
            let zero = $float::zero_t(t);
            let positive = v.simd_gt(zero);
            let result = $float::blend(v.simd_ge(one), one, (p / q) & positive);
            let fallback = (positive & v.simd_lt(floor)) | v.simd_ne(v);
            (fallback, result)
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $try_encode<T: $($bound)+>(t: T, v: $float<T>) -> Option<$float<T>> {
            let (fallback, result) = $encode_poly(t, v);
            if fallback.bitcast_to_i32().any_true() {
                None
            } else {
                Some(result)
            }
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $encode_cfg<T: $($bound)+, const RAILS: bool>(t: T, v: $float<T>) -> $float<T> {
            if !RAILS {
                let a4 = v.fast_sqrt().fast_sqrt();
                let small = a4.simd_lt($float::splat_t(t, 0.1));
                let p = polynomial::$blend_poly(t, a4, small, PQ_INV_P_SMALL, PQ_INV_P_LARGE);
                let q = polynomial::$blend_poly(t, a4, small, PQ_INV_Q_SMALL, PQ_INV_Q_LARGE);
                return p / q;
            }
            let (fallback, result) = $encode_poly(t, v);
            if fallback.bitcast_to_i32().any_true() {
                let src = v.to_array();
                let mut out = result.to_array();
                t.$patch_encode(&src, &mut out);
                $float::from_array_t(t, out)
            } else {
                result
            }
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $encode<T: $($bound)+>(t: T, v: $float<T>) -> $float<T> {
            $encode_cfg::<T, true>(t, v)
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $encode_slice_cfg<T: $($bound)+, const RAILS: bool>(t: T, values: &mut [f32]) {
            if !RAILS {
                #[cfg(not(target_arch = "x86_64"))]
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
            let (chunks, remainder) = values.as_chunks_mut::<$width>();
            for slot in chunks {
                let (fallback, result) = $encode_poly(t, $float::load_t(t, slot));
                if fallback.bitcast_to_i32().any_true() {
                    let mut out = result.to_array();
                    t.$patch_encode(slot, &mut out);
                    *slot = out;
                } else {
                    result.store(slot);
                }
            }
            if !remainder.is_empty() {
                #[cfg(target_arch = "x86_64")]
                {
                    let mut block = [0.0; $width];
                    block[..remainder.len()].copy_from_slice(remainder);
                    let (fallback, result) = $encode_poly(t, $float::load_t(t, &block));
                    let mut out = result.to_array();
                    if fallback.bitcast_to_i32().any_true() {
                        t.$patch_encode(&block, &mut out);
                    }
                    remainder.copy_from_slice(&out[..remainder.len()]);
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    fn tail<T: $($bound)+>(t: T, chunk: [f32; $width]) -> [f32; $width] {
                        $encode(t, $float::from_array_t(t, chunk)).to_array()
                    }
                    crate::kernels::finish_tail_cold(remainder, t, tail::<T>);
                }
            }
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $encode_slice<T: $($bound)+>(t: T, values: &mut [f32]) {
            $encode_slice_cfg::<T, true>(t, values);
        }
    };
}

pq_for!(
    pq_to_linear_poly_x4,
    try_pq_to_linear_x4,
    pq_to_linear_x4_cfg,
    pq_to_linear_x4,
    pq_to_linear_slice_x4_cfg,
    pq_to_linear_slice_x4,
    linear_to_pq_poly_x4,
    try_linear_to_pq_x4,
    linear_to_pq_x4_cfg,
    linear_to_pq_x4,
    linear_to_pq_slice_x4_cfg,
    linear_to_pq_slice_x4,
    patch_decode_x4,
    patch_encode_x4,
    f32x4,
    eval_blended_poly_x4,
    4,
    F32x4Convert + PqDarkToeToken
);
pq_for!(
    pq_to_linear_poly_x8,
    try_pq_to_linear_x8,
    pq_to_linear_x8_cfg,
    pq_to_linear_x8,
    pq_to_linear_slice_x8_cfg,
    pq_to_linear_slice_x8,
    linear_to_pq_poly_x8,
    try_linear_to_pq_x8,
    linear_to_pq_x8_cfg,
    linear_to_pq_x8,
    linear_to_pq_slice_x8_cfg,
    linear_to_pq_slice_x8,
    patch_decode_x8,
    patch_encode_x8,
    f32x8,
    eval_blended_poly_x8,
    8,
    F32x8Convert + F32x4Convert + PqDarkToeToken
);
pq_for!(
    pq_to_linear_poly_x16,
    try_pq_to_linear_x16,
    pq_to_linear_x16_cfg,
    pq_to_linear_x16,
    pq_to_linear_slice_x16_cfg,
    pq_to_linear_slice_x16,
    linear_to_pq_poly_x16,
    try_linear_to_pq_x16,
    linear_to_pq_x16_cfg,
    linear_to_pq_x16,
    linear_to_pq_slice_x16_cfg,
    linear_to_pq_slice_x16,
    patch_decode_x16,
    patch_encode_x16,
    f32x16,
    eval_blended_poly_x16,
    16,
    F32x16Convert + F32x4Convert + PqDarkToeToken
);
