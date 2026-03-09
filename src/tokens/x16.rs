//! 16×f32 `#[rite]` functions (AVX-512 on x86-64).
//!
//! All functions use `[f32; 16]` at the boundary — zero-cost transmute to/from
//! the underlying `__m512` register. No `magetypes` types in the public API.
//!
//! Call these from inside your own `#[arcane]` function with a matching token.
//! They inline fully — no dispatch, no function-pointer indirection.

use archmage::rite;

pub use archmage::X64V4Token;

use magetypes::simd::v4::f32x16 as mt_f32x16;

// sRGB transfer function constants (C0-continuous moxcms, matching rational polynomial)
const SRGB_LINEAR_THRESHOLD: f32 = 0.039_293_37;
const LINEAR_THRESHOLD: f32 = 0.003_041_282_6;
const LINEAR_SCALE: f32 = 1.0 / 12.92;
const TWELVE_92: f32 = 12.92;

// ============================================================================
// x16 functions — operate on [f32; 16]
// ============================================================================

/// Convert 16 sRGB values to linear. Input clamped to \[0, 1\].
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn srgb_to_linear_v4(token: X64V4Token, srgb: [f32; 16]) -> [f32; 16] {
    use crate::rational_poly::{S2L_P, S2L_Q};

    let zero = mt_f32x16::zero(token);
    let one = mt_f32x16::splat(token, 1.0);
    let srgb = mt_f32x16::from_array(token, srgb).max(zero).min(one);

    let linear_result = srgb * mt_f32x16::splat(token, LINEAR_SCALE);

    // Rational polynomial P(x)/Q(x) via Horner's method
    let x = srgb;
    let yp = mt_f32x16::splat(token, S2L_P[4]).mul_add(x, mt_f32x16::splat(token, S2L_P[3]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, S2L_P[2]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, S2L_P[1]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, S2L_P[0]));

    let yq = mt_f32x16::splat(token, S2L_Q[4]).mul_add(x, mt_f32x16::splat(token, S2L_Q[3]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, S2L_Q[2]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, S2L_Q[1]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, S2L_Q[0]));

    let power_result = (yp / yq).min(one);

    let mask = srgb.simd_lt(mt_f32x16::splat(token, SRGB_LINEAR_THRESHOLD));
    mt_f32x16::blend(mask, linear_result, power_result).to_array()
}

/// Convert 16 linear values to sRGB. Input clamped to \[0, 1\].
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn linear_to_srgb_v4(token: X64V4Token, linear: [f32; 16]) -> [f32; 16] {
    use crate::rational_poly::{L2S_P, L2S_Q};

    let zero = mt_f32x16::zero(token);
    let one = mt_f32x16::splat(token, 1.0);
    let linear = mt_f32x16::from_array(token, linear).max(zero).min(one);

    let linear_result = linear * mt_f32x16::splat(token, TWELVE_92);

    // sqrt transform + rational polynomial P(√x)/Q(√x) via Horner's method
    let x = linear.sqrt();
    let yp = mt_f32x16::splat(token, L2S_P[4]).mul_add(x, mt_f32x16::splat(token, L2S_P[3]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, L2S_P[2]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, L2S_P[1]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, L2S_P[0]));

    let yq = mt_f32x16::splat(token, L2S_Q[4]).mul_add(x, mt_f32x16::splat(token, L2S_Q[3]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, L2S_Q[2]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, L2S_Q[1]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, L2S_Q[0]));

    let power_result = (yp / yq).min(one);

    let mask = linear.simd_lt(mt_f32x16::splat(token, LINEAR_THRESHOLD));
    mt_f32x16::blend(mask, linear_result, power_result).to_array()
}

/// Convert 16 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn gamma_to_linear_v4(token: X64V4Token, encoded: [f32; 16], gamma: f32) -> [f32; 16] {
    // 2×x8 via rites/x8 — pow_midp not available on f32x16
    let t3 = token.v3();
    let lo: [f32; 8] = encoded[..8].try_into().unwrap();
    let hi: [f32; 8] = encoded[8..].try_into().unwrap();
    let lo = super::x8::gamma_to_linear_v3(t3, lo, gamma);
    let hi = super::x8::gamma_to_linear_v3(t3, hi, gamma);
    let mut out = [0.0f32; 16];
    out[..8].copy_from_slice(&lo);
    out[8..].copy_from_slice(&hi);
    out
}

/// Convert 16 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn linear_to_gamma_v4(token: X64V4Token, linear: [f32; 16], gamma: f32) -> [f32; 16] {
    // 2×x8 via rites/x8 — pow_midp not available on f32x16
    let t3 = token.v3();
    let lo: [f32; 8] = linear[..8].try_into().unwrap();
    let hi: [f32; 8] = linear[8..].try_into().unwrap();
    let lo = super::x8::linear_to_gamma_v3(t3, lo, gamma);
    let hi = super::x8::linear_to_gamma_v3(t3, hi, gamma);
    let mut out = [0.0f32; 16];
    out[..8].copy_from_slice(&lo);
    out[8..].copy_from_slice(&hi);
    out
}

// ============================================================================
// x16 LUT functions — u8↔f32
// ============================================================================

/// Convert 16 sRGB u8 values to linear f32 via 256-entry LUT.
///
/// Pure table lookup — no math. The 1KB table fits in L1 cache.
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn srgb_u8_to_linear_v4(_token: X64V4Token, srgb: [u8; 16]) -> [f32; 16] {
    let lut = &crate::const_luts::LINEAR_TABLE_8;
    [
        lut[srgb[0] as usize],
        lut[srgb[1] as usize],
        lut[srgb[2] as usize],
        lut[srgb[3] as usize],
        lut[srgb[4] as usize],
        lut[srgb[5] as usize],
        lut[srgb[6] as usize],
        lut[srgb[7] as usize],
        lut[srgb[8] as usize],
        lut[srgb[9] as usize],
        lut[srgb[10] as usize],
        lut[srgb[11] as usize],
        lut[srgb[12] as usize],
        lut[srgb[13] as usize],
        lut[srgb[14] as usize],
        lut[srgb[15] as usize],
    ]
}

/// Convert sRGB u8 values to linear f32 using 16-wide LUT lookup.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn srgb_u8_to_linear_slice_v4(_token: X64V4Token, input: &[u8], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let lut = &crate::const_luts::LINEAR_TABLE_8;
    let (in_chunks, in_remainder) = input.as_chunks::<16>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<16>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = srgb_u8_to_linear_v4(_token, *inp);
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        *out = lut[*inp as usize];
    }
}

/// Convert 16 linear f32 values to sRGB u8 via LUT. Input clamped to \[0, 1\].
///
/// Uses a 4096-entry const LUT with bitmask indexing (`& 0xFFF`) for
/// provably safe bounds. SIMD accelerates the clamp and scale; lookups
/// are scalar from an L1-resident 4KB table.
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn linear_to_srgb_u8_v4(token: X64V4Token, linear: [f32; 16]) -> [u8; 16] {
    let zero = mt_f32x16::zero(token);
    let one = mt_f32x16::splat(token, 1.0);
    let linear = mt_f32x16::from_array(token, linear).max(zero).min(one);
    let scaled = linear * mt_f32x16::splat(token, 4095.0) + mt_f32x16::splat(token, 0.5);
    let arr = scaled.to_array();
    let lut = &crate::const_luts::LINEAR_TO_SRGB_U8;
    [
        lut[arr[0] as usize & 0xFFF],
        lut[arr[1] as usize & 0xFFF],
        lut[arr[2] as usize & 0xFFF],
        lut[arr[3] as usize & 0xFFF],
        lut[arr[4] as usize & 0xFFF],
        lut[arr[5] as usize & 0xFFF],
        lut[arr[6] as usize & 0xFFF],
        lut[arr[7] as usize & 0xFFF],
        lut[arr[8] as usize & 0xFFF],
        lut[arr[9] as usize & 0xFFF],
        lut[arr[10] as usize & 0xFFF],
        lut[arr[11] as usize & 0xFFF],
        lut[arr[12] as usize & 0xFFF],
        lut[arr[13] as usize & 0xFFF],
        lut[arr[14] as usize & 0xFFF],
        lut[arr[15] as usize & 0xFFF],
    ]
}

// ============================================================================
// Slice functions — process &mut [f32] with x16 chunking
// ============================================================================

/// Convert sRGB f32 values to linear in-place using 16-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn srgb_to_linear_slice_v4(token: X64V4Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<16>();

    for chunk in chunks {
        *chunk = srgb_to_linear_v4(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

/// Convert linear f32 values to sRGB in-place using 16-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn linear_to_srgb_slice_v4(token: X64V4Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<16>();

    for chunk in chunks {
        *chunk = linear_to_srgb_v4(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_srgb(*v);
    }
}

/// Convert gamma-encoded f32 values to linear in-place using 16-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn gamma_to_linear_slice_v4(token: X64V4Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<16>();

    for chunk in chunks {
        *chunk = gamma_to_linear_v4(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

/// Convert linear f32 values to gamma-encoded in-place using 16-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn linear_to_gamma_slice_v4(token: X64V4Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<16>();

    for chunk in chunks {
        *chunk = linear_to_gamma_v4(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_gamma(*v, gamma);
    }
}

/// Convert linear f32 values to sRGB u8 using 16-wide SIMD + LUT.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn linear_to_srgb_u8_slice_v4(token: X64V4Token, input: &[f32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());
    let (in_chunks, in_remainder) = input.as_chunks::<16>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<16>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = linear_to_srgb_u8_v4(token, *inp);
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        *out = crate::scalar::linear_to_srgb_u8(*inp);
    }
}

// ============================================================================
// Transfer function rites (behind `transfer` feature)
// ============================================================================

/// Convert 16 sRGB values to linear (rational polynomial, no powf).
#[cfg(feature = "transfer")]
#[rite]
pub fn tf_srgb_to_linear_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    use crate::rational_poly::{LINEAR_SCALE, S2L_P, S2L_Q, SRGB_THRESHOLD};

    let v = mt_f32x16::from_array(token, v);
    let one = mt_f32x16::splat(token, 1.0);
    let threshold = mt_f32x16::splat(token, SRGB_THRESHOLD);
    let inv_12_92 = mt_f32x16::splat(token, LINEAR_SCALE);

    let linear = v * inv_12_92;
    let poly = eval_rational_poly_x16(token, v, S2L_P, S2L_Q).min(one);

    let mask = v.simd_le(threshold);
    mt_f32x16::blend(mask, linear, poly).to_array()
}

/// Convert 16 linear values to sRGB (rational polynomial, no powf).
#[cfg(feature = "transfer")]
#[rite]
pub fn tf_linear_to_srgb_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    use crate::rational_poly::{L2S_P, L2S_Q, LINEAR_THRESHOLD, TWELVE_92};

    let v = mt_f32x16::from_array(token, v);
    let one = mt_f32x16::splat(token, 1.0);
    let threshold = mt_f32x16::splat(token, LINEAR_THRESHOLD);
    let scale = mt_f32x16::splat(token, TWELVE_92);

    let linear = v * scale;
    let s = v.sqrt();
    let poly = eval_rational_poly_x16(token, s, L2S_P, L2S_Q).min(one);

    let mask = v.simd_le(threshold);
    mt_f32x16::blend(mask, linear, poly).to_array()
}

/// Convert 16 PQ signal values to linear.
#[cfg(feature = "transfer")]
#[rite]
pub fn pq_to_linear_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    use crate::tf::pq::{PQ_EOTF_P_LARGE, PQ_EOTF_P_SMALL, PQ_EOTF_Q_LARGE, PQ_EOTF_Q_SMALL};

    let v = mt_f32x16::from_array(token, v);
    let zero = mt_f32x16::zero(token);
    let a = v.max(zero);
    let x = a.mul_add(a, a); // x = a + a*a

    let threshold = mt_f32x16::splat(token, 0.25);
    let large = eval_rational_poly_x16(token, x, PQ_EOTF_P_LARGE, PQ_EOTF_Q_LARGE);
    let small = eval_rational_poly_x16(token, x, PQ_EOTF_P_SMALL, PQ_EOTF_Q_SMALL);

    let mask = a.simd_lt(threshold);
    let result = mt_f32x16::blend(mask, small, large);

    let pos_mask = v.simd_gt(zero);
    (result & pos_mask).to_array()
}

/// Convert 16 linear values to PQ signal.
#[cfg(feature = "transfer")]
#[rite]
pub fn linear_to_pq_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    use crate::tf::pq::{PQ_INV_P_LARGE, PQ_INV_P_SMALL, PQ_INV_Q_LARGE, PQ_INV_Q_SMALL};

    let v = mt_f32x16::from_array(token, v);
    let zero = mt_f32x16::zero(token);
    let a = v.max(zero);
    let a4 = a.sqrt().sqrt();

    let threshold = mt_f32x16::splat(token, 0.1);
    let large = eval_rational_poly_x16(token, a4, PQ_INV_P_LARGE, PQ_INV_Q_LARGE);
    let small = eval_rational_poly_x16(token, a4, PQ_INV_P_SMALL, PQ_INV_Q_SMALL);

    let mask = a4.simd_lt(threshold);
    let result = mt_f32x16::blend(mask, small, large);

    let pos_mask = v.simd_gt(zero);
    (result & pos_mask).to_array()
}

/// Convert 16 BT.709 encoded values to linear (2×x8 via tokens::x8).
#[cfg(feature = "transfer")]
#[rite]
pub fn bt709_to_linear_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    let t3 = token.v3();
    let lo: [f32; 8] = v[..8].try_into().unwrap();
    let hi: [f32; 8] = v[8..].try_into().unwrap();
    let lo = super::x8::bt709_to_linear_v3(t3, lo);
    let hi = super::x8::bt709_to_linear_v3(t3, hi);
    let mut out = [0.0f32; 16];
    out[..8].copy_from_slice(&lo);
    out[8..].copy_from_slice(&hi);
    out
}

/// Convert 16 linear values to BT.709 encoded (2×x8 via tokens::x8).
#[cfg(feature = "transfer")]
#[rite]
pub fn linear_to_bt709_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    let t3 = token.v3();
    let lo: [f32; 8] = v[..8].try_into().unwrap();
    let hi: [f32; 8] = v[8..].try_into().unwrap();
    let lo = super::x8::linear_to_bt709_v3(t3, lo);
    let hi = super::x8::linear_to_bt709_v3(t3, hi);
    let mut out = [0.0f32; 16];
    out[..8].copy_from_slice(&lo);
    out[8..].copy_from_slice(&hi);
    out
}

/// Convert 16 HLG signal values to linear (2×x8 via tokens::x8).
#[cfg(feature = "transfer")]
#[rite]
pub fn hlg_to_linear_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    let t3 = token.v3();
    let lo: [f32; 8] = v[..8].try_into().unwrap();
    let hi: [f32; 8] = v[8..].try_into().unwrap();
    let lo = super::x8::hlg_to_linear_v3(t3, lo);
    let hi = super::x8::hlg_to_linear_v3(t3, hi);
    let mut out = [0.0f32; 16];
    out[..8].copy_from_slice(&lo);
    out[8..].copy_from_slice(&hi);
    out
}

/// Convert 16 linear values to HLG signal (2×x8 via tokens::x8).
#[cfg(feature = "transfer")]
#[rite]
pub fn linear_to_hlg_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    let t3 = token.v3();
    let lo: [f32; 8] = v[..8].try_into().unwrap();
    let hi: [f32; 8] = v[8..].try_into().unwrap();
    let lo = super::x8::linear_to_hlg_v3(t3, lo);
    let hi = super::x8::linear_to_hlg_v3(t3, hi);
    let mut out = [0.0f32; 16];
    out[..8].copy_from_slice(&lo);
    out[8..].copy_from_slice(&hi);
    out
}

#[cfg(feature = "transfer")]
macro_rules! tf_slice_v4 {
    ($name:ident, $rite:ident, $scalar:path) => {
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        #[rite]
        pub fn $name(token: X64V4Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                *chunk = $rite(token, *chunk);
            }
            for v in remainder {
                *v = $scalar(*v);
            }
        }
    };
}

#[cfg(feature = "transfer")]
tf_slice_v4!(
    tf_srgb_to_linear_slice_v4,
    tf_srgb_to_linear_v4,
    crate::tf::srgb_to_linear
);
#[cfg(feature = "transfer")]
tf_slice_v4!(
    tf_linear_to_srgb_slice_v4,
    tf_linear_to_srgb_v4,
    crate::tf::linear_to_srgb
);
#[cfg(feature = "transfer")]
tf_slice_v4!(
    bt709_to_linear_slice_v4,
    bt709_to_linear_v4,
    crate::tf::bt709_to_linear
);
#[cfg(feature = "transfer")]
tf_slice_v4!(
    linear_to_bt709_slice_v4,
    linear_to_bt709_v4,
    crate::tf::linear_to_bt709
);
#[cfg(feature = "transfer")]
tf_slice_v4!(
    pq_to_linear_slice_v4,
    pq_to_linear_v4,
    crate::tf::pq_to_linear
);
#[cfg(feature = "transfer")]
tf_slice_v4!(
    linear_to_pq_slice_v4,
    linear_to_pq_v4,
    crate::tf::linear_to_pq
);
#[cfg(feature = "transfer")]
tf_slice_v4!(
    hlg_to_linear_slice_v4,
    hlg_to_linear_v4,
    crate::tf::hlg_to_linear
);
#[cfg(feature = "transfer")]
tf_slice_v4!(
    linear_to_hlg_slice_v4,
    linear_to_hlg_v4,
    crate::tf::linear_to_hlg
);

#[cfg(feature = "transfer")]
#[inline(always)]
fn eval_rational_poly_x16(
    t: X64V4Token,
    x: magetypes::simd::v4::f32x16,
    p: [f32; 5],
    q: [f32; 5],
) -> magetypes::simd::v4::f32x16 {
    let mut yp = mt_f32x16::splat(t, p[4]);
    yp = yp.mul_add(x, mt_f32x16::splat(t, p[3]));
    yp = yp.mul_add(x, mt_f32x16::splat(t, p[2]));
    yp = yp.mul_add(x, mt_f32x16::splat(t, p[1]));
    yp = yp.mul_add(x, mt_f32x16::splat(t, p[0]));

    let mut yq = mt_f32x16::splat(t, q[4]);
    yq = yq.mul_add(x, mt_f32x16::splat(t, q[3]));
    yq = yq.mul_add(x, mt_f32x16::splat(t, q[2]));
    yq = yq.mul_add(x, mt_f32x16::splat(t, q[1]));
    yq = yq.mul_add(x, mt_f32x16::splat(t, q[0]));

    yp / yq
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use archmage::SimdToken;

    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    fn get_token() -> Option<X64V4Token> {
        X64V4Token::try_new()
    }

    #[archmage::arcane]
    fn call_srgb_to_linear(token: X64V4Token, input: [f32; 16]) -> [f32; 16] {
        srgb_to_linear_v4(token, input)
    }

    #[archmage::arcane]
    fn call_linear_to_srgb(token: X64V4Token, input: [f32; 16]) -> [f32; 16] {
        linear_to_srgb_v4(token, input)
    }

    #[archmage::arcane]
    fn call_srgb_to_linear_slice(token: X64V4Token, values: &mut [f32]) {
        srgb_to_linear_slice_v4(token, values);
    }

    #[archmage::arcane]
    fn call_linear_to_srgb_slice(token: X64V4Token, values: &mut [f32]) {
        linear_to_srgb_slice_v4(token, values);
    }

    #[archmage::arcane]
    fn call_linear_to_srgb_u8(token: X64V4Token, input: [f32; 16]) -> [u8; 16] {
        linear_to_srgb_u8_v4(token, input)
    }

    #[test]
    fn test_x16_linear_to_srgb_u8() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX-512 not available");
            return;
        };

        let input = [
            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
        ];
        let result = call_linear_to_srgb_u8(token, input);

        for (i, (&got, &inp)) in result.iter().zip(input.iter()).enumerate() {
            let expected = crate::scalar::linear_to_srgb_u8(inp);
            assert!(
                (got as i32 - expected as i32).abs() <= 1,
                "u8 mismatch at {}: got {}, expected {} (input={})",
                i,
                got,
                expected,
                inp
            );
        }
    }

    #[test]
    fn test_x16_srgb_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX-512 not available");
            return;
        };

        let input = [
            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
        ];
        let linear = call_srgb_to_linear(token, input);
        let roundtrip = call_linear_to_srgb(token, linear);

        for (i, (&orig, &rt)) in input.iter().zip(roundtrip.iter()).enumerate() {
            assert!(
                (orig - rt).abs() < 1e-4,
                "roundtrip failed at {}: {} -> {}",
                i,
                orig,
                rt
            );
        }
    }

    #[test]
    fn test_x16_matches_scalar() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX-512 not available");
            return;
        };

        let input = [
            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
        ];
        let result = call_srgb_to_linear(token, input);

        for (i, (&got, &inp)) in result.iter().zip(input.iter()).enumerate() {
            let expected = crate::scalar::srgb_to_linear(inp);
            assert!(
                (got - expected).abs() < 1e-5,
                "mismatch at {}: got {}, expected {}",
                i,
                got,
                expected
            );
        }
    }

    #[test]
    fn test_slice_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX-512 not available");
            return;
        };

        let mut values: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        let original = values.clone();

        call_srgb_to_linear_slice(token, &mut values);
        call_linear_to_srgb_slice(token, &mut values);

        for (i, (&orig, &conv)) in original.iter().zip(values.iter()).enumerate() {
            assert!(
                (orig - conv).abs() < 1e-4,
                "roundtrip failed at {}: {} -> {}",
                i,
                orig,
                conv
            );
        }
    }

    #[archmage::arcane]
    fn call_gamma_to_linear(token: X64V4Token, input: [f32; 16], gamma: f32) -> [f32; 16] {
        gamma_to_linear_v4(token, input, gamma)
    }

    #[archmage::arcane]
    fn call_linear_to_gamma(token: X64V4Token, input: [f32; 16], gamma: f32) -> [f32; 16] {
        linear_to_gamma_v4(token, input, gamma)
    }

    #[archmage::arcane]
    fn call_srgb_u8_to_linear(token: X64V4Token, input: [u8; 16]) -> [f32; 16] {
        srgb_u8_to_linear_v4(token, input)
    }

    #[archmage::arcane]
    fn call_srgb_u8_to_linear_slice(token: X64V4Token, input: &[u8], output: &mut [f32]) {
        srgb_u8_to_linear_slice_v4(token, input, output);
    }

    #[archmage::arcane]
    fn call_gamma_to_linear_slice(token: X64V4Token, values: &mut [f32], gamma: f32) {
        gamma_to_linear_slice_v4(token, values, gamma);
    }

    #[archmage::arcane]
    fn call_linear_to_gamma_slice(token: X64V4Token, values: &mut [f32], gamma: f32) {
        linear_to_gamma_slice_v4(token, values, gamma);
    }

    #[archmage::arcane]
    fn call_linear_to_srgb_u8_slice(token: X64V4Token, input: &[f32], output: &mut [u8]) {
        linear_to_srgb_u8_slice_v4(token, input, output);
    }

    #[test]
    fn test_gamma_roundtrip() {
        let Some(token) = get_token() else { return };

        let input: [f32; 16] = core::array::from_fn(|i| i as f32 / 15.0);
        let linear = call_gamma_to_linear(token, input, 2.2);
        let roundtrip = call_linear_to_gamma(token, linear, 2.2);

        for (i, (&orig, &rt)) in input.iter().zip(roundtrip.iter()).enumerate() {
            assert!(
                (orig - rt).abs() < 1e-3,
                "gamma roundtrip failed at {}: {} -> {}",
                i,
                orig,
                rt
            );
        }
    }

    #[test]
    fn test_u8_to_linear_v4() {
        let Some(token) = get_token() else { return };

        let input: [u8; 16] = [
            0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 255,
        ];
        let result = call_srgb_u8_to_linear(token, input);

        for (&got, &inp) in result.iter().zip(input.iter()) {
            let expected = crate::scalar::srgb_u8_to_linear(inp);
            assert!((got - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_u8_to_linear_slice_v4() {
        let Some(token) = get_token() else { return };

        // 19 elements: exercises 16-wide chunk + 3-element remainder
        let input: Vec<u8> = (0..19).map(|i| (i * 13) as u8).collect();
        let mut output = vec![0.0f32; 19];
        call_srgb_u8_to_linear_slice(token, &input, &mut output);

        for (&got, &inp) in output.iter().zip(input.iter()) {
            let expected = crate::scalar::srgb_u8_to_linear(inp);
            assert!((got - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gamma_slice_roundtrip() {
        let Some(token) = get_token() else { return };

        // 19 elements for remainder coverage
        let mut values: Vec<f32> = (0..19).map(|i| i as f32 / 18.0).collect();
        let original = values.clone();

        call_gamma_to_linear_slice(token, &mut values, 2.2);
        call_linear_to_gamma_slice(token, &mut values, 2.2);

        for (i, (&orig, &conv)) in original.iter().zip(values.iter()).enumerate() {
            assert!(
                (orig - conv).abs() < 1e-3,
                "gamma slice roundtrip at {}: {} -> {}",
                i,
                orig,
                conv
            );
        }
    }

    #[test]
    fn test_linear_to_srgb_u8_slice_v4() {
        let Some(token) = get_token() else { return };

        let input: Vec<f32> = (0..19).map(|i| i as f32 / 18.0).collect();
        let mut output = vec![0u8; 19];
        call_linear_to_srgb_u8_slice(token, &input, &mut output);

        for (&got, &inp) in output.iter().zip(input.iter()) {
            let expected = crate::scalar::linear_to_srgb_u8(inp);
            assert!((got as i32 - expected as i32).abs() <= 1);
        }
    }

    #[cfg(feature = "transfer")]
    mod tf_tests {
        use super::*;

        #[archmage::arcane]
        fn call_tf_srgb_to_linear_slice(token: X64V4Token, values: &mut [f32]) {
            tf_srgb_to_linear_slice_v4(token, values);
        }

        #[archmage::arcane]
        fn call_tf_linear_to_srgb_slice(token: X64V4Token, values: &mut [f32]) {
            tf_linear_to_srgb_slice_v4(token, values);
        }

        #[archmage::arcane]
        fn call_bt709_to_linear_slice(token: X64V4Token, values: &mut [f32]) {
            bt709_to_linear_slice_v4(token, values);
        }

        #[archmage::arcane]
        fn call_linear_to_bt709_slice(token: X64V4Token, values: &mut [f32]) {
            linear_to_bt709_slice_v4(token, values);
        }

        #[archmage::arcane]
        fn call_pq_to_linear_slice(token: X64V4Token, values: &mut [f32]) {
            pq_to_linear_slice_v4(token, values);
        }

        #[archmage::arcane]
        fn call_linear_to_pq_slice(token: X64V4Token, values: &mut [f32]) {
            linear_to_pq_slice_v4(token, values);
        }

        #[archmage::arcane]
        fn call_hlg_to_linear_slice(token: X64V4Token, values: &mut [f32]) {
            hlg_to_linear_slice_v4(token, values);
        }

        #[archmage::arcane]
        fn call_linear_to_hlg_slice(token: X64V4Token, values: &mut [f32]) {
            linear_to_hlg_slice_v4(token, values);
        }

        #[test]
        fn test_tf_srgb_slice_roundtrip() {
            let Some(token) = get_token() else { return };

            let mut values: Vec<f32> = (0..19).map(|i| i as f32 / 18.0).collect();
            let original = values.clone();

            call_tf_srgb_to_linear_slice(token, &mut values);
            call_tf_linear_to_srgb_slice(token, &mut values);

            for (i, (&orig, &conv)) in original.iter().zip(values.iter()).enumerate() {
                assert!((orig - conv).abs() < 1e-4, "tf sRGB x16 roundtrip at {}", i);
            }
        }

        #[test]
        fn test_bt709_slice_roundtrip() {
            let Some(token) = get_token() else { return };

            let mut values: Vec<f32> = (0..19).map(|i| i as f32 / 18.0).collect();
            let original = values.clone();

            call_bt709_to_linear_slice(token, &mut values);
            call_linear_to_bt709_slice(token, &mut values);

            for (i, (&orig, &conv)) in original.iter().zip(values.iter()).enumerate() {
                assert!((orig - conv).abs() < 1e-3, "BT.709 x16 roundtrip at {}", i);
            }
        }

        #[test]
        fn test_pq_slice_roundtrip() {
            let Some(token) = get_token() else { return };

            let mut values: Vec<f32> = (1..19).map(|i| i as f32 / 18.0).collect();
            let original = values.clone();

            call_pq_to_linear_slice(token, &mut values);
            call_linear_to_pq_slice(token, &mut values);

            for (i, (&orig, &conv)) in original.iter().zip(values.iter()).enumerate() {
                assert!((orig - conv).abs() < 1e-3, "PQ x16 roundtrip at {}", i);
            }
        }

        #[test]
        fn test_hlg_slice_roundtrip() {
            let Some(token) = get_token() else { return };

            let mut values: Vec<f32> = (0..19).map(|i| i as f32 / 18.0).collect();
            let original = values.clone();

            call_hlg_to_linear_slice(token, &mut values);
            call_linear_to_hlg_slice(token, &mut values);

            for (i, (&orig, &conv)) in original.iter().zip(values.iter()).enumerate() {
                assert!((orig - conv).abs() < 1e-3, "HLG x16 roundtrip at {}", i);
            }
        }
    }
}
