//! SIMD-accelerated sRGB ↔ linear conversion.
//!
//! This module provides high-performance conversion functions using AVX2/FMA SIMD
//! instructions via archmage/magetypes with runtime CPU feature detection.
//!
//! # API Overview
//!
//! ## Slice Functions (process entire slices)
//! - `srgb_to_linear_slice` - &mut \[f32\] sRGB → linear in-place
//! - `linear_to_srgb_slice` - &mut \[f32\] linear → sRGB in-place
//! - `srgb_u8_to_linear_slice` - &\[u8\] sRGB → &mut \[f32\] linear
//! - `linear_to_srgb_u8_slice` - &\[f32\] linear → &mut \[u8\] sRGB
//!
//! ## Single-value LUT Functions
//! - `srgb_u8_to_linear` - u8 → f32 via lookup table

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use archmage::X64V4Token;
use archmage::{ScalarToken, incant};
#[cfg(target_arch = "x86_64")]
use archmage::{X64V3Token, arcane, rite};

// Alias magetypes SIMD types to avoid name clash
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8 as mt_f32x8;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use magetypes::simd::v4::f32x16 as mt_f32x16;

// ============================================================================
// magetypes #[rite] helpers (x86-64 only) — real AVX2+FMA SIMD
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[rite]
fn srgb_to_linear_mt(token: X64V3Token, srgb: mt_f32x8) -> mt_f32x8 {
    use crate::rational_poly::{S2L_P, S2L_Q, SRGB_THRESHOLD};

    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let clamped = srgb.max(zero).min(one);

    let linear_result = clamped * mt_f32x8::splat(token, 1.0 / 12.92);

    let x = clamped;
    let yp = mt_f32x8::splat(token, S2L_P[4]).mul_add(x, mt_f32x8::splat(token, S2L_P[3]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, S2L_P[2]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, S2L_P[1]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, S2L_P[0]));

    let yq = mt_f32x8::splat(token, S2L_Q[4]).mul_add(x, mt_f32x8::splat(token, S2L_Q[3]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, S2L_Q[2]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, S2L_Q[1]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, S2L_Q[0]));

    let power_result = yp / yq;

    let mask = clamped.simd_lt(mt_f32x8::splat(token, SRGB_THRESHOLD));
    let result = mt_f32x8::blend(mask, linear_result, power_result);
    // Force exact 1.0 for inputs >= 1.0 (polynomial may undershoot)
    let ge_one = srgb.simd_ge(one);
    mt_f32x8::blend(ge_one, one, result)
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn linear_to_srgb_mt(token: X64V3Token, linear: mt_f32x8) -> mt_f32x8 {
    use crate::rational_poly::{L2S_P, L2S_Q, LINEAR_THRESHOLD};

    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let clamped = linear.max(zero).min(one);

    let linear_result = clamped * mt_f32x8::splat(token, 12.92);

    let x = clamped.sqrt();
    let yp = mt_f32x8::splat(token, L2S_P[4]).mul_add(x, mt_f32x8::splat(token, L2S_P[3]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, L2S_P[2]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, L2S_P[1]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, L2S_P[0]));

    let yq = mt_f32x8::splat(token, L2S_Q[4]).mul_add(x, mt_f32x8::splat(token, L2S_Q[3]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, L2S_Q[2]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, L2S_Q[1]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, L2S_Q[0]));

    let power_result = yp / yq;

    let mask = clamped.simd_lt(mt_f32x8::splat(token, LINEAR_THRESHOLD));
    let result = mt_f32x8::blend(mask, linear_result, power_result);
    let ge_one = linear.simd_ge(one);
    mt_f32x8::blend(ge_one, one, result)
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn gamma_to_linear_mt(token: X64V3Token, encoded: mt_f32x8, gamma: f32) -> mt_f32x8 {
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let encoded = encoded.max(zero).min(one);
    encoded.pow_midp(gamma)
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn linear_to_gamma_mt(token: X64V3Token, linear: mt_f32x8, gamma: f32) -> mt_f32x8 {
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let linear = linear.max(zero).min(one);
    linear.pow_midp(1.0 / gamma)
}

// ============================================================================
// magetypes #[rite] helpers (x86-64 V4/AVX-512) — native 512-bit SIMD
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[rite]
fn srgb_to_linear_mt_x16(token: X64V4Token, srgb: mt_f32x16) -> mt_f32x16 {
    use crate::rational_poly::{S2L_P, S2L_Q, SRGB_THRESHOLD};

    let zero = mt_f32x16::zero(token);
    let one = mt_f32x16::splat(token, 1.0);
    let clamped = srgb.max(zero).min(one);

    let linear_result = clamped * mt_f32x16::splat(token, 1.0 / 12.92);

    let x = clamped;
    let yp = mt_f32x16::splat(token, S2L_P[4]).mul_add(x, mt_f32x16::splat(token, S2L_P[3]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, S2L_P[2]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, S2L_P[1]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, S2L_P[0]));

    let yq = mt_f32x16::splat(token, S2L_Q[4]).mul_add(x, mt_f32x16::splat(token, S2L_Q[3]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, S2L_Q[2]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, S2L_Q[1]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, S2L_Q[0]));

    let power_result = yp / yq;

    let mask = clamped.simd_lt(mt_f32x16::splat(token, SRGB_THRESHOLD));
    let result = mt_f32x16::blend(mask, linear_result, power_result);
    let ge_one = srgb.simd_ge(one);
    mt_f32x16::blend(ge_one, one, result)
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[rite]
fn linear_to_srgb_mt_x16(token: X64V4Token, linear: mt_f32x16) -> mt_f32x16 {
    use crate::rational_poly::{L2S_P, L2S_Q, LINEAR_THRESHOLD};

    let zero = mt_f32x16::zero(token);
    let one = mt_f32x16::splat(token, 1.0);
    let clamped = linear.max(zero).min(one);

    let linear_result = clamped * mt_f32x16::splat(token, 12.92);

    let x = clamped.sqrt();
    let yp = mt_f32x16::splat(token, L2S_P[4]).mul_add(x, mt_f32x16::splat(token, L2S_P[3]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, L2S_P[2]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, L2S_P[1]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, L2S_P[0]));

    let yq = mt_f32x16::splat(token, L2S_Q[4]).mul_add(x, mt_f32x16::splat(token, L2S_Q[3]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, L2S_Q[2]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, L2S_Q[1]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, L2S_Q[0]));

    let power_result = yp / yq;

    let mask = clamped.simd_lt(mt_f32x16::splat(token, LINEAR_THRESHOLD));
    let result = mt_f32x16::blend(mask, linear_result, power_result);
    let ge_one = linear.simd_ge(one);
    mt_f32x16::blend(ge_one, one, result)
}

// gamma x16: pow_midp not available on f32x16, delegate to 2×x8 via token.v3()
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[rite]
fn gamma_to_linear_x16_2x8(token: X64V4Token, v: [f32; 16], gamma: f32) -> [f32; 16] {
    let t3 = token.v3();
    let lo = mt_f32x8::from_array(t3, <[f32; 8]>::try_from(&v[..8]).unwrap());
    let hi = mt_f32x8::from_array(t3, <[f32; 8]>::try_from(&v[8..]).unwrap());
    let lo = gamma_to_linear_mt(t3, lo, gamma).to_array();
    let hi = gamma_to_linear_mt(t3, hi, gamma).to_array();
    let mut out = [0.0f32; 16];
    out[..8].copy_from_slice(&lo);
    out[8..].copy_from_slice(&hi);
    out
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[rite]
fn linear_to_gamma_x16_2x8(token: X64V4Token, v: [f32; 16], gamma: f32) -> [f32; 16] {
    let t3 = token.v3();
    let lo = mt_f32x8::from_array(t3, <[f32; 8]>::try_from(&v[..8]).unwrap());
    let hi = mt_f32x8::from_array(t3, <[f32; 8]>::try_from(&v[8..]).unwrap());
    let lo = linear_to_gamma_mt(t3, lo, gamma).to_array();
    let hi = linear_to_gamma_mt(t3, hi, gamma).to_array();
    let mut out = [0.0f32; 16];
    out[..8].copy_from_slice(&lo);
    out[8..].copy_from_slice(&hi);
    out
}

// ============================================================================
// Slice tier macros — generate plain and RGBA variants from a single rite
// ============================================================================

/// Generate x86_64 AVX-512 (16-wide) slice tier functions (plain + RGBA).
macro_rules! x16_slice_tiers {
    ($plain:ident, $rgba:ident, $rite:ident, $scalar:path) => {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        #[arcane]
        fn $plain(token: X64V4Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                let v = mt_f32x16::from_array(token, *chunk);
                *chunk = $rite(token, v).to_array();
            }
            for v in remainder {
                *v = $scalar(*v);
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        #[arcane]
        fn $rgba(token: X64V4Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
                let v = mt_f32x16::from_array(token, *chunk);
                *chunk = $rite(token, v).to_array();
                [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
            }
            for pixel in remainder.chunks_exact_mut(4) {
                pixel[0] = $scalar(pixel[0]);
                pixel[1] = $scalar(pixel[1]);
                pixel[2] = $scalar(pixel[2]);
            }
        }
    };
}

/// Generate x86_64 AVX2+FMA (8-wide) slice tier functions (plain + RGBA).
macro_rules! x8_slice_tiers {
    ($plain:ident, $rgba:ident, $rite:ident, $scalar:path) => {
        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $plain(token: X64V3Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<8>();
            for chunk in chunks {
                let v = mt_f32x8::from_array(token, *chunk);
                *chunk = $rite(token, v).to_array();
            }
            for v in remainder {
                *v = $scalar(*v);
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[arcane]
        fn $rgba(token: X64V3Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<8>();
            for chunk in chunks {
                let a = [chunk[3], chunk[7]];
                let v = mt_f32x8::from_array(token, *chunk);
                *chunk = $rite(token, v).to_array();
                [chunk[3], chunk[7]] = a;
            }
            for pixel in remainder.chunks_exact_mut(4) {
                pixel[0] = $scalar(pixel[0]);
                pixel[1] = $scalar(pixel[1]);
                pixel[2] = $scalar(pixel[2]);
            }
        }
    };
}

/// Generate scalar fallback slice tier functions (plain + RGBA).
macro_rules! scalar_slice_tiers {
    ($plain:ident, $rgba:ident, $scalar:path) => {
        fn $plain(_token: ScalarToken, values: &mut [f32]) {
            for v in values.iter_mut() {
                *v = $scalar(*v);
            }
        }

        fn $rgba(_token: ScalarToken, values: &mut [f32]) {
            for pixel in values.chunks_exact_mut(4) {
                pixel[0] = $scalar(pixel[0]);
                pixel[1] = $scalar(pixel[1]);
                pixel[2] = $scalar(pixel[2]);
            }
        }
    };
}

// ============================================================================
// sRGB ↔ Linear Slice Functions (plain + RGBA, generated from macros)
// ============================================================================

x16_slice_tiers!(
    srgb_to_linear_slice_tier_v4,
    srgb_to_linear_rgba_slice_tier_v4,
    srgb_to_linear_mt_x16,
    crate::scalar::srgb_to_linear
);
x8_slice_tiers!(
    srgb_to_linear_slice_tier_v3,
    srgb_to_linear_rgba_slice_tier_v3,
    srgb_to_linear_mt,
    crate::scalar::srgb_to_linear
);
scalar_slice_tiers!(
    srgb_to_linear_slice_tier_scalar,
    srgb_to_linear_rgba_slice_tier_scalar,
    crate::scalar::srgb_to_linear
);

/// Convert sRGB f32 values to linear in-place.
///
/// Uses AVX-512 (16-wide), AVX2+FMA (8-wide), or scalar depending on CPU.
///
/// # Example
/// ```
/// use linear_srgb::default::srgb_to_linear_slice;
///
/// let mut values = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
/// srgb_to_linear_slice(&mut values);
/// ```
#[inline]
pub fn srgb_to_linear_slice(values: &mut [f32]) {
    incant!(srgb_to_linear_slice_tier(values), [v4, v3])
}

/// Convert sRGB RGBA f32 values to linear in-place, preserving alpha.
///
/// Expects interleaved RGBA data (`[R, G, B, A, R, G, B, A, ...]`).
/// Every 4th element (alpha) is left unchanged. Trailing elements that
/// don't form a complete RGBA pixel are ignored.
///
/// # Example
/// ```
/// use linear_srgb::default::srgb_to_linear_rgba_slice;
///
/// let mut rgba = vec![0.5f32, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
/// srgb_to_linear_rgba_slice(&mut rgba);
/// assert_eq!(rgba[3], 0.75); // alpha preserved
/// ```
#[inline]
pub fn srgb_to_linear_rgba_slice(values: &mut [f32]) {
    incant!(srgb_to_linear_rgba_slice_tier(values), [v4, v3])
}

x16_slice_tiers!(
    linear_to_srgb_slice_tier_v4,
    linear_to_srgb_rgba_slice_tier_v4,
    linear_to_srgb_mt_x16,
    crate::scalar::linear_to_srgb
);
x8_slice_tiers!(
    linear_to_srgb_slice_tier_v3,
    linear_to_srgb_rgba_slice_tier_v3,
    linear_to_srgb_mt,
    crate::scalar::linear_to_srgb
);
scalar_slice_tiers!(
    linear_to_srgb_slice_tier_scalar,
    linear_to_srgb_rgba_slice_tier_scalar,
    crate::scalar::linear_to_srgb
);

/// Convert linear f32 values to sRGB in-place.
///
/// Uses AVX-512 (16-wide), AVX2+FMA (8-wide), or scalar depending on CPU.
///
/// # Example
/// ```
/// use linear_srgb::default::linear_to_srgb_slice;
///
/// let mut values = vec![0.0f32, 0.1, 0.2, 0.5, 1.0];
/// linear_to_srgb_slice(&mut values);
/// ```
#[inline]
pub fn linear_to_srgb_slice(values: &mut [f32]) {
    incant!(linear_to_srgb_slice_tier(values), [v4, v3])
}

/// Convert linear RGBA f32 values to sRGB in-place, preserving alpha.
///
/// Expects interleaved RGBA data (`[R, G, B, A, R, G, B, A, ...]`).
/// Every 4th element (alpha) is left unchanged. Trailing elements that
/// don't form a complete RGBA pixel are ignored.
///
/// # Example
/// ```
/// use linear_srgb::default::linear_to_srgb_rgba_slice;
///
/// let mut rgba = vec![0.2f32, 0.2, 0.2, 0.75, 0.8, 0.8, 0.8, 1.0];
/// linear_to_srgb_rgba_slice(&mut rgba);
/// assert_eq!(rgba[3], 0.75); // alpha preserved
/// ```
#[inline]
pub fn linear_to_srgb_rgba_slice(values: &mut [f32]) {
    incant!(linear_to_srgb_rgba_slice_tier(values), [v4, v3])
}

/// Convert sRGB u8 values to linear f32.
///
/// Uses a precomputed LUT for each u8 value, processed in batches of 8.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
///
/// # Example
/// ```
/// use linear_srgb::default::srgb_u8_to_linear_slice;
///
/// let input: Vec<u8> = (0..=255).collect();
/// let mut output = vec![0.0f32; 256];
/// srgb_u8_to_linear_slice(&input, &mut output);
/// ```
#[inline]
pub fn srgb_u8_to_linear_slice(input: &[u8], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());

    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = crate::scalar::srgb_u8_to_linear_x8(*inp);
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        *out = crate::scalar::srgb_u8_to_linear(*inp);
    }
}

/// Convert linear f32 values to sRGB u8.
///
/// Uses a 4096-entry const LUT — no pow/log/exp computation.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
///
/// # Example
/// ```
/// use linear_srgb::default::linear_to_srgb_u8_slice;
///
/// let input: Vec<f32> = (0..=255).map(|i| i as f32 / 255.0).collect();
/// let mut output = vec![0u8; 256];
/// linear_to_srgb_u8_slice(&input, &mut output);
/// ```
pub fn linear_to_srgb_u8_slice(input: &[f32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());
    let lut = &crate::const_luts::LINEAR_TO_SRGB_U8;

    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let clamped = inp.clamp(0.0, 1.0);
        let idx = (clamped * 4095.0 + 0.5) as usize & 0xFFF;
        *out = lut[idx];
    }
}

// ============================================================================
// u16 Batch Functions (LUT-based)
// ============================================================================

/// Convert sRGB u16 values to linear f32 using a 65536-entry const LUT.
///
/// Pure table lookup, no math. The LUT is 256KB.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
pub fn srgb_u16_to_linear_slice(input: &[u16], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let lut = &crate::const_luts_u16::SRGB_U16_TO_LINEAR_F32;

    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = lut[*inp as usize];
    }
}

/// Convert linear f32 values to sRGB u16 using a 65537-entry const LUT.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
pub fn linear_to_srgb_u16_slice(input: &[f32], output: &mut [u16]) {
    assert_eq!(input.len(), output.len());
    let lut = &crate::const_luts_u16::LINEAR_TO_SRGB_U16_65536;

    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let clamped = inp.clamp(0.0, 1.0);
        let idx = (clamped * 65536.0 + 0.5) as usize;
        *out = lut[idx.min(65536)];
    }
}

// ============================================================================
// RGBA u8 Batch Functions (LUT-based, alpha passthrough)
// ============================================================================

/// Convert sRGB RGBA u8 values to linear f32, preserving alpha.
///
/// RGB channels are decoded via LUT. Alpha is passed through as `a / 255.0`
/// without sRGB transfer. Trailing elements that don't form a complete
/// RGBA pixel are ignored.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
///
/// # Example
/// ```
/// use linear_srgb::default::srgb_u8_to_linear_rgba_slice;
///
/// let input = [128u8, 128, 128, 255, 64, 64, 64, 128];
/// let mut output = [0.0f32; 8];
/// srgb_u8_to_linear_rgba_slice(&input, &mut output);
/// assert_eq!(output[3], 1.0);           // alpha 255 → 1.0
/// assert!((output[7] - 0.502).abs() < 0.01); // alpha 128 → ~0.502
/// ```
pub fn srgb_u8_to_linear_rgba_slice(input: &[u8], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());

    // Process in x8 batches (2 RGBA pixels), fix alpha after LUT
    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = crate::scalar::srgb_u8_to_linear_x8(*inp);
        out[3] = inp[3] as f32 / 255.0;
        out[7] = inp[7] as f32 / 255.0;
    }

    // Remainder: at most one RGBA pixel (4 elements)
    let in_rem_pixels = in_remainder.chunks_exact(4);
    let out_rem_pixels = out_remainder.chunks_exact_mut(4);
    for (inp, out) in in_rem_pixels.zip(out_rem_pixels) {
        out[0] = crate::scalar::srgb_u8_to_linear(inp[0]);
        out[1] = crate::scalar::srgb_u8_to_linear(inp[1]);
        out[2] = crate::scalar::srgb_u8_to_linear(inp[2]);
        out[3] = inp[3] as f32 / 255.0;
    }
}

/// Convert linear RGBA f32 values to sRGB u8, preserving alpha.
///
/// RGB channels are encoded via 4096-entry const LUT. Alpha is passed through
/// as `(a * 255 + 0.5) as u8` without sRGB transfer. Trailing elements that
/// don't form a complete RGBA pixel are ignored.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
///
/// # Example
/// ```
/// use linear_srgb::default::linear_to_srgb_u8_rgba_slice;
///
/// let input = [0.2158f32, 0.2158, 0.2158, 1.0, 0.05, 0.05, 0.05, 0.5];
/// let mut output = [0u8; 8];
/// linear_to_srgb_u8_rgba_slice(&input, &mut output);
/// assert_eq!(output[3], 255);  // alpha 1.0 → 255
/// assert_eq!(output[7], 128);  // alpha 0.5 → 128
/// ```
pub fn linear_to_srgb_u8_rgba_slice(input: &[f32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());
    let lut = &crate::const_luts::LINEAR_TO_SRGB_U8;

    let in_pixels = input.chunks_exact(4);
    let out_pixels = output.chunks_exact_mut(4);
    for (inp, out) in in_pixels.zip(out_pixels) {
        // RGB: LUT encode
        for i in 0..3 {
            let clamped = inp[i].clamp(0.0, 1.0);
            let idx = (clamped * 4095.0 + 0.5) as usize & 0xFFF;
            out[i] = lut[idx];
        }
        // Alpha: linear passthrough
        out[3] = (inp[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
}

// ============================================================================
// RGBA u16 Batch Functions (LUT-based, alpha passthrough)
// ============================================================================

/// Convert sRGB RGBA u16 values to linear f32, preserving alpha.
///
/// RGB channels are decoded via 65536-entry const LUT. Alpha is passed
/// through as `a / 65535.0` without sRGB transfer. Trailing elements
/// that don't form a complete RGBA pixel are ignored.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
pub fn srgb_u16_to_linear_rgba_slice(input: &[u16], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let lut = &crate::const_luts_u16::SRGB_U16_TO_LINEAR_F32;

    let in_pixels = input.chunks_exact(4);
    let out_pixels = output.chunks_exact_mut(4);
    for (inp, out) in in_pixels.zip(out_pixels) {
        out[0] = lut[inp[0] as usize];
        out[1] = lut[inp[1] as usize];
        out[2] = lut[inp[2] as usize];
        out[3] = inp[3] as f32 / 65535.0;
    }
}

/// Convert linear RGBA f32 values to sRGB u16, preserving alpha.
///
/// RGB channels are encoded via 65537-entry const LUT. Alpha is passed
/// through as `(a * 65535 + 0.5) as u16` without sRGB transfer. Trailing
/// elements that don't form a complete RGBA pixel are ignored.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
pub fn linear_to_srgb_u16_rgba_slice(input: &[f32], output: &mut [u16]) {
    assert_eq!(input.len(), output.len());
    let lut = &crate::const_luts_u16::LINEAR_TO_SRGB_U16_65536;

    let in_pixels = input.chunks_exact(4);
    let out_pixels = output.chunks_exact_mut(4);
    for (inp, out) in in_pixels.zip(out_pixels) {
        for i in 0..3 {
            let clamped = inp[i].clamp(0.0, 1.0);
            let idx = (clamped * 65536.0 + 0.5) as usize;
            out[i] = lut[idx.min(65536)];
        }
        out[3] = (inp[3].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
}

// ============================================================================
// Custom Gamma Slice Functions
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn gamma_to_linear_slice_tier_v4(token: X64V4Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<16>();
    for chunk in chunks {
        *chunk = gamma_to_linear_x16_2x8(token, *chunk, gamma);
    }
    for v in remainder {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn gamma_to_linear_slice_tier_v3(token: X64V3Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();
    for chunk in chunks {
        let v = mt_f32x8::from_array(token, *chunk);
        let result = gamma_to_linear_mt(token, v, gamma);
        *chunk = result.to_array();
    }
    for v in remainder {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

fn gamma_to_linear_slice_tier_scalar(_token: ScalarToken, values: &mut [f32], gamma: f32) {
    for v in values.iter_mut() {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

/// Convert gamma-encoded f32 values to linear in-place using a custom gamma.
///
/// Uses AVX-512 (16-wide), AVX2+FMA (8-wide), or scalar depending on CPU.
///
/// # Example
/// ```
/// use linear_srgb::default::gamma_to_linear_slice;
///
/// let mut values = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
/// gamma_to_linear_slice(&mut values, 2.2);
/// ```
#[inline]
pub fn gamma_to_linear_slice(values: &mut [f32], gamma: f32) {
    incant!(gamma_to_linear_slice_tier(values, gamma), [v4, v3])
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn linear_to_gamma_slice_tier_v4(token: X64V4Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<16>();
    for chunk in chunks {
        *chunk = linear_to_gamma_x16_2x8(token, *chunk, gamma);
    }
    for v in remainder {
        *v = crate::scalar::linear_to_gamma(*v, gamma);
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn linear_to_gamma_slice_tier_v3(token: X64V3Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();
    for chunk in chunks {
        let v = mt_f32x8::from_array(token, *chunk);
        let result = linear_to_gamma_mt(token, v, gamma);
        *chunk = result.to_array();
    }
    for v in remainder {
        *v = crate::scalar::linear_to_gamma(*v, gamma);
    }
}

fn linear_to_gamma_slice_tier_scalar(_token: ScalarToken, values: &mut [f32], gamma: f32) {
    for v in values.iter_mut() {
        *v = crate::scalar::linear_to_gamma(*v, gamma);
    }
}

/// Convert linear f32 values to gamma-encoded in-place using a custom gamma.
///
/// Uses AVX-512 (16-wide), AVX2+FMA (8-wide), or scalar depending on CPU.
///
/// # Example
/// ```
/// use linear_srgb::default::linear_to_gamma_slice;
///
/// let mut values = vec![0.0f32, 0.1, 0.2, 0.5, 1.0];
/// linear_to_gamma_slice(&mut values, 2.2);
/// ```
#[inline]
pub fn linear_to_gamma_slice(values: &mut [f32], gamma: f32) {
    incant!(linear_to_gamma_slice_tier(values, gamma), [v4, v3])
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    #[test]
    fn test_srgb_u8_to_linear_x8() {
        let input = [0u8, 64, 128, 192, 255, 32, 96, 160];
        let result = crate::scalar::srgb_u8_to_linear_x8(input);

        for (i, (&r, &inp)) in result.iter().zip(input.iter()).enumerate() {
            let expected = crate::scalar::srgb_to_linear(inp as f32 / 255.0);
            assert!(
                (r - expected).abs() < 1e-4,
                "srgb_u8_to_linear_x8 mismatch at {}: got {}, expected {}",
                i,
                r,
                expected
            );
        }
    }

    #[test]
    fn test_slice_roundtrip() {
        let mut values: Vec<f32> = (0..=10).map(|i| i as f32 / 10.0).collect();
        let original = values.clone();

        srgb_to_linear_slice(&mut values);
        linear_to_srgb_slice(&mut values);

        for (i, (orig, conv)) in original.iter().zip(values.iter()).enumerate() {
            assert!(
                (orig - conv).abs() < 1e-5,
                "Slice roundtrip failed at {}: {} -> {}",
                i,
                orig,
                conv
            );
        }
    }

    #[test]
    fn test_srgb_u8_to_linear_slice_basic() {
        let input: Vec<u8> = (0..=255).collect();
        let mut output = vec![0.0f32; 256];
        srgb_u8_to_linear_slice(&input, &mut output);

        for (i, &val) in output.iter().enumerate() {
            let expected = crate::scalar::srgb_to_linear(i as f32 / 255.0);
            assert!(
                (val - expected).abs() < 1e-4,
                "u8_to_linear mismatch at {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }
    }

    #[test]
    fn test_linear_to_srgb_u8_slice_basic() {
        let input: Vec<f32> = (0..=255).map(|i| i as f32 / 255.0).collect();
        let mut output = vec![0u8; 256];
        let mut direct = vec![0.0f32; 256];

        // Convert input to linear first
        for (i, &srgb) in input.iter().enumerate() {
            direct[i] = crate::scalar::srgb_to_linear(srgb);
        }

        linear_to_srgb_u8_slice(&direct, &mut output);

        // Should roundtrip within 1 level
        for (i, &val) in output.iter().enumerate() {
            let diff = (val as i32 - i as i32).unsigned_abs();
            assert!(
                diff <= 1,
                "linear_to_srgb_u8 at {}: got {}, expected {}",
                i,
                val,
                i
            );
        }
    }

    #[test]
    fn test_gamma_slice_roundtrip() {
        let mut values: Vec<f32> = (1..=100).map(|i| i as f32 / 100.0).collect();
        let original = values.clone();

        gamma_to_linear_slice(&mut values, 2.2);
        linear_to_gamma_slice(&mut values, 2.2);

        for (i, (&orig, &conv)) in original.iter().zip(values.iter()).enumerate() {
            assert!(
                (orig - conv).abs() < 1e-3,
                "Gamma roundtrip failed at {}: {} -> {}",
                i,
                orig,
                conv
            );
        }
    }

    // Regression tests for GitHub issue #1:
    // slice functions convert ALL elements including alpha channels.
    // These tests document the current (incorrect) behavior so any fix
    // can be verified against them.

    #[test]
    fn issue_1_srgb_to_linear_slice_modifies_alpha() {
        // RGBA data: 4 pixels, alpha should stay unchanged
        let mut rgba = vec![
            0.5, 0.5, 0.5, 1.0, // pixel 0: mid-gray, full alpha
            0.2, 0.4, 0.8, 0.5, // pixel 1: color, half alpha
            0.0, 0.0, 0.0, 0.0, // pixel 2: transparent black
            1.0, 1.0, 1.0, 0.75, // pixel 3: white, 75% alpha
        ];
        let alphas_before: Vec<f32> = rgba.iter().skip(3).step_by(4).copied().collect();

        srgb_to_linear_slice(&mut rgba);

        let alphas_after: Vec<f32> = rgba.iter().skip(3).step_by(4).copied().collect();

        // Current behavior: alpha IS modified (this is the bug).
        // Alpha 0.0 and 1.0 are fixed points of srgb_to_linear, so they survive.
        // But 0.5 and 0.75 will be changed.
        assert_eq!(
            alphas_before[0], alphas_after[0],
            "alpha=1.0 is a fixed point"
        );
        assert_eq!(
            alphas_before[2], alphas_after[2],
            "alpha=0.0 is a fixed point"
        );
        // These SHOULD be equal but aren't — documenting the bug:
        assert_ne!(
            alphas_before[1], alphas_after[1],
            "BUG(#1): alpha=0.5 is incorrectly converted by srgb_to_linear_slice"
        );
        assert_ne!(
            alphas_before[3], alphas_after[3],
            "BUG(#1): alpha=0.75 is incorrectly converted by srgb_to_linear_slice"
        );
    }

    #[test]
    fn issue_1_linear_to_srgb_slice_modifies_alpha() {
        // Linear RGBA data
        let mut rgba = vec![
            0.2, 0.2, 0.2, 1.0, // pixel 0: full alpha
            0.1, 0.3, 0.5, 0.5, // pixel 1: half alpha
            0.0, 0.0, 0.0, 0.0, // pixel 2: transparent
            0.8, 0.8, 0.8, 0.25, // pixel 3: 25% alpha
        ];
        let alphas_before: Vec<f32> = rgba.iter().skip(3).step_by(4).copied().collect();

        linear_to_srgb_slice(&mut rgba);

        let alphas_after: Vec<f32> = rgba.iter().skip(3).step_by(4).copied().collect();

        // 0.0 is exact fixed point, 1.0 has minor rounding in rational poly
        assert_eq!(
            alphas_before[2], alphas_after[2],
            "alpha=0.0 is a fixed point"
        );
        // Non-trivial alpha values are definitely modified:
        assert_ne!(
            alphas_before[1], alphas_after[1],
            "BUG(#1): alpha=0.5 is incorrectly converted by linear_to_srgb_slice"
        );
        assert_ne!(
            alphas_before[3], alphas_after[3],
            "BUG(#1): alpha=0.25 is incorrectly converted by linear_to_srgb_slice"
        );
    }

    #[test]
    fn issue_1_srgb_u8_to_linear_converts_all_channels() {
        // RGBA u8 data: alpha bytes are at indices 3, 7, 11, 15
        let input: Vec<u8> = vec![
            128, 128, 128, 255, // pixel 0: mid-gray, full alpha
            64, 128, 192, 128, // pixel 1: color, half alpha
        ];
        let mut output = vec![0.0f32; 8];

        srgb_u8_to_linear_slice(&input, &mut output);

        // Alpha=255 maps to 1.0 (fixed point)
        assert_eq!(output[3], 1.0, "alpha=255/255 should map to 1.0");
        // Alpha=128 should stay 128/255 ≈ 0.502 but gets sRGB-decoded instead
        let alpha_128_linear = output[7];
        let expected_passthrough = 128.0 / 255.0;
        assert!(
            (alpha_128_linear - expected_passthrough).abs() > 0.01,
            "BUG(#1): alpha=128 is sRGB-decoded ({}) instead of passed through ({})",
            alpha_128_linear,
            expected_passthrough
        );
    }

    // ====================================================================
    // RGBA variant tests — alpha MUST be preserved
    // ====================================================================

    #[test]
    fn rgba_srgb_to_linear_f32_preserves_alpha() {
        let mut rgba = vec![
            0.5, 0.5, 0.5, 1.0, // full alpha
            0.2, 0.4, 0.8, 0.5, // half alpha
            0.0, 0.0, 0.0, 0.0, // transparent
            1.0, 1.0, 1.0, 0.75, // 75% alpha
        ];
        let alphas_before: Vec<f32> = rgba.iter().skip(3).step_by(4).copied().collect();
        srgb_to_linear_rgba_slice(&mut rgba);
        let alphas_after: Vec<f32> = rgba.iter().skip(3).step_by(4).copied().collect();
        assert_eq!(alphas_before, alphas_after, "all alphas must be preserved");
        // RGB channels should have changed (not fixed points except 0.0/1.0)
        assert_ne!(rgba[0], 0.5, "RGB should be converted");
    }

    #[test]
    fn rgba_linear_to_srgb_f32_preserves_alpha() {
        let mut rgba = vec![
            0.2, 0.2, 0.2, 1.0, 0.1, 0.3, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.25,
        ];
        let alphas_before: Vec<f32> = rgba.iter().skip(3).step_by(4).copied().collect();
        linear_to_srgb_rgba_slice(&mut rgba);
        let alphas_after: Vec<f32> = rgba.iter().skip(3).step_by(4).copied().collect();
        assert_eq!(alphas_before, alphas_after, "all alphas must be preserved");
    }

    #[test]
    fn rgba_srgb_u8_to_linear_preserves_alpha() {
        let input: Vec<u8> = vec![
            128, 128, 128, 255, // full alpha
            64, 128, 192, 128, // half alpha
            0, 0, 0, 0, // transparent
            255, 255, 255, 191, // 75% alpha
        ];
        let mut output = vec![0.0f32; 16];
        srgb_u8_to_linear_rgba_slice(&input, &mut output);

        // Alpha passthrough: a / 255.0
        assert_eq!(output[3], 1.0);
        assert!((output[7] - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(output[11], 0.0);
        assert!((output[15] - 191.0 / 255.0).abs() < 1e-6);

        // RGB should be sRGB-decoded (different from a/255 passthrough)
        // Pixel 1 G channel is 128 → sRGB decoded
        let srgb_decoded_128 = crate::scalar::srgb_u8_to_linear(128);
        assert_eq!(output[5], srgb_decoded_128, "RGB should use sRGB LUT");
        assert_ne!(output[7], srgb_decoded_128, "alpha should NOT use sRGB LUT");
    }

    #[test]
    fn rgba_linear_to_srgb_u8_preserves_alpha() {
        // Linear RGBA → sRGB u8
        let linear: Vec<f32> = vec![0.2158, 0.2158, 0.2158, 1.0, 0.0, 0.0, 0.0, 0.5];
        let mut output = vec![0u8; 8];
        linear_to_srgb_u8_rgba_slice(&linear, &mut output);

        // Alpha: (1.0 * 255 + 0.5) as u8 = 255
        assert_eq!(output[3], 255);
        // Alpha: (0.5 * 255 + 0.5) as u8 = 128
        assert_eq!(output[7], 128);
        // RGB ~128 for 0.2158 linear
        assert!((output[0] as i32 - 128).unsigned_abs() <= 1);
    }

    #[test]
    fn rgba_srgb_u16_to_linear_preserves_alpha() {
        let input: Vec<u16> = vec![
            32768, 32768, 32768, 65535, // full alpha
            16384, 32768, 49152, 32768, // half alpha
        ];
        let mut output = vec![0.0f32; 8];
        srgb_u16_to_linear_rgba_slice(&input, &mut output);

        assert_eq!(output[3], 1.0);
        assert!((output[7] - 32768.0 / 65535.0).abs() < 1e-6);
        // RGB should differ from linear passthrough
        assert_ne!(output[0], 32768.0 / 65535.0, "RGB must be sRGB-decoded");
    }

    #[test]
    fn rgba_linear_to_srgb_u16_preserves_alpha() {
        let linear: Vec<f32> = vec![0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5];
        let mut output = vec![0u16; 8];
        linear_to_srgb_u16_rgba_slice(&linear, &mut output);

        assert_eq!(output[3], 65535);
        // alpha 0.5: (0.5 * 65535 + 0.5) as u16 = 32768
        assert_eq!(output[7], 32768);
    }

    #[test]
    fn rgba_f32_roundtrip_preserves_all() {
        let original = vec![
            0.5f32, 0.3, 0.8, 0.42, 0.1, 0.9, 0.0, 1.0, 1.0, 0.0, 0.5, 0.0,
        ];
        let mut rgba = original.clone();
        srgb_to_linear_rgba_slice(&mut rgba);
        linear_to_srgb_rgba_slice(&mut rgba);

        for (i, (&orig, &conv)) in original.iter().zip(rgba.iter()).enumerate() {
            if i % 4 == 3 {
                assert_eq!(orig, conv, "alpha at pixel {} must be exact", i / 4);
            } else {
                assert!(
                    (orig - conv).abs() < 1e-5,
                    "RGB roundtrip at {}: {} -> {}",
                    i,
                    orig,
                    conv
                );
            }
        }
    }

    // ====================================================================
    // SIMD boundary tests — exercise v4 (16-wide), v3 (8-wide), scalar
    // remainder paths with various pixel counts
    // ====================================================================

    fn make_rgba_srgb(num_pixels: usize) -> Vec<f32> {
        (0..num_pixels * 4)
            .map(|i| {
                if i % 4 == 3 {
                    0.3 + (i as f32 / 400.0) // varying alpha
                } else {
                    (i % 256) as f32 / 255.0
                }
            })
            .collect()
    }

    #[test]
    fn rgba_f32_s2l_various_pixel_counts() {
        // 1 pixel: pure remainder for both v3 and v4
        // 2 pixels: one v3 chunk, remainder for v4
        // 3 pixels: one v3 chunk + remainder, remainder for v4
        // 4 pixels: one v4 chunk, two v3 chunks
        // 5 pixels: one v4 chunk + remainder
        // 9 pixels: multiple v3 chunks + remainder
        // 17 pixels: one v4 chunk + v3 chunk + remainder
        for num_pixels in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let data = make_rgba_srgb(num_pixels);
            let alphas_before: Vec<f32> = data.iter().skip(3).step_by(4).copied().collect();
            let mut rgba = data.clone();
            srgb_to_linear_rgba_slice(&mut rgba);
            let alphas_after: Vec<f32> = rgba.iter().skip(3).step_by(4).copied().collect();
            assert_eq!(
                alphas_before, alphas_after,
                "alpha mismatch at {num_pixels} pixels"
            );
            // Verify RGB channels actually changed
            for px in 0..num_pixels {
                let srgb_r = data[px * 4];
                let linear_r = rgba[px * 4];
                if srgb_r > 0.04045 && srgb_r < 1.0 {
                    assert_ne!(
                        srgb_r, linear_r,
                        "RGB should change at pixel {px}/{num_pixels}"
                    );
                }
            }
        }
    }

    #[test]
    fn rgba_f32_l2s_various_pixel_counts() {
        for num_pixels in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let data = make_rgba_srgb(num_pixels);
            // Pretend data is linear (values in [0,1] are valid linear)
            let alphas_before: Vec<f32> = data.iter().skip(3).step_by(4).copied().collect();
            let mut rgba = data.clone();
            linear_to_srgb_rgba_slice(&mut rgba);
            let alphas_after: Vec<f32> = rgba.iter().skip(3).step_by(4).copied().collect();
            assert_eq!(
                alphas_before, alphas_after,
                "alpha mismatch at {num_pixels} pixels"
            );
        }
    }

    // ====================================================================
    // RGBA consistency — RGB channels must match non-RGBA output
    // ====================================================================

    #[test]
    fn rgba_rgb_channels_match_plain_s2l() {
        let srgb_values = make_rgba_srgb(20);

        // Plain slice: converts everything including alpha
        let mut plain = srgb_values.clone();
        srgb_to_linear_slice(&mut plain);

        // RGBA slice: preserves alpha
        let mut rgba = srgb_values.clone();
        srgb_to_linear_rgba_slice(&mut rgba);

        // RGB channels must be identical
        for px in 0..20 {
            for ch in 0..3 {
                let idx = px * 4 + ch;
                assert_eq!(
                    plain[idx], rgba[idx],
                    "RGB mismatch at pixel {px} channel {ch}"
                );
            }
        }
    }

    #[test]
    fn rgba_rgb_channels_match_plain_l2s() {
        let linear_values = make_rgba_srgb(20);

        let mut plain = linear_values.clone();
        linear_to_srgb_slice(&mut plain);

        let mut rgba = linear_values.clone();
        linear_to_srgb_rgba_slice(&mut rgba);

        for px in 0..20 {
            for ch in 0..3 {
                let idx = px * 4 + ch;
                assert_eq!(
                    plain[idx], rgba[idx],
                    "RGB mismatch at pixel {px} channel {ch}"
                );
            }
        }
    }

    // ====================================================================
    // Empty and edge-case inputs
    // ====================================================================

    #[test]
    fn rgba_f32_empty_slice() {
        let mut empty: Vec<f32> = vec![];
        srgb_to_linear_rgba_slice(&mut empty);
        linear_to_srgb_rgba_slice(&mut empty);
        assert!(empty.is_empty());
    }

    #[test]
    fn rgba_u8_empty_slice() {
        let input: Vec<u8> = vec![];
        let mut output: Vec<f32> = vec![];
        srgb_u8_to_linear_rgba_slice(&input, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn rgba_f32_trailing_elements_ignored() {
        // 5 floats = 1 pixel + 1 trailing element
        let mut data = vec![0.5, 0.5, 0.5, 0.42, 0.99];
        let trailing_before = data[4];
        srgb_to_linear_rgba_slice(&mut data);
        // Alpha preserved
        assert_eq!(data[3], 0.42);
        // Trailing element unchanged (not part of a complete pixel)
        assert_eq!(data[4], trailing_before);
    }

    #[test]
    fn rgba_u8_trailing_elements_ignored() {
        // 6 u8s = 1 pixel + 2 trailing
        let input = vec![128u8, 128, 128, 200, 99, 99];
        let mut output = vec![0.0f32; 6];
        srgb_u8_to_linear_rgba_slice(&input, &mut output);
        // First pixel processed
        assert_eq!(output[3], 200.0 / 255.0);
        // Trailing elements untouched
        assert_eq!(output[4], 0.0);
        assert_eq!(output[5], 0.0);
    }

    // ====================================================================
    // u8 RGBA batch boundary (x8 batch + remainder)
    // ====================================================================

    #[test]
    fn rgba_u8_to_linear_batch_boundaries() {
        // Test pixel counts that exercise x8 batch (2 pixels) + remainder (1 pixel)
        for num_pixels in [1, 2, 3, 4, 5, 8, 9, 16, 17] {
            let input: Vec<u8> = (0..num_pixels * 4)
                .map(|i| {
                    if i % 4 == 3 {
                        ((i / 4) * 30 + 10) as u8 // varying alpha
                    } else {
                        128u8
                    }
                })
                .collect();
            let mut output = vec![0.0f32; num_pixels * 4];
            srgb_u8_to_linear_rgba_slice(&input, &mut output);

            for px in 0..num_pixels {
                let alpha_in = input[px * 4 + 3];
                let alpha_out = output[px * 4 + 3];
                let expected = alpha_in as f32 / 255.0;
                assert!(
                    (alpha_out - expected).abs() < 1e-6,
                    "alpha mismatch at pixel {px}/{num_pixels}: got {alpha_out}, expected {expected}"
                );
                // RGB should be sRGB-decoded 128
                let rgb_out = output[px * 4];
                let expected_rgb = crate::scalar::srgb_u8_to_linear(128);
                assert_eq!(rgb_out, expected_rgb, "RGB mismatch at pixel {px}");
            }
        }
    }

    // ====================================================================
    // u8/u16 RGBA roundtrips
    // ====================================================================

    #[test]
    fn rgba_u8_roundtrip() {
        // u8 sRGB → f32 linear → u8 sRGB, alpha must survive exactly
        let input: Vec<u8> = vec![
            0, 0, 0, 0, // black transparent
            128, 128, 128, 128, // mid-gray, half alpha
            255, 255, 255, 255, // white opaque
            64, 192, 32, 200, // colorful
        ];
        let mut linear = vec![0.0f32; 16];
        srgb_u8_to_linear_rgba_slice(&input, &mut linear);

        let mut output = vec![0u8; 16];
        linear_to_srgb_u8_rgba_slice(&linear, &mut output);

        for px in 0..4 {
            // Alpha must roundtrip exactly
            assert_eq!(
                input[px * 4 + 3],
                output[px * 4 + 3],
                "alpha roundtrip failed at pixel {px}"
            );
            // RGB within 1 level
            for ch in 0..3 {
                let diff = (input[px * 4 + ch] as i32 - output[px * 4 + ch] as i32).unsigned_abs();
                assert!(
                    diff <= 1,
                    "RGB roundtrip at pixel {px} ch {ch}: {} -> {}",
                    input[px * 4 + ch],
                    output[px * 4 + ch]
                );
            }
        }
    }

    #[test]
    fn rgba_u16_roundtrip() {
        let input: Vec<u16> = vec![
            0, 0, 0, 0, 32768, 32768, 32768, 32768, 65535, 65535, 65535, 65535, 16384, 49152, 8192,
            40000,
        ];
        let mut linear = vec![0.0f32; 16];
        srgb_u16_to_linear_rgba_slice(&input, &mut linear);

        let mut output = vec![0u16; 16];
        linear_to_srgb_u16_rgba_slice(&linear, &mut output);

        for px in 0..4 {
            assert_eq!(
                input[px * 4 + 3],
                output[px * 4 + 3],
                "alpha must roundtrip exactly at pixel {px}"
            );
        }
    }

    // ====================================================================
    // Non-RGBA u16 slice roundtrip (was missing)
    // ====================================================================

    #[test]
    fn test_u16_slice_roundtrip() {
        // Full range u16 roundtrip: sRGB u16 → linear f32 → sRGB u16
        // The decode LUT has 65536 entries, encode LUT has 65537. Both pass through
        // f32 which has ~24 bits mantissa — at low sRGB values the linear values
        // are tiny and LUT index quantization causes up to ~10 levels of error.
        let input: Vec<u16> = (0..=255).map(|i| (i * 257) as u16).collect();
        let mut linear = vec![0.0f32; 256];
        srgb_u16_to_linear_slice(&input, &mut linear);

        let mut output = vec![0u16; 256];
        linear_to_srgb_u16_slice(&linear, &mut output);

        let mut max_diff = 0u32;
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            let diff = (inp as i32 - out as i32).unsigned_abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff <= 10,
                "u16 roundtrip at {i}: {inp} -> {out} (diff {diff})"
            );
        }
        // Verify high values roundtrip well (most error is at low sRGB values)
        assert_eq!(output[255], 65535, "u16 max must roundtrip exactly");
    }

    // ====================================================================
    // Threshold boundary — values near the sRGB linear/curve threshold
    // ====================================================================

    #[test]
    fn rgba_f32_near_threshold() {
        // sRGB threshold: 0.04045 (sRGB→linear), 0.0031308 (linear→sRGB)
        let mut rgba = vec![
            0.04044, 0.04045, 0.04046, 0.77, // pixel 0: straddles s2l threshold
            0.003130, 0.0031308, 0.003132, 0.33, // pixel 1: straddles l2s threshold
        ];
        let alphas_before = [rgba[3], rgba[7]];

        srgb_to_linear_rgba_slice(&mut rgba);
        assert_eq!(rgba[3], alphas_before[0]);
        assert_eq!(rgba[7], alphas_before[1]);

        // Values should be monotonic across threshold
        assert!(rgba[0] < rgba[1], "s2l should be monotonic");
        assert!(rgba[1] < rgba[2], "s2l should be monotonic");

        linear_to_srgb_rgba_slice(&mut rgba);
        assert_eq!(rgba[3], alphas_before[0]);
        assert_eq!(rgba[7], alphas_before[1]);
    }

    // ====================================================================
    // Boundary values — 0.0 and 1.0 exact
    // ====================================================================

    #[test]
    fn rgba_f32_boundary_values() {
        let mut rgba = vec![
            0.0, 0.0, 0.0, 0.0, // all zero
            1.0, 1.0, 1.0, 1.0, // all one
        ];
        srgb_to_linear_rgba_slice(&mut rgba);
        assert_eq!(rgba[0], 0.0, "srgb_to_linear(0.0) must be 0.0");
        assert_eq!(rgba[3], 0.0, "alpha 0.0 preserved");
        // 1.0 may have tiny rounding in rational poly, but alpha must be exact
        assert_eq!(rgba[7], 1.0, "alpha 1.0 preserved");

        let mut rgba2 = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        linear_to_srgb_rgba_slice(&mut rgba2);
        assert_eq!(rgba2[0], 0.0, "linear_to_srgb(0.0) must be 0.0");
        assert_eq!(rgba2[3], 0.0, "alpha 0.0 preserved");
        assert_eq!(rgba2[7], 1.0, "alpha 1.0 preserved");
    }

    #[test]
    fn rgba_u8_boundary_values() {
        let input = vec![0u8, 0, 0, 0, 255, 255, 255, 255];
        let mut output = vec![0.0f32; 8];
        srgb_u8_to_linear_rgba_slice(&input, &mut output);
        assert_eq!(output[0], 0.0);
        assert_eq!(output[3], 0.0); // alpha
        assert_eq!(output[4], 1.0); // sRGB 255 → linear 1.0
        assert_eq!(output[7], 1.0); // alpha 255 → 1.0

        let linear_input = vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut u8_output = vec![0u8; 8];
        linear_to_srgb_u8_rgba_slice(&linear_input, &mut u8_output);
        assert_eq!(u8_output[0], 0);
        assert_eq!(u8_output[3], 0); // alpha
        assert_eq!(u8_output[4], 255);
        assert_eq!(u8_output[7], 255); // alpha
    }

    // ====================================================================
    // u8/u16 RGBA consistency with scalar per-channel
    // ====================================================================

    #[test]
    fn rgba_u8_rgb_matches_non_rgba() {
        let input: Vec<u8> = (0..40).map(|i| (i * 6 + 10) as u8).collect(); // 10 RGBA pixels
        let mut rgba_out = vec![0.0f32; 40];
        let mut plain_out = vec![0.0f32; 40];

        srgb_u8_to_linear_rgba_slice(&input, &mut rgba_out);
        srgb_u8_to_linear_slice(&input, &mut plain_out);

        // RGB channels must match
        for px in 0..10 {
            for ch in 0..3 {
                let idx = px * 4 + ch;
                assert_eq!(
                    rgba_out[idx], plain_out[idx],
                    "u8 RGB mismatch at pixel {px} ch {ch}"
                );
            }
            // Alpha should differ: RGBA does passthrough, plain does sRGB decode
            let alpha_val = input[px * 4 + 3];
            if alpha_val > 10 && alpha_val < 245 {
                // Not near fixed points
                assert_ne!(
                    rgba_out[px * 4 + 3],
                    plain_out[px * 4 + 3],
                    "alpha should differ between RGBA and plain at pixel {px}"
                );
            }
        }
    }
}
