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
#[cfg(target_arch = "x86_64")]
use archmage::{X64V3Token, arcane, rite};
use archmage::{ScalarToken, incant};

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
    use crate::rational_poly::{S2L_P, S2L_Q};

    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let srgb = srgb.max(zero).min(one);

    let linear_result = srgb * mt_f32x8::splat(token, 1.0 / 12.92);

    let x = srgb;
    let yp = mt_f32x8::splat(token, S2L_P[4]).mul_add(x, mt_f32x8::splat(token, S2L_P[3]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, S2L_P[2]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, S2L_P[1]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, S2L_P[0]));

    let yq = mt_f32x8::splat(token, S2L_Q[4]).mul_add(x, mt_f32x8::splat(token, S2L_Q[3]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, S2L_Q[2]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, S2L_Q[1]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, S2L_Q[0]));

    let power_result = yp / yq;

    let mask = srgb.simd_lt(mt_f32x8::splat(token, 0.04045));
    mt_f32x8::blend(mask, linear_result, power_result)
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn linear_to_srgb_mt(token: X64V3Token, linear: mt_f32x8) -> mt_f32x8 {
    use crate::rational_poly::{L2S_P, L2S_Q};

    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let linear = linear.max(zero).min(one);

    let linear_result = linear * mt_f32x8::splat(token, 12.92);

    let x = linear.sqrt();
    let yp = mt_f32x8::splat(token, L2S_P[4]).mul_add(x, mt_f32x8::splat(token, L2S_P[3]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, L2S_P[2]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, L2S_P[1]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, L2S_P[0]));

    let yq = mt_f32x8::splat(token, L2S_Q[4]).mul_add(x, mt_f32x8::splat(token, L2S_Q[3]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, L2S_Q[2]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, L2S_Q[1]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, L2S_Q[0]));

    let power_result = yp / yq;

    let mask = linear.simd_lt(mt_f32x8::splat(token, 0.003_130_8));
    mt_f32x8::blend(mask, linear_result, power_result)
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
    use crate::rational_poly::{S2L_P, S2L_Q};

    let zero = mt_f32x16::zero(token);
    let one = mt_f32x16::splat(token, 1.0);
    let srgb = srgb.max(zero).min(one);

    let linear_result = srgb * mt_f32x16::splat(token, 1.0 / 12.92);

    let x = srgb;
    let yp = mt_f32x16::splat(token, S2L_P[4]).mul_add(x, mt_f32x16::splat(token, S2L_P[3]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, S2L_P[2]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, S2L_P[1]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, S2L_P[0]));

    let yq = mt_f32x16::splat(token, S2L_Q[4]).mul_add(x, mt_f32x16::splat(token, S2L_Q[3]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, S2L_Q[2]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, S2L_Q[1]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, S2L_Q[0]));

    let power_result = yp / yq;

    let mask = srgb.simd_lt(mt_f32x16::splat(token, 0.04045));
    mt_f32x16::blend(mask, linear_result, power_result)
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[rite]
fn linear_to_srgb_mt_x16(token: X64V4Token, linear: mt_f32x16) -> mt_f32x16 {
    use crate::rational_poly::{L2S_P, L2S_Q};

    let zero = mt_f32x16::zero(token);
    let one = mt_f32x16::splat(token, 1.0);
    let linear = linear.max(zero).min(one);

    let linear_result = linear * mt_f32x16::splat(token, 12.92);

    let x = linear.sqrt();
    let yp = mt_f32x16::splat(token, L2S_P[4]).mul_add(x, mt_f32x16::splat(token, L2S_P[3]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, L2S_P[2]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, L2S_P[1]));
    let yp = yp.mul_add(x, mt_f32x16::splat(token, L2S_P[0]));

    let yq = mt_f32x16::splat(token, L2S_Q[4]).mul_add(x, mt_f32x16::splat(token, L2S_Q[3]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, L2S_Q[2]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, L2S_Q[1]));
    let yq = yq.mul_add(x, mt_f32x16::splat(token, L2S_Q[0]));

    let power_result = yp / yq;

    let mask = linear.simd_lt(mt_f32x16::splat(token, 0.003_130_8));
    mt_f32x16::blend(mask, linear_result, power_result)
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
// Slice Functions with dispatch
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn srgb_to_linear_slice_tier_v4(token: X64V4Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<16>();
    for chunk in chunks {
        let v = mt_f32x16::from_array(token, *chunk);
        let result = srgb_to_linear_mt_x16(token, v);
        *chunk = result.to_array();
    }
    for v in remainder {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn srgb_to_linear_slice_tier_v3(token: X64V3Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();
    for chunk in chunks {
        let v = mt_f32x8::from_array(token, *chunk);
        let result = srgb_to_linear_mt(token, v);
        *chunk = result.to_array();
    }
    for v in remainder {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

fn srgb_to_linear_slice_tier_scalar(_token: ScalarToken, values: &mut [f32]) {
    for v in values.iter_mut() {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

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

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn linear_to_srgb_slice_tier_v4(token: X64V4Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<16>();
    for chunk in chunks {
        let v = mt_f32x16::from_array(token, *chunk);
        let result = linear_to_srgb_mt_x16(token, v);
        *chunk = result.to_array();
    }
    for v in remainder {
        *v = crate::scalar::linear_to_srgb(*v);
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn linear_to_srgb_slice_tier_v3(token: X64V3Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();
    for chunk in chunks {
        let v = mt_f32x8::from_array(token, *chunk);
        let result = linear_to_srgb_mt(token, v);
        *chunk = result.to_array();
    }
    for v in remainder {
        *v = crate::scalar::linear_to_srgb(*v);
    }
}

fn linear_to_srgb_slice_tier_scalar(_token: ScalarToken, values: &mut [f32]) {
    for v in values.iter_mut() {
        *v = crate::scalar::linear_to_srgb(*v);
    }
}

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

        for i in 0..256 {
            let expected = crate::scalar::srgb_to_linear(i as f32 / 255.0);
            assert!(
                (output[i] - expected).abs() < 1e-4,
                "u8_to_linear mismatch at {}: got {}, expected {}",
                i,
                output[i],
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
        for i in 0..256 {
            let diff = (output[i] as i32 - i as i32).unsigned_abs();
            assert!(
                diff <= 1,
                "linear_to_srgb_u8 at {}: got {}, expected {}",
                i,
                output[i],
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
}
