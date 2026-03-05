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

#[cfg(target_arch = "x86_64")]
use archmage::{Desktop64, arcane, rite};
use archmage::{ScalarToken, incant};

// Alias magetypes f32x8 to avoid name clash
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8 as mt_f32x8;

/// Precomputed sRGB u8 → linear f32 lookup table.
/// Uses the same constants as the transfer module (C0-continuous IEC 61966-2-1).
static SRGB_U8_TO_LINEAR_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        let srgb = i as f64 / 255.0;
        let linear = if srgb <= 0.04045 {
            srgb / 12.92
        } else {
            // Use manual pow via exp(ln(x)*y) since powf isn't const
            let base = (srgb + 0.055) / 1.055;
            // Approximate pow(base, 2.4) using the identity:
            // We precompute these at compile time, so precision doesn't matter
            // for the LUT - we just need f32 precision in the final value
            // Square-and-multiply for 2.4 = 2 + 0.4
            let sq = base * base; // base^2
            // base^0.4 = (base^2)^0.2 = ((base^2)^(1/5))
            // Use Newton's method: find x where x^5 = base^2
            let target = sq; // base^2, we want target^(1/5) = base^0.4
            let mut x = 0.5f64;
            let mut iter = 0;
            while iter < 100 {
                let x4 = x * x * x * x;
                let x5 = x4 * x;
                x = x - (x5 - target) / (5.0 * x4);
                iter += 1;
            }
            sq * x // base^2 * base^0.4 = base^2.4
        };
        lut[i] = linear as f32;
        i += 1;
    }
    lut
};

#[inline]
fn get_lut() -> &'static [f32; 256] {
    &SRGB_U8_TO_LINEAR_LUT
}

/// Convert a single sRGB u8 value to linear f32 using LUT lookup.
///
/// This is the fastest method for u8 input as it uses a precomputed lookup table
/// embedded in the binary. For batch conversions, use [`srgb_u8_to_linear_slice`].
///
/// # Example
/// ```
/// use linear_srgb::simd::srgb_u8_to_linear;
///
/// let linear = srgb_u8_to_linear(128);
/// assert!((linear - 0.2158).abs() < 0.001);
/// ```
#[inline]
pub fn srgb_u8_to_linear(value: u8) -> f32 {
    get_lut()[value as usize]
}

/// Convert 8 sRGB u8 values to linear f32 using LUT lookup.
///
/// # Example
/// ```
/// use linear_srgb::simd::srgb_u8_to_linear_x8;
///
/// let srgb = [0u8, 64, 128, 192, 255, 32, 96, 160];
/// let linear = srgb_u8_to_linear_x8(srgb);
/// ```
#[inline]
pub fn srgb_u8_to_linear_x8(srgb: [u8; 8]) -> [f32; 8] {
    let lut = get_lut();
    [
        lut[srgb[0] as usize],
        lut[srgb[1] as usize],
        lut[srgb[2] as usize],
        lut[srgb[3] as usize],
        lut[srgb[4] as usize],
        lut[srgb[5] as usize],
        lut[srgb[6] as usize],
        lut[srgb[7] as usize],
    ]
}

// ============================================================================
// magetypes #[rite] helpers (x86-64 only) — real AVX2+FMA SIMD
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[rite]
fn srgb_to_linear_mt(token: Desktop64, srgb: mt_f32x8) -> mt_f32x8 {
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
fn linear_to_srgb_mt(token: Desktop64, linear: mt_f32x8) -> mt_f32x8 {
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
fn gamma_to_linear_mt(token: Desktop64, encoded: mt_f32x8, gamma: f32) -> mt_f32x8 {
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let encoded = encoded.max(zero).min(one);
    encoded.pow_midp(gamma)
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn linear_to_gamma_mt(token: Desktop64, linear: mt_f32x8, gamma: f32) -> mt_f32x8 {
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let linear = linear.max(zero).min(one);
    linear.pow_midp(1.0 / gamma)
}

// ============================================================================
// Slice Functions with dispatch
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[arcane]
fn srgb_to_linear_slice_tier_v3(token: Desktop64, values: &mut [f32]) {
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
/// Processes 8 values at a time using SIMD, with scalar fallback for remainder.
///
/// # Example
/// ```
/// use linear_srgb::simd::srgb_to_linear_slice;
///
/// let mut values = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
/// srgb_to_linear_slice(&mut values);
/// ```
#[inline]
pub fn srgb_to_linear_slice(values: &mut [f32]) {
    incant!(srgb_to_linear_slice_tier(values), [v3])
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn linear_to_srgb_slice_tier_v3(token: Desktop64, values: &mut [f32]) {
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
/// Processes 8 values at a time using SIMD, with scalar fallback for remainder.
///
/// # Example
/// ```
/// use linear_srgb::simd::linear_to_srgb_slice;
///
/// let mut values = vec![0.0f32, 0.1, 0.2, 0.5, 1.0];
/// linear_to_srgb_slice(&mut values);
/// ```
#[inline]
pub fn linear_to_srgb_slice(values: &mut [f32]) {
    incant!(linear_to_srgb_slice_tier(values), [v3])
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
/// use linear_srgb::simd::srgb_u8_to_linear_slice;
///
/// let input: Vec<u8> = (0..=255).collect();
/// let mut output = vec![0.0f32; 256];
/// srgb_u8_to_linear_slice(&input, &mut output);
/// ```
#[inline]
pub fn srgb_u8_to_linear_slice(input: &[u8], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let lut = get_lut();

    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = [
            lut[inp[0] as usize],
            lut[inp[1] as usize],
            lut[inp[2] as usize],
            lut[inp[3] as usize],
            lut[inp[4] as usize],
            lut[inp[5] as usize],
            lut[inp[6] as usize],
            lut[inp[7] as usize],
        ];
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        *out = lut[*inp as usize];
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
/// use linear_srgb::simd::linear_to_srgb_u8_slice;
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

#[cfg(target_arch = "x86_64")]
#[arcane]
fn gamma_to_linear_slice_tier_v3(token: Desktop64, values: &mut [f32], gamma: f32) {
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
/// Processes 8 values at a time using SIMD, with scalar fallback for remainder.
///
/// # Example
/// ```
/// use linear_srgb::simd::gamma_to_linear_slice;
///
/// let mut values = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
/// gamma_to_linear_slice(&mut values, 2.2);
/// ```
#[inline]
pub fn gamma_to_linear_slice(values: &mut [f32], gamma: f32) {
    incant!(gamma_to_linear_slice_tier(values, gamma), [v3])
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn linear_to_gamma_slice_tier_v3(token: Desktop64, values: &mut [f32], gamma: f32) {
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
/// Processes 8 values at a time using SIMD, with scalar fallback for remainder.
///
/// # Example
/// ```
/// use linear_srgb::simd::linear_to_gamma_slice;
///
/// let mut values = vec![0.0f32, 0.1, 0.2, 0.5, 1.0];
/// linear_to_gamma_slice(&mut values, 2.2);
/// ```
#[inline]
pub fn linear_to_gamma_slice(values: &mut [f32], gamma: f32) {
    incant!(linear_to_gamma_slice_tier(values, gamma), [v3])
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
        let result = srgb_u8_to_linear_x8(input);

        for (i, (&r, &inp)) in result.iter().zip(input.iter()).enumerate() {
            let expected = crate::scalar::srgb_to_linear(inp as f32 / 255.0);
            assert!(
                (r - expected).abs() < 1e-4,
                "srgb_u8_to_linear_x8 mismatch at {}: got {}, expected {}",
                i, r, expected
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
                i, orig, conv
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
                i, output[i], expected
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
                i, output[i], i
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
                i, orig, conv
            );
        }
    }
}
