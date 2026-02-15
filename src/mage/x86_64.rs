//! x86-64 AVX2+FMA implementation using magetypes for real SIMD.
//!
//! Uses `magetypes::simd::f32x8` with `#[arcane]`/`#[rite]` for FMA-enabled
//! target features. Unlike `wide::f32x8` (which only uses AVX when compiled
//! with `+avx`), magetypes applies `#[target_feature]` at function level,
//! producing real AVX2+FMA instructions regardless of compile flags.

use archmage::Avx2FmaToken;
use archmage::{arcane, rite};
use magetypes::simd::f32x8;

/// Token type for this platform (AVX2+FMA).
pub type Token = Avx2FmaToken;

// sRGB transfer function constants (C0-continuous, moxcms-derived)
const SRGB_LINEAR_THRESHOLD: f32 = 0.039_293_37;
const LINEAR_THRESHOLD: f32 = 0.003_041_282_6;
const LINEAR_SCALE: f32 = 1.0 / 12.92;
const SRGB_OFFSET: f32 = 0.055_010_72;
const SRGB_SCALE: f32 = 1.055_010_7;
const TWELVE_92: f32 = 12.92;

// sRGB→linear degree-11 Chebyshev polynomial (Estrin's scheme)
const S2L_INV_HW: f32 = 2.081_801;
const S2L_BIAS: f32 = -1.081_800_9;
const S2L_C0: f32 = 2.326_832_7e-1;
const S2L_C1: f32 = 4.667_970_8e-1;
const S2L_C2: f32 = 2.731_341e-1;
const S2L_C3: f32 = 3.044_251_2e-2;
const S2L_C4: f32 = -3.802_638_5e-3;
const S2L_C5: f32 = 1.011_499_3e-3;
const S2L_C6: f32 = -4.267_19e-4;
const S2L_C7: f32 = 1.966_666_5e-4;
const S2L_C8: f32 = 2.025_719_4e-5;
const S2L_C9: f32 = -2.400_594_3e-5;
const S2L_C10: f32 = -8.762_017e-5;
const S2L_C11: f32 = 5.557_536_5e-5;

// ============================================================================
// Internal x8 functions (not public - SIMD types hidden)
// ============================================================================

#[rite]
fn srgb_to_linear_x8(token: Avx2FmaToken, srgb: f32x8) -> f32x8 {
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let srgb = srgb.max(zero).min(one);

    let linear_result = srgb * f32x8::splat(token, LINEAR_SCALE);

    // Degree-11 Chebyshev polynomial (Estrin evaluation)
    let u = srgb.mul_add(
        f32x8::splat(token, S2L_INV_HW),
        f32x8::splat(token, S2L_BIAS),
    );
    let u2 = u * u;
    let u4 = u2 * u2;
    let u_8 = u4 * u4;
    let p01 = f32x8::splat(token, S2L_C1).mul_add(u, f32x8::splat(token, S2L_C0));
    let p23 = f32x8::splat(token, S2L_C3).mul_add(u, f32x8::splat(token, S2L_C2));
    let p45 = f32x8::splat(token, S2L_C5).mul_add(u, f32x8::splat(token, S2L_C4));
    let p67 = f32x8::splat(token, S2L_C7).mul_add(u, f32x8::splat(token, S2L_C6));
    let p89 = f32x8::splat(token, S2L_C9).mul_add(u, f32x8::splat(token, S2L_C8));
    let pab = f32x8::splat(token, S2L_C11).mul_add(u, f32x8::splat(token, S2L_C10));
    let p0123 = p23.mul_add(u2, p01);
    let p4567 = p67.mul_add(u2, p45);
    let p8_11 = pab.mul_add(u2, p89);
    let p0_7 = p4567.mul_add(u4, p0123);
    let power_result = p8_11.mul_add(u_8, p0_7);

    let mask = srgb.simd_lt(f32x8::splat(token, SRGB_LINEAR_THRESHOLD));
    f32x8::blend(mask, linear_result, power_result)
}

#[rite]
fn linear_to_srgb_x8(token: Avx2FmaToken, linear: f32x8) -> f32x8 {
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let linear = linear.max(zero).min(one);

    let linear_result = linear * f32x8::splat(token, TWELVE_92);
    let power_result = f32x8::splat(token, SRGB_SCALE) * linear.pow_midp(1.0 / 2.4)
        - f32x8::splat(token, SRGB_OFFSET);

    let mask = linear.simd_lt(f32x8::splat(token, LINEAR_THRESHOLD));
    f32x8::blend(mask, linear_result, power_result)
}

#[rite]
fn gamma_to_linear_x8(token: Avx2FmaToken, encoded: f32x8, gamma: f32) -> f32x8 {
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let encoded = encoded.max(zero).min(one);
    encoded.pow_midp(gamma)
}

#[rite]
fn linear_to_gamma_x8(token: Avx2FmaToken, linear: f32x8, gamma: f32) -> f32x8 {
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let linear = linear.max(zero).min(one);
    linear.pow_midp(1.0 / gamma)
}

// ============================================================================
// Public slice-based API
// ============================================================================

/// Convert sRGB f32 values to linear in-place.
///
/// Processes 8 values at a time using AVX2+FMA SIMD.
/// Remainder elements use scalar fallback.
#[arcane]
pub fn srgb_to_linear_slice(token: Avx2FmaToken, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let v = f32x8::from_array(token, *chunk);
        let result = srgb_to_linear_x8(token, v);
        *chunk = result.to_array();
    }

    for v in remainder {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

/// Convert linear f32 values to sRGB in-place.
///
/// Processes 8 values at a time using AVX2+FMA SIMD.
/// Remainder elements use scalar fallback.
#[arcane]
pub fn linear_to_srgb_slice(token: Avx2FmaToken, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let v = f32x8::from_array(token, *chunk);
        let result = linear_to_srgb_x8(token, v);
        *chunk = result.to_array();
    }

    for v in remainder {
        *v = crate::scalar::linear_to_srgb(*v);
    }
}

/// Convert sRGB u8 values to linear f32.
///
/// Uses a precomputed LUT for each u8 value (no SIMD needed).
/// Delegates to the optimized batch implementation.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
#[inline]
pub fn srgb_u8_to_linear_slice(_token: Avx2FmaToken, input: &[u8], output: &mut [f32]) {
    // Delegate to the optimized simd implementation which does batch LUT lookups
    crate::simd::srgb_u8_to_linear_slice(input, output);
}

/// Convert linear f32 values to sRGB u8.
///
/// Uses a 4096-entry const LUT — no pow/log/exp computation.
/// The token is accepted for API compatibility but not required for this path.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
#[inline]
pub fn linear_to_srgb_u8_slice(_token: Avx2FmaToken, input: &[f32], output: &mut [u8]) {
    // Delegate to the LUT-based implementation (no SIMD dispatch needed)
    crate::simd::linear_to_srgb_u8_slice(input, output);
}

/// Convert gamma-encoded f32 values to linear in-place.
///
/// Processes 8 values at a time using AVX2+FMA SIMD.
#[arcane]
pub fn gamma_to_linear_slice(token: Avx2FmaToken, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let v = f32x8::from_array(token, *chunk);
        let result = gamma_to_linear_x8(token, v, gamma);
        *chunk = result.to_array();
    }

    for v in remainder {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

/// Convert linear f32 values to gamma-encoded in-place.
///
/// Processes 8 values at a time using AVX2+FMA SIMD.
#[arcane]
pub fn linear_to_gamma_slice(token: Avx2FmaToken, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let v = f32x8::from_array(token, *chunk);
        let result = linear_to_gamma_x8(token, v, gamma);
        *chunk = result.to_array();
    }

    for v in remainder {
        *v = crate::scalar::linear_to_gamma(*v, gamma);
    }
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

    fn get_token() -> Option<Token> {
        Token::try_new()
    }

    #[test]
    fn test_srgb_to_linear_slice() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
            return;
        };

        let mut values: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let expected: Vec<f32> = values
            .iter()
            .map(|&v| crate::scalar::srgb_to_linear(v))
            .collect();

        srgb_to_linear_slice(token, &mut values);

        for (i, (&got, &exp)) in values.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "mismatch at {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_linear_to_srgb_slice() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
            return;
        };

        let mut values: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let expected: Vec<f32> = values
            .iter()
            .map(|&v| crate::scalar::linear_to_srgb(v))
            .collect();

        linear_to_srgb_slice(token, &mut values);

        for (i, (&got, &exp)) in values.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "mismatch at {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
            return;
        };

        let mut values: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        let original = values.clone();

        srgb_to_linear_slice(token, &mut values);
        linear_to_srgb_slice(token, &mut values);

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

    #[test]
    fn test_remainder_handling() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
            return;
        };

        // Test with non-multiple-of-8 length
        for len in [1, 3, 7, 9, 15, 17] {
            let mut values: Vec<f32> = (0..len).map(|i| i as f32 / len as f32).collect();
            let expected: Vec<f32> = values
                .iter()
                .map(|&v| crate::scalar::srgb_to_linear(v))
                .collect();

            srgb_to_linear_slice(token, &mut values);

            for (i, (&got, &exp)) in values.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-5,
                    "len={} mismatch at {}: got {}, expected {}",
                    len,
                    i,
                    got,
                    exp
                );
            }
        }
    }
}
