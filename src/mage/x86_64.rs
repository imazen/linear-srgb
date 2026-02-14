//! x86-64 AVX2+FMA implementation.
//!
//! Uses `wide::f32x8` with `#[arcane]`/`#[rite]` for FMA-enabled target features.

use archmage::Avx2FmaToken;
use archmage::{arcane, rite};
use wide::{CmpLt, f32x8};

/// Token type for this platform (AVX2+FMA).
pub type Token = Avx2FmaToken;

// sRGB transfer function constants (C0-continuous, moxcms-derived)
const SRGB_LINEAR_THRESHOLD: f32 = 0.039_293_37;
const LINEAR_THRESHOLD: f32 = 0.003_041_282_6;
const LINEAR_SCALE: f32 = 1.0 / 12.92;
const SRGB_OFFSET: f32 = 0.055_010_72;
const SRGB_SCALE: f32 = 1.055_010_7;
const INV_SRGB_SCALE: f32 = 1.0 / 1.055_010_7;
const TWELVE_92: f32 = 12.92;

// ============================================================================
// Internal x8 functions (not public - SIMD types hidden)
// ============================================================================

/// Compute x^gamma using exp2/log2 approximation.
/// FMA-accelerated when compiled with target_feature.
#[inline(always)]
fn pow_x8(x: f32x8, gamma: f32) -> f32x8 {
    // x^gamma = 2^(gamma * log2(x))
    let log2_x = crate::fast_math::log2_x8(x);
    crate::fast_math::exp2_x8(log2_x * f32x8::splat(gamma))
}

#[rite]
fn srgb_to_linear_x8(_token: Avx2FmaToken, srgb: f32x8) -> f32x8 {
    let zero = f32x8::ZERO;
    let one = f32x8::ONE;
    let srgb = srgb.max(zero).min(one);

    let linear_result = srgb * f32x8::splat(LINEAR_SCALE);
    let power_result = pow_x8(
        (srgb + f32x8::splat(SRGB_OFFSET)) * f32x8::splat(INV_SRGB_SCALE),
        2.4,
    );

    let mask = srgb.simd_lt(f32x8::splat(SRGB_LINEAR_THRESHOLD));
    mask.blend(linear_result, power_result)
}

#[rite]
fn linear_to_srgb_x8(_token: Avx2FmaToken, linear: f32x8) -> f32x8 {
    let zero = f32x8::ZERO;
    let one = f32x8::ONE;
    let linear = linear.max(zero).min(one);

    let linear_result = linear * f32x8::splat(TWELVE_92);
    let power_result =
        f32x8::splat(SRGB_SCALE) * pow_x8(linear, 1.0 / 2.4) - f32x8::splat(SRGB_OFFSET);

    let mask = linear.simd_lt(f32x8::splat(LINEAR_THRESHOLD));
    mask.blend(linear_result, power_result)
}

#[rite]
fn gamma_to_linear_x8(_token: Avx2FmaToken, encoded: f32x8, gamma: f32) -> f32x8 {
    let zero = f32x8::ZERO;
    let one = f32x8::ONE;
    let encoded = encoded.max(zero).min(one);
    pow_x8(encoded, gamma)
}

#[rite]
fn linear_to_gamma_x8(_token: Avx2FmaToken, linear: f32x8, gamma: f32) -> f32x8 {
    let zero = f32x8::ZERO;
    let one = f32x8::ONE;
    let linear = linear.max(zero).min(one);
    pow_x8(linear, 1.0 / gamma)
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
        let v = f32x8::from(*chunk);
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
        let v = f32x8::from(*chunk);
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
/// Uses a 4097-entry const LUT — no pow/log/exp computation.
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
        let v = f32x8::from(*chunk);
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
        let v = f32x8::from(*chunk);
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
