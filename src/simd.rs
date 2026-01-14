//! SIMD-accelerated sRGB ↔ linear conversion.
//!
//! This module provides high-performance conversion functions using AVX2/SSE SIMD
//! instructions via the `wide` crate with runtime CPU feature detection.
//!
//! # API Overview
//!
//! ## x8 Functions (process 8 values at once)
//! - [`srgb_to_linear_x8`] - f32x8 sRGB → f32x8 linear
//! - [`linear_to_srgb_x8`] - f32x8 linear → f32x8 sRGB
//! - [`srgb_u8_to_linear_x8`] - \[u8; 8\] sRGB → f32x8 linear
//! - [`linear_to_srgb_u8_x8`] - f32x8 linear → \[u8; 8\] sRGB
//!
//! ## Slice Functions (process entire slices)
//! - [`srgb_to_linear_slice`] - &mut \[f32\] sRGB → linear in-place
//! - [`linear_to_srgb_slice`] - &mut \[f32\] linear → sRGB in-place
//! - [`srgb_u8_to_linear_slice`] - &\[u8\] sRGB → &mut \[f32\] linear
//! - [`linear_to_srgb_u8_slice`] - &\[f32\] linear → &mut \[u8\] sRGB

use multiversed::multiversed;
use wide::{CmpLt, f32x8};

use crate::fast_math::pow_x8;

// sRGB transfer function constants (IEC 61966-2-1)
const SRGB_LINEAR_THRESHOLD: f32x8 = f32x8::splat(0.039_293_37);
const LINEAR_THRESHOLD: f32x8 = f32x8::splat(0.003_041_282_6);
const LINEAR_SCALE: f32x8 = f32x8::splat(1.0 / 12.92);
const SRGB_OFFSET: f32x8 = f32x8::splat(0.055);
const SRGB_SCALE: f32x8 = f32x8::splat(1.055);
const TWELVE_92: f32x8 = f32x8::splat(12.92);
const ZERO: f32x8 = f32x8::splat(0.0);
const ONE: f32x8 = f32x8::splat(1.0);
const U8_MAX: f32x8 = f32x8::splat(255.0);
const HALF: f32x8 = f32x8::splat(0.5);

/// Precomputed sRGB u8 → linear f32 lookup table (computed once at first use).
/// Uses the same constants as the transfer module for consistency.
fn get_lut() -> &'static [f32; 256] {
    use std::sync::LazyLock;
    static LUT: LazyLock<[f32; 256]> = LazyLock::new(|| {
        let mut lut = [0.0f32; 256];
        for i in 0..256 {
            let s = i as f32 / 255.0;
            // Use the same formula as crate::srgb_to_linear
            lut[i] = crate::srgb_to_linear(s);
        }
        lut
    });
    &LUT
}

// ============================================================================
// x8 Functions - Process 8 values at once
// ============================================================================

/// Convert 8 sRGB f32 values to linear.
///
/// Input values are clamped to \[0, 1\].
///
/// # Example
/// ```
/// use linear_srgb::simd::srgb_to_linear_x8;
/// use wide::f32x8;
///
/// let srgb = f32x8::from([0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9, 0.5]);
/// let linear = srgb_to_linear_x8(srgb);
/// ```
#[multiversed]
#[inline]
pub fn srgb_to_linear_x8(srgb: f32x8) -> f32x8 {
    let srgb = srgb.max(ZERO).min(ONE);
    let linear_result = srgb * LINEAR_SCALE;
    let power_result = pow_x8((srgb + SRGB_OFFSET) / SRGB_SCALE, 2.4);
    let mask = srgb.simd_lt(SRGB_LINEAR_THRESHOLD);
    mask.blend(linear_result, power_result)
}

/// Convert 8 linear f32 values to sRGB.
///
/// Input values are clamped to \[0, 1\].
///
/// # Example
/// ```
/// use linear_srgb::simd::linear_to_srgb_x8;
/// use wide::f32x8;
///
/// let linear = f32x8::from([0.0, 0.1, 0.2, 0.5, 1.0, 0.01, 0.05, 0.8]);
/// let srgb = linear_to_srgb_x8(linear);
/// ```
#[multiversed]
#[inline]
pub fn linear_to_srgb_x8(linear: f32x8) -> f32x8 {
    let linear = linear.max(ZERO).min(ONE);
    let linear_result = linear * TWELVE_92;
    let power_result = SRGB_SCALE * pow_x8(linear, 1.0 / 2.4) - SRGB_OFFSET;
    let mask = linear.simd_lt(LINEAR_THRESHOLD);
    mask.blend(linear_result, power_result)
}

/// Convert 8 sRGB u8 values to linear f32 using LUT lookup.
///
/// This is the fastest method for u8 input as it uses a precomputed lookup table.
///
/// # Example
/// ```
/// use linear_srgb::simd::srgb_u8_to_linear_x8;
///
/// let srgb = [0u8, 64, 128, 192, 255, 32, 96, 160];
/// let linear = srgb_u8_to_linear_x8(srgb);
/// ```
#[inline]
pub fn srgb_u8_to_linear_x8(srgb: [u8; 8]) -> f32x8 {
    let lut = get_lut();
    f32x8::from([
        lut[srgb[0] as usize],
        lut[srgb[1] as usize],
        lut[srgb[2] as usize],
        lut[srgb[3] as usize],
        lut[srgb[4] as usize],
        lut[srgb[5] as usize],
        lut[srgb[6] as usize],
        lut[srgb[7] as usize],
    ])
}

/// Convert 8 linear f32 values to sRGB u8.
///
/// Input values are clamped to \[0, 1\], output is rounded to nearest u8.
///
/// # Example
/// ```
/// use linear_srgb::simd::linear_to_srgb_u8_x8;
/// use wide::f32x8;
///
/// let linear = f32x8::from([0.0, 0.1, 0.2, 0.5, 1.0, 0.01, 0.05, 0.8]);
/// let srgb = linear_to_srgb_u8_x8(linear);
/// ```
#[multiversed]
#[inline]
pub fn linear_to_srgb_u8_x8(linear: f32x8) -> [u8; 8] {
    let srgb = linear_to_srgb_x8(linear);
    let scaled = srgb * U8_MAX + HALF;
    let arr: [f32; 8] = scaled.into();
    [
        arr[0] as u8,
        arr[1] as u8,
        arr[2] as u8,
        arr[3] as u8,
        arr[4] as u8,
        arr[5] as u8,
        arr[6] as u8,
        arr[7] as u8,
    ]
}

// ============================================================================
// Slice Functions - Process entire slices
// ============================================================================

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
#[multiversed]
#[inline]
pub fn srgb_to_linear_slice(values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let result = srgb_to_linear_x8(f32x8::from(*chunk));
        *chunk = result.into();
    }

    for v in remainder {
        *v = crate::srgb_to_linear(*v);
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
#[multiversed]
#[inline]
pub fn linear_to_srgb_slice(values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let result = linear_to_srgb_x8(f32x8::from(*chunk));
        *chunk = result.into();
    }

    for v in remainder {
        *v = crate::linear_to_srgb(*v);
    }
}

/// Convert sRGB u8 values to linear f32.
///
/// Uses a precomputed LUT for each u8 value, processed in SIMD batches of 8.
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
/// Processes 8 values at a time using SIMD, with scalar fallback for remainder.
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
#[multiversed]
#[inline]
pub fn linear_to_srgb_u8_slice(input: &[f32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());

    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = linear_to_srgb_u8_x8(f32x8::from(*inp));
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        let srgb = crate::linear_to_srgb(*inp);
        *out = (srgb * 255.0 + 0.5) as u8;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- x8 function tests ----

    #[test]
    fn test_srgb_to_linear_x8() {
        let input = [0.0f32, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9, 0.04];
        let result = srgb_to_linear_x8(f32x8::from(input));
        let result_arr: [f32; 8] = result.into();

        for (i, &inp) in input.iter().enumerate() {
            let expected = crate::srgb_to_linear(inp);
            assert!(
                (result_arr[i] - expected).abs() < 1e-5,
                "srgb_to_linear_x8 mismatch at {}: got {}, expected {}",
                i,
                result_arr[i],
                expected
            );
        }
    }

    #[test]
    fn test_linear_to_srgb_x8() {
        let input = [0.0f32, 0.1, 0.2, 0.5, 1.0, 0.01, 0.001, 0.8];
        let result = linear_to_srgb_x8(f32x8::from(input));
        let result_arr: [f32; 8] = result.into();

        for (i, &inp) in input.iter().enumerate() {
            let expected = crate::linear_to_srgb(inp);
            assert!(
                (result_arr[i] - expected).abs() < 1e-5,
                "linear_to_srgb_x8 mismatch at {}: got {}, expected {}",
                i,
                result_arr[i],
                expected
            );
        }
    }

    #[test]
    fn test_srgb_u8_to_linear_x8() {
        let input: [u8; 8] = [0, 64, 128, 192, 255, 32, 96, 160];
        let result = srgb_u8_to_linear_x8(input);
        let result_arr: [f32; 8] = result.into();

        for (i, &inp) in input.iter().enumerate() {
            let expected = crate::srgb_u8_to_linear(inp);
            assert!(
                (result_arr[i] - expected).abs() < 1e-6,
                "srgb_u8_to_linear_x8 mismatch at {}: got {}, expected {}",
                i,
                result_arr[i],
                expected
            );
        }
    }

    #[test]
    fn test_linear_to_srgb_u8_x8() {
        let input = [0.0f32, 0.1, 0.2, 0.5, 1.0, 0.01, 0.05, 0.8];
        let result = linear_to_srgb_u8_x8(f32x8::from(input));

        for (i, &inp) in input.iter().enumerate() {
            let expected = (crate::linear_to_srgb(inp) * 255.0 + 0.5) as u8;
            assert!(
                (result[i] as i16 - expected as i16).abs() <= 1,
                "linear_to_srgb_u8_x8 mismatch at {}: got {}, expected {}",
                i,
                result[i],
                expected
            );
        }
    }

    // ---- Slice function tests ----

    #[test]
    fn test_srgb_to_linear_slice() {
        let mut values: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let expected: Vec<f32> = values.iter().map(|&v| crate::srgb_to_linear(v)).collect();

        srgb_to_linear_slice(&mut values);

        for (i, (&got, &exp)) in values.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "srgb_to_linear_slice mismatch at {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_linear_to_srgb_slice() {
        let mut values: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let expected: Vec<f32> = values.iter().map(|&v| crate::linear_to_srgb(v)).collect();

        linear_to_srgb_slice(&mut values);

        for (i, (&got, &exp)) in values.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "linear_to_srgb_slice mismatch at {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_srgb_u8_to_linear_slice() {
        let input: Vec<u8> = (0..=255).collect();
        let mut output = vec![0.0f32; 256];

        srgb_u8_to_linear_slice(&input, &mut output);

        for i in 0..256 {
            let expected = crate::srgb_u8_to_linear(i as u8);
            assert!(
                (output[i] - expected).abs() < 1e-6,
                "srgb_u8_to_linear_slice mismatch at {}: got {}, expected {}",
                i,
                output[i],
                expected
            );
        }
    }

    #[test]
    fn test_linear_to_srgb_u8_slice() {
        let input: Vec<f32> = (0..=255).map(|i| i as f32 / 255.0).collect();
        let mut output = vec![0u8; 256];

        linear_to_srgb_u8_slice(&input, &mut output);

        for i in 0..256 {
            let expected = (crate::linear_to_srgb(input[i]) * 255.0 + 0.5) as u8;
            assert!(
                (output[i] as i16 - expected as i16).abs() <= 1,
                "linear_to_srgb_u8_slice mismatch at {}: got {}, expected {}",
                i,
                output[i],
                expected
            );
        }
    }

    // ---- Roundtrip tests ----

    #[test]
    fn test_f32_roundtrip() {
        let mut values: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        let original = values.clone();

        srgb_to_linear_slice(&mut values);
        linear_to_srgb_slice(&mut values);

        for (i, (&orig, &conv)) in original.iter().zip(values.iter()).enumerate() {
            assert!(
                (orig - conv).abs() < 1e-4,
                "f32 roundtrip failed at {}: {} -> {}",
                i,
                orig,
                conv
            );
        }
    }

    #[test]
    fn test_u8_roundtrip() {
        let input: Vec<u8> = (0..=255).collect();
        let mut linear = vec![0.0f32; 256];
        let mut back = vec![0u8; 256];

        srgb_u8_to_linear_slice(&input, &mut linear);
        linear_to_srgb_u8_slice(&linear, &mut back);

        for i in 0..256 {
            assert!(
                (input[i] as i16 - back[i] as i16).abs() <= 1,
                "u8 roundtrip failed at {}: {} -> {} -> {}",
                i,
                input[i],
                linear[i],
                back[i]
            );
        }
    }

    // ---- Edge case tests ----

    #[test]
    fn test_clamping() {
        // Test that out-of-range values are clamped
        let input = f32x8::from([-0.5, -0.1, 0.0, 0.5, 1.0, 1.5, 2.0, 10.0]);
        let result = srgb_to_linear_x8(input);
        let arr: [f32; 8] = result.into();

        assert_eq!(arr[0], 0.0, "negative should clamp to 0");
        assert_eq!(arr[1], 0.0, "negative should clamp to 0");
        assert!(arr[4] > 0.99 && arr[4] <= 1.0, "1.0 should stay ~1.0");
        assert!(arr[5] > 0.99 && arr[5] <= 1.0, "values > 1 should clamp");
    }

    #[test]
    fn test_linear_segment() {
        // Test values in the linear segment (< 0.04045)
        let input = f32x8::from([0.0, 0.01, 0.02, 0.03, 0.04, 0.005, 0.015, 0.035]);
        let result = srgb_to_linear_x8(input);
        let arr: [f32; 8] = result.into();
        let input_arr: [f32; 8] = input.into();

        for i in 0..8 {
            let expected = input_arr[i] / 12.92;
            assert!(
                (arr[i] - expected).abs() < 1e-6,
                "linear segment mismatch at {}: got {}, expected {}",
                i,
                arr[i],
                expected
            );
        }
    }

    #[test]
    fn test_empty_slice() {
        let mut empty: Vec<f32> = vec![];
        srgb_to_linear_slice(&mut empty);
        assert!(empty.is_empty());

        let empty_u8: Vec<u8> = vec![];
        let mut empty_out: Vec<f32> = vec![];
        srgb_u8_to_linear_slice(&empty_u8, &mut empty_out);
    }

    #[test]
    fn test_non_multiple_of_8() {
        // Test slices that aren't multiples of 8
        for len in [1, 3, 7, 9, 15, 17, 100] {
            let mut values: Vec<f32> = (0..len).map(|i| i as f32 / len as f32).collect();
            let expected: Vec<f32> = values.iter().map(|&v| crate::srgb_to_linear(v)).collect();

            srgb_to_linear_slice(&mut values);

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
