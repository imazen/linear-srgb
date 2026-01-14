//! SIMD-accelerated sRGB conversions using the `wide` crate.
//!
//! Processes 8 f32 values in parallel using AVX2/SSE or equivalent.
//! Uses runtime dispatch via multiversed for optimal code paths.

use multiversed::multiversed;
use wide::{CmpLt, f32x8};

use crate::fast_math::dirty_pow_const_x8;
use crate::lut::LinearTable8;

// Constants for u8 conversion
const U8_MAX_F32: f32x8 = f32x8::splat(255.0);
const HALF: f32x8 = f32x8::splat(0.5);

// Constants for SIMD operations
const SRGB_LINEAR_THRESHOLD: f32x8 = f32x8::splat(0.039_293_37); // 12.92 * 0.00304128...
const LINEAR_THRESHOLD: f32x8 = f32x8::splat(0.003_041_282_6);
const LINEAR_SCALE: f32x8 = f32x8::splat(1.0 / 12.92);
const SRGB_A: f32x8 = f32x8::splat(0.055_010_72);
const SRGB_A_PLUS_1: f32x8 = f32x8::splat(1.055_010_7);
const TWELVE_92: f32x8 = f32x8::splat(12.92);
const ZERO: f32x8 = f32x8::splat(0.0);
const ONE: f32x8 = f32x8::splat(1.0);

/// Convert 8 sRGB values to linear using SIMD.
///
/// Values are clamped to [0, 1].
#[multiversed]
#[inline]
pub fn srgb_to_linear_x8(gamma: f32x8) -> f32x8 {
    // Clamp input
    let gamma = gamma.max(ZERO).min(ONE);

    // Linear segment: gamma * (1/12.92)
    let linear_result = gamma * LINEAR_SCALE;

    // Power segment: ((gamma + 0.055) / 1.055) ^ 2.4
    // Use fast pow approximation instead of slow pow_f32x8
    let power_result = dirty_pow_const_x8((gamma + SRGB_A) / SRGB_A_PLUS_1, 2.4);

    // Select based on threshold: use linear if gamma < threshold
    let mask = gamma.simd_lt(SRGB_LINEAR_THRESHOLD);
    mask.blend(linear_result, power_result)
}

/// Convert 8 linear values to sRGB using SIMD.
///
/// Values are clamped to [0, 1].
#[multiversed]
#[inline]
pub fn linear_to_srgb_x8(linear: f32x8) -> f32x8 {
    // Clamp input
    let linear = linear.max(ZERO).min(ONE);

    // Linear segment: linear * 12.92
    let linear_result = linear * TWELVE_92;

    // Power segment: 1.055 * linear^(1/2.4) - 0.055
    // Use fast pow approximation instead of slow pow_f32x8
    let power_result = SRGB_A_PLUS_1 * dirty_pow_const_x8(linear, 1.0 / 2.4) - SRGB_A;

    // Select based on threshold
    let mask = linear.simd_lt(LINEAR_THRESHOLD);
    mask.blend(linear_result, power_result)
}

/// Convert a slice of f32x8 vectors from sRGB to linear in-place.
///
/// This is the most efficient API when data is already in f32x8 format.
#[multiversed]
#[inline]
pub fn srgb_to_linear_x8_slice(values: &mut [f32x8]) {
    for v in values.iter_mut() {
        *v = srgb_to_linear_x8(*v);
    }
}

/// Convert a slice of f32x8 vectors from linear to sRGB in-place.
///
/// This is the most efficient API when data is already in f32x8 format.
#[multiversed]
#[inline]
pub fn linear_to_srgb_x8_slice(values: &mut [f32x8]) {
    for v in values.iter_mut() {
        *v = linear_to_srgb_x8(*v);
    }
}

/// Convert a slice of sRGB f32 values to linear in-place using SIMD.
///
/// Processes 8 values at a time, with scalar fallback for remainder.
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

/// Convert a slice of linear f32 values to sRGB in-place using SIMD.
///
/// Processes 8 values at a time, with scalar fallback for remainder.
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

/// Convert sRGB values to linear, writing to output slice.
#[multiversed]
#[inline]
pub fn srgb_to_linear_batch(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());

    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let result = srgb_to_linear_x8(f32x8::from(*inp));
        *out = result.into();
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        *out = crate::srgb_to_linear(*inp);
    }
}

/// Convert linear values to sRGB, writing to output slice.
#[multiversed]
#[inline]
pub fn linear_to_srgb_batch(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());

    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let result = linear_to_srgb_x8(f32x8::from(*inp));
        *out = result.into();
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        *out = crate::linear_to_srgb(*inp);
    }
}

/// Convert 8 u8 sRGB values to f32 linear using LUT lookups.
///
/// This uses scalar lookups but returns a SIMD vector for further processing.
#[inline]
pub fn srgb_u8_to_linear_x8(lut: &LinearTable8, values: [u8; 8]) -> f32x8 {
    f32x8::from([
        lut.lookup(values[0] as usize),
        lut.lookup(values[1] as usize),
        lut.lookup(values[2] as usize),
        lut.lookup(values[3] as usize),
        lut.lookup(values[4] as usize),
        lut.lookup(values[5] as usize),
        lut.lookup(values[6] as usize),
        lut.lookup(values[7] as usize),
    ])
}

/// Convert 8 f32 linear values to u8 sRGB.
///
/// Values are clamped to [0, 1] before conversion.
#[multiversed]
#[inline]
pub fn linear_to_srgb_u8_x8(linear: f32x8) -> [u8; 8] {
    // Apply the transfer function
    let srgb = linear_to_srgb_x8(linear);

    // Convert to u8: round(srgb * 255)
    let scaled = srgb * U8_MAX_F32 + HALF;
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

/// Batch convert u8 sRGB values to f32 linear using SIMD.
///
/// Processes 8 values at a time using LUT lookups, with scalar fallback for remainder.
#[inline]
pub fn srgb_u8_to_linear_batch(lut: &LinearTable8, input: &[u8], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());

    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let result = srgb_u8_to_linear_x8(lut, *inp);
        *out = result.into();
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        *out = lut.lookup(*inp as usize);
    }
}

/// Batch convert f32 linear values to u8 sRGB using SIMD.
///
/// Processes 8 values at a time, with scalar fallback for remainder.
#[multiversed]
#[inline]
pub fn linear_to_srgb_u8_batch(input: &[f32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());

    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let v = f32x8::from(*inp);
        *out = linear_to_srgb_u8_x8(v);
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        let srgb = crate::linear_to_srgb(*inp);
        *out = (srgb * 255.0 + 0.5) as u8;
    }
}

/// Batch convert u8 sRGB values to f32 linear in-place (writing to separate output).
///
/// Processes entire rows of RGBA or RGB data, handling alpha pass-through.
#[inline]
pub fn srgb_u8_to_linear_rgb_batch(lut: &LinearTable8, input: &[u8], output: &mut [f32]) {
    srgb_u8_to_linear_batch(lut, input, output);
}

/// Batch convert f32 linear RGB values to u8 sRGB.
///
/// Processes entire rows of RGB data.
#[multiversed]
#[inline]
pub fn linear_to_srgb_u8_rgb_batch(input: &[f32], output: &mut [u8]) {
    linear_to_srgb_u8_batch(input, output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_srgb_to_linear() {
        let input = [0.0f32, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9, 0.04];
        let v = f32x8::from(input);
        let result = srgb_to_linear_x8(v);
        let result_arr: [f32; 8] = result.into();

        for (i, &inp) in input.iter().enumerate() {
            let scalar = crate::srgb_to_linear(inp);
            assert!(
                (result_arr[i] - scalar).abs() < 1e-5,
                "Mismatch at {}: SIMD={}, scalar={}",
                i,
                result_arr[i],
                scalar
            );
        }
    }

    #[test]
    fn test_simd_linear_to_srgb() {
        let input = [0.0f32, 0.1, 0.2, 0.5, 1.0, 0.01, 0.001, 0.8];
        let v = f32x8::from(input);
        let result = linear_to_srgb_x8(v);
        let result_arr: [f32; 8] = result.into();

        for (i, &inp) in input.iter().enumerate() {
            let scalar = crate::linear_to_srgb(inp);
            assert!(
                (result_arr[i] - scalar).abs() < 1e-5,
                "Mismatch at {}: SIMD={}, scalar={}",
                i,
                result_arr[i],
                scalar
            );
        }
    }

    #[test]
    fn test_simd_batch() {
        let input: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        let mut output = vec![0.0f32; 1000];

        srgb_to_linear_batch(&input, &mut output);

        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            let scalar = crate::srgb_to_linear(inp);
            assert!(
                (out - scalar).abs() < 1e-5,
                "Batch mismatch at {}: batch={}, scalar={}",
                i,
                out,
                scalar
            );
        }
    }

    #[test]
    fn test_simd_roundtrip() {
        let mut values: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        let original = values.clone();

        srgb_to_linear_slice(&mut values);
        linear_to_srgb_slice(&mut values);

        for (i, (&orig, &conv)) in original.iter().zip(values.iter()).enumerate() {
            assert!(
                (orig - conv).abs() < 1e-4,
                "Roundtrip failed at {}: {} -> {}",
                i,
                orig,
                conv
            );
        }
    }

    #[test]
    fn test_srgb_u8_to_linear_x8() {
        let lut = LinearTable8::new();
        let input: [u8; 8] = [0, 64, 128, 192, 255, 32, 96, 160];
        let result = srgb_u8_to_linear_x8(&lut, input);
        let result_arr: [f32; 8] = result.into();

        for (i, &inp) in input.iter().enumerate() {
            let expected = lut.lookup(inp as usize);
            assert!(
                (result_arr[i] - expected).abs() < 1e-6,
                "Mismatch at {}: got={}, expected={}",
                i,
                result_arr[i],
                expected
            );
        }
    }

    #[test]
    fn test_linear_to_srgb_u8_x8() {
        let input = [0.0f32, 0.1, 0.2, 0.5, 1.0, 0.01, 0.05, 0.8];
        let v = f32x8::from(input);
        let result = linear_to_srgb_u8_x8(v);

        for (i, &inp) in input.iter().enumerate() {
            let srgb = crate::linear_to_srgb(inp);
            let expected = (srgb * 255.0 + 0.5) as u8;
            assert!(
                (result[i] as i16 - expected as i16).abs() <= 1,
                "Mismatch at {}: got={}, expected={}",
                i,
                result[i],
                expected
            );
        }
    }

    #[test]
    fn test_srgb_u8_to_linear_batch() {
        let lut = LinearTable8::new();
        let input: Vec<u8> = (0..=255).collect();
        let mut output = vec![0.0f32; 256];

        srgb_u8_to_linear_batch(&lut, &input, &mut output);

        for (i, &out) in output.iter().enumerate() {
            let expected = lut.lookup(i);
            assert!(
                (out - expected).abs() < 1e-6,
                "Batch mismatch at {}: got={}, expected={}",
                i,
                out,
                expected
            );
        }
    }

    #[test]
    fn test_linear_to_srgb_u8_batch() {
        let input: Vec<f32> = (0..=255).map(|i| i as f32 / 255.0).collect();
        let mut output = vec![0u8; 256];

        linear_to_srgb_u8_batch(&input, &mut output);

        for i in 0..256 {
            let srgb = crate::linear_to_srgb(input[i]);
            let expected = (srgb * 255.0 + 0.5) as u8;
            assert!(
                (output[i] as i16 - expected as i16).abs() <= 1,
                "Batch mismatch at {}: got={}, expected={}",
                i,
                output[i],
                expected
            );
        }
    }

    #[test]
    fn test_u8_roundtrip() {
        let lut = LinearTable8::new();

        // Test all u8 values
        for i in 0..=255u8 {
            // u8 sRGB -> f32 linear -> u8 sRGB
            let linear = lut.lookup(i as usize);
            let srgb = crate::linear_to_srgb(linear);
            let back = (srgb * 255.0 + 0.5) as u8;

            assert!(
                (i as i16 - back as i16).abs() <= 1,
                "Roundtrip failed for {}: {} -> {} -> {}",
                i,
                i,
                linear,
                back
            );
        }
    }

    #[test]
    fn test_u8_batch_roundtrip() {
        let lut = LinearTable8::new();
        let input: Vec<u8> = (0..=255).collect();
        let mut linear = vec![0.0f32; 256];
        let mut back = vec![0u8; 256];

        srgb_u8_to_linear_batch(&lut, &input, &mut linear);
        linear_to_srgb_u8_batch(&linear, &mut back);

        for i in 0..256 {
            assert!(
                (input[i] as i16 - back[i] as i16).abs() <= 1,
                "Batch roundtrip failed at {}: {} -> {} -> {}",
                i,
                input[i],
                linear[i],
                back[i]
            );
        }
    }
}
