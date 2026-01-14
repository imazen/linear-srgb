//! SIMD-accelerated sRGB conversions using the `wide` crate.
//!
//! Processes 8 f32 values in parallel using AVX2/SSE or equivalent.
//! Uses runtime dispatch via multiversion for optimal code paths.

use wide::{CmpLt, f32x8};

use crate::simd_multiversion;

// Constants for SIMD operations
const SRGB_LINEAR_THRESHOLD: f32x8 = f32x8::splat(0.039_293_37); // 12.92 * 0.00304128...
const LINEAR_THRESHOLD: f32x8 = f32x8::splat(0.003_041_282_6);
const LINEAR_SCALE: f32x8 = f32x8::splat(1.0 / 12.92);
const SRGB_A: f32x8 = f32x8::splat(0.055_010_72);
const SRGB_A_PLUS_1: f32x8 = f32x8::splat(1.055_010_7);
const GAMMA: f32x8 = f32x8::splat(2.4);
const INV_GAMMA: f32x8 = f32x8::splat(1.0 / 2.4);
const TWELVE_92: f32x8 = f32x8::splat(12.92);
const ZERO: f32x8 = f32x8::splat(0.0);
const ONE: f32x8 = f32x8::splat(1.0);

simd_multiversion! {
    /// Convert 8 sRGB values to linear using SIMD.
    ///
    /// Values are clamped to [0, 1].
    #[inline]
    pub fn srgb_to_linear_x8(gamma: f32x8) -> f32x8 {
        // Clamp input
        let gamma = gamma.max(ZERO).min(ONE);

        // Linear segment: gamma * (1/12.92)
        let linear_result = gamma * LINEAR_SCALE;

        // Power segment: ((gamma + 0.055) / 1.055) ^ 2.4
        let power_result = ((gamma + SRGB_A) / SRGB_A_PLUS_1).pow_f32x8(GAMMA);

        // Select based on threshold: use linear if gamma < threshold
        let mask = gamma.simd_lt(SRGB_LINEAR_THRESHOLD);
        mask.blend(linear_result, power_result)
    }
}

simd_multiversion! {
    /// Convert 8 linear values to sRGB using SIMD.
    ///
    /// Values are clamped to [0, 1].
    #[inline]
    pub fn linear_to_srgb_x8(linear: f32x8) -> f32x8 {
        // Clamp input
        let linear = linear.max(ZERO).min(ONE);

        // Linear segment: linear * 12.92
        let linear_result = linear * TWELVE_92;

        // Power segment: 1.055 * linear^(1/2.4) - 0.055
        let power_result = SRGB_A_PLUS_1 * linear.pow_f32x8(INV_GAMMA) - SRGB_A;

        // Select based on threshold
        let mask = linear.simd_lt(LINEAR_THRESHOLD);
        mask.blend(linear_result, power_result)
    }
}

simd_multiversion! {
    /// Convert a slice of f32x8 vectors from sRGB to linear in-place.
    ///
    /// This is the most efficient API when data is already in f32x8 format.
    #[inline]
    pub fn srgb_to_linear_x8_slice(values: &mut [f32x8]) {
        for v in values.iter_mut() {
            *v = srgb_to_linear_x8(*v);
        }
    }
}

simd_multiversion! {
    /// Convert a slice of f32x8 vectors from linear to sRGB in-place.
    ///
    /// This is the most efficient API when data is already in f32x8 format.
    #[inline]
    pub fn linear_to_srgb_x8_slice(values: &mut [f32x8]) {
        for v in values.iter_mut() {
            *v = linear_to_srgb_x8(*v);
        }
    }
}

simd_multiversion! {
    /// Convert a slice of sRGB f32 values to linear in-place using SIMD.
    ///
    /// Processes 8 values at a time, with scalar fallback for remainder.
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
}

simd_multiversion! {
    /// Convert a slice of linear f32 values to sRGB in-place using SIMD.
    ///
    /// Processes 8 values at a time, with scalar fallback for remainder.
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
}

simd_multiversion! {
    /// Convert sRGB values to linear, writing to output slice.
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
}

simd_multiversion! {
    /// Convert linear values to sRGB, writing to output slice.
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
}
