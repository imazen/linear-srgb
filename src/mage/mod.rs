//! Token-based SIMD API with zero dispatch overhead.
//!
//! This module provides an alternative API using archmage tokens for users who
//! want to avoid per-call dispatch overhead. Obtain a token once at startup,
//! then pass it to all conversion functions.
//!
//! # Usage
//!
//! ```rust,ignore
//! use linear_srgb::mage::{self, Avx2FmaToken, SimdToken};
//!
//! fn main() {
//!     // Detection once at startup
//!     let token = Avx2FmaToken::try_new().expect("need AVX2+FMA");
//!
//!     let mut data = vec![0.5f32; 10000];
//!
//!     // Zero dispatch overhead - token proves features available
//!     // Functions compiled with #[target_feature(enable = "avx2,fma")]
//!     mage::srgb_to_linear_slice(token, &mut data);
//! }
//! ```
//!
//! # Performance
//!
//! Unlike the `default::inline::*` functions, the mage module uses `#[arcane]`
//! with archmage's native SIMD types. This allows LLVM to:
//!
//! - Use FMA instructions directly (vfmadd213ps, vfmadd231ps)
//! - Better inline SIMD operations
//! - Optimize across function boundaries
//!
//! # Comparison with Default API
//!
//! | Aspect | `default::` / `simd::` | `mage::` |
//! |--------|------------------------|----------|
//! | Dispatch | Per-call (amortized) | None (compile-time proof) |
//! | Target features | Generic | AVX2+FMA enabled |
//! | Best for | Casual use | Tight loops, max performance |

use archmage::arcane;
use archmage::simd::avx2::f32x8;

// Re-export core types from archmage
pub use archmage::{Avx2FmaToken, SimdToken};

// Also export other common tokens users might want
#[cfg(target_arch = "x86_64")]
pub use archmage::{X64V2Token, X64V3Token, X64V4Token};

#[cfg(target_arch = "aarch64")]
pub use archmage::NeonToken;

// sRGB transfer function constants (C0-continuous, moxcms-derived)
const SRGB_LINEAR_THRESHOLD: f32 = 0.039_293_37;
const LINEAR_THRESHOLD: f32 = 0.003_041_282_6;
const LINEAR_SCALE: f32 = 1.0 / 12.92;
const SRGB_OFFSET: f32 = 0.055_010_72;
const SRGB_SCALE: f32 = 1.055_010_7;
const TWELVE_92: f32 = 12.92;

// ============================================================================
// x8 Functions - process 8 values at once
// ============================================================================

/// Convert 8 sRGB f32 values to linear.
///
/// Compiled with `#[target_feature(enable = "avx2,fma")]` for optimal codegen.
/// Uses archmage's native SIMD types for true FMA instruction generation.
/// Input values are clamped to \[0, 1\].
#[arcane]
pub fn srgb_to_linear_x8(token: Avx2FmaToken, srgb: f32x8) -> f32x8 {
    let zero = f32x8::splat(token, 0.0);
    let one = f32x8::splat(token, 1.0);
    let srgb = srgb.clamp(zero, one);

    let linear_result = srgb * f32x8::splat(token, LINEAR_SCALE);
    let power_result =
        ((srgb + f32x8::splat(token, SRGB_OFFSET)) / f32x8::splat(token, SRGB_SCALE)).pow_midp(2.4);

    let mask = srgb.simd_lt(f32x8::splat(token, SRGB_LINEAR_THRESHOLD));
    f32x8::blend(mask, linear_result, power_result)
}

/// Convert 8 linear f32 values to sRGB.
///
/// Compiled with `#[target_feature(enable = "avx2,fma")]` for optimal codegen.
/// Uses archmage's native SIMD types for true FMA instruction generation.
/// Input values are clamped to \[0, 1\].
#[arcane]
pub fn linear_to_srgb_x8(token: Avx2FmaToken, linear: f32x8) -> f32x8 {
    let zero = f32x8::splat(token, 0.0);
    let one = f32x8::splat(token, 1.0);
    let linear = linear.clamp(zero, one);

    let linear_result = linear * f32x8::splat(token, TWELVE_92);
    let power_result = f32x8::splat(token, SRGB_SCALE) * linear.pow_midp(1.0 / 2.4)
        - f32x8::splat(token, SRGB_OFFSET);

    let mask = linear.simd_lt(f32x8::splat(token, LINEAR_THRESHOLD));
    f32x8::blend(mask, linear_result, power_result)
}

/// Convert 8 linear f32 values to sRGB u8.
///
/// Compiled with `#[target_feature(enable = "avx2,fma")]` for optimal codegen.
#[arcane]
pub fn linear_to_srgb_u8_x8(token: Avx2FmaToken, linear: f32x8) -> [u8; 8] {
    let srgb = linear_to_srgb_x8(token, linear);
    let scaled = srgb * f32x8::splat(token, 255.0) + f32x8::splat(token, 0.5);
    let arr = scaled.to_array();
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

/// Convert 8 gamma-encoded f32 values to linear.
///
/// Compiled with `#[target_feature(enable = "avx2,fma")]` for optimal codegen.
#[arcane]
pub fn gamma_to_linear_x8(token: Avx2FmaToken, encoded: f32x8, gamma: f32) -> f32x8 {
    let zero = f32x8::splat(token, 0.0);
    let one = f32x8::splat(token, 1.0);
    let encoded = encoded.clamp(zero, one);
    encoded.pow_midp(gamma)
}

/// Convert 8 linear f32 values to gamma-encoded.
///
/// Compiled with `#[target_feature(enable = "avx2,fma")]` for optimal codegen.
#[arcane]
pub fn linear_to_gamma_x8(token: Avx2FmaToken, linear: f32x8, gamma: f32) -> f32x8 {
    let zero = f32x8::splat(token, 0.0);
    let one = f32x8::splat(token, 1.0);
    let linear = linear.clamp(zero, one);
    linear.pow_midp(1.0 / gamma)
}

// ============================================================================
// Slice Functions - process entire slices with zero dispatch overhead
// ============================================================================

/// Convert sRGB f32 values to linear in-place.
///
/// Processes 8 values at a time using SIMD with AVX2+FMA enabled.
/// The token parameter proves features are available at compile time.
#[arcane]
pub fn srgb_to_linear_slice(token: Avx2FmaToken, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let v = f32x8::from_array(token, *chunk);
        let result = srgb_to_linear_x8(token, v);
        result.store(chunk);
    }

    for v in remainder {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

/// Convert linear f32 values to sRGB in-place.
///
/// Processes 8 values at a time using SIMD with AVX2+FMA enabled.
/// The token parameter proves features are available at compile time.
#[arcane]
pub fn linear_to_srgb_slice(token: Avx2FmaToken, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let v = f32x8::from_array(token, *chunk);
        let result = linear_to_srgb_x8(token, v);
        result.store(chunk);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_srgb(*v);
    }
}

/// Convert sRGB u8 values to linear f32.
///
/// Uses a precomputed LUT for each u8 value.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
#[inline]
pub fn srgb_u8_to_linear_slice(_token: Avx2FmaToken, input: &[u8], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());

    // LUT-based, no SIMD needed - just prove we have the token
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = crate::simd::srgb_u8_to_linear(*inp);
    }
}

/// Convert linear f32 values to sRGB u8.
///
/// Processes 8 values at a time using SIMD with AVX2+FMA enabled.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
#[arcane]
pub fn linear_to_srgb_u8_slice(token: Avx2FmaToken, input: &[f32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());

    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = linear_to_srgb_u8_x8(token, f32x8::from_array(token, *inp));
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        let srgb = crate::scalar::linear_to_srgb(*inp);
        *out = (srgb * 255.0 + 0.5) as u8;
    }
}

/// Convert gamma-encoded f32 values to linear in-place.
///
/// Processes 8 values at a time using SIMD with AVX2+FMA enabled.
#[arcane]
pub fn gamma_to_linear_slice(token: Avx2FmaToken, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let v = f32x8::from_array(token, *chunk);
        let result = gamma_to_linear_x8(token, v, gamma);
        result.store(chunk);
    }

    for v in remainder {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

/// Convert linear f32 values to gamma-encoded in-place.
///
/// Processes 8 values at a time using SIMD with AVX2+FMA enabled.
#[arcane]
pub fn linear_to_gamma_slice(token: Avx2FmaToken, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let v = f32x8::from_array(token, *chunk);
        let result = linear_to_gamma_x8(token, v, gamma);
        result.store(chunk);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_gamma(*v, gamma);
    }
}

// ============================================================================
// f32x8 Slice Functions (for pre-aligned SIMD data)
// ============================================================================

/// Convert sRGB f32x8 values to linear in-place.
///
/// For data already structured as `f32x8` slices.
#[arcane]
pub fn srgb_to_linear_x8_slice(token: Avx2FmaToken, values: &mut [f32x8]) {
    for v in values.iter_mut() {
        *v = srgb_to_linear_x8(token, *v);
    }
}

/// Convert linear f32x8 values to sRGB in-place.
///
/// For data already structured as `f32x8` slices.
#[arcane]
pub fn linear_to_srgb_x8_slice(token: Avx2FmaToken, values: &mut [f32x8]) {
    for v in values.iter_mut() {
        *v = linear_to_srgb_x8(token, *v);
    }
}

/// Convert gamma-encoded f32x8 values to linear in-place.
///
/// For data already structured as `f32x8` slices.
#[arcane]
pub fn gamma_to_linear_x8_slice(token: Avx2FmaToken, values: &mut [f32x8], gamma: f32) {
    for v in values.iter_mut() {
        *v = gamma_to_linear_x8(token, *v, gamma);
    }
}

/// Convert linear f32x8 values to gamma-encoded in-place.
///
/// For data already structured as `f32x8` slices.
#[arcane]
pub fn linear_to_gamma_x8_slice(token: Avx2FmaToken, values: &mut [f32x8], gamma: f32) {
    for v in values.iter_mut() {
        *v = linear_to_gamma_x8(token, *v, gamma);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    fn get_token() -> Option<Avx2FmaToken> {
        Avx2FmaToken::try_new()
    }

    #[test]
    fn test_srgb_to_linear_x8_with_token() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
            return;
        };

        let input = [0.0f32, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9, 0.04];
        let result = srgb_to_linear_x8(token, f32x8::from_array(token, input));
        let result_arr = result.to_array();

        for (i, &inp) in input.iter().enumerate() {
            let expected = crate::scalar::srgb_to_linear(inp);
            assert!(
                (result_arr[i] - expected).abs() < 1e-5,
                "mismatch at {}: got {}, expected {}",
                i,
                result_arr[i],
                expected
            );
        }
    }

    #[test]
    fn test_srgb_to_linear_slice_with_token() {
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
    fn test_roundtrip_with_token() {
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
}
