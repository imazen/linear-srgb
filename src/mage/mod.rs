//! Token-based SIMD API with zero dispatch overhead.
//!
//! This module provides an alternative API using archmage tokens for users who
//! want to avoid per-call dispatch overhead. Obtain a token once at startup,
//! then pass it to all conversion functions.
//!
//! # Usage
//!
//! ```rust,ignore
//! use linear_srgb::mage::{self, X64V3Token, SimdToken};
//!
//! fn main() {
//!     // Detection once at startup
//!     let token = X64V3Token::try_new().expect("need AVX2+FMA");
//!
//!     let mut data = vec![0.5f32; 10000];
//!
//!     // Zero dispatch overhead - token proves features available
//!     mage::srgb_to_linear_slice(token, &mut data);
//! }
//! ```
//!
//! # Inside `#[multiversed]` Functions
//!
//! When writing your own multiversioned code, obtain a token at runtime:
//!
//! ```rust,ignore
//! use linear_srgb::mage::{self, X64V3Token, SimdToken};
//! use multiversed::multiversed;
//!
//! #[multiversed]
//! fn my_pipeline(data: &mut [f32]) {
//!     // Inside #[multiversed], the CPU features are already proven available
//!     // by the dispatch mechanism, but we still need a token for the API.
//!     // In production, pass the token from outside the hot loop.
//!     if let Some(token) = X64V3Token::try_new() {
//!         mage::srgb_to_linear_slice(token, data);
//!         // ... more processing with token ...
//!     }
//! }
//! ```
//!
//! # Comparison with Default API
//!
//! | Aspect | `default::` / `simd::` | `mage::` |
//! |--------|------------------------|----------|
//! | Dispatch | Per-call (amortized) | None (token proves features) |
//! | API | Simple, no tokens | Requires token threading |
//! | Best for | Casual use | Tight loops, performance-critical |

use wide::f32x8;

// Re-export core types from archmage
pub use archmage::{SimdToken, X64V3Token};

// Also export other common tokens users might want
#[cfg(target_arch = "x86_64")]
pub use archmage::{Avx2FmaToken, X64V2Token, X64V4Token};

#[cfg(target_arch = "aarch64")]
pub use archmage::NeonToken;

// Import the inline implementations
use crate::simd::{
    gamma_to_linear_x8_inline, linear_to_gamma_x8_inline, linear_to_srgb_u8_x8_inline,
    linear_to_srgb_x8_inline, srgb_to_linear_x8_inline, srgb_u8_to_linear,
};

// ============================================================================
// x8 Functions - process 8 values at once
// ============================================================================

/// Convert 8 sRGB f32 values to linear.
///
/// The token parameter proves the required CPU features are available,
/// eliminating runtime dispatch overhead.
///
/// Input values are clamped to \[0, 1\].
#[inline(always)]
pub fn srgb_to_linear_x8(_token: X64V3Token, srgb: f32x8) -> f32x8 {
    srgb_to_linear_x8_inline(srgb)
}

/// Convert 8 linear f32 values to sRGB.
///
/// The token parameter proves the required CPU features are available,
/// eliminating runtime dispatch overhead.
///
/// Input values are clamped to \[0, 1\].
#[inline(always)]
pub fn linear_to_srgb_x8(_token: X64V3Token, linear: f32x8) -> f32x8 {
    linear_to_srgb_x8_inline(linear)
}

/// Convert 8 linear f32 values to sRGB u8.
///
/// The token parameter proves the required CPU features are available,
/// eliminating runtime dispatch overhead.
#[inline(always)]
pub fn linear_to_srgb_u8_x8(_token: X64V3Token, linear: f32x8) -> [u8; 8] {
    linear_to_srgb_u8_x8_inline(linear)
}

/// Convert 8 gamma-encoded f32 values to linear.
///
/// The token parameter proves the required CPU features are available,
/// eliminating runtime dispatch overhead.
#[inline(always)]
pub fn gamma_to_linear_x8(_token: X64V3Token, encoded: f32x8, gamma: f32) -> f32x8 {
    gamma_to_linear_x8_inline(encoded, gamma)
}

/// Convert 8 linear f32 values to gamma-encoded.
///
/// The token parameter proves the required CPU features are available,
/// eliminating runtime dispatch overhead.
#[inline(always)]
pub fn linear_to_gamma_x8(_token: X64V3Token, linear: f32x8, gamma: f32) -> f32x8 {
    linear_to_gamma_x8_inline(linear, gamma)
}

// ============================================================================
// Slice Functions - process entire slices with zero dispatch overhead
// ============================================================================

/// Convert sRGB f32 values to linear in-place.
///
/// Processes 8 values at a time using SIMD, with scalar fallback for remainder.
/// The token parameter eliminates per-call dispatch overhead.
#[inline]
pub fn srgb_to_linear_slice(_token: X64V3Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let result = srgb_to_linear_x8_inline(f32x8::from(*chunk));
        *chunk = result.into();
    }

    for v in remainder {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

/// Convert linear f32 values to sRGB in-place.
///
/// Processes 8 values at a time using SIMD, with scalar fallback for remainder.
/// The token parameter eliminates per-call dispatch overhead.
#[inline]
pub fn linear_to_srgb_slice(_token: X64V3Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let result = linear_to_srgb_x8_inline(f32x8::from(*chunk));
        *chunk = result.into();
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
pub fn srgb_u8_to_linear_slice(_token: X64V3Token, input: &[u8], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());

    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = [
            srgb_u8_to_linear(inp[0]),
            srgb_u8_to_linear(inp[1]),
            srgb_u8_to_linear(inp[2]),
            srgb_u8_to_linear(inp[3]),
            srgb_u8_to_linear(inp[4]),
            srgb_u8_to_linear(inp[5]),
            srgb_u8_to_linear(inp[6]),
            srgb_u8_to_linear(inp[7]),
        ];
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        *out = srgb_u8_to_linear(*inp);
    }
}

/// Convert linear f32 values to sRGB u8.
///
/// Processes 8 values at a time using SIMD, with scalar fallback for remainder.
/// The token parameter eliminates per-call dispatch overhead.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
#[inline]
pub fn linear_to_srgb_u8_slice(_token: X64V3Token, input: &[f32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());

    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = linear_to_srgb_u8_x8_inline(f32x8::from(*inp));
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        let srgb = crate::scalar::linear_to_srgb(*inp);
        *out = (srgb * 255.0 + 0.5) as u8;
    }
}

/// Convert gamma-encoded f32 values to linear in-place.
///
/// Processes 8 values at a time using SIMD, with scalar fallback for remainder.
/// The token parameter eliminates per-call dispatch overhead.
#[inline]
pub fn gamma_to_linear_slice(_token: X64V3Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let result = gamma_to_linear_x8_inline(f32x8::from(*chunk), gamma);
        *chunk = result.into();
    }

    for v in remainder {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

/// Convert linear f32 values to gamma-encoded in-place.
///
/// Processes 8 values at a time using SIMD, with scalar fallback for remainder.
/// The token parameter eliminates per-call dispatch overhead.
#[inline]
pub fn linear_to_gamma_slice(_token: X64V3Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let result = linear_to_gamma_x8_inline(f32x8::from(*chunk), gamma);
        *chunk = result.into();
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
#[inline]
pub fn srgb_to_linear_x8_slice(_token: X64V3Token, values: &mut [f32x8]) {
    for v in values.iter_mut() {
        *v = srgb_to_linear_x8_inline(*v);
    }
}

/// Convert linear f32x8 values to sRGB in-place.
///
/// For data already structured as `f32x8` slices.
#[inline]
pub fn linear_to_srgb_x8_slice(_token: X64V3Token, values: &mut [f32x8]) {
    for v in values.iter_mut() {
        *v = linear_to_srgb_x8_inline(*v);
    }
}

/// Convert gamma-encoded f32x8 values to linear in-place.
///
/// For data already structured as `f32x8` slices.
#[inline]
pub fn gamma_to_linear_x8_slice(_token: X64V3Token, values: &mut [f32x8], gamma: f32) {
    for v in values.iter_mut() {
        *v = gamma_to_linear_x8_inline(*v, gamma);
    }
}

/// Convert linear f32x8 values to gamma-encoded in-place.
///
/// For data already structured as `f32x8` slices.
#[inline]
pub fn linear_to_gamma_x8_slice(_token: X64V3Token, values: &mut [f32x8], gamma: f32) {
    for v in values.iter_mut() {
        *v = linear_to_gamma_x8_inline(*v, gamma);
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

    fn get_token() -> Option<X64V3Token> {
        X64V3Token::try_new()
    }

    #[test]
    fn test_srgb_to_linear_x8_with_token() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: X64V3 not available");
            return;
        };

        let input = [0.0f32, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9, 0.04];
        let result = srgb_to_linear_x8(token, f32x8::from(input));
        let result_arr: [f32; 8] = result.into();

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
            eprintln!("Skipping test: X64V3 not available");
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
            eprintln!("Skipping test: X64V3 not available");
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
