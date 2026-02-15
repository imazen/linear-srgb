//! 16×f32 `#[rite]` functions (AVX-512 on x86-64).
//!
//! All functions use `[f32; 16]` at the boundary — zero-cost transmute to/from
//! the underlying `__m512` register. No `magetypes` types in the public API.
//!
//! Call these from inside your own `#[arcane]` function with a matching token.
//! They inline fully — no dispatch, no function-pointer indirection.

use archmage::rite;

pub use archmage::Server64;

use magetypes::simd::f32x16 as mt_f32x16;

// sRGB transfer function constants (C0-continuous, moxcms-derived)
const SRGB_LINEAR_THRESHOLD: f32 = 0.039_293_37;
const LINEAR_THRESHOLD: f32 = 0.003_041_282_6;
const LINEAR_SCALE: f32 = 1.0 / 12.92;
const SRGB_OFFSET: f32 = 0.055_010_72;
const SRGB_SCALE: f32 = 1.055_010_7;
const INV_SRGB_SCALE: f32 = 1.0 / 1.055_010_7;
const TWELVE_92: f32 = 12.92;

// ============================================================================
// x16 functions — operate on [f32; 16]
// ============================================================================

/// Convert 16 sRGB values to linear. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features (e.g. inside
/// an `#[arcane]` function taking `Server64`). The token proves CPU support.
#[rite]
pub fn srgb_to_linear_v4(token: Server64, srgb: [f32; 16]) -> [f32; 16] {
    let zero = mt_f32x16::zero(token);
    let one = mt_f32x16::splat(token, 1.0);
    let srgb = mt_f32x16::from_array(token, srgb).max(zero).min(one);

    let linear_result = srgb * mt_f32x16::splat(token, LINEAR_SCALE);
    let normalized =
        (srgb + mt_f32x16::splat(token, SRGB_OFFSET)) * mt_f32x16::splat(token, INV_SRGB_SCALE);
    let power_result = normalized.pow_midp(2.4);

    let mask = srgb.simd_lt(mt_f32x16::splat(token, SRGB_LINEAR_THRESHOLD));
    mt_f32x16::blend(mask, linear_result, power_result).to_array()
}

/// Convert 16 linear values to sRGB. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features (e.g. inside
/// an `#[arcane]` function taking `Server64`). The token proves CPU support.
#[rite]
pub fn linear_to_srgb_v4(token: Server64, linear: [f32; 16]) -> [f32; 16] {
    let zero = mt_f32x16::zero(token);
    let one = mt_f32x16::splat(token, 1.0);
    let linear = mt_f32x16::from_array(token, linear).max(zero).min(one);

    let linear_result = linear * mt_f32x16::splat(token, TWELVE_92);
    let power_result = mt_f32x16::splat(token, SRGB_SCALE) * linear.pow_midp(1.0 / 2.4)
        - mt_f32x16::splat(token, SRGB_OFFSET);

    let mask = linear.simd_lt(mt_f32x16::splat(token, LINEAR_THRESHOLD));
    mt_f32x16::blend(mask, linear_result, power_result).to_array()
}

/// Convert 16 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features (e.g. inside
/// an `#[arcane]` function taking `Server64`). The token proves CPU support.
#[rite]
pub fn gamma_to_linear_v4(token: Server64, encoded: [f32; 16], gamma: f32) -> [f32; 16] {
    let zero = mt_f32x16::zero(token);
    let one = mt_f32x16::splat(token, 1.0);
    let encoded = mt_f32x16::from_array(token, encoded).max(zero).min(one);
    encoded.pow_midp(gamma).to_array()
}

/// Convert 16 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features (e.g. inside
/// an `#[arcane]` function taking `Server64`). The token proves CPU support.
#[rite]
pub fn linear_to_gamma_v4(token: Server64, linear: [f32; 16], gamma: f32) -> [f32; 16] {
    let zero = mt_f32x16::zero(token);
    let one = mt_f32x16::splat(token, 1.0);
    let linear = mt_f32x16::from_array(token, linear).max(zero).min(one);
    linear.pow_midp(1.0 / gamma).to_array()
}

// ============================================================================
// Slice functions — process &mut [f32] with x16 chunking
// ============================================================================

/// Convert sRGB f32 values to linear in-place using 16-wide SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[rite]
pub fn srgb_to_linear_slice_v4(token: Server64, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<16>();

    for chunk in chunks {
        *chunk = srgb_to_linear_v4(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

/// Convert linear f32 values to sRGB in-place using 16-wide SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[rite]
pub fn linear_to_srgb_slice_v4(token: Server64, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<16>();

    for chunk in chunks {
        *chunk = linear_to_srgb_v4(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_srgb(*v);
    }
}

/// Convert gamma-encoded f32 values to linear in-place using 16-wide SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[rite]
pub fn gamma_to_linear_slice_v4(token: Server64, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<16>();

    for chunk in chunks {
        *chunk = gamma_to_linear_v4(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

/// Convert linear f32 values to gamma-encoded in-place using 16-wide SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[rite]
pub fn linear_to_gamma_slice_v4(token: Server64, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<16>();

    for chunk in chunks {
        *chunk = linear_to_gamma_v4(token, *chunk, gamma);
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

    fn get_token() -> Option<Server64> {
        Server64::try_new()
    }

    #[archmage::arcane]
    fn call_srgb_to_linear(token: Server64, input: [f32; 16]) -> [f32; 16] {
        srgb_to_linear_v4(token, input)
    }

    #[archmage::arcane]
    fn call_linear_to_srgb(token: Server64, input: [f32; 16]) -> [f32; 16] {
        linear_to_srgb_v4(token, input)
    }

    #[archmage::arcane]
    fn call_srgb_to_linear_slice(token: Server64, values: &mut [f32]) {
        srgb_to_linear_slice_v4(token, values);
    }

    #[archmage::arcane]
    fn call_linear_to_srgb_slice(token: Server64, values: &mut [f32]) {
        linear_to_srgb_slice_v4(token, values);
    }

    #[test]
    fn test_x16_srgb_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX-512 not available");
            return;
        };

        let input = [
            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
        ];
        let linear = call_srgb_to_linear(token, input);
        let roundtrip = call_linear_to_srgb(token, linear);

        for (i, (&orig, &rt)) in input.iter().zip(roundtrip.iter()).enumerate() {
            assert!(
                (orig - rt).abs() < 1e-4,
                "roundtrip failed at {}: {} -> {}",
                i,
                orig,
                rt
            );
        }
    }

    #[test]
    fn test_x16_matches_scalar() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX-512 not available");
            return;
        };

        let input = [
            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
        ];
        let result = call_srgb_to_linear(token, input);

        for (i, (&got, &inp)) in result.iter().zip(input.iter()).enumerate() {
            let expected = crate::scalar::srgb_to_linear(inp);
            assert!(
                (got - expected).abs() < 1e-5,
                "mismatch at {}: got {}, expected {}",
                i,
                got,
                expected
            );
        }
    }

    #[test]
    fn test_slice_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX-512 not available");
            return;
        };

        let mut values: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        let original = values.clone();

        call_srgb_to_linear_slice(token, &mut values);
        call_linear_to_srgb_slice(token, &mut values);

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
