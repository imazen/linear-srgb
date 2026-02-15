//! 8×f32 `#[rite]` functions (AVX2+FMA on x86-64).
//!
//! All functions use `[f32; 8]` at the boundary — zero-cost transmute to/from
//! the underlying SIMD register. No `magetypes` types in the public API.
//!
//! Call these from inside your own `#[arcane]` function with a matching token.
//! They inline fully — no dispatch, no function-pointer indirection.

use archmage::rite;

pub use archmage::Desktop64;

use magetypes::simd::f32x8 as mt_f32x8;

// sRGB transfer function constants (C0-continuous, moxcms-derived)
const SRGB_LINEAR_THRESHOLD: f32 = 0.039_293_37;
const LINEAR_THRESHOLD: f32 = 0.003_041_282_6;
const LINEAR_SCALE: f32 = 1.0 / 12.92;
const SRGB_OFFSET: f32 = 0.055_010_72;
const SRGB_SCALE: f32 = 1.055_010_7;
const INV_SRGB_SCALE: f32 = 1.0 / 1.055_010_7;
const TWELVE_92: f32 = 12.92;

// ============================================================================
// x8 functions — operate on [f32; 8]
// ============================================================================

/// Convert 8 sRGB values to linear. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features (e.g. inside
/// an `#[arcane]` function taking `Desktop64`). The token proves CPU support.
#[rite]
pub fn srgb_to_linear_v3(token: Desktop64, srgb: [f32; 8]) -> [f32; 8] {
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let srgb = mt_f32x8::from_array(token, srgb).max(zero).min(one);

    let linear_result = srgb * mt_f32x8::splat(token, LINEAR_SCALE);
    let normalized =
        (srgb + mt_f32x8::splat(token, SRGB_OFFSET)) * mt_f32x8::splat(token, INV_SRGB_SCALE);
    let power_result = normalized.pow_midp(2.4);

    let mask = srgb.simd_lt(mt_f32x8::splat(token, SRGB_LINEAR_THRESHOLD));
    mt_f32x8::blend(mask, linear_result, power_result).to_array()
}

/// Convert 8 linear values to sRGB. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features (e.g. inside
/// an `#[arcane]` function taking `Desktop64`). The token proves CPU support.
#[rite]
pub fn linear_to_srgb_v3(token: Desktop64, linear: [f32; 8]) -> [f32; 8] {
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let linear = mt_f32x8::from_array(token, linear).max(zero).min(one);

    let linear_result = linear * mt_f32x8::splat(token, TWELVE_92);
    let power_result = mt_f32x8::splat(token, SRGB_SCALE) * linear.pow_midp(1.0 / 2.4)
        - mt_f32x8::splat(token, SRGB_OFFSET);

    let mask = linear.simd_lt(mt_f32x8::splat(token, LINEAR_THRESHOLD));
    mt_f32x8::blend(mask, linear_result, power_result).to_array()
}

/// Convert 8 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features (e.g. inside
/// an `#[arcane]` function taking `Desktop64`). The token proves CPU support.
#[rite]
pub fn gamma_to_linear_v3(token: Desktop64, encoded: [f32; 8], gamma: f32) -> [f32; 8] {
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let encoded = mt_f32x8::from_array(token, encoded).max(zero).min(one);
    encoded.pow_midp(gamma).to_array()
}

/// Convert 8 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features (e.g. inside
/// an `#[arcane]` function taking `Desktop64`). The token proves CPU support.
#[rite]
pub fn linear_to_gamma_v3(token: Desktop64, linear: [f32; 8], gamma: f32) -> [f32; 8] {
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let linear = mt_f32x8::from_array(token, linear).max(zero).min(one);
    linear.pow_midp(1.0 / gamma).to_array()
}

// ============================================================================
// x8 LUT functions — linear f32 → sRGB u8
// ============================================================================

/// Convert 8 linear f32 values to sRGB u8 via LUT. Input clamped to \[0, 1\].
///
/// Uses a 4096-entry const LUT with bitmask indexing (`& 0xFFF`) for
/// provably safe bounds. SIMD accelerates the clamp and scale; lookups
/// are scalar from an L1-resident 4KB table.
///
/// # Safety
///
/// Safe when called from a context with matching target features (e.g. inside
/// an `#[arcane]` function taking `Desktop64`). The token proves CPU support.
#[rite]
pub fn linear_to_srgb_u8_v3(token: Desktop64, linear: [f32; 8]) -> [u8; 8] {
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let linear = mt_f32x8::from_array(token, linear).max(zero).min(one);
    let scaled = linear * mt_f32x8::splat(token, 4095.0) + mt_f32x8::splat(token, 0.5);
    let arr = scaled.to_array();
    let lut = &crate::const_luts::LINEAR_TO_SRGB_U8;
    [
        lut[arr[0] as usize & 0xFFF],
        lut[arr[1] as usize & 0xFFF],
        lut[arr[2] as usize & 0xFFF],
        lut[arr[3] as usize & 0xFFF],
        lut[arr[4] as usize & 0xFFF],
        lut[arr[5] as usize & 0xFFF],
        lut[arr[6] as usize & 0xFFF],
        lut[arr[7] as usize & 0xFFF],
    ]
}

// ============================================================================
// Slice functions — process &mut [f32] with x8 chunking
// ============================================================================

/// Convert sRGB f32 values to linear in-place using 8-wide SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[rite]
pub fn srgb_to_linear_slice_v3(token: Desktop64, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        *chunk = srgb_to_linear_v3(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

/// Convert linear f32 values to sRGB in-place using 8-wide SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[rite]
pub fn linear_to_srgb_slice_v3(token: Desktop64, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        *chunk = linear_to_srgb_v3(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_srgb(*v);
    }
}

/// Convert gamma-encoded f32 values to linear in-place using 8-wide SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[rite]
pub fn gamma_to_linear_slice_v3(token: Desktop64, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        *chunk = gamma_to_linear_v3(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

/// Convert linear f32 values to gamma-encoded in-place using 8-wide SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[rite]
pub fn linear_to_gamma_slice_v3(token: Desktop64, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        *chunk = linear_to_gamma_v3(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_gamma(*v, gamma);
    }
}

/// Convert linear f32 values to sRGB u8 using 8-wide SIMD + LUT.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[rite]
pub fn linear_to_srgb_u8_slice_v3(token: Desktop64, input: &[f32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());
    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = linear_to_srgb_u8_v3(token, *inp);
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        *out = crate::scalar::linear_to_srgb_u8(*inp);
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

    fn get_token() -> Option<Desktop64> {
        Desktop64::try_new()
    }

    // We need an #[arcane] wrapper to safely call #[rite] functions in tests.
    #[archmage::arcane]
    fn call_srgb_to_linear(token: Desktop64, input: [f32; 8]) -> [f32; 8] {
        srgb_to_linear_v3(token, input)
    }

    #[archmage::arcane]
    fn call_linear_to_srgb(token: Desktop64, input: [f32; 8]) -> [f32; 8] {
        linear_to_srgb_v3(token, input)
    }

    #[archmage::arcane]
    fn call_srgb_to_linear_slice(token: Desktop64, values: &mut [f32]) {
        srgb_to_linear_slice_v3(token, values);
    }

    #[archmage::arcane]
    fn call_linear_to_srgb_slice(token: Desktop64, values: &mut [f32]) {
        linear_to_srgb_slice_v3(token, values);
    }

    #[archmage::arcane]
    fn call_linear_to_srgb_u8(token: Desktop64, input: [f32; 8]) -> [u8; 8] {
        linear_to_srgb_u8_v3(token, input)
    }

    #[test]
    fn test_x8_linear_to_srgb_u8() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
            return;
        };

        let input = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
        let result = call_linear_to_srgb_u8(token, input);

        for (i, (&got, &inp)) in result.iter().zip(input.iter()).enumerate() {
            let expected = crate::scalar::linear_to_srgb_u8(inp);
            assert!(
                (got as i32 - expected as i32).abs() <= 1,
                "u8 mismatch at {}: got {}, expected {} (input={})",
                i,
                got,
                expected,
                inp
            );
        }
    }

    #[test]
    fn test_x8_srgb_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
            return;
        };

        let input = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
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
    fn test_x8_matches_scalar() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
            return;
        };

        let input = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
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
    fn test_slice_matches_scalar() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
            return;
        };

        let mut values: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let expected: Vec<f32> = values
            .iter()
            .map(|&v| crate::scalar::srgb_to_linear(v))
            .collect();

        call_srgb_to_linear_slice(token, &mut values);

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
    fn test_slice_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
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
