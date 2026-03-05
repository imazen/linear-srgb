//! 4×f32 `#[rite]` functions (NEON on AArch64, SIMD128 on WebAssembly).
//!
//! All functions use `[f32; 4]` at the boundary — zero-cost transmute to/from
//! the underlying SIMD register. No `magetypes` types in the public API.
//!
//! Call these from inside your own `#[arcane]` function with a matching token.
//! They inline fully — no dispatch, no function-pointer indirection.
//!
//! # Suffix convention
//!
//! - `_neon` — requires [`Arm64`] (AArch64 NEON)
//! - `_wasm128` — requires [`Wasm128Token`] (WebAssembly SIMD128)

use archmage::rite;

#[cfg(target_arch = "aarch64")]
pub use archmage::Arm64;

#[cfg(target_arch = "wasm32")]
pub use archmage::Wasm128Token;

use magetypes::simd::f32x4 as mt_f32x4;

// sRGB transfer function constants (IEC 61966-2-1, matching rational polynomial)
const SRGB_LINEAR_THRESHOLD: f32 = 0.04045;
const LINEAR_THRESHOLD: f32 = 0.003_130_8;
const LINEAR_SCALE: f32 = 1.0 / 12.92;
const TWELVE_92: f32 = 12.92;

// ============================================================================
// AArch64 NEON — 4×f32 with Arm64 token
// ============================================================================

/// Convert 4 sRGB values to linear. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features (e.g. inside
/// an `#[arcane]` function taking `Arm64`). The token proves CPU support.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn srgb_to_linear_neon(token: Arm64, srgb: [f32; 4]) -> [f32; 4] {
    use crate::rational_poly::{S2L_P, S2L_Q};

    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let srgb = mt_f32x4::from_array(token, srgb).max(zero).min(one);

    let linear_result = srgb * mt_f32x4::splat(token, LINEAR_SCALE);

    // Rational polynomial P(x)/Q(x) via Horner's method
    let x = srgb;
    let yp = mt_f32x4::splat(token, S2L_P[4]).mul_add(x, mt_f32x4::splat(token, S2L_P[3]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[2]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[1]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[0]));

    let yq = mt_f32x4::splat(token, S2L_Q[4]).mul_add(x, mt_f32x4::splat(token, S2L_Q[3]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[2]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[1]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[0]));

    let power_result = yp / yq;

    let mask = srgb.simd_lt(mt_f32x4::splat(token, SRGB_LINEAR_THRESHOLD));
    mt_f32x4::blend(mask, linear_result, power_result).to_array()
}

/// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn linear_to_srgb_neon(token: Arm64, linear: [f32; 4]) -> [f32; 4] {
    use crate::rational_poly::{L2S_P, L2S_Q};

    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let linear = mt_f32x4::from_array(token, linear).max(zero).min(one);

    let linear_result = linear * mt_f32x4::splat(token, TWELVE_92);

    // sqrt transform + rational polynomial P(√x)/Q(√x) via Horner's method
    let x = linear.sqrt();
    let yp = mt_f32x4::splat(token, L2S_P[4]).mul_add(x, mt_f32x4::splat(token, L2S_P[3]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[2]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[1]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[0]));

    let yq = mt_f32x4::splat(token, L2S_Q[4]).mul_add(x, mt_f32x4::splat(token, L2S_Q[3]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[2]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[1]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[0]));

    let power_result = yp / yq;

    let mask = linear.simd_lt(mt_f32x4::splat(token, LINEAR_THRESHOLD));
    mt_f32x4::blend(mask, linear_result, power_result).to_array()
}

/// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn gamma_to_linear_neon(token: Arm64, encoded: [f32; 4], gamma: f32) -> [f32; 4] {
    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let encoded = mt_f32x4::from_array(token, encoded).max(zero).min(one);
    encoded.pow_midp(gamma).to_array()
}

/// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn linear_to_gamma_neon(token: Arm64, linear: [f32; 4], gamma: f32) -> [f32; 4] {
    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let linear = mt_f32x4::from_array(token, linear).max(zero).min(one);
    linear.pow_midp(1.0 / gamma).to_array()
}

/// Convert sRGB f32 values to linear in-place using 4-wide NEON SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn srgb_to_linear_slice_neon(token: Arm64, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = srgb_to_linear_neon(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

/// Convert linear f32 values to sRGB in-place using 4-wide NEON SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn linear_to_srgb_slice_neon(token: Arm64, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = linear_to_srgb_neon(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_srgb(*v);
    }
}

/// Convert gamma-encoded f32 values to linear in-place using 4-wide NEON SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn gamma_to_linear_slice_neon(token: Arm64, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = gamma_to_linear_neon(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

/// Convert linear f32 values to gamma-encoded in-place using 4-wide NEON SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn linear_to_gamma_slice_neon(token: Arm64, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = linear_to_gamma_neon(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_gamma(*v, gamma);
    }
}

// ============================================================================
// WebAssembly SIMD128 — 4×f32 with Wasm128Token
// ============================================================================

/// Convert 4 sRGB values to linear. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features (e.g. inside
/// an `#[arcane]` function taking `Wasm128Token`).
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn srgb_to_linear_wasm128(token: Wasm128Token, srgb: [f32; 4]) -> [f32; 4] {
    use crate::rational_poly::{S2L_P, S2L_Q};

    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let srgb = mt_f32x4::from_array(token, srgb).max(zero).min(one);

    let linear_result = srgb * mt_f32x4::splat(token, LINEAR_SCALE);

    // Rational polynomial P(x)/Q(x) via Horner's method
    let x = srgb;
    let yp = mt_f32x4::splat(token, S2L_P[4]).mul_add(x, mt_f32x4::splat(token, S2L_P[3]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[2]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[1]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[0]));

    let yq = mt_f32x4::splat(token, S2L_Q[4]).mul_add(x, mt_f32x4::splat(token, S2L_Q[3]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[2]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[1]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[0]));

    let power_result = yp / yq;

    let mask = srgb.simd_lt(mt_f32x4::splat(token, SRGB_LINEAR_THRESHOLD));
    mt_f32x4::blend(mask, linear_result, power_result).to_array()
}

/// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn linear_to_srgb_wasm128(token: Wasm128Token, linear: [f32; 4]) -> [f32; 4] {
    use crate::rational_poly::{L2S_P, L2S_Q};

    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let linear = mt_f32x4::from_array(token, linear).max(zero).min(one);

    let linear_result = linear * mt_f32x4::splat(token, TWELVE_92);

    // sqrt transform + rational polynomial P(√x)/Q(√x) via Horner's method
    let x = linear.sqrt();
    let yp = mt_f32x4::splat(token, L2S_P[4]).mul_add(x, mt_f32x4::splat(token, L2S_P[3]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[2]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[1]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[0]));

    let yq = mt_f32x4::splat(token, L2S_Q[4]).mul_add(x, mt_f32x4::splat(token, L2S_Q[3]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[2]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[1]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[0]));

    let power_result = yp / yq;

    let mask = linear.simd_lt(mt_f32x4::splat(token, LINEAR_THRESHOLD));
    mt_f32x4::blend(mask, linear_result, power_result).to_array()
}

/// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn gamma_to_linear_wasm128(token: Wasm128Token, encoded: [f32; 4], gamma: f32) -> [f32; 4] {
    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let encoded = mt_f32x4::from_array(token, encoded).max(zero).min(one);
    encoded.pow_midp(gamma).to_array()
}

/// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn linear_to_gamma_wasm128(token: Wasm128Token, linear: [f32; 4], gamma: f32) -> [f32; 4] {
    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let linear = mt_f32x4::from_array(token, linear).max(zero).min(one);
    linear.pow_midp(1.0 / gamma).to_array()
}

/// Convert sRGB f32 values to linear in-place using 4-wide WASM SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn srgb_to_linear_slice_wasm128(token: Wasm128Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = srgb_to_linear_wasm128(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

/// Convert linear f32 values to sRGB in-place using 4-wide WASM SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn linear_to_srgb_slice_wasm128(token: Wasm128Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = linear_to_srgb_wasm128(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_srgb(*v);
    }
}

/// Convert gamma-encoded f32 values to linear in-place using 4-wide WASM SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn gamma_to_linear_slice_wasm128(token: Wasm128Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = gamma_to_linear_wasm128(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

/// Convert linear f32 values to gamma-encoded in-place using 4-wide WASM SIMD.
///
/// # Safety
///
/// Safe when called from a context with matching target features.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn linear_to_gamma_slice_wasm128(token: Wasm128Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = linear_to_gamma_wasm128(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_gamma(*v, gamma);
    }
}

// ============================================================================
// Tests (AArch64 only — WASM tests require wasm runtime)
// ============================================================================

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;
    use archmage::SimdToken;

    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    fn get_token() -> Option<Arm64> {
        Arm64::try_new()
    }

    #[archmage::arcane]
    fn call_srgb_to_linear(token: Arm64, input: [f32; 4]) -> [f32; 4] {
        srgb_to_linear_neon(token, input)
    }

    #[archmage::arcane]
    fn call_linear_to_srgb(token: Arm64, input: [f32; 4]) -> [f32; 4] {
        linear_to_srgb_neon(token, input)
    }

    #[archmage::arcane]
    fn call_srgb_to_linear_slice(token: Arm64, values: &mut [f32]) {
        srgb_to_linear_slice_neon(token, values);
    }

    #[archmage::arcane]
    fn call_linear_to_srgb_slice(token: Arm64, values: &mut [f32]) {
        linear_to_srgb_slice_neon(token, values);
    }

    #[test]
    fn test_x4_srgb_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: NEON not available");
            return;
        };

        let input = [0.0, 0.3, 0.7, 1.0];
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
    fn test_x4_matches_scalar() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: NEON not available");
            return;
        };

        let input = [0.0, 0.3, 0.7, 1.0];
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
            eprintln!("Skipping test: NEON not available");
            return;
        };

        let mut values: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
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
