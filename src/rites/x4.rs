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

// sRGB transfer function constants (C0-continuous, moxcms-derived)
const SRGB_LINEAR_THRESHOLD: f32 = 0.039_293_37;
const LINEAR_THRESHOLD: f32 = 0.003_041_282_6;
const LINEAR_SCALE: f32 = 1.0 / 12.92;
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

// linear→sRGB degree-15 Chebyshev polynomial via sqrt transform (Estrin's scheme)
const L2S_INV_HW: f32 = 2.116_733_3;
const L2S_BIAS: f32 = -1.116_733_2;
const L2S_C0: f32 = 5.641_828e-1;
const L2S_C1: f32 = 4.620_569_3e-1;
const L2S_C2: f32 = -3.450_065e-2;
const L2S_C3: f32 = 1.202_464_2e-2;
const L2S_C4: f32 = -5.398_721e-3;
const L2S_C5: f32 = 2.946_610_3e-3;
const L2S_C6: f32 = -5.274_399_6e-3;
const L2S_C7: f32 = 4.055_202e-3;
const L2S_C8: f32 = 1.062_489_9e-2;
const L2S_C9: f32 = -9.012_202e-3;
const L2S_C10: f32 = -2.186_026_6e-2;
const L2S_C11: f32 = 1.824_478_4e-2;
const L2S_C12: f32 = 1.958_387_2e-2;
const L2S_C13: f32 = -1.638_288e-2;
const L2S_C14: f32 = -7.710_282_7e-3;
const L2S_C15: f32 = 6.419_743e-3;

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
    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let srgb = mt_f32x4::from_array(token, srgb).max(zero).min(one);

    let linear_result = srgb * mt_f32x4::splat(token, LINEAR_SCALE);

    // Degree-11 Chebyshev polynomial (Estrin evaluation)
    let u = srgb.mul_add(
        mt_f32x4::splat(token, S2L_INV_HW),
        mt_f32x4::splat(token, S2L_BIAS),
    );
    let u2 = u * u;
    let u4 = u2 * u2;
    let u_8 = u4 * u4;
    let p01 = mt_f32x4::splat(token, S2L_C1).mul_add(u, mt_f32x4::splat(token, S2L_C0));
    let p23 = mt_f32x4::splat(token, S2L_C3).mul_add(u, mt_f32x4::splat(token, S2L_C2));
    let p45 = mt_f32x4::splat(token, S2L_C5).mul_add(u, mt_f32x4::splat(token, S2L_C4));
    let p67 = mt_f32x4::splat(token, S2L_C7).mul_add(u, mt_f32x4::splat(token, S2L_C6));
    let p89 = mt_f32x4::splat(token, S2L_C9).mul_add(u, mt_f32x4::splat(token, S2L_C8));
    let pab = mt_f32x4::splat(token, S2L_C11).mul_add(u, mt_f32x4::splat(token, S2L_C10));
    let p0123 = p23.mul_add(u2, p01);
    let p4567 = p67.mul_add(u2, p45);
    let p8_11 = pab.mul_add(u2, p89);
    let p0_7 = p4567.mul_add(u4, p0123);
    let power_result = p8_11.mul_add(u_8, p0_7);

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
    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let linear = mt_f32x4::from_array(token, linear).max(zero).min(one);

    let linear_result = linear * mt_f32x4::splat(token, TWELVE_92);

    // sqrt transform + degree-15 Chebyshev polynomial (Estrin evaluation)
    let s = linear.sqrt();
    let u = s.mul_add(
        mt_f32x4::splat(token, L2S_INV_HW),
        mt_f32x4::splat(token, L2S_BIAS),
    );
    let u2 = u * u;
    let u4 = u2 * u2;
    let u_8 = u4 * u4;
    let p01 = mt_f32x4::splat(token, L2S_C1).mul_add(u, mt_f32x4::splat(token, L2S_C0));
    let p23 = mt_f32x4::splat(token, L2S_C3).mul_add(u, mt_f32x4::splat(token, L2S_C2));
    let p45 = mt_f32x4::splat(token, L2S_C5).mul_add(u, mt_f32x4::splat(token, L2S_C4));
    let p67 = mt_f32x4::splat(token, L2S_C7).mul_add(u, mt_f32x4::splat(token, L2S_C6));
    let p89 = mt_f32x4::splat(token, L2S_C9).mul_add(u, mt_f32x4::splat(token, L2S_C8));
    let pab = mt_f32x4::splat(token, L2S_C11).mul_add(u, mt_f32x4::splat(token, L2S_C10));
    let pcd = mt_f32x4::splat(token, L2S_C13).mul_add(u, mt_f32x4::splat(token, L2S_C12));
    let pef = mt_f32x4::splat(token, L2S_C15).mul_add(u, mt_f32x4::splat(token, L2S_C14));
    let p0123 = p23.mul_add(u2, p01);
    let p4567 = p67.mul_add(u2, p45);
    let p89ab = pab.mul_add(u2, p89);
    let pcdef = pef.mul_add(u2, pcd);
    let p0_7 = p4567.mul_add(u4, p0123);
    let p8_f = pcdef.mul_add(u4, p89ab);
    let power_result = p8_f.mul_add(u_8, p0_7);

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
    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let srgb = mt_f32x4::from_array(token, srgb).max(zero).min(one);

    let linear_result = srgb * mt_f32x4::splat(token, LINEAR_SCALE);

    // Degree-11 Chebyshev polynomial (Estrin evaluation)
    let u = srgb.mul_add(
        mt_f32x4::splat(token, S2L_INV_HW),
        mt_f32x4::splat(token, S2L_BIAS),
    );
    let u2 = u * u;
    let u4 = u2 * u2;
    let u_8 = u4 * u4;
    let p01 = mt_f32x4::splat(token, S2L_C1).mul_add(u, mt_f32x4::splat(token, S2L_C0));
    let p23 = mt_f32x4::splat(token, S2L_C3).mul_add(u, mt_f32x4::splat(token, S2L_C2));
    let p45 = mt_f32x4::splat(token, S2L_C5).mul_add(u, mt_f32x4::splat(token, S2L_C4));
    let p67 = mt_f32x4::splat(token, S2L_C7).mul_add(u, mt_f32x4::splat(token, S2L_C6));
    let p89 = mt_f32x4::splat(token, S2L_C9).mul_add(u, mt_f32x4::splat(token, S2L_C8));
    let pab = mt_f32x4::splat(token, S2L_C11).mul_add(u, mt_f32x4::splat(token, S2L_C10));
    let p0123 = p23.mul_add(u2, p01);
    let p4567 = p67.mul_add(u2, p45);
    let p8_11 = pab.mul_add(u2, p89);
    let p0_7 = p4567.mul_add(u4, p0123);
    let power_result = p8_11.mul_add(u_8, p0_7);

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
    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let linear = mt_f32x4::from_array(token, linear).max(zero).min(one);

    let linear_result = linear * mt_f32x4::splat(token, TWELVE_92);

    // sqrt transform + degree-15 Chebyshev polynomial (Estrin evaluation)
    let s = linear.sqrt();
    let u = s.mul_add(
        mt_f32x4::splat(token, L2S_INV_HW),
        mt_f32x4::splat(token, L2S_BIAS),
    );
    let u2 = u * u;
    let u4 = u2 * u2;
    let u_8 = u4 * u4;
    let p01 = mt_f32x4::splat(token, L2S_C1).mul_add(u, mt_f32x4::splat(token, L2S_C0));
    let p23 = mt_f32x4::splat(token, L2S_C3).mul_add(u, mt_f32x4::splat(token, L2S_C2));
    let p45 = mt_f32x4::splat(token, L2S_C5).mul_add(u, mt_f32x4::splat(token, L2S_C4));
    let p67 = mt_f32x4::splat(token, L2S_C7).mul_add(u, mt_f32x4::splat(token, L2S_C6));
    let p89 = mt_f32x4::splat(token, L2S_C9).mul_add(u, mt_f32x4::splat(token, L2S_C8));
    let pab = mt_f32x4::splat(token, L2S_C11).mul_add(u, mt_f32x4::splat(token, L2S_C10));
    let pcd = mt_f32x4::splat(token, L2S_C13).mul_add(u, mt_f32x4::splat(token, L2S_C12));
    let pef = mt_f32x4::splat(token, L2S_C15).mul_add(u, mt_f32x4::splat(token, L2S_C14));
    let p0123 = p23.mul_add(u2, p01);
    let p4567 = p67.mul_add(u2, p45);
    let p89ab = pab.mul_add(u2, p89);
    let pcdef = pef.mul_add(u2, pcd);
    let p0_7 = p4567.mul_add(u4, p0123);
    let p8_f = pcdef.mul_add(u4, p89ab);
    let power_result = p8_f.mul_add(u_8, p0_7);

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
