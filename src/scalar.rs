//! Scalar (single-value) sRGB conversion functions.
//!
//! Implements the IEC 61966-2-1:1999 sRGB transfer functions with optimizations:
//! - Piecewise function avoids pow() for ~1.2% of values in the linear segment
//! - Early exit for out-of-range values (0 and 1) avoids transcendentals
//! - FMA instructions for the encoding formula

use crate::mlaf::fmla;
#[allow(unused_imports)]
use num_traits::Float; // provides powf/sqrt/mul_add via libm in no_std

// sRGB constants (IEC 61966-2-1:1999)
// These are the exact values for the continuous piecewise function.
// Precision is intentional - these come from the moxcms reference implementation.

/// Linear threshold for linearization: 12.92 * 0.0030412825601275209 ≈ 0.04045
#[allow(clippy::excessive_precision)]
const SRGB_LINEAR_THRESHOLD: f64 = 12.92 * 0.003_041_282_560_127_521;
pub(crate) const SRGB_LINEAR_THRESHOLD_F32: f32 = SRGB_LINEAR_THRESHOLD as f32;

/// Linear threshold for encoding (the inverse cutoff point)
#[allow(clippy::excessive_precision)]
const LINEAR_THRESHOLD: f64 = 0.003_041_282_560_127_521;
pub(crate) const LINEAR_THRESHOLD_F32: f32 = LINEAR_THRESHOLD as f32;

/// Linear scale factor (1/12.92)
const LINEAR_SCALE: f64 = 1.0 / 12.92;
pub(crate) const LINEAR_SCALE_F32: f32 = LINEAR_SCALE as f32;

/// sRGB encoding constants
const SRGB_A: f64 = 0.055_010_718_947_586_6;
const SRGB_A_F32: f32 = SRGB_A as f32;
const SRGB_A_PLUS_1: f64 = 1.055_010_718_947_586_6;
const SRGB_A_PLUS_1_F32: f32 = SRGB_A_PLUS_1 as f32;

/// Gamma exponent
const GAMMA: f64 = 2.4;
const INV_GAMMA: f64 = 1.0 / GAMMA;
const INV_GAMMA_F32: f32 = INV_GAMMA as f32;

/// Convert sRGB gamma-encoded value to linear light (f64).
///
/// Input: sRGB value in \[0, 1\]
/// Output: Linear light value in \[0, 1\]
///
/// **Clamps** inputs to \[0, 1\]. No extended-range f64 variant exists.
#[inline]
pub fn srgb_to_linear_f64(gamma: f64) -> f64 {
    if gamma < 0.0 {
        0.0
    } else if gamma < SRGB_LINEAR_THRESHOLD {
        // Linear segment (cheap multiply)
        gamma * LINEAR_SCALE
    } else if gamma < 1.0 {
        // Power segment
        ((gamma + SRGB_A) / SRGB_A_PLUS_1).powf(GAMMA)
    } else {
        1.0
    }
}

/// Convert sRGB gamma-encoded value to linear light (f32).
///
/// Input: sRGB value in \[0, 1\]
/// Output: Linear light value in \[0, 1\]
///
/// **Clamps** inputs to \[0, 1\]. For HDR/ICC workflows with out-of-range
/// values, use [`srgb_to_linear_extended`] instead.
#[inline]
pub fn srgb_to_linear(gamma: f32) -> f32 {
    if gamma < 0.0 {
        0.0
    } else if gamma < SRGB_LINEAR_THRESHOLD_F32 {
        gamma * LINEAR_SCALE_F32
    } else if gamma < 1.0 {
        ((gamma + SRGB_A_F32) / SRGB_A_PLUS_1_F32).powf(GAMMA as f32)
    } else {
        1.0
    }
}

/// Convert linear light value to sRGB gamma-encoded (f64).
///
/// Input: Linear light value in \[0, 1\]
/// Output: sRGB value in \[0, 1\]
///
/// **Clamps** inputs to \[0, 1\]. No extended-range f64 variant exists.
#[inline]
pub fn linear_to_srgb_f64(linear: f64) -> f64 {
    if linear < 0.0 {
        0.0
    } else if linear < LINEAR_THRESHOLD {
        // Linear segment (cheap multiply)
        linear * 12.92
    } else if linear < 1.0 {
        // Power segment with FMA: 1.055 * pow(linear, 1/2.4) - 0.055
        fmla(SRGB_A_PLUS_1, linear.powf(INV_GAMMA), -SRGB_A)
    } else {
        1.0
    }
}

/// Convert linear light value to sRGB gamma-encoded (f32).
///
/// Input: Linear light value in \[0, 1\]
/// Output: sRGB value in \[0, 1\]
///
/// **Clamps** inputs to \[0, 1\]. For HDR/ICC workflows with out-of-range
/// values, use [`linear_to_srgb_extended`] instead.
#[inline]
pub fn linear_to_srgb(linear: f32) -> f32 {
    if linear < 0.0 {
        0.0
    } else if linear < LINEAR_THRESHOLD_F32 {
        linear * 12.92
    } else if linear < 1.0 {
        fmla(SRGB_A_PLUS_1_F32, linear.powf(INV_GAMMA_F32), -SRGB_A_F32)
    } else {
        1.0
    }
}

// ============================================================================
// Fast rational polynomial variants (no powf — see function docs for ULP accuracy)
// ============================================================================

/// Convert sRGB gamma-encoded value to linear light using a rational polynomial (f32).
///
/// Same as [`srgb_to_linear`] but replaces `powf()` with a 5/5 rational
/// polynomial (Horner's method). Faster than powf on all platforms and
/// identical to the SIMD path.
///
/// Max error vs f64 reference (exhaustive over all f32):
/// - Full power segment: ~8 ULP max, ~1 ULP avg
///
/// Coefficients from libjxl (BSD-3-Clause).
///
/// **Clamps** inputs to \[0, 1\].
#[inline]
#[allow(dead_code)] // used by tests and alt::accuracy
pub fn srgb_to_linear_fast(gamma: f32) -> f32 {
    crate::rational_poly::srgb_to_linear_fast(gamma)
}

/// Convert linear light value to sRGB gamma-encoded using a rational polynomial (f32).
///
/// Same as [`linear_to_srgb`] but replaces `powf()` with sqrt + 5/5 rational
/// polynomial (Horner's method). Faster than powf on all platforms and
/// identical to the SIMD path.
///
/// Max error vs f64 reference (exhaustive over all f32):
/// - Full power segment: ~8 ULP max, ~1 ULP avg
///
/// Coefficients from libjxl (BSD-3-Clause).
///
/// **Clamps** inputs to \[0, 1\].
#[inline]
#[allow(dead_code)] // used by tests and alt::accuracy
pub fn linear_to_srgb_fast(linear: f32) -> f32 {
    crate::rational_poly::linear_to_srgb_fast(linear)
}

/// Convert sRGB gamma-encoded value to linear light without clamping (f32).
///
/// Unlike [`srgb_to_linear`], this does **not** clamp to \[0, 1\]. Use this for:
/// - **HDR content** where values exceed 1.0
/// - **ICC profile conversions** where negative values are possible
/// - **Scene-referred** workflows with unbounded linear light
///
/// Negative inputs pass through the linear segment (scaled by 1/12.92).
/// Inputs above 1.0 pass through the power segment.
#[inline]
pub fn srgb_to_linear_extended(gamma: f32) -> f32 {
    if gamma < SRGB_LINEAR_THRESHOLD_F32 {
        gamma * LINEAR_SCALE_F32
    } else {
        ((gamma + SRGB_A_F32) / SRGB_A_PLUS_1_F32).powf(GAMMA as f32)
    }
}

/// Convert linear light value to sRGB gamma-encoded without clamping (f32).
///
/// Unlike [`linear_to_srgb`], this does **not** clamp to \[0, 1\]. Use this for:
/// - **HDR content** where linear values exceed 1.0
/// - **ICC profile conversions** where negative values are possible
/// - **Scene-referred** workflows with unbounded linear light
///
/// Negative inputs pass through the linear segment (scaled by 12.92).
/// Inputs above 1.0 pass through the power segment.
#[inline]
pub fn linear_to_srgb_extended(linear: f32) -> f32 {
    if linear < LINEAR_THRESHOLD_F32 {
        linear * 12.92
    } else {
        fmla(SRGB_A_PLUS_1_F32, linear.powf(INV_GAMMA_F32), -SRGB_A_F32)
    }
}

/// Convert linear to 8-bit sRGB using const LUT.
///
/// Uses a 4096-entry lookup table (4KB, fits L1 cache). No transcendental math.
/// Bitmask indexing (`& 0xFFF`) guarantees bounds safety with zero overhead.
/// Max error: ±1 u8 level vs exact computation.
#[inline]
pub fn linear_to_srgb_u8(linear: f32) -> u8 {
    let idx = (linear.clamp(0.0, 1.0) * 4095.0 + 0.5) as usize & 0xFFF;
    crate::const_luts::LINEAR_TO_SRGB_U8[idx]
}

// ============================================================================
// u16 LUT-based conversions
// ============================================================================

/// Convert 16-bit sRGB to linear f32 using a 65536-entry const LUT.
///
/// Zero math — pure table lookup. The LUT is 256KB and generated at compile time
/// with f64 precision. Roundtrip error is 0 u16 levels.
#[inline]
pub fn srgb_u16_to_linear(value: u16) -> f32 {
    crate::const_luts_u16::SRGB_U16_TO_LINEAR_F32[value as usize]
}

/// Convert linear f32 to 16-bit sRGB using a 65537-entry const LUT.
///
/// Uses the same pattern as [`linear_to_srgb_u8`] but with 16-bit resolution.
/// Max error: ±0 u16 levels at 1/65536 resolution.
#[inline]
pub fn linear_to_srgb_u16(linear: f32) -> u16 {
    let idx = (linear.clamp(0.0, 1.0) * 65536.0 + 0.5) as usize;
    crate::const_luts_u16::LINEAR_TO_SRGB_U16_65536[idx]
}

// ============================================================================
// Custom Gamma Functions (pure power, no linear segment)
// ============================================================================

/// Convert gamma-encoded value to linear using a custom gamma exponent (f32).
///
/// This is a pure power function: `linear = gamma_encoded.powf(gamma)`
///
/// Unlike sRGB, there is no linear segment near black. Common gamma values:
/// - 2.2: Traditional "gamma 2.2" used in many applications
/// - 2.4: The power portion of sRGB (but sRGB also has a linear segment)
/// - 1.8: Historic Mac gamma
///
/// Input values are clamped to [0, 1].
///
/// # Example
/// ```
/// use linear_srgb::default::gamma_to_linear;
///
/// let linear = gamma_to_linear(0.5, 2.2);
/// assert!((linear - 0.218).abs() < 0.001);
/// ```
#[inline]
pub fn gamma_to_linear(encoded: f32, gamma: f32) -> f32 {
    if encoded <= 0.0 {
        0.0
    } else if encoded >= 1.0 {
        1.0
    } else {
        encoded.powf(gamma)
    }
}

/// Convert linear value to gamma-encoded using a custom gamma exponent (f32).
///
/// This is a pure power function: `gamma_encoded = linear.powf(1.0 / gamma)`
///
/// Unlike sRGB, there is no linear segment near black.
///
/// Input values are clamped to [0, 1].
///
/// # Example
/// ```
/// use linear_srgb::default::linear_to_gamma;
///
/// let encoded = linear_to_gamma(0.218, 2.2);
/// assert!((encoded - 0.5).abs() < 0.01);
/// ```
#[inline]
pub fn linear_to_gamma(linear: f32, gamma: f32) -> f32 {
    if linear <= 0.0 {
        0.0
    } else if linear >= 1.0 {
        1.0
    } else {
        linear.powf(1.0 / gamma)
    }
}

/// Convert gamma-encoded value to linear using a custom gamma exponent (f64).
///
/// High-precision version of [`gamma_to_linear`].
#[inline]
pub fn gamma_to_linear_f64(encoded: f64, gamma: f64) -> f64 {
    if encoded <= 0.0 {
        0.0
    } else if encoded >= 1.0 {
        1.0
    } else {
        encoded.powf(gamma)
    }
}

/// Convert linear value to gamma-encoded using a custom gamma exponent (f64).
///
/// High-precision version of [`linear_to_gamma`].
#[inline]
pub fn linear_to_gamma_f64(linear: f64, gamma: f64) -> f64 {
    if linear <= 0.0 {
        0.0
    } else if linear >= 1.0 {
        1.0
    } else {
        linear.powf(1.0 / gamma)
    }
}

// ============================================================================
// u8 LUT-based conversion (moved from simd.rs — pure table lookup, not SIMD)
// ============================================================================

/// Precomputed sRGB u8 → linear f32 lookup table.
/// Uses IEC 61966-2-1 constants (0.04045 threshold, 0.055 offset).
/// At u8 precision, these produce identical values to the C0-continuous constants.
static SRGB_U8_TO_LINEAR_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        let srgb = i as f64 / 255.0;
        let linear = if srgb <= 0.04045 {
            srgb / 12.92
        } else {
            // Use manual pow via exp(ln(x)*y) since powf isn't const
            let base = (srgb + 0.055) / 1.055;
            // Approximate pow(base, 2.4) using the identity:
            // We precompute these at compile time, so precision doesn't matter
            // for the LUT - we just need f32 precision in the final value
            // Square-and-multiply for 2.4 = 2 + 0.4
            let sq = base * base; // base^2
            // base^0.4 = (base^2)^0.2 = ((base^2)^(1/5))
            // Use Newton's method: find x where x^5 = base^2
            let target = sq; // base^2, we want target^(1/5) = base^0.4
            let mut x = 0.5f64;
            let mut iter = 0;
            while iter < 100 {
                let x4 = x * x * x * x;
                let x5 = x4 * x;
                x = x - (x5 - target) / (5.0 * x4);
                iter += 1;
            }
            sq * x // base^2 * base^0.4 = base^2.4
        };
        lut[i] = linear as f32;
        i += 1;
    }
    lut
};

#[inline]
fn get_lut() -> &'static [f32; 256] {
    &SRGB_U8_TO_LINEAR_LUT
}

/// Convert a single sRGB u8 value to linear f32 using LUT lookup.
///
/// This is the fastest method for u8 input as it uses a precomputed lookup table
/// embedded in the binary. For batch conversions, use [`crate::default::srgb_u8_to_linear_slice`].
///
/// # Example
/// ```
/// use linear_srgb::default::srgb_u8_to_linear;
///
/// let linear = srgb_u8_to_linear(128);
/// assert!((linear - 0.2158).abs() < 0.001);
/// ```
#[inline]
pub fn srgb_u8_to_linear(value: u8) -> f32 {
    get_lut()[value as usize]
}

/// Convert 8 sRGB u8 values to linear f32 using LUT lookup.
///
/// # Example
/// ```ignore
/// use linear_srgb::scalar::srgb_u8_to_linear_x8;
///
/// let srgb = [0u8, 64, 128, 192, 255, 32, 96, 160];
/// let linear = srgb_u8_to_linear_x8(srgb);
/// ```
#[inline]
pub fn srgb_u8_to_linear_x8(srgb: [u8; 8]) -> [f32; 8] {
    let lut = get_lut();
    [
        lut[srgb[0] as usize],
        lut[srgb[1] as usize],
        lut[srgb[2] as usize],
        lut[srgb[3] as usize],
        lut[srgb[4] as usize],
        lut[srgb[5] as usize],
        lut[srgb[6] as usize],
        lut[srgb[7] as usize],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_to_linear_boundaries() {
        assert_eq!(srgb_to_linear(-0.1), 0.0);
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert_eq!(srgb_to_linear(1.0), 1.0);
        assert_eq!(srgb_to_linear(1.1), 1.0);
    }

    #[test]
    fn test_linear_to_srgb_boundaries() {
        assert_eq!(linear_to_srgb(-0.1), 0.0);
        assert_eq!(linear_to_srgb(0.0), 0.0);
        assert_eq!(linear_to_srgb(1.0), 1.0);
        assert_eq!(linear_to_srgb(1.1), 1.0);
    }

    #[test]
    fn test_roundtrip_f32() {
        for i in 0..=255 {
            let srgb = i as f32 / 255.0;
            let linear = srgb_to_linear(srgb);
            let back = linear_to_srgb(linear);
            assert!(
                (srgb - back).abs() < 1e-5,
                "Roundtrip failed for {}: {} -> {} -> {}",
                i,
                srgb,
                linear,
                back
            );
        }
    }

    #[test]
    fn test_roundtrip_f64() {
        for i in 0..=255 {
            let srgb = i as f64 / 255.0;
            let linear = srgb_to_linear_f64(srgb);
            let back = linear_to_srgb_f64(linear);
            assert!(
                (srgb - back).abs() < 1e-10,
                "Roundtrip failed for {}: {} -> {} -> {}",
                i,
                srgb,
                linear,
                back
            );
        }
    }

    #[test]
    fn test_linear_segment() {
        // Values below threshold should use linear formula
        let test_val = 0.02f32;
        let linear = srgb_to_linear(test_val);
        let expected = test_val / 12.92;
        assert!((linear - expected).abs() < 1e-7);
    }

    #[test]
    fn test_known_values() {
        // Middle gray (sRGB 0.5 ≈ linear 0.214)
        let linear = srgb_to_linear(0.5);
        assert!((linear - 0.214).abs() < 0.001);

        // 18% gray is roughly linear 0.18, sRGB ~0.46
        let srgb = linear_to_srgb(0.18);
        assert!((srgb - 0.46).abs() < 0.01);
    }

    #[test]
    #[allow(deprecated)]
    fn test_u8_conversion() {
        assert_eq!(srgb_u8_to_linear(0), 0.0);
        assert_eq!(linear_to_srgb_u8(0.0), 0);
        assert_eq!(linear_to_srgb_u8(1.0), 255);

        // Roundtrip
        for i in 0..=255u8 {
            let linear = srgb_u8_to_linear(i);
            let back = linear_to_srgb_u8(linear);
            assert!(
                (i as i32 - back as i32).abs() <= 1,
                "u8 roundtrip failed for {}",
                i
            );
        }
    }

    #[test]
    fn test_u16_conversion() {
        // Boundary values
        assert_eq!(srgb_u16_to_linear(0), 0.0);
        assert_eq!(srgb_u16_to_linear(65535), 1.0);
        assert_eq!(linear_to_srgb_u16(0.0), 0);
        assert_eq!(linear_to_srgb_u16(1.0), 65535);

        // Clamping
        assert_eq!(linear_to_srgb_u16(-0.1), 0);
        assert_eq!(linear_to_srgb_u16(1.1), 65535);

        // Monotonicity
        let mut prev = 0.0f32;
        for i in 0..=65535u16 {
            let linear = srgb_u16_to_linear(i);
            assert!(
                linear >= prev,
                "non-monotonic at {}: {} < {}",
                i,
                linear,
                prev
            );
            prev = linear;
        }

        // Roundtrip at 257-spaced values (equivalent to u8 levels scaled to u16).
        // Low sRGB values have higher error because the LUT index quantization
        // through f32 intermediate loses precision in the dark region.
        for i in 0..=255u16 {
            let val = i * 257; // 0, 257, 514, ..., 65535
            let linear = srgb_u16_to_linear(val);
            let back = linear_to_srgb_u16(linear);
            let diff = (val as i32 - back as i32).unsigned_abs();
            assert!(
                diff <= 10,
                "u16 roundtrip failed for {}: {} -> {} -> {} (diff {})",
                i,
                val,
                linear,
                back,
                diff
            );
        }
        // High values should roundtrip exactly
        assert_eq!(linear_to_srgb_u16(srgb_u16_to_linear(65535)), 65535);
    }

    #[test]
    fn test_custom_gamma_boundaries() {
        // Test clamping
        assert_eq!(gamma_to_linear(-0.1, 2.2), 0.0);
        assert_eq!(gamma_to_linear(0.0, 2.2), 0.0);
        assert_eq!(gamma_to_linear(1.0, 2.2), 1.0);
        assert_eq!(gamma_to_linear(1.1, 2.2), 1.0);

        assert_eq!(linear_to_gamma(-0.1, 2.2), 0.0);
        assert_eq!(linear_to_gamma(0.0, 2.2), 0.0);
        assert_eq!(linear_to_gamma(1.0, 2.2), 1.0);
        assert_eq!(linear_to_gamma(1.1, 2.2), 1.0);
    }

    #[test]
    fn test_custom_gamma_known_values() {
        // gamma 2.2: 0.5^2.2 ≈ 0.2176
        let linear = gamma_to_linear(0.5, 2.2);
        assert!(
            (linear - 0.2176).abs() < 0.001,
            "gamma_to_linear(0.5, 2.2) = {}, expected ~0.2176",
            linear
        );

        // Inverse: 0.2176^(1/2.2) ≈ 0.5
        let encoded = linear_to_gamma(0.2176, 2.2);
        assert!(
            (encoded - 0.5).abs() < 0.01,
            "linear_to_gamma(0.2176, 2.2) = {}, expected ~0.5",
            encoded
        );
    }

    #[test]
    fn test_custom_gamma_roundtrip() {
        for gamma in [1.8, 2.0, 2.2, 2.4, 2.6] {
            for i in 0..=255 {
                let encoded = i as f32 / 255.0;
                let linear = gamma_to_linear(encoded, gamma);
                let back = linear_to_gamma(linear, gamma);
                assert!(
                    (encoded - back).abs() < 1e-5,
                    "Roundtrip failed for gamma={}, value={}: {} -> {} -> {}",
                    gamma,
                    i,
                    encoded,
                    linear,
                    back
                );
            }
        }
    }

    #[test]
    fn test_custom_gamma_f64_precision() {
        // f64 should have higher precision
        let encoded = 0.5_f64;
        let gamma = 2.2_f64;
        let linear = gamma_to_linear_f64(encoded, gamma);
        let back = linear_to_gamma_f64(linear, gamma);
        assert!(
            (encoded - back).abs() < 1e-14,
            "f64 roundtrip: {} -> {} -> {}",
            encoded,
            linear,
            back
        );
    }

    #[test]
    fn test_srgb_to_linear_fast_boundaries() {
        assert_eq!(srgb_to_linear_fast(-0.1), 0.0);
        assert_eq!(srgb_to_linear_fast(0.0), 0.0);
        assert_eq!(srgb_to_linear_fast(1.0), 1.0);
        assert_eq!(srgb_to_linear_fast(1.1), 1.0);
    }

    #[test]
    fn test_linear_to_srgb_fast_boundaries() {
        assert_eq!(linear_to_srgb_fast(-0.1), 0.0);
        assert_eq!(linear_to_srgb_fast(0.0), 0.0);
        assert_eq!(linear_to_srgb_fast(1.0), 1.0);
        assert_eq!(linear_to_srgb_fast(1.1), 1.0);
    }

    #[test]
    fn test_fast_vs_powf() {
        // _fast should closely match powf-based functions
        for i in 0..=255 {
            let srgb = i as f32 / 255.0;
            let exact = srgb_to_linear(srgb);
            let fast = srgb_to_linear_fast(srgb);
            assert!(
                (exact - fast).abs() < 1e-5,
                "srgb_to_linear_fast mismatch at {}/255: exact={}, fast={}, diff={}",
                i,
                exact,
                fast,
                (exact - fast).abs()
            );
        }
        for i in 0..=255 {
            let linear = i as f32 / 255.0;
            let exact = linear_to_srgb(linear);
            let fast = linear_to_srgb_fast(linear);
            assert!(
                (exact - fast).abs() < 1e-5,
                "linear_to_srgb_fast mismatch at {}/255: exact={}, fast={}, diff={}",
                i,
                exact,
                fast,
                (exact - fast).abs()
            );
        }
    }

    #[test]
    fn test_fast_roundtrip() {
        for i in 0..=255 {
            let srgb = i as f32 / 255.0;
            let linear = srgb_to_linear_fast(srgb);
            let back = linear_to_srgb_fast(linear);
            assert!(
                (srgb - back).abs() < 1e-4,
                "Fast roundtrip failed at {}/255: {} -> {} -> {}, diff={}",
                i,
                srgb,
                linear,
                back,
                (srgb - back).abs()
            );
        }
    }

    #[test]
    fn test_fast_linear_segment() {
        // Below threshold, _fast should use the same linear formula
        let test_val = 0.02f32;
        let fast = srgb_to_linear_fast(test_val);
        let expected = test_val / 12.92;
        assert!((fast - expected).abs() < 1e-7);
    }
}
