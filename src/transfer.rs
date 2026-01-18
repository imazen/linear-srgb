//! Core sRGB transfer functions.
//!
//! Implements the IEC 61966-2-1:1999 sRGB transfer functions with optimizations:
//! - Piecewise function avoids pow() for ~1.2% of values in the linear segment
//! - Early exit for out-of-range values (0 and 1) avoids transcendentals
//! - FMA instructions for the encoding formula

use crate::mlaf::fmla;

// sRGB constants (IEC 61966-2-1:1999)
// These are the exact values for the continuous piecewise function.
// Precision is intentional - these come from the moxcms reference implementation.

/// Linear threshold for linearization: 12.92 * 0.0030412825601275209 ≈ 0.04045
#[allow(clippy::excessive_precision)]
const SRGB_LINEAR_THRESHOLD: f64 = 12.92 * 0.003_041_282_560_127_521;
const SRGB_LINEAR_THRESHOLD_F32: f32 = SRGB_LINEAR_THRESHOLD as f32;

/// Linear threshold for encoding (the inverse cutoff point)
#[allow(clippy::excessive_precision)]
const LINEAR_THRESHOLD: f64 = 0.003_041_282_560_127_521;
const LINEAR_THRESHOLD_F32: f32 = LINEAR_THRESHOLD as f32;

/// Linear scale factor (1/12.92)
const LINEAR_SCALE: f64 = 1.0 / 12.92;
const LINEAR_SCALE_F32: f32 = LINEAR_SCALE as f32;

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
/// Input: sRGB value in [0, 1]
/// Output: Linear light value in [0, 1]
///
/// Values outside [0, 1] are clamped.
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
/// Input: sRGB value in [0, 1]
/// Output: Linear light value in [0, 1]
///
/// Values outside [0, 1] are clamped.
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
/// Input: Linear light value in [0, 1]
/// Output: sRGB value in [0, 1]
///
/// Values outside [0, 1] are clamped.
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
/// Input: Linear light value in [0, 1]
/// Output: sRGB value in [0, 1]
///
/// Values outside [0, 1] are clamped.
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

/// Convert sRGB gamma-encoded value to linear light without clamping (f32).
///
/// For extended range HDR workflows where values may exceed [0, 1].
/// Uses the transfer function for all values, following the mathematical definition.
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
/// For extended range HDR workflows where values may exceed [0, 1].
#[inline]
pub fn linear_to_srgb_extended(linear: f32) -> f32 {
    if linear < LINEAR_THRESHOLD_F32 {
        linear * 12.92
    } else {
        fmla(SRGB_A_PLUS_1_F32, linear.powf(INV_GAMMA_F32), -SRGB_A_F32)
    }
}

/// Convert 8-bit sRGB to linear (using direct computation).
#[inline]
pub fn srgb_u8_to_linear(value: u8) -> f32 {
    srgb_to_linear(value as f32 / 255.0)
}

/// Convert linear to 8-bit sRGB (using direct computation).
#[inline]
pub fn linear_to_srgb_u8(linear: f32) -> u8 {
    (linear_to_srgb(linear) * 255.0 + 0.5) as u8
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
/// use linear_srgb::transfer::gamma_to_linear;
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
/// use linear_srgb::transfer::linear_to_gamma;
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
}
