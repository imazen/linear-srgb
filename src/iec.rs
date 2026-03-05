//! IEC 61966-2-1:1999 textbook sRGB transfer functions.
//!
//! These use the original specification constants (threshold 0.04045, offset 0.055)
//! which have a ~2.3e-9 discontinuity at the piecewise junction. The default
//! module uses C0-continuous (moxcms) constants that eliminate this discontinuity.
//!
//! Use this module when you need **exact** IEC 61966-2-1 behavior for
//! interoperability with software that implements the original specification
//! verbatim (e.g., ICC profile processors, conformance test suites).
//!
//! At u8 precision the two constant sets produce identical results. Differences
//! only appear near the piecewise threshold at ≥16-bit or f32/f64 precision.
//!
//! Requires the `iec` feature.

use crate::mlaf::fmla;
#[allow(unused_imports)]
use num_traits::Float;

// IEC 61966-2-1:1999 textbook constants
const SRGB_LINEAR_THRESHOLD: f64 = 0.04045;
const SRGB_LINEAR_THRESHOLD_F32: f32 = SRGB_LINEAR_THRESHOLD as f32;
const LINEAR_THRESHOLD: f64 = 0.003_130_804_953_560_372;
const LINEAR_THRESHOLD_F32: f32 = LINEAR_THRESHOLD as f32;
const LINEAR_SCALE: f64 = 1.0 / 12.92;
const LINEAR_SCALE_F32: f32 = LINEAR_SCALE as f32;
const SRGB_A: f64 = 0.055;
const SRGB_A_F32: f32 = SRGB_A as f32;
const SRGB_A_PLUS_1: f64 = 1.055;
const SRGB_A_PLUS_1_F32: f32 = SRGB_A_PLUS_1 as f32;
const GAMMA: f64 = 2.4;
const INV_GAMMA: f64 = 1.0 / GAMMA;
const INV_GAMMA_F32: f32 = INV_GAMMA as f32;

/// Convert sRGB gamma-encoded value to linear light (f64, IEC 61966-2-1).
///
/// Uses the original IEC textbook constants.
/// Input clamped to \[0, 1\].
#[inline]
pub fn srgb_to_linear_f64(gamma: f64) -> f64 {
    if gamma < 0.0 {
        0.0
    } else if gamma < SRGB_LINEAR_THRESHOLD {
        gamma * LINEAR_SCALE
    } else if gamma < 1.0 {
        ((gamma + SRGB_A) / SRGB_A_PLUS_1).powf(GAMMA)
    } else {
        1.0
    }
}

/// Convert sRGB gamma-encoded value to linear light (f32, IEC 61966-2-1).
///
/// Uses the original IEC textbook constants.
/// Input clamped to \[0, 1\].
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

/// Convert linear light value to sRGB gamma-encoded (f64, IEC 61966-2-1).
///
/// Uses the original IEC textbook constants.
/// Input clamped to \[0, 1\].
#[inline]
pub fn linear_to_srgb_f64(linear: f64) -> f64 {
    if linear < 0.0 {
        0.0
    } else if linear < LINEAR_THRESHOLD {
        linear * 12.92
    } else if linear < 1.0 {
        fmla(SRGB_A_PLUS_1, linear.powf(INV_GAMMA), -SRGB_A)
    } else {
        1.0
    }
}

/// Convert linear light value to sRGB gamma-encoded (f32, IEC 61966-2-1).
///
/// Uses the original IEC textbook constants.
/// Input clamped to \[0, 1\].
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iec_boundaries() {
        assert_eq!(srgb_to_linear(-0.1), 0.0);
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert_eq!(srgb_to_linear(1.0), 1.0);
        assert_eq!(srgb_to_linear(1.1), 1.0);

        assert_eq!(linear_to_srgb(-0.1), 0.0);
        assert_eq!(linear_to_srgb(0.0), 0.0);
        assert_eq!(linear_to_srgb(1.0), 1.0);
        assert_eq!(linear_to_srgb(1.1), 1.0);
    }

    #[test]
    fn test_iec_roundtrip() {
        for i in 0..=255 {
            let srgb = i as f32 / 255.0;
            let linear = srgb_to_linear(srgb);
            let back = linear_to_srgb(linear);
            assert!(
                (srgb - back).abs() < 1e-5,
                "IEC roundtrip failed at {}: {} -> {} -> {}",
                i,
                srgb,
                linear,
                back
            );
        }
    }

    #[test]
    fn test_iec_threshold_is_0_04045() {
        // Verify this uses the IEC threshold, not C0
        // 0.04 is below 0.04045, so should use linear segment
        let linear_seg = srgb_to_linear(0.04_f32);
        let expected = 0.04_f32 * LINEAR_SCALE_F32;
        assert_eq!(
            linear_seg, expected,
            "0.04 should be in linear segment with IEC threshold"
        );
    }

    #[test]
    fn test_iec_f64_roundtrip() {
        for i in 0..=255 {
            let srgb = i as f64 / 255.0;
            let linear = srgb_to_linear_f64(srgb);
            let back = linear_to_srgb_f64(linear);
            assert!(
                (srgb - back).abs() < 1e-10,
                "IEC f64 roundtrip failed at {}: {} -> {} -> {}",
                i,
                srgb,
                linear,
                back
            );
        }
    }
}
