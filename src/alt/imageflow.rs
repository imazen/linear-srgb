//! Imageflow's sRGB conversion algorithms for benchmarking comparison.
//!
//! This module contains conversions equivalent to imageflow_core/src/graphics/color.rs
//! and math.rs, rewritten to use safe Rust bit manipulation.

#[cfg(feature = "std")]
use std::sync::LazyLock;

// ============================================================================
// From imageflow_core/src/graphics/math.rs (rewritten with safe bit ops)
// ============================================================================

/// Fast approximate 2^p using bit manipulation.
/// Equivalent to imageflow's fastpow2, using safe f32::from_bits.
#[inline]
fn fastpow2(p: f32) -> f32 {
    let offset: f32 = if p < 0.0 { 1.0 } else { 0.0 };
    let clipp: f32 = if p < -126.0 { -126.0 } else { p };
    let z: f32 = clipp - (clipp as i32) as f32 + offset;
    let bits = ((1_i32 << 23) as f32
        * (clipp + 121.274_055_f32 + 27.728_024_f32 / (4.842_525_5_f32 - z) - 1.490_129_1_f32 * z))
        as u32;
    f32::from_bits(bits)
}

/// Fast approximate log2(x) using bit manipulation.
/// Equivalent to imageflow's fastlog2, using safe f32::to_bits/from_bits.
#[inline]
fn fastlog2(x: f32) -> f32 {
    let vx_bits = x.to_bits();
    let mx_bits = (vx_bits & 0x007f_ffff) | 0x3f00_0000;
    let mx = f32::from_bits(mx_bits);
    let mut y: f32 = vx_bits as f32;
    y *= 1.192_092_9e-7_f32;
    y - 124.225_52_f32 - 1.498_030_3_f32 * mx - 1.725_88_f32 / (0.352_088_72_f32 + mx)
}

/// Fast approximate power function using bit manipulation.
/// From imageflow_core/src/graphics/math.rs
#[inline]
pub fn fastpow(x: f32, p: f32) -> f32 {
    fastpow2(p * fastlog2(x))
}

// ============================================================================
// From imageflow_core/src/graphics/color.rs
// ============================================================================

/// sRGB to linear using imageflow's constants (standard 0.04045 threshold).
#[inline]
pub fn srgb_to_linear(s: f32) -> f32 {
    if s <= 0.04045f32 {
        s / 12.92f32
    } else {
        f32::powf((s + 0.055f32) / (1_f32 + 0.055f32), 2.4f32)
    }
}

/// Linear to sRGB using imageflow's formula with fastpow.
/// Returns value in [0, 255] range (not normalized).
#[inline]
pub fn linear_to_srgb_raw(clr: f32) -> f32 {
    if clr <= 0.0031308f32 {
        12.92f32 * clr * 255.0f32
    } else {
        1.055f32 * 255.0f32 * fastpow(clr, 0.41666666f32) - 14.025f32
    }
}

/// Linear to sRGB normalized (0 to 1) output using fastpow.
#[inline]
pub fn linear_to_srgb(clr: f32) -> f32 {
    if clr <= 0.0031308f32 {
        12.92f32 * clr
    } else {
        1.055f32 * fastpow(clr, 0.41666666f32) - 0.055f32
    }
}

/// Clamp float to u8 using imageflow's method.
#[inline]
pub fn uchar_clamp_ff(clr: f32) -> u8 {
    let mut result: u16;
    result = (clr as f64 + 0.5f64) as i16 as u16;
    if result as i32 > 255_i32 {
        result = if clr < 0_i32 as f32 { 0_i32 } else { 255_i32 } as u16
    }
    result as u8
}

/// Convert linear f32 to sRGB u8 using fastpow formula.
#[inline]
pub fn linear_to_srgb_u8_fastpow(linear: f32) -> u8 {
    uchar_clamp_ff(linear_to_srgb_raw(linear))
}

/// Fast linear→sRGB using precomputed 16K LUT.
#[inline]
pub fn linear_to_srgb_lut(linear: f32) -> u8 {
    let idx = (linear * 16383.0).clamp(0.0, 16383.0) as usize;
    LINEAR_TO_SRGB_LUT[idx]
}

/// Convert sRGB u8 to linear f32 using standard formula.
#[inline]
pub fn srgb_u8_to_linear(value: u8) -> f32 {
    srgb_to_linear(value as f32 / 255.0)
}

/// Pre-computed sRGB u8→linear f32 lookup table (256 entries).
pub struct SrgbToLinearLut {
    table: [f32; 256],
}

impl SrgbToLinearLut {
    /// Create the LUT using imageflow's srgb_to_linear formula.
    pub fn new() -> Self {
        let mut table = [0.0f32; 256];
        for (i, entry) in table.iter_mut().enumerate() {
            *entry = srgb_to_linear(i as f32 / 255.0);
        }
        Self { table }
    }

    #[inline]
    pub fn lookup(&self, value: u8) -> f32 {
        // Safe: u8 is always in bounds for a 256-element array
        self.table[value as usize]
    }
}

impl Default for SrgbToLinearLut {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate the linear→sRGB LUT using imageflow's exact formula.
/// Returns 16384 entries matching imageflow's static table.
fn generate_linear_to_srgb_lut() -> [u8; 16384] {
    let mut lut = [0u8; 16384];
    for (i, entry) in lut.iter_mut().enumerate() {
        let linear = i as f32 / 16383.0;
        // Use imageflow's exact formula (with accurate powf for LUT generation)
        let srgb_raw = if linear <= 0.0031308f32 {
            12.92f32 * linear * 255.0f32
        } else {
            1.055f32 * 255.0f32 * linear.powf(0.41666666f32) - 14.025f32
        };
        *entry = uchar_clamp_ff(srgb_raw);
    }
    lut
}

/// Precomputed linear→sRGB LUT (16384 entries)
/// Generated using imageflow's exact formula
pub static LINEAR_TO_SRGB_LUT: LazyLock<[u8; 16384]> = LazyLock::new(generate_linear_to_srgb_lut);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_to_linear_boundaries() {
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_to_srgb_lut_boundaries() {
        assert_eq!(linear_to_srgb_lut(0.0), 0);
        assert_eq!(linear_to_srgb_lut(1.0), 255);
    }

    #[test]
    fn test_fastpow_approximation() {
        // Test that fastpow gives reasonable results for sRGB gamma
        let x = 0.5f32;
        let p = 1.0 / 2.4;
        let fast = fastpow(x, p);
        let accurate = x.powf(p);
        // Allow ~5% error for the approximation
        assert!((fast - accurate).abs() / accurate < 0.05);
    }

    #[test]
    fn test_roundtrip_u8() {
        // Test that u8 roundtrip works within ±1
        for i in 0..=255u8 {
            let linear = srgb_u8_to_linear(i);
            let back = linear_to_srgb_lut(linear);
            assert!(
                (i as i32 - back as i32).abs() <= 1,
                "Roundtrip failed for {}: {} -> {} -> {}",
                i,
                i,
                linear,
                back
            );
        }
    }
}
