//! Lazily-initialized u16 sRGB lookup tables.
//!
//! Tables are generated on first use via `OnceLock` — no binary bloat,
//! no compile-time cost. The 512KB (256KB decode + 256KB encode) is only
//! allocated if a caller actually uses the u16 API.
//!
//! Generation uses f64 `powf` for maximum accuracy, matching the quality
//! of hand-written const tables.

use std::sync::OnceLock;

// ============================================================================
// Decode: sRGB u16 → linear f32 (65536 entries, 256KB)
// ============================================================================

static DECODE_LUT: OnceLock<Box<[f32; 65536]>> = OnceLock::new();

/// C0-continuous sRGB constants (moxcms).
const A: f64 = 0.055_010_718_947_586_6;
const A_PLUS_1: f64 = 1.055_010_718_947_586_6;
const THRESHOLD: f64 = 12.92 * 0.003_041_282_560_127_521;

fn srgb_to_linear_f64(srgb: f64) -> f64 {
    if srgb <= 0.0 {
        0.0
    } else if srgb <= THRESHOLD {
        srgb / 12.92
    } else if srgb < 1.0 {
        ((srgb + A) / A_PLUS_1).powf(2.4)
    } else {
        1.0
    }
}

fn generate_decode_lut() -> Box<[f32; 65536]> {
    let mut lut = Box::new([0.0f32; 65536]);
    for i in 0..65536 {
        lut[i] = srgb_to_linear_f64(i as f64 / 65535.0) as f32;
    }
    lut
}

/// Get the decode LUT, initializing on first call.
#[inline]
pub(crate) fn decode_lut() -> &'static [f32; 65536] {
    DECODE_LUT.get_or_init(generate_decode_lut)
}

// ============================================================================
// Encode: linear f32 → sRGB u16 (65537 entries, ~128KB)
// ============================================================================

static ENCODE_LUT: OnceLock<Box<[u16; 65537]>> = OnceLock::new();

fn linear_to_srgb_f64(linear: f64) -> f64 {
    if linear <= 0.0 {
        0.0
    } else if linear <= 0.003_041_282_560_127_521 {
        linear * 12.92
    } else if linear < 1.0 {
        A_PLUS_1 * linear.powf(1.0 / 2.4) - A
    } else {
        1.0
    }
}

fn generate_encode_lut() -> Box<[u16; 65537]> {
    let mut lut = Box::new([0u16; 65537]);
    for i in 0..=65536 {
        let linear = i as f64 / 65536.0;
        let srgb = linear_to_srgb_f64(linear);
        lut[i] = (srgb * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
    }
    lut
}

/// Get the encode LUT, initializing on first call.
#[inline]
pub(crate) fn encode_lut() -> &'static [u16; 65537] {
    ENCODE_LUT.get_or_init(generate_encode_lut)
}
