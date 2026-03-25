//! Lazily-initialized u16 sRGB lookup tables.
//!
//! Tables are generated on first use via `OnceLock` — no binary bloat,
//! no compile-time cost. The ~384KB is only allocated if a caller actually
//! uses the u16 API.
//!
//! Generation uses SIMD-dispatched `srgb_to_linear_slice` / `linear_to_srgb_slice`.
//! The SIMD rational polynomial may produce slightly different f32 bits than the
//! scalar path due to FMA, but both are ≤14 ULP of the f64 reference.

use std::sync::OnceLock;

// ============================================================================
// Decode: sRGB u16 → linear f32 (65536 entries, 256KB)
// ============================================================================

static DECODE_LUT: OnceLock<Box<[f32; 65536]>> = OnceLock::new();

/// Generate the decode LUT using SIMD-accelerated sRGB→linear conversion.
#[doc(hidden)]
pub fn generate_decode_lut() -> Box<[f32; 65536]> {
    let mut v: Vec<f32> = Vec::with_capacity(65536);
    // Fill and convert: the fill IS the input, SIMD converts in-place
    v.extend((0..65536u32).map(|i| i as f32 * (1.0 / 65535.0)));
    crate::simd::srgb_to_linear_slice(&mut v);
    v.into_boxed_slice().try_into().ok().unwrap()
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

/// Generate the encode LUT using SIMD-accelerated linear→sRGB conversion.
#[doc(hidden)]
pub fn generate_encode_lut() -> Box<[u16; 65537]> {
    let mut srgb: Vec<f32> = Vec::with_capacity(65537);
    srgb.extend((0..=65536u32).map(|i| i as f32 * (1.0 / 65536.0)));
    crate::simd::linear_to_srgb_slice(&mut srgb);
    // Fused quantize: read f32, write u16 directly
    let v: Vec<u16> = srgb
        .into_iter()
        .map(|s| (s * 65535.0 + 0.5) as u16) // no clamp needed: slice already clamped [0,1]
        .collect();
    v.into_boxed_slice().try_into().ok().unwrap()
}

/// Get the encode LUT, initializing on first call.
#[inline]
pub(crate) fn encode_lut() -> &'static [u16; 65537] {
    ENCODE_LUT.get_or_init(generate_encode_lut)
}
