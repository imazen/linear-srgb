//! Lazily-initialized u16 sRGB lookup tables.
//!
//! Tables are generated on first use via `OnceLock` — no binary bloat,
//! no compile-time cost. Only allocated if the u16 API is actually called.
//!
//! - **Decode** (sRGB u16 → linear f32): 65536-entry direct lookup, 256KB.
//!   Exact — each u16 value has its own f32 entry.
//!
//! - **Encode** (linear f32 → sRGB u16): 65537-entry sqrt-indexed LUT, 128KB.
//!   Indexed by `sqrt(linear) * 65536`, which concentrates resolution where
//!   the sRGB curve is steepest (near black). Max roundtrip error ±1 u16 level,
//!   94.2% exact roundtrip (vs 71.3% / ±6 with uniform indexing).
//!
//! Generation uses SIMD-dispatched slice functions in L1-sized chunks.

use std::sync::OnceLock;

/// Chunk size for LUT generation — 16KB fits in L1 cache.
const CHUNK: usize = 4096;

// ============================================================================
// Decode: sRGB u16 → linear f32 (65536 entries, 256KB)
// ============================================================================

static DECODE_LUT: OnceLock<Box<[f32; 65536]>> = OnceLock::new();

/// Generate the decode LUT using SIMD-accelerated sRGB→linear conversion.
#[doc(hidden)]
pub fn generate_decode_lut() -> Box<[f32; 65536]> {
    let mut v: Vec<f32> = (0..65536u32).map(|i| i as f32 * (1.0 / 65535.0)).collect();
    for chunk in v.chunks_mut(CHUNK) {
        crate::simd::srgb_to_linear_slice(chunk);
    }
    v.into_boxed_slice().try_into().ok().unwrap()
}

/// Get the decode LUT, initializing on first call.
#[inline]
pub(crate) fn decode_lut() -> &'static [f32; 65536] {
    DECODE_LUT.get_or_init(generate_decode_lut)
}

// ============================================================================
// Encode: linear f32 → sRGB u16 (65537 entries, ~128KB, sqrt-indexed)
// ============================================================================

static ENCODE_LUT: OnceLock<Box<[u16; 65537]>> = OnceLock::new();

/// Number of entries in the sqrt-indexed encode LUT.
pub(crate) const ENCODE_LUT_N: usize = 65537;
/// Scale factor for sqrt index: `idx = (sqrt(linear) * ENCODE_SQRT_SCALE + 0.5) as usize`
pub(crate) const ENCODE_SQRT_SCALE: f32 = (ENCODE_LUT_N - 1) as f32;

/// Generate the sqrt-indexed encode LUT using SIMD-accelerated linear→sRGB.
///
/// Entry `i` stores the sRGB u16 value for `linear = (i / 65536)²`.
/// Lookup uses `idx = (sqrt(linear) * 65536 + 0.5)`.
#[doc(hidden)]
pub fn generate_encode_lut() -> Box<[u16; 65537]> {
    let mut lut: Vec<u16> = Vec::with_capacity(ENCODE_LUT_N);
    let mut scratch = [0.0f32; CHUNK];
    let mut i = 0u32;
    while (i as usize) < ENCODE_LUT_N {
        let n = CHUNK.min(ENCODE_LUT_N - i as usize);
        for (j, s) in scratch[..n].iter_mut().enumerate() {
            // Inverse of sqrt: index i maps to linear = (i/65536)²
            let t = (i + j as u32) as f32 * (1.0 / ENCODE_SQRT_SCALE);
            *s = t * t;
        }
        crate::simd::linear_to_srgb_slice(&mut scratch[..n]);
        for &s in &scratch[..n] {
            lut.push((s * 65535.0 + 0.5) as u16);
        }
        i += n as u32;
    }
    lut.into_boxed_slice().try_into().ok().unwrap()
}

/// Get the sqrt-indexed encode LUT, initializing on first call.
#[inline]
pub(crate) fn encode_lut() -> &'static [u16; 65537] {
    ENCODE_LUT.get_or_init(generate_encode_lut)
}
