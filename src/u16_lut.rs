//! Lazily-initialized u16 sRGB lookup tables.
//!
//! Tables are generated on first use via `OnceLock` — no binary bloat,
//! no compile-time cost. The ~384KB is only allocated if a caller actually
//! uses the u16 API.
//!
//! Generation uses SIMD-dispatched `srgb_to_linear_slice` / `linear_to_srgb_slice`
//! in L1-sized chunks for cache locality. The SIMD rational polynomial may
//! produce slightly different f32 bits than the scalar path due to FMA, but
//! both are ≤14 ULP of the f64 reference.

use std::sync::OnceLock;

/// Chunk size for LUT generation. 4096 f32s = 16KB — fits in L1 cache,
/// so SIMD conversion hits warm data and quantization reads don't miss.
const CHUNK: usize = 4096;

// ============================================================================
// Decode: sRGB u16 → linear f32 (65536 entries, 256KB)
// ============================================================================

static DECODE_LUT: OnceLock<Box<[f32; 65536]>> = OnceLock::new();

/// Generate the decode LUT using SIMD-accelerated sRGB→linear conversion.
///
/// Fills the output Vec with `i / 65535.0`, then SIMD-converts in L1-sized
/// chunks. One allocation (256KB), one write pass for fill, one read+write
/// pass for conversion with each chunk hot in L1.
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
// Encode: linear f32 → sRGB u16 (65537 entries, ~128KB)
// ============================================================================

static ENCODE_LUT: OnceLock<Box<[u16; 65537]>> = OnceLock::new();

/// Generate the encode LUT using SIMD-accelerated linear→sRGB conversion.
///
/// Uses a 16KB f32 scratch buffer (L1-resident) for chunked SIMD conversion,
/// writing u16 values directly into the output. Only one heap allocation
/// (128KB for the u16 output) instead of two (256KB f32 + 128KB u16).
#[doc(hidden)]
pub fn generate_encode_lut() -> Box<[u16; 65537]> {
    let mut lut: Vec<u16> = Vec::with_capacity(65537);
    let mut scratch = [0.0f32; CHUNK];
    let mut i = 0u32;
    while (i as usize) < 65537 {
        let n = CHUNK.min(65537 - i as usize);
        for j in 0..n {
            scratch[j] = (i + j as u32) as f32 * (1.0 / 65536.0);
        }
        crate::simd::linear_to_srgb_slice(&mut scratch[..n]);
        for j in 0..n {
            lut.push((scratch[j] * 65535.0 + 0.5) as u16);
        }
        i += n as u32;
    }
    lut.into_boxed_slice().try_into().ok().unwrap()
}

/// Get the encode LUT, initializing on first call.
#[inline]
pub(crate) fn encode_lut() -> &'static [u16; 65537] {
    ENCODE_LUT.get_or_init(generate_encode_lut)
}
