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
//! Generation uses the scalar f64-intermediate polynomial for platform-
//! independent precision (see `generate_decode_lut` for rationale).

use std::sync::OnceLock;

// ============================================================================
// Decode: sRGB u16 → linear f32 (65536 entries, 256KB)
// ============================================================================

static DECODE_LUT: OnceLock<Box<[f32; 65536]>> = OnceLock::new();

/// Generate the decode LUT using the scalar f64-intermediate polynomial.
///
/// Uses `rational_poly::srgb_to_linear_fast` (f64 Horner evaluation) instead
/// of SIMD f32 dispatch to guarantee exact roundtrip with the scalar encode
/// path (`linear_to_srgb_u16`). The f64 intermediate eliminates platform-
/// dependent rounding differences — WASM SIMD128 lacks fused multiply-add,
/// so its f32 `mul_add` (two roundings) can diverge from f64 by enough ULPs
/// to break u16 roundtrip at boundary values.
///
/// LUT generation is one-time (`OnceLock`), so the cost is negligible.
#[doc(hidden)]
pub fn generate_decode_lut() -> Box<[f32; 65536]> {
    let v: Vec<f32> = (0..65536u32)
        .map(|i| crate::rational_poly::srgb_to_linear_fast(i as f32 * (1.0 / 65535.0)))
        .collect();
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

/// Generate the sqrt-indexed encode LUT using the scalar f64-intermediate polynomial.
///
/// Entry `i` stores the sRGB u16 value for `linear = (i / 65536)²`.
/// Lookup uses `idx = (sqrt(linear) * 65536 + 0.5)`.
///
/// Uses `rational_poly::linear_to_srgb_fast` for platform-independent precision
/// (see `generate_decode_lut` for rationale).
#[doc(hidden)]
pub fn generate_encode_lut() -> Box<[u16; 65537]> {
    let lut: Vec<u16> = (0..ENCODE_LUT_N as u32)
        .map(|i| {
            let t = i as f32 * (1.0 / ENCODE_SQRT_SCALE);
            let linear = t * t;
            let srgb = crate::rational_poly::linear_to_srgb_fast(linear);
            (srgb * 65535.0 + 0.5) as u16
        })
        .collect();
    lut.into_boxed_slice().try_into().ok().unwrap()
}

/// Get the sqrt-indexed encode LUT, initializing on first call.
#[inline]
pub(crate) fn encode_lut() -> &'static [u16; 65537] {
    ENCODE_LUT.get_or_init(generate_encode_lut)
}
