//! 8×f32 `#[rite]` functions (AVX2+FMA on x86-64).
//!
//! All functions use `[f32; 8]` at the boundary — zero-cost transmute to/from
//! the underlying SIMD register. No `magetypes` types in the public API.
//!
//! Call these from inside your own `#[arcane]` function with a matching token.
//! They inline fully — no dispatch, no function-pointer indirection.

use archmage::rite;

pub use archmage::{NeonToken, Wasm128Token, X64V3Token};

use magetypes::simd::f32x8 as mt_f32x8;
use magetypes::simd::generic::f32x8 as gen_f32x8;

// sRGB transfer function constants (C0-continuous moxcms, matching rational polynomial)
const SRGB_LINEAR_THRESHOLD: f32 = 0.039_293_37;
const LINEAR_THRESHOLD: f32 = 0.003_041_282_6;
const LINEAR_SCALE: f32 = 1.0 / 12.92;
const TWELVE_92: f32 = 12.92;

// ============================================================================
// x8 functions — operate on [f32; 8]
// ============================================================================

/// Convert 8 sRGB values to linear. Input clamped to \[0, 1\].
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn srgb_to_linear_v3(token: X64V3Token, srgb: [f32; 8]) -> [f32; 8] {
    use crate::rational_poly::{S2L_P, S2L_Q};

    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let srgb = mt_f32x8::from_array(token, srgb).max(zero).min(one);

    let linear_result = srgb * mt_f32x8::splat(token, LINEAR_SCALE);

    // Rational polynomial P(x)/Q(x) via Horner's method
    let x = srgb;
    let yp = mt_f32x8::splat(token, S2L_P[4]).mul_add(x, mt_f32x8::splat(token, S2L_P[3]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, S2L_P[2]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, S2L_P[1]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, S2L_P[0]));

    let yq = mt_f32x8::splat(token, S2L_Q[4]).mul_add(x, mt_f32x8::splat(token, S2L_Q[3]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, S2L_Q[2]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, S2L_Q[1]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, S2L_Q[0]));

    let power_result = (yp / yq).min(one);

    let mask = srgb.simd_lt(mt_f32x8::splat(token, SRGB_LINEAR_THRESHOLD));
    mt_f32x8::blend(mask, linear_result, power_result).to_array()
}

/// Convert 8 linear values to sRGB. Input clamped to \[0, 1\].
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn linear_to_srgb_v3(token: X64V3Token, linear: [f32; 8]) -> [f32; 8] {
    use crate::rational_poly::{L2S_P, L2S_Q};

    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let linear = mt_f32x8::from_array(token, linear).max(zero).min(one);

    let linear_result = linear * mt_f32x8::splat(token, TWELVE_92);

    // sqrt transform + rational polynomial P(√x)/Q(√x) via Horner's method
    let x = linear.sqrt();
    let yp = mt_f32x8::splat(token, L2S_P[4]).mul_add(x, mt_f32x8::splat(token, L2S_P[3]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, L2S_P[2]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, L2S_P[1]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, L2S_P[0]));

    let yq = mt_f32x8::splat(token, L2S_Q[4]).mul_add(x, mt_f32x8::splat(token, L2S_Q[3]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, L2S_Q[2]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, L2S_Q[1]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, L2S_Q[0]));

    let power_result = (yp / yq).min(one);

    let mask = linear.simd_lt(mt_f32x8::splat(token, LINEAR_THRESHOLD));
    mt_f32x8::blend(mask, linear_result, power_result).to_array()
}

/// Convert 8 sRGB values to linear without clamping (extended range).
///
/// Uses abs+sign with a 6/6 rational polynomial fitted to \[0, 8\].
/// 5 ULP max in \[0,1\], u16-safe to |encoded| ≤ 6.18, u8-safe to 8.0.
/// Pure SIMD — no per-lane branching or scalar fallback.
#[rite]
pub fn srgb_to_linear_extended_v3(token: X64V3Token, srgb: [f32; 8]) -> [f32; 8] {
    use crate::rational_poly::{EXT_S2L_P as P, EXT_S2L_Q as Q};

    let zero = mt_f32x8::zero(token);
    let v = mt_f32x8::from_array(token, srgb);
    let neg_mask = v.simd_lt(zero);
    let abs_v = v.abs();

    let linear_result = abs_v * mt_f32x8::splat(token, LINEAR_SCALE);

    let x = abs_v;
    let yp = mt_f32x8::splat(token, P[6]).mul_add(x, mt_f32x8::splat(token, P[5]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, P[4]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, P[3]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, P[2]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, P[1]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, P[0]));

    let yq = mt_f32x8::splat(token, Q[6]).mul_add(x, mt_f32x8::splat(token, Q[5]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, Q[4]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, Q[3]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, Q[2]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, Q[1]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, Q[0]));

    let power_result = yp / yq;

    let thresh_mask = abs_v.simd_lt(mt_f32x8::splat(token, SRGB_LINEAR_THRESHOLD));
    let result = mt_f32x8::blend(thresh_mask, linear_result, power_result);
    let result = mt_f32x8::blend(neg_mask, -result, result);
    result.to_array()
}

/// Convert 8 linear values to sRGB without clamping (extended range).
///
/// Uses abs+sign with a 6/6 rational polynomial fitted on √x to \[0, 64\].
/// 5 ULP max in \[0,1\], u16-safe across the full \[0, 64\] domain.
/// Pure SIMD — no per-lane branching or scalar fallback.
#[rite]
pub fn linear_to_srgb_extended_v3(token: X64V3Token, linear: [f32; 8]) -> [f32; 8] {
    use crate::rational_poly::{EXT_L2S_P as P, EXT_L2S_Q as Q};

    let zero = mt_f32x8::zero(token);
    let v = mt_f32x8::from_array(token, linear);
    let neg_mask = v.simd_lt(zero);
    let abs_v = v.abs();

    let linear_result = abs_v * mt_f32x8::splat(token, TWELVE_92);

    let x = abs_v.sqrt();
    let yp = mt_f32x8::splat(token, P[6]).mul_add(x, mt_f32x8::splat(token, P[5]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, P[4]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, P[3]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, P[2]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, P[1]));
    let yp = yp.mul_add(x, mt_f32x8::splat(token, P[0]));

    let yq = mt_f32x8::splat(token, Q[6]).mul_add(x, mt_f32x8::splat(token, Q[5]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, Q[4]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, Q[3]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, Q[2]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, Q[1]));
    let yq = yq.mul_add(x, mt_f32x8::splat(token, Q[0]));

    let power_result = yp / yq;

    let thresh_mask = abs_v.simd_lt(mt_f32x8::splat(token, LINEAR_THRESHOLD));
    let result = mt_f32x8::blend(thresh_mask, linear_result, power_result);
    let result = mt_f32x8::blend(neg_mask, -result, result);
    result.to_array()
}

/// Convert 8 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn gamma_to_linear_v3(token: X64V3Token, encoded: [f32; 8], gamma: f32) -> [f32; 8] {
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let encoded = mt_f32x8::from_array(token, encoded).max(zero).min(one);
    encoded.pow_midp(gamma).to_array()
}

/// Convert 8 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn linear_to_gamma_v3(token: X64V3Token, linear: [f32; 8], gamma: f32) -> [f32; 8] {
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let linear = mt_f32x8::from_array(token, linear).max(zero).min(one);
    linear.pow_midp(1.0 / gamma).to_array()
}

// ============================================================================
// x8 LUT functions — u8↔f32
// ============================================================================

/// Convert 8 sRGB u8 values to linear f32 via 256-entry LUT.
///
/// Pure table lookup — no math. The 1KB table fits in L1 cache.
/// The token is accepted for API consistency; the operation itself
/// is scalar lookups assembled into an array.
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn srgb_u8_to_linear_v3(_token: X64V3Token, srgb: [u8; 8]) -> [f32; 8] {
    let lut = crate::const_luts::linear_table_8();
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

/// Convert sRGB u8 values to linear f32 using 8-wide LUT lookup.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn srgb_u8_to_linear_slice_v3(_token: X64V3Token, input: &[u8], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let lut = crate::const_luts::linear_table_8();
    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = [
            lut[inp[0] as usize],
            lut[inp[1] as usize],
            lut[inp[2] as usize],
            lut[inp[3] as usize],
            lut[inp[4] as usize],
            lut[inp[5] as usize],
            lut[inp[6] as usize],
            lut[inp[7] as usize],
        ];
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        *out = lut[*inp as usize];
    }
}

// ============================================================================
// x8 SIMD functions — u16↔f32 (rational polynomial, no LUT)
// ============================================================================

/// Convert 8 sRGB u16 values to linear f32 via rational polynomial.
///
/// Pure SIMD — no LUT, no cache footprint. Normalizes u16 to [0, 1] via
/// `*(1.0/65535.0)` and applies the C0-continuous `S2L_P/Q` rational
/// polynomial. Boundary values (0, 65535) map exactly to (0.0, 1.0).
///
/// The `#[magetypes(...)]` macro generates `_v3`, `_neon`, `_wasm128`
/// variants from this single body; f32x8 is native on V3 and polyfills
/// to 2×f32x4 on NEON / WASM128 (fully inlined). Call from inside an
/// `#[arcane]` function for zero-overhead inlining.
#[archmage::magetypes(v3, neon, wasm128)]
#[rite]
pub fn srgb_u16_to_linear(token: Token, srgb: [u16; 8]) -> [f32; 8] {
    use crate::rational_poly::{S2L_P, S2L_Q};
    #[allow(non_camel_case_types)]
    type f32x8 = gen_f32x8<Token>;

    // u16 → f32 / 65535: auto-vectorizes to vpmovzxwd + vcvtdq2ps + vmulps
    // (or the NEON / WASM equivalents).
    let mut f = [0.0f32; 8];
    for i in 0..8 {
        f[i] = srgb[i] as f32;
    }
    let v = f32x8::from_array(token, f) * f32x8::splat(token, 1.0 / 65535.0);
    let one = f32x8::splat(token, 1.0);

    let linear_result = v * f32x8::splat(token, LINEAR_SCALE);

    let yp = f32x8::splat(token, S2L_P[4]).mul_add(v, f32x8::splat(token, S2L_P[3]));
    let yp = yp.mul_add(v, f32x8::splat(token, S2L_P[2]));
    let yp = yp.mul_add(v, f32x8::splat(token, S2L_P[1]));
    let yp = yp.mul_add(v, f32x8::splat(token, S2L_P[0]));

    let yq = f32x8::splat(token, S2L_Q[4]).mul_add(v, f32x8::splat(token, S2L_Q[3]));
    let yq = yq.mul_add(v, f32x8::splat(token, S2L_Q[2]));
    let yq = yq.mul_add(v, f32x8::splat(token, S2L_Q[1]));
    let yq = yq.mul_add(v, f32x8::splat(token, S2L_Q[0]));

    let power_result = (yp / yq).min(one);

    let mask = v.simd_lt(f32x8::splat(token, SRGB_LINEAR_THRESHOLD));
    let result = f32x8::blend(mask, linear_result, power_result);
    // Match the scalar `srgb_to_linear_fast` early-exit so u16=65535 → 1.0
    // bit-exact: the normalized v rounds to 1 + 2^-16 in f32 and the
    // polynomial alone drifts a few ULP short of 1.0 at v=1.
    let ge_one = v.simd_ge(one);
    f32x8::blend(ge_one, one, result).to_array()
}

/// Convert 8 linear f32 values to sRGB u8 via LUT. Input clamped to \[0, 1\].
///
/// Uses a 4096-entry const LUT with bitmask indexing (`& 0xFFF`) for
/// provably safe bounds. SIMD accelerates the clamp and scale; lookups
/// are scalar from an L1-resident 4KB table.
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn linear_to_srgb_u8_v3(token: X64V3Token, linear: [f32; 8]) -> [u8; 8] {
    let zero = mt_f32x8::zero(token);
    let one = mt_f32x8::splat(token, 1.0);
    let linear = mt_f32x8::from_array(token, linear).max(zero).min(one);
    let scaled = linear * mt_f32x8::splat(token, 4095.0) + mt_f32x8::splat(token, 0.5);
    let arr = scaled.to_array();
    let lut = crate::const_luts::linear_to_srgb_u8();
    [
        lut[arr[0] as usize & 0xFFF],
        lut[arr[1] as usize & 0xFFF],
        lut[arr[2] as usize & 0xFFF],
        lut[arr[3] as usize & 0xFFF],
        lut[arr[4] as usize & 0xFFF],
        lut[arr[5] as usize & 0xFFF],
        lut[arr[6] as usize & 0xFFF],
        lut[arr[7] as usize & 0xFFF],
    ]
}

/// Convert 8 linear f32 values to sRGB u16 via rational polynomial.
///
/// Pure SIMD — no LUT, no cache footprint. Clamps to [0, 1], applies the
/// C0-continuous `L2S_P/Q` rational polynomial on √x, then scales by
/// 65535 + 0.5 and truncates to u16. Perfect roundtrip with the decode
/// polynomial. Inputs ≥ 1.0 map to 65535 bit-exact; inputs ≤ 0.0 map to 0.
///
/// The `#[magetypes(...)]` macro generates `_v3`, `_neon`, `_wasm128`
/// variants from this single body; f32x8 is native on V3 and polyfills
/// to 2×f32x4 on NEON / WASM128 (fully inlined). Call from inside an
/// `#[arcane]` function for zero-overhead inlining.
#[archmage::magetypes(v3, neon, wasm128)]
#[rite]
pub fn linear_to_srgb_u16(token: Token, linear: [f32; 8]) -> [u16; 8] {
    use crate::rational_poly::{L2S_P, L2S_Q};
    #[allow(non_camel_case_types)]
    type f32x8 = gen_f32x8<Token>;

    let v = f32x8::from_array(token, linear);
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let clamped = v.max(zero).min(one);

    let linear_result = clamped * f32x8::splat(token, TWELVE_92);

    let x = clamped.sqrt();
    let yp = f32x8::splat(token, L2S_P[4]).mul_add(x, f32x8::splat(token, L2S_P[3]));
    let yp = yp.mul_add(x, f32x8::splat(token, L2S_P[2]));
    let yp = yp.mul_add(x, f32x8::splat(token, L2S_P[1]));
    let yp = yp.mul_add(x, f32x8::splat(token, L2S_P[0]));

    let yq = f32x8::splat(token, L2S_Q[4]).mul_add(x, f32x8::splat(token, L2S_Q[3]));
    let yq = yq.mul_add(x, f32x8::splat(token, L2S_Q[2]));
    let yq = yq.mul_add(x, f32x8::splat(token, L2S_Q[1]));
    let yq = yq.mul_add(x, f32x8::splat(token, L2S_Q[0]));

    let power_result = (yp / yq).min(one);

    let thresh_mask = clamped.simd_lt(f32x8::splat(token, LINEAR_THRESHOLD));
    let srgb = f32x8::blend(thresh_mask, linear_result, power_result);

    // Force 1.0 for inputs >= 1.0 so the u16 endpoint lands exactly on 65535
    // (mirrors the clamp in linear_to_srgb_u16_slice_tier).
    let ge_one = v.simd_ge(one);
    let srgb = f32x8::blend(ge_one, one, srgb);

    let scaled = srgb.mul_add(f32x8::splat(token, 65535.0), f32x8::splat(token, 0.5));
    let idx = scaled.to_i32().to_array();
    [
        idx[0] as u16,
        idx[1] as u16,
        idx[2] as u16,
        idx[3] as u16,
        idx[4] as u16,
        idx[5] as u16,
        idx[6] as u16,
        idx[7] as u16,
    ]
}

// ============================================================================
// Slice functions — process &mut [f32] with x8 chunking
// ============================================================================

/// Convert sRGB f32 values to linear in-place using 8-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn srgb_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        *chunk = srgb_to_linear_v3(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

/// Convert linear f32 values to sRGB in-place using 8-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn linear_to_srgb_slice_v3(token: X64V3Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        *chunk = linear_to_srgb_v3(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_srgb(*v);
    }
}

/// Convert sRGB f32 values to linear in-place (extended range, no clamping).
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn srgb_to_linear_extended_slice_v3(token: X64V3Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        *chunk = srgb_to_linear_extended_v3(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::srgb_to_linear_extended(*v);
    }
}

/// Convert linear f32 values to sRGB in-place (extended range, no clamping).
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn linear_to_srgb_extended_slice_v3(token: X64V3Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        *chunk = linear_to_srgb_extended_v3(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_srgb_extended(*v);
    }
}

/// Convert gamma-encoded f32 values to linear in-place using 8-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn gamma_to_linear_slice_v3(token: X64V3Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        *chunk = gamma_to_linear_v3(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

/// Convert linear f32 values to gamma-encoded in-place using 8-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn linear_to_gamma_slice_v3(token: X64V3Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        *chunk = linear_to_gamma_v3(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_gamma(*v, gamma);
    }
}

/// Convert linear f32 values to sRGB u8 using 8-wide SIMD + LUT.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn linear_to_srgb_u8_slice_v3(token: X64V3Token, input: &[f32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());
    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = linear_to_srgb_u8_v3(token, *inp);
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        *out = crate::scalar::linear_to_srgb_u8(*inp);
    }
}

// ============================================================================
// Transfer function rites (behind `transfer` feature)
// ============================================================================

/// Convert 8 sRGB values to linear (rational polynomial, no powf).
#[cfg(feature = "transfer")]
#[rite]
pub fn tf_srgb_to_linear_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::srgb::srgb_to_linear_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

/// Convert 8 linear values to sRGB (rational polynomial, no powf).
#[cfg(feature = "transfer")]
#[rite]
pub fn tf_linear_to_srgb_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::srgb::linear_to_srgb_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

/// Convert 8 BT.709 encoded values to linear.
#[cfg(feature = "transfer")]
#[rite]
pub fn bt709_to_linear_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::bt709::bt709_to_linear_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

/// Convert 8 linear values to BT.709 encoded.
#[cfg(feature = "transfer")]
#[rite]
pub fn linear_to_bt709_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::bt709::linear_to_bt709_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

/// Convert 8 PQ signal values to linear.
#[cfg(feature = "transfer")]
#[rite]
pub fn pq_to_linear_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::pq::pq_to_linear_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

/// Convert 8 linear values to PQ signal.
#[cfg(feature = "transfer")]
#[rite]
pub fn linear_to_pq_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::pq::linear_to_pq_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

/// Convert 8 HLG signal values to linear.
#[cfg(feature = "transfer")]
#[rite]
pub fn hlg_to_linear_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::hlg::hlg_to_linear_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

/// Convert 8 linear values to HLG signal.
#[cfg(feature = "transfer")]
#[rite]
pub fn linear_to_hlg_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::hlg::linear_to_hlg_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

/// Convert sRGB f32 values to linear in-place, 8-wide (TF module version).
#[cfg(feature = "transfer")]
#[rite]
pub fn tf_srgb_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
    tf_slice_x8(
        values,
        |v| tf_srgb_to_linear_v3(token, v),
        crate::tf::srgb_to_linear,
    );
}

/// Convert linear f32 values to sRGB in-place, 8-wide (TF module version).
#[cfg(feature = "transfer")]
#[rite]
pub fn tf_linear_to_srgb_slice_v3(token: X64V3Token, values: &mut [f32]) {
    tf_slice_x8(
        values,
        |v| tf_linear_to_srgb_v3(token, v),
        crate::tf::linear_to_srgb,
    );
}

/// Convert BT.709 f32 values to linear in-place, 8-wide.
#[cfg(feature = "transfer")]
#[rite]
pub fn bt709_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
    tf_slice_x8(
        values,
        |v| bt709_to_linear_v3(token, v),
        crate::tf::bt709_to_linear,
    );
}

/// Convert linear f32 values to BT.709 in-place, 8-wide.
#[cfg(feature = "transfer")]
#[rite]
pub fn linear_to_bt709_slice_v3(token: X64V3Token, values: &mut [f32]) {
    tf_slice_x8(
        values,
        |v| linear_to_bt709_v3(token, v),
        crate::tf::linear_to_bt709,
    );
}

/// Convert PQ f32 values to linear in-place, 8-wide.
#[cfg(feature = "transfer")]
#[rite]
pub fn pq_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
    tf_slice_x8(
        values,
        |v| pq_to_linear_v3(token, v),
        crate::tf::pq_to_linear,
    );
}

/// Convert linear f32 values to PQ in-place, 8-wide.
#[cfg(feature = "transfer")]
#[rite]
pub fn linear_to_pq_slice_v3(token: X64V3Token, values: &mut [f32]) {
    tf_slice_x8(
        values,
        |v| linear_to_pq_v3(token, v),
        crate::tf::linear_to_pq,
    );
}

/// Convert HLG f32 values to linear in-place, 8-wide.
#[cfg(feature = "transfer")]
#[rite]
pub fn hlg_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
    tf_slice_x8(
        values,
        |v| hlg_to_linear_v3(token, v),
        crate::tf::hlg_to_linear,
    );
}

/// Convert linear f32 values to HLG in-place, 8-wide.
#[cfg(feature = "transfer")]
#[rite]
pub fn linear_to_hlg_slice_v3(token: X64V3Token, values: &mut [f32]) {
    tf_slice_x8(
        values,
        |v| linear_to_hlg_v3(token, v),
        crate::tf::linear_to_hlg,
    );
}

#[cfg(feature = "transfer")]
#[inline(always)]
fn tf_slice_x8(
    values: &mut [f32],
    tf_x8: impl Fn([f32; 8]) -> [f32; 8],
    tf_scalar: fn(f32) -> f32,
) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();
    for chunk in chunks {
        *chunk = tf_x8(*chunk);
    }
    for v in remainder {
        *v = tf_scalar(*v);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use archmage::SimdToken;

    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    fn get_token() -> Option<X64V3Token> {
        X64V3Token::try_new()
    }

    // We need an #[arcane] wrapper to safely call #[rite] functions in tests.
    #[archmage::arcane]
    fn call_srgb_to_linear(token: X64V3Token, input: [f32; 8]) -> [f32; 8] {
        srgb_to_linear_v3(token, input)
    }

    #[archmage::arcane]
    fn call_linear_to_srgb(token: X64V3Token, input: [f32; 8]) -> [f32; 8] {
        linear_to_srgb_v3(token, input)
    }

    #[archmage::arcane]
    fn call_srgb_to_linear_slice(token: X64V3Token, values: &mut [f32]) {
        srgb_to_linear_slice_v3(token, values);
    }

    #[archmage::arcane]
    fn call_linear_to_srgb_slice(token: X64V3Token, values: &mut [f32]) {
        linear_to_srgb_slice_v3(token, values);
    }

    #[archmage::arcane]
    fn call_linear_to_srgb_u8(token: X64V3Token, input: [f32; 8]) -> [u8; 8] {
        linear_to_srgb_u8_v3(token, input)
    }

    #[test]
    fn test_x8_linear_to_srgb_u8() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
            return;
        };

        let input = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
        let result = call_linear_to_srgb_u8(token, input);

        for (i, (&got, &inp)) in result.iter().zip(input.iter()).enumerate() {
            let expected = crate::scalar::linear_to_srgb_u8(inp);
            assert!(
                (got as i32 - expected as i32).abs() <= 1,
                "u8 mismatch at {}: got {}, expected {} (input={})",
                i,
                got,
                expected,
                inp
            );
        }
    }

    #[test]
    fn test_x8_srgb_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
            return;
        };

        let input = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
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
    fn test_x8_matches_scalar() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
            return;
        };

        let input = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
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
    fn test_slice_matches_scalar() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
            return;
        };

        let mut values: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let expected: Vec<f32> = values
            .iter()
            .map(|&v| crate::scalar::srgb_to_linear(v))
            .collect();

        call_srgb_to_linear_slice(token, &mut values);

        for (i, (&got, &exp)) in values.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "mismatch at {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_slice_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX2+FMA not available");
            return;
        };

        let mut values: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
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
