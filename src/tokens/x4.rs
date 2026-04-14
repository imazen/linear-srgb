//! 4×f32 `#[rite]` functions (128-bit SIMD on all platforms).
//!
//! All functions use `[f32; 4]` at the boundary — zero-cost transmute to/from
//! the underlying SIMD register. No `magetypes` types in the public API.
//!
//! Call these from inside your own `#[arcane]` function with a matching token.
//! They inline fully — no dispatch, no function-pointer indirection.
//!
//! # Suffix convention
//!
//! - `_v3` — requires `X64V3Token` (x86-64 AVX2+FMA, using 128-bit SSE registers)
//! - `_neon` — requires [`NeonToken`](archmage::NeonToken) (AArch64 NEON)
//! - `_wasm128` — requires [`Wasm128Token`](archmage::Wasm128Token) (WebAssembly SIMD128)
//!
//! # When to use x4 vs x8 on x86-64
//!
//! On x86-64 with AVX2+FMA, both `x4` and `x8` are available. Use `x4` when
//! your data naturally comes in groups of 4 (e.g., RGBA pixels) or when writing
//! portable code that mirrors the NEON/WASM API shape. Use `x8` for maximum
//! throughput on contiguous f32 slices.

use archmage::rite;

#[cfg(target_arch = "x86_64")]
pub use archmage::X64V3Token;

#[cfg(target_arch = "aarch64")]
pub use archmage::NeonToken;

#[cfg(target_arch = "wasm32")]
pub use archmage::Wasm128Token;

use magetypes::simd::f32x4 as mt_f32x4;
use magetypes::simd::generic::f32x4 as gen_f32x4;

// sRGB transfer function constants (C0-continuous moxcms, matching rational polynomial)
const SRGB_LINEAR_THRESHOLD: f32 = 0.039_293_37;
const LINEAR_THRESHOLD: f32 = 0.003_041_282_6;
const LINEAR_SCALE: f32 = 1.0 / 12.92;
const TWELVE_92: f32 = 12.92;

// ============================================================================
// x86-64 V3 (AVX2+FMA) — 4×f32 with X64V3Token
// ============================================================================

/// Convert 4 sRGB values to linear. Input clamped to \[0, 1\].
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Uses 128-bit SSE registers (VEX-encoded) within the AVX2+FMA context.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn srgb_to_linear_v3(token: X64V3Token, srgb: [f32; 4]) -> [f32; 4] {
    use crate::rational_poly::{S2L_P, S2L_Q};

    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let srgb_v = mt_f32x4::from_array(token, srgb);
    let clamped = srgb_v.max(zero).min(one);

    let linear_result = clamped * mt_f32x4::splat(token, LINEAR_SCALE);

    // Rational polynomial P(x)/Q(x) via Horner's method
    let x = clamped;
    let yp = mt_f32x4::splat(token, S2L_P[4]).mul_add(x, mt_f32x4::splat(token, S2L_P[3]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[2]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[1]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[0]));

    let yq = mt_f32x4::splat(token, S2L_Q[4]).mul_add(x, mt_f32x4::splat(token, S2L_Q[3]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[2]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[1]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[0]));

    let power_result = (yp / yq).min(one);

    let mask = clamped.simd_lt(mt_f32x4::splat(token, SRGB_LINEAR_THRESHOLD));
    let result = mt_f32x4::blend(mask, linear_result, power_result);
    // Force exact 1.0 for inputs >= 1.0 (polynomial may undershoot at boundary)
    let ge_one = srgb_v.simd_ge(one);
    mt_f32x4::blend(ge_one, one, result).to_array()
}

/// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_srgb_v3(token: X64V3Token, linear: [f32; 4]) -> [f32; 4] {
    use crate::rational_poly::{L2S_P, L2S_Q};

    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let linear_v = mt_f32x4::from_array(token, linear);
    let clamped = linear_v.max(zero).min(one);

    let linear_result = clamped * mt_f32x4::splat(token, TWELVE_92);

    // sqrt transform + rational polynomial P(√x)/Q(√x) via Horner's method
    let x = clamped.sqrt();
    let yp = mt_f32x4::splat(token, L2S_P[4]).mul_add(x, mt_f32x4::splat(token, L2S_P[3]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[2]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[1]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[0]));

    let yq = mt_f32x4::splat(token, L2S_Q[4]).mul_add(x, mt_f32x4::splat(token, L2S_Q[3]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[2]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[1]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[0]));

    let power_result = (yp / yq).min(one);

    let mask = clamped.simd_lt(mt_f32x4::splat(token, LINEAR_THRESHOLD));
    let result = mt_f32x4::blend(mask, linear_result, power_result);
    // Force exact 1.0 for inputs >= 1.0 (polynomial may undershoot at boundary)
    let ge_one = linear_v.simd_ge(one);
    mt_f32x4::blend(ge_one, one, result).to_array()
}

/// Convert 4 sRGB values to linear without clamping (extended range).
///
/// 6/6 rational polynomial fitted to \[0, 8\]. 5 ULP in \[0,1\], u16-safe to ~4.2.
#[archmage::magetypes(v3, neon, wasm128)]
#[rite]
pub fn srgb_to_linear_extended(token: Token, srgb: [f32; 4]) -> [f32; 4] {
    use crate::rational_poly::{EXT_S2L_P as P, EXT_S2L_Q as Q};
    #[allow(non_camel_case_types)]
    type f32x4 = gen_f32x4<Token>;
    let zero = f32x4::zero(token);
    let v = f32x4::from_array(token, srgb);
    let neg_mask = v.simd_lt(zero);
    let abs_v = v.abs();
    let linear_result = abs_v * f32x4::splat(token, LINEAR_SCALE);
    let x = abs_v;
    let yp = f32x4::splat(token, P[6]).mul_add(x, f32x4::splat(token, P[5]));
    let yp = yp.mul_add(x, f32x4::splat(token, P[4]));
    let yp = yp.mul_add(x, f32x4::splat(token, P[3]));
    let yp = yp.mul_add(x, f32x4::splat(token, P[2]));
    let yp = yp.mul_add(x, f32x4::splat(token, P[1]));
    let yp = yp.mul_add(x, f32x4::splat(token, P[0]));
    let yq = f32x4::splat(token, Q[6]).mul_add(x, f32x4::splat(token, Q[5]));
    let yq = yq.mul_add(x, f32x4::splat(token, Q[4]));
    let yq = yq.mul_add(x, f32x4::splat(token, Q[3]));
    let yq = yq.mul_add(x, f32x4::splat(token, Q[2]));
    let yq = yq.mul_add(x, f32x4::splat(token, Q[1]));
    let yq = yq.mul_add(x, f32x4::splat(token, Q[0]));
    let power_result = yp / yq;
    let thresh_mask = abs_v.simd_lt(f32x4::splat(token, SRGB_LINEAR_THRESHOLD));
    let result = f32x4::blend(thresh_mask, linear_result, power_result);
    f32x4::blend(neg_mask, -result, result).to_array()
}

/// Convert 4 linear values to sRGB without clamping (extended range).
///
/// 6/6 rational polynomial fitted on √x to \[0, 64\]. 5 ULP in \[0,1\], u16-safe to 64.
#[archmage::magetypes(v3, neon, wasm128)]
#[rite]
pub fn linear_to_srgb_extended(token: Token, linear: [f32; 4]) -> [f32; 4] {
    use crate::rational_poly::{EXT_L2S_P as P, EXT_L2S_Q as Q};
    #[allow(non_camel_case_types)]
    type f32x4 = gen_f32x4<Token>;
    let zero = f32x4::zero(token);
    let v = f32x4::from_array(token, linear);
    let neg_mask = v.simd_lt(zero);
    let abs_v = v.abs();
    let linear_result = abs_v * f32x4::splat(token, TWELVE_92);
    let x = abs_v.sqrt();
    let yp = f32x4::splat(token, P[6]).mul_add(x, f32x4::splat(token, P[5]));
    let yp = yp.mul_add(x, f32x4::splat(token, P[4]));
    let yp = yp.mul_add(x, f32x4::splat(token, P[3]));
    let yp = yp.mul_add(x, f32x4::splat(token, P[2]));
    let yp = yp.mul_add(x, f32x4::splat(token, P[1]));
    let yp = yp.mul_add(x, f32x4::splat(token, P[0]));
    let yq = f32x4::splat(token, Q[6]).mul_add(x, f32x4::splat(token, Q[5]));
    let yq = yq.mul_add(x, f32x4::splat(token, Q[4]));
    let yq = yq.mul_add(x, f32x4::splat(token, Q[3]));
    let yq = yq.mul_add(x, f32x4::splat(token, Q[2]));
    let yq = yq.mul_add(x, f32x4::splat(token, Q[1]));
    let yq = yq.mul_add(x, f32x4::splat(token, Q[0]));
    let power_result = yp / yq;
    let thresh_mask = abs_v.simd_lt(f32x4::splat(token, LINEAR_THRESHOLD));
    let result = f32x4::blend(thresh_mask, linear_result, power_result);
    f32x4::blend(neg_mask, -result, result).to_array()
}

/// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn gamma_to_linear_v3(token: X64V3Token, encoded: [f32; 4], gamma: f32) -> [f32; 4] {
    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let encoded = mt_f32x4::from_array(token, encoded).max(zero).min(one);
    encoded.pow_midp(gamma).to_array()
}

/// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_gamma_v3(token: X64V3Token, linear: [f32; 4], gamma: f32) -> [f32; 4] {
    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let linear = mt_f32x4::from_array(token, linear).max(zero).min(one);
    linear.pow_midp(1.0 / gamma).to_array()
}

/// Convert sRGB f32 values to linear in-place using 4-wide SSE SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn srgb_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = srgb_to_linear_v3(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

/// Convert linear f32 values to sRGB in-place using 4-wide SSE SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_srgb_slice_v3(token: X64V3Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = linear_to_srgb_v3(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_srgb(*v);
    }
}

/// Convert gamma-encoded f32 values to linear in-place using 4-wide SSE SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn gamma_to_linear_slice_v3(token: X64V3Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = gamma_to_linear_v3(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

/// Convert linear f32 values to gamma-encoded in-place using 4-wide SSE SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_gamma_slice_v3(token: X64V3Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = linear_to_gamma_v3(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_gamma(*v, gamma);
    }
}

// ============================================================================
// AArch64 NEON — 4×f32 with NeonToken token
// ============================================================================

/// Convert 4 sRGB values to linear. Input clamped to \[0, 1\].
///
/// The `NeonToken` parameter proves NEON support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn srgb_to_linear_neon(token: NeonToken, srgb: [f32; 4]) -> [f32; 4] {
    use crate::rational_poly::{S2L_P, S2L_Q};

    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let srgb_v = mt_f32x4::from_array(token, srgb);
    let clamped = srgb_v.max(zero).min(one);

    let linear_result = clamped * mt_f32x4::splat(token, LINEAR_SCALE);

    // Rational polynomial P(x)/Q(x) via Horner's method
    let x = clamped;
    let yp = mt_f32x4::splat(token, S2L_P[4]).mul_add(x, mt_f32x4::splat(token, S2L_P[3]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[2]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[1]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[0]));

    let yq = mt_f32x4::splat(token, S2L_Q[4]).mul_add(x, mt_f32x4::splat(token, S2L_Q[3]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[2]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[1]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[0]));

    let power_result = (yp / yq).min(one);

    let mask = clamped.simd_lt(mt_f32x4::splat(token, SRGB_LINEAR_THRESHOLD));
    let result = mt_f32x4::blend(mask, linear_result, power_result);
    // Force exact 1.0 for inputs >= 1.0 (polynomial may undershoot at boundary)
    let ge_one = srgb_v.simd_ge(one);
    mt_f32x4::blend(ge_one, one, result).to_array()
}

/// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn linear_to_srgb_neon(token: NeonToken, linear: [f32; 4]) -> [f32; 4] {
    use crate::rational_poly::{L2S_P, L2S_Q};

    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let linear_v = mt_f32x4::from_array(token, linear);
    let clamped = linear_v.max(zero).min(one);

    let linear_result = clamped * mt_f32x4::splat(token, TWELVE_92);

    // sqrt transform + rational polynomial P(√x)/Q(√x) via Horner's method
    let x = clamped.sqrt();
    let yp = mt_f32x4::splat(token, L2S_P[4]).mul_add(x, mt_f32x4::splat(token, L2S_P[3]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[2]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[1]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[0]));

    let yq = mt_f32x4::splat(token, L2S_Q[4]).mul_add(x, mt_f32x4::splat(token, L2S_Q[3]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[2]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[1]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[0]));

    let power_result = (yp / yq).min(one);

    let mask = clamped.simd_lt(mt_f32x4::splat(token, LINEAR_THRESHOLD));
    let result = mt_f32x4::blend(mask, linear_result, power_result);
    // Force exact 1.0 for inputs >= 1.0 (polynomial may undershoot at boundary)
    let ge_one = linear_v.simd_ge(one);
    mt_f32x4::blend(ge_one, one, result).to_array()
}

/// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn gamma_to_linear_neon(token: NeonToken, encoded: [f32; 4], gamma: f32) -> [f32; 4] {
    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let encoded = mt_f32x4::from_array(token, encoded).max(zero).min(one);
    encoded.pow_midp(gamma).to_array()
}

/// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn linear_to_gamma_neon(token: NeonToken, linear: [f32; 4], gamma: f32) -> [f32; 4] {
    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let linear = mt_f32x4::from_array(token, linear).max(zero).min(one);
    linear.pow_midp(1.0 / gamma).to_array()
}

/// Convert sRGB f32 values to linear in-place using 4-wide NEON SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn srgb_to_linear_slice_neon(token: NeonToken, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = srgb_to_linear_neon(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

/// Convert linear f32 values to sRGB in-place using 4-wide NEON SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn linear_to_srgb_slice_neon(token: NeonToken, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = linear_to_srgb_neon(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_srgb(*v);
    }
}

/// Convert gamma-encoded f32 values to linear in-place using 4-wide NEON SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn gamma_to_linear_slice_neon(token: NeonToken, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = gamma_to_linear_neon(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

/// Convert linear f32 values to gamma-encoded in-place using 4-wide NEON SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[rite]
pub fn linear_to_gamma_slice_neon(token: NeonToken, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = linear_to_gamma_neon(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_gamma(*v, gamma);
    }
}

// ============================================================================
// WebAssembly SIMD128 — 4×f32 with Wasm128Token
// ============================================================================

/// Convert 4 sRGB values to linear. Input clamped to \[0, 1\].
///
/// The `Wasm128Token` parameter proves SIMD128 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn srgb_to_linear_wasm128(token: Wasm128Token, srgb: [f32; 4]) -> [f32; 4] {
    use crate::rational_poly::{S2L_P, S2L_Q};

    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let srgb_v = mt_f32x4::from_array(token, srgb);
    let clamped = srgb_v.max(zero).min(one);

    let linear_result = clamped * mt_f32x4::splat(token, LINEAR_SCALE);

    // Rational polynomial P(x)/Q(x) via Horner's method
    let x = clamped;
    let yp = mt_f32x4::splat(token, S2L_P[4]).mul_add(x, mt_f32x4::splat(token, S2L_P[3]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[2]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[1]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, S2L_P[0]));

    let yq = mt_f32x4::splat(token, S2L_Q[4]).mul_add(x, mt_f32x4::splat(token, S2L_Q[3]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[2]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[1]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, S2L_Q[0]));

    let power_result = (yp / yq).min(one);

    let mask = clamped.simd_lt(mt_f32x4::splat(token, SRGB_LINEAR_THRESHOLD));
    let result = mt_f32x4::blend(mask, linear_result, power_result);
    // Force exact 1.0 for inputs >= 1.0 (polynomial may undershoot at boundary)
    let ge_one = srgb_v.simd_ge(one);
    mt_f32x4::blend(ge_one, one, result).to_array()
}

/// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn linear_to_srgb_wasm128(token: Wasm128Token, linear: [f32; 4]) -> [f32; 4] {
    use crate::rational_poly::{L2S_P, L2S_Q};

    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let linear_v = mt_f32x4::from_array(token, linear);
    let clamped = linear_v.max(zero).min(one);

    let linear_result = clamped * mt_f32x4::splat(token, TWELVE_92);

    // sqrt transform + rational polynomial P(√x)/Q(√x) via Horner's method
    let x = clamped.sqrt();
    let yp = mt_f32x4::splat(token, L2S_P[4]).mul_add(x, mt_f32x4::splat(token, L2S_P[3]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[2]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[1]));
    let yp = yp.mul_add(x, mt_f32x4::splat(token, L2S_P[0]));

    let yq = mt_f32x4::splat(token, L2S_Q[4]).mul_add(x, mt_f32x4::splat(token, L2S_Q[3]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[2]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[1]));
    let yq = yq.mul_add(x, mt_f32x4::splat(token, L2S_Q[0]));

    let power_result = (yp / yq).min(one);

    let mask = clamped.simd_lt(mt_f32x4::splat(token, LINEAR_THRESHOLD));
    let result = mt_f32x4::blend(mask, linear_result, power_result);
    // Force exact 1.0 for inputs >= 1.0 (polynomial may undershoot at boundary)
    let ge_one = linear_v.simd_ge(one);
    mt_f32x4::blend(ge_one, one, result).to_array()
}

/// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn gamma_to_linear_wasm128(token: Wasm128Token, encoded: [f32; 4], gamma: f32) -> [f32; 4] {
    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let encoded = mt_f32x4::from_array(token, encoded).max(zero).min(one);
    encoded.pow_midp(gamma).to_array()
}

/// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn linear_to_gamma_wasm128(token: Wasm128Token, linear: [f32; 4], gamma: f32) -> [f32; 4] {
    let zero = mt_f32x4::zero(token);
    let one = mt_f32x4::splat(token, 1.0);
    let linear = mt_f32x4::from_array(token, linear).max(zero).min(one);
    linear.pow_midp(1.0 / gamma).to_array()
}

/// Convert sRGB f32 values to linear in-place using 4-wide WASM SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn srgb_to_linear_slice_wasm128(token: Wasm128Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = srgb_to_linear_wasm128(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::srgb_to_linear(*v);
    }
}

/// Convert linear f32 values to sRGB in-place using 4-wide WASM SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn linear_to_srgb_slice_wasm128(token: Wasm128Token, values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = linear_to_srgb_wasm128(token, *chunk);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_srgb(*v);
    }
}

/// Convert gamma-encoded f32 values to linear in-place using 4-wide WASM SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn gamma_to_linear_slice_wasm128(token: Wasm128Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = gamma_to_linear_wasm128(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::gamma_to_linear(*v, gamma);
    }
}

/// Convert linear f32 values to gamma-encoded in-place using 4-wide WASM SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[rite]
pub fn linear_to_gamma_slice_wasm128(token: Wasm128Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        *chunk = linear_to_gamma_wasm128(token, *chunk, gamma);
    }

    for v in remainder {
        *v = crate::scalar::linear_to_gamma(*v, gamma);
    }
}

// ============================================================================
// Transfer function rites — x86-64 V3 (behind `transfer` feature)
// ============================================================================

macro_rules! x86_tf_rite {
    ($name:ident, $inner:path) => {
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        #[cfg(all(feature = "transfer", target_arch = "x86_64"))]
        #[rite]
        pub fn $name(token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            $inner(token, gen_f32x4::from_array(token, v)).to_array()
        }
    };
}

x86_tf_rite!(tf_srgb_to_linear_v3, crate::tf::srgb::srgb_to_linear_x4);
x86_tf_rite!(tf_linear_to_srgb_v3, crate::tf::srgb::linear_to_srgb_x4);
x86_tf_rite!(bt709_to_linear_v3, crate::tf::bt709::bt709_to_linear_x4);
x86_tf_rite!(linear_to_bt709_v3, crate::tf::bt709::linear_to_bt709_x4);
x86_tf_rite!(pq_to_linear_v3, crate::tf::pq::pq_to_linear_x4);
x86_tf_rite!(linear_to_pq_v3, crate::tf::pq::linear_to_pq_x4);
x86_tf_rite!(hlg_to_linear_v3, crate::tf::hlg::hlg_to_linear_x4);
x86_tf_rite!(linear_to_hlg_v3, crate::tf::hlg::linear_to_hlg_x4);

macro_rules! x86_tf_slice_rite {
    ($name:ident, $rite:ident, $scalar:path) => {
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        #[cfg(all(feature = "transfer", target_arch = "x86_64"))]
        #[rite]
        pub fn $name(token: X64V3Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = $rite(token, *chunk);
            }
            for v in remainder {
                *v = $scalar(*v);
            }
        }
    };
}

x86_tf_slice_rite!(
    tf_srgb_to_linear_slice_v3,
    tf_srgb_to_linear_v3,
    crate::tf::srgb_to_linear
);
x86_tf_slice_rite!(
    tf_linear_to_srgb_slice_v3,
    tf_linear_to_srgb_v3,
    crate::tf::linear_to_srgb
);
x86_tf_slice_rite!(
    bt709_to_linear_slice_v3,
    bt709_to_linear_v3,
    crate::tf::bt709_to_linear
);
x86_tf_slice_rite!(
    linear_to_bt709_slice_v3,
    linear_to_bt709_v3,
    crate::tf::linear_to_bt709
);
x86_tf_slice_rite!(
    pq_to_linear_slice_v3,
    pq_to_linear_v3,
    crate::tf::pq_to_linear
);
x86_tf_slice_rite!(
    linear_to_pq_slice_v3,
    linear_to_pq_v3,
    crate::tf::linear_to_pq
);
x86_tf_slice_rite!(
    hlg_to_linear_slice_v3,
    hlg_to_linear_v3,
    crate::tf::hlg_to_linear
);
x86_tf_slice_rite!(
    linear_to_hlg_slice_v3,
    linear_to_hlg_v3,
    crate::tf::linear_to_hlg
);

// ============================================================================
// Transfer function rites — AArch64 NEON (behind `transfer` feature)
// ============================================================================

macro_rules! neon_tf_rite {
    ($name:ident, $inner:path) => {
        /// Transfer function rite (4×f32, AArch64 NEON). Requires `transfer` feature.
        #[cfg(all(feature = "transfer", target_arch = "aarch64"))]
        #[rite]
        pub fn $name(token: NeonToken, v: [f32; 4]) -> [f32; 4] {
            $inner(token, gen_f32x4::from_array(token, v)).to_array()
        }
    };
}

neon_tf_rite!(tf_srgb_to_linear_neon, crate::tf::srgb::srgb_to_linear_x4);
neon_tf_rite!(tf_linear_to_srgb_neon, crate::tf::srgb::linear_to_srgb_x4);
neon_tf_rite!(bt709_to_linear_neon, crate::tf::bt709::bt709_to_linear_x4);
neon_tf_rite!(linear_to_bt709_neon, crate::tf::bt709::linear_to_bt709_x4);
neon_tf_rite!(pq_to_linear_neon, crate::tf::pq::pq_to_linear_x4);
neon_tf_rite!(linear_to_pq_neon, crate::tf::pq::linear_to_pq_x4);
neon_tf_rite!(hlg_to_linear_neon, crate::tf::hlg::hlg_to_linear_x4);
neon_tf_rite!(linear_to_hlg_neon, crate::tf::hlg::linear_to_hlg_x4);

macro_rules! neon_tf_slice_rite {
    ($name:ident, $rite:ident, $scalar:path) => {
        /// Transfer function slice rite (4×f32, AArch64 NEON). Requires `transfer` feature.
        #[cfg(all(feature = "transfer", target_arch = "aarch64"))]
        #[rite]
        pub fn $name(token: NeonToken, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = $rite(token, *chunk);
            }
            for v in remainder {
                *v = $scalar(*v);
            }
        }
    };
}

neon_tf_slice_rite!(
    tf_srgb_to_linear_slice_neon,
    tf_srgb_to_linear_neon,
    crate::tf::srgb_to_linear
);
neon_tf_slice_rite!(
    tf_linear_to_srgb_slice_neon,
    tf_linear_to_srgb_neon,
    crate::tf::linear_to_srgb
);
neon_tf_slice_rite!(
    bt709_to_linear_slice_neon,
    bt709_to_linear_neon,
    crate::tf::bt709_to_linear
);
neon_tf_slice_rite!(
    linear_to_bt709_slice_neon,
    linear_to_bt709_neon,
    crate::tf::linear_to_bt709
);
neon_tf_slice_rite!(
    pq_to_linear_slice_neon,
    pq_to_linear_neon,
    crate::tf::pq_to_linear
);
neon_tf_slice_rite!(
    linear_to_pq_slice_neon,
    linear_to_pq_neon,
    crate::tf::linear_to_pq
);
neon_tf_slice_rite!(
    hlg_to_linear_slice_neon,
    hlg_to_linear_neon,
    crate::tf::hlg_to_linear
);
neon_tf_slice_rite!(
    linear_to_hlg_slice_neon,
    linear_to_hlg_neon,
    crate::tf::linear_to_hlg
);

// ============================================================================
// Transfer function rites — WebAssembly SIMD128 (behind `transfer` feature)
// ============================================================================

macro_rules! wasm_tf_rite {
    ($name:ident, $inner:path) => {
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        #[cfg(all(feature = "transfer", target_arch = "wasm32"))]
        #[rite]
        pub fn $name(token: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            $inner(token, gen_f32x4::from_array(token, v)).to_array()
        }
    };
}

wasm_tf_rite!(
    tf_srgb_to_linear_wasm128,
    crate::tf::srgb::srgb_to_linear_x4
);
wasm_tf_rite!(
    tf_linear_to_srgb_wasm128,
    crate::tf::srgb::linear_to_srgb_x4
);
wasm_tf_rite!(
    bt709_to_linear_wasm128,
    crate::tf::bt709::bt709_to_linear_x4
);
wasm_tf_rite!(
    linear_to_bt709_wasm128,
    crate::tf::bt709::linear_to_bt709_x4
);
wasm_tf_rite!(pq_to_linear_wasm128, crate::tf::pq::pq_to_linear_x4);
wasm_tf_rite!(linear_to_pq_wasm128, crate::tf::pq::linear_to_pq_x4);
wasm_tf_rite!(hlg_to_linear_wasm128, crate::tf::hlg::hlg_to_linear_x4);
wasm_tf_rite!(linear_to_hlg_wasm128, crate::tf::hlg::linear_to_hlg_x4);

macro_rules! wasm_tf_slice_rite {
    ($name:ident, $rite:ident, $scalar:path) => {
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        #[cfg(all(feature = "transfer", target_arch = "wasm32"))]
        #[rite]
        pub fn $name(token: Wasm128Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = $rite(token, *chunk);
            }
            for v in remainder {
                *v = $scalar(*v);
            }
        }
    };
}

wasm_tf_slice_rite!(
    tf_srgb_to_linear_slice_wasm128,
    tf_srgb_to_linear_wasm128,
    crate::tf::srgb_to_linear
);
wasm_tf_slice_rite!(
    tf_linear_to_srgb_slice_wasm128,
    tf_linear_to_srgb_wasm128,
    crate::tf::linear_to_srgb
);
wasm_tf_slice_rite!(
    bt709_to_linear_slice_wasm128,
    bt709_to_linear_wasm128,
    crate::tf::bt709_to_linear
);
wasm_tf_slice_rite!(
    linear_to_bt709_slice_wasm128,
    linear_to_bt709_wasm128,
    crate::tf::linear_to_bt709
);
wasm_tf_slice_rite!(
    pq_to_linear_slice_wasm128,
    pq_to_linear_wasm128,
    crate::tf::pq_to_linear
);
wasm_tf_slice_rite!(
    linear_to_pq_slice_wasm128,
    linear_to_pq_wasm128,
    crate::tf::linear_to_pq
);
wasm_tf_slice_rite!(
    hlg_to_linear_slice_wasm128,
    hlg_to_linear_wasm128,
    crate::tf::hlg_to_linear
);
wasm_tf_slice_rite!(
    linear_to_hlg_slice_wasm128,
    linear_to_hlg_wasm128,
    crate::tf::linear_to_hlg
);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests_x86 {
    use super::*;
    use archmage::SimdToken;

    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    fn get_token() -> Option<X64V3Token> {
        X64V3Token::try_new()
    }

    #[archmage::arcane]
    fn call_srgb_to_linear(token: X64V3Token, input: [f32; 4]) -> [f32; 4] {
        srgb_to_linear_v3(token, input)
    }

    #[archmage::arcane]
    fn call_linear_to_srgb(token: X64V3Token, input: [f32; 4]) -> [f32; 4] {
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

    #[test]
    fn test_x4_srgb_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: X64V3 not available");
            return;
        };

        let input = [0.0, 0.3, 0.7, 1.0];
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
    fn test_x4_matches_scalar() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: X64V3 not available");
            return;
        };

        let input = [0.0, 0.3, 0.7, 1.0];
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
    fn test_x4_matches_x8() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: X64V3 not available");
            return;
        };

        // Verify x4 produces same results as x8 for the same inputs
        let input = [0.1, 0.4, 0.7, 0.95];
        let x4_result = call_srgb_to_linear(token, input);

        // Compare to scalar (which x8 also matches)
        for (i, (&got, &inp)) in x4_result.iter().zip(input.iter()).enumerate() {
            let expected = crate::rational_poly::srgb_to_linear_fast(inp);
            assert!(
                (got - expected).abs() < 1e-6,
                "x4 vs rational_poly mismatch at {}: got {}, expected {}",
                i,
                got,
                expected
            );
        }
    }

    #[test]
    fn test_slice_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: X64V3 not available");
            return;
        };

        let mut values: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
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

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests_aarch64 {
    use super::*;
    use archmage::SimdToken;

    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    fn get_token() -> Option<NeonToken> {
        NeonToken::try_new()
    }

    #[archmage::arcane]
    fn call_srgb_to_linear(token: NeonToken, input: [f32; 4]) -> [f32; 4] {
        srgb_to_linear_neon(token, input)
    }

    #[archmage::arcane]
    fn call_linear_to_srgb(token: NeonToken, input: [f32; 4]) -> [f32; 4] {
        linear_to_srgb_neon(token, input)
    }

    #[archmage::arcane]
    fn call_srgb_to_linear_slice(token: NeonToken, values: &mut [f32]) {
        srgb_to_linear_slice_neon(token, values);
    }

    #[archmage::arcane]
    fn call_linear_to_srgb_slice(token: NeonToken, values: &mut [f32]) {
        linear_to_srgb_slice_neon(token, values);
    }

    #[test]
    fn test_x4_srgb_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: NEON not available");
            return;
        };

        let input = [0.0, 0.3, 0.7, 1.0];
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
    fn test_x4_matches_scalar() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: NEON not available");
            return;
        };

        let input = [0.0, 0.3, 0.7, 1.0];
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
    fn test_slice_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: NEON not available");
            return;
        };

        let mut values: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
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
