//! Custom-gamma (pure power-law) transfer function — generic SIMD wrappers.
//!
//! Pure `v.powf(gamma)` / `v.powf(1.0 / gamma)`, clamped to `[0, 1]`.
//! Matches the V3-only `tokens::x4::gamma_to_linear_core` / `tokens::x16::gamma_to_linear_v4`
//! pattern but is generic over any width's backend trait, so it works on
//! NEON / WASM128 / V3 / V4 / V4x with one body each.
//!
//! Used for Adobe RGB (gamma 2.19921875), gamma-2.2, gamma-1.8 ICC profiles, etc.

use magetypes::simd::backends::{F32x4Convert, F32x8Convert, F32x16Convert};
use magetypes::simd::generic::{f32x4, f32x8, f32x16};

/// Custom-gamma encoded → linear, 4 lanes. Clamps input to `[0, 1]`.
#[inline(always)]
pub fn gamma_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>, gamma: f32) -> f32x4<T> {
    let zero = f32x4::zero(t);
    let one = f32x4::splat(t, 1.0);
    let clamped = v.max(zero).min(one);
    clamped.pow_midp(gamma)
}

/// Linear → custom-gamma encoded, 4 lanes. Clamps input to `[0, 1]`.
#[inline(always)]
pub fn linear_to_gamma_x4<T: F32x4Convert>(t: T, v: f32x4<T>, gamma: f32) -> f32x4<T> {
    let zero = f32x4::zero(t);
    let one = f32x4::splat(t, 1.0);
    let clamped = v.max(zero).min(one);
    clamped.pow_midp(1.0 / gamma)
}

/// Custom-gamma encoded → linear, 8 lanes. Clamps input to `[0, 1]`.
#[inline(always)]
pub fn gamma_to_linear_x8<T: F32x8Convert>(t: T, v: f32x8<T>, gamma: f32) -> f32x8<T> {
    let zero = f32x8::zero(t);
    let one = f32x8::splat(t, 1.0);
    let clamped = v.max(zero).min(one);
    clamped.pow_midp(gamma)
}

/// Linear → custom-gamma encoded, 8 lanes. Clamps input to `[0, 1]`.
#[inline(always)]
pub fn linear_to_gamma_x8<T: F32x8Convert>(t: T, v: f32x8<T>, gamma: f32) -> f32x8<T> {
    let zero = f32x8::zero(t);
    let one = f32x8::splat(t, 1.0);
    let clamped = v.max(zero).min(one);
    clamped.pow_midp(1.0 / gamma)
}

/// Custom-gamma encoded → linear, 16 lanes. Clamps input to `[0, 1]`.
#[inline(always)]
pub fn gamma_to_linear_x16<T: F32x16Convert>(t: T, v: f32x16<T>, gamma: f32) -> f32x16<T> {
    let zero = f32x16::zero(t);
    let one = f32x16::splat(t, 1.0);
    let clamped = v.max(zero).min(one);
    clamped.pow_midp(gamma)
}

/// Linear → custom-gamma encoded, 16 lanes. Clamps input to `[0, 1]`.
#[inline(always)]
pub fn linear_to_gamma_x16<T: F32x16Convert>(t: T, v: f32x16<T>, gamma: f32) -> f32x16<T> {
    let zero = f32x16::zero(t);
    let one = f32x16::splat(t, 1.0);
    let clamped = v.max(zero).min(one);
    clamped.pow_midp(1.0 / gamma)
}
