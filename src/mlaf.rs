//! Single-rounding arithmetic for the formula and LUT paths.

use magetypes::scalar::F32ScalarBackend;
pub(crate) use magetypes::scalar::{mul_add_fused_f32 as fmla_f32, mul_add_fused_f64 as fmla_f64};

#[inline(always)]
pub(crate) fn fmla_f32_with_token<T: F32ScalarBackend>(token: T, a: f32, b: f32, c: f32) -> f32 {
    // Native SIMD proofs guarantee hardware fusion. Scalar and WASM need the
    // portable implementation when the caller requires single rounding.
    if cfg!(target_arch = "wasm32") || token.as_scalar().is_some() {
        fmla_f32(a, b, c)
    } else {
        magetypes::scalar::mul_add_relaxed_f32(token, a, b, c)
    }
}
