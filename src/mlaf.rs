//! Single-rounding arithmetic and hardware square-root helpers for formula and LUT paths.

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

/// IEEE-754 correctly-rounded `f32` square root via a single hardware instruction
/// (`vsqrtss`/`sqrtss` on x86 SSE, `fsqrt` on AArch64 NEON, `f32.sqrt` / `f32x4.sqrt`
/// on WASM) in both `std` and `#![no_std]` builds, falling back to software IEEE
/// `sqrt_f32` only when hardware SIMD/FP is unavailable (`x86_64-unknown-none`, etc.).
#[inline(always)]
pub(crate) fn hw_sqrt_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        {
            use archmage::SimdToken;
            #[archmage::arcane]
            fn sqrt_v1(_token: archmage::X64V1Token, v: f32) -> f32 {
                use core::arch::x86_64::{_mm_cvtss_f32, _mm_set_ss, _mm_sqrt_ss};
                _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(v)))
            }
            if let Some(t) = archmage::X64V1Token::summon() {
                return sqrt_v1(t, x);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            use archmage::SimdToken;
            #[archmage::arcane]
            fn sqrt_neon(_token: archmage::NeonToken, v: f32) -> f32 {
                use core::arch::aarch64::{vdup_n_f32, vget_lane_f32, vsqrt_f32};
                vget_lane_f32::<0>(vsqrt_f32(vdup_n_f32(v)))
            }
            if let Some(t) = archmage::NeonToken::summon() {
                return sqrt_neon(t, x);
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            use archmage::SimdToken;
            #[archmage::arcane]
            fn sqrt_wasm(_token: archmage::Wasm128Token, v: f32) -> f32 {
                use core::arch::wasm32::{f32x4_extract_lane, f32x4_splat, f32x4_sqrt};
                f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(v)))
            }
            if let Some(t) = archmage::Wasm128Token::summon() {
                return sqrt_wasm(t, x);
            }
        }
        magetypes::scalar::sqrt_f32(x)
    }
}
