//! Private 4-lane kernels used by the canonical CPU-feature entry points.

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use super::float::x4::{
    gamma_to_linear as gamma_to_linear_core, linear_to_gamma as linear_to_gamma_core,
    linear_to_srgb as linear_to_srgb_core, srgb_to_linear as srgb_to_linear_core,
};
use archmage::attune;

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
use archmage::X64V3Token;

#[cfg(target_arch = "aarch64")]
use archmage::NeonToken;

#[cfg(target_arch = "wasm32")]
use archmage::Wasm128Token;

#[cfg(all(
    feature = "transfer",
    any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
use magetypes::simd::generic::f32x4 as gen_f32x4;

// sRGB transfer function constants (C0-continuous moxcms, matching rational polynomial)

// ============================================================================
// Generic core implementations
// ============================================================================
// x86-64 V3 (AVX2+FMA) — 4×f32 with X64V3Token
// ============================================================================

/// Convert 4 sRGB values to linear. Input clamped to \[0, 1\].
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Uses 128-bit SSE registers (VEX-encoded) within the AVX2+FMA context.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[attune(inline(hint))]
pub(crate) fn srgb_to_linear_v3(token: X64V3Token, srgb: [f32; 4]) -> [f32; 4] {
    srgb_to_linear_core(token, srgb)
}

/// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_v3(token: X64V3Token, linear: [f32; 4]) -> [f32; 4] {
    linear_to_srgb_core(token, linear)
}

/// Convert 4 sRGB values to linear without clamping (extended range).
///
/// 6/6 rational polynomial fitted to \[0, 8\]. sampled max 8 ULP in \[0,1\], u16-safe to ~4.2.
#[attune(_v3, _neon, _wasm128, _scalar, inline(hint))]
pub(crate) fn srgb_to_linear_extended(token: Token, srgb: [f32; 4]) -> [f32; 4] {
    super::float::x4::srgb_to_linear_extended(token, srgb)
}

/// Convert 4 linear values to sRGB without clamping (extended range).
///
/// 6/6 rational polynomial fitted on √x to \[0, 64\]. sampled max 7 ULP in \[0,1\], u16-safe to 64.
#[attune(_v3, _neon, _wasm128, _scalar, inline(hint))]
pub(crate) fn linear_to_srgb_extended(token: Token, linear: [f32; 4]) -> [f32; 4] {
    super::float::x4::linear_to_srgb_extended(token, linear)
}

/// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[attune(inline(hint))]
pub(crate) fn gamma_to_linear_v3(token: X64V3Token, encoded: [f32; 4], gamma: f32) -> [f32; 4] {
    gamma_to_linear_core(token, encoded, gamma)
}

/// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[attune(inline(hint))]
pub(crate) fn linear_to_gamma_v3(token: X64V3Token, linear: [f32; 4], gamma: f32) -> [f32; 4] {
    linear_to_gamma_core(token, linear, gamma)
}

/// Convert sRGB f32 values to linear in-place using 4-wide SSE SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[attune(inline(hint))]
pub(crate) fn srgb_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
    super::float::x4::srgb_to_linear_slice(token, values);
}

/// Convert linear f32 values to sRGB in-place using 4-wide SSE SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_slice_v3(token: X64V3Token, values: &mut [f32]) {
    super::float::x4::linear_to_srgb_slice(token, values);
}

/// Convert gamma-encoded f32 values to linear in-place using 4-wide SSE SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[attune(inline(hint))]
pub(crate) fn gamma_to_linear_slice_v3(token: X64V3Token, values: &mut [f32], gamma: f32) {
    super::float::x4::gamma_to_linear_slice(token, values, gamma);
}

/// Convert linear f32 values to gamma-encoded in-place using 4-wide SSE SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[attune(inline(hint))]
pub(crate) fn linear_to_gamma_slice_v3(token: X64V3Token, values: &mut [f32], gamma: f32) {
    super::float::x4::linear_to_gamma_slice(token, values, gamma);
}

// ============================================================================
// AArch64 NEON — 4×f32 with NeonToken token
// ============================================================================

/// Convert 4 sRGB values to linear. Input clamped to \[0, 1\].
///
/// The `NeonToken` parameter proves NEON support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[cfg(target_arch = "aarch64")]
#[attune(inline(hint))]
pub(crate) fn srgb_to_linear_neon(token: NeonToken, srgb: [f32; 4]) -> [f32; 4] {
    srgb_to_linear_core(token, srgb)
}

/// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_neon(token: NeonToken, linear: [f32; 4]) -> [f32; 4] {
    linear_to_srgb_core(token, linear)
}

/// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[attune(inline(hint))]
pub(crate) fn gamma_to_linear_neon(token: NeonToken, encoded: [f32; 4], gamma: f32) -> [f32; 4] {
    gamma_to_linear_core(token, encoded, gamma)
}

/// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[attune(inline(hint))]
pub(crate) fn linear_to_gamma_neon(token: NeonToken, linear: [f32; 4], gamma: f32) -> [f32; 4] {
    linear_to_gamma_core(token, linear, gamma)
}

/// Convert sRGB f32 values to linear in-place using 4-wide NEON SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[attune(inline(hint))]
pub(crate) fn srgb_to_linear_slice_neon(token: NeonToken, values: &mut [f32]) {
    super::float::x4::srgb_to_linear_slice(token, values);
}

/// Convert linear f32 values to sRGB in-place using 4-wide NEON SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_slice_neon(token: NeonToken, values: &mut [f32]) {
    super::float::x4::linear_to_srgb_slice(token, values);
}

/// Convert gamma-encoded f32 values to linear in-place using 4-wide NEON SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[attune(inline(hint))]
pub(crate) fn gamma_to_linear_slice_neon(token: NeonToken, values: &mut [f32], gamma: f32) {
    super::float::x4::gamma_to_linear_slice(token, values, gamma);
}

/// Convert linear f32 values to gamma-encoded in-place using 4-wide NEON SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[attune(inline(hint))]
pub(crate) fn linear_to_gamma_slice_neon(token: NeonToken, values: &mut [f32], gamma: f32) {
    super::float::x4::linear_to_gamma_slice(token, values, gamma);
}

// ============================================================================
// WebAssembly SIMD128 — 4×f32 with Wasm128Token
// ============================================================================

/// Convert 4 sRGB values to linear. Input clamped to \[0, 1\].
///
/// The `Wasm128Token` parameter proves SIMD128 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[cfg(target_arch = "wasm32")]
#[attune(inline(hint))]
pub(crate) fn srgb_to_linear_wasm128(token: Wasm128Token, srgb: [f32; 4]) -> [f32; 4] {
    srgb_to_linear_core(token, srgb)
}

/// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_wasm128(token: Wasm128Token, linear: [f32; 4]) -> [f32; 4] {
    linear_to_srgb_core(token, linear)
}

/// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[attune(inline(hint))]
pub(crate) fn gamma_to_linear_wasm128(
    token: Wasm128Token,
    encoded: [f32; 4],
    gamma: f32,
) -> [f32; 4] {
    gamma_to_linear_core(token, encoded, gamma)
}

/// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[attune(inline(hint))]
pub(crate) fn linear_to_gamma_wasm128(
    token: Wasm128Token,
    linear: [f32; 4],
    gamma: f32,
) -> [f32; 4] {
    linear_to_gamma_core(token, linear, gamma)
}

/// Convert sRGB f32 values to linear in-place using 4-wide WASM SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[attune(inline(hint))]
pub(crate) fn srgb_to_linear_slice_wasm128(token: Wasm128Token, values: &mut [f32]) {
    super::float::x4::srgb_to_linear_slice(token, values);
}

/// Convert linear f32 values to sRGB in-place using 4-wide WASM SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_slice_wasm128(token: Wasm128Token, values: &mut [f32]) {
    super::float::x4::linear_to_srgb_slice(token, values);
}

/// Convert gamma-encoded f32 values to linear in-place using 4-wide WASM SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[attune(inline(hint))]
pub(crate) fn gamma_to_linear_slice_wasm128(token: Wasm128Token, values: &mut [f32], gamma: f32) {
    super::float::x4::gamma_to_linear_slice(token, values, gamma);
}

/// Convert linear f32 values to gamma-encoded in-place using 4-wide WASM SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[attune(inline(hint))]
pub(crate) fn linear_to_gamma_slice_wasm128(token: Wasm128Token, values: &mut [f32], gamma: f32) {
    super::float::x4::linear_to_gamma_slice(token, values, gamma);
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
        #[cfg(all(
            feature = "transfer",
            all(target_arch = "x86_64", target_feature = "sse2")
        ))]
        #[attune(inline(hint))]
        pub(crate) fn $name(token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            $inner(token, gen_f32x4::from_array_t(token, v)).to_array()
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
    ($name:ident, $rite:ident) => {
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        #[cfg(all(
            feature = "transfer",
            all(target_arch = "x86_64", target_feature = "sse2")
        ))]
        #[attune(inline(hint))]
        pub(crate) fn $name(token: X64V3Token, values: &mut [f32]) {
            crate::kernels::map_slice_inline!(values, 4, |chunk| $rite(token, chunk));
        }
    };
    ($name:ident => $slice_fn:path) => {
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        #[cfg(all(
            feature = "transfer",
            all(target_arch = "x86_64", target_feature = "sse2")
        ))]
        #[attune(inline(hint))]
        pub(crate) fn $name(token: X64V3Token, values: &mut [f32]) {
            $slice_fn(token, values);
        }
    };
}

x86_tf_slice_rite!(tf_srgb_to_linear_slice_v3 => super::float::x4::srgb_to_linear_slice);
x86_tf_slice_rite!(tf_linear_to_srgb_slice_v3 => super::float::x4::linear_to_srgb_slice);
x86_tf_slice_rite!(bt709_to_linear_slice_v3 => crate::tf::bt709::bt709_to_linear_slice_x4);
x86_tf_slice_rite!(linear_to_bt709_slice_v3 => crate::tf::bt709::linear_to_bt709_slice_x4);
x86_tf_slice_rite!(pq_to_linear_slice_v3 => crate::tf::pq::pq_to_linear_slice_x4);
x86_tf_slice_rite!(linear_to_pq_slice_v3 => crate::tf::pq::linear_to_pq_slice_x4);
x86_tf_slice_rite!(hlg_to_linear_slice_v3 => crate::tf::hlg::hlg_to_linear_slice_x4);
x86_tf_slice_rite!(linear_to_hlg_slice_v3 => crate::tf::hlg::linear_to_hlg_slice_x4);

// ============================================================================
// Transfer function rites — AArch64 NEON (behind `transfer` feature)
// ============================================================================

macro_rules! neon_tf_rite {
    ($name:ident, $inner:path) => {
        /// Transfer function rite (4×f32, AArch64 NEON). Requires `transfer` feature.
        #[cfg(all(feature = "transfer", target_arch = "aarch64"))]
        #[attune(inline(hint))]
        pub(crate) fn $name(token: NeonToken, v: [f32; 4]) -> [f32; 4] {
            $inner(token, gen_f32x4::from_array_t(token, v)).to_array()
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
    ($name:ident, $rite:ident) => {
        /// Transfer function slice rite (4×f32, AArch64 NEON). Requires `transfer` feature.
        #[cfg(all(feature = "transfer", target_arch = "aarch64"))]
        #[attune(inline(hint))]
        pub(crate) fn $name(token: NeonToken, values: &mut [f32]) {
            crate::kernels::map_slice_inline!(values, 4, |chunk| $rite(token, chunk));
        }
    };
    ($name:ident => $slice_fn:path) => {
        /// Transfer function slice rite (4×f32, AArch64 NEON). Requires `transfer` feature.
        #[cfg(all(feature = "transfer", target_arch = "aarch64"))]
        #[attune(inline(hint))]
        pub(crate) fn $name(token: NeonToken, values: &mut [f32]) {
            $slice_fn(token, values);
        }
    };
}

neon_tf_slice_rite!(tf_srgb_to_linear_slice_neon => super::float::x4::srgb_to_linear_slice);
neon_tf_slice_rite!(tf_linear_to_srgb_slice_neon => super::float::x4::linear_to_srgb_slice);
neon_tf_slice_rite!(bt709_to_linear_slice_neon => crate::tf::bt709::bt709_to_linear_slice_x4);
neon_tf_slice_rite!(linear_to_bt709_slice_neon => crate::tf::bt709::linear_to_bt709_slice_x4);
neon_tf_slice_rite!(pq_to_linear_slice_neon => crate::tf::pq::pq_to_linear_slice_x4);
neon_tf_slice_rite!(linear_to_pq_slice_neon => crate::tf::pq::linear_to_pq_slice_x4);
neon_tf_slice_rite!(hlg_to_linear_slice_neon => crate::tf::hlg::hlg_to_linear_slice_x4);
neon_tf_slice_rite!(linear_to_hlg_slice_neon => crate::tf::hlg::linear_to_hlg_slice_x4);

// ============================================================================
// Transfer function rites — WebAssembly SIMD128 (behind `transfer` feature)
// ============================================================================

macro_rules! wasm_tf_rite {
    ($name:ident, $inner:path) => {
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        #[cfg(all(feature = "transfer", target_arch = "wasm32"))]
        #[attune(inline(hint))]
        pub(crate) fn $name(token: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            $inner(token, gen_f32x4::from_array_t(token, v)).to_array()
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
    ($name:ident, $rite:ident) => {
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        #[cfg(all(feature = "transfer", target_arch = "wasm32"))]
        #[attune(inline(hint))]
        pub(crate) fn $name(token: Wasm128Token, values: &mut [f32]) {
            crate::kernels::map_slice_inline!(values, 4, |chunk| $rite(token, chunk));
        }
    };
    ($name:ident => $slice_fn:path) => {
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        #[cfg(all(feature = "transfer", target_arch = "wasm32"))]
        #[attune(inline(hint))]
        pub(crate) fn $name(token: Wasm128Token, values: &mut [f32]) {
            $slice_fn(token, values);
        }
    };
}

wasm_tf_slice_rite!(tf_srgb_to_linear_slice_wasm128 => super::float::x4::srgb_to_linear_slice);
wasm_tf_slice_rite!(tf_linear_to_srgb_slice_wasm128 => super::float::x4::linear_to_srgb_slice);
wasm_tf_slice_rite!(bt709_to_linear_slice_wasm128 => crate::tf::bt709::bt709_to_linear_slice_x4);
wasm_tf_slice_rite!(linear_to_bt709_slice_wasm128 => crate::tf::bt709::linear_to_bt709_slice_x4);
wasm_tf_slice_rite!(pq_to_linear_slice_wasm128 => crate::tf::pq::pq_to_linear_slice_x4);
wasm_tf_slice_rite!(linear_to_pq_slice_wasm128 => crate::tf::pq::linear_to_pq_slice_x4);
wasm_tf_slice_rite!(hlg_to_linear_slice_wasm128 => crate::tf::hlg::hlg_to_linear_slice_x4);
wasm_tf_slice_rite!(linear_to_hlg_slice_wasm128 => crate::tf::hlg::linear_to_hlg_slice_x4);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod tests_x86 {
    use super::*;
    use archmage::SimdToken;

    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

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
            std::eprintln!("Skipping test: X64V3 not available");
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
            std::eprintln!("Skipping test: X64V3 not available");
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
            std::eprintln!("Skipping test: X64V3 not available");
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
            std::eprintln!("Skipping test: X64V3 not available");
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
    use alloc::vec::Vec;

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
            std::eprintln!("Skipping test: NEON not available");
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
            std::eprintln!("Skipping test: NEON not available");
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
            std::eprintln!("Skipping test: NEON not available");
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
