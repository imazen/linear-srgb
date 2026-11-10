//! Compatibility adapters for the deprecated 4-lane token API.
//! Numerical implementation lives behind `crate::target_features::x4`.

use archmage::attune;

#[cfg(target_arch = "x86_64")]
pub use archmage::X64V3Token;

#[cfg(target_arch = "aarch64")]
pub use archmage::NeonToken;

#[cfg(target_arch = "wasm32")]
pub use archmage::Wasm128Token;

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
#[cfg(target_arch = "x86_64")]
#[attune]
pub fn srgb_to_linear_v3(_token: X64V3Token, srgb: [f32; 4]) -> [f32; 4] {
    crate::target_features::x4::srgb_to_linear_v3(srgb)
}

/// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "x86_64")]
#[attune]
pub fn linear_to_srgb_v3(_token: X64V3Token, linear: [f32; 4]) -> [f32; 4] {
    crate::target_features::x4::linear_to_srgb_v3(linear)
}

/// Convert 4 sRGB values to linear without clamping (extended range).
///
/// 6/6 rational polynomial fitted to \[0, 8\]. sampled max 8 ULP in \[0,1\], u16-safe to ~4.2.
#[attune(_v3, _neon, _wasm128, _scalar, inline(hint))]
pub fn srgb_to_linear_extended(_token: Token, srgb: [f32; 4]) -> [f32; 4] {
    archmage::incant!(crate::target_features::x4::srgb_to_linear_extended(srgb) without token)
}

/// Convert 4 linear values to sRGB without clamping (extended range).
///
/// 6/6 rational polynomial fitted on √x to \[0, 64\]. sampled max 7 ULP in \[0,1\], u16-safe to 64.
#[attune(_v3, _neon, _wasm128, _scalar, inline(hint))]
pub fn linear_to_srgb_extended(_token: Token, linear: [f32; 4]) -> [f32; 4] {
    archmage::incant!(crate::target_features::x4::linear_to_srgb_extended(linear) without token)
}

/// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "x86_64")]
#[attune]
pub fn gamma_to_linear_v3(_token: X64V3Token, encoded: [f32; 4], gamma: f32) -> [f32; 4] {
    crate::target_features::x4::gamma_to_linear_v3(encoded, gamma)
}

/// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "x86_64")]
#[attune]
pub fn linear_to_gamma_v3(_token: X64V3Token, linear: [f32; 4], gamma: f32) -> [f32; 4] {
    crate::target_features::x4::linear_to_gamma_v3(linear, gamma)
}

/// Convert sRGB f32 values to linear in-place using 4-wide SSE SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "x86_64")]
#[attune]
pub fn srgb_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
    crate::target_features::x4::srgb_to_linear_slice_v3(values)
}

/// Convert linear f32 values to sRGB in-place using 4-wide SSE SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "x86_64")]
#[attune]
pub fn linear_to_srgb_slice_v3(_token: X64V3Token, values: &mut [f32]) {
    crate::target_features::x4::linear_to_srgb_slice_v3(values)
}

/// Convert gamma-encoded f32 values to linear in-place using 4-wide SSE SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "x86_64")]
#[attune]
pub fn gamma_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32], gamma: f32) {
    crate::target_features::x4::gamma_to_linear_slice_v3(values, gamma)
}

/// Convert linear f32 values to gamma-encoded in-place using 4-wide SSE SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "x86_64")]
#[attune]
pub fn linear_to_gamma_slice_v3(_token: X64V3Token, values: &mut [f32], gamma: f32) {
    crate::target_features::x4::linear_to_gamma_slice_v3(values, gamma)
}

// ============================================================================
// AArch64 NEON — 4×f32 with NeonToken token
// ============================================================================

/// Convert 4 sRGB values to linear. Input clamped to \[0, 1\].
///
/// The `NeonToken` parameter proves NEON support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[cfg(target_arch = "aarch64")]
#[attune]
pub fn srgb_to_linear_neon(_token: NeonToken, srgb: [f32; 4]) -> [f32; 4] {
    crate::target_features::x4::srgb_to_linear_neon(srgb)
}

/// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[attune]
pub fn linear_to_srgb_neon(_token: NeonToken, linear: [f32; 4]) -> [f32; 4] {
    crate::target_features::x4::linear_to_srgb_neon(linear)
}

/// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[attune]
pub fn gamma_to_linear_neon(_token: NeonToken, encoded: [f32; 4], gamma: f32) -> [f32; 4] {
    crate::target_features::x4::gamma_to_linear_neon(encoded, gamma)
}

/// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[attune]
pub fn linear_to_gamma_neon(_token: NeonToken, linear: [f32; 4], gamma: f32) -> [f32; 4] {
    crate::target_features::x4::linear_to_gamma_neon(linear, gamma)
}

/// Convert sRGB f32 values to linear in-place using 4-wide NEON SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[attune]
pub fn srgb_to_linear_slice_neon(_token: NeonToken, values: &mut [f32]) {
    crate::target_features::x4::srgb_to_linear_slice_neon(values)
}

/// Convert linear f32 values to sRGB in-place using 4-wide NEON SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[attune]
pub fn linear_to_srgb_slice_neon(_token: NeonToken, values: &mut [f32]) {
    crate::target_features::x4::linear_to_srgb_slice_neon(values)
}

/// Convert gamma-encoded f32 values to linear in-place using 4-wide NEON SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[attune]
pub fn gamma_to_linear_slice_neon(_token: NeonToken, values: &mut [f32], gamma: f32) {
    crate::target_features::x4::gamma_to_linear_slice_neon(values, gamma)
}

/// Convert linear f32 values to gamma-encoded in-place using 4-wide NEON SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "aarch64")]
#[attune]
pub fn linear_to_gamma_slice_neon(_token: NeonToken, values: &mut [f32], gamma: f32) {
    crate::target_features::x4::linear_to_gamma_slice_neon(values, gamma)
}

// ============================================================================
// WebAssembly SIMD128 — 4×f32 with Wasm128Token
// ============================================================================

/// Convert 4 sRGB values to linear. Input clamped to \[0, 1\].
///
/// The `Wasm128Token` parameter proves SIMD128 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[cfg(target_arch = "wasm32")]
#[attune]
pub fn srgb_to_linear_wasm128(_token: Wasm128Token, srgb: [f32; 4]) -> [f32; 4] {
    crate::target_features::x4::srgb_to_linear_wasm128(srgb)
}

/// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[attune]
pub fn linear_to_srgb_wasm128(_token: Wasm128Token, linear: [f32; 4]) -> [f32; 4] {
    crate::target_features::x4::linear_to_srgb_wasm128(linear)
}

/// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[attune]
pub fn gamma_to_linear_wasm128(_token: Wasm128Token, encoded: [f32; 4], gamma: f32) -> [f32; 4] {
    crate::target_features::x4::gamma_to_linear_wasm128(encoded, gamma)
}

/// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[attune]
pub fn linear_to_gamma_wasm128(_token: Wasm128Token, linear: [f32; 4], gamma: f32) -> [f32; 4] {
    crate::target_features::x4::linear_to_gamma_wasm128(linear, gamma)
}

/// Convert sRGB f32 values to linear in-place using 4-wide WASM SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[attune]
pub fn srgb_to_linear_slice_wasm128(_token: Wasm128Token, values: &mut [f32]) {
    crate::target_features::x4::srgb_to_linear_slice_wasm128(values)
}

/// Convert linear f32 values to sRGB in-place using 4-wide WASM SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[attune]
pub fn linear_to_srgb_slice_wasm128(_token: Wasm128Token, values: &mut [f32]) {
    crate::target_features::x4::linear_to_srgb_slice_wasm128(values)
}

/// Convert gamma-encoded f32 values to linear in-place using 4-wide WASM SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[attune]
pub fn gamma_to_linear_slice_wasm128(_token: Wasm128Token, values: &mut [f32], gamma: f32) {
    crate::target_features::x4::gamma_to_linear_slice_wasm128(values, gamma)
}

/// Convert linear f32 values to gamma-encoded in-place using 4-wide WASM SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[cfg(target_arch = "wasm32")]
#[attune]
pub fn linear_to_gamma_slice_wasm128(_token: Wasm128Token, values: &mut [f32], gamma: f32) {
    crate::target_features::x4::linear_to_gamma_slice_wasm128(values, gamma)
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
        #[attune]
        pub fn $name(_token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::$name(v)
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
        #[attune]
        pub fn $name(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x4::$name(values)
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
        #[attune]
        pub fn $name(_token: NeonToken, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::$name(v)
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
        #[attune]
        pub fn $name(_token: NeonToken, values: &mut [f32]) {
            crate::target_features::x4::$name(values)
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
        #[attune]
        pub fn $name(_token: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::$name(v)
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
        #[attune]
        pub fn $name(_token: Wasm128Token, values: &mut [f32]) {
            crate::target_features::x4::$name(values)
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
