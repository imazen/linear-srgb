//! Compatibility adapters for the deprecated 16-lane token API.
//! Numerical implementation lives behind `crate::target_features::x16`.

use archmage::rite;

pub use archmage::X64V4Token;

// sRGB transfer function constants (C0-continuous moxcms, matching rational polynomial)

// ============================================================================
// x16 functions — operate on [f32; 16]
// ============================================================================

/// Convert 16 sRGB values to linear. Input clamped to \[0, 1\].
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn srgb_to_linear_v4(_token: X64V4Token, srgb: [f32; 16]) -> [f32; 16] {
    crate::target_features::x16::srgb_to_linear_v4(srgb)
}

/// Convert 16 linear values to sRGB. Input clamped to \[0, 1\].
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn linear_to_srgb_v4(_token: X64V4Token, linear: [f32; 16]) -> [f32; 16] {
    crate::target_features::x16::linear_to_srgb_v4(linear)
}

/// Convert 16 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn gamma_to_linear_v4(_token: X64V4Token, encoded: [f32; 16], gamma: f32) -> [f32; 16] {
    crate::target_features::x16::gamma_to_linear_v4(encoded, gamma)
}

/// Convert 16 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn linear_to_gamma_v4(_token: X64V4Token, linear: [f32; 16], gamma: f32) -> [f32; 16] {
    crate::target_features::x16::linear_to_gamma_v4(linear, gamma)
}

// ============================================================================
// x16 LUT functions — u8↔f32
// ============================================================================

/// Convert 16 sRGB u8 values to linear f32 via 256-entry LUT.
///
/// Pure table lookup — no math. The 1KB table fits in L1 cache.
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn srgb_u8_to_linear_v4(_token: X64V4Token, srgb: [u8; 16]) -> [f32; 16] {
    crate::target_features::x16::srgb_u8_to_linear_v4(srgb)
}

/// Convert sRGB u8 values to linear f32 using 16-wide LUT lookup.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn srgb_u8_to_linear_slice_v4(_token: X64V4Token, input: &[u8], output: &mut [f32]) {
    crate::target_features::x16::srgb_u8_to_linear_slice_v4(input, output)
}

/// Convert 16 linear f32 values to sRGB u8 via LUT. Input clamped to \[0, 1\].
///
/// Uses a 4096-entry const LUT with bitmask indexing (`& 0xFFF`) for
/// provably safe bounds. SIMD accelerates the clamp and scale; lookups
/// are scalar from an L1-resident 4KB table.
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[rite]
pub fn linear_to_srgb_u8_v4(_token: X64V4Token, linear: [f32; 16]) -> [u8; 16] {
    crate::target_features::x16::linear_to_srgb_u8_v4(linear)
}

// ============================================================================
// Slice functions — process &mut [f32] with x16 chunking
// ============================================================================

/// Convert sRGB f32 values to linear in-place using 16-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn srgb_to_linear_slice_v4(_token: X64V4Token, values: &mut [f32]) {
    crate::target_features::x16::srgb_to_linear_slice_v4(values)
}

/// Convert linear f32 values to sRGB in-place using 16-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn linear_to_srgb_slice_v4(_token: X64V4Token, values: &mut [f32]) {
    crate::target_features::x16::linear_to_srgb_slice_v4(values)
}

/// Convert gamma-encoded f32 values to linear in-place using 16-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn gamma_to_linear_slice_v4(_token: X64V4Token, values: &mut [f32], gamma: f32) {
    crate::target_features::x16::gamma_to_linear_slice_v4(values, gamma)
}

/// Convert linear f32 values to gamma-encoded in-place using 16-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn linear_to_gamma_slice_v4(_token: X64V4Token, values: &mut [f32], gamma: f32) {
    crate::target_features::x16::linear_to_gamma_slice_v4(values, gamma)
}

/// Convert linear f32 values to sRGB u8 using 16-wide SIMD + LUT.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[rite]
pub fn linear_to_srgb_u8_slice_v4(_token: X64V4Token, input: &[f32], output: &mut [u8]) {
    crate::target_features::x16::linear_to_srgb_u8_slice_v4(input, output)
}

// ============================================================================
// Transfer function rites (behind `transfer` feature)
// ============================================================================

/// Convert 16 sRGB values to linear (rational polynomial, no powf).
#[cfg(feature = "transfer")]
#[rite]
pub fn tf_srgb_to_linear_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    crate::target_features::x16::tf_srgb_to_linear_v4(v)
}

/// Convert 16 linear values to sRGB (rational polynomial, no powf).
#[cfg(feature = "transfer")]
#[rite]
pub fn tf_linear_to_srgb_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    crate::target_features::x16::tf_linear_to_srgb_v4(v)
}

/// Convert 16 PQ signal values to linear.
#[cfg(feature = "transfer")]
#[rite]
pub fn pq_to_linear_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    crate::target_features::x16::pq_to_linear_v4(v)
}

/// Convert 16 linear values to PQ signal.
#[cfg(feature = "transfer")]
#[rite]
pub fn linear_to_pq_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    crate::target_features::x16::linear_to_pq_v4(v)
}

/// Convert 16 BT.709 encoded values to linear (2×x8 via tokens::x8).
#[cfg(feature = "transfer")]
#[rite]
pub fn bt709_to_linear_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    crate::target_features::x16::bt709_to_linear_v4(v)
}

/// Convert 16 linear values to BT.709 encoded (2×x8 via tokens::x8).
#[cfg(feature = "transfer")]
#[rite]
pub fn linear_to_bt709_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    crate::target_features::x16::linear_to_bt709_v4(v)
}

/// Convert 16 HLG signal values to linear (2×x8 via tokens::x8).
#[cfg(feature = "transfer")]
#[rite]
pub fn hlg_to_linear_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    crate::target_features::x16::hlg_to_linear_v4(v)
}

/// Convert 16 linear values to HLG signal (2×x8 via tokens::x8).
#[cfg(feature = "transfer")]
#[rite]
pub fn linear_to_hlg_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    crate::target_features::x16::linear_to_hlg_v4(v)
}

#[cfg(feature = "transfer")]
macro_rules! tf_slice_v4 {
    ($name:ident, $rite:ident, $scalar:path) => {
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        #[rite]
        pub fn $name(_token: X64V4Token, values: &mut [f32]) {
            crate::target_features::x16::$name(values)
        }
    };
}

#[cfg(feature = "transfer")]
tf_slice_v4!(
    tf_srgb_to_linear_slice_v4,
    tf_srgb_to_linear_v4,
    crate::tf::srgb_to_linear
);
#[cfg(feature = "transfer")]
tf_slice_v4!(
    tf_linear_to_srgb_slice_v4,
    tf_linear_to_srgb_v4,
    crate::tf::linear_to_srgb
);
#[cfg(feature = "transfer")]
tf_slice_v4!(
    bt709_to_linear_slice_v4,
    bt709_to_linear_v4,
    crate::tf::bt709_to_linear
);
#[cfg(feature = "transfer")]
tf_slice_v4!(
    linear_to_bt709_slice_v4,
    linear_to_bt709_v4,
    crate::tf::linear_to_bt709
);
#[cfg(feature = "transfer")]
tf_slice_v4!(
    pq_to_linear_slice_v4,
    pq_to_linear_v4,
    crate::tf::pq_to_linear
);
#[cfg(feature = "transfer")]
tf_slice_v4!(
    linear_to_pq_slice_v4,
    linear_to_pq_v4,
    crate::tf::linear_to_pq
);
#[cfg(feature = "transfer")]
tf_slice_v4!(
    hlg_to_linear_slice_v4,
    hlg_to_linear_v4,
    crate::tf::hlg_to_linear
);
#[cfg(feature = "transfer")]
tf_slice_v4!(
    linear_to_hlg_slice_v4,
    linear_to_hlg_v4,
    crate::tf::linear_to_hlg
);
