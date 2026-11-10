//! Compatibility adapters for the deprecated 8-lane token API.
//! Numerical implementation lives behind `crate::target_features::x8`.

use archmage::attune;

// 8-wide is AVX2/V3-only; NEON and Wasm128 are 4-wide and live in `tokens::x4`.
pub use archmage::X64V3Token;

// sRGB transfer function constants (C0-continuous moxcms, matching rational polynomial)

// ============================================================================
// x8 functions — operate on [f32; 8]
// ============================================================================

/// Convert 8 sRGB values to linear. Input clamped to \[0, 1\].
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune]
pub fn srgb_to_linear_v3(_token: X64V3Token, srgb: [f32; 8]) -> [f32; 8] {
    crate::target_features::x8::srgb_to_linear_v3(srgb)
}

/// Convert 8 linear values to sRGB. Input clamped to \[0, 1\].
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune]
pub fn linear_to_srgb_v3(_token: X64V3Token, linear: [f32; 8]) -> [f32; 8] {
    crate::target_features::x8::linear_to_srgb_v3(linear)
}

/// Convert 8 sRGB values to linear without clamping (extended range).
///
/// Uses abs+sign with a 6/6 rational polynomial fitted to \[0, 8\].
/// Sampled max 8 ULP in \[0,1\], u16-safe to |encoded| ≤ 6.18, u8-safe to 8.0.
/// Pure SIMD — no per-lane branching or scalar fallback.
#[attune]
pub fn srgb_to_linear_extended_v3(_token: X64V3Token, srgb: [f32; 8]) -> [f32; 8] {
    crate::target_features::x8::srgb_to_linear_extended_v3(srgb)
}

/// Convert 8 linear values to sRGB without clamping (extended range).
///
/// Uses abs+sign with a 6/6 rational polynomial fitted on √x to \[0, 64\].
/// Sampled max 7 ULP in \[0,1\], u16-safe across the full \[0, 64\] domain.
/// Pure SIMD — no per-lane branching or scalar fallback.
#[attune]
pub fn linear_to_srgb_extended_v3(_token: X64V3Token, linear: [f32; 8]) -> [f32; 8] {
    crate::target_features::x8::linear_to_srgb_extended_v3(linear)
}

/// Convert 8 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune]
pub fn gamma_to_linear_v3(_token: X64V3Token, encoded: [f32; 8], gamma: f32) -> [f32; 8] {
    crate::target_features::x8::gamma_to_linear_v3(encoded, gamma)
}

/// Convert 8 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune]
pub fn linear_to_gamma_v3(_token: X64V3Token, linear: [f32; 8], gamma: f32) -> [f32; 8] {
    crate::target_features::x8::linear_to_gamma_v3(linear, gamma)
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
#[attune]
pub fn srgb_u8_to_linear_v3(_token: X64V3Token, srgb: [u8; 8]) -> [f32; 8] {
    crate::target_features::x8::srgb_u8_to_linear_v3(srgb)
}

/// Convert sRGB u8 values to linear f32 using 8-wide LUT lookup.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune]
pub fn srgb_u8_to_linear_slice_v3(_token: X64V3Token, input: &[u8], output: &mut [f32]) {
    crate::target_features::x8::srgb_u8_to_linear_slice_v3(input, output)
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
#[attune(_v3, _neon, _wasm128, _scalar, inline(hint))]
pub fn srgb_u16_to_linear(_token: Token, srgb: [u16; 8]) -> [f32; 8] {
    archmage::incant!(crate::target_features::x8::srgb_u16_to_linear(srgb) without token)
}

/// Convert 8 linear f32 values to sRGB u8 via LUT. Input clamped to \[0, 1\].
///
/// Uses a 4096-entry const LUT with bitmask indexing (`& 0xFFF`) for
/// provably safe bounds. SIMD accelerates the clamp and scale; lookups
/// are scalar from an L1-resident 4KB table.
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune]
pub fn linear_to_srgb_u8_v3(_token: X64V3Token, linear: [f32; 8]) -> [u8; 8] {
    crate::target_features::x8::linear_to_srgb_u8_v3(linear)
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
#[attune(_v3, _neon, _wasm128, _scalar, inline(hint))]
pub fn linear_to_srgb_u16(_token: Token, linear: [f32; 8]) -> [u16; 8] {
    archmage::incant!(crate::target_features::x8::linear_to_srgb_u16(linear) without token)
}

// ============================================================================
// Slice functions — process &mut [f32] with x8 chunking
// ============================================================================

/// Convert sRGB f32 values to linear in-place using 8-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune]
pub fn srgb_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
    crate::target_features::x8::srgb_to_linear_slice_v3(values)
}

/// Convert linear f32 values to sRGB in-place using 8-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune]
pub fn linear_to_srgb_slice_v3(_token: X64V3Token, values: &mut [f32]) {
    crate::target_features::x8::linear_to_srgb_slice_v3(values)
}

/// Convert sRGB f32 values to linear in-place (extended range, no clamping).
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune]
pub fn srgb_to_linear_extended_slice_v3(_token: X64V3Token, values: &mut [f32]) {
    crate::target_features::x8::srgb_to_linear_extended_slice_v3(values)
}

/// Convert linear f32 values to sRGB in-place (extended range, no clamping).
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune]
pub fn linear_to_srgb_extended_slice_v3(_token: X64V3Token, values: &mut [f32]) {
    crate::target_features::x8::linear_to_srgb_extended_slice_v3(values)
}

/// Convert gamma-encoded f32 values to linear in-place using 8-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune]
pub fn gamma_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32], gamma: f32) {
    crate::target_features::x8::gamma_to_linear_slice_v3(values, gamma)
}

/// Convert linear f32 values to gamma-encoded in-place using 8-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune]
pub fn linear_to_gamma_slice_v3(_token: X64V3Token, values: &mut [f32], gamma: f32) {
    crate::target_features::x8::linear_to_gamma_slice_v3(values, gamma)
}

/// Convert linear f32 values to sRGB u8 using 8-wide SIMD + LUT.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune]
pub fn linear_to_srgb_u8_slice_v3(_token: X64V3Token, input: &[f32], output: &mut [u8]) {
    crate::target_features::x8::linear_to_srgb_u8_slice_v3(input, output)
}

// ============================================================================
// Transfer function rites (behind `transfer` feature)
// ============================================================================

/// Convert 8 sRGB values to linear (rational polynomial, no powf).
#[cfg(feature = "transfer")]
#[attune]
pub fn tf_srgb_to_linear_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::target_features::x8::tf_srgb_to_linear_v3(v)
}

/// Convert 8 linear values to sRGB (rational polynomial, no powf).
#[cfg(feature = "transfer")]
#[attune]
pub fn tf_linear_to_srgb_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::target_features::x8::tf_linear_to_srgb_v3(v)
}

/// Convert 8 BT.709 encoded values to linear.
#[cfg(feature = "transfer")]
#[attune]
pub fn bt709_to_linear_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::target_features::x8::bt709_to_linear_v3(v)
}

/// Convert 8 linear values to BT.709 encoded.
#[cfg(feature = "transfer")]
#[attune]
pub fn linear_to_bt709_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::target_features::x8::linear_to_bt709_v3(v)
}

/// Convert 8 PQ signal values to linear.
#[cfg(feature = "transfer")]
#[attune]
pub fn pq_to_linear_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::target_features::x8::pq_to_linear_v3(v)
}

/// Convert 8 linear values to PQ signal.
#[cfg(feature = "transfer")]
#[attune]
pub fn linear_to_pq_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::target_features::x8::linear_to_pq_v3(v)
}

/// Convert 8 HLG signal values to linear.
#[cfg(feature = "transfer")]
#[attune]
pub fn hlg_to_linear_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::target_features::x8::hlg_to_linear_v3(v)
}

/// Convert 8 linear values to HLG signal.
#[cfg(feature = "transfer")]
#[attune]
pub fn linear_to_hlg_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::target_features::x8::linear_to_hlg_v3(v)
}

/// Convert sRGB f32 values to linear in-place, 8-wide (TF module version).
#[cfg(feature = "transfer")]
#[attune]
pub fn tf_srgb_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
    crate::target_features::x8::tf_srgb_to_linear_slice_v3(values)
}

/// Convert linear f32 values to sRGB in-place, 8-wide (TF module version).
#[cfg(feature = "transfer")]
#[attune]
pub fn tf_linear_to_srgb_slice_v3(_token: X64V3Token, values: &mut [f32]) {
    crate::target_features::x8::tf_linear_to_srgb_slice_v3(values)
}

/// Convert BT.709 f32 values to linear in-place, 8-wide.
#[cfg(feature = "transfer")]
#[attune]
pub fn bt709_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
    crate::target_features::x8::bt709_to_linear_slice_v3(values)
}

/// Convert linear f32 values to BT.709 in-place, 8-wide.
#[cfg(feature = "transfer")]
#[attune]
pub fn linear_to_bt709_slice_v3(_token: X64V3Token, values: &mut [f32]) {
    crate::target_features::x8::linear_to_bt709_slice_v3(values)
}

/// Convert PQ f32 values to linear in-place, 8-wide.
#[cfg(feature = "transfer")]
#[attune]
pub fn pq_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
    crate::target_features::x8::pq_to_linear_slice_v3(values)
}

/// Convert linear f32 values to PQ in-place, 8-wide.
#[cfg(feature = "transfer")]
#[attune]
pub fn linear_to_pq_slice_v3(_token: X64V3Token, values: &mut [f32]) {
    crate::target_features::x8::linear_to_pq_slice_v3(values)
}

/// Convert HLG f32 values to linear in-place, 8-wide.
#[cfg(feature = "transfer")]
#[attune]
pub fn hlg_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
    crate::target_features::x8::hlg_to_linear_slice_v3(values)
}

/// Convert linear f32 values to HLG in-place, 8-wide.
#[cfg(feature = "transfer")]
#[attune]
pub fn linear_to_hlg_slice_v3(_token: X64V3Token, values: &mut [f32]) {
    crate::target_features::x8::linear_to_hlg_slice_v3(values)
}
