//! Private 8-lane kernels used by the canonical CPU-feature entry points.

use crate::tf::polynomial::Mla;
use archmage::attune;

// 8-wide is AVX2/V3-only; NEON and Wasm128 are 4-wide and live in `tokens::x4`.
use archmage::X64V3Token;

use magetypes::simd::f32x8 as mt_f32x8;
use magetypes::simd::generic::f32x8 as gen_f32x8;

// ============================================================================
// x8 functions — operate on [f32; 8]
// ============================================================================

/// Convert 8 sRGB values to linear. Input clamped to \[0, 1\].
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune(inline(hint))]
pub(crate) fn srgb_to_linear_v3(token: X64V3Token, srgb: [f32; 8]) -> [f32; 8] {
    super::float::x8::srgb_to_linear(token, srgb)
}

/// Convert 8 linear values to sRGB. Input clamped to \[0, 1\].
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_v3(token: X64V3Token, linear: [f32; 8]) -> [f32; 8] {
    super::float::x8::linear_to_srgb(token, linear)
}

/// Convert 8 sRGB values to linear without clamping (extended range).
///
/// Uses abs+sign with a 6/6 rational polynomial fitted to \[0, 8\].
/// Sampled max 8 ULP in \[0,1\], u16-safe to |encoded| ≤ 6.18, u8-safe to 8.0.
/// Pure SIMD — no per-lane branching or scalar fallback.
#[attune(inline(hint))]
pub(crate) fn srgb_to_linear_extended_v3(token: X64V3Token, srgb: [f32; 8]) -> [f32; 8] {
    super::float::x8::srgb_to_linear_extended(token, srgb)
}

/// Convert 8 linear values to sRGB without clamping (extended range).
///
/// Uses abs+sign with a 6/6 rational polynomial fitted on √x to \[0, 64\].
/// Sampled max 7 ULP in \[0,1\], u16-safe across the full \[0, 64\] domain.
/// Pure SIMD — no per-lane branching or scalar fallback.
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_extended_v3(token: X64V3Token, linear: [f32; 8]) -> [f32; 8] {
    super::float::x8::linear_to_srgb_extended(token, linear)
}

/// Convert 8 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune(inline(hint))]
pub(crate) fn gamma_to_linear_v3(token: X64V3Token, encoded: [f32; 8], gamma: f32) -> [f32; 8] {
    super::float::x8::gamma_to_linear(token, encoded, gamma)
}

/// Convert 8 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune(inline(hint))]
pub(crate) fn linear_to_gamma_v3(token: X64V3Token, linear: [f32; 8], gamma: f32) -> [f32; 8] {
    super::float::x8::linear_to_gamma(token, linear, gamma)
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
#[attune(inline(hint))]
pub(crate) fn srgb_u8_to_linear_v3(_token: X64V3Token, srgb: [u8; 8]) -> [f32; 8] {
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
#[attune(inline(hint))]
pub(crate) fn srgb_u8_to_linear_slice_v3(_token: X64V3Token, input: &[u8], output: &mut [f32]) {
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
#[attune(_v3, _neon, _wasm128, _scalar, inline(hint))]
pub(crate) fn srgb_u16_to_linear(token: Token, srgb: [u16; 8]) -> [f32; 8] {
    #[allow(non_camel_case_types)]
    type f32x8 = gen_f32x8<Token>;

    // u16 → f32 / 65535: auto-vectorizes to vpmovzxwd + vcvtdq2ps + vmulps
    // (or the NEON / WASM equivalents). Since `v` is already in `[0, 1]`,
    // `SRGB_UNIT_NONNEG` ablates redundant pre-clamps and post-div `.min(1.0)`
    // while keeping exact `1.0` at `65535`.
    let mut f = [0.0f32; 8];
    for i in 0..8 {
        f[i] = srgb[i] as f32;
    }
    let v = f32x8::from_array_t(token, f) * f32x8::splat_t(token, 1.0 / 65535.0);
    super::float::x8::srgb_to_linear_vector_cfg::<_, { super::float::cfg::SRGB_UNIT_NONNEG }>(
        token, v,
    )
    .to_array()
}

/// Convert 8 linear f32 values to sRGB u8 via LUT. Input clamped to \[0, 1\].
///
/// Uses a 4096-entry const LUT with bitmask indexing (`& 0xFFF`) for
/// provably safe bounds. SIMD accelerates the clamp and scale; lookups
/// are scalar from an L1-resident 4KB table.
///
/// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_u8_v3(token: X64V3Token, linear: [f32; 8]) -> [u8; 8] {
    const TRUNC: u32 = magetypes::numerics::IntConvert::new()
        .trunc()
        .in_range()
        .bits();
    let v = mt_f32x8::from_array_t(token, linear);
    let zero = mt_f32x8::zero_t(token);
    let one = mt_f32x8::splat_t(token, 1.0);
    let clamped = crate::tf::polynomial::floor_zero!(v, zero).min_fast(one);
    let scaled = clamped * mt_f32x8::splat_t(token, 4095.0) + mt_f32x8::splat_t(token, 0.5);
    let idx =
        (scaled.to_i32_with::<TRUNC>() & magetypes::simd::i32x8::splat_t(token, 0xFFF)).to_array();
    let lut = crate::const_luts::linear_to_srgb_u8();
    [
        lut[idx[0] as u32 as usize],
        lut[idx[1] as u32 as usize],
        lut[idx[2] as u32 as usize],
        lut[idx[3] as u32 as usize],
        lut[idx[4] as u32 as usize],
        lut[idx[5] as u32 as usize],
        lut[idx[6] as u32 as usize],
        lut[idx[7] as u32 as usize],
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
#[attune(_v3, _neon, _wasm128, _scalar, inline(hint))]
pub(crate) fn linear_to_srgb_u16(token: Token, linear: [f32; 8]) -> [u16; 8] {
    const TRUNC: u32 = magetypes::numerics::IntConvert::new()
        .trunc()
        .in_range()
        .bits();
    #[allow(non_camel_case_types)]
    type f32x8 = gen_f32x8<Token>;

    let v = f32x8::from_array_t(token, linear);
    let srgb = super::float::x8::linear_to_srgb_vector(token, v);
    let scaled = srgb.mla(f32x8::splat_t(token, 65535.0), f32x8::splat_t(token, 0.5));
    let idx = scaled.to_i32_with::<TRUNC>().to_array();
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
#[attune(inline(hint))]
pub(crate) fn srgb_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
    super::float::x8::srgb_to_linear_slice(token, values);
}

/// Convert linear f32 values to sRGB in-place using 8-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_slice_v3(token: X64V3Token, values: &mut [f32]) {
    super::float::x8::linear_to_srgb_slice(token, values);
}

/// Convert sRGB f32 values to linear in-place (extended range, no clamping).
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune(inline(hint))]
pub(crate) fn srgb_to_linear_extended_slice_v3(token: X64V3Token, values: &mut [f32]) {
    super::float::x8::srgb_to_linear_extended_slice(token, values);
}

/// Convert linear f32 values to sRGB in-place (extended range, no clamping).
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_extended_slice_v3(token: X64V3Token, values: &mut [f32]) {
    super::float::x8::linear_to_srgb_extended_slice(token, values);
}

/// Convert gamma-encoded f32 values to linear in-place using 8-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune(inline(hint))]
pub(crate) fn gamma_to_linear_slice_v3(token: X64V3Token, values: &mut [f32], gamma: f32) {
    super::float::x8::gamma_to_linear_slice(token, values, gamma);
}

/// Convert linear f32 values to gamma-encoded in-place using 8-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune(inline(hint))]
pub(crate) fn linear_to_gamma_slice_v3(token: X64V3Token, values: &mut [f32], gamma: f32) {
    super::float::x8::linear_to_gamma_slice(token, values, gamma);
}

/// Convert linear f32 values to sRGB u8 using 8-wide SIMD + LUT.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_u8_slice_v3(token: X64V3Token, input: &[f32], output: &mut [u8]) {
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
#[attune(inline(hint))]
pub(crate) fn tf_srgb_to_linear_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::srgb::srgb_to_linear_x8(token, mt_f32x8::from_array_t(token, v)).to_array()
}

/// Convert 8 linear values to sRGB (rational polynomial, no powf).
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn tf_linear_to_srgb_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::srgb::linear_to_srgb_x8(token, mt_f32x8::from_array_t(token, v)).to_array()
}

/// Convert 8 BT.709 encoded values to linear.
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn bt709_to_linear_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::bt709::bt709_to_linear_x8(token, mt_f32x8::from_array_t(token, v)).to_array()
}

/// Convert 8 linear values to BT.709 encoded.
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn linear_to_bt709_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::bt709::linear_to_bt709_x8(token, mt_f32x8::from_array_t(token, v)).to_array()
}

/// Convert 8 PQ signal values to linear.
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn pq_to_linear_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::pq::pq_to_linear_x8(token, mt_f32x8::from_array_t(token, v)).to_array()
}

/// Convert 8 linear values to PQ signal.
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn linear_to_pq_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::pq::linear_to_pq_x8(token, mt_f32x8::from_array_t(token, v)).to_array()
}

/// Convert 8 HLG signal values to linear.
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn hlg_to_linear_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::hlg::hlg_to_linear_x8(token, mt_f32x8::from_array_t(token, v)).to_array()
}

/// Convert 8 linear values to HLG signal.
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn linear_to_hlg_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
    crate::tf::hlg::linear_to_hlg_x8(token, mt_f32x8::from_array_t(token, v)).to_array()
}

/// Convert sRGB f32 values to linear in-place, 8-wide (TF module version).
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn tf_srgb_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
    super::float::x8::srgb_to_linear_slice(token, values);
}

/// Convert linear f32 values to sRGB in-place, 8-wide (TF module version).
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn tf_linear_to_srgb_slice_v3(token: X64V3Token, values: &mut [f32]) {
    super::float::x8::linear_to_srgb_slice(token, values);
}

/// Convert BT.709 f32 values to linear in-place, 8-wide.
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn bt709_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
    crate::tf::bt709::bt709_to_linear_slice_x8(token, values);
}

/// Convert linear f32 values to BT.709 in-place, 8-wide.
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn linear_to_bt709_slice_v3(token: X64V3Token, values: &mut [f32]) {
    crate::tf::bt709::linear_to_bt709_slice_x8(token, values);
}

/// Convert PQ f32 values to linear in-place, 8-wide.
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn pq_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
    crate::tf::pq::pq_to_linear_slice_x8(token, values);
}

/// Convert linear f32 values to PQ in-place, 8-wide.
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn linear_to_pq_slice_v3(token: X64V3Token, values: &mut [f32]) {
    crate::tf::pq::linear_to_pq_slice_x8(token, values);
}

/// Convert HLG f32 values to linear in-place, 8-wide.
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn hlg_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
    crate::tf::hlg::hlg_to_linear_slice_x8(token, values);
}

/// Convert linear f32 values to HLG in-place, 8-wide.
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn linear_to_hlg_slice_v3(token: X64V3Token, values: &mut [f32]) {
    crate::tf::hlg::linear_to_hlg_slice_x8(token, values);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use archmage::SimdToken;

    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    fn get_token() -> Option<X64V3Token> {
        X64V3Token::try_new()
    }

    // We need an #[arcane] wrapper to safely call #[attune] functions in tests.
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
            std::eprintln!("Skipping test: AVX2+FMA not available");
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
            std::eprintln!("Skipping test: AVX2+FMA not available");
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
            std::eprintln!("Skipping test: AVX2+FMA not available");
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
            std::eprintln!("Skipping test: AVX2+FMA not available");
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
            std::eprintln!("Skipping test: AVX2+FMA not available");
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
