//! Private 16-lane kernels used by the canonical CPU-feature entry points.

use archmage::attune;

use archmage::X64V4Token;

use magetypes::simd::v4::f32x16 as mt_f32x16;

// sRGB transfer function constants (C0-continuous moxcms, matching rational polynomial)

// ============================================================================
// x16 functions — operate on [f32; 16]
// ============================================================================

/// Convert 16 sRGB values to linear. Input clamped to \[0, 1\].
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune(inline(hint))]
pub(crate) fn srgb_to_linear_v4(token: X64V4Token, srgb: [f32; 16]) -> [f32; 16] {
    super::float::x16::srgb_to_linear(token, srgb)
}

/// Convert 16 linear values to sRGB. Input clamped to \[0, 1\].
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_v4(token: X64V4Token, linear: [f32; 16]) -> [f32; 16] {
    super::float::x16::linear_to_srgb(token, linear)
}

/// Convert 16 gamma-encoded values to linear. Input clamped to \[0, 1\].
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune(inline(hint))]
pub(crate) fn gamma_to_linear_v4(token: X64V4Token, encoded: [f32; 16], gamma: f32) -> [f32; 16] {
    super::float::x16::gamma_to_linear(token, encoded, gamma)
}

/// Convert 16 linear values to gamma-encoded. Input clamped to \[0, 1\].
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune(inline(hint))]
pub(crate) fn linear_to_gamma_v4(token: X64V4Token, linear: [f32; 16], gamma: f32) -> [f32; 16] {
    super::float::x16::linear_to_gamma(token, linear, gamma)
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
#[attune(inline(hint))]
pub(crate) fn srgb_u8_to_linear_v4(_token: X64V4Token, srgb: [u8; 16]) -> [f32; 16] {
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
        lut[srgb[8] as usize],
        lut[srgb[9] as usize],
        lut[srgb[10] as usize],
        lut[srgb[11] as usize],
        lut[srgb[12] as usize],
        lut[srgb[13] as usize],
        lut[srgb[14] as usize],
        lut[srgb[15] as usize],
    ]
}

/// Convert sRGB u8 values to linear f32 using 16-wide LUT lookup.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune(inline(hint))]
pub(crate) fn srgb_u8_to_linear_slice_v4(_token: X64V4Token, input: &[u8], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let lut = crate::const_luts::linear_table_8();
    let (in_chunks, in_remainder) = input.as_chunks::<16>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<16>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = srgb_u8_to_linear_v4(_token, *inp);
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        *out = lut[*inp as usize];
    }
}

/// Convert 16 linear f32 values to sRGB u8 via LUT. Input clamped to \[0, 1\].
///
/// Uses a 4096-entry const LUT with bitmask indexing (`& 0xFFF`) for
/// provably safe bounds. SIMD accelerates the clamp and scale; lookups
/// are scalar from an L1-resident 4KB table.
///
/// The `X64V4Token` parameter proves AVX-512 support at compile time.
/// Call from inside an `#[arcane]` function for zero-overhead inlining.
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_u8_v4(token: X64V4Token, linear: [f32; 16]) -> [u8; 16] {
    const TRUNC: u32 = magetypes::numerics::IntConvert::new()
        .trunc()
        .in_range()
        .bits();
    let v = mt_f32x16::from_array_t(token, linear);
    let zero = mt_f32x16::zero_t(token);
    let one = mt_f32x16::splat_t(token, 1.0);
    let clamped = crate::tf::polynomial::floor_zero!(v, zero).min_fast(one);
    let scaled = clamped * mt_f32x16::splat_t(token, 4095.0) + mt_f32x16::splat_t(token, 0.5);
    let idx = (scaled.to_i32_with::<TRUNC>()
        & magetypes::simd::generic::i32x16::splat_t(token, 0xFFF))
    .to_array();
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
        lut[idx[8] as u32 as usize],
        lut[idx[9] as u32 as usize],
        lut[idx[10] as u32 as usize],
        lut[idx[11] as u32 as usize],
        lut[idx[12] as u32 as usize],
        lut[idx[13] as u32 as usize],
        lut[idx[14] as u32 as usize],
        lut[idx[15] as u32 as usize],
    ]
}

// ============================================================================
// Slice functions — process &mut [f32] with x16 chunking
// ============================================================================

/// Convert sRGB f32 values to linear in-place using 16-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune(inline(hint))]
pub(crate) fn srgb_to_linear_slice_v4(token: X64V4Token, values: &mut [f32]) {
    super::float::x16::srgb_to_linear_slice(token, values);
}

/// Convert linear f32 values to sRGB in-place using 16-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_slice_v4(token: X64V4Token, values: &mut [f32]) {
    super::float::x16::linear_to_srgb_slice(token, values);
}

/// Convert gamma-encoded f32 values to linear in-place using 16-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune(inline(hint))]
pub(crate) fn gamma_to_linear_slice_v4(token: X64V4Token, values: &mut [f32], gamma: f32) {
    super::float::x16::gamma_to_linear_slice(token, values, gamma);
}

/// Convert linear f32 values to gamma-encoded in-place using 16-wide SIMD.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune(inline(hint))]
pub(crate) fn linear_to_gamma_slice_v4(token: X64V4Token, values: &mut [f32], gamma: f32) {
    super::float::x16::linear_to_gamma_slice(token, values, gamma);
}

/// Convert linear f32 values to sRGB u8 using 16-wide SIMD + LUT.
///
/// Token parameter proves CPU support. Call from `#[arcane]` context.
#[attune(inline(hint))]
pub(crate) fn linear_to_srgb_u8_slice_v4(token: X64V4Token, input: &[f32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());
    let (in_chunks, in_remainder) = input.as_chunks::<16>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<16>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = linear_to_srgb_u8_v4(token, *inp);
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        *out = crate::scalar::linear_to_srgb_u8(*inp);
    }
}

// ============================================================================
// Transfer function rites (behind `transfer` feature)
// ============================================================================

/// Convert 16 sRGB values to linear (rational polynomial, no powf).
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn tf_srgb_to_linear_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    super::float::x16::srgb_to_linear(token, v)
}

/// Convert 16 linear values to sRGB (rational polynomial, no powf).
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn tf_linear_to_srgb_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    super::float::x16::linear_to_srgb(token, v)
}

/// Convert 16 PQ signal values to linear.
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn pq_to_linear_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    crate::tf::pq::pq_to_linear_x16(token, mt_f32x16::from_array_t(token, v)).to_array()
}

/// Convert 16 linear values to PQ signal.
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn linear_to_pq_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    crate::tf::pq::linear_to_pq_x16(token, mt_f32x16::from_array_t(token, v)).to_array()
}

/// Convert 16 BT.709 encoded values to linear (2×x8 via tokens::x8).
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn bt709_to_linear_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    crate::tf::bt709::bt709_to_linear_x16(token, mt_f32x16::from_array_t(token, v)).to_array()
}

/// Convert 16 linear values to BT.709 encoded (2×x8 via tokens::x8).
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn linear_to_bt709_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    crate::tf::bt709::linear_to_bt709_x16(token, mt_f32x16::from_array_t(token, v)).to_array()
}

/// Convert 16 HLG signal values to linear (2×x8 via tokens::x8).
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn hlg_to_linear_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    crate::tf::hlg::hlg_to_linear_x16(token, mt_f32x16::from_array_t(token, v)).to_array()
}

/// Convert 16 linear values to HLG signal (2×x8 via tokens::x8).
#[cfg(feature = "transfer")]
#[attune(inline(hint))]
pub(crate) fn linear_to_hlg_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
    crate::tf::hlg::linear_to_hlg_x16(token, mt_f32x16::from_array_t(token, v)).to_array()
}

#[cfg(feature = "transfer")]
macro_rules! tf_slice_v4 {
    ($name:ident, $rite:ident) => {
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        #[attune(inline(hint))]
        pub(crate) fn $name(token: X64V4Token, values: &mut [f32]) {
            crate::kernels::map_slice_inline!(values, 16, |chunk| $rite(token, chunk));
        }
    };
    ($name:ident => $slice_fn:path) => {
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        #[attune(inline(hint))]
        pub(crate) fn $name(token: X64V4Token, values: &mut [f32]) {
            $slice_fn(token, values);
        }
    };
}

#[cfg(feature = "transfer")]
tf_slice_v4!(tf_srgb_to_linear_slice_v4 => super::float::x16::srgb_to_linear_slice);
#[cfg(feature = "transfer")]
tf_slice_v4!(tf_linear_to_srgb_slice_v4 => super::float::x16::linear_to_srgb_slice);
#[cfg(feature = "transfer")]
tf_slice_v4!(bt709_to_linear_slice_v4 => crate::tf::bt709::bt709_to_linear_slice_x16);
#[cfg(feature = "transfer")]
tf_slice_v4!(linear_to_bt709_slice_v4 => crate::tf::bt709::linear_to_bt709_slice_x16);
#[cfg(feature = "transfer")]
tf_slice_v4!(pq_to_linear_slice_v4 => crate::tf::pq::pq_to_linear_slice_x16);
#[cfg(feature = "transfer")]
tf_slice_v4!(linear_to_pq_slice_v4 => crate::tf::pq::linear_to_pq_slice_x16);
#[cfg(feature = "transfer")]
tf_slice_v4!(hlg_to_linear_slice_v4 => crate::tf::hlg::hlg_to_linear_slice_x16);
#[cfg(feature = "transfer")]
tf_slice_v4!(linear_to_hlg_slice_v4 => crate::tf::hlg::linear_to_hlg_slice_x16);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use archmage::SimdToken;

    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    fn get_token() -> Option<X64V4Token> {
        X64V4Token::try_new()
    }

    #[archmage::arcane]
    fn call_srgb_to_linear(token: X64V4Token, input: [f32; 16]) -> [f32; 16] {
        srgb_to_linear_v4(token, input)
    }

    #[archmage::arcane]
    fn call_linear_to_srgb(token: X64V4Token, input: [f32; 16]) -> [f32; 16] {
        linear_to_srgb_v4(token, input)
    }

    #[archmage::arcane]
    fn call_srgb_to_linear_slice(token: X64V4Token, values: &mut [f32]) {
        srgb_to_linear_slice_v4(token, values);
    }

    #[archmage::arcane]
    fn call_linear_to_srgb_slice(token: X64V4Token, values: &mut [f32]) {
        linear_to_srgb_slice_v4(token, values);
    }

    #[archmage::arcane]
    fn call_linear_to_srgb_u8(token: X64V4Token, input: [f32; 16]) -> [u8; 16] {
        linear_to_srgb_u8_v4(token, input)
    }

    #[test]
    fn test_x16_linear_to_srgb_u8() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX-512 not available");
            return;
        };

        let input = [
            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
        ];
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
    fn test_x16_srgb_roundtrip() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX-512 not available");
            return;
        };

        let input = [
            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
        ];
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
    fn test_x16_matches_scalar() {
        let Some(token) = get_token() else {
            eprintln!("Skipping test: AVX-512 not available");
            return;
        };

        let input = [
            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
        ];
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
            eprintln!("Skipping test: AVX-512 not available");
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
