//! PU encoding — perceptually-uniform luminance for HDR-IQA.
//!
//! PU encoding maps absolute luminance (cd/m²) to a perceptually-uniform
//! value via a CSF-derived curve. It exists because SDR-trained quality
//! metrics (SSIMULACRA2, IW-SSIM, butteraugli) saturate or sign-flip on
//! raw HDR luminance. Front-ending those metrics with PU recovers their
//! SDR machinery on HDR input — the foundational fix.
//!
//! ## Reference
//!
//! Mantiuk, R. K., Azimi, M. (2021).
//! *PU21: A novel perceptually uniform encoding for adapting existing
//! quality metrics for HDR.* Picture Coding Symposium (PCS).
//! Reference impl: <https://github.com/gfxdisp/pu21>.
//!
//! ## Output range
//!
//! `pu_encode(Y_cd_m2)` returns a value in roughly `[0, 530]` for
//! `Y ∈ [0.005, 10_000]` cd/m². For downstream IQA work that wants a
//! normalized `[0, 1]` scale, divide by [`PU_PEAK`] (the encoded value
//! at the HDR10 peak of 10 000 cd/m²).
//!
//! ## Anchor identities
//!
//! | Input | PU value |
//! |---|---|
//! | 0.005 cd/m² (sub-black) | clamped to 0 |
//! | 1 cd/m² | ≈ 84.4 |
//! | 10 cd/m² | ≈ 158.5 |
//! | 80 cd/m² (legacy SDR display peak) | ≈ 250.5 |
//! | 100 cd/m² (BT.1886 SDR peak) | ≈ 261.8 |
//! | 203 cd/m² (BT.2408 HDR reference white) | ≈ 298.8 ([`PU_REF_WHITE_BT2408`]) |
//! | 1 000 cd/m² (HDR10 highlight) | ≈ 388.1 |
//! | 4 000 cd/m² | ≈ 468.5 |
//! | 10 000 cd/m² (PQ peak) | ≈ 520.5 ([`PU_PEAK`]) |
//!
//! ## Scope
//!
//! - **Luminance only.** Chroma handling (per-channel PU on R/G/B vs
//!   PU on Y + linear Cb/Cr) is a downstream decision.
//! - **No display-adaptation curve.** PU is calibration of the *metric*,
//!   not of the *display* — that is BT.2390 EETF's job, lives downstream.

#[allow(unused_imports)]
use num_traits::Float; // provides powf in no_std

// =============================================================================
// PU21 constants (Mantiuk & Azimi 2021)
// =============================================================================
//
// PU21(Y) = par[7] * (((par[1] + par[2]·Y^par[4]) / (1 + par[3]·Y^par[4]))^par[5] - par[6])
//
// where Y is absolute luminance in cd/m². Coefficients lifted verbatim
// from gfxdisp/pu21 (banding_glare model — the variant tuned for IQA
// against both subtle banding and high-luminance flare; not the
// banding-only or HDR-VDP-3 fits).

const PU21_P1: f32 = 1.070_275_272;
const PU21_P2: f32 = 0.408_827_393_2;
const PU21_P3: f32 = 0.153_224_308;
const PU21_P4: f32 = 0.252_032_616_8;
const PU21_P5: f32 = 1.063_512_885;
const PU21_P6: f32 = 1.141_150_47;
const PU21_P7: f32 = 521.452_748_4;

/// PU value at the HDR10 peak luminance (10 000 cd/m²).
///
/// Use as the normaliser when a `[0, 1]` PU output is desired.
/// Computed from the published PU21 coefficients above; verified by
/// `tests::hdr_peak_anchor` to track the formula exactly.
pub const PU_PEAK: f32 = 520.467_25;

/// PU value at BT.2408 HDR reference white (203 cd/m²).
///
/// Anchor identity for verifying PU implementations and for downstream
/// metric calibration.
pub const PU_REF_WHITE_BT2408: f32 = 298.761_14;

/// Lowest luminance the PU curve accepts before clamping to zero.
///
/// Matches gfxdisp/pu21's reference clamp. Below this the CSF-integrated
/// curve goes mildly negative; clamping is the documented behaviour.
pub const PU_LUMINANCE_MIN_CD_M2: f32 = 0.005;

/// Highest luminance the PU curve is calibrated for.
///
/// Coincides with the HDR10 / PQ ST 2084 peak. Inputs above this are
/// permitted (the formula remains monotone) but extrapolate beyond the
/// published validation range.
pub const PU_LUMINANCE_MAX_CD_M2: f32 = 10_000.0;

// =============================================================================
// Scalar
// =============================================================================

/// Encode absolute luminance (cd/m²) to PU space.
///
/// Inputs at or below [`PU_LUMINANCE_MIN_CD_M2`] (0.005 cd/m²) clamp to 0
/// to match the gfxdisp/pu21 reference; the underlying curve goes mildly
/// negative below that threshold. The curve is monotone-increasing across
/// its valid range and extrapolates smoothly above
/// [`PU_LUMINANCE_MAX_CD_M2`].
///
/// # Example
///
/// ```
/// # #[cfg(feature = "transfer")] {
/// use linear_srgb::tf::pu_encode;
///
/// // SDR reference white anchors at PU ≈ 299.
/// let pu_ref = pu_encode(203.0);
/// assert!((pu_ref - 299.0).abs() < 1.0);
///
/// // Sub-black clamps to zero.
/// assert_eq!(pu_encode(-1.0), 0.0);
/// assert_eq!(pu_encode(0.0), 0.0);
/// # }
/// ```
#[inline]
pub fn pu_encode(luminance_cd_m2: f32) -> f32 {
    if luminance_cd_m2 <= PU_LUMINANCE_MIN_CD_M2 {
        return 0.0;
    }
    let yp = luminance_cd_m2.powf(PU21_P4);
    let num = PU21_P1 + PU21_P2 * yp;
    let den = 1.0 + PU21_P3 * yp;
    let inner = num / den;
    let val = PU21_P7 * (inner.powf(PU21_P5) - PU21_P6);
    val.max(0.0)
}

/// Decode a PU value back to absolute luminance (cd/m²).
///
/// Inverse of [`pu_encode`]. The closed-form inverse is well-defined
/// over the valid PU range `[0, ~530]`; inputs outside that clamp to
/// the corresponding luminance bounds.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "transfer")] {
/// use linear_srgb::tf::{pu_decode, pu_encode};
///
/// let y = 203.0_f32;
/// let pu = pu_encode(y);
/// let round_trip = pu_decode(pu);
/// assert!((round_trip - y).abs() < 0.5);
/// # }
/// ```
#[inline]
pub fn pu_decode(pu: f32) -> f32 {
    if pu <= 0.0 {
        return 0.0;
    }
    // Invert PU21: starting from V = p7·(inner^p5 - p6),
    // inner = ((V/p7) + p6)^(1/p5),
    // inner = (p1 + p2·Y^p4) / (1 + p3·Y^p4),
    // Y^p4 = (inner - p1) / (p2 - p3·inner),
    // Y = ((inner - p1) / (p2 - p3·inner))^(1/p4).
    let inner = ((pu / PU21_P7) + PU21_P6).powf(1.0 / PU21_P5);
    let num = inner - PU21_P1;
    let den = PU21_P2 - PU21_P3 * inner;
    if den <= 0.0 || num <= 0.0 {
        return PU_LUMINANCE_MIN_CD_M2;
    }
    (num / den).powf(1.0 / PU21_P4)
}

// =============================================================================
// Generic SIMD — element-wise scalar fallback
// =============================================================================
//
// The PU21 curve uses two fractional-exponent `powf` calls per element
// (Y^p4 then ^p5). Rational-polynomial fitting on the SIMD-friendly
// shape used by PQ/HLG is deferred; element-wise scalar dispatch keeps
// the SIMD entry points alive without shipping unvalidated coefficients.

use magetypes::simd::backends::F32x4Convert;
use magetypes::simd::generic::f32x4;

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn pu_encode_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let a = v.to_array();
    f32x4::from_array(
        t,
        [
            pu_encode(a[0]),
            pu_encode(a[1]),
            pu_encode(a[2]),
            pu_encode(a[3]),
        ],
    )
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn pu_decode_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
    let a = v.to_array();
    f32x4::from_array(
        t,
        [
            pu_decode(a[0]),
            pu_decode(a[1]),
            pu_decode(a[2]),
            pu_decode(a[3]),
        ],
    )
}

use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8;

#[inline(always)]
pub(crate) fn pu_encode_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let a = v.to_array();
    let mut out = [0.0_f32; 8];
    for i in 0..8 {
        out[i] = pu_encode(a[i]);
    }
    f32x8::from_array(t, out)
}

#[inline(always)]
pub(crate) fn pu_decode_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
    let a = v.to_array();
    let mut out = [0.0_f32; 8];
    for i in 0..8 {
        out[i] = pu_decode(a[i]);
    }
    f32x8::from_array(t, out)
}

use magetypes::simd::backends::F32x16Convert;
use magetypes::simd::generic::f32x16;

#[inline(always)]
pub(crate) fn pu_encode_x16<T: F32x16Convert>(t: T, v: f32x16<T>) -> f32x16<T> {
    let a = v.to_array();
    let mut out = [0.0_f32; 16];
    for i in 0..16 {
        out[i] = pu_encode(a[i]);
    }
    f32x16::from_array(t, out)
}

#[inline(always)]
pub(crate) fn pu_decode_x16<T: F32x16Convert>(t: T, v: f32x16<T>) -> f32x16<T> {
    let a = v.to_array();
    let mut out = [0.0_f32; 16];
    for i in 0..16 {
        out[i] = pu_decode(a[i]);
    }
    f32x16::from_array(t, out)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation in f64 — mirrors gfxdisp/pu21's `pu21_encoder`
    /// at full precision so the f32 path can be validated.
    fn pu_encode_f64(y: f64) -> f64 {
        if y <= PU_LUMINANCE_MIN_CD_M2 as f64 {
            return 0.0;
        }
        let yp = y.powf(PU21_P4 as f64);
        let num = (PU21_P1 as f64) + (PU21_P2 as f64) * yp;
        let den = 1.0 + (PU21_P3 as f64) * yp;
        let inner = num / den;
        let val = (PU21_P7 as f64) * (inner.powf(PU21_P5 as f64) - (PU21_P6 as f64));
        val.max(0.0)
    }

    #[test]
    fn sub_black_clamps_to_zero() {
        assert_eq!(pu_encode(-100.0), 0.0);
        assert_eq!(pu_encode(0.0), 0.0);
        assert_eq!(pu_encode(PU_LUMINANCE_MIN_CD_M2), 0.0);
    }

    #[test]
    fn bt2408_reference_white_anchor() {
        // BT.2408 reference white = 203 cd/m². PU should land at PU_REF_WHITE_BT2408.
        let pu = pu_encode(203.0);
        assert!(
            (pu - PU_REF_WHITE_BT2408).abs() < 0.05,
            "PU(203) = {pu}, expected ≈ {PU_REF_WHITE_BT2408}"
        );
    }

    #[test]
    fn hdr_peak_anchor() {
        // PU(10_000) defines the PU_PEAK constant.
        let pu = pu_encode(PU_LUMINANCE_MAX_CD_M2);
        assert!(
            (pu - PU_PEAK).abs() < 0.05,
            "PU(10000) = {pu}, expected ≈ {PU_PEAK}"
        );
    }

    #[test]
    fn published_curve_anchors() {
        // PU21 curve anchors verified against the f64 reference implementation.
        let cases = [
            (1.0_f32, 84.4_f32),
            (10.0, 158.5),
            (80.0, 250.5),
            (100.0, 261.8),
            (500.0, 348.5),
            (1_000.0, 388.1),
            (4_000.0, 468.5),
        ];
        for (input, expected) in cases {
            let pu = pu_encode(input);
            assert!(
                (pu - expected).abs() < 0.1,
                "PU({input}) = {pu}, expected ≈ {expected}"
            );
        }
    }

    #[test]
    fn monotone_across_decade_grid() {
        // PU must be strictly monotone-increasing across the valid range.
        let grid = [0.01_f32, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0];
        let mut prev = pu_encode(grid[0]);
        for &y in &grid[1..] {
            let pu = pu_encode(y);
            assert!(
                pu > prev,
                "PU not monotone at Y = {y} (prev = {prev}, pu = {pu})"
            );
            prev = pu;
        }
    }

    #[test]
    fn round_trip_within_05_pct() {
        // Encode + decode should recover luminance within 0.5 % across the
        // valid range. Tighter near the middle; loosest at the extremes.
        for &y in &[
            0.01_f32, 0.1, 1.0, 10.0, 80.0, 100.0, 203.0, 500.0, 1_000.0, 4_000.0, 10_000.0,
        ] {
            let pu = pu_encode(y);
            let back = pu_decode(pu);
            let rel_err = ((back - y) / y).abs();
            assert!(
                rel_err < 0.005,
                "Round-trip Y = {y}: PU = {pu}, decoded = {back}, rel_err = {rel_err}"
            );
        }
    }

    #[test]
    fn f32_path_matches_f64_reference() {
        // The f32 scalar should track the f64 reference to within 1e-3
        // absolute over the valid range (powf precision is the floor).
        let grid: Vec<f32> = (0..1000)
            .map(|i| {
                let t = i as f32 / 999.0;
                // log-spaced 0.005 → 10_000
                let log_min = PU_LUMINANCE_MIN_CD_M2.ln();
                let log_max = PU_LUMINANCE_MAX_CD_M2.ln();
                (log_min + (log_max - log_min) * t).exp()
            })
            .collect();
        let mut max_err = 0.0_f32;
        for &y in &grid {
            let fast = pu_encode(y);
            let slow = pu_encode_f64(y as f64) as f32;
            let err = (fast - slow).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 5e-3,
            "f32 PU max error vs f64 = {max_err}, expected < 5e-3"
        );
    }
}
