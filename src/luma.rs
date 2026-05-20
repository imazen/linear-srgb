//! Luma coefficients for common color-space standards.
//!
//! These coefficients define how to derive luma `Y` from linear-light
//! RGB. They appear in HDR-IQA front-ends, YCbCr matrices, and
//! gamut-conversion paths where downstream callers don't want to
//! reconstruct them from primaries.
//!
//! Each set sums to `1.0` and matches the published ITU recommendation
//! verbatim. Values are exposed as `[f32; 3]` (R, G, B order) for direct
//! consumption by SIMD dot-product kernels.
//!
//! ## Source recommendations
//!
//! | Constant | Spec | Where used |
//! |---|---|---|
//! | [`BT2020_NCL_LUMA`] | ITU-R BT.2020 | UHDTV, HDR10, PQ chain |
//! | [`BT709_LUMA`] | ITU-R BT.709 | sRGB / HDTV |
//! | [`BT601_LUMA`] | ITU-R BT.601 | SDTV, JPEG YCbCr |

/// BT.2020 NCL (non-constant-luminance) luma coefficients.
///
/// Defined by ITU-R BT.2020, used by HDR10, PQ, BT.2100, and the HDR
/// gain-map specs. The non-constant-luminance variant is the matrix
/// used in every real-world container — the constant-luminance variant
/// in BT.2020 is theoretical only.
pub const BT2020_NCL_LUMA: [f32; 3] = [0.2627, 0.6780, 0.0593];

/// BT.709 luma coefficients.
///
/// Defined by ITU-R BT.709. Used by sRGB-primary YCbCr at HDTV
/// resolutions and as the default for SDR JPEG / JPEG XL on Rec.709
/// primaries.
pub const BT709_LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// BT.601 luma coefficients (legacy SDTV).
///
/// Defined by ITU-R BT.601. Used by SDTV YCbCr and the canonical
/// JPEG color-space matrix when the encoder targets pre-HD content.
/// Modern web content uses [`BT709_LUMA`].
pub const BT601_LUMA: [f32; 3] = [0.299, 0.587, 0.114];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_set_sums_to_one() {
        for (name, set) in [
            ("BT2020_NCL", BT2020_NCL_LUMA),
            ("BT709", BT709_LUMA),
            ("BT601", BT601_LUMA),
        ] {
            let sum: f32 = set.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "{name} luma coefficients sum to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn green_is_largest_in_every_standard() {
        // Every photopic-luminance set must put the most weight on
        // green (human eye peak ~555 nm).
        for set in [BT2020_NCL_LUMA, BT709_LUMA, BT601_LUMA] {
            assert!(set[1] > set[0] && set[1] > set[2]);
        }
    }
}
