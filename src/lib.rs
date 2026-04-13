//! Fast linear↔sRGB color space conversion.
//!
//! This crate provides efficient conversion between linear light values and
//! sRGB gamma-encoded values, with multiple implementation strategies for
//! different accuracy/performance tradeoffs.
//!
//! # Module Organization
//!
//! - [`default`] — **Start here.** Rational polynomial for f32, LUT for integers, SIMD for slices.
//! - [`precise`] — Exact `powf()` with C0-continuous constants. f32/f64, extended range. Slower.
//! - [`tokens`] — Inlineable `#[rite]` functions for embedding in your own `#[arcane]` SIMD code.
//! - [`lut`] — Lookup tables for custom bit depths (10-bit, 12-bit, 16-bit).
//! - **`tf`** — Transfer functions beyond sRGB: BT.709, PQ, HLG. Requires `transfer` feature.
//! - **`iec`** — IEC 61966-2-1 textbook constants (legacy interop). Requires `iec` feature.
//!
//! # Quick Start
//!
//! ```rust
//! use linear_srgb::default::{srgb_to_linear, linear_to_srgb};
//!
//! // Convert sRGB 0.5 to linear
//! let linear = srgb_to_linear(0.5);
//! assert!((linear - 0.214).abs() < 0.001);
//!
//! // Convert back to sRGB
//! let srgb = linear_to_srgb(linear);
//! assert!((srgb - 0.5).abs() < 0.001);
//! ```
//!
//! # Batch Processing (SIMD)
//!
//! For maximum throughput on slices:
//!
//! ```rust
//! use linear_srgb::default::{srgb_to_linear_slice, linear_to_srgb_slice};
//!
//! let mut values = vec![0.5f32; 10000];
//! srgb_to_linear_slice(&mut values);  // SIMD-accelerated
//! linear_to_srgb_slice(&mut values);
//! ```
//!
//! # Custom Gamma
//!
//! For non-sRGB gamma (pure power function without linear segment):
//!
//! ```rust
//! use linear_srgb::default::{gamma_to_linear, linear_to_gamma};
//!
//! let linear = gamma_to_linear(0.5, 2.2);  // gamma 2.2
//! let encoded = linear_to_gamma(linear, 2.2);
//! ```
//!
//! # LUT-based Conversion
//!
//! For batch processing with pre-computed lookup tables:
//!
//! ```rust
//! use linear_srgb::default::SrgbConverter;
//!
//! let conv = SrgbConverter::new();  // Zero-cost, const tables
//!
//! // Fast 8-bit conversions
//! let linear = conv.srgb_u8_to_linear(128);
//! let srgb = conv.linear_to_srgb_u8(linear);
//! ```
//!
//! # Choosing the Right API
//!
//! | Use Case | Recommended Function |
//! |----------|---------------------|
//! | Single f32 value | [`default::srgb_to_linear`] |
//! | Single u8 value | [`default::srgb_u8_to_linear`] |
//! | f32 slice (in-place) | [`default::srgb_to_linear_slice`] |
//! | RGBA f32 slice (alpha-preserving) | [`default::srgb_to_linear_rgba_slice`] |
//! | u8 slice → f32 slice | [`default::srgb_u8_to_linear_slice`] |
//! | RGBA u8 → f32 (alpha-preserving) | [`default::srgb_u8_to_linear_rgba_slice`] |
//! | RGBA f32 sRGB → linear premul | [`default::srgb_to_linear_premultiply_rgba_slice`] |
//! | RGBA u8 sRGB → linear premul f32 | [`default::srgb_u8_to_linear_premultiply_rgba_slice`] |
//! | RGBA f32 linear premul → sRGB | [`default::unpremultiply_linear_to_srgb_rgba_slice`] |
//! | RGBA f32 linear premul → sRGB u8 | [`default::unpremultiply_linear_to_srgb_u8_rgba_slice`] |
//! | u16 → f32 slice | [`default::srgb_u16_to_linear_slice`] |
//! | f32 → u16 (exact RT) | [`default::linear_to_srgb_u16`] |
//! | f32 → u16 (fast, ±1 RT) | [`default::linear_to_srgb_u16_fast`] |
//! | Exact f32/f64 (powf) | [`precise::srgb_to_linear`] |
//! | Extended range (HDR) | [`precise::srgb_to_linear_extended`] |
//! | Inside `#[arcane]` | `tokens::x8::srgb_to_linear_v3` |
//! | Custom bit depth LUT | [`lut::LinearTable16`] |
//!
//! # Clamping and Extended Range
//!
//! The f32↔f32 conversion functions come in two flavors: **clamped** (default)
//! and **extended** (unclamped). Integer paths (u8, u16) always clamp since
//! out-of-range values can't be represented in the output format.
//!
//! ## Clamped (default) — use for same-gamut pipelines
//!
//! All functions except the `_extended` variants clamp inputs to \[0, 1\]:
//! negatives become 0, values above 1 become 1.
//!
//! This is correct whenever the source and destination share the same color
//! space (gamut + transfer function). The typical pipeline:
//!
//! 1. Decode sRGB image (u8 → linear f32 via LUT, or f32 via TRC)
//! 2. Process in linear light (resize, blur, blend, composite)
//! 3. Re-encode to sRGB (linear f32 → sRGB f32 or u8)
//!
//! In this pipeline, out-of-range values only come from processing artifacts:
//! resize filters with negative lobes (Lanczos, Mitchell, etc.) produce small
//! negatives near dark edges and values slightly above 1.0 near bright edges.
//! These are ringing artifacts, not real colors — clamping is correct.
//!
//! Float decoders like jpegli can also produce small out-of-range values from
//! YCbCr quantization noise. When the image is sRGB, these are compression
//! artifacts and clamping is correct — gives the same result as decoding to
//! u8 first.
//!
//! ## Extended (unclamped) — use for cross-gamut pipelines
//!
//! [`precise::srgb_to_linear_extended`] and [`precise::linear_to_srgb_extended`]
//! do not clamp. They follow the mathematical sRGB transfer function for all
//! inputs: negatives pass through the linear segment, values above 1.0 pass
//! through the power segment.
//!
//! Use these when the sRGB transfer function is applied to values from a
//! **different, wider gamut**. A 3×3 matrix converting Rec. 2020 linear or
//! Display P3 linear to sRGB linear can produce values well outside \[0, 1\]:
//! a saturated Rec. 2020 green maps to deeply negative sRGB red and blue.
//! These are real out-of-gamut colors, not artifacts — clamping destroys
//! information that downstream gamut mapping or compositing may need.
//!
//! This matters in practice: JPEG and JPEG XL images can carry Rec. 2020 or
//! Display P3 ICC profiles. Phones shoot Rec. 2020 HLG, cameras embed
//! wide-gamut profiles. Decoding such an image and converting to sRGB for
//! display produces out-of-gamut values that should survive until final
//! output.
//!
//! If a float decoder (jpegli, libjxl) outputs wide-gamut data directly to
//! f32, the output contains both small compression artifacts and real
//! out-of-gamut values. The artifacts are tiny; the gamut excursions
//! dominate. Using `_extended` preserves both — the artifacts are harmless
//! noise that vanishes at quantization.
//!
//! The `_extended` variants also cover **scRGB** (float sRGB with values
//! outside \[0, 1\] for HDR and wide color) and any pipeline where
//! intermediate f32 values are not yet at the final output stage.
//!
//! ## Summary
//!
//! | Function | Range | Pipeline |
//! |----------|-------|----------|
//! | All `default::*_slice`, `tokens::*`, `lut::*` | \[0, 1\] | Same-gamut batch processing |
//! | [`default::srgb_to_linear`] | \[0, 1\] | Same-gamut single values |
//! | [`default::linear_to_srgb`] | \[0, 1\] | Same-gamut single values |
//! | [`precise::srgb_to_linear_extended`] | Unbounded | Cross-gamut, scRGB, HDR (scalar) |
//! | [`precise::linear_to_srgb_extended`] | Unbounded | Cross-gamut, scRGB, HDR (scalar) |
//! | [`default::srgb_to_linear_extended_slice`] | Unbounded | Cross-gamut, scRGB, HDR (SIMD batch) |
//! | [`default::linear_to_srgb_extended_slice`] | Unbounded | Cross-gamut, scRGB, HDR (SIMD batch) |
//! | All u8/u16 paths | \[0, 1\] | Final quantization (clamp inherent) |
//!
//! The `_extended_slice` functions use the fast SIMD polynomial for the
//! common \[0, 1\] case and fix up out-of-range lanes with scalar `powf`.
//! This is optimal when most pixels are in-gamut — only out-of-gamut
//! lanes pay the `powf` cost.
//!
//! # Feature Flags
//!
//! - **`std`** (default) — Enable runtime SIMD dispatch. Required for slice functions.
//! - **`avx512`** (default) — Enable AVX-512 code paths and `tokens::x16` module.
//! - **`transfer`** — BT.709, PQ, and HLG transfer functions in `tf` and [`tokens`].
//! - **`iec`** — IEC 61966-2-1 textbook sRGB functions for legacy interop.
//! - **`alt`** — Alternative implementations for benchmarking (not stable API).
//! - **`unsafe_simd`** — No-op (kept for backward compatibility, will be removed in 0.7).
//!
//! # `no_std` Support
//!
//! This crate is `no_std` compatible. Without `std`, u16 functions use the
//! rational polynomial instead of LUT (slower but no heap allocation).
//! Disable the `std` feature:
//!
//! ```toml
//! linear-srgb = { version = "0.6", default-features = false }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(all(test, not(feature = "std")))]
extern crate std;

// ============================================================================
// Public modules
// ============================================================================

/// Recommended API with optimal implementations for each use case.
///
/// Uses a rational polynomial for single f32 values (≤14 ULP, perfectly
/// monotonic), LUT for integer types, and SIMD-dispatched batch processing
/// for slices.
pub mod default;

/// Exact `powf()`-based conversions with C0-continuous constants.
///
/// Uses C0-continuous constants (from the moxcms reference implementation) that
/// eliminate the IEC 61966-2-1 piecewise discontinuity. ~6 ULP max error
/// vs f64 reference. See the module docs for the constant comparison table.
///
/// Also provides f64, extended-range (unclamped), and custom gamma functions.
/// For faster alternatives, use [`default`].
pub mod precise;

/// Lookup table types for sRGB conversion.
///
/// Provides both build-time const tables ([`SrgbConverter`](lut::SrgbConverter))
/// and runtime-generated tables for custom bit depths (10-bit, 12-bit, 16-bit).
pub mod lut;

/// Inlineable `#[rite]` functions for embedding in your own `#[arcane]` code.
///
/// These carry `#[target_feature]` + `#[inline]` directly — no wrapper, no
/// dispatch. When called from a matching `#[arcane]` context, LLVM inlines
/// them fully. Organized by SIMD width; suffixed by required token tier.
///
/// Also re-exports token types for convenience: `X64V3Token`, `X64V4Token`,
/// `NeonToken`, `Wasm128Token` (each gated to its target architecture).
///
/// When the `transfer` feature is enabled, each width module also provides
/// rites for BT.709, PQ, and HLG (prefixed with `tf_` for sRGB to avoid
/// name collisions with the rational polynomial sRGB rites).
pub mod tokens;

/// Transfer functions: sRGB, BT.709, PQ (ST 2084), HLG (ARIB STD-B67).
///
/// Provides scalar functions for all four transfer curves. SIMD `#[rite]`
/// versions live in [`tokens`] (x4/x8/x16).
///
/// Requires the `transfer` feature.
#[cfg(feature = "transfer")]
pub mod tf;

/// IEC 61966-2-1:1999 textbook sRGB transfer functions.
///
/// Provides the original specification constants (threshold 0.04045, offset 0.055)
/// for interoperability with software that implements IEC 61966-2-1 verbatim.
/// The default module uses C0-continuous constants that eliminate the spec's
/// ~2.3e-9 piecewise discontinuity.
///
/// Requires the `iec` feature.
#[cfg(feature = "iec")]
pub mod iec;

// ============================================================================
// Internal modules
// ============================================================================

pub(crate) mod scalar;
pub(crate) mod simd;

mod mlaf;

// Rational polynomial sRGB approximation (shared coefficients + scalar evaluator)
pub(crate) mod rational_poly;

// Pre-computed const lookup tables (embedded in binary)
mod const_luts;

// Lazily-initialized u16 sRGB LUTs (OnceLock, allocated on first use)
#[cfg(feature = "std")]
#[doc(hidden)]
pub mod u16_lut;

// Alternative/experimental implementations (for benchmarking, not stable API)
#[cfg(feature = "alt")]
#[doc(hidden)]
pub mod alt;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use crate::default::*;

    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    #[test]
    fn test_api_consistency() {
        // Ensure direct and LUT-based conversions are consistent
        let conv = SrgbConverter::new();

        for i in 0..=255u8 {
            let direct = srgb_u8_to_linear(i);
            let lut = conv.srgb_u8_to_linear(i);
            assert!(
                (direct - lut).abs() < 1e-5,
                "Mismatch at {}: direct={}, lut={}",
                i,
                direct,
                lut
            );
        }
    }

    #[test]
    fn test_slice_conversion() {
        let mut values: Vec<f32> = (0..=10).map(|i| i as f32 / 10.0).collect();
        let original = values.clone();

        srgb_to_linear_slice(&mut values);
        linear_to_srgb_slice(&mut values);

        for (i, (orig, conv)) in original.iter().zip(values.iter()).enumerate() {
            assert!(
                (orig - conv).abs() < 1e-5,
                "Slice roundtrip failed at {}: {} -> {}",
                i,
                orig,
                conv
            );
        }
    }
}
