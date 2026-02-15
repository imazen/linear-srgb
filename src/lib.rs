//! Fast linear↔sRGB color space conversion.
//!
//! This crate provides efficient conversion between linear light values and
//! sRGB gamma-encoded values following the IEC 61966-2-1:1999 standard.
//!
//! # Module Organization
//!
//! - [`default`] - **Recommended API** with optimal implementations for each use case
//! - [`simd`] - SIMD-accelerated functions with full control over dispatch
//! - [`scalar`] - Single-value conversion functions (f32/f64)
//! - [`lut`] - Lookup table types for custom bit depths
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
//! | u8 slice → f32 slice | [`default::srgb_u8_to_linear_slice`] |
//! | Manual SIMD (8 values) | [`default::srgb_to_linear_x8`] |
//! | Inside `#[magetypes]` | [`default::inline::srgb_to_linear_x8`] |
//! | Inside `#[arcane]` (token) | [`rites::x8::srgb_to_linear_v3`] |
//! | Custom bit depth LUT | [`lut::LinearTable16`] |
//!
//! # Feature Flags
//!
//! - `std` (default): Enable std library support
//! - `unsafe_simd`: Enable unsafe optimizations for maximum performance
//!
//! # `no_std` Support
//!
//! This crate is `no_std` compatible. Disable the `std` feature:
//!
//! ```toml
//! linear-srgb = { version = "0.2", default-features = false }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "unsafe_simd"), deny(unsafe_code))]
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
/// See module documentation for details.
pub mod default;

/// Lookup table types for sRGB conversion.
///
/// Provides both build-time const tables ([`SrgbConverter`](lut::SrgbConverter))
/// and runtime-generated tables for custom bit depths.
pub mod lut;

/// SIMD-accelerated conversion functions.
///
/// Provides full control over CPU dispatch with `_dispatch` and `_inline` variants.
pub mod simd;

/// Scalar (single-value) conversion functions.
///
/// Direct computation without SIMD. Best for individual value conversions.
pub mod scalar;

/// Inlineable `#[rite]` functions for embedding in your own `#[arcane]` code.
///
/// These carry `#[target_feature]` + `#[inline]` directly — no wrapper, no
/// dispatch. When called from a matching `#[arcane]` context, LLVM inlines
/// them fully. Organized by SIMD unit width; suffixed by required token tier.
pub mod rites;

/// Token-based API using archmage for zero dispatch overhead.
///
/// This module provides an alternative API using archmage tokens for users who
/// want to avoid per-call dispatch overhead. Obtain a token once at startup,
/// then pass it to all conversion functions.
///
/// Requires the `mage` feature.
#[cfg(feature = "mage")]
pub mod mage;

// ============================================================================
// Internal modules
// ============================================================================

mod mlaf;

// Internal fast math for SIMD (not public API)
pub(crate) mod fast_math;

// Pre-computed const lookup tables (embedded in binary)
mod const_luts;
mod const_luts_u16;

// Alternative/experimental implementations (for benchmarking)
#[cfg(feature = "alt")]
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
