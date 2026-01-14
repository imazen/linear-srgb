//! Fast linear↔sRGB color space conversion.
//!
//! This crate is `no_std` compatible by default. Enable the `std` feature
//! if you need std library support.
//!
//! This crate provides efficient conversion between linear light values and
//! sRGB gamma-encoded values following the IEC 61966-2-1:1999 standard.
//!
//! # Features
//!
//! - **Direct computation**: Single value conversion with piecewise functions
//! - **FMA acceleration**: Uses hardware FMA when available (x86 FMA, ARM64 NEON)
//! - **LUT-based conversion**: Pre-computed tables for batch processing
//! - **Multiple precisions**: f32, f64, and u8/u16 conversions
//!
//! # Quick Start
//!
//! ```rust
//! use linear_srgb::{srgb_to_linear, linear_to_srgb};
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
//! # LUT-based Conversion
//!
//! For batch processing, use `SrgbConverter` which pre-computes lookup tables:
//!
//! ```rust
//! use linear_srgb::SrgbConverter;
//!
//! let conv = SrgbConverter::new();
//!
//! // Fast 8-bit conversions
//! let linear = conv.srgb_u8_to_linear(128);
//! let srgb = conv.linear_to_srgb_u8(linear);
//! ```
//!
//! # Performance
//!
//! The implementation uses several optimizations:
//! - Piecewise functions avoid `pow()` for ~1.2% of values in the linear segment
//! - Early exit for out-of-range values avoids expensive transcendentals
//! - FMA instructions combine multiply+add into single-cycle operations
//! - Pre-computed LUTs trade memory for compute time
//!
//! # Feature Flags
//!
//! - `fast-math`: Use faster but slightly less accurate pow approximation for
//!   extended range conversions (affects `linear_to_srgb_extended` only)
//!
//! # SIMD Acceleration
//!
//! For maximum throughput on large batches, use the `simd` module:
//!
//! ```rust
//! use linear_srgb::simd;
//!
//! let mut values = vec![0.5f32; 10000];
//! simd::srgb_to_linear_slice(&mut values);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "unsafe_simd"), deny(unsafe_code))]
#![warn(missing_docs)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(all(test, not(feature = "std")))]
extern crate std;

pub mod lut;
mod mlaf;
pub mod simd;
mod targets;
pub mod transfer;

// Internal fast math for SIMD (not public API)
pub(crate) mod fast_math;

// Alternative/experimental implementations (for benchmarking)
#[cfg(feature = "alt")]
pub mod alt;

// Re-export main types and functions
pub use lut::{
    EncodeTable8, EncodeTable12, EncodeTable16, EncodingTable, LinearTable8, LinearTable10,
    LinearTable12, LinearTable16, LinearizationTable, SrgbConverter, lut_interp_linear_float,
    lut_interp_linear_u16,
};

pub use transfer::{
    linear_to_srgb, linear_to_srgb_extended, linear_to_srgb_f64, linear_to_srgb_u8, srgb_to_linear,
    srgb_to_linear_extended, srgb_to_linear_f64, srgb_u8_to_linear,
};

/// Convert a slice of sRGB f32 values to linear in-place.
#[inline]
pub fn srgb_to_linear_slice(values: &mut [f32]) {
    for v in values.iter_mut() {
        *v = srgb_to_linear(*v);
    }
}

/// Convert a slice of linear f32 values to sRGB in-place.
#[inline]
pub fn linear_to_srgb_slice(values: &mut [f32]) {
    for v in values.iter_mut() {
        *v = linear_to_srgb(*v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
