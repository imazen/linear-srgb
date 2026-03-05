//! Recommended API for sRGB ↔ linear conversion.
//!
//! This module re-exports the optimal implementation for each use case:
//!
//! - **Single f32 values**: Rational polynomial (~110 ULP max at the piecewise
//!   threshold, <8 ULP elsewhere — no `powf`)
//! - **Single u8/u16 values**: LUT lookup (zero math)
//! - **Slices**: SIMD-accelerated with runtime CPU dispatch
//! - **Custom gamma**: Pure power function (f32, slices)
//!
//! For exact `powf()` conversions with C0-continuous constants, see [`crate::precise`].
//!
//! # Quick Start
//!
//! ```rust
//! use linear_srgb::default::{srgb_to_linear, linear_to_srgb};
//!
//! let linear = srgb_to_linear(0.5);
//! let srgb = linear_to_srgb(linear);
//! ```
//!
//! # Batch Processing
//!
//! ```rust
//! use linear_srgb::default::{srgb_to_linear_slice, linear_to_srgb_slice};
//!
//! let mut values = vec![0.5f32; 10000];
//! srgb_to_linear_slice(&mut values);  // SIMD-accelerated
//! linear_to_srgb_slice(&mut values);
//! ```

// ============================================================================
// Single-value sRGB f32 (rational polynomial — fast, ~110 ULP max at threshold)
// ============================================================================

pub use crate::rational_poly::{
    linear_to_srgb_fast as linear_to_srgb, srgb_to_linear_fast as srgb_to_linear,
};

// ============================================================================
// Single-value sRGB integer (LUT lookup — zero math)
// ============================================================================

pub use crate::scalar::{
    linear_to_srgb_u8, linear_to_srgb_u16, srgb_u8_to_linear, srgb_u16_to_linear,
};

// ============================================================================
// Slice functions (SIMD-dispatched)
// ============================================================================

pub use crate::simd::{
    // Custom gamma slices
    gamma_to_linear_slice,
    linear_to_gamma_slice,
    // f32 slices (in-place)
    linear_to_srgb_slice,
    // u8 ↔ f32 slices
    linear_to_srgb_u8_slice,
    // u16 slices
    linear_to_srgb_u16_slice,
    srgb_to_linear_slice,
    srgb_u8_to_linear_slice,
    srgb_u16_to_linear_slice,
};

// ============================================================================
// Custom gamma (scalar)
// ============================================================================

pub use crate::scalar::{gamma_to_linear, linear_to_gamma};

// ============================================================================
// Transfer functions (behind `transfer` feature)
// ============================================================================

#[cfg(feature = "transfer")]
pub use crate::tf::{
    bt709_to_linear, hlg_to_linear, linear_to_bt709, linear_to_hlg, linear_to_pq, pq_to_linear,
};

// ============================================================================
// LUT converter (zero-cost const tables)
// ============================================================================

pub use crate::lut::SrgbConverter;
