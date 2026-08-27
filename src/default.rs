//! Recommended API for sRGB ↔ linear conversion.
//!
//! This module re-exports the optimal implementation for each use case:
//!
//! - **Single f32 values**: Rational polynomial (~14 ULP max, perfectly
//!   monotonic — no `powf`)
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
// Single-value sRGB f32 (rational polynomial — fast, ≤14 ULP, monotonic)
// ============================================================================

pub use crate::rational_poly::{
    linear_to_srgb_fast as linear_to_srgb, srgb_to_linear_fast as srgb_to_linear,
};

// ============================================================================
// Single-value sRGB integer (LUT lookup — zero math)
// ============================================================================

pub use crate::scalar::{
    linear_to_srgb_u8, linear_to_srgb_u16, linear_to_srgb_u16_fast, srgb_u8_to_linear,
    srgb_u16_to_linear,
};

// ============================================================================
// Slice functions (SIMD-dispatched)
// ============================================================================

pub use crate::simd::{
    // Custom gamma RGBA slices (alpha-preserving)
    gamma_to_linear_rgba_slice,
    // Custom gamma slices
    gamma_to_linear_slice,
    linear_to_gamma_rgba_slice,
    linear_to_gamma_slice,
    // Extended-range f32 RGBA slices (no clamping, alpha-preserving)
    linear_to_srgb_extended_rgba_slice,
    // Extended-range f32 slices (no clamping, sign-preserving)
    linear_to_srgb_extended_slice,
    // f32 RGBA slices (alpha-preserving)
    linear_to_srgb_rgba_slice,
    // f32 slices (in-place)
    linear_to_srgb_slice,
    linear_to_srgb_u8_rgba_slice,
    // u8 slices
    linear_to_srgb_u8_slice,
    linear_to_srgb_u16_rgba_slice,
    linear_to_srgb_u16_rgba_slice_fast,
    // u16 slices
    linear_to_srgb_u16_slice,
    linear_to_srgb_u16_slice_fast,
    srgb_to_linear_extended_rgba_slice,
    srgb_to_linear_extended_slice,
    // Fused premultiply/unpremultiply
    srgb_to_linear_premultiply_rgba_slice,
    // f32 RGBA slices (alpha-preserving)
    srgb_to_linear_rgba_slice,
    // f32 slices (in-place)
    srgb_to_linear_slice,
    srgb_u8_to_linear_premultiply_rgba_slice,
    srgb_u8_to_linear_rgba_slice,
    // u8 slices
    srgb_u8_to_linear_slice,
    srgb_u16_to_linear_rgba_slice,
    // u16 slices
    srgb_u16_to_linear_slice,
    unpremultiply_linear_to_srgb_rgba_slice,
    unpremultiply_linear_to_srgb_u8_rgba_slice,
};

// Deprecated gamma premultiply/unpremultiply re-exports
#[allow(deprecated)]
pub use crate::simd::{
    gamma_to_linear_premultiply_rgba_slice, unpremultiply_linear_to_gamma_rgba_slice,
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
// Transfer function slice operations (behind `transfer` feature)
// ============================================================================

#[cfg(feature = "transfer")]
pub use crate::simd::{
    // RGBA variants (alpha-preserving; every 4th element left unchanged)
    bt709_to_linear_rgba_slice,
    // Channel-agnostic slice (applies TF to every element — including alpha
    // if RGBA data is passed; use the *_rgba_slice variants for alpha-preserving)
    bt709_to_linear_slice,
    hlg_to_linear_rgba_slice,
    hlg_to_linear_slice,
    linear_to_bt709_rgba_slice,
    linear_to_bt709_slice,
    linear_to_hlg_rgba_slice,
    linear_to_hlg_slice,
    linear_to_pq_rgba_slice,
    linear_to_pq_slice,
    pq_to_linear_rgba_slice,
    pq_to_linear_slice,
};

// ============================================================================
// LUT converter (zero-cost const tables)
// ============================================================================

pub use crate::lut::SrgbConverter;
