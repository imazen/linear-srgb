//! Recommended API for sRGB ↔ linear conversion.
//!
//! This module provides the optimal implementation for each use case:
//!
//! - **Single values**: Scalar functions (SIMD overhead not worthwhile)
//! - **Slices**: SIMD-accelerated with runtime CPU dispatch
//!
//! # Quick Start
//!
//! ```rust
//! use linear_srgb::default::{srgb_to_linear, linear_to_srgb};
//!
//! // Single value conversion
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
// Single-value functions (scalar - best for individual values)
// ============================================================================

pub use crate::scalar::{
    // Custom gamma (pure power function)
    gamma_to_linear,
    gamma_to_linear_f64,
    linear_to_gamma,
    linear_to_gamma_f64,
    // f32 sRGB (exact powf)
    linear_to_srgb,
    linear_to_srgb_extended,
    // f64 sRGB (high precision)
    linear_to_srgb_f64,
    // f32 sRGB (fast polynomial, no powf)
    linear_to_srgb_fast,
    linear_to_srgb_u8,
    // u16 sRGB (LUT-based)
    linear_to_srgb_u16,
    srgb_to_linear,
    srgb_to_linear_extended,
    srgb_to_linear_f64,
    // f32 sRGB (fast polynomial, no powf)
    srgb_to_linear_fast,
    srgb_u16_to_linear,
};

// u8 → f32 uses LUT (20x faster than scalar powf)
pub use crate::simd::srgb_u8_to_linear;

// ============================================================================
// Slice functions (SIMD with dispatch - best for batches)
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
// LUT converter (zero-cost const tables)
// ============================================================================

pub use crate::lut::SrgbConverter;
