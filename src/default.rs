//! Recommended API for sRGB ↔ linear conversion.
//!
//! This module provides the optimal implementation for each use case:
//!
//! - **Single values**: Scalar functions (SIMD overhead not worthwhile)
//! - **Slices**: SIMD-accelerated with runtime CPU dispatch
//! - **x8 batches**: SIMD with dispatch (`_dispatch`) or inlineable (`_inline`)
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
//!
//! # x8 SIMD Functions
//!
//! For manual SIMD control, use the x8 functions:
//!
//! - `*_x8` - Default with CPU dispatch
//! - `*_x8_inline` - `#[inline(always)]`, for use inside your own `#[multiversed]` code
//!
//! ```rust
//! use linear_srgb::default::{srgb_to_linear_x8, linear_to_srgb_x8};
//! use wide::f32x8;
//!
//! let srgb = f32x8::splat(0.5);
//! let linear = srgb_to_linear_x8(srgb);  // CPU dispatch
//! let back = linear_to_srgb_x8(linear);
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
    // f32 sRGB
    linear_to_srgb,
    linear_to_srgb_extended,
    // f64 sRGB (high precision)
    linear_to_srgb_f64,
    linear_to_srgb_u8,
    srgb_to_linear,
    srgb_to_linear_extended,
    srgb_to_linear_f64,
    srgb_u8_to_linear,
};

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
    srgb_to_linear_slice,
    srgb_u8_to_linear_slice,
};

// ============================================================================
// x8 SIMD functions with CPU dispatch (default)
// ============================================================================

pub use crate::simd::{
    // Custom gamma x8 with dispatch
    gamma_to_linear_x8_dispatch as gamma_to_linear_x8,
    linear_to_gamma_x8_dispatch as linear_to_gamma_x8,
    // sRGB x8 with dispatch
    linear_to_srgb_u8_x8_dispatch as linear_to_srgb_u8_x8,
    linear_to_srgb_x8_dispatch as linear_to_srgb_x8,
    srgb_to_linear_x8_dispatch as srgb_to_linear_x8,
    // u8 x8 (LUT-based, no dispatch needed)
    srgb_u8_to_linear_x8,
};

// ============================================================================
// x8 SIMD inline functions (for embedding in caller's multiversed code)
// ============================================================================

pub use crate::simd::{
    gamma_to_linear_x8_inline, linear_to_gamma_x8_inline, linear_to_srgb_u8_x8_inline,
    linear_to_srgb_x8_inline, srgb_to_linear_x8_inline,
};

// ============================================================================
// LUT converter (zero-cost const tables)
// ============================================================================

pub use crate::lut::SrgbConverter;
