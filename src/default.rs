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
//! - `*_x8` - Default with CPU dispatch (standalone use)
//! - [`inline`] module - `#[inline(always)]` variants for use inside your own `#[magetypes]` code
//!
//! ```rust
//! use linear_srgb::default::{linear_to_srgb_x8, linear_to_srgb_u8_x8};
//! use wide::f32x8;
//!
//! let linear = f32x8::splat(0.214);
//! let srgb = linear_to_srgb_x8(linear);  // CPU dispatch
//! let srgb_u8 = linear_to_srgb_u8_x8(linear);
//! ```
//!
//! For use inside `#[magetypes]` functions (no dispatch overhead):
//! ```rust,ignore
//! use linear_srgb::default::inline::*;
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
    // u16 sRGB (LUT-based)
    linear_to_srgb_u16,
    srgb_to_linear,
    srgb_to_linear_extended,
    srgb_to_linear_f64,
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
    // f32x8 slices (for pre-aligned SIMD data)
    gamma_to_linear_x8_slice,
    linear_to_gamma_slice,
    linear_to_gamma_x8_slice,
    // f32 slices (in-place)
    linear_to_srgb_slice,
    // u8 ↔ f32 slices
    linear_to_srgb_u8_slice,
    // u16 slices
    linear_to_srgb_u16_slice,
    linear_to_srgb_x8_slice,
    srgb_to_linear_slice,
    srgb_to_linear_x8_slice,
    srgb_u8_to_linear_slice,
    srgb_u16_to_linear_slice,
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
// LUT converter (zero-cost const tables)
// ============================================================================

pub use crate::lut::SrgbConverter;

pub mod inline {
    //! Dispatch-free inline variants for use inside `#[magetypes]` functions.
    //!
    //! When building your own SIMD-accelerated functions with `archmage`,
    //! use these `_inline` variants to avoid nested dispatch overhead.
    //! These functions are `#[inline(always)]` and contain no dispatch overhead.
    //!
    //! # Example
    //!
    //! ```rust,ignore
    //! use linear_srgb::default::inline::*;
    //! use archmage::magetypes;
    //! use wide::f32x8;
    //!
    //! #[magetypes(v3)]  // Your function handles dispatch
    //! fn process_pixels(_token: Token, data: &mut [f32]) {
    //!     for chunk in data.chunks_exact_mut(8) {
    //!         let v = f32x8::from([
    //!             chunk[0], chunk[1], chunk[2], chunk[3],
    //!             chunk[4], chunk[5], chunk[6], chunk[7],
    //!         ]);
    //!         // Use inline variants - no dispatch, just raw SIMD
    //!         let linear = srgb_to_linear_x8(v);
    //!         let processed = linear * f32x8::splat(1.5);
    //!         let result = linear_to_srgb_x8(processed);
    //!         let arr: [f32; 8] = result.into();
    //!         chunk.copy_from_slice(&arr);
    //!     }
    //! }
    //! ```

    // Re-export inline x8 functions with clean names (no _inline suffix)
    pub use crate::simd::{
        gamma_to_linear_x8_inline as gamma_to_linear_x8,
        linear_to_gamma_x8_inline as linear_to_gamma_x8,
        linear_to_srgb_u8_x8_inline as linear_to_srgb_u8_x8,
        linear_to_srgb_x8_inline as linear_to_srgb_x8,
        srgb_to_linear_x8_inline as srgb_to_linear_x8,
    };

    // Re-export inline x8 slice functions with clean names (no _inline suffix)
    pub use crate::simd::{
        gamma_to_linear_x8_slice_inline as gamma_to_linear_x8_slice,
        linear_to_gamma_x8_slice_inline as linear_to_gamma_x8_slice,
        linear_to_srgb_x8_slice_inline as linear_to_srgb_x8_slice,
        srgb_to_linear_x8_slice_inline as srgb_to_linear_x8_slice,
    };
}
