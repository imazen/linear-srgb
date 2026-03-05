//! Slow but exact conversion functions using `powf()`.
//!
//! These functions use the mathematical sRGB transfer function with `powf()`,
//! providing maximum accuracy at the cost of speed. For faster alternatives,
//! use [`crate::default`] which provides rational polynomial approximations
//! with ~8 ULP max error.
//!
//! # Extended-range variants
//!
//! The `_extended` functions do not clamp to \[0, 1\], making them suitable
//! for HDR, ICC, and cross-gamut pipelines. See the crate-level docs on
//! clamping for details.

// sRGB f32 (powf, clamped)
pub use crate::scalar::{linear_to_srgb, srgb_to_linear};

// sRGB f32 (powf, extended/unclamped)
pub use crate::scalar::{linear_to_srgb_extended, srgb_to_linear_extended};

// sRGB f64 (powf)
pub use crate::scalar::{linear_to_srgb_f64, srgb_to_linear_f64};

// Custom gamma f64
pub use crate::scalar::{gamma_to_linear_f64, linear_to_gamma_f64};
