//! Exact conversion functions using `powf()`.
//!
//! These functions use the mathematical sRGB transfer function with `powf()`,
//! providing maximum accuracy at the cost of speed (~6 ULP max vs f64 reference).
//! For faster alternatives, use [`crate::default`] which provides rational
//! polynomial approximations (~110 ULP max at threshold, <8 ULP elsewhere).
//!
//! # Constants: C0-continuous vs IEC textbook
//!
//! The IEC 61966-2-1 sRGB specification defines a piecewise transfer function
//! with a linear segment and a power curve. The textbook constants
//! (threshold 0.04045, offset 0.055) create a tiny discontinuity at the
//! segment boundary — the two pieces don't quite meet (~2.3×10⁻⁹ in f64).
//!
//! This module uses adjusted C0-continuous constants (from the moxcms reference
//! implementation) that eliminate the discontinuity:
//!
//! | Constant | IEC textbook | This module |
//! |----------|-------------|-------------|
//! | Gamma threshold | 0.04045 | 0.039293... |
//! | Linear threshold | 0.0031308 | 0.003041... |
//! | Offset (*a*) | 0.055 | 0.055011... |
//!
//! The adjusted constants solve `12.92 × threshold = (1+a) × threshold^(1/2.4) − a`
//! exactly, making the piecewise function mathematically continuous. The difference
//! is unmeasurable at u8 precision and within 1 LSB at u16.
//!
//! The [`crate::default`] module uses IEC textbook thresholds because its rational
//! polynomial was fitted to the IEC power curve — see the crate-level docs for
//! the full comparison.
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
