//! Inlineable `#[rite]` functions for embedding in your own `#[arcane]` code.
//!
//! These functions carry `#[target_feature]` + `#[inline]` directly — no wrapper,
//! no dispatch overhead. When called from a context with matching features (e.g.
//! your own `#[arcane]` entry point), LLVM inlines them fully.
//!
//! # Modules
//!
//! Organized by SIMD unit width:
//!
//! - [`x8`] — 8×f32 operations (AVX2+FMA on x86-64)
//!
//! # Naming Convention
//!
//! Function suffixes match the required token type:
//!
//! - `_v3` — requires [`Desktop64`] (x86-64-v3: AVX2+FMA)
//!
//! # Example
//!
//! ```rust,ignore
//! use archmage::{arcane, Desktop64};
//! use linear_srgb::rites::x8;
//!
//! #[arcane]
//! fn process_pixels(token: Desktop64, data: &mut [f32]) {
//!     // This inlines — no dispatch boundary
//!     x8::srgb_to_linear_slice_v3(token, data);
//! }
//! ```

#[cfg(target_arch = "x86_64")]
pub mod x8;
