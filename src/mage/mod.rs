//! Token-based SIMD API with zero dispatch overhead.
//!
//! This module provides an alternative API using archmage tokens for users who
//! want to avoid per-call dispatch overhead. Obtain a token once at startup,
//! then pass it to all conversion functions.
//!
//! # Usage
//!
//! ```rust,ignore
//! use linear_srgb::mage::{self, Token, SimdToken};
//!
//! fn main() {
//!     // Detection once at startup
//!     let token = Token::try_new().expect("SIMD not available");
//!
//!     let mut data = vec![0.5f32; 10000];
//!
//!     // Zero dispatch overhead - token proves features available
//!     mage::srgb_to_linear_slice(token, &mut data);
//! }
//! ```
//!
//! # Platform Support
//!
//! Currently supports x86-64 with AVX2+FMA. The API is designed for future
//! aarch64/NEON support once archmage adds transcendental functions for ARM.
//!
//! # Performance
//!
//! Uses archmage's native SIMD types with true FMA instructions, providing
//! ~1.7x speedup over the `simd` module's wide-based implementation.

// Platform-specific implementations
#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "x86_64")]
pub use x86_64::*;

// Re-export SimdToken trait for token creation
pub use archmage::SimdToken;

// Future: aarch64/NEON and wasm32/SIMD128 implementations will go here.
// Currently a no-op on non-x86_64 — the module compiles but exports nothing
// beyond the SimdToken trait re-export.
