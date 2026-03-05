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
//! - `x4` — 4×f32 operations (NEON on AArch64, SIMD128 on WebAssembly)
//! - `x8` — 8×f32 operations (AVX2+FMA on x86-64)
//! - `x16` — 16×f32 operations (AVX-512 on x86-64)
//!
//! # Naming Convention
//!
//! Function suffixes match the required token type:
//!
//! - `_neon` — requires [`NeonToken`](archmage::NeonToken) (AArch64 NEON)
//! - `_wasm128` — requires [`Wasm128Token`](archmage::Wasm128Token) (WebAssembly SIMD128)
//! - `_v3` — requires [`X64V3Token`](archmage::X64V3Token) (x86-64-v3: AVX2+FMA)
//! - `_v4` — requires [`X64V4Token`](archmage::X64V4Token) (x86-64-v4: AVX-512)
//!
//! # Example
//!
//! ```rust,ignore
//! use archmage::arcane;
//! use linear_srgb::tokens::{X64V3Token, x8};
//!
//! #[arcane]
//! fn process_pixels(token: X64V3Token, data: &mut [f32]) {
//!     // This inlines — no dispatch boundary
//!     x8::srgb_to_linear_slice_v3(token, data);
//! }
//! ```

#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
pub mod x4;

#[cfg(target_arch = "x86_64")]
pub mod x8;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod x16;

// Re-export token types so users can `use linear_srgb::tokens::X64V3Token` etc.
#[cfg(target_arch = "aarch64")]
pub use archmage::NeonToken;
#[cfg(target_arch = "wasm32")]
pub use archmage::Wasm128Token;
#[cfg(target_arch = "x86_64")]
pub use archmage::X64V3Token;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use archmage::X64V4Token;
