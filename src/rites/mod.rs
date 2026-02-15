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
//! - `_neon` — requires [`Arm64`](archmage::Arm64) (AArch64 NEON)
//! - `_wasm128` — requires [`Wasm128Token`](archmage::Wasm128Token) (WebAssembly SIMD128)
//! - `_v3` — requires [`Desktop64`](archmage::Desktop64) (x86-64-v3: AVX2+FMA)
//! - `_v4` — requires [`Server64`](archmage::Server64) (x86-64-v4: AVX-512)
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

#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
pub mod x4;

#[cfg(target_arch = "x86_64")]
pub mod x8;

#[cfg(target_arch = "x86_64")]
pub mod x16;
