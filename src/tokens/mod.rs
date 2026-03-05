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
//! - `x4` — 4×f32 operations (128-bit: SSE on x86-64, NEON on AArch64, SIMD128 on WebAssembly)
//! - `x8` — 8×f32 operations (256-bit: AVX2+FMA on x86-64)
//! - `x16` — 16×f32 operations (512-bit: AVX-512 on x86-64)
//!
//! # Naming Convention
//!
//! Function suffixes match the required token type:
//!
//! - `_v3` — requires [`X64V3Token`](archmage::X64V3Token) (x86-64-v3: AVX2+FMA)
//! - `_v4` — requires [`X64V4Token`](archmage::X64V4Token) (x86-64-v4: AVX-512)
//! - `_neon` — requires [`NeonToken`](archmage::NeonToken) (AArch64 NEON)
//! - `_wasm128` — requires [`Wasm128Token`](archmage::Wasm128Token) (WebAssembly SIMD128)
//!
//! On x86-64, `x4` and `x8` both use `_v3` suffix (both require AVX2+FMA). The
//! module name (`x4` vs `x8`) distinguishes the SIMD width.
//!
//! # Example
//!
//! ```rust,ignore
//! use archmage::arcane;
//! use linear_srgb::tokens::{X64V3Token, x4, x8};
//!
//! #[arcane]
//! fn process_pixel(token: X64V3Token, rgba: [f32; 4]) -> [f32; 4] {
//!     // x4 for single RGBA pixels
//!     x4::srgb_to_linear_v3(token, rgba)
//! }
//!
//! #[arcane]
//! fn process_batch(token: X64V3Token, data: &mut [f32]) {
//!     // x8 for maximum throughput on slices
//!     x8::srgb_to_linear_slice_v3(token, data);
//! }
//! ```

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
