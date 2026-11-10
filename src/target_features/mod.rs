//! Conversions for callers already compiled with the required CPU features.
//!
//! **Start with [`crate::default`] for ordinary calls.** It selects CPU features
//! at runtime. This module is for composing conversions inside an existing SIMD
//! loop: arrays and slices cross the API boundary, with no archmage tokens or
//! magetypes vectors in the signatures.
//!
//! # Calling context
//!
//! On native targets, call these functions inside a function with matching
//! `#[target_feature]` attributes. A runtime feature check alone does not give
//! rustc that context. Calling from an ordinary function produces E0133 even
//! though the conversion is a safe function. Move the call into your feature
//! function instead of adding an `unsafe` block. An archmage `#[attune(v3)]` or
//! `#[arcane]` body supplies the same context; its token stays with the caller.
//!
//! WebAssembly validates the required SIMD instructions when loading the
//! module, so Rust also permits safe calls from ordinary WASM functions.
//! Functions ending in `_scalar` need no feature context on any target.
//!
//! ```
//! # #[cfg(target_arch = "x86_64")]
//! # {
//! use archmage::{arcane, SimdToken, X64V3Token};
//! use linear_srgb::target_features::x8;
//!
//! // The entry point checks the CPU once, outside the loop.
//! #[arcane]
//! fn convert_rows(_token: X64V3Token, rows: &mut [[f32; 8]]) {
//!     for row in rows {
//!         *row = x8::srgb_to_linear_v3(*row); // no token argument
//!     }
//! }
//!
//! if let Some(token) = X64V3Token::summon() {
//!     convert_rows(token, &mut [[0.5; 8]]);
//! }
//! # }
//! ```
//!
//! # Names and requirements
//!
//! Width modules describe array size, not CPU detection: `x4` and `x8`.
//! `x8` is available on x86-64, matching its deprecated token API. Suffixes
//! specify the required feature context. `x4` on x86 still requires V3,
//! despite operating on four values.
//!
//! | Suffix | Required `target_feature(enable = "…")` |
//! | --- | --- |
//! | `_v3` | `sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe` |
//! | `_neon` | `neon` (AArch64) |
//! | `_wasm128` | `simd128` (WebAssembly) |
//! | `_scalar` | None |
//!
//! The deprecated `tokens::x4` and `tokens::x8` functions forward here and retain the same
//! conversion, clamping, rounding and slice-length contracts. Transfer curves other than
//! the base sRGB/gamma operations require Cargo feature `transfer`. They do not
//! inspect the CPU or select a fallback. The internal SIMD dependencies may
//! change independently of the caller's copies; only the feature requirements
//! and ordinary Rust input/output types cross this boundary.

// Canonical feature entry points call private numerical kernels.

//! # Replacement scope
//!
//! Each public function replaces the same function in `tokens::x4` or
//! `tokens::x8`, omitting its token argument. Additional widths, tiers and
//! scalar fallbacks used by internal dispatch are crate-private.
//! `tokens::x16` is deprecated without a direct replacement. Ordinary callers
//! can use [`crate::default`] for dispatched slice conversions.
//!
//! The internal sixteen-value entry points are not a public API:
//!
//! ```compile_fail,E0603
//! use linear_srgb::target_features::x16;
//! ```
//!
//! Internal fallbacks also stay outside the public replacement surface:
//!
//! ```compile_fail,E0603
//! let _ = linear_srgb::target_features::x4::srgb_to_linear_scalar([0.5; 4]);
//! ```

macro_rules! entry {
    ($width:ident, scalar, $token:path; $($item:tt)*) => {
        entry!(@body $width, $token; #[inline] $($item)*);
    };
    ($width:ident, $tier:ident, $token:path; $($item:tt)*) => {
        entry!(@body $width, $token; #[archmage::attune($tier)] $($item)*);
    };
    (@body $width:ident, $token:path; $(#[$meta:meta])* fn $name:ident($($arg:ident: $ty:ty),*) $(-> $out:ty)?) => {
        #[doc = concat!("CPU-feature entry point corresponding to [`crate::tokens::", stringify!($width), "::", stringify!($name), "`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.")]
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        $(#[$meta])*
        pub fn $name($($arg: $ty),*) $(-> $out)? {
            crate::kernels::$width::$name(<$token>::from_context(), $($arg),*)
        }
    };
}

/// Four-value arrays, or slices processed four values at a time.
pub mod x4;

/// Eight-value arrays, or slices processed eight values at a time.
///
/// On x86, ordinary calls lack the required compilation context:
///
/// ```compile_fail,E0133
/// let _ = linear_srgb::target_features::x8::srgb_to_linear_v3([0.5; 8]);
/// ```
///
/// A matching feature context makes the same call safe:
///
/// ```
/// #![forbid(unsafe_code)]
/// #[archmage::attune(v3)]
/// fn convert(values: [f32; 8]) -> [f32; 8] {
///     linear_srgb::target_features::x8::srgb_to_linear_v3(values)
/// }
/// ```
#[cfg(target_arch = "x86_64")]
pub mod x8;
#[cfg(not(target_arch = "x86_64"))]
pub(crate) mod x8;

/// Internal wide dispatch and compatibility entry points.
pub(crate) mod x16;

#[cfg(test)]
mod compatibility_tests;
#[cfg(test)]
mod dispatch_tests;
#[cfg(test)]
mod fingerprint;
