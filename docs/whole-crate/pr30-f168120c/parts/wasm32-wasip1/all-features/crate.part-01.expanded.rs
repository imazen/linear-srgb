#![feature(prelude_import)]
//! Fast linear↔sRGB color space conversion.
//!
//! This crate provides efficient conversion between linear light values and
//! sRGB gamma-encoded values, with multiple implementation strategies for
//! different accuracy/performance tradeoffs.
//!
//! # Module Organization
//!
//! - [`default`] — **Start here.** Rational polynomial for f32, LUT for integers, SIMD for slices.
//! - [`precise`] — Platform `powf()` with C0-continuous constants. f32/f64, extended range. Slower.
//! - [`target_features`] — Array/slice conversions inside matching CPU-feature functions.
//! - [`tokens`] — Deprecated token-taking compatibility API.
//! - [`lut`] — Lookup tables for custom bit depths (10-bit, 12-bit, 16-bit).
//! - **`tf`** — Transfer functions beyond sRGB: BT.709, PQ, HLG. Requires `transfer` feature.
//! - **`iec`** — IEC 61966-2-1 textbook constants (legacy interop). Requires `iec` feature.
//!
//! # Quick Start
//!
//! ```rust
//! use linear_srgb::default::{srgb_to_linear, linear_to_srgb};
//!
//! // Convert sRGB 0.5 to linear
//! let linear = srgb_to_linear(0.5);
//! assert!((linear - 0.214).abs() < 0.001);
//!
//! // Convert back to sRGB
//! let srgb = linear_to_srgb(linear);
//! assert!((srgb - 0.5).abs() < 0.001);
//! ```
//!
//! # Batch Processing (SIMD)
//!
//! For maximum throughput on slices:
//!
//! ```rust
//! use linear_srgb::default::{srgb_to_linear_slice, linear_to_srgb_slice};
//!
//! let mut values = vec![0.5f32; 10000];
//! srgb_to_linear_slice(&mut values);  // SIMD-accelerated
//! linear_to_srgb_slice(&mut values);
//! ```
//!
//! # Custom Gamma
//!
//! For non-sRGB gamma (pure power function without linear segment):
//!
//! ```rust
//! use linear_srgb::default::{gamma_to_linear, linear_to_gamma};
//!
//! let linear = gamma_to_linear(0.5, 2.2);  // gamma 2.2
//! let encoded = linear_to_gamma(linear, 2.2);
//! ```
//!
//! # LUT-based Conversion
//!
//! For batch processing with pre-computed lookup tables:
//!
//! ```rust
//! use linear_srgb::default::SrgbConverter;
//!
//! let conv = SrgbConverter::new();  // Zero-cost, const tables
//!
//! // Fast 8-bit conversions
//! let linear = conv.srgb_u8_to_linear(128);
//! let srgb = conv.linear_to_srgb_u8(linear);
//! ```
//!
//! # Choosing the Right API
//!
//! | Use Case | Recommended Function |
//! |----------|---------------------|
//! | Single f32 value | [`default::srgb_to_linear`] |
//! | Single u8 value | [`default::srgb_u8_to_linear`] |
//! | f32 slice (in-place) | [`default::srgb_to_linear_slice`] |
//! | RGBA f32 slice (alpha-preserving) | [`default::srgb_to_linear_rgba_slice`] |
//! | u8 slice → f32 slice | [`default::srgb_u8_to_linear_slice`] |
//! | RGBA u8 → f32 (alpha-preserving) | [`default::srgb_u8_to_linear_rgba_slice`] |
//! | RGBA f32 sRGB → linear premul | [`default::srgb_to_linear_premultiply_rgba_slice`] |
//! | RGBA u8 sRGB → linear premul f32 | [`default::srgb_u8_to_linear_premultiply_rgba_slice`] |
//! | RGBA f32 linear premul → sRGB | [`default::unpremultiply_linear_to_srgb_rgba_slice`] |
//! | RGBA f32 linear premul → sRGB u8 | [`default::unpremultiply_linear_to_srgb_u8_rgba_slice`] |
//! | u16 → f32 slice | [`default::srgb_u16_to_linear_slice`] |
//! | f32 → u16 (exact RT) | [`default::linear_to_srgb_u16`] |
//! | f32 → u16 (fast, ±1 RT) | [`default::linear_to_srgb_u16_fast`] |
//! | Platform powf f32/f64 | [`precise::srgb_to_linear`] |
//! | Extended range (HDR) | [`precise::srgb_to_linear_extended`] |
//! | Extended range f32 slice (SIMD) | [`default::srgb_to_linear_extended_slice`] |
//! | Extended range RGBA f32 slice (alpha-preserving) | [`default::srgb_to_linear_extended_rgba_slice`] |
//! | Custom gamma f32 slice | [`default::gamma_to_linear_slice`] |
//! | Custom gamma RGBA f32 slice (alpha-preserving) | [`default::gamma_to_linear_rgba_slice`] |
//! | Inside `#[arcane]` | `target_features::x8::srgb_to_linear_v3` |
//! | Custom bit depth LUT | [`lut::LinearTable16`] |
//!
//! # Clamping and Extended Range
//!
//! The f32↔f32 conversion functions come in two flavors: **clamped** (default)
//! and **extended** (unclamped). Integer paths (u8, u16) always clamp since
//! out-of-range values can't be represented in the output format.
//!
//! ## Clamped (default) — use for same-gamut pipelines
//!
//! The ordinary sRGB and custom-gamma APIs clamp inputs to \[0, 1\]:
//! negatives become 0, values above 1 become 1. BT.709/PQ/HLG helpers have
//! curve-specific domains and do not share this general upper clamp.
//!
//! This is correct whenever the source and destination share the same color
//! space (gamut + transfer function). The typical pipeline:
//!
//! 1. Decode sRGB image (u8 → linear f32 via LUT, or f32 via TRC)
//! 2. Process in linear light (resize, blur, blend, composite)
//! 3. Re-encode to sRGB (linear f32 → sRGB f32 or u8)
//!
//! In this pipeline, out-of-range values only come from processing artifacts:
//! resize filters with negative lobes (Lanczos, Mitchell, etc.) produce small
//! negatives near dark edges and values slightly above 1.0 near bright edges.
//! These are ringing artifacts, not real colors — clamping is correct.
//!
//! Float decoders like jpegli can also produce small out-of-range values from
//! YCbCr quantization noise. When the image is sRGB, these are compression
//! artifacts and clamping is correct — gives the same result as decoding to
//! u8 first.
//!
//! ## Extended (unclamped) — use for cross-gamut pipelines
//!
//! [`precise::srgb_to_linear_extended`] and [`precise::linear_to_srgb_extended`]
//! do not clamp. They follow the mathematical sRGB transfer function for all
//! inputs: negatives pass through the linear segment, values above 1.0 pass
//! through the power segment.
//!
//! Use these when the sRGB transfer function is applied to values from a
//! **different, wider gamut**. A 3×3 matrix converting Rec. 2020 linear or
//! Display P3 linear to sRGB linear can produce values well outside \[0, 1\]:
//! a saturated Rec. 2020 green maps to deeply negative sRGB red and blue.
//! These are real out-of-gamut colors, not artifacts — clamping destroys
//! information that downstream gamut mapping or compositing may need.
//!
//! This matters in practice: JPEG and JPEG XL images can carry Rec. 2020 or
//! Display P3 ICC profiles. Phones shoot Rec. 2020 HLG, cameras embed
//! wide-gamut profiles. Decoding such an image and converting to sRGB for
//! display produces out-of-gamut values that should survive until final
//! output.
//!
//! If a float decoder (jpegli, libjxl) outputs wide-gamut data directly to
//! f32, the output contains both small compression artifacts and real
//! out-of-gamut values. The artifacts are tiny; the gamut excursions
//! dominate. Using `_extended` preserves both — the artifacts are harmless
//! noise that vanishes at quantization.
//!
//! The `_extended` variants also cover **scRGB** (float sRGB with values
//! outside \[0, 1\] for HDR and wide color) and any pipeline where
//! intermediate f32 values are not yet at the final output stage.
//!
//! ## Summary
//!
//! | Function | Range | Pipeline |
//! |----------|-------|----------|
//! | Ordinary sRGB/custom-gamma slices and integer LUT helpers | \[0, 1\] | Same-gamut batch processing |
//! | [`default::srgb_to_linear`] | \[0, 1\] | Same-gamut single values |
//! | [`default::linear_to_srgb`] | \[0, 1\] | Same-gamut single values |
//! | [`precise::srgb_to_linear_extended`] | Unbounded | Cross-gamut, scRGB, HDR |
//! | [`precise::linear_to_srgb_extended`] | Unbounded | Cross-gamut, scRGB, HDR |
//! | [`default::srgb_to_linear_extended_slice`] / [`default::srgb_to_linear_extended_rgba_slice`] | Unbounded (SIMD) | Cross-gamut, scRGB batch |
//! | [`default::linear_to_srgb_extended_slice`] / [`default::linear_to_srgb_extended_rgba_slice`] | Unbounded (SIMD) | Cross-gamut, scRGB batch |
//! | All u8/u16 paths | \[0, 1\] | Final quantization (clamp inherent) |
//!
//! The SIMD `_extended_slice` variants use a wider-domain rational
//! polynomial (fitted to \[0, 8\] / \[0, 64\] — see the per-function
//! accuracy tables) rather than the \[0, 1\] polynomial of the clamped
//! path, so they stay accurate on the out-of-gamut values a 3×3 gamut
//! matrix produces. The `_rgba_slice` forms leave alpha bit-identical.
//!
//! # Feature Flags
//!
//! - **`std`** (default) — Enable runtime SIMD dispatch. Required for slice functions.
//! - **`avx512`** (default) — Enable AVX-512 code paths and `tokens::x16` module.
//! - **`transfer`** — BT.709, PQ, and HLG transfer functions in `tf` and [`tokens`].
//! - **`iec`** — IEC 61966-2-1 textbook sRGB functions for legacy interop.
//! - **`alt`** — Alternative implementations for benchmarking (not stable API).
//! - **`unsafe_simd`** — No-op (kept for backward compatibility, will be removed in 0.7).
//!
//! # `no_std` Support
//!
//! This crate is `no_std` compatible. Without `std`, u16 functions use the
//! rational polynomial instead of LUT (slower but no heap allocation).
//! Disable the `std` feature:
//!
//! ```toml
//! linear-srgb = { version = "0.6", default-features = false }
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]
extern crate std;
#[prelude_import]
use std::prelude::rust_2024::*;
/// Recommended API with optimal implementations for each use case.
///
/// Uses a rational polynomial for single f32 values (≤14 ULP, perfectly
/// monotonic), LUT for integer types, and SIMD-dispatched batch processing
/// for slices.
pub mod default {
    //! Recommended API for sRGB ↔ linear conversion.
    //!
    //! This module re-exports the optimal implementation for each use case:
    //!
    //! - **Single f32 values**: Rational polynomial (~14 ULP max, perfectly
    //!   monotonic — no `powf`)
    //! - **Single u8/u16 values**: LUT lookup (zero math)
    //! - **Slices**: SIMD-accelerated with runtime CPU dispatch
    //! - **Custom gamma**: Pure power function (f32, slices)
    //!
    //! For platform `powf()` conversions with C0-continuous constants, see [`crate::precise`].
    //!
    //! # Quick Start
    //!
    //! ```rust
    //! use linear_srgb::default::{srgb_to_linear, linear_to_srgb};
    //!
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
    pub use crate::rational_poly::{
        linear_to_srgb_fast as linear_to_srgb, srgb_to_linear_fast as srgb_to_linear,
    };
    pub use crate::scalar::{
        linear_to_srgb_u8, linear_to_srgb_u16, linear_to_srgb_u16_fast,
        srgb_u8_to_linear, srgb_u16_to_linear,
    };
    pub use crate::simd::{
        gamma_to_linear_rgba_slice, gamma_to_linear_slice, linear_to_gamma_rgba_slice,
        linear_to_gamma_slice, linear_to_srgb_extended_rgba_slice,
        linear_to_srgb_extended_slice, linear_to_srgb_rgba_slice, linear_to_srgb_slice,
        linear_to_srgb_u8_rgba_slice, linear_to_srgb_u8_slice,
        linear_to_srgb_u16_rgba_slice, linear_to_srgb_u16_rgba_slice_fast,
        linear_to_srgb_u16_slice, linear_to_srgb_u16_slice_fast,
        srgb_to_linear_extended_rgba_slice, srgb_to_linear_extended_slice,
        srgb_to_linear_premultiply_rgba_slice, srgb_to_linear_rgba_slice,
        srgb_to_linear_slice, srgb_u8_to_linear_premultiply_rgba_slice,
        srgb_u8_to_linear_rgba_slice, srgb_u8_to_linear_slice,
        srgb_u16_to_linear_rgba_slice, srgb_u16_to_linear_slice,
        unpremultiply_linear_to_srgb_rgba_slice,
        unpremultiply_linear_to_srgb_u8_rgba_slice,
    };
    #[allow(deprecated)]
    pub use crate::simd::{
        gamma_to_linear_premultiply_rgba_slice, unpremultiply_linear_to_gamma_rgba_slice,
    };
    pub use crate::scalar::{gamma_to_linear, linear_to_gamma};
    pub use crate::tf::{
        bt709_to_linear, hlg_to_linear, linear_to_bt709, linear_to_hlg, linear_to_pq,
        pq_to_linear,
    };
    pub use crate::simd::{
        bt709_to_linear_rgba_slice, bt709_to_linear_slice, hlg_to_linear_rgba_slice,
        hlg_to_linear_slice, linear_to_bt709_rgba_slice, linear_to_bt709_slice,
        linear_to_hlg_rgba_slice, linear_to_hlg_slice, linear_to_pq_rgba_slice,
        linear_to_pq_slice, pq_to_linear_rgba_slice, pq_to_linear_slice,
    };
    pub use crate::lut::SrgbConverter;
}
/// Platform `powf()`-based conversions with C0-continuous constants.
///
/// Uses C0-continuous constants (from the moxcms reference implementation) that
/// eliminate the IEC 61966-2-1 piecewise discontinuity. ~6 ULP max error
/// vs f64 reference. See the module docs for the constant comparison table.
///
/// Also provides f64, extended-range (unclamped), and custom gamma functions.
/// For faster alternatives, use [`default`].
pub mod precise {
    //! Formula-based conversion functions using platform `powf()`.
    //!
    //! These functions use the mathematical sRGB transfer function with `powf()`,
    //! with platform libm rounding (sampled max 6 ULP vs the f64 reference).
    //! For faster alternatives, use [`crate::default`] which provides rational
    //! polynomial approximations (~14 ULP max, ~0.5 ULP avg, perfectly monotonic).
    //!
    //! # Constants: C0-continuous (moxcms)
    //!
    //! Both this module and [`crate::default`] use C0-continuous constants (from the
    //! moxcms reference implementation) that eliminate the ~2.3×10⁻⁹ discontinuity
    //! present in the IEC 61966-2-1 textbook constants:
    //!
    //! | Constant | IEC textbook | This crate |
    //! |----------|-------------|------------|
    //! | Gamma threshold | 0.04045 | 0.039293... |
    //! | Linear threshold | 0.0031308 | 0.003041... |
    //! | Offset (*a*) | 0.055 | 0.055011... |
    //!
    //! The adjusted constants solve `12.92 × threshold = (1+a) × threshold^(1/2.4) − a`
    //! exactly, making the piecewise function mathematically continuous. The difference
    //! is unmeasurable at u8 precision and within 1 LSB at u16.
    //!
    //! # Extended-range variants
    //!
    //! The `_extended` functions do not clamp to \[0, 1\], making them suitable
    //! for HDR, ICC, and cross-gamut pipelines. See the crate-level docs on
    //! clamping for details.
    pub use crate::scalar::{linear_to_srgb, srgb_to_linear};
    pub use crate::scalar::{linear_to_srgb_extended, srgb_to_linear_extended};
    pub use crate::scalar::{linear_to_srgb_f64, srgb_to_linear_f64};
    pub use crate::scalar::{gamma_to_linear_f64, linear_to_gamma_f64};
}
/// Lookup table types for sRGB conversion.
///
/// Provides both build-time const tables ([`SrgbConverter`](lut::SrgbConverter))
/// and runtime-generated tables for custom bit depths (10-bit, 12-bit, 16-bit).
pub mod lut {
    //! Lookup table (LUT) based sRGB conversions.
    //!
    //! ## For u8 and u16 data
    //!
    //! Use the functions in [`crate::default`] instead of this module — they're
    //! faster and easier:
    //! - u8: [`srgb_u8_to_linear`](crate::default::srgb_u8_to_linear), [`linear_to_srgb_u8`](crate::default::linear_to_srgb_u8)
    //! - u16: [`srgb_u16_to_linear`](crate::default::srgb_u16_to_linear), [`linear_to_srgb_u16`](crate::default::linear_to_srgb_u16)
    //!
    //! ## For non-standard bit depths (10-bit, 12-bit)
    //!
    //! This module provides generic `LinearizationTable<N>` and `EncodingTable<N>`
    //! for arbitrary bit depths, plus interpolation helpers. Use these when working
    //! with 10-bit HDR, 12-bit medical imaging, or other non-standard formats.
    //!
    //! ## `SrgbConverter`
    //!
    //! A zero-sized type wrapping const 8-bit tables. Largely superseded by the
    //! free functions in [`crate::default`], but kept for API compatibility.
    use crate::scalar::{linear_to_srgb_f64, srgb_to_linear_f64};
    #[allow(unused_imports)]
    use num_traits::Float;
    /// Pre-computed linearization table for sRGB to linear conversion.
    pub struct LinearizationTable<const N: usize> {
        table: Box<[f32; N]>,
    }
    impl<const N: usize> LinearizationTable<N> {
        /// Create a linearization table for the given bit depth.
        ///
        /// N must equal 2^BIT_DEPTH (e.g., N=256 for 8-bit, N=65536 for 16-bit).
        pub fn new() -> Self {
            let mut table = ::alloc::vec::from_elem(0.0f32, N).into_boxed_slice();
            let max_value = (N - 1) as f64;
            for (i, entry) in table.iter_mut().enumerate() {
                let srgb = i as f64 / max_value;
                *entry = srgb_to_linear_f64(srgb) as f32;
            }
            let table: Box<[f32; N]> = table.try_into().expect("size mismatch");
            Self { table }
        }
        /// Direct lookup (no interpolation) - for exact bit-depth matches.
        #[inline]
        pub fn lookup(&self, index: usize) -> f32 {
            self.table[index.min(N - 1)]
        }
        /// Get the raw table for custom interpolation.
        #[inline]
        pub fn as_slice(&self) -> &[f32] {
            &self.table[..]
        }
    }
    impl<const N: usize> Default for LinearizationTable<N> {
        fn default() -> Self {
            Self::new()
        }
    }
    /// Pre-computed encoding table for linear to sRGB conversion.
    pub struct EncodingTable<const N: usize> {
        table: Box<[f32; N]>,
    }
    impl<const N: usize> EncodingTable<N> {
        /// Create an encoding table for the given resolution.
        ///
        /// Higher N means finer granularity for interpolation.
        pub fn new() -> Self {
            let mut table = ::alloc::vec::from_elem(0.0f32, N).into_boxed_slice();
            let max_value = (N - 1) as f64;
            for (i, entry) in table.iter_mut().enumerate() {
                let linear = i as f64 / max_value;
                *entry = linear_to_srgb_f64(linear) as f32;
            }
            let table: Box<[f32; N]> = table.try_into().expect("size mismatch");
            Self { table }
        }
        /// Direct lookup (no interpolation).
        #[inline]
        pub fn lookup(&self, index: usize) -> f32 {
            self.table[index.min(N - 1)]
        }
        /// Get the raw table for custom interpolation.
        #[inline]
        pub fn as_slice(&self) -> &[f32] {
            &self.table[..]
        }
    }
    impl<const N: usize> Default for EncodingTable<N> {
        fn default() -> Self {
            Self::new()
        }
    }
    /// Linear interpolation in a float LUT.
    ///
    /// `x` should be in [0, 1]. Values outside are clamped.
    /// Uses FMA for the interpolation calculation.
    #[inline]
    pub fn lut_interp_linear_float(x: f32, table: &[f32]) -> f32 {
        let x = x.clamp(0.0, 1.0);
        let value = x * (table.len() - 1) as f32;
        let upper = value.ceil() as usize;
        let lower = value.floor() as usize;
        let tu = table[upper.min(table.len() - 1)];
        let tl = table[lower];
        (tu - tl).mul_add(value - lower as f32, tl)
    }
    /// Linear interpolation in a u16 LUT using fixed-point arithmetic.
    ///
    /// Avoids floating-point entirely for integer-only pipelines.
    #[inline]
    pub fn lut_interp_linear_u16(input_value: u16, table: &[u16]) -> u16 {
        let table_len = table.len() as u32;
        let mut value: u32 = input_value as u32 * (table_len - 1);
        let upper = value.div_ceil(65535) as usize;
        let lower = (value / 65535) as usize;
        let interp: u32 = value % 65535;
        let upper = upper.min(table.len() - 1);
        let lower = lower.min(table.len() - 1);
        value = (table[upper] as u32 * interp + table[lower] as u32 * (65535 - interp))
            / 65535;
        value as u16
    }
    /// Pre-computed 8-bit linearization table (256 entries).
    pub type LinearTable8 = LinearizationTable<256>;
    /// Pre-computed 10-bit linearization table (1024 entries).
    pub type LinearTable10 = LinearizationTable<1024>;
    /// Pre-computed 12-bit linearization table (4096 entries).
    pub type LinearTable12 = LinearizationTable<4096>;
    /// Pre-computed 16-bit linearization table (65536 entries).
    ///
    /// **Deprecated:** Use [`crate::default::srgb_u16_to_linear`] instead — it's
    /// faster (shared OnceLock LUT, SIMD-generated) and doesn't add 1+ seconds
    /// of compile time from const evaluation.
    #[deprecated(
        since = "0.7.0",
        note = "use default::srgb_u16_to_linear instead (faster, shared LUT)"
    )]
    pub type LinearTable16 = LinearizationTable<65536>;
    /// Pre-computed 8-bit encoding table (256 entries).
    pub type EncodeTable8 = EncodingTable<256>;
    /// Pre-computed 12-bit encoding table (4096 entries).
    pub type EncodeTable12 = EncodingTable<4096>;
    /// Pre-computed 16-bit encoding table (65536 entries).
    ///
    /// **Deprecated:** Use [`crate::default::linear_to_srgb_u16`] (exact) or
    /// [`crate::default::linear_to_srgb_u16_fast`] (10×, ±1 max error) instead.
    #[deprecated(
        since = "0.7.0",
        note = "use default::linear_to_srgb_u16 or linear_to_srgb_u16_fast instead"
    )]
    pub type EncodeTable16 = EncodingTable<65536>;
    /// Converter using pre-computed const LUTs for fast batch conversion.
    ///
    /// This is a **zero-sized type** - the tables are embedded in the binary at
    /// compile time with no runtime initialization cost. This is the recommended
    /// approach for standard 8-bit sRGB workflows.
    ///
    /// # Example
    ///
    /// ```rust
    /// use linear_srgb::lut::SrgbConverter;
    ///
    /// let conv = SrgbConverter::new();  // No allocation, just a marker type
    ///
    /// // Fast 8-bit lookups
    /// let linear = conv.srgb_u8_to_linear(128);
    /// let srgb = conv.linear_to_srgb_u8(linear);
    /// ```
    ///
    /// # Implementation Details
    ///
    /// - Uses a 256-entry const table for sRGB u8 → linear f32 (direct lookup)
    /// - Uses a 4096-entry const table for linear f32 → sRGB f32 (linear interpolation)
    pub struct SrgbConverter;
    impl SrgbConverter {
        /// Create a new converter.
        ///
        /// This is a no-op since tables are const and embedded in binary.
        #[inline]
        pub const fn new() -> Self {
            Self
        }
        /// Convert 8-bit sRGB to linear using direct table lookup.
        #[inline]
        pub fn srgb_u8_to_linear(&self, value: u8) -> f32 {
            crate::const_luts::linear_table_8()[value as usize]
        }
        /// Convert linear to sRGB using table interpolation.
        #[inline]
        pub fn linear_to_srgb(&self, linear: f32) -> f32 {
            lut_interp_linear_float(linear, crate::const_luts::encode_table_12())
        }
        /// Convert linear to 8-bit sRGB using direct LUT lookup.
        ///
        /// Uses a 4096-entry const u8 table — no interpolation or transcendental math.
        #[inline]
        pub fn linear_to_srgb_u8(&self, linear: f32) -> u8 {
            crate::scalar::linear_to_srgb_u8(linear)
        }
        /// Batch convert sRGB u8 values to linear f32.
        #[inline]
        pub fn batch_srgb_to_linear(&self, input: &[u8], output: &mut [f32]) {
            {
                match (&input.len(), &output.len()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                }
            };
            for (i, o) in input.iter().zip(output.iter_mut()) {
                *o = self.srgb_u8_to_linear(*i);
            }
        }
        /// Batch convert linear f32 values to sRGB u8.
        ///
        /// Uses [`crate::default::linear_to_srgb_u8_slice`] for SIMD-accelerated
        /// index computation with LUT lookup.
        #[inline]
        pub fn batch_linear_to_srgb(&self, input: &[f32], output: &mut [u8]) {
            crate::simd::linear_to_srgb_u8_slice(input, output);
        }
    }
    impl Default for SrgbConverter {
        fn default() -> Self {
            Self::new()
        }
    }
}
pub mod target_features {
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
    /// Four-value arrays, or slices processed four values at a time.
    pub mod x4 {
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::srgb_to_linear_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_to_linear_wasm128(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::srgb_to_linear_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::srgb_to_linear_slice_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_to_linear_slice_wasm128(values: &mut [f32]) {
            crate::kernels::x4::srgb_to_linear_slice_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_srgb_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_wasm128(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::linear_to_srgb_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_srgb_slice_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_slice_wasm128(values: &mut [f32]) {
            crate::kernels::x4::linear_to_srgb_slice_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::srgb_to_linear_extended_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_to_linear_extended_wasm128(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::srgb_to_linear_extended_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_srgb_extended_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_extended_wasm128(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::linear_to_srgb_extended_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::gamma_to_linear_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn gamma_to_linear_wasm128(values: [f32; 4], gamma: f32) -> [f32; 4] {
            crate::kernels::x4::gamma_to_linear_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::gamma_to_linear_slice_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn gamma_to_linear_slice_wasm128(values: &mut [f32], gamma: f32) {
            crate::kernels::x4::gamma_to_linear_slice_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_gamma_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_gamma_wasm128(values: [f32; 4], gamma: f32) -> [f32; 4] {
            crate::kernels::x4::linear_to_gamma_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_gamma_slice_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_gamma_slice_wasm128(values: &mut [f32], gamma: f32) {
            crate::kernels::x4::linear_to_gamma_slice_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::tf_srgb_to_linear_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_srgb_to_linear_wasm128(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::tf_srgb_to_linear_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::tf_srgb_to_linear_slice_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_srgb_to_linear_slice_wasm128(values: &mut [f32]) {
            crate::kernels::x4::tf_srgb_to_linear_slice_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::tf_linear_to_srgb_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_linear_to_srgb_wasm128(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::tf_linear_to_srgb_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::tf_linear_to_srgb_slice_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_linear_to_srgb_slice_wasm128(values: &mut [f32]) {
            crate::kernels::x4::tf_linear_to_srgb_slice_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::bt709_to_linear_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn bt709_to_linear_wasm128(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::bt709_to_linear_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::bt709_to_linear_slice_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn bt709_to_linear_slice_wasm128(values: &mut [f32]) {
            crate::kernels::x4::bt709_to_linear_slice_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_bt709_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_bt709_wasm128(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::linear_to_bt709_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_bt709_slice_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_bt709_slice_wasm128(values: &mut [f32]) {
            crate::kernels::x4::linear_to_bt709_slice_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::pq_to_linear_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn pq_to_linear_wasm128(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::pq_to_linear_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::pq_to_linear_slice_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn pq_to_linear_slice_wasm128(values: &mut [f32]) {
            crate::kernels::x4::pq_to_linear_slice_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_pq_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_pq_wasm128(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::linear_to_pq_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_pq_slice_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_pq_slice_wasm128(values: &mut [f32]) {
            crate::kernels::x4::linear_to_pq_slice_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::hlg_to_linear_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn hlg_to_linear_wasm128(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::hlg_to_linear_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::hlg_to_linear_slice_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn hlg_to_linear_slice_wasm128(values: &mut [f32]) {
            crate::kernels::x4::hlg_to_linear_slice_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_hlg_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_hlg_wasm128(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::linear_to_hlg_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_hlg_slice_wasm128`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_hlg_slice_wasm128(values: &mut [f32]) {
            crate::kernels::x4::linear_to_hlg_slice_wasm128(
                <archmage::Wasm128Token>::from_context(),
                values,
            )
        }
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::srgb_to_linear_extended_scalar`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        #[inline]
        pub fn srgb_to_linear_extended_scalar(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::srgb_to_linear_extended_scalar(
                <archmage::ScalarToken>::from_context(),
                values,
            )
        }
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_srgb_extended_scalar`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        #[inline]
        pub fn linear_to_srgb_extended_scalar(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::linear_to_srgb_extended_scalar(
                <archmage::ScalarToken>::from_context(),
                values,
            )
        }
        /// Convert 4 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn srgb_to_linear_scalar(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::float::x4::srgb_to_linear(archmage::ScalarToken, values)
        }
        /// Convert 4 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn linear_to_srgb_scalar(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::float::x4::linear_to_srgb(archmage::ScalarToken, values)
        }
        /// Convert 4 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn gamma_to_linear_scalar(values: [f32; 4], gamma: f32) -> [f32; 4] {
            crate::kernels::float::x4::gamma_to_linear(
                archmage::ScalarToken,
                values,
                gamma,
            )
        }
        /// Convert 4 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn linear_to_gamma_scalar(values: [f32; 4], gamma: f32) -> [f32; 4] {
            crate::kernels::float::x4::linear_to_gamma(
                archmage::ScalarToken,
                values,
                gamma,
            )
        }
        /// Convert 4 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn bt709_to_linear_scalar(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::scalar_array(
                values,
                |chunk| {
                    let token = archmage::ScalarToken;
                    crate::tf::bt709::bt709_to_linear_x4(
                            token,
                            magetypes::simd::generic::f32x4::from_array_t(token, chunk),
                        )
                        .to_array()
                },
                crate::lanes::bt709_to_linear,
            )
        }
        /// Convert 4 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn linear_to_bt709_scalar(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::scalar_array(
                values,
                |chunk| {
                    let token = archmage::ScalarToken;
                    crate::tf::bt709::linear_to_bt709_x4(
                            token,
                            magetypes::simd::generic::f32x4::from_array_t(token, chunk),
                        )
                        .to_array()
                },
                crate::lanes::linear_to_bt709,
            )
        }
        /// Convert 4 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn pq_to_linear_scalar(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::scalar_array(
                values,
                |chunk| {
                    let token = archmage::ScalarToken;
                    crate::tf::pq::pq_to_linear_x4(
                            token,
                            magetypes::simd::generic::f32x4::from_array_t(token, chunk),
                        )
                        .to_array()
                },
                crate::lanes::pq_to_linear,
            )
        }
        /// Convert 4 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn linear_to_pq_scalar(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::scalar_array(
                values,
                |chunk| {
                    let token = archmage::ScalarToken;
                    crate::tf::pq::linear_to_pq_x4(
                            token,
                            magetypes::simd::generic::f32x4::from_array_t(token, chunk),
                        )
                        .to_array()
                },
                crate::lanes::linear_to_pq,
            )
        }
        /// Convert 4 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn hlg_to_linear_scalar(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::scalar_array(
                values,
                |chunk| {
                    let token = archmage::ScalarToken;
                    crate::tf::hlg::hlg_to_linear_x4(
                            token,
                            magetypes::simd::generic::f32x4::from_array_t(token, chunk),
                        )
                        .to_array()
                },
                crate::lanes::hlg_to_linear,
            )
        }
        /// Convert 4 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn linear_to_hlg_scalar(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::scalar_array(
                values,
                |chunk| {
                    let token = archmage::ScalarToken;
                    crate::tf::hlg::linear_to_hlg_x4(
                            token,
                            magetypes::simd::generic::f32x4::from_array_t(token, chunk),
                        )
                        .to_array()
                },
                crate::lanes::linear_to_hlg,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn srgb_to_linear_extended_slice_wasm128(values: &mut [f32]) {
            crate::kernels::map_slice::<
                4,
            >(values, |chunk| srgb_to_linear_extended_wasm128(chunk));
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_srgb_extended_slice_wasm128(values: &mut [f32]) {
            crate::kernels::map_slice::<
                4,
            >(values, |chunk| linear_to_srgb_extended_wasm128(chunk));
        }
    }
    pub(crate) mod x8 {}
    /// Internal wide dispatch and compatibility entry points.
    pub(crate) mod x16 {
        /// Convert 16 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn srgb_to_linear_scalar(values: [f32; 16]) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::srgb_to_linear_scalar(*src);
                dst.copy_from_slice(&converted);
            }
            output
        }
        /// Convert all values, including a final partial array.
        #[inline]
        pub(crate) fn srgb_to_linear_slice_scalar(values: &mut [f32]) {
            crate::kernels::scalar_slice::<
                16,
            >(values, srgb_to_linear_scalar, crate::lanes::srgb_to_linear);
        }
        /// Convert 16 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn linear_to_srgb_scalar(values: [f32; 16]) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::linear_to_srgb_scalar(*src);
                dst.copy_from_slice(&converted);
            }
            output
        }
        /// Convert all values, including a final partial array.
        #[inline]
        pub(crate) fn linear_to_srgb_slice_scalar(values: &mut [f32]) {
            crate::kernels::scalar_slice::<
                16,
            >(values, linear_to_srgb_scalar, crate::lanes::linear_to_srgb);
        }
        /// Convert 16 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn srgb_to_linear_extended_scalar(values: [f32; 16]) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::srgb_to_linear_extended_scalar(*src);
                dst.copy_from_slice(&converted);
            }
            output
        }
        /// Convert all values, including a final partial array.
        #[inline]
        pub(crate) fn srgb_to_linear_extended_slice_scalar(values: &mut [f32]) {
            crate::kernels::scalar_slice::<
                16,
            >(
                values,
                srgb_to_linear_extended_scalar,
                crate::lanes::srgb_to_linear_extended,
            );
        }
        /// Convert 16 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn linear_to_srgb_extended_scalar(values: [f32; 16]) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::linear_to_srgb_extended_scalar(*src);
                dst.copy_from_slice(&converted);
            }
            output
        }
        /// Convert all values, including a final partial array.
        #[inline]
        pub(crate) fn linear_to_srgb_extended_slice_scalar(values: &mut [f32]) {
            crate::kernels::scalar_slice::<
                16,
            >(
                values,
                linear_to_srgb_extended_scalar,
                crate::lanes::linear_to_srgb_extended,
            );
        }
        /// Convert 16 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn gamma_to_linear_scalar(
            values: [f32; 16],
            gamma: f32,
        ) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::gamma_to_linear_scalar(*src, gamma);
                dst.copy_from_slice(&converted);
            }
            output
        }
        /// Convert all values, including a final partial array.
        #[inline]
        pub(crate) fn gamma_to_linear_slice_scalar(values: &mut [f32], gamma: f32) {
            crate::kernels::scalar_slice::<
                16,
            >(
                values,
                |chunk| gamma_to_linear_scalar(chunk, gamma),
                |value| crate::lanes::gamma_to_linear(value, gamma),
            );
        }
        /// Convert 16 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn linear_to_gamma_scalar(
            values: [f32; 16],
            gamma: f32,
        ) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::linear_to_gamma_scalar(*src, gamma);
                dst.copy_from_slice(&converted);
            }
            output
        }
        /// Convert all values, including a final partial array.
        #[inline]
        pub(crate) fn linear_to_gamma_slice_scalar(values: &mut [f32], gamma: f32) {
            crate::kernels::scalar_slice::<
                16,
            >(
                values,
                |chunk| linear_to_gamma_scalar(chunk, gamma),
                |value| crate::lanes::linear_to_gamma(value, gamma),
            );
        }
        /// Convert 16 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn bt709_to_linear_scalar(values: [f32; 16]) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::bt709_to_linear_scalar(*src);
                dst.copy_from_slice(&converted);
            }
            output
        }
        /// Convert all values, including a final partial array.
        #[inline]
        pub(crate) fn bt709_to_linear_slice_scalar(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, bt709_to_linear_scalar);
        }
        /// Convert 16 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn linear_to_bt709_scalar(values: [f32; 16]) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::linear_to_bt709_scalar(*src);
                dst.copy_from_slice(&converted);
            }
            output
        }
        /// Convert all values, including a final partial array.
        #[inline]
        pub(crate) fn linear_to_bt709_slice_scalar(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, linear_to_bt709_scalar);
        }
        /// Convert 16 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn pq_to_linear_scalar(values: [f32; 16]) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::pq_to_linear_scalar(*src);
                dst.copy_from_slice(&converted);
            }
            output
        }
        /// Convert all values, including a final partial array.
        #[inline]
        pub(crate) fn pq_to_linear_slice_scalar(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, pq_to_linear_scalar);
        }
        /// Convert 16 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn linear_to_pq_scalar(values: [f32; 16]) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::linear_to_pq_scalar(*src);
                dst.copy_from_slice(&converted);
            }
            output
        }
        /// Convert all values, including a final partial array.
        #[inline]
        pub(crate) fn linear_to_pq_slice_scalar(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, linear_to_pq_scalar);
        }
        /// Convert 16 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn hlg_to_linear_scalar(values: [f32; 16]) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::hlg_to_linear_scalar(*src);
                dst.copy_from_slice(&converted);
            }
            output
        }
        /// Convert all values, including a final partial array.
        #[inline]
        pub(crate) fn hlg_to_linear_slice_scalar(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, hlg_to_linear_scalar);
        }
        /// Convert 16 values; see the module calling-context requirements.
        #[inline]
        pub(crate) fn linear_to_hlg_scalar(values: [f32; 16]) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::linear_to_hlg_scalar(*src);
                dst.copy_from_slice(&converted);
            }
            output
        }
        /// Convert all values, including a final partial array.
        #[inline]
        pub(crate) fn linear_to_hlg_slice_scalar(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, linear_to_hlg_scalar);
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn srgb_to_linear_wasm128(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::float::x16::srgb_to_linear(
                archmage::Wasm128Token::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn srgb_to_linear_slice_wasm128(values: &mut [f32]) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| srgb_to_linear_wasm128(chunk));
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn linear_to_srgb_wasm128(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::float::x16::linear_to_srgb(
                archmage::Wasm128Token::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_srgb_slice_wasm128(values: &mut [f32]) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| linear_to_srgb_wasm128(chunk));
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn srgb_to_linear_extended_wasm128(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::float::x16::srgb_to_linear_extended(
                archmage::Wasm128Token::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn srgb_to_linear_extended_slice_wasm128(values: &mut [f32]) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| srgb_to_linear_extended_wasm128(chunk));
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn linear_to_srgb_extended_wasm128(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::float::x16::linear_to_srgb_extended(
                archmage::Wasm128Token::from_context(),
                values,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_srgb_extended_slice_wasm128(values: &mut [f32]) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| linear_to_srgb_extended_wasm128(chunk));
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn gamma_to_linear_wasm128(
            values: [f32; 16],
            gamma: f32,
        ) -> [f32; 16] {
            crate::kernels::float::x16::gamma_to_linear(
                archmage::Wasm128Token::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn gamma_to_linear_slice_wasm128(values: &mut [f32], gamma: f32) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| gamma_to_linear_wasm128(chunk, gamma));
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn linear_to_gamma_wasm128(
            values: [f32; 16],
            gamma: f32,
        ) -> [f32; 16] {
            crate::kernels::float::x16::linear_to_gamma(
                archmage::Wasm128Token::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_gamma_slice_wasm128(values: &mut [f32], gamma: f32) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| linear_to_gamma_wasm128(chunk, gamma));
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn tf_srgb_to_linear_wasm128(values: [f32; 16]) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::tf_srgb_to_linear_wasm128(*src);
                dst.copy_from_slice(&converted);
            }
            output
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn tf_srgb_to_linear_slice_wasm128(values: &mut [f32]) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| tf_srgb_to_linear_wasm128(chunk));
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn tf_linear_to_srgb_wasm128(values: [f32; 16]) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::tf_linear_to_srgb_wasm128(*src);
                dst.copy_from_slice(&converted);
            }
            output
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn tf_linear_to_srgb_slice_wasm128(values: &mut [f32]) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| tf_linear_to_srgb_wasm128(chunk));
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn bt709_to_linear_wasm128(values: [f32; 16]) -> [f32; 16] {
            let token = archmage::Wasm128Token::from_context();
            crate::tf::bt709::bt709_to_linear_x16(
                    token,
                    magetypes::simd::generic::f32x16::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn bt709_to_linear_slice_wasm128(values: &mut [f32]) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| bt709_to_linear_wasm128(chunk));
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn linear_to_bt709_wasm128(values: [f32; 16]) -> [f32; 16] {
            let token = archmage::Wasm128Token::from_context();
            crate::tf::bt709::linear_to_bt709_x16(
                    token,
                    magetypes::simd::generic::f32x16::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_bt709_slice_wasm128(values: &mut [f32]) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| linear_to_bt709_wasm128(chunk));
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn pq_to_linear_wasm128(values: [f32; 16]) -> [f32; 16] {
            let token = archmage::Wasm128Token::from_context();
            crate::tf::pq::pq_to_linear_x16(
                    token,
                    magetypes::simd::generic::f32x16::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn pq_to_linear_slice_wasm128(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, |chunk| pq_to_linear_wasm128(chunk));
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn linear_to_pq_wasm128(values: [f32; 16]) -> [f32; 16] {
            let token = archmage::Wasm128Token::from_context();
            crate::tf::pq::linear_to_pq_x16(
                    token,
                    magetypes::simd::generic::f32x16::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_pq_slice_wasm128(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, |chunk| linear_to_pq_wasm128(chunk));
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn hlg_to_linear_wasm128(values: [f32; 16]) -> [f32; 16] {
            let token = archmage::Wasm128Token::from_context();
            crate::tf::hlg::hlg_to_linear_x16(
                    token,
                    magetypes::simd::generic::f32x16::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn hlg_to_linear_slice_wasm128(values: &mut [f32]) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| hlg_to_linear_wasm128(chunk));
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn linear_to_hlg_wasm128(values: [f32; 16]) -> [f32; 16] {
            let token = archmage::Wasm128Token::from_context();
            crate::tf::hlg::linear_to_hlg_x16(
                    token,
                    magetypes::simd::generic::f32x16::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_hlg_slice_wasm128(values: &mut [f32]) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| linear_to_hlg_wasm128(chunk));
        }
    }
}
/// Deprecated token-taking compatibility API. Use [`crate::target_features`]
/// for x4/x8 inside a matching CPU-feature function and omit the token argument.
/// The x16 compatibility API is deprecated without a direct replacement.
#[deprecated(
    note = "use target_features::{x4, x8} inside a matching CPU-feature function; x16 is deprecated without a direct replacement"
)]
#[allow(deprecated)]
pub mod tokens {
    //! Deprecated: `x4` and `x8` callers use [`crate::target_features`] with the
    //! same function name and arguments except for the token. The `x16` API is
    //! deprecated without a direct replacement. See the calling-context example.
    //!
    //! Compatibility functions retain their original calling-context contracts.
    //! New feature-context code uses the `#[attune]` entry points in
    //! [`crate::target_features`]. Scalar replacements are ordinary functions.
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
    pub mod x4 {
        //! Compatibility adapters for the deprecated 4-lane token API.
        //! Numerical implementation lives behind `crate::target_features::x4`.
        use archmage::attune;
        pub use archmage::Wasm128Token;
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 4 sRGB values to linear without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted to \[0, 8\]. sampled max 8 ULP in \[0,1\], u16-safe to ~4.2.
        fn __attune_srgb_to_linear_extended_wasm128(
            _token: ::archmage::Wasm128Token,
            srgb: [f32; 4],
        ) -> [f32; 4] {
            crate::target_features::x4::srgb_to_linear_extended_wasm128(srgb)
        }
        /// Convert 4 sRGB values to linear without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted to \[0, 8\]. sampled max 8 ULP in \[0,1\], u16-safe to ~4.2.
        #[inline(always)]
        #[allow(clippy::inline_always)]
        pub fn srgb_to_linear_extended_wasm128(
            __archmage_token: ::archmage::Wasm128Token,
            srgb: [f32; 4],
        ) -> [f32; 4] {
            __attune_srgb_to_linear_extended_wasm128(__archmage_token, srgb)
        }
        #[inline]
        /// Convert 4 sRGB values to linear without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted to \[0, 8\]. sampled max 8 ULP in \[0,1\], u16-safe to ~4.2.
        fn __attune_srgb_to_linear_extended_scalar(
            _token: ::archmage::ScalarToken,
            srgb: [f32; 4],
        ) -> [f32; 4] {
            crate::target_features::x4::srgb_to_linear_extended_scalar(srgb)
        }
        /// Convert 4 sRGB values to linear without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted to \[0, 8\]. sampled max 8 ULP in \[0,1\], u16-safe to ~4.2.
        #[inline(always)]
        #[allow(clippy::inline_always)]
        pub fn srgb_to_linear_extended_scalar(
            __archmage_token: ::archmage::ScalarToken,
            srgb: [f32; 4],
        ) -> [f32; 4] {
            __attune_srgb_to_linear_extended_scalar(__archmage_token, srgb)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 4 linear values to sRGB without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted on √x to \[0, 64\]. sampled max 7 ULP in \[0,1\], u16-safe to 64.
        fn __attune_linear_to_srgb_extended_wasm128(
            _token: ::archmage::Wasm128Token,
            linear: [f32; 4],
        ) -> [f32; 4] {
            crate::target_features::x4::linear_to_srgb_extended_wasm128(linear)
        }
        /// Convert 4 linear values to sRGB without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted on √x to \[0, 64\]. sampled max 7 ULP in \[0,1\], u16-safe to 64.
        #[inline(always)]
        #[allow(clippy::inline_always)]
        pub fn linear_to_srgb_extended_wasm128(
            __archmage_token: ::archmage::Wasm128Token,
            linear: [f32; 4],
        ) -> [f32; 4] {
            __attune_linear_to_srgb_extended_wasm128(__archmage_token, linear)
        }
        #[inline]
        /// Convert 4 linear values to sRGB without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted on √x to \[0, 64\]. sampled max 7 ULP in \[0,1\], u16-safe to 64.
        fn __attune_linear_to_srgb_extended_scalar(
            _token: ::archmage::ScalarToken,
            linear: [f32; 4],
        ) -> [f32; 4] {
            crate::target_features::x4::linear_to_srgb_extended_scalar(linear)
        }
        /// Convert 4 linear values to sRGB without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted on √x to \[0, 64\]. sampled max 7 ULP in \[0,1\], u16-safe to 64.
        #[inline(always)]
        #[allow(clippy::inline_always)]
        pub fn linear_to_srgb_extended_scalar(
            __archmage_token: ::archmage::ScalarToken,
            linear: [f32; 4],
        ) -> [f32; 4] {
            __attune_linear_to_srgb_extended_scalar(__archmage_token, linear)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 4 sRGB values to linear. Input clamped to \[0, 1\].
        ///
        /// The `Wasm128Token` parameter proves SIMD128 support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub fn srgb_to_linear_wasm128(_token: Wasm128Token, srgb: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::srgb_to_linear_wasm128(srgb)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_srgb_wasm128(
            _token: Wasm128Token,
            linear: [f32; 4],
        ) -> [f32; 4] {
            crate::target_features::x4::linear_to_srgb_wasm128(linear)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn gamma_to_linear_wasm128(
            _token: Wasm128Token,
            encoded: [f32; 4],
            gamma: f32,
        ) -> [f32; 4] {
            crate::target_features::x4::gamma_to_linear_wasm128(encoded, gamma)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_gamma_wasm128(
            _token: Wasm128Token,
            linear: [f32; 4],
            gamma: f32,
        ) -> [f32; 4] {
            crate::target_features::x4::linear_to_gamma_wasm128(linear, gamma)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert sRGB f32 values to linear in-place using 4-wide WASM SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn srgb_to_linear_slice_wasm128(_token: Wasm128Token, values: &mut [f32]) {
            crate::target_features::x4::srgb_to_linear_slice_wasm128(values)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert linear f32 values to sRGB in-place using 4-wide WASM SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_srgb_slice_wasm128(_token: Wasm128Token, values: &mut [f32]) {
            crate::target_features::x4::linear_to_srgb_slice_wasm128(values)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert gamma-encoded f32 values to linear in-place using 4-wide WASM SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn gamma_to_linear_slice_wasm128(
            _token: Wasm128Token,
            values: &mut [f32],
            gamma: f32,
        ) {
            crate::target_features::x4::gamma_to_linear_slice_wasm128(values, gamma)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert linear f32 values to gamma-encoded in-place using 4-wide WASM SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_gamma_slice_wasm128(
            _token: Wasm128Token,
            values: &mut [f32],
            gamma: f32,
        ) {
            crate::target_features::x4::linear_to_gamma_slice_wasm128(values, gamma)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn tf_srgb_to_linear_wasm128(_token: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::tf_srgb_to_linear_wasm128(v)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn tf_linear_to_srgb_wasm128(_token: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::tf_linear_to_srgb_wasm128(v)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn bt709_to_linear_wasm128(_token: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::bt709_to_linear_wasm128(v)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn linear_to_bt709_wasm128(_token: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::linear_to_bt709_wasm128(v)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn pq_to_linear_wasm128(_token: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::pq_to_linear_wasm128(v)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn linear_to_pq_wasm128(_token: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::linear_to_pq_wasm128(v)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn hlg_to_linear_wasm128(_token: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::hlg_to_linear_wasm128(v)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn linear_to_hlg_wasm128(_token: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::linear_to_hlg_wasm128(v)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn tf_srgb_to_linear_slice_wasm128(
            _token: Wasm128Token,
            values: &mut [f32],
        ) {
            crate::target_features::x4::tf_srgb_to_linear_slice_wasm128(values)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn tf_linear_to_srgb_slice_wasm128(
            _token: Wasm128Token,
            values: &mut [f32],
        ) {
            crate::target_features::x4::tf_linear_to_srgb_slice_wasm128(values)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn bt709_to_linear_slice_wasm128(_token: Wasm128Token, values: &mut [f32]) {
            crate::target_features::x4::bt709_to_linear_slice_wasm128(values)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn linear_to_bt709_slice_wasm128(_token: Wasm128Token, values: &mut [f32]) {
            crate::target_features::x4::linear_to_bt709_slice_wasm128(values)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn pq_to_linear_slice_wasm128(_token: Wasm128Token, values: &mut [f32]) {
            crate::target_features::x4::pq_to_linear_slice_wasm128(values)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn linear_to_pq_slice_wasm128(_token: Wasm128Token, values: &mut [f32]) {
            crate::target_features::x4::linear_to_pq_slice_wasm128(values)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn hlg_to_linear_slice_wasm128(_token: Wasm128Token, values: &mut [f32]) {
            crate::target_features::x4::hlg_to_linear_slice_wasm128(values)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub fn linear_to_hlg_slice_wasm128(_token: Wasm128Token, values: &mut [f32]) {
            crate::target_features::x4::linear_to_hlg_slice_wasm128(values)
        }
    }
    pub use archmage::Wasm128Token;
}
/// Transfer functions: sRGB, BT.709, PQ (ST 2084), HLG (ARIB STD-B67).
///
/// **Prefer [`default`]** — every scalar in this module is already
/// re-exported there with the same name (gated on `transfer` for the HDR
/// curves). `tf` exists primarily for internal organization of the
/// per-curve implementations consumed by [`tokens`]; it is `pub` for
/// backward compatibility with 0.6.x callers that imported through this
/// path. New code should use `linear_srgb::default::*`.
///
/// BT.709 / PQ / HLG scalars require the `transfer` feature. sRGB-only
/// helpers compile unconditionally so the extended-range slice functions
/// (not gated on `transfer`) can link.
pub mod tf {
    #![allow(dead_code)]
    #![allow(clippy::excessive_precision)]
    //! Transfer function implementations for sRGB, BT.709, PQ, and HLG.
    //!
    //! This module provides scalar functions for all four transfer curves.
    //! SIMD implementations are private. Feature-context callers use the x4/x8
    //! array/slice replacements in [`crate::target_features`].
    //!
    //! # Scalar functions
    //!
    //! | Function | Transfer | Method |
    //! |---|---|---|
    //! | `srgb_to_linear` | sRGB | rational polynomial |
    //! | `linear_to_srgb` | sRGB | rational polynomial |
    //! | `bt709_to_linear` | BT.709 | fast_powf approximation |
    //! | `linear_to_bt709` | BT.709 | fast_powf approximation |
    //! | `pq_to_linear` | PQ ST 2084 | rational polynomial |
    //! | `linear_to_pq` | PQ ST 2084 | rational polynomial |
    //! | `hlg_to_linear` | HLG ARIB STD-B67 | fast_pow2f approximation |
    //! | `linear_to_hlg` | HLG ARIB STD-B67 | fast_log2f approximation |
    //!
    //! # Accuracy
    //!
    //! | TF | Max error vs f64 | Within u8? | Within u16? |
    //! |---|---|---|---|
    //! | sRGB | ~5e-7 | Yes | Yes |
    //! | BT.709 forward | ~3e-7 | Yes | Yes |
    //! | BT.709 inverse | ~3e-5 (fast_powf) | Yes | Yes (1 LSB = 1.5e-5) |
    //! | PQ forward | ~7e-7 | Yes | Yes |
    //! | PQ inverse | ~3e-6 | Yes | Yes |
    //! | HLG | ~5e-6 | Yes | Yes |
    #[allow(unused_imports)]
    use num_traits::Float;
    pub(crate) mod bt709 {
        //! BT.709 transfer function (scalar + generic SIMD).
        use super::fast_math;
        use crate::tf::fast_math::Mla;
        pub(crate) const BT709_ALPHA: f32 = 0.09929682680944;
        pub(crate) const BT709_BETA: f32 = 0.018053968510807;
        /// BT.709 EOTF: encoded → linear. Uses fast_powf, max error ~3e-6.
        #[inline(always)]
        pub fn bt709_to_linear(v: f32) -> f32 {
            if v < 4.5 * BT709_BETA {
                v / 4.5
            } else {
                fast_math::fast_powf((v + BT709_ALPHA) / (1.0 + BT709_ALPHA), 1.0 / 0.45)
                    .max(BT709_BETA)
            }
        }
        /// BT.709 inverse EOTF: linear → encoded. Uses fast_powf, max error ~3e-5.
        #[inline(always)]
        pub fn linear_to_bt709(v: f32) -> f32 {
            if v < BT709_BETA {
                4.5 * v
            } else {
                (1.0 + BT709_ALPHA) * fast_math::fast_powf(v, 0.45) - BT709_ALPHA
            }
        }
        use magetypes::simd::backends::F32x4Convert;
        use magetypes::simd::generic::f32x4;
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn bt709_to_linear_x4<T: F32x4Convert>(
            t: T,
            v: f32x4<T>,
        ) -> f32x4<T> {
            let threshold = f32x4::splat_t(t, 4.5 * BT709_BETA);
            let inv_4_5 = f32x4::splat_t(t, 1.0 / 4.5);
            let alpha = f32x4::splat_t(t, BT709_ALPHA);
            let one_plus_alpha = f32x4::splat_t(t, 1.0 + BT709_ALPHA);
            let linear = v * inv_4_5;
            let normalized = (v + alpha) / one_plus_alpha;
            let power = fast_math::fast_powf_x4(t, normalized, 1.0 / 0.45);
            let power = power.max(f32x4::splat_t(t, (4.5 * BT709_BETA) * (1.0 / 4.5)));
            let mask = v.simd_lt(threshold);
            f32x4::blend(mask, linear, power)
        }
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn linear_to_bt709_x4<T: F32x4Convert>(
            t: T,
            v: f32x4<T>,
        ) -> f32x4<T> {
            let threshold = f32x4::splat_t(t, BT709_BETA);
            let scale_4_5 = f32x4::splat_t(t, 4.5);
            let one_plus_alpha = f32x4::splat_t(t, 1.0 + BT709_ALPHA);
            let alpha = f32x4::splat_t(t, BT709_ALPHA);
            let linear = v * scale_4_5;
            let power = fast_math::fast_powf_x4(t, v, 0.45);
            let power = one_plus_alpha.mla(power, -alpha);
            let mask = v.simd_lt(threshold);
            f32x4::blend(mask, linear, power)
        }
        use magetypes::simd::backends::F32x8Convert;
        use magetypes::simd::generic::f32x8;
        #[inline(always)]
        pub(crate) fn bt709_to_linear_x8<T: F32x8Convert>(
            t: T,
            v: f32x8<T>,
        ) -> f32x8<T> {
            let threshold = f32x8::splat_t(t, 4.5 * BT709_BETA);
            let inv_4_5 = f32x8::splat_t(t, 1.0 / 4.5);
            let alpha = f32x8::splat_t(t, BT709_ALPHA);
            let one_plus_alpha = f32x8::splat_t(t, 1.0 + BT709_ALPHA);
            let linear = v * inv_4_5;
            let normalized = (v + alpha) / one_plus_alpha;
            let power = fast_math::fast_powf_x8(t, normalized, 1.0 / 0.45);
            let power = power.max(f32x8::splat_t(t, (4.5 * BT709_BETA) * (1.0 / 4.5)));
            let mask = v.simd_lt(threshold);
            f32x8::blend(mask, linear, power)
        }
        #[inline(always)]
        pub(crate) fn linear_to_bt709_x8<T: F32x8Convert>(
            t: T,
            v: f32x8<T>,
        ) -> f32x8<T> {
            let threshold = f32x8::splat_t(t, BT709_BETA);
            let scale_4_5 = f32x8::splat_t(t, 4.5);
            let one_plus_alpha = f32x8::splat_t(t, 1.0 + BT709_ALPHA);
            let alpha = f32x8::splat_t(t, BT709_ALPHA);
            let linear = v * scale_4_5;
            let power = fast_math::fast_powf_x8(t, v, 0.45);
            let power = one_plus_alpha.mla(power, -alpha);
            let mask = v.simd_lt(threshold);
            f32x8::blend(mask, linear, power)
        }
        use magetypes::simd::backends::F32x16Convert;
        use magetypes::simd::generic::f32x16;
        #[inline(always)]
        pub(crate) fn bt709_to_linear_x16<T: F32x16Convert>(
            t: T,
            v: f32x16<T>,
        ) -> f32x16<T> {
            let threshold = f32x16::splat_t(t, 4.5 * BT709_BETA);
            let inv_4_5 = f32x16::splat_t(t, 1.0 / 4.5);
            let alpha = f32x16::splat_t(t, BT709_ALPHA);
            let one_plus_alpha = f32x16::splat_t(t, 1.0 + BT709_ALPHA);
            let linear = v * inv_4_5;
            let normalized = (v + alpha) / one_plus_alpha;
            let power = fast_math::fast_powf_x16(t, normalized, 1.0 / 0.45);
            let power = power.max(f32x16::splat_t(t, (4.5 * BT709_BETA) * (1.0 / 4.5)));
            let mask = v.simd_lt(threshold);
            f32x16::blend(mask, linear, power)
        }
        #[inline(always)]
        pub(crate) fn linear_to_bt709_x16<T: F32x16Convert>(
            t: T,
            v: f32x16<T>,
        ) -> f32x16<T> {
            let threshold = f32x16::splat_t(t, BT709_BETA);
            let scale_4_5 = f32x16::splat_t(t, 4.5);
            let one_plus_alpha = f32x16::splat_t(t, 1.0 + BT709_ALPHA);
            let alpha = f32x16::splat_t(t, BT709_ALPHA);
            let linear = v * scale_4_5;
            let power = fast_math::fast_powf_x16(t, v, 0.45);
            let power = one_plus_alpha.mla(power, -alpha);
            let mask = v.simd_lt(threshold);
            f32x16::blend(mask, linear, power)
        }
    }
    pub(crate) mod fast_math {
        //! Fast transfer function math using rational polynomials and bit tricks.
        //!
        //! Replaces libm `powf()` (~40-100ns, non-vectorizable) with rational polynomial
        //! approximations (~5ns, SIMD-vectorizable). Coefficients from libjxl (BSD licensed).
        #![allow(clippy::excessive_precision, clippy::assign_op_pattern)]
        #[allow(unused_imports)]
        use num_traits::Float;
        /// Evaluate a rational polynomial P(x)/Q(x) using Horner's method.
        ///
        /// Coefficients are stored lowest-degree-first: `p[0] + p[1]*x + p[2]*x^2 + ...`
        #[inline(always)]
        pub(crate) fn eval_rational_poly<const P: usize, const Q: usize>(
            x: f32,
            p: [f32; P],
            q: [f32; Q],
        ) -> f32 {
            let mut yp = p[P - 1];
            for i in (0..P - 1).rev() {
                yp = yp.mul_add(x, p[i]);
            }
            let mut yq = q[Q - 1];
            for i in (0..Q - 1).rev() {
                yq = yq.mul_add(x, q[i]);
            }
            yp / yq
        }
        /// Fast base-2 logarithm (~3e-6 relative error).
        ///
        /// Uses integer bit extraction for exponent + rational polynomial on mantissa.
        /// Coefficients from libjxl (via jxl crate).
        #[inline(always)]
        pub(crate) fn fast_log2f(x: f32) -> f32 {
            let x_bits = x.to_bits() as i32;
            let exp_bits = x_bits.wrapping_sub(0x3f2aaaab);
            let exp_shifted = exp_bits >> 23;
            let mantissa = f32::from_bits(
                (x_bits.wrapping_sub(exp_shifted << 23)) as u32,
            );
            let exp_val = exp_shifted as f32;
            eval_rational_poly(mantissa - 1.0, LOG2_P, LOG2_Q) + exp_val
        }
        /// Fast base-2 exponentiation (~3e-6 relative error).
        ///
        /// Splits into integer exponent (bit shift) + fractional part (rational polynomial).
        /// Coefficients from libjxl (via jxl crate).
        #[inline(always)]
        pub(crate) fn fast_pow2f(x: f32) -> f32 {
            let x = x.clamp(-126.0, 128.0);
            let x_floor = x.floor();
            let exp = f32::from_bits(((x_floor as i32 + 127) as u32) << 23);
            let frac = x - x_floor;
            let num = frac.mul_add(1.0, POW2_NUM[0]);
            let num = num.mul_add(frac, POW2_NUM[1]);
            let num = num.mul_add(frac, POW2_NUM[2]);
            let num = num * exp;
            let den = POW2_DEN[0].mul_add(frac, POW2_DEN[1]);
            let den = den.mul_add(frac, POW2_DEN[2]);
            let den = den.mul_add(frac, POW2_DEN[3]);
            num / den
        }
        /// Fast power function: `base^exp` (~3e-5 relative error).
        ///
        /// Computed as `pow2(exp * log2(base))`.
        #[inline(always)]
        pub(crate) fn fast_powf(base: f32, exp: f32) -> f32 {
            fast_pow2f(fast_log2f(base) * exp)
        }
        pub(crate) const LOG2_P: [f32; 3] = [
            -1.8503833400518310e-6,
            1.4287160470083755,
            7.4245873327820566e-1,
        ];
        pub(crate) const LOG2_Q: [f32; 3] = [
            9.9032814277590719e-1,
            1.0096718572241148,
            1.7409343003366853e-1,
        ];
        pub(crate) const POW2_NUM: [f32; 3] = [1.01749063e1, 4.88687798e1, 9.85506591e1];
        pub(crate) const POW2_DEN: [f32; 4] = [
            2.10242958e-1,
            -2.22328856e-2,
            -1.94414990e1,
            9.85506633e1,
        ];
        use magetypes::simd::backends::{
            F32x4Backend, F32x4Convert, F32x8Backend, F32x16Backend,
        };
        use magetypes::simd::generic::{f32x4, i32x4};
        /// Existing arithmetic contract: fused on native, opt-in guaranteed fusion on WASM.
        /// Magetypes owns the hardware/engine probe and software fallback behind `_fused`.
        pub(crate) const USE_FUSED_MATH: bool = !false;
        /// Multiply-add in the kernels. Fused wherever that costs nothing or the scalar tier needs it, so
        /// every native tier gives the same bits. On WASM, `mul_add_fused` is software unless the engine
        /// fuses, so it stays relaxed there unless the `precise-portable` feature asks for the same bits.
        pub(crate) trait Mla: Sized {
            fn mla(self, a: Self, b: Self) -> Self;
        }
        impl<T: F32x4Backend> Mla for f32x4<T> {
            #[inline(always)]
            fn mla(self, a: Self, b: Self) -> Self {
                if USE_FUSED_MATH {
                    self.mul_add_fused(a, b)
                } else {
                    self.mul_add_relaxed(a, b)
                }
            }
        }
        impl<T: F32x8Backend> Mla for f32x8<T> {
            #[inline(always)]
            fn mla(self, a: Self, b: Self) -> Self {
                if USE_FUSED_MATH {
                    self.mul_add_fused(a, b)
                } else {
                    self.mul_add_relaxed(a, b)
                }
            }
        }
        impl<T: F32x16Backend> Mla for f32x16<T> {
            #[inline(always)]
            fn mla(self, a: Self, b: Self) -> Self {
                if USE_FUSED_MATH {
                    self.mul_add_fused(a, b)
                } else {
                    self.mul_add_relaxed(a, b)
                }
            }
        }
        pub(crate) const POW_SQRT1_2: u32 = 0x3f35_04f3;
        pub(crate) const POW_LOG2: [f32; 4] = [
            2.885_39,
            0.961_800_76,
            0.576_974_45,
            0.434_411_97,
        ];
        pub(crate) const POW_EXP2: [f32; 7] = [
            1.0,
            core::f32::consts::LN_2,
            0.240_226_46,
            0.055_504_545,
            0.009_618_055,
            0.001_333_37,
            0.000_154_47,
        ];
        /// Evaluate degree-4 rational polynomial P(x)/Q(x) on 4 f32 values via FMA.
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn eval_rational_poly_x4<T: F32x4Convert>(
            t: T,
            x: f32x4<T>,
            p: [f32; 5],
            q: [f32; 5],
        ) -> f32x4<T> {
            let mut yp = f32x4::splat_t(t, p[4]);
            yp = yp.mla(x, f32x4::splat_t(t, p[3]));
            yp = yp.mla(x, f32x4::splat_t(t, p[2]));
            yp = yp.mla(x, f32x4::splat_t(t, p[1]));
            yp = yp.mla(x, f32x4::splat_t(t, p[0]));
            let mut yq = f32x4::splat_t(t, q[4]);
            yq = yq.mla(x, f32x4::splat_t(t, q[3]));
            yq = yq.mla(x, f32x4::splat_t(t, q[2]));
            yq = yq.mla(x, f32x4::splat_t(t, q[1]));
            yq = yq.mla(x, f32x4::splat_t(t, q[0]));
            yp / yq
        }
        /// fast_log2f on 4 f32 values — bit manipulation + rational polynomial.
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn fast_log2f_x4<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            let x_bits = x.bitcast_to_i32();
            let magic = i32x4::splat_t(t, 0x3f2aaaab_u32 as i32);
            let exp_bits = x_bits - magic;
            let exp_shifted = exp_bits.shr_arithmetic_const::<23>();
            let shifted_back = exp_shifted.shl_const::<23>();
            let mantissa_bits = x_bits - shifted_back;
            let mantissa = f32x4::from_i32_bitcast_t(t, mantissa_bits);
            let exp_f = f32x4::from_i32_t(t, exp_shifted);
            let one = f32x4::splat_t(t, 1.0);
            let m = mantissa - one;
            let mut yp = f32x4::splat_t(t, LOG2_P[2]);
            yp = yp.mla(m, f32x4::splat_t(t, LOG2_P[1]));
            yp = yp.mla(m, f32x4::splat_t(t, LOG2_P[0]));
            let mut yq = f32x4::splat_t(t, LOG2_Q[2]);
            yq = yq.mla(m, f32x4::splat_t(t, LOG2_Q[1]));
            yq = yq.mla(m, f32x4::splat_t(t, LOG2_Q[0]));
            let poly = yp / yq;
            poly + exp_f
        }
        /// fast_pow2f on 4 f32 values — integer bit manipulation + polynomial.
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn fast_pow2f_x4<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
            let x_floor = x.floor();
            let frac = x - x_floor;
            let x_floor_i = x_floor.to_i32();
            let bias = i32x4::splat_t(t, 127);
            let exp_bits = (x_floor_i + bias).shl_const::<23>();
            let exp = f32x4::from_i32_bitcast_t(t, exp_bits);
            let mut num = frac + f32x4::splat_t(t, POW2_NUM[0]);
            num = num.mla(frac, f32x4::splat_t(t, POW2_NUM[1]));
            num = num.mla(frac, f32x4::splat_t(t, POW2_NUM[2]));
            num = num * exp;
            let mut den = f32x4::splat_t(t, POW2_DEN[0])
                .mla(frac, f32x4::splat_t(t, POW2_DEN[1]));
            den = den.mla(frac, f32x4::splat_t(t, POW2_DEN[2]));
            den = den.mla(frac, f32x4::splat_t(t, POW2_DEN[3]));
            num / den
        }
        /// fast_powf(base, exp) on 4 f32 values: pow2f(exp * log2f(base)).
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn fast_powf_x4<T: F32x4Convert>(
            t: T,
            base: f32x4<T>,
            exponent: f32,
        ) -> f32x4<T> {
            let log2 = fast_log2f_x4(t, base);
            let scaled = log2 * f32x4::splat_t(t, exponent);
            fast_pow2f_x4(t, scaled)
        }
        use magetypes::simd::backends::F32x8Convert;
        use magetypes::simd::generic::{f32x8, i32x8};
        /// Evaluate degree-4 rational polynomial P(x)/Q(x) on 8 f32 values via FMA.
        #[inline(always)]
        pub(crate) fn eval_rational_poly_x8<T: F32x8Convert>(
            t: T,
            x: f32x8<T>,
            p: [f32; 5],
            q: [f32; 5],
        ) -> f32x8<T> {
            let mut yp = f32x8::splat_t(t, p[4]);
            yp = yp.mla(x, f32x8::splat_t(t, p[3]));
            yp = yp.mla(x, f32x8::splat_t(t, p[2]));
            yp = yp.mla(x, f32x8::splat_t(t, p[1]));
            yp = yp.mla(x, f32x8::splat_t(t, p[0]));
            let mut yq = f32x8::splat_t(t, q[4]);
            yq = yq.mla(x, f32x8::splat_t(t, q[3]));
            yq = yq.mla(x, f32x8::splat_t(t, q[2]));
            yq = yq.mla(x, f32x8::splat_t(t, q[1]));
            yq = yq.mla(x, f32x8::splat_t(t, q[0]));
            yp / yq
        }
        /// fast_log2f on 8 f32 values — bit manipulation + rational polynomial.
        #[inline(always)]
        pub(crate) fn fast_log2f_x8<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
            let x_bits = x.bitcast_to_i32();
            let magic = i32x8::splat_t(t, 0x3f2aaaab_u32 as i32);
            let exp_bits = x_bits - magic;
            let exp_shifted = exp_bits.shr_arithmetic_const::<23>();
            let shifted_back = exp_shifted.shl_const::<23>();
            let mantissa_bits = x_bits - shifted_back;
            let mantissa = f32x8::from_i32_bitcast_t(t, mantissa_bits);
            let exp_f = f32x8::from_i32_t(t, exp_shifted);
            let one = f32x8::splat_t(t, 1.0);
            let m = mantissa - one;
            let mut yp = f32x8::splat_t(t, LOG2_P[2]);
            yp = yp.mla(m, f32x8::splat_t(t, LOG2_P[1]));
            yp = yp.mla(m, f32x8::splat_t(t, LOG2_P[0]));
            let mut yq = f32x8::splat_t(t, LOG2_Q[2]);
            yq = yq.mla(m, f32x8::splat_t(t, LOG2_Q[1]));
            yq = yq.mla(m, f32x8::splat_t(t, LOG2_Q[0]));
            let poly = yp / yq;
            poly + exp_f
        }
        /// fast_pow2f on 8 f32 values — integer bit manipulation + polynomial.
        #[inline(always)]
        pub(crate) fn fast_pow2f_x8<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
            let x_floor = x.floor();
            let frac = x - x_floor;
            let x_floor_i = x_floor.to_i32();
            let bias = i32x8::splat_t(t, 127);
            let exp_bits = (x_floor_i + bias).shl_const::<23>();
            let exp = f32x8::from_i32_bitcast_t(t, exp_bits);
            let mut num = frac + f32x8::splat_t(t, POW2_NUM[0]);
            num = num.mla(frac, f32x8::splat_t(t, POW2_NUM[1]));
            num = num.mla(frac, f32x8::splat_t(t, POW2_NUM[2]));
            num = num * exp;
            let mut den = f32x8::splat_t(t, POW2_DEN[0])
                .mla(frac, f32x8::splat_t(t, POW2_DEN[1]));
            den = den.mla(frac, f32x8::splat_t(t, POW2_DEN[2]));
            den = den.mla(frac, f32x8::splat_t(t, POW2_DEN[3]));
            num / den
        }
        /// fast_powf(base, exp) on 8 f32 values: pow2f(exp * log2f(base)).
        #[inline(always)]
        pub(crate) fn fast_powf_x8<T: F32x8Convert>(
            t: T,
            base: f32x8<T>,
            exponent: f32,
        ) -> f32x8<T> {
            let log2 = fast_log2f_x8(t, base);
            let scaled = log2 * f32x8::splat_t(t, exponent);
            fast_pow2f_x8(t, scaled)
        }
        use magetypes::simd::backends::F32x16Convert;
        use magetypes::simd::generic::{f32x16, i32x16};
        /// Evaluate degree-4 rational polynomial P(x)/Q(x) on 16 f32 values via FMA.
        #[inline(always)]
        pub(crate) fn eval_rational_poly_x16<T: F32x16Convert>(
            t: T,
            x: f32x16<T>,
            p: [f32; 5],
            q: [f32; 5],
        ) -> f32x16<T> {
            let mut yp = f32x16::splat_t(t, p[4]);
            yp = yp.mla(x, f32x16::splat_t(t, p[3]));
            yp = yp.mla(x, f32x16::splat_t(t, p[2]));
            yp = yp.mla(x, f32x16::splat_t(t, p[1]));
            yp = yp.mla(x, f32x16::splat_t(t, p[0]));
            let mut yq = f32x16::splat_t(t, q[4]);
            yq = yq.mla(x, f32x16::splat_t(t, q[3]));
            yq = yq.mla(x, f32x16::splat_t(t, q[2]));
            yq = yq.mla(x, f32x16::splat_t(t, q[1]));
            yq = yq.mla(x, f32x16::splat_t(t, q[0]));
            yp / yq
        }
        /// fast_log2f on 16 f32 values — bit manipulation + rational polynomial.
        #[inline(always)]
        pub(crate) fn fast_log2f_x16<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            let x_bits = x.bitcast_to_i32();
            let magic = i32x16::splat_t(t, 0x3f2aaaab_u32 as i32);
            let exp_bits = x_bits - magic;
            let exp_shifted = exp_bits.shr_arithmetic_const::<23>();
            let shifted_back = exp_shifted.shl_const::<23>();
            let mantissa_bits = x_bits - shifted_back;
            let mantissa = f32x16::from_i32_bitcast_t(t, mantissa_bits);
            let exp_f = f32x16::from_i32_t(t, exp_shifted);
            let one = f32x16::splat_t(t, 1.0);
            let m = mantissa - one;
            let mut yp = f32x16::splat_t(t, LOG2_P[2]);
            yp = yp.mla(m, f32x16::splat_t(t, LOG2_P[1]));
            yp = yp.mla(m, f32x16::splat_t(t, LOG2_P[0]));
            let mut yq = f32x16::splat_t(t, LOG2_Q[2]);
            yq = yq.mla(m, f32x16::splat_t(t, LOG2_Q[1]));
            yq = yq.mla(m, f32x16::splat_t(t, LOG2_Q[0]));
            let poly = yp / yq;
            poly + exp_f
        }
        /// fast_pow2f on 16 f32 values — integer bit manipulation + polynomial.
        #[inline(always)]
        pub(crate) fn fast_pow2f_x16<T: F32x16Convert>(t: T, x: f32x16<T>) -> f32x16<T> {
            let x_floor = x.floor();
            let frac = x - x_floor;
            let x_floor_i = x_floor.to_i32();
            let bias = i32x16::splat_t(t, 127);
            let exp_bits = (x_floor_i + bias).shl_const::<23>();
            let exp = f32x16::from_i32_bitcast_t(t, exp_bits);
            let mut num = frac + f32x16::splat_t(t, POW2_NUM[0]);
            num = num.mla(frac, f32x16::splat_t(t, POW2_NUM[1]));
            num = num.mla(frac, f32x16::splat_t(t, POW2_NUM[2]));
            num = num * exp;
            let mut den = f32x16::splat_t(t, POW2_DEN[0])
                .mla(frac, f32x16::splat_t(t, POW2_DEN[1]));
            den = den.mla(frac, f32x16::splat_t(t, POW2_DEN[2]));
            den = den.mla(frac, f32x16::splat_t(t, POW2_DEN[3]));
            num / den
        }
        /// fast_powf(base, exp) on 16 f32 values: pow2f(exp * log2f(base)).
        #[inline(always)]
        pub(crate) fn fast_powf_x16<T: F32x16Convert>(
            t: T,
            base: f32x16<T>,
            exponent: f32,
        ) -> f32x16<T> {
            let log2 = fast_log2f_x16(t, base);
            let scaled = log2 * f32x16::splat_t(t, exponent);
            fast_pow2f_x16(t, scaled)
        }
        #[inline(always)]
        pub(crate) fn pow_x4<T: F32x4Convert>(t: T, x: f32x4<T>, n: f32) -> f32x4<T> {
            let f = |v: f32| f32x4::splat_t(t, v);
            let i = |v: i32| i32x4::splat_t(t, v);
            let adjusted = x.bitcast_to_i32() + i((0x3f80_0000u32 - POW_SQRT1_2) as i32);
            let exponent = f32x4::from_i32_t(
                t,
                adjusted.shr_arithmetic_const::<23>() - i(127),
            );
            let a = f32x4::from_i32_bitcast_t(
                t,
                (adjusted & i(0x007f_ffff)) + i(POW_SQRT1_2 as i32),
            );
            let y = (a - f(1.0)) / (a + f(1.0));
            let y2 = y * y;
            let poly = f(POW_LOG2[3]).mla(y2, f(POW_LOG2[2]));
            let poly = poly.mla(y2, f(POW_LOG2[1]));
            let poly = poly.mla(y2, f(POW_LOG2[0]));
            let log = poly.mla(y, exponent);
            let log = f32x4::blend(x.simd_eq(f(0.0)), f(f32::NEG_INFINITY), log);
            let log = f32x4::blend(x.simd_lt(f(0.0)), f(f32::NAN), log);
            let log = f32x4::blend(x.simd_eq(f(f32::INFINITY)), f(f32::INFINITY), log);
            let e = log * f(n);
            let clamped = e.max(f(-126.0)).min(f(128.0));
            let xi = clamped.round().min(f(127.0));
            let xf = clamped - xi;
            let poly = f(POW_EXP2[6]).mla(xf, f(POW_EXP2[5]));
            let poly = poly.mla(xf, f(POW_EXP2[4]));
            let poly = poly.mla(xf, f(POW_EXP2[3]));
            let poly = poly.mla(xf, f(POW_EXP2[2]));
            let poly = poly.mla(xf, f(POW_EXP2[1]));
            let poly = poly.mla(xf, f(POW_EXP2[0]));
            let scale = (xi.to_i32_round() + i(127)).shl_const::<23>();
            let result = poly * f32x4::from_i32_bitcast_t(t, scale);
            let result = f32x4::blend(e.simd_lt(f(-126.0)), f(0.0), result);
            f32x4::blend(e.simd_ge(f(128.0)), f(f32::INFINITY), result)
        }
        #[inline(always)]
        pub(crate) fn pow_x8<T: F32x8Convert>(t: T, x: f32x8<T>, n: f32) -> f32x8<T> {
            let f = |v: f32| f32x8::splat_t(t, v);
            let i = |v: i32| i32x8::splat_t(t, v);
            let adjusted = x.bitcast_to_i32() + i((0x3f80_0000u32 - POW_SQRT1_2) as i32);
            let exponent = f32x8::from_i32_t(
                t,
                adjusted.shr_arithmetic_const::<23>() - i(127),
            );
            let a = f32x8::from_i32_bitcast_t(
                t,
                (adjusted & i(0x007f_ffff)) + i(POW_SQRT1_2 as i32),
            );
            let y = (a - f(1.0)) / (a + f(1.0));
            let y2 = y * y;
            let poly = f(POW_LOG2[3]).mla(y2, f(POW_LOG2[2]));
            let poly = poly.mla(y2, f(POW_LOG2[1]));
            let poly = poly.mla(y2, f(POW_LOG2[0]));
            let log = poly.mla(y, exponent);
            let log = f32x8::blend(x.simd_eq(f(0.0)), f(f32::NEG_INFINITY), log);
            let log = f32x8::blend(x.simd_lt(f(0.0)), f(f32::NAN), log);
            let log = f32x8::blend(x.simd_eq(f(f32::INFINITY)), f(f32::INFINITY), log);
            let e = log * f(n);
            let clamped = e.max(f(-126.0)).min(f(128.0));
            let xi = clamped.round().min(f(127.0));
            let xf = clamped - xi;
            let poly = f(POW_EXP2[6]).mla(xf, f(POW_EXP2[5]));
            let poly = poly.mla(xf, f(POW_EXP2[4]));
            let poly = poly.mla(xf, f(POW_EXP2[3]));
            let poly = poly.mla(xf, f(POW_EXP2[2]));
            let poly = poly.mla(xf, f(POW_EXP2[1]));
            let poly = poly.mla(xf, f(POW_EXP2[0]));
            let scale = (xi.to_i32_round() + i(127)).shl_const::<23>();
            let result = poly * f32x8::from_i32_bitcast_t(t, scale);
            let result = f32x8::blend(e.simd_lt(f(-126.0)), f(0.0), result);
            f32x8::blend(e.simd_ge(f(128.0)), f(f32::INFINITY), result)
        }
        #[inline(always)]
        pub(crate) fn pow_x16<T: F32x16Convert>(
            t: T,
            x: f32x16<T>,
            n: f32,
        ) -> f32x16<T> {
            let f = |v: f32| f32x16::splat_t(t, v);
            let i = |v: i32| i32x16::splat_t(t, v);
            let adjusted = x.bitcast_to_i32() + i((0x3f80_0000u32 - POW_SQRT1_2) as i32);
            let exponent = f32x16::from_i32_t(
                t,
                adjusted.shr_arithmetic_const::<23>() - i(127),
            );
            let a = f32x16::from_i32_bitcast_t(
                t,
                (adjusted & i(0x007f_ffff)) + i(POW_SQRT1_2 as i32),
            );
            let y = (a - f(1.0)) / (a + f(1.0));
            let y2 = y * y;
            let poly = f(POW_LOG2[3]).mla(y2, f(POW_LOG2[2]));
            let poly = poly.mla(y2, f(POW_LOG2[1]));
            let poly = poly.mla(y2, f(POW_LOG2[0]));
            let log = poly.mla(y, exponent);
            let log = f32x16::blend(x.simd_eq(f(0.0)), f(f32::NEG_INFINITY), log);
            let log = f32x16::blend(x.simd_lt(f(0.0)), f(f32::NAN), log);
            let log = f32x16::blend(x.simd_eq(f(f32::INFINITY)), f(f32::INFINITY), log);
            let e = log * f(n);
            let clamped = e.max(f(-126.0)).min(f(128.0));
            let xi = clamped.round().min(f(127.0));
            let xf = clamped - xi;
            let poly = f(POW_EXP2[6]).mla(xf, f(POW_EXP2[5]));
            let poly = poly.mla(xf, f(POW_EXP2[4]));
            let poly = poly.mla(xf, f(POW_EXP2[3]));
            let poly = poly.mla(xf, f(POW_EXP2[2]));
            let poly = poly.mla(xf, f(POW_EXP2[1]));
            let poly = poly.mla(xf, f(POW_EXP2[0]));
            let scale = (xi.to_i32_round() + i(127)).shl_const::<23>();
            let result = poly * f32x16::from_i32_bitcast_t(t, scale);
            let result = f32x16::blend(e.simd_lt(f(-126.0)), f(0.0), result);
            f32x16::blend(e.simd_ge(f(128.0)), f(f32::INFINITY), result)
        }
    }
    pub(crate) mod hlg {
        //! HLG (ARIB STD-B67) transfer function (scalar + generic SIMD).
        use super::fast_math;
        use crate::tf::fast_math::Mla;
        #[allow(unused_imports)]
        use num_traits::Float;
        const HLG_A: f32 = 0.17883277;
        pub(crate) const HLG_B: f32 = 0.28466892;
        pub(crate) const HLG_C: f32 = 0.55991073;
        pub(crate) const HLG_A_INV_LOG2E: f32 = HLG_A * core::f32::consts::LN_2;
        pub(crate) const HLG_INV_A_LOG2E: f32 = core::f32::consts::LOG2_E / HLG_A;
        /// HLG inverse OETF: signal → scene linear. Uses fast_pow2f for exp().
        #[inline(always)]
        pub fn hlg_to_linear(v: f32) -> f32 {
            if v <= 0.0 {
                0.0
            } else if v <= 0.5 {
                (v * v) / 3.0
            } else {
                (fast_math::fast_pow2f((v - HLG_C) * HLG_INV_A_LOG2E) + HLG_B) / 12.0
            }
        }
        /// HLG OETF: scene linear → signal. Uses fast_log2f for ln().
        #[inline(always)]
        pub fn linear_to_hlg(v: f32) -> f32 {
            if v <= 0.0 {
                0.0
            } else if v <= 1.0 / 12.0 {
                (3.0 * v).sqrt()
            } else {
                (HLG_A_INV_LOG2E * fast_math::fast_log2f(12.0 * v - HLG_B) + HLG_C)
                    .max(0.5)
            }
        }
        use magetypes::simd::backends::F32x4Convert;
        use magetypes::simd::generic::f32x4;
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn hlg_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
            let zero = f32x4::zero_t(t);
            let half = f32x4::splat_t(t, 0.5);
            let third = f32x4::splat_t(t, 1.0 / 3.0);
            let inv_12 = f32x4::splat_t(t, 1.0 / 12.0);
            let hlg_b = f32x4::splat_t(t, HLG_B);
            let hlg_c = f32x4::splat_t(t, HLG_C);
            let hlg_inv_a_log2e = f32x4::splat_t(t, HLG_INV_A_LOG2E);
            let a = v.max(zero);
            let low = (a * a) * third;
            let exp_arg = (a - hlg_c) * hlg_inv_a_log2e;
            let exp_val = fast_math::fast_pow2f_x4(t, exp_arg);
            let high = (exp_val + hlg_b) * inv_12;
            let mask = a.simd_le(half);
            f32x4::blend(mask, low, high)
        }
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn linear_to_hlg_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
            let zero = f32x4::zero_t(t);
            let threshold = f32x4::splat_t(t, 1.0 / 12.0);
            let three = f32x4::splat_t(t, 3.0);
            let twelve = f32x4::splat_t(t, 12.0);
            let hlg_a_ln2 = f32x4::splat_t(t, HLG_A_INV_LOG2E);
            let hlg_b = f32x4::splat_t(t, HLG_B);
            let hlg_c = f32x4::splat_t(t, HLG_C);
            let a = v.max(zero);
            let low = (three * a).sqrt();
            let arg = twelve * a - hlg_b;
            let log2_val = fast_math::fast_log2f_x4(t, arg);
            let high = hlg_a_ln2.mla(log2_val, hlg_c).max(f32x4::splat_t(t, 0.5));
            let mask = a.simd_le(threshold);
            f32x4::blend(mask, low, high)
        }
        use magetypes::simd::backends::F32x8Convert;
        use magetypes::simd::generic::f32x8;
        #[inline(always)]
        pub(crate) fn hlg_to_linear_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
            let zero = f32x8::zero_t(t);
            let half = f32x8::splat_t(t, 0.5);
            let third = f32x8::splat_t(t, 1.0 / 3.0);
            let inv_12 = f32x8::splat_t(t, 1.0 / 12.0);
            let hlg_b = f32x8::splat_t(t, HLG_B);
            let hlg_c = f32x8::splat_t(t, HLG_C);
            let hlg_inv_a_log2e = f32x8::splat_t(t, HLG_INV_A_LOG2E);
            let a = v.max(zero);
            let low = (a * a) * third;
            let exp_arg = (a - hlg_c) * hlg_inv_a_log2e;
            let exp_val = fast_math::fast_pow2f_x8(t, exp_arg);
            let high = (exp_val + hlg_b) * inv_12;
            let mask = a.simd_le(half);
            f32x8::blend(mask, low, high)
        }
        #[inline(always)]
        pub(crate) fn linear_to_hlg_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
            let zero = f32x8::zero_t(t);
            let threshold = f32x8::splat_t(t, 1.0 / 12.0);
            let three = f32x8::splat_t(t, 3.0);
            let twelve = f32x8::splat_t(t, 12.0);
            let hlg_a_ln2 = f32x8::splat_t(t, HLG_A_INV_LOG2E);
            let hlg_b = f32x8::splat_t(t, HLG_B);
            let hlg_c = f32x8::splat_t(t, HLG_C);
            let a = v.max(zero);
            let low = (three * a).sqrt();
            let arg = twelve * a - hlg_b;
            let log2_val = fast_math::fast_log2f_x8(t, arg);
            let high = hlg_a_ln2.mla(log2_val, hlg_c).max(f32x8::splat_t(t, 0.5));
            let mask = a.simd_le(threshold);
            f32x8::blend(mask, low, high)
        }
        use magetypes::simd::backends::F32x16Convert;
        use magetypes::simd::generic::f32x16;
        #[inline(always)]
        pub(crate) fn hlg_to_linear_x16<T: F32x16Convert>(
            t: T,
            v: f32x16<T>,
        ) -> f32x16<T> {
            let zero = f32x16::zero_t(t);
            let half = f32x16::splat_t(t, 0.5);
            let third = f32x16::splat_t(t, 1.0 / 3.0);
            let inv_12 = f32x16::splat_t(t, 1.0 / 12.0);
            let hlg_b = f32x16::splat_t(t, HLG_B);
            let hlg_c = f32x16::splat_t(t, HLG_C);
            let hlg_inv_a_log2e = f32x16::splat_t(t, HLG_INV_A_LOG2E);
            let a = v.max(zero);
            let low = (a * a) * third;
            let exp_arg = (a - hlg_c) * hlg_inv_a_log2e;
            let exp_val = fast_math::fast_pow2f_x16(t, exp_arg);
            let high = (exp_val + hlg_b) * inv_12;
            let mask = a.simd_le(half);
            f32x16::blend(mask, low, high)
        }
        #[inline(always)]
        pub(crate) fn linear_to_hlg_x16<T: F32x16Convert>(
            t: T,
            v: f32x16<T>,
        ) -> f32x16<T> {
            let zero = f32x16::zero_t(t);
            let threshold = f32x16::splat_t(t, 1.0 / 12.0);
            let three = f32x16::splat_t(t, 3.0);
            let twelve = f32x16::splat_t(t, 12.0);
            let hlg_a_ln2 = f32x16::splat_t(t, HLG_A_INV_LOG2E);
            let hlg_b = f32x16::splat_t(t, HLG_B);
            let hlg_c = f32x16::splat_t(t, HLG_C);
            let a = v.max(zero);
            let low = (three * a).sqrt();
            let arg = twelve * a - hlg_b;
            let log2_val = fast_math::fast_log2f_x16(t, arg);
            let high = hlg_a_ln2.mla(log2_val, hlg_c).max(f32x16::splat_t(t, 0.5));
            let mask = a.simd_le(threshold);
            f32x16::blend(mask, low, high)
        }
    }
    pub(crate) mod pq {
        //! PQ (SMPTE ST 2084) transfer function (scalar + generic SIMD).
        use super::fast_math;
        use crate::tf::fast_math::Mla;
        #[allow(unused_imports)]
        use num_traits::Float;
        /// PQ EOTF: signal → linear. Two-range rational polynomial, sub-U16 roundtrip.
        ///
        /// Zero `powf()` calls for signal > 0.02 — uses `x + x*x` input transformation
        /// with separate polynomial fits for low and high signal ranges.
        /// Very small signal values use exact formula.
        #[inline(always)]
        pub fn pq_to_linear(v: f32) -> f32 {
            if v <= 0.0 {
                return 0.0;
            }
            if v < EOTF_POLY_FLOOR {
                return pq_to_linear_exact(v);
            }
            let x = v + v * v;
            if v < 0.25 {
                fast_math::eval_rational_poly(x, PQ_EOTF_P_SMALL, PQ_EOTF_Q_SMALL)
            } else {
                fast_math::eval_rational_poly(x, PQ_EOTF_P_LARGE, PQ_EOTF_Q_LARGE)
            }
        }
        /// PQ inverse EOTF: linear → signal. Rational polynomial on x^(1/4), max error ~3e-6.
        ///
        /// Two-range approximation. Very small values (v < 1e-7) use exact powf.
        /// Coefficients from libjxl.
        #[inline(always)]
        pub fn linear_to_pq(v: f32) -> f32 {
            if v <= 0.0 {
                return 0.0;
            }
            if v < OETF_POLY_FLOOR {
                return pq_from_linear_exact(v);
            }
            let a = v.sqrt().sqrt();
            if a < 0.1 {
                fast_math::eval_rational_poly(a, PQ_INV_P_SMALL, PQ_INV_Q_SMALL)
            } else {
                fast_math::eval_rational_poly(a, PQ_INV_P_LARGE, PQ_INV_Q_LARGE)
            }
        }
        pub(crate) const M1: f32 = 0.1593017578125;
        pub(crate) const M2: f32 = 78.84375;
        pub(crate) const C1: f32 = 0.8359375;
        pub(crate) const C2: f32 = 18.8515625;
        pub(crate) const C3: f32 = 18.6875;
        /// Below this signal the EOTF polynomial fails: it has a pole near 0.007, so the exact formula
        /// runs.
        pub(crate) const EOTF_POLY_FLOOR: f32 = 0.02;
        /// Below this linear value (0.001 cd/m²) the inverse-EOTF polynomial loses relative accuracy:
        /// worst 7.4e-6 relative in 1e-7..1e-6, but 9.2e-5 in 1e-8..1e-7.
        pub(crate) const OETF_POLY_FLOOR: f32 = 1e-7;
        #[inline(always)]
        fn pq_to_linear_exact(v: f32) -> f32 {
            let vp = v.powf(1.0 / M2);
            let num = (vp - C1).max(0.0);
            let den = C2 - C3 * vp;
            if den <= 0.0 {
                return 1.0;
            }
            (num / den).powf(1.0 / M1)
        }
        #[inline(always)]
        fn pq_from_linear_exact(v: f32) -> f32 {
            let vp = v.powf(M1);
            let num = C1 + C2 * vp;
            let den = 1.0 + C3 * vp;
            (num / den).powf(M2)
        }
        pub(crate) const PQ_EOTF_P_SMALL: [f32; 5] = [
            4.36033203e-10,
            -1.53651077e-07,
            2.06864196e-06,
            2.80304711e-03,
            1.47809948e-02,
        ];
        pub(crate) const PQ_EOTF_Q_SMALL: [f32; 5] = [
            -9.89955022e-03,
            1.89911767e+00,
            -1.34828020e+00,
            -7.10547502e-01,
            1.00000000e+00,
        ];
        pub(crate) const PQ_EOTF_P_LARGE: [f32; 5] = [
            5.43046166e-04,
            -6.31017431e-03,
            2.90480734e-01,
            9.80363032e-01,
            2.22819483e-01,
        ];
        pub(crate) const PQ_EOTF_Q_LARGE: [f32; 5] = [
            1.58688561e+02,
            -1.60764060e+02,
            6.51370372e+01,
            -1.26438391e+01,
            1.00000000e+00,
        ];
        pub(crate) const PQ_INV_P_LARGE: [f32; 5] = [
            1.351392e-2,
            -1.095778,
            5.522776e1,
            1.492516e2,
            4.838434e1,
        ];
        pub(crate) const PQ_INV_Q_LARGE: [f32; 5] = [
            1.012416,
            2.016708e1,
            9.26371e1,
            1.120607e2,
            2.590418e1,
        ];
        pub(crate) const PQ_INV_P_SMALL: [f32; 5] = [
            9.863406e-6,
            3.881234e-1,
            1.352821e2,
            6.889862e4,
            -2.864824e5,
        ];
        pub(crate) const PQ_INV_Q_SMALL: [f32; 5] = [
            3.371868e1,
            1.477719e3,
            1.608477e4,
            -4.389884e4,
            -2.072546e5,
        ];
        use magetypes::simd::backends::F32x4Convert;
        use magetypes::simd::generic::f32x4;
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn pq_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
            let zero = f32x4::zero_t(t);
            let a = v.max(zero);
            let x = a.mla(a, a);
            let threshold = f32x4::splat_t(t, 0.25);
            let large = fast_math::eval_rational_poly_x4(
                t,
                x,
                PQ_EOTF_P_LARGE,
                PQ_EOTF_Q_LARGE,
            );
            let small = fast_math::eval_rational_poly_x4(
                t,
                x,
                PQ_EOTF_P_SMALL,
                PQ_EOTF_Q_SMALL,
            );
            let mask = a.simd_lt(threshold);
            let result = f32x4::blend(mask, small, large);
            let pos_mask = v.simd_gt(zero);
            let result = result & pos_mask;
            let low = v.simd_lt(f32x4::splat_t(t, EOTF_POLY_FLOOR)) & pos_mask;
            if low.bitcast_to_i32().any_true() {
                let vp = fast_math::pow_x4(t, a, 1.0 / M2);
                let num = (vp - f32x4::splat_t(t, C1)).max(zero);
                let den = f32x4::splat_t(t, C2) - f32x4::splat_t(t, C3) * vp;
                return f32x4::blend(
                    low,
                    fast_math::pow_x4(t, num / den, 1.0 / M1),
                    result,
                );
            }
            result
        }
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn linear_to_pq_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
            let zero = f32x4::zero_t(t);
            let a = v.max(zero);
            let a4 = a.sqrt().sqrt();
            let threshold = f32x4::splat_t(t, 0.1);
            let large = fast_math::eval_rational_poly_x4(
                t,
                a4,
                PQ_INV_P_LARGE,
                PQ_INV_Q_LARGE,
            );
            let small = fast_math::eval_rational_poly_x4(
                t,
                a4,
                PQ_INV_P_SMALL,
                PQ_INV_Q_SMALL,
            );
            let mask = a4.simd_lt(threshold);
            let result = f32x4::blend(mask, small, large);
            let pos_mask = v.simd_gt(zero);
            let result = result & pos_mask;
            let low = v.simd_lt(f32x4::splat_t(t, OETF_POLY_FLOOR)) & pos_mask;
            if low.bitcast_to_i32().any_true() {
                let vp = fast_math::pow_x4(t, a, M1);
                let num = f32x4::splat_t(t, C1) + f32x4::splat_t(t, C2) * vp;
                let den = f32x4::splat_t(t, 1.0) + f32x4::splat_t(t, C3) * vp;
                return f32x4::blend(low, fast_math::pow_x4(t, num / den, M2), result);
            }
            result
        }
        use magetypes::simd::backends::F32x8Convert;
        use magetypes::simd::generic::f32x8;
        #[inline(always)]
        pub(crate) fn pq_to_linear_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
            let zero = f32x8::zero_t(t);
            let a = v.max(zero);
            let x = a.mla(a, a);
            let threshold = f32x8::splat_t(t, 0.25);
            let large = fast_math::eval_rational_poly_x8(
                t,
                x,
                PQ_EOTF_P_LARGE,
                PQ_EOTF_Q_LARGE,
            );
            let small = fast_math::eval_rational_poly_x8(
                t,
                x,
                PQ_EOTF_P_SMALL,
                PQ_EOTF_Q_SMALL,
            );
            let mask = a.simd_lt(threshold);
            let result = f32x8::blend(mask, small, large);
            let pos_mask = v.simd_gt(zero);
            let result = result & pos_mask;
            let low = v.simd_lt(f32x8::splat_t(t, EOTF_POLY_FLOOR)) & pos_mask;
            if low.bitcast_to_i32().any_true() {
                let vp = fast_math::pow_x8(t, a, 1.0 / M2);
                let num = (vp - f32x8::splat_t(t, C1)).max(zero);
                let den = f32x8::splat_t(t, C2) - f32x8::splat_t(t, C3) * vp;
                return f32x8::blend(
                    low,
                    fast_math::pow_x8(t, num / den, 1.0 / M1),
                    result,
                );
            }
            result
        }
        #[inline(always)]
        pub(crate) fn linear_to_pq_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
            let zero = f32x8::zero_t(t);
            let a = v.max(zero);
            let a4 = a.sqrt().sqrt();
            let threshold = f32x8::splat_t(t, 0.1);
            let large = fast_math::eval_rational_poly_x8(
                t,
                a4,
                PQ_INV_P_LARGE,
                PQ_INV_Q_LARGE,
            );
            let small = fast_math::eval_rational_poly_x8(
                t,
                a4,
                PQ_INV_P_SMALL,
                PQ_INV_Q_SMALL,
            );
            let mask = a4.simd_lt(threshold);
            let result = f32x8::blend(mask, small, large);
            let pos_mask = v.simd_gt(zero);
            let result = result & pos_mask;
            let low = v.simd_lt(f32x8::splat_t(t, OETF_POLY_FLOOR)) & pos_mask;
            if low.bitcast_to_i32().any_true() {
                let vp = fast_math::pow_x8(t, a, M1);
                let num = f32x8::splat_t(t, C1) + f32x8::splat_t(t, C2) * vp;
                let den = f32x8::splat_t(t, 1.0) + f32x8::splat_t(t, C3) * vp;
                return f32x8::blend(low, fast_math::pow_x8(t, num / den, M2), result);
            }
            result
        }
        use magetypes::simd::backends::F32x16Convert;
        use magetypes::simd::generic::f32x16;
        #[inline(always)]
        pub(crate) fn pq_to_linear_x16<T: F32x16Convert>(
            t: T,
            v: f32x16<T>,
        ) -> f32x16<T> {
            let zero = f32x16::zero_t(t);
            let a = v.max(zero);
            let x = a.mla(a, a);
            let threshold = f32x16::splat_t(t, 0.25);
            let large = fast_math::eval_rational_poly_x16(
                t,
                x,
                PQ_EOTF_P_LARGE,
                PQ_EOTF_Q_LARGE,
            );
            let small = fast_math::eval_rational_poly_x16(
                t,
                x,
                PQ_EOTF_P_SMALL,
                PQ_EOTF_Q_SMALL,
            );
            let mask = a.simd_lt(threshold);
            let result = f32x16::blend(mask, small, large);
            let pos_mask = v.simd_gt(zero);
            let result = result & pos_mask;
            let low = v.simd_lt(f32x16::splat_t(t, EOTF_POLY_FLOOR)) & pos_mask;
            if low.bitcast_to_i32().any_true() {
                let vp = fast_math::pow_x16(t, a, 1.0 / M2);
                let num = (vp - f32x16::splat_t(t, C1)).max(zero);
                let den = f32x16::splat_t(t, C2) - f32x16::splat_t(t, C3) * vp;
                return f32x16::blend(
                    low,
                    fast_math::pow_x16(t, num / den, 1.0 / M1),
                    result,
                );
            }
            result
        }
        #[inline(always)]
        pub(crate) fn linear_to_pq_x16<T: F32x16Convert>(
            t: T,
            v: f32x16<T>,
        ) -> f32x16<T> {
            let zero = f32x16::zero_t(t);
            let a = v.max(zero);
            let a4 = a.sqrt().sqrt();
            let threshold = f32x16::splat_t(t, 0.1);
            let large = fast_math::eval_rational_poly_x16(
                t,
                a4,
                PQ_INV_P_LARGE,
                PQ_INV_Q_LARGE,
            );
            let small = fast_math::eval_rational_poly_x16(
                t,
                a4,
                PQ_INV_P_SMALL,
                PQ_INV_Q_SMALL,
            );
            let mask = a4.simd_lt(threshold);
            let result = f32x16::blend(mask, small, large);
            let pos_mask = v.simd_gt(zero);
            let result = result & pos_mask;
            let low = v.simd_lt(f32x16::splat_t(t, OETF_POLY_FLOOR)) & pos_mask;
            if low.bitcast_to_i32().any_true() {
                let vp = fast_math::pow_x16(t, a, M1);
                let num = f32x16::splat_t(t, C1) + f32x16::splat_t(t, C2) * vp;
                let den = f32x16::splat_t(t, 1.0) + f32x16::splat_t(t, C3) * vp;
                return f32x16::blend(low, fast_math::pow_x16(t, num / den, M2), result);
            }
            result
        }
    }
    pub(crate) mod srgb {
        //! sRGB transfer function — generic SIMD wrappers.
        //!
        //! Scalar sRGB delegates to `crate::rational_poly` (C0-continuous rational polynomial).
        //! SIMD versions use the same polynomial coefficients inline.
        use crate::rational_poly;
        use crate::tf::fast_math::Mla;
        use magetypes::simd::backends::F32x4Convert;
        use magetypes::simd::generic::f32x4;
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn srgb_to_linear_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
            let one = f32x4::splat_t(t, 1.0);
            let threshold = f32x4::splat_t(t, rational_poly::SRGB_THRESHOLD);
            let inv_12_92 = f32x4::splat_t(t, rational_poly::LINEAR_SCALE);
            let linear = v * inv_12_92;
            let poly = super::fast_math::eval_rational_poly_x4(
                    t,
                    v,
                    rational_poly::S2L_P,
                    rational_poly::S2L_Q,
                )
                .min(one);
            let mask = v.simd_le(threshold);
            f32x4::blend(mask, linear, poly)
        }
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn linear_to_srgb_x4<T: F32x4Convert>(t: T, v: f32x4<T>) -> f32x4<T> {
            let one = f32x4::splat_t(t, 1.0);
            let threshold = f32x4::splat_t(t, rational_poly::LINEAR_THRESHOLD);
            let scale = f32x4::splat_t(t, rational_poly::TWELVE_92);
            let linear = v * scale;
            let s = v.sqrt();
            let poly = super::fast_math::eval_rational_poly_x4(
                    t,
                    s,
                    rational_poly::L2S_P,
                    rational_poly::L2S_Q,
                )
                .min(one);
            let mask = v.simd_le(threshold);
            f32x4::blend(mask, linear, poly)
        }
        use magetypes::simd::backends::F32x8Convert;
        use magetypes::simd::generic::f32x8;
        #[inline(always)]
        pub(crate) fn srgb_to_linear_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
            let one = f32x8::splat_t(t, 1.0);
            let threshold = f32x8::splat_t(t, rational_poly::SRGB_THRESHOLD);
            let inv_12_92 = f32x8::splat_t(t, rational_poly::LINEAR_SCALE);
            let linear = v * inv_12_92;
            let poly = super::fast_math::eval_rational_poly_x8(
                    t,
                    v,
                    rational_poly::S2L_P,
                    rational_poly::S2L_Q,
                )
                .min(one);
            let mask = v.simd_le(threshold);
            f32x8::blend(mask, linear, poly)
        }
        #[inline(always)]
        pub(crate) fn linear_to_srgb_x8<T: F32x8Convert>(t: T, v: f32x8<T>) -> f32x8<T> {
            let one = f32x8::splat_t(t, 1.0);
            let threshold = f32x8::splat_t(t, rational_poly::LINEAR_THRESHOLD);
            let scale = f32x8::splat_t(t, rational_poly::TWELVE_92);
            let linear = v * scale;
            let s = v.sqrt();
            let poly = super::fast_math::eval_rational_poly_x8(
                    t,
                    s,
                    rational_poly::L2S_P,
                    rational_poly::L2S_Q,
                )
                .min(one);
            let mask = v.simd_le(threshold);
            f32x8::blend(mask, linear, poly)
        }
        use magetypes::simd::backends::F32x16Backend;
        use magetypes::simd::generic::f32x16;
        /// Extended-range sRGB→linear, 16-wide. `sign(v) * eotf(|v|)` per CSS Color 4.
        #[inline(always)]
        pub(crate) fn srgb_to_linear_extended_x16<T: F32x16Backend>(
            t: T,
            v: f32x16<T>,
        ) -> f32x16<T> {
            use rational_poly::{EXT_S2L_P as P, EXT_S2L_Q as Q};
            let zero = f32x16::zero_t(t);
            let neg_mask = v.simd_lt(zero);
            let abs_v = v.abs();
            let linear_result = abs_v * f32x16::splat_t(t, rational_poly::LINEAR_SCALE);
            let x = abs_v;
            let yp = f32x16::splat_t(t, P[6]).mla(x, f32x16::splat_t(t, P[5]));
            let yp = yp.mla(x, f32x16::splat_t(t, P[4]));
            let yp = yp.mla(x, f32x16::splat_t(t, P[3]));
            let yp = yp.mla(x, f32x16::splat_t(t, P[2]));
            let yp = yp.mla(x, f32x16::splat_t(t, P[1]));
            let yp = yp.mla(x, f32x16::splat_t(t, P[0]));
            let yq = f32x16::splat_t(t, Q[6]).mla(x, f32x16::splat_t(t, Q[5]));
            let yq = yq.mla(x, f32x16::splat_t(t, Q[4]));
            let yq = yq.mla(x, f32x16::splat_t(t, Q[3]));
            let yq = yq.mla(x, f32x16::splat_t(t, Q[2]));
            let yq = yq.mla(x, f32x16::splat_t(t, Q[1]));
            let yq = yq.mla(x, f32x16::splat_t(t, Q[0]));
            let power_result = yp / yq;
            let thresh_mask = abs_v
                .simd_lt(f32x16::splat_t(t, rational_poly::SRGB_THRESHOLD));
            let result = f32x16::blend(thresh_mask, linear_result, power_result);
            f32x16::blend(neg_mask, -result, result)
        }
        /// Extended-range linear→sRGB, 16-wide. `sign(v) * oetf(|v|)` per CSS Color 4.
        #[inline(always)]
        pub(crate) fn linear_to_srgb_extended_x16<T: F32x16Backend>(
            t: T,
            v: f32x16<T>,
        ) -> f32x16<T> {
            use rational_poly::{EXT_L2S_P as P, EXT_L2S_Q as Q};
            let zero = f32x16::zero_t(t);
            let neg_mask = v.simd_lt(zero);
            let abs_v = v.abs();
            let linear_result = abs_v * f32x16::splat_t(t, rational_poly::TWELVE_92);
            let x = abs_v.sqrt();
            let yp = f32x16::splat_t(t, P[6]).mla(x, f32x16::splat_t(t, P[5]));
            let yp = yp.mla(x, f32x16::splat_t(t, P[4]));
            let yp = yp.mla(x, f32x16::splat_t(t, P[3]));
            let yp = yp.mla(x, f32x16::splat_t(t, P[2]));
            let yp = yp.mla(x, f32x16::splat_t(t, P[1]));
            let yp = yp.mla(x, f32x16::splat_t(t, P[0]));
            let yq = f32x16::splat_t(t, Q[6]).mla(x, f32x16::splat_t(t, Q[5]));
            let yq = yq.mla(x, f32x16::splat_t(t, Q[4]));
            let yq = yq.mla(x, f32x16::splat_t(t, Q[3]));
            let yq = yq.mla(x, f32x16::splat_t(t, Q[2]));
            let yq = yq.mla(x, f32x16::splat_t(t, Q[1]));
            let yq = yq.mla(x, f32x16::splat_t(t, Q[0]));
            let power_result = yp / yq;
            let thresh_mask = abs_v
                .simd_lt(f32x16::splat_t(t, rational_poly::LINEAR_THRESHOLD));
            let result = f32x16::blend(thresh_mask, linear_result, power_result);
            f32x16::blend(neg_mask, -result, result)
        }
    }
    /// sRGB EOTF: encoded → linear. Rational polynomial, max error ~5e-7.
    ///
    /// Uses rational polynomial fitted to C0-continuous sRGB — no `powf()` calls.
    /// Equivalent to `crate::rational_poly::srgb_to_linear_fast`.
    #[inline(always)]
    pub fn srgb_to_linear(v: f32) -> f32 {
        crate::rational_poly::srgb_to_linear_fast(v)
    }
    /// sRGB inverse EOTF: linear → encoded. Rational polynomial, max error ~5e-7.
    ///
    /// Uses sqrt + libjxl rational polynomial — no `powf()` calls.
    /// Equivalent to `crate::rational_poly::linear_to_srgb_fast`.
    #[inline(always)]
    pub fn linear_to_srgb(v: f32) -> f32 {
        crate::rational_poly::linear_to_srgb_fast(v)
    }
    pub use bt709::{bt709_to_linear, linear_to_bt709};
    pub use hlg::{hlg_to_linear, linear_to_hlg};
    pub use pq::{linear_to_pq, pq_to_linear};
}
/// IEC 61966-2-1:1999 textbook sRGB transfer functions.
///
/// Provides the original specification constants (threshold 0.04045, offset 0.055)
/// for interoperability with software that implements IEC 61966-2-1 verbatim.
/// The default module uses C0-continuous constants that eliminate the spec's
/// ~2.3e-9 piecewise discontinuity.
///
/// Requires the `iec` feature.
pub mod iec {
    //! IEC 61966-2-1:1999 textbook sRGB transfer functions.
    //!
    //! These use the original specification constants (threshold 0.04045, offset 0.055)
    //! which have a ~2.3e-9 discontinuity at the piecewise junction. The default
    //! module uses C0-continuous (moxcms) constants that eliminate this discontinuity.
    //!
    //! Use this module when you need **exact** IEC 61966-2-1 behavior for
    //! interoperability with software that implements the original specification
    //! verbatim (e.g., ICC profile processors, conformance test suites).
    //!
    //! The decoded f32 values differ at 244 of the 256 u8 codes (up to 2.28e-4
    //! relative at code 12). The adjusted offset changes the entire power branch,
    //! not only the piecewise junction. An exact u8 decode→encode round trip does
    //! not imply that the intermediate floating-point values are identical.
    //!
    //! Requires the `iec` feature.
    use crate::mlaf::fmla;
    #[allow(unused_imports)]
    use num_traits::Float;
    const SRGB_LINEAR_THRESHOLD: f64 = 0.04045;
    const SRGB_LINEAR_THRESHOLD_F32: f32 = SRGB_LINEAR_THRESHOLD as f32;
    const LINEAR_THRESHOLD: f64 = 0.003_130_8;
    const LINEAR_THRESHOLD_F32: f32 = LINEAR_THRESHOLD as f32;
    const LINEAR_SCALE: f64 = 1.0 / 12.92;
    const LINEAR_SCALE_F32: f32 = LINEAR_SCALE as f32;
    const SRGB_A: f64 = 0.055;
    const SRGB_A_F32: f32 = SRGB_A as f32;
    const SRGB_A_PLUS_1: f64 = 1.055;
    const SRGB_A_PLUS_1_F32: f32 = SRGB_A_PLUS_1 as f32;
    const GAMMA: f64 = 2.4;
    const INV_GAMMA: f64 = 1.0 / GAMMA;
    const INV_GAMMA_F32: f32 = INV_GAMMA as f32;
    /// Convert sRGB gamma-encoded value to linear light (f64, IEC 61966-2-1).
    ///
    /// Uses the original IEC textbook constants.
    /// Input clamped to \[0, 1\].
    #[inline]
    pub fn srgb_to_linear_f64(gamma: f64) -> f64 {
        if gamma < 0.0 {
            0.0
        } else if gamma <= SRGB_LINEAR_THRESHOLD {
            gamma * LINEAR_SCALE
        } else if gamma < 1.0 {
            ((gamma + SRGB_A) / SRGB_A_PLUS_1).powf(GAMMA)
        } else {
            1.0
        }
    }
    /// Convert sRGB gamma-encoded value to linear light (f32, IEC 61966-2-1).
    ///
    /// Uses the original IEC textbook constants.
    /// Input clamped to \[0, 1\].
    #[inline]
    pub fn srgb_to_linear(gamma: f32) -> f32 {
        if gamma < 0.0 {
            0.0
        } else if gamma <= SRGB_LINEAR_THRESHOLD_F32 {
            gamma * LINEAR_SCALE_F32
        } else if gamma < 1.0 {
            ((gamma + SRGB_A_F32) / SRGB_A_PLUS_1_F32).powf(GAMMA as f32)
        } else {
            1.0
        }
    }
    /// Convert linear light value to sRGB gamma-encoded (f64, IEC 61966-2-1).
    ///
    /// Uses the original IEC textbook constants.
    /// Input clamped to \[0, 1\].
    #[inline]
    pub fn linear_to_srgb_f64(linear: f64) -> f64 {
        if linear < 0.0 {
            0.0
        } else if linear <= LINEAR_THRESHOLD {
            linear * 12.92
        } else if linear < 1.0 {
            fmla(SRGB_A_PLUS_1, linear.powf(INV_GAMMA), -SRGB_A)
        } else {
            1.0
        }
    }
    /// Convert linear light value to sRGB gamma-encoded (f32, IEC 61966-2-1).
    ///
    /// Uses the original IEC textbook constants.
    /// Input clamped to \[0, 1\].
    #[inline]
    pub fn linear_to_srgb(linear: f32) -> f32 {
        if linear < 0.0 {
            0.0
        } else if linear <= LINEAR_THRESHOLD_F32 {
            linear * 12.92
        } else if linear < 1.0 {
            fmla(SRGB_A_PLUS_1_F32, linear.powf(INV_GAMMA_F32), -SRGB_A_F32)
        } else {
            1.0
        }
    }
}
/// Alpha threshold for unpremultiply operations.
///
/// Pixels with alpha at or below this value skip the divide-by-alpha step;
/// their RGB channels are zeroed instead. This prevents division by
/// near-zero alpha from amplifying filter-ringing noise into bright
/// white fringe artifacts at transparent edges.
///
/// The value `1.0 / 1024.0` (~0.000977) is well below the smallest alpha
/// that rounds to u8=1 (`0.5 / 255.0` ≈ 0.00196), so it never affects
/// visible pixels in u8 output. It matches the threshold used by
/// zenresize's SIMD unpremultiply and Chrome Skia's approach.
pub const UNPREMUL_ALPHA_THRESHOLD: f32 = 1.0 / 1024.0;
mod kernels {
    //! Private numerical kernels shared by target-feature entry points and runtime dispatch.
    pub(crate) mod float {
        //! Shared sRGB and custom-gamma kernels. Public adapters own no math.
        use crate::rational_poly::{
            LINEAR_SCALE, LINEAR_THRESHOLD, SRGB_THRESHOLD as SRGB_LINEAR_THRESHOLD,
            TWELVE_92,
        };
        use crate::tf::fast_math::Mla;
        use magetypes::simd::{backends::*, generic::*};
        pub(crate) mod x4 {
            use super::*;
            #[inline(always)]
            pub(crate) fn srgb_to_linear<T: F32x4Backend>(
                token: T,
                srgb: [f32; 4],
            ) -> [f32; 4] {
                use crate::rational_poly::{S2L_P, S2L_Q};
                let zero = f32x4::<T>::zero_t(token);
                let one = f32x4::<T>::splat_t(token, 1.0);
                let v = f32x4::<T>::from_array_t(token, srgb);
                let clamped = v.max(zero).min(one);
                let linear_result = clamped * f32x4::<T>::splat_t(token, LINEAR_SCALE);
                let x = clamped;
                let yp = f32x4::<T>::splat_t(token, S2L_P[4])
                    .mla(x, f32x4::<T>::splat_t(token, S2L_P[3]));
                let yp = yp.mla(x, f32x4::<T>::splat_t(token, S2L_P[2]));
                let yp = yp.mla(x, f32x4::<T>::splat_t(token, S2L_P[1]));
                let yp = yp.mla(x, f32x4::<T>::splat_t(token, S2L_P[0]));
                let yq = f32x4::<T>::splat_t(token, S2L_Q[4])
                    .mla(x, f32x4::<T>::splat_t(token, S2L_Q[3]));
                let yq = yq.mla(x, f32x4::<T>::splat_t(token, S2L_Q[2]));
                let yq = yq.mla(x, f32x4::<T>::splat_t(token, S2L_Q[1]));
                let yq = yq.mla(x, f32x4::<T>::splat_t(token, S2L_Q[0]));
                let power_result = (yp / yq).min(one);
                let mask = clamped
                    .simd_lt(f32x4::<T>::splat_t(token, SRGB_LINEAR_THRESHOLD));
                let result = f32x4::<T>::blend(mask, linear_result, power_result);
                let ge_one = v.simd_ge(one);
                f32x4::<T>::blend(ge_one, one, result).to_array()
            }
            #[inline(always)]
            pub(crate) fn linear_to_srgb<T: F32x4Backend>(
                token: T,
                linear: [f32; 4],
            ) -> [f32; 4] {
                use crate::rational_poly::{L2S_P, L2S_Q};
                let zero = f32x4::<T>::zero_t(token);
                let one = f32x4::<T>::splat_t(token, 1.0);
                let v = f32x4::<T>::from_array_t(token, linear);
                let clamped = v.max(zero).min(one);
                let linear_result = clamped * f32x4::<T>::splat_t(token, TWELVE_92);
                let x = clamped.sqrt();
                let yp = f32x4::<T>::splat_t(token, L2S_P[4])
                    .mla(x, f32x4::<T>::splat_t(token, L2S_P[3]));
                let yp = yp.mla(x, f32x4::<T>::splat_t(token, L2S_P[2]));
                let yp = yp.mla(x, f32x4::<T>::splat_t(token, L2S_P[1]));
                let yp = yp.mla(x, f32x4::<T>::splat_t(token, L2S_P[0]));
                let yq = f32x4::<T>::splat_t(token, L2S_Q[4])
                    .mla(x, f32x4::<T>::splat_t(token, L2S_Q[3]));
                let yq = yq.mla(x, f32x4::<T>::splat_t(token, L2S_Q[2]));
                let yq = yq.mla(x, f32x4::<T>::splat_t(token, L2S_Q[1]));
                let yq = yq.mla(x, f32x4::<T>::splat_t(token, L2S_Q[0]));
                let power_result = (yp / yq).min(one);
                let mask = clamped.simd_lt(f32x4::<T>::splat_t(token, LINEAR_THRESHOLD));
                let result = f32x4::<T>::blend(mask, linear_result, power_result);
                let ge_one = v.simd_ge(one);
                f32x4::<T>::blend(ge_one, one, result).to_array()
            }
            #[inline(always)]
            pub(crate) fn gamma_to_linear<T: F32x4Convert>(
                token: T,
                encoded: [f32; 4],
                gamma: f32,
            ) -> [f32; 4] {
                let zero = f32x4::<T>::zero_t(token);
                let one = f32x4::<T>::splat_t(token, 1.0);
                let v = f32x4::<T>::from_array_t(token, encoded).max(zero).min(one);
                crate::tf::fast_math::pow_x4(token, v, gamma).to_array()
            }
            #[inline(always)]
            pub(crate) fn linear_to_gamma<T: F32x4Convert>(
                token: T,
                linear: [f32; 4],
                gamma: f32,
            ) -> [f32; 4] {
                let zero = f32x4::<T>::zero_t(token);
                let one = f32x4::<T>::splat_t(token, 1.0);
                let v = f32x4::<T>::from_array_t(token, linear).max(zero).min(one);
                crate::tf::fast_math::pow_x4(token, v, 1.0 / gamma).to_array()
            }
            #[inline(always)]
            pub(crate) fn srgb_to_linear_extended<T: F32x4Backend>(
                token: T,
                srgb: [f32; 4],
            ) -> [f32; 4] {
                use crate::rational_poly::{EXT_S2L_P as P, EXT_S2L_Q as Q};
                let zero = f32x4::zero_t(token);
                let v = f32x4::from_array_t(token, srgb);
                let neg_mask = v.simd_lt(zero);
                let abs_v = v.abs();
                let linear_result = abs_v * f32x4::splat_t(token, LINEAR_SCALE);
                let x = abs_v;
                let yp = f32x4::splat_t(token, P[6]).mla(x, f32x4::splat_t(token, P[5]));
                let yp = yp.mla(x, f32x4::splat_t(token, P[4]));
                let yp = yp.mla(x, f32x4::splat_t(token, P[3]));
                let yp = yp.mla(x, f32x4::splat_t(token, P[2]));
                let yp = yp.mla(x, f32x4::splat_t(token, P[1]));
                let yp = yp.mla(x, f32x4::splat_t(token, P[0]));
                let yq = f32x4::splat_t(token, Q[6]).mla(x, f32x4::splat_t(token, Q[5]));
                let yq = yq.mla(x, f32x4::splat_t(token, Q[4]));
                let yq = yq.mla(x, f32x4::splat_t(token, Q[3]));
                let yq = yq.mla(x, f32x4::splat_t(token, Q[2]));
                let yq = yq.mla(x, f32x4::splat_t(token, Q[1]));
                let yq = yq.mla(x, f32x4::splat_t(token, Q[0]));
                let power_result = yp / yq;
                let thresh_mask = abs_v
                    .simd_lt(f32x4::splat_t(token, SRGB_LINEAR_THRESHOLD));
                let result = f32x4::blend(thresh_mask, linear_result, power_result);
                let result = f32x4::blend(neg_mask, -result, result);
                result.to_array()
            }
            #[inline(always)]
            pub(crate) fn linear_to_srgb_extended<T: F32x4Backend>(
                token: T,
                linear: [f32; 4],
            ) -> [f32; 4] {
                use crate::rational_poly::{EXT_L2S_P as P, EXT_L2S_Q as Q};
                let zero = f32x4::zero_t(token);
                let v = f32x4::from_array_t(token, linear);
                let neg_mask = v.simd_lt(zero);
                let abs_v = v.abs();
                let linear_result = abs_v * f32x4::splat_t(token, TWELVE_92);
                let x = abs_v.sqrt();
                let yp = f32x4::splat_t(token, P[6]).mla(x, f32x4::splat_t(token, P[5]));
                let yp = yp.mla(x, f32x4::splat_t(token, P[4]));
                let yp = yp.mla(x, f32x4::splat_t(token, P[3]));
                let yp = yp.mla(x, f32x4::splat_t(token, P[2]));
                let yp = yp.mla(x, f32x4::splat_t(token, P[1]));
                let yp = yp.mla(x, f32x4::splat_t(token, P[0]));
                let yq = f32x4::splat_t(token, Q[6]).mla(x, f32x4::splat_t(token, Q[5]));
                let yq = yq.mla(x, f32x4::splat_t(token, Q[4]));
                let yq = yq.mla(x, f32x4::splat_t(token, Q[3]));
                let yq = yq.mla(x, f32x4::splat_t(token, Q[2]));
                let yq = yq.mla(x, f32x4::splat_t(token, Q[1]));
                let yq = yq.mla(x, f32x4::splat_t(token, Q[0]));
                let power_result = yp / yq;
                let thresh_mask = abs_v.simd_lt(f32x4::splat_t(token, LINEAR_THRESHOLD));
                let result = f32x4::blend(thresh_mask, linear_result, power_result);
                let result = f32x4::blend(neg_mask, -result, result);
                result.to_array()
            }
        }
        pub(crate) mod x16 {
            use super::*;
            #[inline(always)]
            pub(crate) fn srgb_to_linear<T: F32x16Backend>(
                token: T,
                srgb: [f32; 16],
            ) -> [f32; 16] {
                use crate::rational_poly::{S2L_P, S2L_Q};
                let zero = f32x16::<T>::zero_t(token);
                let one = f32x16::<T>::splat_t(token, 1.0);
                let v = f32x16::<T>::from_array_t(token, srgb);
                let clamped = v.max(zero).min(one);
                let linear_result = clamped * f32x16::<T>::splat_t(token, LINEAR_SCALE);
                let x = clamped;
                let yp = f32x16::<T>::splat_t(token, S2L_P[4])
                    .mla(x, f32x16::<T>::splat_t(token, S2L_P[3]));
                let yp = yp.mla(x, f32x16::<T>::splat_t(token, S2L_P[2]));
                let yp = yp.mla(x, f32x16::<T>::splat_t(token, S2L_P[1]));
                let yp = yp.mla(x, f32x16::<T>::splat_t(token, S2L_P[0]));
                let yq = f32x16::<T>::splat_t(token, S2L_Q[4])
                    .mla(x, f32x16::<T>::splat_t(token, S2L_Q[3]));
                let yq = yq.mla(x, f32x16::<T>::splat_t(token, S2L_Q[2]));
                let yq = yq.mla(x, f32x16::<T>::splat_t(token, S2L_Q[1]));
                let yq = yq.mla(x, f32x16::<T>::splat_t(token, S2L_Q[0]));
                let power_result = (yp / yq).min(one);
                let mask = clamped
                    .simd_lt(f32x16::<T>::splat_t(token, SRGB_LINEAR_THRESHOLD));
                let result = f32x16::<T>::blend(mask, linear_result, power_result);
                let ge_one = v.simd_ge(one);
                f32x16::<T>::blend(ge_one, one, result).to_array()
            }
            #[inline(always)]
            pub(crate) fn linear_to_srgb<T: F32x16Backend>(
                token: T,
                linear: [f32; 16],
            ) -> [f32; 16] {
                use crate::rational_poly::{L2S_P, L2S_Q};
                let zero = f32x16::<T>::zero_t(token);
                let one = f32x16::<T>::splat_t(token, 1.0);
                let v = f32x16::<T>::from_array_t(token, linear);
                let clamped = v.max(zero).min(one);
                let linear_result = clamped * f32x16::<T>::splat_t(token, TWELVE_92);
                let x = clamped.sqrt();
                let yp = f32x16::<T>::splat_t(token, L2S_P[4])
                    .mla(x, f32x16::<T>::splat_t(token, L2S_P[3]));
                let yp = yp.mla(x, f32x16::<T>::splat_t(token, L2S_P[2]));
                let yp = yp.mla(x, f32x16::<T>::splat_t(token, L2S_P[1]));
                let yp = yp.mla(x, f32x16::<T>::splat_t(token, L2S_P[0]));
                let yq = f32x16::<T>::splat_t(token, L2S_Q[4])
                    .mla(x, f32x16::<T>::splat_t(token, L2S_Q[3]));
                let yq = yq.mla(x, f32x16::<T>::splat_t(token, L2S_Q[2]));
                let yq = yq.mla(x, f32x16::<T>::splat_t(token, L2S_Q[1]));
                let yq = yq.mla(x, f32x16::<T>::splat_t(token, L2S_Q[0]));
                let power_result = (yp / yq).min(one);
                let mask = clamped
                    .simd_lt(f32x16::<T>::splat_t(token, LINEAR_THRESHOLD));
                let result = f32x16::<T>::blend(mask, linear_result, power_result);
                let ge_one = v.simd_ge(one);
                f32x16::<T>::blend(ge_one, one, result).to_array()
            }
            #[inline(always)]
            pub(crate) fn gamma_to_linear<T: F32x16Convert>(
                token: T,
                encoded: [f32; 16],
                gamma: f32,
            ) -> [f32; 16] {
                let zero = f32x16::<T>::zero_t(token);
                let one = f32x16::<T>::splat_t(token, 1.0);
                let v = f32x16::<T>::from_array_t(token, encoded).max(zero).min(one);
                crate::tf::fast_math::pow_x16(token, v, gamma).to_array()
            }
            #[inline(always)]
            pub(crate) fn linear_to_gamma<T: F32x16Convert>(
                token: T,
                linear: [f32; 16],
                gamma: f32,
            ) -> [f32; 16] {
                let zero = f32x16::<T>::zero_t(token);
                let one = f32x16::<T>::splat_t(token, 1.0);
                let v = f32x16::<T>::from_array_t(token, linear).max(zero).min(one);
                crate::tf::fast_math::pow_x16(token, v, 1.0 / gamma).to_array()
            }
            #[inline(always)]
            pub(crate) fn srgb_to_linear_extended<T: F32x16Backend>(
                token: T,
                srgb: [f32; 16],
            ) -> [f32; 16] {
                use crate::rational_poly::{EXT_S2L_P as P, EXT_S2L_Q as Q};
                let zero = f32x16::zero_t(token);
                let v = f32x16::from_array_t(token, srgb);
                let neg_mask = v.simd_lt(zero);
                let abs_v = v.abs();
                let linear_result = abs_v * f32x16::splat_t(token, LINEAR_SCALE);
                let x = abs_v;
                let yp = f32x16::splat_t(token, P[6])
                    .mla(x, f32x16::splat_t(token, P[5]));
                let yp = yp.mla(x, f32x16::splat_t(token, P[4]));
                let yp = yp.mla(x, f32x16::splat_t(token, P[3]));
                let yp = yp.mla(x, f32x16::splat_t(token, P[2]));
                let yp = yp.mla(x, f32x16::splat_t(token, P[1]));
                let yp = yp.mla(x, f32x16::splat_t(token, P[0]));
                let yq = f32x16::splat_t(token, Q[6])
                    .mla(x, f32x16::splat_t(token, Q[5]));
                let yq = yq.mla(x, f32x16::splat_t(token, Q[4]));
                let yq = yq.mla(x, f32x16::splat_t(token, Q[3]));
                let yq = yq.mla(x, f32x16::splat_t(token, Q[2]));
                let yq = yq.mla(x, f32x16::splat_t(token, Q[1]));
                let yq = yq.mla(x, f32x16::splat_t(token, Q[0]));
                let power_result = yp / yq;
                let thresh_mask = abs_v
                    .simd_lt(f32x16::splat_t(token, SRGB_LINEAR_THRESHOLD));
                let result = f32x16::blend(thresh_mask, linear_result, power_result);
                let result = f32x16::blend(neg_mask, -result, result);
                result.to_array()
            }
            #[inline(always)]
            pub(crate) fn linear_to_srgb_extended<T: F32x16Backend>(
                token: T,
                linear: [f32; 16],
            ) -> [f32; 16] {
                use crate::rational_poly::{EXT_L2S_P as P, EXT_L2S_Q as Q};
                let zero = f32x16::zero_t(token);
                let v = f32x16::from_array_t(token, linear);
                let neg_mask = v.simd_lt(zero);
                let abs_v = v.abs();
                let linear_result = abs_v * f32x16::splat_t(token, TWELVE_92);
                let x = abs_v.sqrt();
                let yp = f32x16::splat_t(token, P[6])
                    .mla(x, f32x16::splat_t(token, P[5]));
                let yp = yp.mla(x, f32x16::splat_t(token, P[4]));
                let yp = yp.mla(x, f32x16::splat_t(token, P[3]));
                let yp = yp.mla(x, f32x16::splat_t(token, P[2]));
                let yp = yp.mla(x, f32x16::splat_t(token, P[1]));
                let yp = yp.mla(x, f32x16::splat_t(token, P[0]));
                let yq = f32x16::splat_t(token, Q[6])
                    .mla(x, f32x16::splat_t(token, Q[5]));
                let yq = yq.mla(x, f32x16::splat_t(token, Q[4]));
                let yq = yq.mla(x, f32x16::splat_t(token, Q[3]));
                let yq = yq.mla(x, f32x16::splat_t(token, Q[2]));
                let yq = yq.mla(x, f32x16::splat_t(token, Q[1]));
                let yq = yq.mla(x, f32x16::splat_t(token, Q[0]));
                let power_result = yp / yq;
                let thresh_mask = abs_v
                    .simd_lt(f32x16::splat_t(token, LINEAR_THRESHOLD));
                let result = f32x16::blend(thresh_mask, linear_result, power_result);
                let result = f32x16::blend(neg_mask, -result, result);
                result.to_array()
            }
        }
    }
    pub(crate) mod x4 {
        //! Private 4-lane kernels used by the canonical CPU-feature entry points.
        use super::float::x4::{
            gamma_to_linear as gamma_to_linear_core,
            linear_to_gamma as linear_to_gamma_core,
            linear_to_srgb as linear_to_srgb_core, srgb_to_linear as srgb_to_linear_core,
        };
        use archmage::attune;
        use archmage::Wasm128Token;
        use magetypes::simd::generic::f32x4 as gen_f32x4;
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 4 sRGB values to linear without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted to \[0, 8\]. sampled max 8 ULP in \[0,1\], u16-safe to ~4.2.
        pub(crate) fn srgb_to_linear_extended_wasm128(
            token: ::archmage::Wasm128Token,
            srgb: [f32; 4],
        ) -> [f32; 4] {
            super::float::x4::srgb_to_linear_extended(token, srgb)
        }
        #[inline]
        /// Convert 4 sRGB values to linear without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted to \[0, 8\]. sampled max 8 ULP in \[0,1\], u16-safe to ~4.2.
        pub(crate) fn srgb_to_linear_extended_scalar(
            token: ::archmage::ScalarToken,
            srgb: [f32; 4],
        ) -> [f32; 4] {
            super::float::x4::srgb_to_linear_extended(token, srgb)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 4 linear values to sRGB without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted on √x to \[0, 64\]. sampled max 7 ULP in \[0,1\], u16-safe to 64.
        pub(crate) fn linear_to_srgb_extended_wasm128(
            token: ::archmage::Wasm128Token,
            linear: [f32; 4],
        ) -> [f32; 4] {
            super::float::x4::linear_to_srgb_extended(token, linear)
        }
        #[inline]
        /// Convert 4 linear values to sRGB without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted on √x to \[0, 64\]. sampled max 7 ULP in \[0,1\], u16-safe to 64.
        pub(crate) fn linear_to_srgb_extended_scalar(
            token: ::archmage::ScalarToken,
            linear: [f32; 4],
        ) -> [f32; 4] {
            super::float::x4::linear_to_srgb_extended(token, linear)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 4 sRGB values to linear. Input clamped to \[0, 1\].
        ///
        /// The `Wasm128Token` parameter proves SIMD128 support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn srgb_to_linear_wasm128(
            token: Wasm128Token,
            srgb: [f32; 4],
        ) -> [f32; 4] {
            srgb_to_linear_core(token, srgb)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_srgb_wasm128(
            token: Wasm128Token,
            linear: [f32; 4],
        ) -> [f32; 4] {
            linear_to_srgb_core(token, linear)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn gamma_to_linear_wasm128(
            token: Wasm128Token,
            encoded: [f32; 4],
            gamma: f32,
        ) -> [f32; 4] {
            gamma_to_linear_core(token, encoded, gamma)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_gamma_wasm128(
            token: Wasm128Token,
            linear: [f32; 4],
            gamma: f32,
        ) -> [f32; 4] {
            linear_to_gamma_core(token, linear, gamma)
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert sRGB f32 values to linear in-place using 4-wide WASM SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn srgb_to_linear_slice_wasm128(
            token: Wasm128Token,
            values: &mut [f32],
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = srgb_to_linear_core(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = srgb_to_linear_core(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert linear f32 values to sRGB in-place using 4-wide WASM SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_srgb_slice_wasm128(
            token: Wasm128Token,
            values: &mut [f32],
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = linear_to_srgb_core(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_srgb_core(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert gamma-encoded f32 values to linear in-place using 4-wide WASM SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn gamma_to_linear_slice_wasm128(
            token: Wasm128Token,
            values: &mut [f32],
            gamma: f32,
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = gamma_to_linear_core(token, *chunk, gamma);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = gamma_to_linear_core(token, padded, gamma);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Convert linear f32 values to gamma-encoded in-place using 4-wide WASM SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_gamma_slice_wasm128(
            token: Wasm128Token,
            values: &mut [f32],
            gamma: f32,
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = linear_to_gamma_core(token, *chunk, gamma);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_gamma_core(token, padded, gamma);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn tf_srgb_to_linear_wasm128(
            token: Wasm128Token,
            v: [f32; 4],
        ) -> [f32; 4] {
            crate::tf::srgb::srgb_to_linear_x4(token, gen_f32x4::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn tf_linear_to_srgb_wasm128(
            token: Wasm128Token,
            v: [f32; 4],
        ) -> [f32; 4] {
            crate::tf::srgb::linear_to_srgb_x4(token, gen_f32x4::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn bt709_to_linear_wasm128(
            token: Wasm128Token,
            v: [f32; 4],
        ) -> [f32; 4] {
            crate::tf::bt709::bt709_to_linear_x4(
                    token,
                    gen_f32x4::from_array_t(token, v),
                )
                .to_array()
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn linear_to_bt709_wasm128(
            token: Wasm128Token,
            v: [f32; 4],
        ) -> [f32; 4] {
            crate::tf::bt709::linear_to_bt709_x4(
                    token,
                    gen_f32x4::from_array_t(token, v),
                )
                .to_array()
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn pq_to_linear_wasm128(
            token: Wasm128Token,
            v: [f32; 4],
        ) -> [f32; 4] {
            crate::tf::pq::pq_to_linear_x4(token, gen_f32x4::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn linear_to_pq_wasm128(
            token: Wasm128Token,
            v: [f32; 4],
        ) -> [f32; 4] {
            crate::tf::pq::linear_to_pq_x4(token, gen_f32x4::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn hlg_to_linear_wasm128(
            token: Wasm128Token,
            v: [f32; 4],
        ) -> [f32; 4] {
            crate::tf::hlg::hlg_to_linear_x4(token, gen_f32x4::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn linear_to_hlg_wasm128(
            token: Wasm128Token,
            v: [f32; 4],
        ) -> [f32; 4] {
            crate::tf::hlg::linear_to_hlg_x4(token, gen_f32x4::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn tf_srgb_to_linear_slice_wasm128(
            token: Wasm128Token,
            values: &mut [f32],
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = tf_srgb_to_linear_wasm128(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = tf_srgb_to_linear_wasm128(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn tf_linear_to_srgb_slice_wasm128(
            token: Wasm128Token,
            values: &mut [f32],
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = tf_linear_to_srgb_wasm128(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = tf_linear_to_srgb_wasm128(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn bt709_to_linear_slice_wasm128(
            token: Wasm128Token,
            values: &mut [f32],
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = bt709_to_linear_wasm128(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = bt709_to_linear_wasm128(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn linear_to_bt709_slice_wasm128(
            token: Wasm128Token,
            values: &mut [f32],
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = linear_to_bt709_wasm128(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_bt709_wasm128(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn pq_to_linear_slice_wasm128(
            token: Wasm128Token,
            values: &mut [f32],
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = pq_to_linear_wasm128(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = pq_to_linear_wasm128(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn linear_to_pq_slice_wasm128(
            token: Wasm128Token,
            values: &mut [f32],
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = linear_to_pq_wasm128(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_pq_wasm128(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn hlg_to_linear_slice_wasm128(
            token: Wasm128Token,
            values: &mut [f32],
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = hlg_to_linear_wasm128(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = hlg_to_linear_wasm128(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(enable = "simd128")]
        #[inline]
        /// Transfer function slice rite (4×f32, WASM SIMD128). Requires `transfer` feature.
        pub(crate) fn linear_to_hlg_slice_wasm128(
            token: Wasm128Token,
            values: &mut [f32],
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = linear_to_hlg_wasm128(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_hlg_wasm128(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
    }
    /// A slice tail padded to one vector block, or `None` for an empty tail.
    ///
    /// Tails run through the same kernel as the body, so a value converts to the same bits
    /// wherever it sits in a slice and however a row is split.
    #[inline(always)]
    pub(crate) fn tail_block<const N: usize>(tail: &[f32]) -> Option<[f32; N]> {
        if tail.is_empty() {
            return None;
        }
        let mut block = [0.0; N];
        block[..tail.len()].copy_from_slice(tail);
        Some(block)
    }
    /// Writes a converted [`tail_block`] back.
    #[inline(always)]
    pub(crate) fn store_tail<const N: usize>(
        tail: &mut [f32],
        block: Option<&[f32; N]>,
    ) {
        if let Some(block) = block {
            tail.copy_from_slice(&block[..tail.len()]);
        }
    }
    /// Apply one array kernel to full blocks and one padded tail.
    #[inline(always)]
    pub(crate) fn map_slice<const N: usize>(
        values: &mut [f32],
        mut convert: impl FnMut([f32; N]) -> [f32; N],
    ) {
        let (chunks, remainder) = values.as_chunks_mut::<N>();
        let mut tail = tail_block::<N>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            *chunk = convert(*chunk);
        }
        store_tail(remainder, tail.as_ref());
    }
    /// Keep scalar HDR arrays on the same arithmetic policy as their SIMD peers.
    #[inline(always)]
    pub(crate) fn scalar_array<const N: usize>(
        values: [f32; N],
        vector: impl FnOnce([f32; N]) -> [f32; N],
        fused: impl FnMut(f32) -> f32,
    ) -> [f32; N] {
        if crate::tf::fast_math::USE_FUSED_MATH {
            values.map(fused)
        } else {
            vector(values)
        }
    }
    /// Preserve per-sample fused scalar loops and relaxed-WASM padded vector tails.
    #[inline(always)]
    pub(crate) fn scalar_slice<const N: usize>(
        values: &mut [f32],
        vector: impl FnMut([f32; N]) -> [f32; N],
        mut fused: impl FnMut(f32) -> f32,
    ) {
        if crate::tf::fast_math::USE_FUSED_MATH {
            for value in values {
                *value = fused(*value);
            }
        } else {
            map_slice::<N>(values, vector);
        }
    }
}
mod lanes {
    //! One sample through the same operations as the vector kernels, with every multiply-add fused.
    //!
    //! Native and precise-portable scalar tiers use these. Default WASM uses the relaxed
    //! generic vector kernels for float entry points and slice tails. These are
    //! crate-private kernels written out per lane. Keep their operations paired with
    //! the private vector copies; `tests/tier_parity.rs` holds them together.
    #[allow(unused_imports)]
    use num_traits::Float;
    use crate::rational_poly::{
        EXT_L2S_P, EXT_L2S_Q, EXT_S2L_P, EXT_S2L_Q, LINEAR_SCALE, S2L_P, S2L_Q,
        SRGB_THRESHOLD,
    };
    use crate::rational_poly::{L2S_P, L2S_Q, LINEAR_THRESHOLD, TWELVE_92};
    use crate::tf::fast_math;
    use crate::tf::fast_math::eval_rational_poly;
    use crate::tf::fast_math::fast_log2f;
    /// `max(v, 0)` as the kernels do it: zero for negative zero and NaN too.
    #[inline(always)]
    fn non_negative(v: f32) -> f32 {
        if v > 0.0 { v } else { 0.0 }
    }
    /// `max(min(v, 1), 0)` as the kernels do it.
    #[inline(always)]
    fn unit(v: f32) -> f32 {
        let v = non_negative(v);
        if v < 1.0 { v } else { 1.0 }
    }
    /// `2^x` without the scalar function's clamp.
    #[inline(always)]
    fn pow2(x: f32) -> f32 {
        let floor = x.floor();
        let frac = x - floor;
        let exp = f32::from_bits(((floor as i32).wrapping_add(127) as u32) << 23);
        let num = frac + fast_math::POW2_NUM[0];
        let num = num.mul_add(frac, fast_math::POW2_NUM[1]);
        let num = num.mul_add(frac, fast_math::POW2_NUM[2]) * exp;
        let den = fast_math::POW2_DEN[0].mul_add(frac, fast_math::POW2_DEN[1]);
        let den = den.mul_add(frac, fast_math::POW2_DEN[2]);
        let den = den.mul_add(frac, fast_math::POW2_DEN[3]);
        num / den
    }
    /// The `pow` of [`fast_math::POW_LOG2`] and [`fast_math::POW_EXP2`], for the vector `pow_*`.
    #[inline(always)]
    pub(crate) fn pow(x: f32, n: f32) -> f32 {
        exp2(log2(x) * n)
    }
    #[inline(always)]
    fn log2(x: f32) -> f32 {
        use fast_math::POW_LOG2 as C;
        let adjusted = (x.to_bits() as i32)
            .wrapping_add((0x3f80_0000u32 - fast_math::POW_SQRT1_2) as i32);
        let n = ((adjusted >> 23) - 127) as f32;
        let a = f32::from_bits(
            ((adjusted & 0x007f_ffff).wrapping_add(fast_math::POW_SQRT1_2 as i32)) as u32,
        );
        let y = (a - 1.0) / (a + 1.0);
        let y2 = y * y;
        let poly = C[3].mul_add(y2, C[2]);
        let poly = poly.mul_add(y2, C[1]);
        let poly = poly.mul_add(y2, C[0]);
        let result = poly.mul_add(y, n);
        if x == 0.0 {
            f32::NEG_INFINITY
        } else if x < 0.0 {
            f32::NAN
        } else if x == f32::INFINITY {
            f32::INFINITY
        } else {
            result
        }
    }
    /// Round to nearest with ties to even, like the vector `round`.
    #[inline(always)]
    fn round_even(x: f32) -> f32 {
        let floor = x.floor();
        let frac = x - floor;
        if frac < 0.5 {
            floor
        } else if frac > 0.5 || (floor as i32) & 1 != 0 {
            floor + 1.0
        } else {
            floor
        }
    }
    #[inline(always)]
    fn exp2(x: f32) -> f32 {
        use fast_math::POW_EXP2 as C;
        if x >= 128.0 {
            return f32::INFINITY;
        }
        if x < -126.0 {
            return 0.0;
        }
        let xi = round_even(x);
        let xi = if xi < 127.0 { xi } else { 127.0 };
        let xf = x - xi;
        let poly = C[6].mul_add(xf, C[5]);
        let poly = poly.mul_add(xf, C[4]);
        let poly = poly.mul_add(xf, C[3]);
        let poly = poly.mul_add(xf, C[2]);
        let poly = poly.mul_add(xf, C[1]);
        let poly = poly.mul_add(xf, C[0]);
        poly * f32::from_bits(((xi as i32).wrapping_add(127) as u32) << 23)
    }
    pub(crate) fn srgb_to_linear(v: f32) -> f32 {
        let clamped = unit(v);
        if v >= 1.0 {
            return 1.0;
        }
        if clamped < SRGB_THRESHOLD {
            return clamped * LINEAR_SCALE;
        }
        let power = eval_rational_poly(clamped, S2L_P, S2L_Q);
        if power < 1.0 { power } else { 1.0 }
    }
    pub(crate) fn linear_to_srgb(v: f32) -> f32 {
        let clamped = unit(v);
        if v >= 1.0 {
            return 1.0;
        }
        if clamped < LINEAR_THRESHOLD {
            return clamped * TWELVE_92;
        }
        let power = eval_rational_poly(clamped.sqrt(), L2S_P, L2S_Q);
        if power < 1.0 { power } else { 1.0 }
    }
    pub(crate) fn linear_to_srgb_u16(v: f32) -> u16 {
        linear_to_srgb(v).mul_add(65535.0, 0.5) as i32 as u16
    }
    pub(crate) fn bt709_to_linear(v: f32) -> f32 {
        use crate::tf::bt709::{BT709_ALPHA, BT709_BETA};
        if v < 4.5 * BT709_BETA {
            return v * (1.0 / 4.5);
        }
        let normalized = (v + BT709_ALPHA) / (1.0 + BT709_ALPHA);
        let power = pow2(fast_log2f(normalized) * (1.0 / 0.45));
        let seam = (4.5 * BT709_BETA) * (1.0 / 4.5);
        if power > seam { power } else { seam }
    }
    pub(crate) fn linear_to_bt709(v: f32) -> f32 {
        use crate::tf::bt709::{BT709_ALPHA, BT709_BETA};
        if v < BT709_BETA {
            return v * 4.5;
        }
        let power = pow2(fast_log2f(v) * 0.45);
        (1.0 + BT709_ALPHA).mul_add(power, -BT709_ALPHA)
    }
    pub(crate) fn hlg_to_linear(v: f32) -> f32 {
        use crate::tf::hlg::{HLG_B, HLG_C, HLG_INV_A_LOG2E};
        let a = non_negative(v);
        if a <= 0.5 {
            return (a * a) * (1.0 / 3.0);
        }
        (pow2((a - HLG_C) * HLG_INV_A_LOG2E) + HLG_B) * (1.0 / 12.0)
    }
    pub(crate) fn linear_to_hlg(v: f32) -> f32 {
        use crate::tf::hlg::{HLG_A_INV_LOG2E, HLG_B, HLG_C};
        let a = non_negative(v);
        if a <= 1.0 / 12.0 {
            return (3.0 * a).sqrt();
        }
        let high = HLG_A_INV_LOG2E.mul_add(fast_log2f(12.0 * a - HLG_B), HLG_C);
        if high > 0.5 { high } else { 0.5 }
    }
    pub(crate) fn pq_to_linear(v: f32) -> f32 {
        use crate::tf::pq::{
            C1, C2, C3, EOTF_POLY_FLOOR, M1, M2, PQ_EOTF_P_LARGE, PQ_EOTF_P_SMALL,
            PQ_EOTF_Q_LARGE, PQ_EOTF_Q_SMALL,
        };
        if v.is_nan() || v <= 0.0 {
            return 0.0;
        }
        if v < EOTF_POLY_FLOOR {
            let vp = pow(v, 1.0 / M2);
            let num = non_negative(vp - C1);
            let den = C2 - C3 * vp;
            return pow(num / den, 1.0 / M1);
        }
        let x = v.mul_add(v, v);
        if v < 0.25 {
            eval_rational_poly(x, PQ_EOTF_P_SMALL, PQ_EOTF_Q_SMALL)
        } else {
            eval_rational_poly(x, PQ_EOTF_P_LARGE, PQ_EOTF_Q_LARGE)
        }
    }
    pub(crate) fn linear_to_pq(v: f32) -> f32 {
        use crate::tf::pq::{
            C1, C2, C3, M1, M2, OETF_POLY_FLOOR, PQ_INV_P_LARGE, PQ_INV_P_SMALL,
            PQ_INV_Q_LARGE, PQ_INV_Q_SMALL,
        };
        if v.is_nan() || v <= 0.0 {
            return 0.0;
        }
        if v < OETF_POLY_FLOOR {
            let vp = pow(v, M1);
            return pow((C1 + C2 * vp) / (1.0 + C3 * vp), M2);
        }
        let a4 = v.sqrt().sqrt();
        if a4 < 0.1 {
            eval_rational_poly(a4, PQ_INV_P_SMALL, PQ_INV_Q_SMALL)
        } else {
            eval_rational_poly(a4, PQ_INV_P_LARGE, PQ_INV_Q_LARGE)
        }
    }
    pub(crate) fn srgb_to_linear_extended(v: f32) -> f32 {
        let magnitude = v.abs();
        let result = if magnitude < SRGB_THRESHOLD {
            magnitude * LINEAR_SCALE
        } else {
            eval_rational_poly(magnitude, EXT_S2L_P, EXT_S2L_Q)
        };
        if v < 0.0 { -result } else { result }
    }
    pub(crate) fn linear_to_srgb_extended(v: f32) -> f32 {
        let magnitude = v.abs();
        let result = if magnitude < LINEAR_THRESHOLD {
            magnitude * TWELVE_92
        } else {
            eval_rational_poly(magnitude.sqrt(), EXT_L2S_P, EXT_L2S_Q)
        };
        if v < 0.0 { -result } else { result }
    }
    pub(crate) fn gamma_to_linear(v: f32, gamma: f32) -> f32 {
        pow(unit(v), gamma)
    }
    pub(crate) fn linear_to_gamma(v: f32, gamma: f32) -> f32 {
        pow(unit(v), 1.0 / gamma)
    }
}
pub(crate) mod scalar {
    //! Scalar (single-value) sRGB conversion functions.
    //!
    //! Implements the sRGB transfer function with C0-continuous (moxcms) constants
    //! that eliminate the tiny discontinuity in the IEC 61966-2-1 textbook version.
    //! Optimizations:
    //! - Piecewise function avoids pow() for ~1.2% of values in the linear segment
    //! - Early exit for out-of-range values (0 and 1) avoids transcendentals
    //! - FMA instructions for the encoding formula
    use crate::mlaf::fmla;
    #[allow(unused_imports)]
    use num_traits::Float;
    /// Linear threshold for linearization: 12.92 * 0.0030412825601275209 ≈ 0.03929
    const SRGB_LINEAR_THRESHOLD: f64 = 12.92 * 0.003_041_282_560_127_521;
    pub(crate) const SRGB_LINEAR_THRESHOLD_F32: f32 = SRGB_LINEAR_THRESHOLD as f32;
    /// Linear threshold for encoding (the inverse cutoff point)
    const LINEAR_THRESHOLD: f64 = 0.003_041_282_560_127_521;
    pub(crate) const LINEAR_THRESHOLD_F32: f32 = LINEAR_THRESHOLD as f32;
    /// Linear scale factor (1/12.92)
    const LINEAR_SCALE: f64 = 1.0 / 12.92;
    pub(crate) const LINEAR_SCALE_F32: f32 = LINEAR_SCALE as f32;
    /// sRGB encoding constants
    const SRGB_A: f64 = 0.055_010_718_947_586_6;
    const SRGB_A_F32: f32 = SRGB_A as f32;
    const SRGB_A_PLUS_1: f64 = 1.055_010_718_947_586_6;
    const SRGB_A_PLUS_1_F32: f32 = SRGB_A_PLUS_1 as f32;
    /// Gamma exponent
    const GAMMA: f64 = 2.4;
    const INV_GAMMA: f64 = 1.0 / GAMMA;
    const INV_GAMMA_F32: f32 = INV_GAMMA as f32;
    /// Convert sRGB gamma-encoded value to linear light (f64).
    ///
    /// Input: sRGB value in \[0, 1\]
    /// Output: Linear light value in \[0, 1\]
    ///
    /// **Clamps** inputs to \[0, 1\]. No extended-range f64 variant exists.
    #[inline]
    pub fn srgb_to_linear_f64(gamma: f64) -> f64 {
        if gamma < 0.0 {
            0.0
        } else if gamma < SRGB_LINEAR_THRESHOLD {
            gamma * LINEAR_SCALE
        } else if gamma < 1.0 {
            ((gamma + SRGB_A) / SRGB_A_PLUS_1).powf(GAMMA)
        } else {
            1.0
        }
    }
    /// Convert sRGB gamma-encoded value to linear light (f32).
    ///
    /// Input: sRGB value in \[0, 1\]
    /// Output: Linear light value in \[0, 1\]
    ///
    /// **Clamps** inputs to \[0, 1\]. For HDR/ICC workflows with out-of-range
    /// values, use [`srgb_to_linear_extended`] instead.
    #[inline]
    pub fn srgb_to_linear(gamma: f32) -> f32 {
        if gamma < 0.0 {
            0.0
        } else if gamma < SRGB_LINEAR_THRESHOLD_F32 {
            gamma * LINEAR_SCALE_F32
        } else if gamma < 1.0 {
            ((gamma + SRGB_A_F32) / SRGB_A_PLUS_1_F32).powf(GAMMA as f32)
        } else {
            1.0
        }
    }
    /// Convert linear light value to sRGB gamma-encoded (f64).
    ///
    /// Input: Linear light value in \[0, 1\]
    /// Output: sRGB value in \[0, 1\]
    ///
    /// **Clamps** inputs to \[0, 1\]. No extended-range f64 variant exists.
    #[inline]
    pub fn linear_to_srgb_f64(linear: f64) -> f64 {
        if linear < 0.0 {
            0.0
        } else if linear < LINEAR_THRESHOLD {
            linear * 12.92
        } else if linear < 1.0 {
            fmla(SRGB_A_PLUS_1, linear.powf(INV_GAMMA), -SRGB_A)
        } else {
            1.0
        }
    }
    /// Convert linear light value to sRGB gamma-encoded (f32).
    ///
    /// Input: Linear light value in \[0, 1\]
    /// Output: sRGB value in \[0, 1\]
    ///
    /// **Clamps** inputs to \[0, 1\]. For HDR/ICC workflows with out-of-range
    /// values, use [`linear_to_srgb_extended`] instead.
    #[inline]
    pub fn linear_to_srgb(linear: f32) -> f32 {
        if linear < 0.0 {
            0.0
        } else if linear < LINEAR_THRESHOLD_F32 {
            linear * 12.92
        } else if linear < 1.0 {
            fmla(SRGB_A_PLUS_1_F32, linear.powf(INV_GAMMA_F32), -SRGB_A_F32)
        } else {
            1.0
        }
    }
    /// Convert sRGB gamma-encoded value to linear light using a rational polynomial (f32).
    ///
    /// Same as [`srgb_to_linear`] but replaces `powf()` with a 5/5 rational
    /// polynomial (Horner's method). Faster than powf on all platforms and
    /// identical to the SIMD path.
    ///
    /// Max error vs f64 reference (exhaustive over all f32):
    /// - Full power segment: ~8 ULP max, ~1 ULP avg
    ///
    /// Coefficients from libjxl (BSD-3-Clause).
    ///
    /// **Clamps** inputs to \[0, 1\].
    #[inline]
    #[allow(dead_code)]
    pub fn srgb_to_linear_fast(gamma: f32) -> f32 {
        crate::rational_poly::srgb_to_linear_fast(gamma)
    }
    /// Convert linear light value to sRGB gamma-encoded using a rational polynomial (f32).
    ///
    /// Same as [`linear_to_srgb`] but replaces `powf()` with sqrt + 5/5 rational
    /// polynomial (Horner's method). Faster than powf on all platforms and
    /// identical to the SIMD path.
    ///
    /// Max error vs f64 reference (exhaustive over all f32):
    /// - Full power segment: ~8 ULP max, ~1 ULP avg
    ///
    /// Coefficients from libjxl (BSD-3-Clause).
    ///
    /// **Clamps** inputs to \[0, 1\].
    #[inline]
    #[allow(dead_code)]
    pub fn linear_to_srgb_fast(linear: f32) -> f32 {
        crate::rational_poly::linear_to_srgb_fast(linear)
    }
    /// Convert sRGB gamma-encoded value to linear light without clamping (f32).
    ///
    /// Unlike [`srgb_to_linear`], this does **not** clamp to \[0, 1\]. Use this for:
    /// - **HDR content** where values exceed 1.0
    /// - **Cross-gamut conversion** (P3 → sRGB, BT.2020 → sRGB) where 3×3 matrix
    ///   output may be negative
    /// - **Scene-referred** workflows with unbounded linear light
    ///
    /// Uses sign-preserving extension per CSS Color 4: `sign(v) * eotf(|v|)`.
    /// The power curve is applied to the magnitude and the sign is restored.
    #[inline]
    pub fn srgb_to_linear_extended(gamma: f32) -> f32 {
        let sign = gamma.signum();
        let abs_v = gamma.abs();
        if abs_v < SRGB_LINEAR_THRESHOLD_F32 {
            gamma * LINEAR_SCALE_F32
        } else {
            sign * ((abs_v + SRGB_A_F32) / SRGB_A_PLUS_1_F32).powf(GAMMA as f32)
        }
    }
    /// Convert linear light value to sRGB gamma-encoded without clamping (f32).
    ///
    /// Unlike [`linear_to_srgb`], this does **not** clamp to \[0, 1\]. Use this for:
    /// - **HDR content** where linear values exceed 1.0
    /// - **Cross-gamut conversion** (P3 → sRGB, BT.2020 → sRGB) where 3×3 matrix
    ///   output may be negative
    /// - **Scene-referred** workflows with unbounded linear light
    ///
    /// Uses sign-preserving extension per CSS Color 4: `sign(v) * oetf(|v|)`.
    /// The power curve is applied to the magnitude and the sign is restored.
    #[inline]
    pub fn linear_to_srgb_extended(linear: f32) -> f32 {
        let sign = linear.signum();
        let abs_v = linear.abs();
        if abs_v < LINEAR_THRESHOLD_F32 {
            linear * 12.92
        } else {
            sign * fmla(SRGB_A_PLUS_1_F32, abs_v.powf(INV_GAMMA_F32), -SRGB_A_F32)
        }
    }
    /// Convert linear to 8-bit sRGB using const LUT.
    ///
    /// Uses a 4096-entry lookup table (4KB, fits L1 cache). No transcendental math.
    /// Bitmask indexing (`& 0xFFF`) guarantees bounds safety with zero overhead.
    /// Max error: ±1 u8 level vs exact computation.
    #[inline]
    pub fn linear_to_srgb_u8(linear: f32) -> u8 {
        let idx = (linear.clamp(0.0, 1.0) * 4095.0 + 0.5) as usize & 0xFFF;
        crate::const_luts::linear_to_srgb_u8()[idx]
    }
    /// Convert 16-bit sRGB to linear f32 using a 65536-entry LUT.
    ///
    /// Zero math — pure table lookup. The LUT is lazily initialized on first call
    /// via `OnceLock` (256KB heap, SIMD-generated in ~75µs).
    ///
    /// Without `std`, falls back to the rational polynomial.
    #[inline]
    pub fn srgb_u16_to_linear(value: u16) -> f32 {
        { crate::u16_lut::decode_lut()[value as usize] }
    }
    /// Convert linear f32 to 16-bit sRGB via rational polynomial.
    ///
    /// Perfect roundtrip with any decode method (LUT, polynomial, or precise).
    /// ~89 Mops/s. For 10× faster encode with ±1 max roundtrip error, use
    /// [`linear_to_srgb_u16_fast`].
    #[inline]
    pub fn linear_to_srgb_u16(linear: f32) -> u16 {
        let srgb = crate::rational_poly::linear_to_srgb_fast(linear);
        (srgb * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16
    }
    /// Convert linear f32 to 16-bit sRGB using a sqrt-indexed LUT (~10× faster).
    ///
    /// The sqrt indexing concentrates table resolution where the sRGB curve is
    /// steepest (near black), giving max ±1 u16 roundtrip error with 94.2%
    /// exact. The LUT is lazily initialized on first call via `OnceLock`.
    ///
    /// Use this in image pipelines where encode throughput matters more than
    /// bit-perfect roundtrip. Use [`linear_to_srgb_u16`] for lossless roundtrip.
    ///
    /// Without `std`, falls back to the rational polynomial (same as
    /// [`linear_to_srgb_u16`]).
    #[inline]
    pub fn linear_to_srgb_u16_fast(linear: f32) -> u16 {
        {
            let idx = (linear.clamp(0.0, 1.0).sqrt() * crate::u16_lut::ENCODE_SQRT_SCALE
                + 0.5) as usize;
            crate::u16_lut::encode_lut()[idx.min(crate::u16_lut::ENCODE_LUT_N - 1)]
        }
    }
    /// Convert gamma-encoded value to linear using a custom gamma exponent (f32).
    ///
    /// This is a pure power function: `linear = gamma_encoded.powf(gamma)`
    ///
    /// Unlike sRGB, there is no linear segment near black. Common gamma values:
    /// - 2.2: Traditional "gamma 2.2" used in many applications
    /// - 2.4: The power portion of sRGB (but sRGB also has a linear segment)
    /// - 1.8: Historic Mac gamma
    ///
    /// Input values are clamped to [0, 1].
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::gamma_to_linear;
    ///
    /// let linear = gamma_to_linear(0.5, 2.2);
    /// assert!((linear - 0.218).abs() < 0.001);
    /// ```
    #[inline]
    pub fn gamma_to_linear(encoded: f32, gamma: f32) -> f32 {
        if encoded <= 0.0 {
            0.0
        } else if encoded >= 1.0 {
            1.0
        } else {
            encoded.powf(gamma)
        }
    }
    /// Convert linear value to gamma-encoded using a custom gamma exponent (f32).
    ///
    /// This is a pure power function: `gamma_encoded = linear.powf(1.0 / gamma)`
    ///
    /// Unlike sRGB, there is no linear segment near black.
    ///
    /// Input values are clamped to [0, 1].
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::linear_to_gamma;
    ///
    /// let encoded = linear_to_gamma(0.218, 2.2);
    /// assert!((encoded - 0.5).abs() < 0.01);
    /// ```
    #[inline]
    pub fn linear_to_gamma(linear: f32, gamma: f32) -> f32 {
        if linear <= 0.0 {
            0.0
        } else if linear >= 1.0 {
            1.0
        } else {
            linear.powf(1.0 / gamma)
        }
    }
    /// Convert gamma-encoded value to linear using a custom gamma exponent (f64).
    ///
    /// High-precision version of [`gamma_to_linear`].
    #[inline]
    pub fn gamma_to_linear_f64(encoded: f64, gamma: f64) -> f64 {
        if encoded <= 0.0 {
            0.0
        } else if encoded >= 1.0 {
            1.0
        } else {
            encoded.powf(gamma)
        }
    }
    /// Convert linear value to gamma-encoded using a custom gamma exponent (f64).
    ///
    /// High-precision version of [`linear_to_gamma`].
    #[inline]
    pub fn linear_to_gamma_f64(linear: f64, gamma: f64) -> f64 {
        if linear <= 0.0 {
            0.0
        } else if linear >= 1.0 {
            1.0
        } else {
            linear.powf(1.0 / gamma)
        }
    }
    /// Precomputed sRGB u8 → linear f32 lookup table.
    /// Uses C0-continuous (moxcms) constants for the compile-time Newton's method computation.
    static SRGB_U8_TO_LINEAR_LUT: [f32; 256] = {
        const THRESHOLD: f64 = 12.92 * 0.003_041_282_560_127_521;
        const A: f64 = 0.055_010_718_947_586_6;
        const A_PLUS_1: f64 = 1.055_010_718_947_586_6;
        let mut lut = [0.0f32; 256];
        let mut i = 0;
        while i < 256 {
            let srgb = i as f64 / 255.0;
            let linear = if srgb <= THRESHOLD {
                srgb / 12.92
            } else {
                let base = (srgb + A) / A_PLUS_1;
                let sq = base * base;
                let target = sq;
                let mut x = 0.5f64;
                let mut iter = 0;
                while iter < 100 {
                    let x4 = x * x * x * x;
                    let x5 = x4 * x;
                    x = x - (x5 - target) / (5.0 * x4);
                    iter += 1;
                }
                sq * x
            };
            lut[i] = linear as f32;
            i += 1;
        }
        lut
    };
    #[inline]
    fn get_lut() -> &'static [f32; 256] {
        &SRGB_U8_TO_LINEAR_LUT
    }
    /// Convert a single sRGB u8 value to linear f32 using LUT lookup.
    ///
    /// This is the fastest method for u8 input as it uses a precomputed lookup table
    /// embedded in the binary. For batch conversions, use [`crate::default::srgb_u8_to_linear_slice`].
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::srgb_u8_to_linear;
    ///
    /// let linear = srgb_u8_to_linear(128);
    /// assert!((linear - 0.2158).abs() < 0.001);
    /// ```
    #[inline]
    pub fn srgb_u8_to_linear(value: u8) -> f32 {
        get_lut()[value as usize]
    }
    /// Convert 8 sRGB u8 values to linear f32 using LUT lookup.
    ///
    /// # Example
    /// ```ignore
    /// use linear_srgb::scalar::srgb_u8_to_linear_x8;
    ///
    /// let srgb = [0u8, 64, 128, 192, 255, 32, 96, 160];
    /// let linear = srgb_u8_to_linear_x8(srgb);
    /// ```
    #[inline]
    pub fn srgb_u8_to_linear_x8(srgb: [u8; 8]) -> [f32; 8] {
        let lut = get_lut();
        [
            lut[srgb[0] as usize],
            lut[srgb[1] as usize],
            lut[srgb[2] as usize],
            lut[srgb[3] as usize],
            lut[srgb[4] as usize],
            lut[srgb[5] as usize],
            lut[srgb[6] as usize],
            lut[srgb[7] as usize],
        ]
    }
}
pub(crate) mod simd {
    //! SIMD-accelerated sRGB ↔ linear conversion.
    //!
    //! This module provides high-performance conversion functions using AVX2/FMA SIMD
    //! instructions via archmage/magetypes with runtime CPU feature detection.
    //!
    //! # API Overview
    //!
    //! ## Slice Functions (process entire slices)
    //! - `srgb_to_linear_slice` - &mut \[f32\] sRGB → linear in-place
    //! - `linear_to_srgb_slice` - &mut \[f32\] linear → sRGB in-place
    //! - `srgb_u8_to_linear_slice` - &\[u8\] sRGB → &mut \[f32\] linear
    //! - `linear_to_srgb_u8_slice` - &\[f32\] linear → &mut \[u8\] sRGB
    //!
    //! ## Single-value LUT Functions
    //! - `srgb_u8_to_linear` - u8 → f32 via lookup table
    use crate::tf::fast_math::Mla;
    #[allow(unused_imports)]
    use archmage::{
        NeonToken, ScalarToken, Wasm128Token, X64V3Token, X64V4Token, arcane, incant,
        rite,
    };
    use archmage::{attune, attuned};
    use crate::kernels::float::x16::{
        gamma_to_linear as mt_gamma_to_linear, linear_to_gamma as mt_linear_to_gamma,
        linear_to_srgb as mt_linear_to_srgb, srgb_to_linear as mt_srgb_to_linear,
    };
    use crate::kernels::{store_tail, tail_block};
    #[target_feature(enable = "simd128")]
    #[inline]
    fn __attune_srgb_to_linear_slice_tier_wasm128(
        _token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::Wasm128Token>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::srgb_to_linear_slice_wasm128(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn srgb_to_linear_slice_tier_wasm128_t(
        __archmage_token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        __attune_srgb_to_linear_slice_tier_wasm128(__archmage_token, values)
    }
    fn srgb_to_linear_slice_tier_scalar_t(_token: ScalarToken, values: &mut [f32]) {
        crate::target_features::x16::srgb_to_linear_slice_scalar(values);
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn srgb_to_linear_rgba_slice_tier_wasm128(
        _token: archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            *chunk = mt_srgb_to_linear(_token, *chunk);
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    fn srgb_to_linear_rgba_slice_tier_scalar(
        _token: archmage::ScalarToken,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            *chunk = mt_srgb_to_linear(_token, *chunk);
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    /// Convert sRGB f32 values to linear in-place.
    ///
    /// Uses AVX-512 (16-wide), AVX2+FMA (8-wide), NEON (4-wide), or scalar depending on CPU.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::srgb_to_linear_slice;
    ///
    /// let mut values = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
    /// srgb_to_linear_slice(&mut values);
    /// ```
    #[inline]
    pub fn srgb_to_linear_slice(values: &mut [f32]) {
        {
            '__attune_call_0: {
                {
                    if let Some(__attune_proof) = <::archmage::Wasm128Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 srgb_to_linear_slice_tier_wasm128_t(
                            __attune_proof,
                            values,
                        );
                    }
                }
                { srgb_to_linear_slice_tier_scalar_t(::archmage::ScalarToken, values) }
            }
        }
    }
    /// Convert sRGB RGBA f32 values to linear in-place, preserving alpha.
    ///
    /// Expects interleaved RGBA data (`[R, G, B, A, R, G, B, A, ...]`).
    /// Every 4th element (alpha) is left unchanged. Trailing elements that
    /// don't form a complete RGBA pixel are ignored.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::srgb_to_linear_rgba_slice;
    ///
    /// let mut rgba = vec![0.5f32, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
    /// srgb_to_linear_rgba_slice(&mut rgba);
    /// assert_eq!(rgba[3], 0.75); // alpha preserved
    /// ```
    #[inline]
    pub fn srgb_to_linear_rgba_slice(values: &mut [f32]) {
        '__incant: {
            #[allow(unused_imports)]
            use archmage::SimdToken;
            {
                if let Some(__t) = archmage::Wasm128Token::summon() {
                    break '__incant srgb_to_linear_rgba_slice_tier_wasm128(__t, values);
                }
            }
            srgb_to_linear_rgba_slice_tier_scalar(archmage::ScalarToken, values)
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn __attune_linear_to_srgb_slice_tier_wasm128(
        _token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::Wasm128Token>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::linear_to_srgb_slice_wasm128(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn linear_to_srgb_slice_tier_wasm128_t(
        __archmage_token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        __attune_linear_to_srgb_slice_tier_wasm128(__archmage_token, values)
    }
    fn linear_to_srgb_slice_tier_scalar_t(_token: ScalarToken, values: &mut [f32]) {
        crate::target_features::x16::linear_to_srgb_slice_scalar(values);
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn linear_to_srgb_rgba_slice_tier_wasm128(
        _token: archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            *chunk = mt_linear_to_srgb(_token, *chunk);
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    fn linear_to_srgb_rgba_slice_tier_scalar(
        _token: archmage::ScalarToken,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            *chunk = mt_linear_to_srgb(_token, *chunk);
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    /// Convert linear f32 values to sRGB in-place.
    ///
    /// Uses AVX-512 (16-wide), AVX2+FMA (8-wide), NEON (4-wide), or scalar depending on CPU.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::linear_to_srgb_slice;
    ///
    /// let mut values = vec![0.0f32, 0.1, 0.2, 0.5, 1.0];
    /// linear_to_srgb_slice(&mut values);
    /// ```
    #[inline]
    pub fn linear_to_srgb_slice(values: &mut [f32]) {
        {
            '__attune_call_0: {
                {
                    if let Some(__attune_proof) = <::archmage::Wasm128Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 linear_to_srgb_slice_tier_wasm128_t(
                            __attune_proof,
                            values,
                        );
                    }
                }
                { linear_to_srgb_slice_tier_scalar_t(::archmage::ScalarToken, values) }
            }
        }
    }
    /// Convert linear RGBA f32 values to sRGB in-place, preserving alpha.
    ///
    /// Expects interleaved RGBA data (`[R, G, B, A, R, G, B, A, ...]`).
    /// Every 4th element (alpha) is left unchanged. Trailing elements that
    /// don't form a complete RGBA pixel are ignored.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::linear_to_srgb_rgba_slice;
    ///
    /// let mut rgba = vec![0.2f32, 0.2, 0.2, 0.75, 0.8, 0.8, 0.8, 1.0];
    /// linear_to_srgb_rgba_slice(&mut rgba);
    /// assert_eq!(rgba[3], 0.75); // alpha preserved
    /// ```
    #[inline]
    pub fn linear_to_srgb_rgba_slice(values: &mut [f32]) {
        '__incant: {
            #[allow(unused_imports)]
            use archmage::SimdToken;
            {
                if let Some(__t) = archmage::Wasm128Token::summon() {
                    break '__incant linear_to_srgb_rgba_slice_tier_wasm128(__t, values);
                }
            }
            linear_to_srgb_rgba_slice_tier_scalar(archmage::ScalarToken, values)
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn __attune_srgb_to_linear_extended_slice_tier_wasm128(
        _token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::Wasm128Token>;
        {
            '__attune_call_0: {
                {
                    crate::target_features::x16::srgb_to_linear_extended_slice_wasm128(
                        values,
                    )
                }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn srgb_to_linear_extended_slice_tier_wasm128_t(
        __archmage_token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        __attune_srgb_to_linear_extended_slice_tier_wasm128(__archmage_token, values)
    }
    fn srgb_to_linear_extended_slice_tier_scalar_t(
        _token: ScalarToken,
        values: &mut [f32],
    ) {
        crate::target_features::x16::srgb_to_linear_extended_slice_scalar(values);
    }
    /// Convert sRGB f32 values to linear in-place without clamping (extended range).
    ///
    /// Applies the transfer function to **every element of the slice** — if you
    /// pass RGBA data, alpha will be decoded too. Use
    /// [`srgb_to_linear_extended_rgba_slice`] for alpha-preserving RGBA conversion.
    ///
    /// Uses sign-preserving extension per CSS Color 4: `sign(v) * eotf(|v|)`.
    /// Pure SIMD polynomial on x86-64 (AVX2+FMA), AArch64 (NEON), and
    /// WebAssembly (SIMD128). Scalar `powf` fallback on other architectures.
    ///
    /// Designed for **SDR cross-gamut conversion** — converting between gamuts
    /// (P3→sRGB, BT.2020→sRGB, etc.) at SDR peak (linear 1.0). Out-of-gamut
    /// values from the 3×3 matrix survive the transfer function and can be
    /// gamut-mapped or composited downstream.
    ///
    /// # Accuracy
    ///
    /// Uses a 6/6 rational polynomial fitted to \[0, 8\] — wider domain
    /// than the clamped path's 4/4 polynomial, with comparable in-range
    /// accuracy (sampled max 8 ULP in \[0,1\] vs 11 for clamped).
    ///
    /// | Precision | Max |encoded| for < 0.5 LSB error | Headroom |
    /// |-----------|-------------------------------------|----------|
    /// | u8 (8-bit) | 8.0 | 8× SDR |
    /// | u16 (16-bit) | ~4.2 | ~4× SDR |
    ///
    /// Covers all SDR cross-gamut conversions at u16 precision (ACES AP0
    /// worst case is 1.50). u16 boundary measured via SIMD FMA dispatch.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::srgb_to_linear_extended_slice;
    ///
    /// let mut values = vec![-0.1f32, 0.0, 0.5, 1.0, 1.5];
    /// srgb_to_linear_extended_slice(&mut values);
    /// assert!(values[0] < 0.0);  // negative preserved
    /// assert!(values[4] > 1.0);  // super-white preserved
    /// ```
    #[inline]
    pub fn srgb_to_linear_extended_slice(values: &mut [f32]) {
        {
            '__attune_call_0: {
                {
                    if let Some(__attune_proof) = <::archmage::Wasm128Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 srgb_to_linear_extended_slice_tier_wasm128_t(
                            __attune_proof,
                            values,
                        );
                    }
                }
                {
                    srgb_to_linear_extended_slice_tier_scalar_t(
                        ::archmage::ScalarToken,
                        values,
                    )
                }
            }
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn __attune_linear_to_srgb_extended_slice_tier_wasm128(
        _token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::Wasm128Token>;
        {
            '__attune_call_0: {
                {
                    crate::target_features::x16::linear_to_srgb_extended_slice_wasm128(
                        values,
                    )
                }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn linear_to_srgb_extended_slice_tier_wasm128_t(
        __archmage_token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        __attune_linear_to_srgb_extended_slice_tier_wasm128(__archmage_token, values)
    }
    fn linear_to_srgb_extended_slice_tier_scalar_t(
        _token: ScalarToken,
        values: &mut [f32],
    ) {
        crate::target_features::x16::linear_to_srgb_extended_slice_scalar(values);
    }
    /// Convert linear f32 values to sRGB in-place without clamping (extended range).
    ///
    /// Applies the transfer function to **every element of the slice** — if you
    /// pass RGBA data, alpha will be encoded too. Use
    /// [`linear_to_srgb_extended_rgba_slice`] for alpha-preserving RGBA conversion.
    ///
    /// Uses sign-preserving extension per CSS Color 4: `sign(v) * oetf(|v|)`.
    /// Pure SIMD polynomial on x86-64 (AVX2+FMA), AArch64 (NEON), and
    /// WebAssembly (SIMD128). Scalar `powf` fallback on other architectures.
    ///
    /// Designed for **SDR cross-gamut conversion** — converting between gamuts
    /// (P3→sRGB, BT.2020→sRGB, etc.) at SDR peak (linear 1.0).
    ///
    /// # Accuracy
    ///
    /// Uses a 6/6 rational polynomial fitted on √x to \[0, 64\]. The sqrt
    /// compression gives excellent extrapolation — u16-safe across the
    /// entire fitted domain (sampled max 7 ULP in \[0,1\]).
    ///
    /// | Precision | Max |linear| for < 0.5 LSB error | Headroom |
    /// |-----------|--------------------------------------|----------|
    /// | u8 (8-bit) | 64.0 | 64× SDR |
    /// | u16 (16-bit) | 64.0 | 64× SDR |
    ///
    /// Covers all SDR cross-gamut conversions and all practical HDR
    /// scenarios at u16 precision.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::linear_to_srgb_extended_slice;
    ///
    /// let mut values = vec![-0.1f32, 0.0, 0.5, 1.0, 1.5];
    /// linear_to_srgb_extended_slice(&mut values);
    /// assert!(values[0] < 0.0);  // negative preserved
    /// assert!(values[4] > 1.0);  // super-white preserved
    /// ```
    #[inline]
    pub fn linear_to_srgb_extended_slice(values: &mut [f32]) {
        {
            '__attune_call_0: {
                {
                    if let Some(__attune_proof) = <::archmage::Wasm128Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 linear_to_srgb_extended_slice_tier_wasm128_t(
                            __attune_proof,
                            values,
                        );
                    }
                }
                {
                    linear_to_srgb_extended_slice_tier_scalar_t(
                        ::archmage::ScalarToken,
                        values,
                    )
                }
            }
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn srgb_to_linear_extended_rgba_slice_tier_wasm128(
        token: archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            let v = f32x16::from_array_t(token, *chunk);
            *chunk = crate::tf::srgb::srgb_to_linear_extended_x16(token, v).to_array();
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    fn srgb_to_linear_extended_rgba_slice_tier_scalar(
        token: archmage::ScalarToken,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            let v = f32x16::from_array_t(token, *chunk);
            *chunk = crate::tf::srgb::srgb_to_linear_extended_x16(token, v).to_array();
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    /// Convert sRGB RGBA f32 values to linear in-place without clamping
    /// (extended range), preserving alpha.
    ///
    /// Expects interleaved RGBA data (`[R, G, B, A, R, G, B, A, ...]`).
    /// Every 4th element (alpha) is left unchanged. Trailing elements that
    /// don't form a complete RGBA pixel are ignored.
    ///
    /// R/G/B use the same sign-preserving CSS Color 4 extension and the same
    /// SIMD polynomial as [`srgb_to_linear_extended_slice`] — see that
    /// function for the accuracy table and the intended cross-gamut / scRGB
    /// use cases. This is the variant to reach for on the interleaved RGBA
    /// output of a wide-gamut decode.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::srgb_to_linear_extended_rgba_slice;
    ///
    /// let mut rgba = vec![-0.1f32, 0.5, 1.5, 0.75, 1.0, 1.0, 1.0, 2.0];
    /// srgb_to_linear_extended_rgba_slice(&mut rgba);
    /// assert!(rgba[0] < 0.0);   // negative preserved
    /// assert!(rgba[2] > 1.0);   // super-white preserved
    /// assert_eq!(rgba[3], 0.75); // alpha untouched
    /// assert_eq!(rgba[7], 2.0);  // alpha untouched, even out of [0, 1]
    /// ```
    #[inline]
    pub fn srgb_to_linear_extended_rgba_slice(values: &mut [f32]) {
        '__incant: {
            #[allow(unused_imports)]
            use archmage::SimdToken;
            {
                if let Some(__t) = archmage::Wasm128Token::summon() {
                    break '__incant srgb_to_linear_extended_rgba_slice_tier_wasm128(
                        __t,
                        values,
                    );
                }
            }
            srgb_to_linear_extended_rgba_slice_tier_scalar(archmage::ScalarToken, values)
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn linear_to_srgb_extended_rgba_slice_tier_wasm128(
        token: archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            let v = f32x16::from_array_t(token, *chunk);
            *chunk = crate::tf::srgb::linear_to_srgb_extended_x16(token, v).to_array();
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    fn linear_to_srgb_extended_rgba_slice_tier_scalar(
        token: archmage::ScalarToken,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            let v = f32x16::from_array_t(token, *chunk);
            *chunk = crate::tf::srgb::linear_to_srgb_extended_x16(token, v).to_array();
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    /// Convert linear RGBA f32 values to sRGB in-place without clamping
    /// (extended range), preserving alpha.
    ///
    /// Expects interleaved RGBA data (`[R, G, B, A, R, G, B, A, ...]`).
    /// Every 4th element (alpha) is left unchanged. Trailing elements that
    /// don't form a complete RGBA pixel are ignored.
    ///
    /// R/G/B use the same sign-preserving CSS Color 4 extension and the same
    /// SIMD polynomial as [`linear_to_srgb_extended_slice`] — see that
    /// function for the accuracy table and the intended cross-gamut / scRGB
    /// use cases.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::linear_to_srgb_extended_rgba_slice;
    ///
    /// let mut rgba = vec![-0.1f32, 0.5, 1.5, 0.75, 1.0, 1.0, 1.0, 2.0];
    /// linear_to_srgb_extended_rgba_slice(&mut rgba);
    /// assert!(rgba[0] < 0.0);   // negative preserved
    /// assert!(rgba[2] > 1.0);   // super-white preserved
    /// assert_eq!(rgba[3], 0.75); // alpha untouched
    /// assert_eq!(rgba[7], 2.0);  // alpha untouched, even out of [0, 1]
    /// ```
    #[inline]
    pub fn linear_to_srgb_extended_rgba_slice(values: &mut [f32]) {
        '__incant: {
            #[allow(unused_imports)]
            use archmage::SimdToken;
            {
                if let Some(__t) = archmage::Wasm128Token::summon() {
                    break '__incant linear_to_srgb_extended_rgba_slice_tier_wasm128(
                        __t,
                        values,
                    );
                }
            }
            linear_to_srgb_extended_rgba_slice_tier_scalar(archmage::ScalarToken, values)
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn srgb_to_linear_premultiply_rgba_slice_tier_wasm128(
        token: archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            let converted_arr = mt_srgb_to_linear(token, *chunk);
            let converted = f32x16::from_array_t(token, converted_arr);
            let alpha = f32x16::from_array_t(
                token,
                [
                    a[0],
                    a[0],
                    a[0],
                    1.0,
                    a[1],
                    a[1],
                    a[1],
                    1.0,
                    a[2],
                    a[2],
                    a[2],
                    1.0,
                    a[3],
                    a[3],
                    a[3],
                    1.0,
                ],
            );
            *chunk = (converted * alpha).to_array();
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    fn srgb_to_linear_premultiply_rgba_slice_tier_scalar(
        token: archmage::ScalarToken,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            let converted_arr = mt_srgb_to_linear(token, *chunk);
            let converted = f32x16::from_array_t(token, converted_arr);
            let alpha = f32x16::from_array_t(
                token,
                [
                    a[0],
                    a[0],
                    a[0],
                    1.0,
                    a[1],
                    a[1],
                    a[1],
                    1.0,
                    a[2],
                    a[2],
                    a[2],
                    1.0,
                    a[3],
                    a[3],
                    a[3],
                    1.0,
                ],
            );
            *chunk = (converted * alpha).to_array();
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    /// Convert sRGB RGBA to linear premultiplied RGBA in-place.
    ///
    /// Each RGB channel is converted from sRGB to linear light, then multiplied
    /// by the alpha channel — in a single SIMD pass (no second memory traversal).
    /// Alpha is left unchanged.
    ///
    /// Input: `[R_srgb, G_srgb, B_srgb, A, ...]` (straight alpha)
    /// Output: `[R_linear*A, G_linear*A, B_linear*A, A, ...]` (premultiplied alpha)
    ///
    /// Trailing elements that don't form a complete RGBA pixel are ignored.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::srgb_to_linear_premultiply_rgba_slice;
    ///
    /// let mut rgba = vec![0.5f32, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0];
    /// srgb_to_linear_premultiply_rgba_slice(&mut rgba);
    /// // RGB converted to linear then multiplied by alpha
    /// assert!(rgba[0] < 0.15); // srgb_to_linear(0.5) ≈ 0.214, × 0.5 ≈ 0.107
    /// assert_eq!(rgba[3], 0.5); // alpha preserved
    /// ```
    #[inline]
    pub fn srgb_to_linear_premultiply_rgba_slice(values: &mut [f32]) {
        '__incant: {
            #[allow(unused_imports)]
            use archmage::SimdToken;
            {
                if let Some(__t) = archmage::Wasm128Token::summon() {
                    break '__incant srgb_to_linear_premultiply_rgba_slice_tier_wasm128(
                        __t,
                        values,
                    );
                }
            }
            srgb_to_linear_premultiply_rgba_slice_tier_scalar(
                archmage::ScalarToken,
                values,
            )
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn unpremultiply_linear_to_srgb_rgba_slice_tier_wasm128(
        token: archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            let inv = [
                if a[0] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[0] } else { 0.0 },
                if a[1] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[1] } else { 0.0 },
                if a[2] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[2] } else { 0.0 },
                if a[3] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[3] } else { 0.0 },
            ];
            let v = f32x16::from_array_t(token, *chunk);
            let inv_alpha = f32x16::from_array_t(
                token,
                [
                    inv[0],
                    inv[0],
                    inv[0],
                    1.0,
                    inv[1],
                    inv[1],
                    inv[1],
                    1.0,
                    inv[2],
                    inv[2],
                    inv[2],
                    1.0,
                    inv[3],
                    inv[3],
                    inv[3],
                    1.0,
                ],
            );
            let unpremul = (v * inv_alpha).to_array();
            *chunk = mt_linear_to_srgb(token, unpremul);
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    fn unpremultiply_linear_to_srgb_rgba_slice_tier_scalar(
        token: archmage::ScalarToken,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            let inv = [
                if a[0] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[0] } else { 0.0 },
                if a[1] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[1] } else { 0.0 },
                if a[2] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[2] } else { 0.0 },
                if a[3] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[3] } else { 0.0 },
            ];
            let v = f32x16::from_array_t(token, *chunk);
            let inv_alpha = f32x16::from_array_t(
                token,
                [
                    inv[0],
                    inv[0],
                    inv[0],
                    1.0,
                    inv[1],
                    inv[1],
                    inv[1],
                    1.0,
                    inv[2],
                    inv[2],
                    inv[2],
                    1.0,
                    inv[3],
                    inv[3],
                    inv[3],
                    1.0,
                ],
            );
            let unpremul = (v * inv_alpha).to_array();
            *chunk = mt_linear_to_srgb(token, unpremul);
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    /// Convert linear premultiplied RGBA to sRGB straight-alpha RGBA in-place.
    ///
    /// Each RGB channel is divided by alpha (unpremultiplied), then converted
    /// from linear light to sRGB — in a single SIMD pass. Alpha is left unchanged.
    ///
    /// Input: `[R_linear*A, G_linear*A, B_linear*A, A, ...]` (premultiplied)
    /// Output: `[R_srgb, G_srgb, B_srgb, A, ...]` (straight alpha)
    ///
    /// When alpha is zero, the RGB channels are set to zero (fully transparent
    /// pixels have no meaningful color). This avoids division-by-zero artifacts.
    ///
    /// Trailing elements that don't form a complete RGBA pixel are ignored.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::{srgb_to_linear_premultiply_rgba_slice,
    ///     unpremultiply_linear_to_srgb_rgba_slice};
    ///
    /// let mut rgba = vec![0.5f32, 0.5, 0.5, 0.75, 0.0, 0.0, 0.0, 0.0];
    /// srgb_to_linear_premultiply_rgba_slice(&mut rgba);
    /// unpremultiply_linear_to_srgb_rgba_slice(&mut rgba);
    /// assert!((rgba[0] - 0.5).abs() < 1e-4); // roundtrips
    /// assert_eq!(rgba[3], 0.75);              // alpha preserved
    /// assert_eq!(rgba[4], 0.0);               // transparent pixel stays zero
    /// ```
    #[inline]
    pub fn unpremultiply_linear_to_srgb_rgba_slice(values: &mut [f32]) {
        '__incant: {
            #[allow(unused_imports)]
            use archmage::SimdToken;
            {
                if let Some(__t) = archmage::Wasm128Token::summon() {
                    break '__incant unpremultiply_linear_to_srgb_rgba_slice_tier_wasm128(
                        __t,
                        values,
                    );
                }
            }
            unpremultiply_linear_to_srgb_rgba_slice_tier_scalar(
                archmage::ScalarToken,
                values,
            )
        }
    }
    /// Convert sRGB u8 values to linear f32.
    ///
    /// Uses a precomputed LUT for each u8 value, processed in batches of 8.
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::srgb_u8_to_linear_slice;
    ///
    /// let input: Vec<u8> = (0..=255).collect();
    /// let mut output = vec![0.0f32; 256];
    /// srgb_u8_to_linear_slice(&input, &mut output);
    /// ```
    #[inline]
    pub fn srgb_u8_to_linear_slice(input: &[u8], output: &mut [f32]) {
        {
            match (&input.len(), &output.len()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            }
        };
        let (in_chunks, in_remainder) = input.as_chunks::<8>();
        let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            *out = crate::scalar::srgb_u8_to_linear_x8(*inp);
        }
        for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
            *out = crate::scalar::srgb_u8_to_linear(*inp);
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn linear_to_srgb_u8_slice_tier_wasm128(
        token: archmage::Wasm128Token,
        input: &[f32],
        output: &mut [u8],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let lut = crate::const_luts::linear_to_srgb_u8();
        let (in_chunks, in_rem) = input.as_chunks::<16>();
        let (out_chunks, out_rem) = output.as_chunks_mut::<16>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let v = f32x16::from_array_t(token, *inp);
            let clamped = v.max(f32x16::zero_t(token)).min(f32x16::splat_t(token, 1.0));
            let idx_f = clamped
                .mla(f32x16::splat_t(token, 4095.0), f32x16::splat_t(token, 0.5));
            let idx_i = idx_f.to_i32().to_array();
            for i in 0..16 {
                out[i] = lut[(idx_i[i] as usize) & 0xFFF];
            }
        }
        for (inp, out) in in_rem.iter().zip(out_rem.iter_mut()) {
            let clamped = inp.clamp(0.0, 1.0);
            let idx = (clamped * 4095.0 + 0.5) as usize & 0xFFF;
            *out = lut[idx];
        }
    }
    fn linear_to_srgb_u8_slice_tier_scalar(
        token: archmage::ScalarToken,
        input: &[f32],
        output: &mut [u8],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let lut = crate::const_luts::linear_to_srgb_u8();
        let (in_chunks, in_rem) = input.as_chunks::<16>();
        let (out_chunks, out_rem) = output.as_chunks_mut::<16>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let v = f32x16::from_array_t(token, *inp);
            let clamped = v.max(f32x16::zero_t(token)).min(f32x16::splat_t(token, 1.0));
            let idx_f = clamped
                .mla(f32x16::splat_t(token, 4095.0), f32x16::splat_t(token, 0.5));
            let idx_i = idx_f.to_i32().to_array();
            for i in 0..16 {
                out[i] = lut[(idx_i[i] as usize) & 0xFFF];
            }
        }
        for (inp, out) in in_rem.iter().zip(out_rem.iter_mut()) {
            let clamped = inp.clamp(0.0, 1.0);
            let idx = (clamped * 4095.0 + 0.5) as usize & 0xFFF;
            *out = lut[idx];
        }
    }
    /// Convert linear f32 values to sRGB u8.
    ///
    /// Uses a 4096-entry const LUT with SIMD-computed indices (AVX-512 / AVX2+FMA
    /// / NEON / WASM SIMD128) so the 4/8/16 scalar LUT loads per chunk can issue
    /// in parallel rather than serializing through the float-to-int cast.
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::linear_to_srgb_u8_slice;
    ///
    /// let input: Vec<f32> = (0..=255).map(|i| i as f32 / 255.0).collect();
    /// let mut output = vec![0u8; 256];
    /// linear_to_srgb_u8_slice(&input, &mut output);
    /// ```
    #[inline]
    pub fn linear_to_srgb_u8_slice(input: &[f32], output: &mut [u8]) {
        {
            match (&input.len(), &output.len()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            }
        };
        '__incant: {
            #[allow(unused_imports)]
            use archmage::SimdToken;
            {
                if let Some(__t) = archmage::Wasm128Token::summon() {
                    break '__incant linear_to_srgb_u8_slice_tier_wasm128(
                        __t,
                        input,
                        output,
                    );
                }
            }
            linear_to_srgb_u8_slice_tier_scalar(archmage::ScalarToken, input, output)
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn srgb_u16_to_linear_slice_tier_wasm128(
        token: archmage::Wasm128Token,
        input: &[u16],
        output: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        use crate::rational_poly::{LINEAR_SCALE, S2L_P, S2L_Q, SRGB_THRESHOLD};
        let (in_chunks, in_rem) = input.as_chunks::<16>();
        let (out_chunks, out_rem) = output.as_chunks_mut::<16>();
        let convert = |inp: &[u16; 16]| -> [f32; 16] {
            let mut f = [0.0f32; 16];
            for i in 0..16 {
                f[i] = inp[i] as f32;
            }
            let v = f32x16::from_array_t(token, f)
                * f32x16::splat_t(token, 1.0 / 65535.0);
            let one = f32x16::splat_t(token, 1.0);
            let linear_result = v * f32x16::splat_t(token, LINEAR_SCALE);
            let yp = f32x16::splat_t(token, S2L_P[4])
                .mla(v, f32x16::splat_t(token, S2L_P[3]));
            let yp = yp.mla(v, f32x16::splat_t(token, S2L_P[2]));
            let yp = yp.mla(v, f32x16::splat_t(token, S2L_P[1]));
            let yp = yp.mla(v, f32x16::splat_t(token, S2L_P[0]));
            let yq = f32x16::splat_t(token, S2L_Q[4])
                .mla(v, f32x16::splat_t(token, S2L_Q[3]));
            let yq = yq.mla(v, f32x16::splat_t(token, S2L_Q[2]));
            let yq = yq.mla(v, f32x16::splat_t(token, S2L_Q[1]));
            let yq = yq.mla(v, f32x16::splat_t(token, S2L_Q[0]));
            let power_result = (yp / yq).min(one);
            let mask = v.simd_le(f32x16::splat_t(token, SRGB_THRESHOLD));
            let result = f32x16::blend(mask, linear_result, power_result);
            let ge_one = v.simd_ge(one);
            let result = f32x16::blend(ge_one, one, result);
            result.to_array()
        };
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            *out = convert(inp);
        }
        if crate::tf::fast_math::USE_FUSED_MATH {
            for (inp, out) in in_rem.iter().zip(out_rem.iter_mut()) {
                *out = crate::lanes::srgb_to_linear(*inp as f32 * (1.0 / 65535.0));
            }
        } else if !in_rem.is_empty() {
            let mut padded = [0u16; 16];
            padded[..in_rem.len()].copy_from_slice(in_rem);
            out_rem.copy_from_slice(&convert(&padded)[..in_rem.len()]);
        }
    }
    fn srgb_u16_to_linear_slice_tier_scalar(
        token: archmage::ScalarToken,
        input: &[u16],
        output: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        use crate::rational_poly::{LINEAR_SCALE, S2L_P, S2L_Q, SRGB_THRESHOLD};
        let (in_chunks, in_rem) = input.as_chunks::<16>();
        let (out_chunks, out_rem) = output.as_chunks_mut::<16>();
        let convert = |inp: &[u16; 16]| -> [f32; 16] {
            let mut f = [0.0f32; 16];
            for i in 0..16 {
                f[i] = inp[i] as f32;
            }
            let v = f32x16::from_array_t(token, f)
                * f32x16::splat_t(token, 1.0 / 65535.0);
            let one = f32x16::splat_t(token, 1.0);
            let linear_result = v * f32x16::splat_t(token, LINEAR_SCALE);
            let yp = f32x16::splat_t(token, S2L_P[4])
                .mla(v, f32x16::splat_t(token, S2L_P[3]));
            let yp = yp.mla(v, f32x16::splat_t(token, S2L_P[2]));
            let yp = yp.mla(v, f32x16::splat_t(token, S2L_P[1]));
            let yp = yp.mla(v, f32x16::splat_t(token, S2L_P[0]));
            let yq = f32x16::splat_t(token, S2L_Q[4])
                .mla(v, f32x16::splat_t(token, S2L_Q[3]));
            let yq = yq.mla(v, f32x16::splat_t(token, S2L_Q[2]));
            let yq = yq.mla(v, f32x16::splat_t(token, S2L_Q[1]));
            let yq = yq.mla(v, f32x16::splat_t(token, S2L_Q[0]));
            let power_result = (yp / yq).min(one);
            let mask = v.simd_le(f32x16::splat_t(token, SRGB_THRESHOLD));
            let result = f32x16::blend(mask, linear_result, power_result);
            let ge_one = v.simd_ge(one);
            let result = f32x16::blend(ge_one, one, result);
            result.to_array()
        };
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            *out = convert(inp);
        }
        if crate::tf::fast_math::USE_FUSED_MATH {
            for (inp, out) in in_rem.iter().zip(out_rem.iter_mut()) {
                *out = crate::lanes::srgb_to_linear(*inp as f32 * (1.0 / 65535.0));
            }
        } else if !in_rem.is_empty() {
            let mut padded = [0u16; 16];
            padded[..in_rem.len()].copy_from_slice(in_rem);
            out_rem.copy_from_slice(&convert(&padded)[..in_rem.len()]);
        }
    }
    /// Convert sRGB u16 values to linear f32 via rational polynomial (SIMD).
    ///
    /// Dispatches through `incant!` over `[v4, v3, neon, wasm128, scalar]` so
    /// the polynomial evaluates 4/8/16 wide. No LUT and no cache footprint —
    /// replaces the 256KB decode LUT on the slice path. Boundary values
    /// (0, 65535) map exactly to (0.0, 1.0). Output may differ from the
    /// scalar LUT at 31,766 of 65,536 codes, by at most 4 f32 ULP
    /// (exhaustive native-tier measurement, November 2026).
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    #[inline]
    pub fn srgb_u16_to_linear_slice(input: &[u16], output: &mut [f32]) {
        {
            match (&input.len(), &output.len()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            }
        };
        '__incant: {
            #[allow(unused_imports)]
            use archmage::SimdToken;
            {
                if let Some(__t) = archmage::Wasm128Token::summon() {
                    break '__incant srgb_u16_to_linear_slice_tier_wasm128(
                        __t,
                        input,
                        output,
                    );
                }
            }
            srgb_u16_to_linear_slice_tier_scalar(archmage::ScalarToken, input, output)
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn __attune_linear_to_srgb_u16_slice_tier_wasm128(
        token: ::archmage::Wasm128Token,
        input: &[f32],
        output: &mut [u16],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::Wasm128Token>;
        use crate::rational_poly::{L2S_P, L2S_Q, LINEAR_THRESHOLD};
        let (in_chunks, in_rem) = input.as_chunks::<16>();
        let (out_chunks, out_rem) = output.as_chunks_mut::<16>();
        let convert = |inp: &[f32; 16]| -> [u16; 16] {
            let linear = f32x16::from_array_t(token, *inp);
            let zero = f32x16::zero_t(token);
            let one = f32x16::splat_t(token, 1.0);
            let clamped = linear.max(zero).min(one);
            let linear_result = clamped * f32x16::splat_t(token, 12.92);
            let x = clamped.sqrt();
            let yp = f32x16::splat_t(token, L2S_P[4])
                .mla(x, f32x16::splat_t(token, L2S_P[3]));
            let yp = yp.mla(x, f32x16::splat_t(token, L2S_P[2]));
            let yp = yp.mla(x, f32x16::splat_t(token, L2S_P[1]));
            let yp = yp.mla(x, f32x16::splat_t(token, L2S_P[0]));
            let yq = f32x16::splat_t(token, L2S_Q[4])
                .mla(x, f32x16::splat_t(token, L2S_Q[3]));
            let yq = yq.mla(x, f32x16::splat_t(token, L2S_Q[2]));
            let yq = yq.mla(x, f32x16::splat_t(token, L2S_Q[1]));
            let yq = yq.mla(x, f32x16::splat_t(token, L2S_Q[0]));
            let power_result = (yp / yq).min(one);
            let thresh_mask = clamped.simd_lt(f32x16::splat_t(token, LINEAR_THRESHOLD));
            let srgb = f32x16::blend(thresh_mask, linear_result, power_result);
            let ge_one = linear.simd_ge(one);
            let srgb = f32x16::blend(ge_one, one, srgb);
            let scaled = srgb
                .mla(f32x16::splat_t(token, 65535.0), f32x16::splat_t(token, 0.5));
            let idx_i = scaled.to_i32().to_array();
            idx_i.map(|value| value as u16)
        };
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            *out = convert(inp);
        }
        if crate::tf::fast_math::USE_FUSED_MATH {
            for (inp, out) in in_rem.iter().zip(out_rem.iter_mut()) {
                *out = crate::lanes::linear_to_srgb_u16(*inp);
            }
        } else if !in_rem.is_empty() {
            let mut padded = [0.0f32; 16];
            padded[..in_rem.len()].copy_from_slice(in_rem);
            out_rem.copy_from_slice(&convert(&padded)[..in_rem.len()]);
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn linear_to_srgb_u16_slice_tier_wasm128_t(
        token: ::archmage::Wasm128Token,
        input: &[f32],
        output: &mut [u16],
    ) {
        __attune_linear_to_srgb_u16_slice_tier_wasm128(token, input, output)
    }
    fn linear_to_srgb_u16_slice_tier_scalar_t(
        _token: ScalarToken,
        input: &[f32],
        output: &mut [u16],
    ) {
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = crate::lanes::linear_to_srgb_u16(*inp);
        }
    }
    /// Convert linear f32 values to sRGB u16 via rational polynomial.
    ///
    /// Perfect roundtrip. Dispatches through `incant!` over
    /// `[v4, v3, neon, wasm128, scalar]` so the polynomial evaluates 4/8/16 wide
    /// before f32→u16 quantization. For ~10× faster encode with ±1 max error,
    /// use [`linear_to_srgb_u16_slice_fast`].
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    #[inline]
    pub fn linear_to_srgb_u16_slice(input: &[f32], output: &mut [u16]) {
        {
            match (&input.len(), &output.len()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            }
        };
        {
            '__attune_call_0: {
                {
                    if let Some(__attune_proof) = <::archmage::Wasm128Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 linear_to_srgb_u16_slice_tier_wasm128_t(
                            __attune_proof,
                            input,
                            output,
                        );
                    }
                }
                {
                    linear_to_srgb_u16_slice_tier_scalar_t(
                        ::archmage::ScalarToken,
                        input,
                        output,
                    )
                }
            }
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn linear_to_srgb_u16_slice_fast_tier_wasm128(
        token: archmage::Wasm128Token,
        input: &[f32],
        output: &mut [u16],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let lut = crate::u16_lut::encode_lut();
        let max_idx = crate::u16_lut::ENCODE_LUT_N - 1;
        let scale = crate::u16_lut::ENCODE_SQRT_SCALE;
        let (in_chunks, in_rem) = input.as_chunks::<16>();
        let (out_chunks, out_rem) = output.as_chunks_mut::<16>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let v = f32x16::from_array_t(token, *inp);
            let clamped = v.max(f32x16::zero_t(token)).min(f32x16::splat_t(token, 1.0));
            let idx_f = clamped
                .sqrt()
                .mla(f32x16::splat_t(token, scale), f32x16::splat_t(token, 0.5));
            let idx_i = idx_f.to_i32().to_array();
            for i in 0..16 {
                out[i] = lut[(idx_i[i] as usize).min(max_idx)];
            }
        }
        for (inp, out) in in_rem.iter().zip(out_rem.iter_mut()) {
            *out = crate::scalar::linear_to_srgb_u16_fast(*inp);
        }
    }
    fn linear_to_srgb_u16_slice_fast_tier_scalar(
        token: archmage::ScalarToken,
        input: &[f32],
        output: &mut [u16],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let lut = crate::u16_lut::encode_lut();
        let max_idx = crate::u16_lut::ENCODE_LUT_N - 1;
        let scale = crate::u16_lut::ENCODE_SQRT_SCALE;
        let (in_chunks, in_rem) = input.as_chunks::<16>();
        let (out_chunks, out_rem) = output.as_chunks_mut::<16>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let v = f32x16::from_array_t(token, *inp);
            let clamped = v.max(f32x16::zero_t(token)).min(f32x16::splat_t(token, 1.0));
            let idx_f = clamped
                .sqrt()
                .mla(f32x16::splat_t(token, scale), f32x16::splat_t(token, 0.5));
            let idx_i = idx_f.to_i32().to_array();
            for i in 0..16 {
                out[i] = lut[(idx_i[i] as usize).min(max_idx)];
            }
        }
        for (inp, out) in in_rem.iter().zip(out_rem.iter_mut()) {
            *out = crate::scalar::linear_to_srgb_u16_fast(*inp);
        }
    }
    /// Convert linear f32 values to sRGB u16 using sqrt-indexed LUT (~10× faster).
    ///
    /// Max ±1 u16 roundtrip error, 94.2% exact. SIMD-computes the sqrt-based
    /// index 4/8/16-wide per chunk so the scalar LUT loads can issue in
    /// parallel. See [`linear_to_srgb_u16_fast`](crate::scalar::linear_to_srgb_u16_fast)
    /// for the exact formula.
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    #[inline]
    pub fn linear_to_srgb_u16_slice_fast(input: &[f32], output: &mut [u16]) {
        {
            match (&input.len(), &output.len()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            }
        };
        {
            '__incant: {
                #[allow(unused_imports)]
                use archmage::SimdToken;
                {
                    if let Some(__t) = archmage::Wasm128Token::summon() {
                        break '__incant linear_to_srgb_u16_slice_fast_tier_wasm128(
                            __t,
                            input,
                            output,
                        );
                    }
                }
                linear_to_srgb_u16_slice_fast_tier_scalar(
                    archmage::ScalarToken,
                    input,
                    output,
                )
            }
        }
    }
    /// Convert sRGB RGBA u8 values to linear f32, preserving alpha.
    ///
    /// RGB channels are decoded via LUT. Alpha is passed through as `a / 255.0`
    /// without sRGB transfer. Trailing elements that don't form a complete
    /// RGBA pixel are ignored.
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::srgb_u8_to_linear_rgba_slice;
    ///
    /// let input = [128u8, 128, 128, 255, 64, 64, 64, 128];
    /// let mut output = [0.0f32; 8];
    /// srgb_u8_to_linear_rgba_slice(&input, &mut output);
    /// assert_eq!(output[3], 1.0);           // alpha 255 → 1.0
    /// assert!((output[7] - 0.502).abs() < 0.01); // alpha 128 → ~0.502
    /// ```
    pub fn srgb_u8_to_linear_rgba_slice(input: &[u8], output: &mut [f32]) {
        {
            match (&input.len(), &output.len()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            }
        };
        let (in_chunks, in_remainder) = input.as_chunks::<8>();
        let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            *out = crate::scalar::srgb_u8_to_linear_x8(*inp);
            out[3] = inp[3] as f32 / 255.0;
            out[7] = inp[7] as f32 / 255.0;
        }
        let in_rem_pixels = in_remainder.as_chunks::<4>().0.iter();
        let out_rem_pixels = out_remainder.as_chunks_mut::<4>().0.iter_mut();
        for (inp, out) in in_rem_pixels.zip(out_rem_pixels) {
            out[0] = crate::scalar::srgb_u8_to_linear(inp[0]);
            out[1] = crate::scalar::srgb_u8_to_linear(inp[1]);
            out[2] = crate::scalar::srgb_u8_to_linear(inp[2]);
            out[3] = inp[3] as f32 / 255.0;
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    /// Convert linear RGBA f32 values to sRGB u8, preserving alpha.
    ///
    /// RGB channels are encoded via 4096-entry const LUT. Alpha is passed through
    /// as `(a * 255 + 0.5) as u8` without sRGB transfer. Trailing elements that
    /// don't form a complete RGBA pixel are ignored.
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::linear_to_srgb_u8_rgba_slice;
    ///
    /// let input = [0.2158f32, 0.2158, 0.2158, 1.0, 0.05, 0.05, 0.05, 0.5];
    /// let mut output = [0u8; 8];
    /// linear_to_srgb_u8_rgba_slice(&input, &mut output);
    /// assert_eq!(output[3], 255);  // alpha 1.0 → 255
    /// assert_eq!(output[7], 128);  // alpha 0.5 → 128
    /// ```
    fn linear_to_srgb_u8_rgba_slice_tier_wasm128(
        token: archmage::Wasm128Token,
        input: &[f32],
        output: &mut [u8],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let lut = crate::const_luts::linear_to_srgb_u8();
        let (in_chunks, in_rem) = input.as_chunks::<16>();
        let (out_chunks, out_rem) = output.as_chunks_mut::<16>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let v = f32x16::from_array_t(token, *inp);
            let clamped = v.max(f32x16::zero_t(token)).min(f32x16::splat_t(token, 1.0));
            let idx_f = clamped
                .mla(f32x16::splat_t(token, 4095.0), f32x16::splat_t(token, 0.5));
            let idx_i = idx_f.to_i32().to_array();
            for px in 0..4 {
                let base = px * 4;
                out[base] = lut[(idx_i[base] as usize) & 0xFFF];
                out[base + 1] = lut[(idx_i[base + 1] as usize) & 0xFFF];
                out[base + 2] = lut[(idx_i[base + 2] as usize) & 0xFFF];
                out[base + 3] = (f64::from(inp[base + 3]).clamp(0.0, 1.0) * 255.0 + 0.5)
                    as u8;
            }
        }
        for (inp, out) in in_rem
            .as_chunks::<4>()
            .0
            .iter()
            .zip(out_rem.as_chunks_mut::<4>().0.iter_mut())
        {
            for i in 0..3 {
                let clamped = inp[i].clamp(0.0, 1.0);
                let idx = (clamped * 4095.0 + 0.5) as usize & 0xFFF;
                out[i] = lut[idx];
            }
            out[3] = (f64::from(inp[3]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    /// Convert linear RGBA f32 values to sRGB u8, preserving alpha.
    ///
    /// RGB channels are encoded via 4096-entry const LUT. Alpha is passed through
    /// as `(a * 255 + 0.5) as u8` without sRGB transfer. Trailing elements that
    /// don't form a complete RGBA pixel are ignored.
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::linear_to_srgb_u8_rgba_slice;
    ///
    /// let input = [0.2158f32, 0.2158, 0.2158, 1.0, 0.05, 0.05, 0.05, 0.5];
    /// let mut output = [0u8; 8];
    /// linear_to_srgb_u8_rgba_slice(&input, &mut output);
    /// assert_eq!(output[3], 255);  // alpha 1.0 → 255
    /// assert_eq!(output[7], 128);  // alpha 0.5 → 128
    /// ```
    fn linear_to_srgb_u8_rgba_slice_tier_scalar(
        token: archmage::ScalarToken,
        input: &[f32],
        output: &mut [u8],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let lut = crate::const_luts::linear_to_srgb_u8();
        let (in_chunks, in_rem) = input.as_chunks::<16>();
        let (out_chunks, out_rem) = output.as_chunks_mut::<16>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let v = f32x16::from_array_t(token, *inp);
            let clamped = v.max(f32x16::zero_t(token)).min(f32x16::splat_t(token, 1.0));
            let idx_f = clamped
                .mla(f32x16::splat_t(token, 4095.0), f32x16::splat_t(token, 0.5));
            let idx_i = idx_f.to_i32().to_array();
            for px in 0..4 {
                let base = px * 4;
                out[base] = lut[(idx_i[base] as usize) & 0xFFF];
                out[base + 1] = lut[(idx_i[base + 1] as usize) & 0xFFF];
                out[base + 2] = lut[(idx_i[base + 2] as usize) & 0xFFF];
                out[base + 3] = (f64::from(inp[base + 3]).clamp(0.0, 1.0) * 255.0 + 0.5)
                    as u8;
            }
        }
        for (inp, out) in in_rem
            .as_chunks::<4>()
            .0
            .iter()
            .zip(out_rem.as_chunks_mut::<4>().0.iter_mut())
        {
            for i in 0..3 {
                let clamped = inp[i].clamp(0.0, 1.0);
                let idx = (clamped * 4095.0 + 0.5) as usize & 0xFFF;
                out[i] = lut[idx];
            }
            out[3] = (f64::from(inp[3]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    /// Convert linear RGBA f32 to sRGB RGBA u8, preserving alpha.
    ///
    /// RGB channels are LUT-encoded with SIMD-computed indices; alpha is scaled
    /// linearly by 255 and rounded. Trailing elements that don't form a complete
    /// RGBA pixel are ignored.
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    #[inline]
    pub fn linear_to_srgb_u8_rgba_slice(input: &[f32], output: &mut [u8]) {
        {
            match (&input.len(), &output.len()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            }
        };
        '__incant: {
            #[allow(unused_imports)]
            use archmage::SimdToken;
            {
                if let Some(__t) = archmage::Wasm128Token::summon() {
                    break '__incant linear_to_srgb_u8_rgba_slice_tier_wasm128(
                        __t,
                        input,
                        output,
                    );
                }
            }
            linear_to_srgb_u8_rgba_slice_tier_scalar(
                archmage::ScalarToken,
                input,
                output,
            )
        }
    }
    /// Convert sRGB RGBA u8 to linear premultiplied RGBA f32.
    ///
    /// Fused operation: RGB channels are decoded via LUT to linear f32, then
    /// multiplied by alpha. Alpha is passed through as `a / 255.0`.
    ///
    /// Trailing elements that don't form a complete RGBA pixel are ignored.
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::srgb_u8_to_linear_premultiply_rgba_slice;
    ///
    /// let input = [128u8, 128, 128, 128, 255, 255, 255, 255];
    /// let mut output = [0.0f32; 8];
    /// srgb_u8_to_linear_premultiply_rgba_slice(&input, &mut output);
    /// // RGB decoded then premultiplied by alpha (128/255 ≈ 0.502)
    /// assert!(output[0] < 0.12); // srgb_to_linear(128/255) * 0.502 ≈ 0.107
    /// assert!((output[3] - 128.0 / 255.0).abs() < 1e-6); // alpha passthrough
    /// ```
    pub fn srgb_u8_to_linear_premultiply_rgba_slice(input: &[u8], output: &mut [f32]) {
        {
            match (&input.len(), &output.len()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            }
        };
        let (in_chunks, in_remainder) = input.as_chunks::<8>();
        let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            *out = crate::scalar::srgb_u8_to_linear_x8(*inp);
            let a0 = inp[3] as f32 / 255.0;
            let a1 = inp[7] as f32 / 255.0;
            out[0] *= a0;
            out[1] *= a0;
            out[2] *= a0;
            out[3] = a0;
            out[4] *= a1;
            out[5] *= a1;
            out[6] *= a1;
            out[7] = a1;
        }
        let in_rem_pixels = in_remainder.as_chunks::<4>().0.iter();
        let out_rem_pixels = out_remainder.as_chunks_mut::<4>().0.iter_mut();
        for (inp, out) in in_rem_pixels.zip(out_rem_pixels) {
            let a = inp[3] as f32 / 255.0;
            out[0] = crate::scalar::srgb_u8_to_linear(inp[0]) * a;
            out[1] = crate::scalar::srgb_u8_to_linear(inp[1]) * a;
            out[2] = crate::scalar::srgb_u8_to_linear(inp[2]) * a;
            out[3] = a;
        }
    }
    #[inline(always)]
    fn inv_alpha_or_zero(a: f32) -> f32 {
        if a > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a } else { 0.0 }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    #[allow(clippy::needless_range_loop)]
    fn unpremultiply_linear_to_srgb_u8_rgba_slice_tier_wasm128(
        token: archmage::Wasm128Token,
        input: &[f32],
        output: &mut [u8],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let lut = crate::const_luts::linear_to_srgb_u8();
        let (in_chunks, in_rem) = input.as_chunks::<16>();
        let (out_chunks, out_rem) = output.as_chunks_mut::<16>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let a = [inp[3], inp[7], inp[11], inp[15]];
            let inv_a = [
                inv_alpha_or_zero(a[0]),
                inv_alpha_or_zero(a[1]),
                inv_alpha_or_zero(a[2]),
                inv_alpha_or_zero(a[3]),
            ];
            let mul_arr = [
                inv_a[0],
                inv_a[0],
                inv_a[0],
                1.0,
                inv_a[1],
                inv_a[1],
                inv_a[1],
                1.0,
                inv_a[2],
                inv_a[2],
                inv_a[2],
                1.0,
                inv_a[3],
                inv_a[3],
                inv_a[3],
                1.0,
            ];
            let mul_vec = f32x16::from_array_t(token, mul_arr);
            let v = f32x16::from_array_t(token, *inp);
            let unpremul = v * mul_vec;
            let clamped = unpremul
                .max(f32x16::zero_t(token))
                .min(f32x16::splat_t(token, 1.0));
            let idx_f = clamped
                .mla(f32x16::splat_t(token, 4095.0), f32x16::splat_t(token, 0.5));
            let idx_i = idx_f.to_i32().to_array();
            for px in 0..4 {
                let base = px * 4;
                out[base] = lut[(idx_i[base] as usize) & 0xFFF];
                out[base + 1] = lut[(idx_i[base + 1] as usize) & 0xFFF];
                out[base + 2] = lut[(idx_i[base + 2] as usize) & 0xFFF];
                out[base + 3] = (f64::from(a[px]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
        for (inp, out) in in_rem
            .as_chunks::<4>()
            .0
            .iter()
            .zip(out_rem.as_chunks_mut::<4>().0.iter_mut())
        {
            let a = inp[3];
            let inv_a = inv_alpha_or_zero(a);
            for i in 0..3 {
                let clamped = (inp[i] * inv_a).clamp(0.0, 1.0);
                let idx = (clamped * 4095.0 + 0.5) as usize & 0xFFF;
                out[i] = lut[idx];
            }
            out[3] = (f64::from(a).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    #[allow(clippy::needless_range_loop)]
    fn unpremultiply_linear_to_srgb_u8_rgba_slice_tier_scalar(
        token: archmage::ScalarToken,
        input: &[f32],
        output: &mut [u8],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let lut = crate::const_luts::linear_to_srgb_u8();
        let (in_chunks, in_rem) = input.as_chunks::<16>();
        let (out_chunks, out_rem) = output.as_chunks_mut::<16>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let a = [inp[3], inp[7], inp[11], inp[15]];
            let inv_a = [
                inv_alpha_or_zero(a[0]),
                inv_alpha_or_zero(a[1]),
                inv_alpha_or_zero(a[2]),
                inv_alpha_or_zero(a[3]),
            ];
            let mul_arr = [
                inv_a[0],
                inv_a[0],
                inv_a[0],
                1.0,
                inv_a[1],
                inv_a[1],
                inv_a[1],
                1.0,
                inv_a[2],
                inv_a[2],
                inv_a[2],
                1.0,
                inv_a[3],
                inv_a[3],
                inv_a[3],
                1.0,
            ];
            let mul_vec = f32x16::from_array_t(token, mul_arr);
            let v = f32x16::from_array_t(token, *inp);
            let unpremul = v * mul_vec;
            let clamped = unpremul
                .max(f32x16::zero_t(token))
                .min(f32x16::splat_t(token, 1.0));
            let idx_f = clamped
                .mla(f32x16::splat_t(token, 4095.0), f32x16::splat_t(token, 0.5));
            let idx_i = idx_f.to_i32().to_array();
            for px in 0..4 {
                let base = px * 4;
                out[base] = lut[(idx_i[base] as usize) & 0xFFF];
                out[base + 1] = lut[(idx_i[base + 1] as usize) & 0xFFF];
                out[base + 2] = lut[(idx_i[base + 2] as usize) & 0xFFF];
                out[base + 3] = (f64::from(a[px]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
        for (inp, out) in in_rem
            .as_chunks::<4>()
            .0
            .iter()
            .zip(out_rem.as_chunks_mut::<4>().0.iter_mut())
        {
            let a = inp[3];
            let inv_a = inv_alpha_or_zero(a);
            for i in 0..3 {
                let clamped = (inp[i] * inv_a).clamp(0.0, 1.0);
                let idx = (clamped * 4095.0 + 0.5) as usize & 0xFFF;
                out[i] = lut[idx];
            }
            out[3] = (f64::from(a).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    /// Convert linear premultiplied RGBA f32 to sRGB straight-alpha RGBA u8.
    ///
    /// Fused SIMD operation: each RGB channel is divided by alpha
    /// (unpremultiplied), then encoded to sRGB u8 via LUT. Alpha is passed
    /// through as `(a * 255 + 0.5) as u8`. When alpha is at or below
    /// [`UNPREMUL_ALPHA_THRESHOLD`](crate::UNPREMUL_ALPHA_THRESHOLD), RGB
    /// output bytes are set to zero — this falls out of the math naturally
    /// (`inv_a = 0 → lut[0] = 0`) with no per-pixel branch on the write side.
    ///
    /// Trailing elements that don't form a complete RGBA pixel are ignored.
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::{srgb_u8_to_linear_premultiply_rgba_slice,
    ///     unpremultiply_linear_to_srgb_u8_rgba_slice};
    ///
    /// let input = [128u8, 128, 128, 255, 0, 0, 0, 0];
    /// let mut linear = [0.0f32; 8];
    /// srgb_u8_to_linear_premultiply_rgba_slice(&input, &mut linear);
    /// let mut output = [0u8; 8];
    /// unpremultiply_linear_to_srgb_u8_rgba_slice(&linear, &mut output);
    /// assert!((output[0] as i32 - 128).abs() <= 1); // roundtrips within 1
    /// assert_eq!(output[7], 0); // transparent pixel
    /// ```
    #[inline]
    pub fn unpremultiply_linear_to_srgb_u8_rgba_slice(input: &[f32], output: &mut [u8]) {
        {
            match (&input.len(), &output.len()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            }
        };
        '__incant: {
            #[allow(unused_imports)]
            use archmage::SimdToken;
            {
                if let Some(__t) = archmage::Wasm128Token::summon() {
                    break '__incant unpremultiply_linear_to_srgb_u8_rgba_slice_tier_wasm128(
                        __t,
                        input,
                        output,
                    );
                }
            }
            unpremultiply_linear_to_srgb_u8_rgba_slice_tier_scalar(
                archmage::ScalarToken,
                input,
                output,
            )
        }
    }
    /// Convert sRGB RGBA u16 values to linear f32, preserving alpha.
    ///
    /// RGB channels are decoded via lazily-initialized 65536-entry LUT. Alpha is passed
    /// through as `a / 65535.0` without sRGB transfer. Trailing elements
    /// that don't form a complete RGBA pixel are ignored.
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    pub fn srgb_u16_to_linear_rgba_slice(input: &[u16], output: &mut [f32]) {
        {
            match (&input.len(), &output.len()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            }
        };
        let in_pixels = input.as_chunks::<4>().0.iter();
        let out_pixels = output.as_chunks_mut::<4>().0.iter_mut();
        for (inp, out) in in_pixels.zip(out_pixels) {
            out[0] = crate::scalar::srgb_u16_to_linear(inp[0]);
            out[1] = crate::scalar::srgb_u16_to_linear(inp[1]);
            out[2] = crate::scalar::srgb_u16_to_linear(inp[2]);
            out[3] = inp[3] as f32 / 65535.0;
        }
    }
    /// Convert linear RGBA f32 to sRGB u16, preserving alpha.
    ///
    /// Uses the same RGB kernel as [`linear_to_srgb_u16_slice`] at every row
    /// length. Alpha is quantized linearly. Incomplete trailing pixels are ignored.
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    #[inline]
    pub fn linear_to_srgb_u16_rgba_slice(input: &[f32], output: &mut [u16]) {
        {
            match (&input.len(), &output.len()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            }
        };
        let len = input.len() / 4 * 4;
        linear_to_srgb_u16_slice(&input[..len], &mut output[..len]);
        for (src, dst) in input[..len]
            .as_chunks::<4>()
            .0
            .iter()
            .zip(output[..len].as_chunks_mut::<4>().0)
        {
            dst[3] = (f64::from(src[3]).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    /// Convert linear RGBA f32 to sRGB u16, preserving alpha. Sqrt-indexed LUT
    /// encode (~10× faster, max ±1 roundtrip error).
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    fn linear_to_srgb_u16_rgba_slice_fast_tier_wasm128(
        token: archmage::Wasm128Token,
        input: &[f32],
        output: &mut [u16],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let lut = crate::u16_lut::encode_lut();
        let max_idx = crate::u16_lut::ENCODE_LUT_N - 1;
        let scale = crate::u16_lut::ENCODE_SQRT_SCALE;
        let (in_chunks, in_rem) = input.as_chunks::<16>();
        let (out_chunks, out_rem) = output.as_chunks_mut::<16>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let v = f32x16::from_array_t(token, *inp);
            let clamped = v.max(f32x16::zero_t(token)).min(f32x16::splat_t(token, 1.0));
            let idx_f = clamped
                .sqrt()
                .mla(f32x16::splat_t(token, scale), f32x16::splat_t(token, 0.5));
            let idx_i = idx_f.to_i32().to_array();
            for px in 0..4 {
                let base = px * 4;
                out[base] = lut[(idx_i[base] as usize).min(max_idx)];
                out[base + 1] = lut[(idx_i[base + 1] as usize).min(max_idx)];
                out[base + 2] = lut[(idx_i[base + 2] as usize).min(max_idx)];
                out[base + 3] = (f64::from(inp[base + 3]).clamp(0.0, 1.0) * 65535.0
                    + 0.5) as u16;
            }
        }
        for (inp, out) in in_rem
            .as_chunks::<4>()
            .0
            .iter()
            .zip(out_rem.as_chunks_mut::<4>().0.iter_mut())
        {
            out[0] = crate::scalar::linear_to_srgb_u16_fast(inp[0]);
            out[1] = crate::scalar::linear_to_srgb_u16_fast(inp[1]);
            out[2] = crate::scalar::linear_to_srgb_u16_fast(inp[2]);
            out[3] = (f64::from(inp[3]).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
    }
    /// Convert linear RGBA f32 to sRGB u16, preserving alpha. Sqrt-indexed LUT
    /// encode (~10× faster, max ±1 roundtrip error).
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    fn linear_to_srgb_u16_rgba_slice_fast_tier_scalar(
        token: archmage::ScalarToken,
        input: &[f32],
        output: &mut [u16],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let lut = crate::u16_lut::encode_lut();
        let max_idx = crate::u16_lut::ENCODE_LUT_N - 1;
        let scale = crate::u16_lut::ENCODE_SQRT_SCALE;
        let (in_chunks, in_rem) = input.as_chunks::<16>();
        let (out_chunks, out_rem) = output.as_chunks_mut::<16>();
        for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let v = f32x16::from_array_t(token, *inp);
            let clamped = v.max(f32x16::zero_t(token)).min(f32x16::splat_t(token, 1.0));
            let idx_f = clamped
                .sqrt()
                .mla(f32x16::splat_t(token, scale), f32x16::splat_t(token, 0.5));
            let idx_i = idx_f.to_i32().to_array();
            for px in 0..4 {
                let base = px * 4;
                out[base] = lut[(idx_i[base] as usize).min(max_idx)];
                out[base + 1] = lut[(idx_i[base + 1] as usize).min(max_idx)];
                out[base + 2] = lut[(idx_i[base + 2] as usize).min(max_idx)];
                out[base + 3] = (f64::from(inp[base + 3]).clamp(0.0, 1.0) * 65535.0
                    + 0.5) as u16;
            }
        }
        for (inp, out) in in_rem
            .as_chunks::<4>()
            .0
            .iter()
            .zip(out_rem.as_chunks_mut::<4>().0.iter_mut())
        {
            out[0] = crate::scalar::linear_to_srgb_u16_fast(inp[0]);
            out[1] = crate::scalar::linear_to_srgb_u16_fast(inp[1]);
            out[2] = crate::scalar::linear_to_srgb_u16_fast(inp[2]);
            out[3] = (f64::from(inp[3]).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
    }
    /// Convert linear RGBA f32 to sRGB u16, preserving alpha. Sqrt-indexed LUT
    /// encode with SIMD-computed indices (~10× faster than the polynomial
    /// variant, max ±1 u16 roundtrip error).
    ///
    /// # Panics
    /// Panics if `input.len() != output.len()`.
    #[inline]
    pub fn linear_to_srgb_u16_rgba_slice_fast(input: &[f32], output: &mut [u16]) {
        {
            match (&input.len(), &output.len()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            }
        };
        {
            '__incant: {
                #[allow(unused_imports)]
                use archmage::SimdToken;
                {
                    if let Some(__t) = archmage::Wasm128Token::summon() {
                        break '__incant linear_to_srgb_u16_rgba_slice_fast_tier_wasm128(
                            __t,
                            input,
                            output,
                        );
                    }
                }
                linear_to_srgb_u16_rgba_slice_fast_tier_scalar(
                    archmage::ScalarToken,
                    input,
                    output,
                )
            }
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn __attune_gamma_to_linear_slice_tier_wasm128(
        _token: ::archmage::Wasm128Token,
        values: &mut [f32],
        gamma: f32,
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::Wasm128Token>;
        {
            '__attune_call_0: {
                {
                    crate::target_features::x16::gamma_to_linear_slice_wasm128(
                        values,
                        gamma,
                    )
                }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn gamma_to_linear_slice_tier_wasm128_t(
        __archmage_token: ::archmage::Wasm128Token,
        values: &mut [f32],
        gamma: f32,
    ) {
        __attune_gamma_to_linear_slice_tier_wasm128(__archmage_token, values, gamma)
    }
    fn gamma_to_linear_slice_tier_scalar_t(
        _token: ScalarToken,
        values: &mut [f32],
        gamma: f32,
    ) {
        crate::target_features::x16::gamma_to_linear_slice_scalar(values, gamma);
    }
    /// Convert gamma-encoded f32 values to linear in-place using a custom gamma.
    ///
    /// Applies the power function to **every element of the slice** — if you
    /// pass RGBA data, alpha will be decoded too. Use
    /// [`gamma_to_linear_rgba_slice`] for alpha-preserving RGBA conversion.
    ///
    /// Uses AVX-512 (16-wide), AVX2+FMA (8-wide), NEON (4-wide), WASM SIMD128 (4-wide), or scalar depending on CPU.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::gamma_to_linear_slice;
    ///
    /// let mut values = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
    /// gamma_to_linear_slice(&mut values, 2.2);
    /// ```
    #[inline]
    pub fn gamma_to_linear_slice(values: &mut [f32], gamma: f32) {
        {
            '__attune_call_0: {
                {
                    if let Some(__attune_proof) = <::archmage::Wasm128Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 gamma_to_linear_slice_tier_wasm128_t(
                            __attune_proof,
                            values,
                            gamma,
                        );
                    }
                }
                {
                    gamma_to_linear_slice_tier_scalar_t(
                        ::archmage::ScalarToken,
                        values,
                        gamma,
                    )
                }
            }
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn __attune_linear_to_gamma_slice_tier_wasm128(
        _token: ::archmage::Wasm128Token,
        values: &mut [f32],
        gamma: f32,
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::Wasm128Token>;
        {
            '__attune_call_0: {
                {
                    crate::target_features::x16::linear_to_gamma_slice_wasm128(
                        values,
                        gamma,
                    )
                }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn linear_to_gamma_slice_tier_wasm128_t(
        __archmage_token: ::archmage::Wasm128Token,
        values: &mut [f32],
        gamma: f32,
    ) {
        __attune_linear_to_gamma_slice_tier_wasm128(__archmage_token, values, gamma)
    }
    fn linear_to_gamma_slice_tier_scalar_t(
        _token: ScalarToken,
        values: &mut [f32],
        gamma: f32,
    ) {
        crate::target_features::x16::linear_to_gamma_slice_scalar(values, gamma);
    }
    /// Convert linear f32 values to gamma-encoded in-place using a custom gamma.
    ///
    /// Applies the power function to **every element of the slice** — if you
    /// pass RGBA data, alpha will be encoded too. Use
    /// [`linear_to_gamma_rgba_slice`] for alpha-preserving RGBA conversion.
    ///
    /// Uses AVX-512 (16-wide), AVX2+FMA (8-wide), NEON (4-wide), WASM SIMD128 (4-wide), or scalar depending on CPU.
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::linear_to_gamma_slice;
    ///
    /// let mut values = vec![0.0f32, 0.1, 0.2, 0.5, 1.0];
    /// linear_to_gamma_slice(&mut values, 2.2);
    /// ```
    #[inline]
    pub fn linear_to_gamma_slice(values: &mut [f32], gamma: f32) {
        {
            '__attune_call_0: {
                {
                    if let Some(__attune_proof) = <::archmage::Wasm128Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 linear_to_gamma_slice_tier_wasm128_t(
                            __attune_proof,
                            values,
                            gamma,
                        );
                    }
                }
                {
                    linear_to_gamma_slice_tier_scalar_t(
                        ::archmage::ScalarToken,
                        values,
                        gamma,
                    )
                }
            }
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn gamma_to_linear_rgba_slice_tier_wasm128(
        token: archmage::Wasm128Token,
        values: &mut [f32],
        gamma: f32,
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            let v = f32x16::from_array_t(token, *chunk);
            let clamped = v.max(f32x16::zero_t(token)).min(f32x16::splat_t(token, 1.0));
            *chunk = crate::tf::fast_math::pow_x16(token, clamped, gamma).to_array();
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    fn gamma_to_linear_rgba_slice_tier_scalar(
        token: archmage::ScalarToken,
        values: &mut [f32],
        gamma: f32,
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            let v = f32x16::from_array_t(token, *chunk);
            let clamped = v.max(f32x16::zero_t(token)).min(f32x16::splat_t(token, 1.0));
            *chunk = crate::tf::fast_math::pow_x16(token, clamped, gamma).to_array();
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    /// Convert gamma-encoded RGBA f32 values to linear in-place using a custom
    /// gamma, preserving alpha.
    ///
    /// Expects interleaved RGBA data (`[R, G, B, A, R, G, B, A, ...]`).
    /// Every 4th element (alpha) is left unchanged. R/G/B are clamped to
    /// \[0, 1\] and decoded via `x.powf(gamma)`, exactly as
    /// [`gamma_to_linear_slice`] does for every element. Trailing elements
    /// that don't form a complete RGBA pixel are ignored.
    ///
    /// Straight (non-premultiplied) alpha in, straight alpha out. For the
    /// fused decode-and-premultiply form see
    /// [`srgb_to_linear_premultiply_rgba_slice`].
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::gamma_to_linear_rgba_slice;
    ///
    /// let mut rgba = vec![0.5f32, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 2.0];
    /// gamma_to_linear_rgba_slice(&mut rgba, 2.2);
    /// assert!((rgba[0] - 0.5f32.powf(2.2)).abs() < 1e-3);
    /// assert_eq!(rgba[3], 0.75); // alpha untouched
    /// assert_eq!(rgba[7], 2.0);  // alpha untouched, not clamped
    /// ```
    #[inline]
    pub fn gamma_to_linear_rgba_slice(values: &mut [f32], gamma: f32) {
        '__incant: {
            #[allow(unused_imports)]
            use archmage::SimdToken;
            {
                if let Some(__t) = archmage::Wasm128Token::summon() {
                    break '__incant gamma_to_linear_rgba_slice_tier_wasm128(
                        __t,
                        values,
                        gamma,
                    );
                }
            }
            gamma_to_linear_rgba_slice_tier_scalar(archmage::ScalarToken, values, gamma)
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn linear_to_gamma_rgba_slice_tier_wasm128(
        token: archmage::Wasm128Token,
        values: &mut [f32],
        gamma: f32,
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            let v = f32x16::from_array_t(token, *chunk);
            let clamped = v.max(f32x16::zero_t(token)).min(f32x16::splat_t(token, 1.0));
            *chunk = crate::tf::fast_math::pow_x16(token, clamped, 1.0 / gamma)
                .to_array();
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    fn linear_to_gamma_rgba_slice_tier_scalar(
        token: archmage::ScalarToken,
        values: &mut [f32],
        gamma: f32,
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            let v = f32x16::from_array_t(token, *chunk);
            let clamped = v.max(f32x16::zero_t(token)).min(f32x16::splat_t(token, 1.0));
            *chunk = crate::tf::fast_math::pow_x16(token, clamped, 1.0 / gamma)
                .to_array();
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    /// Convert linear RGBA f32 values to gamma-encoded in-place using a custom
    /// gamma, preserving alpha.
    ///
    /// Expects interleaved RGBA data (`[R, G, B, A, R, G, B, A, ...]`).
    /// Every 4th element (alpha) is left unchanged. R/G/B are clamped to
    /// \[0, 1\] and encoded via `x.powf(1.0 / gamma)`, exactly as
    /// [`linear_to_gamma_slice`] does for every element. Trailing elements
    /// that don't form a complete RGBA pixel are ignored.
    ///
    /// Straight (non-premultiplied) alpha in, straight alpha out. For the
    /// fused unpremultiply-and-encode form see
    /// [`unpremultiply_linear_to_srgb_rgba_slice`].
    ///
    /// # Example
    /// ```
    /// use linear_srgb::default::linear_to_gamma_rgba_slice;
    ///
    /// let mut rgba = vec![0.2f32, 0.2, 0.2, 0.75, 0.8, 0.8, 0.8, 2.0];
    /// linear_to_gamma_rgba_slice(&mut rgba, 2.2);
    /// assert!((rgba[0] - 0.2f32.powf(1.0 / 2.2)).abs() < 1e-3);
    /// assert_eq!(rgba[3], 0.75); // alpha untouched
    /// assert_eq!(rgba[7], 2.0);  // alpha untouched, not clamped
    /// ```
    #[inline]
    pub fn linear_to_gamma_rgba_slice(values: &mut [f32], gamma: f32) {
        '__incant: {
            #[allow(unused_imports)]
            use archmage::SimdToken;
            {
                if let Some(__t) = archmage::Wasm128Token::summon() {
                    break '__incant linear_to_gamma_rgba_slice_tier_wasm128(
                        __t,
                        values,
                        gamma,
                    );
                }
            }
            linear_to_gamma_rgba_slice_tier_scalar(archmage::ScalarToken, values, gamma)
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn gamma_to_linear_premultiply_rgba_slice_tier_wasm128(
        token: archmage::Wasm128Token,
        values: &mut [f32],
        gamma: f32,
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            *chunk = mt_gamma_to_linear(token, *chunk, gamma);
            let v = f32x16::from_array_t(token, *chunk);
            let alpha = f32x16::from_array_t(
                token,
                [
                    a[0],
                    a[0],
                    a[0],
                    1.0,
                    a[1],
                    a[1],
                    a[1],
                    1.0,
                    a[2],
                    a[2],
                    a[2],
                    1.0,
                    a[3],
                    a[3],
                    a[3],
                    1.0,
                ],
            );
            *chunk = (v * alpha).to_array();
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    fn gamma_to_linear_premultiply_rgba_slice_tier_scalar(
        token: archmage::ScalarToken,
        values: &mut [f32],
        gamma: f32,
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            *chunk = mt_gamma_to_linear(token, *chunk, gamma);
            let v = f32x16::from_array_t(token, *chunk);
            let alpha = f32x16::from_array_t(
                token,
                [
                    a[0],
                    a[0],
                    a[0],
                    1.0,
                    a[1],
                    a[1],
                    a[1],
                    1.0,
                    a[2],
                    a[2],
                    a[2],
                    1.0,
                    a[3],
                    a[3],
                    a[3],
                    1.0,
                ],
            );
            *chunk = (v * alpha).to_array();
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    /// Convert gamma-encoded RGBA to linear premultiplied RGBA in-place.
    ///
    /// Each RGB channel is decoded via `x.powf(gamma)`, then multiplied by the
    /// alpha channel — in a single SIMD pass. Alpha is left unchanged.
    ///
    /// Input: `[R_gamma, G_gamma, B_gamma, A, ...]` (straight alpha)
    /// Output: `[R_linear*A, G_linear*A, B_linear*A, A, ...]` (premultiplied)
    ///
    /// Trailing elements that don't form a complete RGBA pixel are ignored.
    ///
    /// # Example
    /// ```
    /// #[allow(deprecated)]
    /// # fn main() {
    /// use linear_srgb::default::gamma_to_linear_premultiply_rgba_slice;
    ///
    /// let mut rgba = vec![0.5f32, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0];
    /// gamma_to_linear_premultiply_rgba_slice(&mut rgba, 2.2);
    /// // gamma_to_linear(0.5, 2.2) ≈ 0.218, × 0.5 ≈ 0.109
    /// assert!(rgba[0] < 0.12);
    /// assert_eq!(rgba[3], 0.5); // alpha preserved
    /// # }
    /// ```
    #[deprecated(
        since = "0.6.4",
        note = "use srgb_to_linear_premultiply_rgba_slice instead; gamma-based premultiply will be removed in a future release"
    )]
    #[inline]
    pub fn gamma_to_linear_premultiply_rgba_slice(values: &mut [f32], gamma: f32) {
        '__incant: {
            #[allow(unused_imports)]
            use archmage::SimdToken;
            {
                if let Some(__t) = archmage::Wasm128Token::summon() {
                    break '__incant gamma_to_linear_premultiply_rgba_slice_tier_wasm128(
                        __t,
                        values,
                        gamma,
                    );
                }
            }
            gamma_to_linear_premultiply_rgba_slice_tier_scalar(
                archmage::ScalarToken,
                values,
                gamma,
            )
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn unpremultiply_linear_to_gamma_rgba_slice_tier_wasm128(
        token: archmage::Wasm128Token,
        values: &mut [f32],
        gamma: f32,
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::Wasm128Token>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            let inv = [
                if a[0] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[0] } else { 0.0 },
                if a[1] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[1] } else { 0.0 },
                if a[2] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[2] } else { 0.0 },
                if a[3] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[3] } else { 0.0 },
            ];
            let v = f32x16::from_array_t(token, *chunk);
            let inv_alpha = f32x16::from_array_t(
                token,
                [
                    inv[0],
                    inv[0],
                    inv[0],
                    1.0,
                    inv[1],
                    inv[1],
                    inv[1],
                    1.0,
                    inv[2],
                    inv[2],
                    inv[2],
                    1.0,
                    inv[3],
                    inv[3],
                    inv[3],
                    1.0,
                ],
            );
            let unpremul = (v * inv_alpha).to_array();
            *chunk = mt_linear_to_gamma(token, unpremul, gamma);
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    fn unpremultiply_linear_to_gamma_rgba_slice_tier_scalar(
        token: archmage::ScalarToken,
        values: &mut [f32],
        gamma: f32,
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::ScalarToken>;
        let (chunks, remainder) = values.as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut tail = tail_block::<16>(remainder);
        for chunk in chunks.iter_mut().chain(tail.as_mut()) {
            let a = [chunk[3], chunk[7], chunk[11], chunk[15]];
            let inv = [
                if a[0] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[0] } else { 0.0 },
                if a[1] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[1] } else { 0.0 },
                if a[2] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[2] } else { 0.0 },
                if a[3] > crate::UNPREMUL_ALPHA_THRESHOLD { 1.0 / a[3] } else { 0.0 },
            ];
            let v = f32x16::from_array_t(token, *chunk);
            let inv_alpha = f32x16::from_array_t(
                token,
                [
                    inv[0],
                    inv[0],
                    inv[0],
                    1.0,
                    inv[1],
                    inv[1],
                    inv[1],
                    1.0,
                    inv[2],
                    inv[2],
                    inv[2],
                    1.0,
                    inv[3],
                    inv[3],
                    inv[3],
                    1.0,
                ],
            );
            let unpremul = (v * inv_alpha).to_array();
            *chunk = mt_linear_to_gamma(token, unpremul, gamma);
            [chunk[3], chunk[7], chunk[11], chunk[15]] = a;
        }
        store_tail(remainder, tail.as_ref());
    }
    /// Convert linear premultiplied RGBA to gamma-encoded straight-alpha RGBA in-place.
    ///
    /// Each RGB channel is divided by alpha, then encoded via `x.powf(1.0 / gamma)`
    /// — in a single SIMD pass. Alpha is left unchanged.
    ///
    /// Input: `[R_linear*A, G_linear*A, B_linear*A, A, ...]` (premultiplied)
    /// Output: `[R_gamma, G_gamma, B_gamma, A, ...]` (straight alpha)
    ///
    /// When alpha is zero, the RGB channels are set to zero. Trailing elements
    /// that don't form a complete RGBA pixel are ignored.
    ///
    /// # Example
    /// ```
    /// #[allow(deprecated)]
    /// # fn main() {
    /// use linear_srgb::default::{gamma_to_linear_premultiply_rgba_slice,
    ///     unpremultiply_linear_to_gamma_rgba_slice};
    ///
    /// let mut rgba = vec![0.5f32, 0.5, 0.5, 0.75, 0.0, 0.0, 0.0, 0.0];
    /// gamma_to_linear_premultiply_rgba_slice(&mut rgba, 2.2);
    /// unpremultiply_linear_to_gamma_rgba_slice(&mut rgba, 2.2);
    /// assert!((rgba[0] - 0.5).abs() < 1e-3); // roundtrips
    /// assert_eq!(rgba[3], 0.75);              // alpha preserved
    /// assert_eq!(rgba[4], 0.0);               // transparent pixel stays zero
    /// # }
    /// ```
    #[deprecated(
        since = "0.6.4",
        note = "use unpremultiply_linear_to_srgb_rgba_slice instead; gamma-based unpremultiply will be removed in a future release"
    )]
    #[inline]
    pub fn unpremultiply_linear_to_gamma_rgba_slice(values: &mut [f32], gamma: f32) {
        '__incant: {
            #[allow(unused_imports)]
            use archmage::SimdToken;
            {
                if let Some(__t) = archmage::Wasm128Token::summon() {
                    break '__incant unpremultiply_linear_to_gamma_rgba_slice_tier_wasm128(
                        __t,
                        values,
                        gamma,
                    );
                }
            }
            unpremultiply_linear_to_gamma_rgba_slice_tier_scalar(
                archmage::ScalarToken,
                values,
                gamma,
            )
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn __attune_bt709_to_linear_slice_tier_wasm128(
        _token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::Wasm128Token>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::bt709_to_linear_slice_wasm128(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn bt709_to_linear_slice_tier_wasm128_t(
        __archmage_token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        __attune_bt709_to_linear_slice_tier_wasm128(__archmage_token, values)
    }
    fn bt709_to_linear_slice_tier_scalar_t(_token: ScalarToken, values: &mut [f32]) {
        crate::target_features::x16::bt709_to_linear_slice_scalar(values);
    }
    /// Convert BT.709 signal f32 values to linear in-place.
    ///
    /// Applies the transfer function to **every element of the slice** — if you
    /// pass RGBA data, alpha will be decoded too. Use
    /// [`bt709_to_linear_rgba_slice`] for alpha-preserving RGBA conversion.
    ///
    /// Uses AVX-512 (16-wide), AVX2+FMA (8-wide), NEON (4-wide), WASM SIMD128
    /// (4-wide), or scalar depending on CPU.
    #[inline]
    pub fn bt709_to_linear_slice(values: &mut [f32]) {
        {
            '__attune_call_0: {
                {
                    if let Some(__attune_proof) = <::archmage::Wasm128Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 bt709_to_linear_slice_tier_wasm128_t(
                            __attune_proof,
                            values,
                        );
                    }
                }
                { bt709_to_linear_slice_tier_scalar_t(::archmage::ScalarToken, values) }
            }
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn __attune_linear_to_bt709_slice_tier_wasm128(
        _token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::Wasm128Token>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::linear_to_bt709_slice_wasm128(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn linear_to_bt709_slice_tier_wasm128_t(
        __archmage_token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        __attune_linear_to_bt709_slice_tier_wasm128(__archmage_token, values)
    }
    fn linear_to_bt709_slice_tier_scalar_t(_token: ScalarToken, values: &mut [f32]) {
        crate::target_features::x16::linear_to_bt709_slice_scalar(values);
    }
    /// Convert linear f32 values to BT.709 signal in-place.
    ///
    /// Applies the transfer function to **every element of the slice** — if you
    /// pass RGBA data, alpha will be encoded too. Use
    /// [`linear_to_bt709_rgba_slice`] for alpha-preserving RGBA conversion.
    ///
    /// Uses AVX-512 (16-wide), AVX2+FMA (8-wide), NEON (4-wide), WASM SIMD128
    /// (4-wide), or scalar depending on CPU.
    #[inline]
    pub fn linear_to_bt709_slice(values: &mut [f32]) {
        {
            '__attune_call_0: {
                {
                    if let Some(__attune_proof) = <::archmage::Wasm128Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 linear_to_bt709_slice_tier_wasm128_t(
                            __attune_proof,
                            values,
                        );
                    }
                }
                { linear_to_bt709_slice_tier_scalar_t(::archmage::ScalarToken, values) }
            }
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn __attune_pq_to_linear_slice_tier_wasm128(
        _token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::Wasm128Token>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::pq_to_linear_slice_wasm128(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn pq_to_linear_slice_tier_wasm128_t(
        __archmage_token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        __attune_pq_to_linear_slice_tier_wasm128(__archmage_token, values)
    }
    fn pq_to_linear_slice_tier_scalar_t(_token: ScalarToken, values: &mut [f32]) {
        crate::target_features::x16::pq_to_linear_slice_scalar(values);
    }
    /// Convert PQ (ST 2084) signal f32 values to linear in-place.
    ///
    /// Applies the transfer function to **every element of the slice** — if you
    /// pass RGBA data, alpha will be decoded too. Use [`pq_to_linear_rgba_slice`]
    /// for alpha-preserving RGBA conversion.
    ///
    /// Uses AVX-512 (16-wide), AVX2+FMA (8-wide), NEON (4-wide), WASM SIMD128
    /// (4-wide), or scalar depending on CPU.
    #[inline]
    pub fn pq_to_linear_slice(values: &mut [f32]) {
        {
            '__attune_call_0: {
                {
                    if let Some(__attune_proof) = <::archmage::Wasm128Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 pq_to_linear_slice_tier_wasm128_t(
                            __attune_proof,
                            values,
                        );
                    }
                }
                { pq_to_linear_slice_tier_scalar_t(::archmage::ScalarToken, values) }
            }
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn __attune_linear_to_pq_slice_tier_wasm128(
        _token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::Wasm128Token>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::linear_to_pq_slice_wasm128(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn linear_to_pq_slice_tier_wasm128_t(
        __archmage_token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        __attune_linear_to_pq_slice_tier_wasm128(__archmage_token, values)
    }
    fn linear_to_pq_slice_tier_scalar_t(_token: ScalarToken, values: &mut [f32]) {
        crate::target_features::x16::linear_to_pq_slice_scalar(values);
    }
    /// Convert linear f32 values to PQ (ST 2084) signal in-place.
    ///
    /// Applies the transfer function to **every element of the slice** — if you
    /// pass RGBA data, alpha will be encoded too. Use [`linear_to_pq_rgba_slice`]
    /// for alpha-preserving RGBA conversion.
    ///
    /// Uses AVX-512 (16-wide), AVX2+FMA (8-wide), NEON (4-wide), WASM SIMD128
    /// (4-wide), or scalar depending on CPU.
    #[inline]
    pub fn linear_to_pq_slice(values: &mut [f32]) {
        {
            '__attune_call_0: {
                {
                    if let Some(__attune_proof) = <::archmage::Wasm128Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 linear_to_pq_slice_tier_wasm128_t(
                            __attune_proof,
                            values,
                        );
                    }
                }
                { linear_to_pq_slice_tier_scalar_t(::archmage::ScalarToken, values) }
            }
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn __attune_hlg_to_linear_slice_tier_wasm128(
        _token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::Wasm128Token>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::hlg_to_linear_slice_wasm128(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn hlg_to_linear_slice_tier_wasm128_t(
        __archmage_token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        __attune_hlg_to_linear_slice_tier_wasm128(__archmage_token, values)
    }
    fn hlg_to_linear_slice_tier_scalar_t(_token: ScalarToken, values: &mut [f32]) {
        crate::target_features::x16::hlg_to_linear_slice_scalar(values);
    }
    /// Convert HLG (ARIB STD-B67) signal f32 values to linear in-place.
    ///
    /// Applies the transfer function to **every element of the slice** — if you
    /// pass RGBA data, alpha will be decoded too. Use [`hlg_to_linear_rgba_slice`]
    /// for alpha-preserving RGBA conversion.
    ///
    /// Uses AVX-512 (16-wide), AVX2+FMA (8-wide), NEON (4-wide), WASM SIMD128
    /// (4-wide), or scalar depending on CPU.
    #[inline]
    pub fn hlg_to_linear_slice(values: &mut [f32]) {
        {
            '__attune_call_0: {
                {
                    if let Some(__attune_proof) = <::archmage::Wasm128Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 hlg_to_linear_slice_tier_wasm128_t(
                            __attune_proof,
                            values,
                        );
                    }
                }
                { hlg_to_linear_slice_tier_scalar_t(::archmage::ScalarToken, values) }
            }
        }
    }
    #[target_feature(enable = "simd128")]
    #[inline]
    fn __attune_linear_to_hlg_slice_tier_wasm128(
        _token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::Wasm128Token>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::linear_to_hlg_slice_wasm128(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn linear_to_hlg_slice_tier_wasm128_t(
        __archmage_token: ::archmage::Wasm128Token,
        values: &mut [f32],
    ) {
        __attune_linear_to_hlg_slice_tier_wasm128(__archmage_token, values)
    }
    fn linear_to_hlg_slice_tier_scalar_t(_token: ScalarToken, values: &mut [f32]) {
        crate::target_features::x16::linear_to_hlg_slice_scalar(values);
    }
    /// Convert linear f32 values to HLG (ARIB STD-B67) signal in-place.
    ///
    /// Applies the transfer function to **every element of the slice** — if you
    /// pass RGBA data, alpha will be encoded too. Use [`linear_to_hlg_rgba_slice`]
    /// for alpha-preserving RGBA conversion.
    ///
    /// Uses AVX-512 (16-wide), AVX2+FMA (8-wide), NEON (4-wide), WASM SIMD128
    /// (4-wide), or scalar depending on CPU.
    #[inline]
    pub fn linear_to_hlg_slice(values: &mut [f32]) {
        {
            '__attune_call_0: {
                {
                    if let Some(__attune_proof) = <::archmage::Wasm128Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 linear_to_hlg_slice_tier_wasm128_t(
                            __attune_proof,
                            values,
                        );
                    }
                }
                { linear_to_hlg_slice_tier_scalar_t(::archmage::ScalarToken, values) }
            }
        }
    }
    ///Convert BT.709 signal RGBA f32 values to linear in-place, preserving alpha.
    /// Alpha is preserved bit-for-bit; incomplete trailing pixels are ignored.
    pub fn bt709_to_linear_rgba_slice(values: &mut [f32]) {
        let len = values.len() / 4 * 4;
        for chunk in values[..len].chunks_mut(256) {
            let mut alpha = [0.0f32; 64];
            for (i, px) in chunk.as_chunks::<4>().0.iter().enumerate() {
                alpha[i] = px[3];
            }
            bt709_to_linear_slice(chunk);
            for (i, px) in chunk.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                px[3] = alpha[i];
            }
        }
    }
    ///Convert linear RGBA f32 values to BT.709 signal in-place, preserving alpha.
    /// Alpha is preserved bit-for-bit; incomplete trailing pixels are ignored.
    pub fn linear_to_bt709_rgba_slice(values: &mut [f32]) {
        let len = values.len() / 4 * 4;
        for chunk in values[..len].chunks_mut(256) {
            let mut alpha = [0.0f32; 64];
            for (i, px) in chunk.as_chunks::<4>().0.iter().enumerate() {
                alpha[i] = px[3];
            }
            linear_to_bt709_slice(chunk);
            for (i, px) in chunk.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                px[3] = alpha[i];
            }
        }
    }
    ///Convert PQ (ST 2084) signal RGBA f32 values to linear in-place, preserving alpha.
    /// Alpha is preserved bit-for-bit; incomplete trailing pixels are ignored.
    pub fn pq_to_linear_rgba_slice(values: &mut [f32]) {
        let len = values.len() / 4 * 4;
        for chunk in values[..len].chunks_mut(256) {
            let mut alpha = [0.0f32; 64];
            for (i, px) in chunk.as_chunks::<4>().0.iter().enumerate() {
                alpha[i] = px[3];
            }
            pq_to_linear_slice(chunk);
            for (i, px) in chunk.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                px[3] = alpha[i];
            }
        }
    }
    ///Convert linear RGBA f32 values to PQ (ST 2084) signal in-place, preserving alpha.
    /// Alpha is preserved bit-for-bit; incomplete trailing pixels are ignored.
    pub fn linear_to_pq_rgba_slice(values: &mut [f32]) {
        let len = values.len() / 4 * 4;
        for chunk in values[..len].chunks_mut(256) {
            let mut alpha = [0.0f32; 64];
            for (i, px) in chunk.as_chunks::<4>().0.iter().enumerate() {
                alpha[i] = px[3];
            }
            linear_to_pq_slice(chunk);
            for (i, px) in chunk.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                px[3] = alpha[i];
            }
        }
    }
    ///Convert HLG (ARIB STD-B67) signal RGBA f32 values to linear in-place, preserving alpha.
    /// Alpha is preserved bit-for-bit; incomplete trailing pixels are ignored.
    pub fn hlg_to_linear_rgba_slice(values: &mut [f32]) {
        let len = values.len() / 4 * 4;
        for chunk in values[..len].chunks_mut(256) {
            let mut alpha = [0.0f32; 64];
            for (i, px) in chunk.as_chunks::<4>().0.iter().enumerate() {
                alpha[i] = px[3];
            }
            hlg_to_linear_slice(chunk);
            for (i, px) in chunk.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                px[3] = alpha[i];
            }
        }
    }
    ///Convert linear RGBA f32 values to HLG (ARIB STD-B67) signal in-place, preserving alpha.
    /// Alpha is preserved bit-for-bit; incomplete trailing pixels are ignored.
    pub fn linear_to_hlg_rgba_slice(values: &mut [f32]) {
        let len = values.len() / 4 * 4;
        for chunk in values[..len].chunks_mut(256) {
            let mut alpha = [0.0f32; 64];
            for (i, px) in chunk.as_chunks::<4>().0.iter().enumerate() {
                alpha[i] = px[3];
            }
            linear_to_hlg_slice(chunk);
            for (i, px) in chunk.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                px[3] = alpha[i];
            }
        }
    }
}
mod mlaf {
    //! Fused multiply-add abstraction with compile-time hardware detection.
    //!
    //! When FMA is available (x86 with FMA feature or ARM64 NEON), uses single-cycle
    //! hardware FMA instructions. Otherwise falls back to separate multiply+add.
    /// Computes `acc + a * b` (fallback without hardware FMA).
    #[inline(always)]
    pub fn mlaf<T: core::ops::Add<Output = T> + core::ops::Mul<Output = T>>(
        acc: T,
        a: T,
        b: T,
    ) -> T {
        acc + a * b
    }
    /// Computes `a * b + c` (reordered FMA for ergonomics).
    ///
    /// This matches the mathematical notation `a * b + c` more naturally.
    #[inline(always)]
    pub fn fmla<T>(a: T, b: T, c: T) -> T
    where
        T: MlaCompatible,
    {
        T::mla(c, a, b)
    }
    /// Trait for types that support MLA operations.
    /// This abstracts over the cfg-dependent mlaf implementations.
    pub trait MlaCompatible: Sized {
        fn mla(acc: Self, a: Self, b: Self) -> Self;
    }
    impl MlaCompatible for f32 {
        #[inline(always)]
        fn mla(acc: Self, a: Self, b: Self) -> Self {
            mlaf(acc, a, b)
        }
    }
    impl MlaCompatible for f64 {
        #[inline(always)]
        fn mla(acc: Self, a: Self, b: Self) -> Self {
            mlaf(acc, a, b)
        }
    }
}
pub(crate) mod rational_poly {
    //! Rational polynomial sRGB transfer function approximation.
    //!
    //! Single source of truth for all polynomial constants used across scalar,
    //! SIMD, magetypes rites (x4/x8/x16), and mage implementations.
    //!
    //! Evaluates P(x)/Q(x) with Horner multiply-adds and one division per direction.
    //!
    //! Uses C0-continuous thresholds (moxcms) that eliminate the ~2.3e-9
    //! discontinuity present in the IEC 61966-2-1 textbook constants. The
    //! polynomials are fitted to the C0-continuous curve with a boundary
    //! constraint ensuring exact continuity at the piecewise threshold.
    //!
    //! # Accuracy (exhaustive f32 sweep)
    //!
    //! | Direction | Max ULP (power segment) | Max ULP (boundary) | Avg ULP |
    //! |---|---|---|---|
    //! | sRGB → linear | 9 | 1 | ~3.4 |
    //! | linear → sRGB | 10 | 0 | ~3.0 |
    //!
    //! The scalar evaluator uses f64 intermediate precision, which guarantees
    //! perfect monotonicity (zero reversals across all ~1B f32 values in \[0, 1\]).
    //!
    //! Roundtrip error is sub-U16 (< 1/65535) across the full [0, 1] range.
    #[allow(unused_imports)]
    use num_traits::Float;
    /// sRGB EOTF (encoded → linear) numerator coefficients.
    /// Polyfit: weighted constraint w=50000, 8 restarts + f32 ULP search.
    /// Max 9 ULP, boundary 1 ULP (vs 11/1 previously).
    #[allow(clippy::excessive_precision)]
    pub(crate) const S2L_P: [f32; 5] = [
        1.672_814_6e-2,
        8.089_536_4e-1,
        1.288_439_66e1,
        6.854_374_7e1,
        8.224_625_4e1,
    ];
    /// sRGB EOTF (encoded → linear) denominator coefficients.
    #[allow(clippy::excessive_precision)]
    pub(crate) const S2L_Q: [f32; 5] = [
        2.003_807_83e1,
        9.681_468_2e1,
        5.377_303_3e1,
        -7.125_638,
        1.0,
    ];
    /// sRGB inverse EOTF (linear → encoded) numerator coefficients.
    /// Evaluated on `sqrt(linear)`.
    /// Polyfit: weighted constraint w=1000, 8 restarts + f32 ULP search.
    /// Max 10 ULP, boundary 0 ULP (vs 14/0 previously).
    #[allow(clippy::excessive_precision)]
    pub(crate) const L2S_P: [f32; 5] = [
        -1.356_440_97e-2,
        9.336_896_24e-2,
        1.157_270_53e1,
        5.002_124_79e1,
        2.792_420_96e1,
    ];
    /// sRGB inverse EOTF (linear → encoded) denominator coefficients.
    /// Evaluated on `sqrt(linear)`.
    #[allow(clippy::excessive_precision)]
    pub(crate) const L2S_Q: [f32; 5] = [
        2.633_812_73e-1,
        8.998_917_58,
        4.477_739_72e1,
        3.455_830_76e1,
        1.0,
    ];
    /// Extended sRGB EOTF numerator (degree 6, fitted to [threshold, 8]).
    /// Sampled max 8 ULP in [0,1], u16-safe to |encoded| ≤ ~4.2 (SIMD FMA), u8-safe to 8.0.
    pub(crate) const EXT_S2L_P: [f32; 7] = [
        1.802_136_5e1,
        9.110_411_4e2,
        1.570_602_1e4,
        1.020_638_2e5,
        2.199_931_2e5,
        1.338_269_2e5,
        1.706_519_4e4,
    ];
    /// Extended sRGB EOTF denominator (degree 6, fitted to [threshold, 8]).
    #[allow(clippy::excessive_precision)]
    pub(crate) const EXT_S2L_Q: [f32; 7] = [
        2.159_401_7e4,
        1.508_555_1e5,
        2.303_299_0e5,
        8.239_410_8e4,
        4.473_249_1e3,
        -6.359_000_1e1,
        1.0,
    ];
    /// Extended sRGB inverse EOTF numerator (degree 6, fitted on sqrt to [threshold, 64]).
    /// Sampled max 7 ULP in [0,1], u16-safe across full [0, 64] domain.
    /// Coefficients from polyfit (SK init + LM + f32 ULP grid search).
    pub(crate) const EXT_L2S_P: [f32; 7] = [
        -1.025_467_4,
        -3.075_361_5e-1,
        1.027_286e3,
        7.093_665e3,
        1.006_868_9e4,
        3.230_716e3,
        1.769_130_4e2,
    ];
    /// Extended sRGB inverse EOTF denominator (degree 6, fitted on sqrt to [threshold, 64]).
    pub(crate) const EXT_L2S_Q: [f32; 7] = [
        1.977_460_5e1,
        8.308_271e2,
        6.024_792_5e3,
        1.024_407_5e4,
        4.157_534e3,
        3.179_324_6e2,
        1.0,
    ];
    /// sRGB linearization threshold in gamma domain (C0-continuous).
    /// Values below this use the linear segment `v / 12.92`.
    pub(crate) const SRGB_THRESHOLD: f32 = 0.039_293_37;
    /// sRGB linearization threshold in linear domain (C0-continuous).
    /// Values below this use the linear segment `v * 12.92`.
    pub(crate) const LINEAR_THRESHOLD: f32 = 0.003_041_282_6;
    /// Scale factor for the linear segment (1 / 12.92).
    pub(crate) const LINEAR_SCALE: f32 = 1.0 / 12.92;
    /// Scale factor for the inverse linear segment.
    pub(crate) const TWELVE_92: f32 = 12.92;
    /// Evaluate a degree-4 rational polynomial P(x)/Q(x) using Horner's method.
    ///
    /// Evaluates in f64 to eliminate monotonicity violations from f32 rounding.
    /// The f64→f32 cast (round-to-nearest-even) preserves monotonicity because
    /// the f64 result has far more precision than needed for f32 output.
    ///
    /// Coefficients are lowest-degree-first: `p[0] + p[1]*x + p[2]*x^2 + ...`
    #[inline(always)]
    fn eval_rational_poly_5(x: f32, p: [f32; 5], q: [f32; 5]) -> f32 {
        let x = x as f64;
        let yp = (p[4] as f64).mul_add(x, p[3] as f64);
        let yp = yp.mul_add(x, p[2] as f64);
        let yp = yp.mul_add(x, p[1] as f64);
        let yp = yp.mul_add(x, p[0] as f64);
        let yq = (q[4] as f64).mul_add(x, q[3] as f64);
        let yq = yq.mul_add(x, q[2] as f64);
        let yq = yq.mul_add(x, q[1] as f64);
        let yq = yq.mul_add(x, q[0] as f64);
        (yp / yq) as f32
    }
    /// Convert sRGB gamma-encoded value to linear light using a rational polynomial (f32).
    ///
    /// Replaces `powf()` with a 5/5 rational polynomial (Horner's method).
    /// Max error: ~8 ULP in the power segment, 1 ULP at the piecewise threshold.
    /// Uses C0-continuous (moxcms) thresholds for exact continuity.
    ///
    /// **Clamps** inputs to \[0, 1\]. For platform `powf()`, see [`crate::precise::srgb_to_linear`].
    #[inline]
    pub fn srgb_to_linear_fast(gamma: f32) -> f32 {
        if gamma < 0.0 {
            return 0.0;
        }
        if gamma >= 1.0 {
            return 1.0;
        }
        if gamma <= SRGB_THRESHOLD {
            return gamma * LINEAR_SCALE;
        }
        eval_rational_poly_5(gamma, S2L_P, S2L_Q)
    }
    /// Convert linear light value to sRGB gamma-encoded using a rational polynomial (f32).
    ///
    /// Uses sqrt + 5/5 rational polynomial (Horner's method).
    /// Sampled max error: 10 ULP on the uniform/dark grid in benchmarks/precision_f32_2026-11-05.md.
    /// Uses C0-continuous (moxcms) thresholds for exact continuity.
    ///
    /// **Clamps** inputs to \[0, 1\]. For platform `powf()`, see [`crate::precise::linear_to_srgb`].
    #[inline]
    pub fn linear_to_srgb_fast(linear: f32) -> f32 {
        if linear < 0.0 {
            return 0.0;
        }
        if linear >= 1.0 {
            return 1.0;
        }
        if linear <= LINEAR_THRESHOLD {
            return linear * TWELVE_92;
        }
        let s = linear.sqrt();
        eval_rational_poly_5(s, L2S_P, L2S_Q)
    }
}
mod const_luts {
    //! Pre-computed lookup tables embedded in the binary.
    //!
    //! Tables are stored as raw little-endian bytes in `src/data/` and cast to
    //! `&[f32]` at runtime via `bytemuck::cast_slice`. A `#[repr(C, align(4))]`
    //! wrapper guarantees f32 alignment so `cast_slice` is a trivial pointer cast
    //! (no copy, no allocation — LLVM optimizes it to nothing after inlining).
    //!
    //! Generated using C0-continuous sRGB constants (moxcms).
    /// Wrapper that forces 4-byte alignment on an included byte array.
    /// `include_bytes!` has no alignment guarantee; this ensures the data
    /// can be safely cast to `&[f32]` via `bytemuck::cast_slice`.
    #[repr(C, align(4))]
    struct Aligned<const N: usize>([u8; N]);
    static LINEAR_TABLE_8: Aligned<{ 256 * 4 }> = Aligned(
        *b"\x00\x00\x00\x00\xb4\"\x9f9\xb4\"\x1f:\x0e\xb4n:\xb4\"\x9f:a\xeb\xc6:\x0e\xb4\xee:]>\x0b;\xb4\"\x1f;\n\x073;a\xebF;\xe7^[;\x9a\xffp;%\xe9\x83;\xd0\xed\x8f;\x12\x90\x9c;&\xd2\xa9;;\xb6\xb7;r>\xc6;\xe5l\xd5;\xa3C\xe5;\xb0\xc4\xf5;\x05y\x03<\xd1f\x0c<\xb2\xac\x15<\x9aK\x1f<vD)<2\x983<\xb4G><\xdeSI<\x92\xbdT<\xac\x85`<\x07\xadl<z4y<l\x0e\x83<{\xb3\x89<\xcf\x89\x90<\xd1\x91\x97<\xe4\xcb\x9e<m8\xa6<\xcf\xd7\xad<k\xaa\xb5<\xa1\xb0\xbd<\xd2\xea\xc5<[Y\xce<\x9b\xfc\xd6<\xed\xd4\xdf<\xae\xe2\xe8<8&\xf2<\xe5\x9f\xfb<\x07\xa8\x02=\x85\x9b\x07=\x99\xaa\x0c=n\xd5\x11=.\x1c\x17=\x04\x7f\x1c=\x1b\xfe!=\x9c\x99\'=\xb0Q-=\x81&3=7\x189=\xfa&?=\xf2RE=H\x9cK=\"\x03R=\xa8\x87X=\x00*_=Q\xeae=\xc2\xc8l=w\xc5s=\x97\xe0z=$\r\x81=V\xb9\x84=\xf6t\x88=\x15@\x8c=\xc6\x1a\x90=\x1a\x05\x94=$\xff\x97=\xf4\x08\x9c=\x9e\"\xa0=3L\xa4=\xc3\x85\xa8=a\xcf\xac=\x1d)\xb1=\n\x93\xb5=6\r\xba=\xb5\x97\xbe=\x962\xc3=\xeb\xdd\xc7=\xc3\x99\xcc=0f\xd1=AC\xd6=\x081\xdb=\x94/\xe0=\xf6>\xe5=>_\xea=z\x90\xef=\xbd\xd2\xf4=\x15&\xfa=\x91\x8a\xff=!\x80\x02>\x9cC\x05>\xc1\x0f\x08>\x97\xe4\n>&\xc2\r>v\xa8\x10>\x8f\x97\x13>w\x8f\x16>7\x90\x19>\xd5\x99\x1c>Z\xac\x1f>\xcd\xc7\">4\xec%>\x98\x19)>\x00P,>r\x8f/>\xf6\xd72>\x94)6>R\x849>8\xe8<>LU@>\x96\xcbC>\x1dKG>\xe7\xd3J>\xfceN>c\x01R>\"\xa6U>@TY>\xc5\x0b]>\xb6\xcc`>\x1c\x97d>\xfbjh>]Hl>F/p>\xbd\x1ft>\xca\x19x>s\x1d|>_\x15\x80>\xd9 \x82>+1\x84>XF\x86>c`\x88>P\x7f\x8a>!\xa3\x8c>\xdb\xcb\x8e>\x7f\xf9\x90>\x12,\x93>\x97c\x95>\x0f\xa0\x97>\x80\xe1\x99>\xec\'\x9c>Us\x9e>\xc0\xc3\xa0>/\x19\xa3>\xa6s\xa5>\'\xd3\xa7>\xb57\xaa>T\xa1\xac>\x06\x10\xaf>\xcf\x83\xb1>\xb2\xfc\xb3>\xb1z\xb6>\xd0\xfd\xb8>\x12\x86\xbb>y\x13\xbe>\t\xa6\xc0>\xc4=\xc3>\xae\xda\xc5>\xc9|\xc8>\x19$\xcb>\xa0\xd0\xcd>a\x82\xd0>`9\xd3>\x9e\xf5\xd5>\x1f\xb7\xd8>\xe7}\xdb>\xf6I\xde>R\x1b\xe1>\xfb\xf1\xe3>\xf6\xcd\xe6>E\xaf\xe9>\xeb\x95\xec>\xea\x81\xef>Fs\xf2>\x01j\xf5>\x1ef\xf8>\xa0g\xfb>\x8an\xfe>o\xbd\x00?PF\x02?\xe8\xd1\x03?:`\x05?F\xf1\x06?\x0f\x85\x08?\x95\x1b\n?\xd9\xb4\x0b?\xdeP\r?\xa4\xef\x0e?-\x91\x10?z5\x12?\x8d\xdc\x13?f\x86\x15?\x073\x17?r\xe2\x18?\xa7\x94\x1a?\xa9I\x1c?x\x01\x1e?\x16\xbc\x1f?\x83y!?\xc29#?\xd4\xfc$?\xb9\xc2&?t\x8b(?\x05W*?n%,?\xaf\xf6-?\xcb\xca/?\xc3\xa11?\x98{3?JX5?\xdc77?O\x1a9?\xa3\xff:?\xdb\xe7<?\xf7\xd2>?\xf9\xc0@?\xe1\xb1B?\xb2\xa5D?l\x9cF?\x11\x96H?\xa2\x92J?\x1f\x92L?\x8b\x94N?\xe7\x99P?3\xa2R?q\xadT?\xa2\xbbV?\xc8\xccX?\xe3\xe0Z?\xf5\xf7\\?\xff\x11_?\x03/a?\x00Oc?\xfaqe?\xf0\x97g?\xe4\xc0i?\xd7\xeck?\xca\x1bn?\xbfMp?\xb7\x82r?\xb3\xbat?\xb4\xf5v?\xba3y?\xc9t{?\xdf\xb8}?\x00\x00\x80?",
    );
    static ENCODE_TABLE_12: Aligned<{ 4096 * 4 }> = Aligned(
        *b"\x00\x00\x00\x00>\xc5N;>\xc5\xce;\xef\x13\x1b<>\xc5N<G;\x81<\xef\x13\x9b<\x96\xec\xb4<>\xc5\xce<\xe6\x9d\xe8<G;\x01=\x9b\'\x0e=\xef\x13\x1b=\xb0\xe9\'=\xa7>4=\xa5\x12@=\x08sK=\x11kV=O\x04a=\xfaFk=.:u= \xe4~=!%\x84=\xb4\xb8\x88=\xed.\x8d=\xb6\x89\x91=\xc6\xca\x95=\xab\xf3\x99=\xcc\x05\x9e=o\x02\xa2=\xbd\xea\xa5=\xc7\xbf\xa9=\x86\x82\xad=\xe03\xb1=\xa9\xd4\xb4=\xa5e\xb8=\x89\xe7\xbb=\xfdZ\xbf=\xa0\xc0\xc2=\x02\x19\xc6=\xaed\xc9=\"\xa4\xcc=\xd6\xd7\xcf=<\x00\xd3=\xbe\x1d\xd6=\xbe0\xd9=\x9b9\xdc=\xad8\xdf=J.\xe2=\xbf\x1a\xe5=X\xfe\xe7=^\xd9\xea=\x12\xac\xed=\xb7v\xf0=\x899\xf3=\xc3\xf4\xf5=\x9c\xa8\xf8=JU\xfb=\xff\xfa\xfd=\xf6L\x00>\x1f\x99\x01>\x12\xe2\x02>\xe3\'\x04>\xa7j\x05>q\xaa\x06>U\xe7\x07>d!\t>\xafX\n>H\x8d\x0b>=\xbf\x0c>\x9f\xee\r>}\x1b\x0f>\xe4E\x10>\xe3m\x11>\x87\x93\x12>\xdd\xb6\x13>\xf1\xd7\x14>\xce\xf6\x15>\x82\x13\x17>\x17.\x18>\x97F\x19>\x0e]\x1a>\x85q\x1b>\x06\x84\x1c>\x9b\x94\x1d>M\xa3\x1e>&\xb0\x1f>-\xbb >k\xc4!>\xe9\xcb\">\xad\xd1#>\xc1\xd5$>,\xd8%>\xf4\xd8&>\"\xd8\'>\xbb\xd5(>\xc6\xd1)>K\xcc*>P\xc5+>\xda\xbc,>\xef\xb2->\x97\xa7.>\xd5\x9a/>\xb1\x8c0>.}1>Tl2>&Z3>\xaaF4>\xe515>\xdc\x1b6>\x92\x047>\x0e\xec7>S\xd28>f\xb79>K\x9b:>\x06~;>\x9c_<>\x10@=>f\x1f>>\xa2\xfd>>\xc8\xda?>\xdb\xb6@>\xdf\x91A>\xd8kB>\xc9DC>\xb5\x1cD>\xa0\xf3D>\x8d\xc9E>\x7f\x9eF>yrG>~EH>\x91\x17I>\xb6\xe8I>\xee\xb8J>=\x88K>\xa5VL>*$M>\xcd\xf0M>\x92\xbcN>z\x87O>\x8aQP>\xc2\x1aQ>%\xe3Q>\xb6\xaaR>wqS>k7T>\x93\xfcT>\xf2\xc0U>\x89\x84V>\\GW>l\tX>\xbb\xcaX>K\x8bY>\x1eKZ>7\n[>\x96\xc8[>?\x86\\>2C]>r\xff]>\x00\xbb^>\xdeu_>\x0f0`>\x93\xe9`>l\xa2a>\x9cZb>%\x12c>\x08\xc9c>F\x7fd>\xe24e>\xdd\xe9e>8\x9ef>\xf5Qg>\x16\x05h>\x9b\xb7h>\x86ii>\xd9\x1aj>\x95\xcbj>\xbb{k>L+l>K\xdal>\xb8\x88m>\x946n>\xe1\xe3n>\xa1\x90o>\xd3<p>z\xe8p>\x97\x93q>*>r>5\xe8r>\xba\x91s>\xb9:t>3\xe3t>*\x8bu>\x9e2v>\x91\xd9v>\x03\x80w>\xf7%x>l\xcbx>cpy>\xdf\x14z>\xdf\xb8z>e\\{>r\xff{>\x06\xa2|>#D}>\xca\xe5}>\xfa\x86~>\xb7\'\x7f>\xff\xc7\x7f>\xea3\x80>\x9c\x83\x80>\x15\xd3\x80>V\"\x81>_q\x81>1\xc0\x81>\xcc\x0e\x82>0]\x82>^\xab\x82>W\xf9\x82>\x1aG\x83>\xa8\x94\x83>\x02\xe2\x83>(/\x84>\x19|\x84>\xd7\xc8\x84>b\x15\x85>\xbba\x85>\xe1\xad\x85>\xd5\xf9\x85>\x97E\x86>(\x91\x86>\x88\xdc\x86>\xb7\'\x87>\xb6r\x87>\x85\xbd\x87>$\x08\x88>\x94R\x88>\xd5\x9c\x88>\xe7\xe6\x88>\xcb0\x89>\x80z\x89>\x08\xc4\x89>b\r\x8a>\x8fV\x8a>\x8f\x9f\x8a>c\xe8\x8a>\n1\x8b>\x85y\x8b>\xd4\xc1\x8b>\xf8\t\x8c>\xf1Q\x8c>\xbf\x99\x8c>b\xe1\x8c>\xda(\x8d>)p\x8d>M\xb7\x8d>H\xfe\x8d>\x1aE\x8e>\xc2\x8b\x8e>B\xd2\x8e>\x99\x18\x8f>\xc8^\x8f>\xce\xa4\x8f>\xad\xea\x8f>d0\x90>\xf4u\x90>]\xbb\x90>\x9e\x00\x91>\xb9E\x91>\xae\x8a\x91>|\xcf\x91>$\x14\x92>\xa7X\x92>\x03\x9d\x92>;\xe1\x92>M%\x93>;i\x93>\x04\xad\x93>\xa8\xf0\x93>(4\x94>\x84w\x94>\xbb\xba\x94>\xd0\xfd\x94>\xc0@\x95>\x8e\x83\x95>8\xc6\x95>\xc0\x08\x96>%K\x96>g\x8d\x96>\x87\xcf\x96>\x85\x11\x97>aS\x97>\x1b\x95\x97>\xb3\xd6\x97>*\x18\x98>\x80Y\x98>\xb5\x9a\x98>\xc9\xdb\x98>\xbd\x1c\x99>\x8f]\x99>B\x9e\x99>\xd4\xde\x99>G\x1f\x9a>\x99_\x9a>\xcc\x9f\x9a>\xdf\xdf\x9a>\xd3\x1f\x9b>\xa8_\x9b>]\x9f\x9b>\xf4\xde\x9b>m\x1e\x9c>\xc6]\x9c>\x01\x9d\x9c>\x1e\xdc\x9c>\x1d\x1b\x9d>\xfeY\x9d>\xc1\x98\x9d>g\xd7\x9d>\xef\x15\x9e>ZT\x9e>\xa7\x92\x9e>\xd8\xd0\x9e>\xec\x0e\x9f>\xe2L\x9f>\xbd\x8a\x9f>z\xc8\x9f>\x1c\x06\xa0>\xa1C\xa0>\n\x81\xa0>W\xbe\xa0>\x89\xfb\xa0>\x9e8\xa1>\x98u\xa1>w\xb2\xa1>;\xef\xa1>\xe3+\xa2>ph\xa2>\xe3\xa4\xa2>;\xe1\xa2>x\x1d\xa3>\x9aY\xa3>\xa2\x95\xa3>\x90\xd1\xa3>d\r\xa4>\x1eI\xa4>\xbd\x84\xa4>C\xc0\xa4>\xb0\xfb\xa4>\x037\xa5><r\xa5>\\\xad\xa5>c\xe8\xa5>Q#\xa6>%^\xa6>\xe1\x98\xa6>\x84\xd3\xa6>\x0f\x0e\xa7>\x81H\xa7>\xda\x82\xa7>\x1c\xbd\xa7>E\xf7\xa7>U1\xa8>Nk\xa8>/\xa5\xa8>\xf9\xde\xa8>\xaa\x18\xa9>DR\xa9>\xc6\x8b\xa9>2\xc5\xa9>\x85\xfe\xa9>\xc27\xaa>\xe7p\xaa>\xf6\xa9\xaa>\xee\xe2\xaa>\xce\x1b\xab>\x99T\xab>L\x8d\xab>\xe9\xc5\xab>p\xfe\xab>\xe06\xac>:o\xac>~\xa7\xac>\xac\xdf\xac>\xc4\x17\xad>\xc7O\xad>\xb3\x87\xad>\x8a\xbf\xad>K\xf7\xad>\xf6.\xae>\x8df\xae>\x0e\x9e\xae>y\xd5\xae>\xd0\x0c\xaf>\x11D\xaf>>{\xaf>U\xb2\xaf>X\xe9\xaf>F \xb0>\x1fW\xb0>\xe4\x8d\xb0>\x95\xc4\xb0>0\xfb\xb0>\xb81\xb1>+h\xb1>\x8b\x9e\xb1>\xd6\xd4\xb1>\r\x0b\xb2>0A\xb2>?w\xb2>;\xad\xb2>#\xe3\xb2>\xf7\x18\xb3>\xb8N\xb3>e\x84\xb3>\xff\xb9\xb3>\x85\xef\xb3>\xf9$\xb4>YZ\xb4>\xa6\x8f\xb4>\xe0\xc4\xb4>\x07\xfa\xb4>\x1b/\xb5>\x1cd\xb5>\x0b\x99\xb5>\xe7\xcd\xb5>\xb0\x02\xb6>g7\xb6>\x0bl\xb6>\x9d\xa0\xb6>\x1c\xd5\xb6>\x8a\t\xb7>\xe5=\xb7>.r\xb7>e\xa6\xb7>\x8a\xda\xb7>\x9d\x0e\xb8>\x9eB\xb8>\x8dv\xb8>k\xaa\xb8>7\xde\xb8>\xf1\x11\xb9>\x9aE\xb9>1y\xb9>\xb7\xac\xb9>+\xe0\xb9>\x8f\x13\xba>\xe1F\xba>!z\xba>Q\xad\xba>p\xe0\xba>}\x13\xbb>zF\xbb>fy\xbb>A\xac\xbb>\x0b\xdf\xbb>\xc5\x11\xbc>nD\xbc>\x06w\xbc>\x8e\xa9\xbc>\x05\xdc\xbc>l\x0e\xbd>\xc3@\xbd>\ts\xbd>?\xa5\xbd>e\xd7\xbd>{\t\xbe>\x81;\xbe>wm\xbe>\\\x9f\xbe>2\xd1\xbe>\xf8\x02\xbf>\xae4\xbf>Uf\xbf>\xeb\x97\xbf>r\xc9\xbf>\xea\xfa\xbf>R,\xc0>\xaa]\xc0>\xf3\x8e\xc0>-\xc0\xc0>W\xf1\xc0>r\"\xc1>~S\xc1>z\x84\xc1>h\xb5\xc1>F\xe6\xc1>\x15\x17\xc2>\xd5G\xc2>\x87x\xc2>)\xa9\xc2>\xbd\xd9\xc2>B\n\xc3>\xb8:\xc3>\x1fk\xc3>x\x9b\xc3>\xc2\xcb\xc3>\xfe\xfb\xc3>+,\xc4>I\\\xc4>Y\x8c\xc4>[\xbc\xc4>O\xec\xc4>4\x1c\xc5>\x0bL\xc5>\xd3{\xc5>\x8e\xab\xc5>:\xdb\xc5>\xd9\n\xc6>i:\xc6>\xeci\xc6>`\x99\xc6>\xc7\xc8\xc6>\x1f\xf8\xc6>j\'\xc7>\xa8V\xc7>\xd7\x85\xc7>\xf9\xb4\xc7>\r\xe4\xc7>\x14\x13\xc8>\rB\xc8>\xf8p\xc8>\xd6\x9f\xc8>\xa7\xce\xc8>j\xfd\xc8> ,\xc9>\xc9Z\xc9>d\x89\xc9>\xf2\xb7\xc9>s\xe6\xc9>\xe7\x14\xca>NC\xca>\xa7q\xca>\xf4\x9f\xca>3\xce\xca>f\xfc\xca>\x8c*\xcb>\xa5X\xcb>\xb1\x86\xcb>\xb0\xb4\xcb>\xa2\xe2\xcb>\x88\x10\xcc>a>\xcc>-l\xcc>\xed\x99\xcc>\xa0\xc7\xcc>G\xf5\xcc>\xe1\"\xcd>oP\xcd>\xf0}\xcd>e\xab\xcd>\xce\xd8\xcd>*\x06\xce>z3\xce>\xbd`\xce>\xf5\x8d\xce> \xbb\xce>?\xe8\xce>R\x15\xcf>YB\xcf>To\xcf>C\x9c\xcf>&\xc9\xcf>\xfd\xf5\xcf>\xc8\"\xd0>\x87O\xd0>:|\xd0>\xe2\xa8\xd0>~\xd5\xd0>\x0e\x02\xd1>\x92.\xd1>\x0b[\xd1>x\x87\xd1>\xd9\xb3\xd1>/\xe0\xd1>y\x0c\xd2>\xb88\xd2>\xebd\xd2>\x13\x91\xd2>/\xbd\xd2>@\xe9\xd2>F\x15\xd3>@A\xd3>/m\xd3>\x12\x99\xd3>\xeb\xc4\xd3>\xb8\xf0\xd3>z\x1c\xd4>1H\xd4>\xdds\xd4>~\x9f\xd4>\x14\xcb\xd4>\x9e\xf6\xd4>\x1e\"\xd5>\x93M\xd5>\xfcx\xd5>[\xa4\xd5>\xaf\xcf\xd5>\xf8\xfa\xd5>7&\xd6>jQ\xd6>\x93|\xd6>\xb1\xa7\xd6>\xc4\xd2\xd6>\xcd\xfd\xd6>\xcb(\xd7>\xbeS\xd7>\xa7~\xd7>\x85\xa9\xd7>Y\xd4\xd7>\"\xff\xd7>\xe0)\xd8>\x94T\xd8>>\x7f\xd8>\xde\xa9\xd8>s\xd4\xd8>\xfd\xfe\xd8>})\xd9>\xf3S\xd9>_~\xd9>\xc1\xa8\xd9>\x18\xd3\xd9>e\xfd\xd9>\xa8\'\xda>\xe0Q\xda>\x0f|\xda>4\xa6\xda>N\xd0\xda>_\xfa\xda>e$\xdb>aN\xdb>Tx\xdb><\xa2\xdb>\x1b\xcc\xdb>\xf0\xf5\xdb>\xbb\x1f\xdc>|I\xdc>3s\xdc>\xe0\x9c\xdc>\x84\xc6\xdc>\x1e\xf0\xdc>\xae\x19\xdd>4C\xdd>\xb1l\xdd>$\x96\xdd>\x8e\xbf\xdd>\xee\xe8\xdd>D\x12\xde>\x91;\xde>\xd5d\xde>\x0e\x8e\xde>?\xb7\xde>f\xe0\xde>\x83\t\xdf>\x972\xdf>\xa2[\xdf>\xa3\x84\xdf>\x9b\xad\xdf>\x89\xd6\xdf>o\xff\xdf>K(\xe0>\x1dQ\xe0>\xe7y\xe0>\xa7\xa2\xe0>^\xcb\xe0>\x0c\xf4\xe0>\xb1\x1c\xe1>ME\xe1>\xdfm\xe1>i\x96\xe1>\xe9\xbe\xe1>a\xe7\xe1>\xcf\x0f\xe2>58\xe2>\x91`\xe2>\xe5\x88\xe2>/\xb1\xe2>q\xd9\xe2>\xa9\x01\xe3>\xd9)\xe3>\x00R\xe3>\x1fz\xe3>4\xa2\xe3>A\xca\xe3>D\xf2\xe3>@\x1a\xe4>2B\xe4>\x1cj\xe4>\xfd\x91\xe4>\xd5\xb9\xe4>\xa5\xe1\xe4>l\t\xe5>*1\xe5>\xe0X\xe5>\x8d\x80\xe5>2\xa8\xe5>\xce\xcf\xe5>b\xf7\xe5>\xed\x1e\xe6>pF\xe6>\xeam\xe6>\\\x95\xe6>\xc6\xbc\xe6>\'\xe4\xe6>\x7f\x0b\xe7>\xd02\xe7>\x18Z\xe7>W\x81\xe7>\x8f\xa8\xe7>\xbe\xcf\xe7>\xe5\xf6\xe7>\x03\x1e\xe8>\x1aE\xe8>(l\xe8>.\x93\xe8>,\xba\xe8>\"\xe1\xe8>\x0f\x08\xe9>\xf4.\xe9>\xd2U\xe9>\xa7|\xe9>t\xa3\xe9>9\xca\xe9>\xf7\xf0\xe9>\xac\x17\xea>Y>\xea>\xfed\xea>\x9b\x8b\xea>0\xb2\xea>\xbe\xd8\xea>C\xff\xea>\xc1%\xeb>6L\xeb>\xa4r\xeb>\n\x99\xeb>h\xbf\xeb>\xbe\xe5\xeb>\r\x0c\xec>T2\xec>\x93X\xec>\xca~\xec>\xfa\xa4\xec>\"\xcb\xec>B\xf1\xec>Z\x17\xed>k=\xed>tc\xed>v\x89\xed>p\xaf\xed>b\xd5\xed>M\xfb\xed>1!\xee>\x0cG\xee>\xe0l\xee>\xad\x92\xee>r\xb8\xee>0\xde\xee>\xe6\x03\xef>\x95)\xef>=O\xef>\xddt\xef>u\x9a\xef>\x06\xc0\xef>\x90\xe5\xef>\x13\x0b\xf0>\x8e0\xf0>\x02V\xf0>n{\xf0>\xd3\xa0\xf0>1\xc6\xf0>\x88\xeb\xf0>\xd7\x10\xf1> 6\xf1>a[\xf1>\x9a\x80\xf1>\xcd\xa5\xf1>\xf8\xca\xf1>\x1d\xf0\xf1>:\x15\xf2>P:\xf2>^_\xf2>f\x84\xf2>g\xa9\xf2>`\xce\xf2>S\xf3\xf2>?\x18\xf3>#=\xf3>\x00b\xf3>\xd7\x86\xf3>\xa6\xab\xf3>o\xd0\xf3>0\xf5\xf3>\xeb\x19\xf4>\x9f>\xf4>Kc\xf4>\xf1\x87\xf4>\x90\xac\xf4>(\xd1\xf4>\xb9\xf5\xf4>D\x1a\xf5>\xc7>\xf5>Dc\xf5>\xba\x87\xf5>)\xac\xf5>\x91\xd0\xf5>\xf3\xf4\xf5>M\x19\xf6>\xa2=\xf6>\xefa\xf6>5\x86\xf6>u\xaa\xf6>\xaf\xce\xf6>\xe1\xf2\xf6>\r\x17\xf7>2;\xf7>Q_\xf7>i\x83\xf7>z\xa7\xf7>\x85\xcb\xf7>\x89\xef\xf7>\x87\x13\xf8>~7\xf8>n[\xf8>X\x7f\xf8>;\xa3\xf8>\x18\xc7\xf8>\xef\xea\xf8>\xbf\x0e\xf9>\x882\xf9>KV\xf9>\x08z\xf9>\xbe\x9d\xf9>m\xc1\xf9>\x17\xe5\xf9>\xba\x08\xfa>V,\xfa>\xecO\xfa>|s\xfa>\x05\x97\xfa>\x89\xba\xfa>\x05\xde\xfa>|\x01\xfb>\xec$\xfb>VH\xfb>\xb9k\xfb>\x17\x8f\xfb>n\xb2\xfb>\xbe\xd5\xfb>\t\xf9\xfb>M\x1c\xfc>\x8b?\xfc>\xc3b\xfc>\xf5\x85\xfc>!\xa9\xfc>F\xcc\xfc>e\xef\xfc>\x7f\x12\xfd>\x925\xfd>\x9eX\xfd>\xa5{\xfd>\xa6\x9e\xfd>\xa1\xc1\xfd>\x95\xe4\xfd>\x84\x07\xfe>l*\xfe>NM\xfe>+p\xfe>\x01\x93\xfe>\xd1\xb5\xfe>\x9c\xd8\xfe>`\xfb\xfe>\x1e\x1e\xff>\xd7@\xff>\x89c\xff>6\x86\xff>\xdc\xa8\xff>}\xcb\xff>\x18\xee\xff>V\x08\x00?\x9e\x19\x00?\xe2*\x00?$<\x00?cM\x00?\x9f^\x00?\xd7o\x00?\r\x81\x00?@\x92\x00?q\xa3\x00?\x9e\xb4\x00?\xc8\xc5\x00?\xf0\xd6\x00?\x14\xe8\x00?6\xf9\x00?U\n\x01?q\x1b\x01?\x8a,\x01?\xa0=\x01?\xb3N\x01?\xc4_\x01?\xd2p\x01?\xdd\x81\x01?\xe5\x92\x01?\xea\xa3\x01?\xec\xb4\x01?\xec\xc5\x01?\xe9\xd6\x01?\xe3\xe7\x01?\xda\xf8\x01?\xce\t\x02?\xc0\x1a\x02?\xaf+\x02?\x9b<\x02?\x84M\x02?j^\x02?No\x02?/\x80\x02?\r\x91\x02?\xe9\xa1\x02?\xc2\xb2\x02?\x97\xc3\x02?k\xd4\x02?;\xe5\x02?\t\xf6\x02?\xd4\x06\x03?\x9c\x17\x03?b(\x03?%9\x03?\xe5I\x03?\xa2Z\x03?]k\x03?\x15|\x03?\xcb\x8c\x03?}\x9d\x03?.\xae\x03?\xdb\xbe\x03?\x86\xcf\x03?.\xe0\x03?\xd3\xf0\x03?v\x01\x04?\x16\x12\x04?\xb3\"\x04?N3\x04?\xe6C\x04?|T\x04?\x0fe\x04?\x9fu\x04?-\x86\x04?\xb8\x96\x04?@\xa7\x04?\xc6\xb7\x04?I\xc8\x04?\xca\xd8\x04?H\xe9\x04?\xc3\xf9\x04?<\n\x05?\xb2\x1a\x05?&+\x05?\x97;\x05?\x06L\x05?r\\\x05?\xdbl\x05?B}\x05?\xa6\x8d\x05?\x08\x9e\x05?g\xae\x05?\xc4\xbe\x05?\x1e\xcf\x05?v\xdf\x05?\xcb\xef\x05?\x1e\x00\x06?n\x10\x06?\xbb \x06?\x061\x06?OA\x06?\x95Q\x06?\xd8a\x06?\x19r\x06?X\x82\x06?\x94\x92\x06?\xce\xa2\x06?\x05\xb3\x06?9\xc3\x06?l\xd3\x06?\x9b\xe3\x06?\xc9\xf3\x06?\xf3\x03\x07?\x1c\x14\x07?B$\x07?e4\x07?\x86D\x07?\xa5T\x07?\xc1d\x07?\xdbt\x07?\xf2\x84\x07?\x07\x95\x07?\x19\xa5\x07?)\xb5\x07?7\xc5\x07?B\xd5\x07?K\xe5\x07?Q\xf5\x07?U\x05\x08?W\x15\x08?V%\x08?S5\x08?NE\x08?FU\x08?;e\x08?/u\x08? \x85\x08?\x0e\x95\x08?\xfb\xa4\x08?\xe5\xb4\x08?\xcc\xc4\x08?\xb1\xd4\x08?\x94\xe4\x08?u\xf4\x08?S\x04\t?/\x14\t?\t$\t?\xe03\t?\xb5C\t?\x87S\t?Xc\t?&s\t?\xf1\x82\t?\xbb\x92\t?\x82\xa2\t?F\xb2\t?\t\xc2\t?\xc9\xd1\t?\x87\xe1\t?C\xf1\t?\xfc\x00\n?\xb3\x10\n?h \n?\x1a0\n?\xcb?\n?yO\n?$_\n?\xcen\n?u~\n?\x1a\x8e\n?\xbd\x9d\n?]\xad\n?\xfc\xbc\n?\x98\xcc\n?1\xdc\n?\xc9\xeb\n?^\xfb\n?\xf2\n\x0b?\x82\x1a\x0b?\x11*\x0b?\x9e9\x0b?(I\x0b?\xb0X\x0b?6h\x0b?\xbaw\x0b?;\x87\x0b?\xba\x96\x0b?8\xa6\x0b?\xb2\xb5\x0b?+\xc5\x0b?\xa2\xd4\x0b?\x16\xe4\x0b?\x88\xf3\x0b?\xf9\x02\x0c?f\x12\x0c?\xd2!\x0c?<1\x0c?\xa3@\x0c?\tP\x0c?l_\x0c?\xcdn\x0c?+~\x0c?\x88\x8d\x0c?\xe3\x9c\x0c?;\xac\x0c?\x92\xbb\x0c?\xe6\xca\x0c?8\xda\x0c?\x88\xe9\x0c?\xd6\xf8\x0c?!\x08\r?k\x17\r?\xb2&\r?\xf85\r?;E\r?|T\r?\xbbc\r?\xf8r\r?3\x82\r?l\x91\r?\xa3\xa0\r?\xd7\xaf\r?\n\xbf\r?;\xce\r?i\xdd\r?\x95\xec\r?\xc0\xfb\r?\xe8\n\x0e?\x0e\x1a\x0e?2)\x0e?T8\x0e?tG\x0e?\x92V\x0e?\xaee\x0e?\xc8t\x0e?\xe0\x83\x0e?\xf6\x92\x0e?\n\xa2\x0e?\x1b\xb1\x0e?+\xc0\x0e?9\xcf\x0e?E\xde\x0e?N\xed\x0e?V\xfc\x0e?\\\x0b\x0f?_\x1a\x0f?a)\x0f?`8\x0f?^G\x0f?ZV\x0f?Se\x0f?Kt\x0f?A\x83\x0f?4\x92\x0f?&\xa1\x0f?\x16\xb0\x0f?\x03\xbf\x0f?\xef\xcd\x0f?\xd9\xdc\x0f?\xc1\xeb\x0f?\xa7\xfa\x0f?\x8a\t\x10?l\x18\x10?L\'\x10?*6\x10?\x06E\x10?\xe0S\x10?\xb8b\x10?\x8fq\x10?c\x80\x10?5\x8f\x10?\x06\x9e\x10?\xd4\xac\x10?\xa1\xbb\x10?k\xca\x10?4\xd9\x10?\xfa\xe7\x10?\xbf\xf6\x10?\x82\x05\x11?C\x14\x11?\x02#\x11?\xbf1\x11?z@\x11?4O\x11?\xeb]\x11?\xa1l\x11?T{\x11?\x06\x8a\x11?\xb6\x98\x11?d\xa7\x11?\x10\xb6\x11?\xba\xc4\x11?b\xd3\x11?\t\xe2\x11?\xad\xf0\x11?P\xff\x11?\xf1\r\x12?\x90\x1c\x12?-+\x12?\xc89\x12?aH\x12?\xf9V\x12?\x8ee\x12?\"t\x12?\xb4\x82\x12?D\x91\x12?\xd2\x9f\x12?_\xae\x12?\xe9\xbc\x12?r\xcb\x12?\xf9\xd9\x12?~\xe8\x12?\x01\xf7\x12?\x82\x05\x13?\x02\x14\x13?\x7f\"\x13?\xfb0\x13?u?\x13?\xedM\x13?d\\\x13?\xd8j\x13?Ky\x13?\xbc\x87\x13?+\x96\x13?\x99\xa4\x13?\x04\xb3\x13?n\xc1\x13?\xd6\xcf\x13?<\xde\x13?\xa1\xec\x13?\x03\xfb\x13?d\t\x14?\xc3\x17\x14? &\x14?|4\x14?\xd5B\x14?-Q\x14?\x83_\x14?\xd8m\x14?*|\x14?{\x8a\x14?\xca\x98\x14?\x18\xa7\x14?c\xb5\x14?\xad\xc3\x14?\xf5\xd1\x14?;\xe0\x14?\x80\xee\x14?\xc3\xfc\x14?\x04\x0b\x15?C\x19\x15?\x80\'\x15?\xbc5\x15?\xf6C\x15?/R\x15?e`\x15?\x9an\x15?\xcd|\x15?\xff\x8a\x15?/\x99\x15?]\xa7\x15?\x89\xb5\x15?\xb3\xc3\x15?\xdc\xd1\x15?\x03\xe0\x15?)\xee\x15?M\xfc\x15?o\n\x16?\x8f\x18\x16?\xae&\x16?\xca4\x16?\xe6B\x16?\xffP\x16?\x17_\x16?-m\x16?B{\x16?T\x89\x16?e\x97\x16?u\xa5\x16?\x83\xb3\x16?\x8f\xc1\x16?\x99\xcf\x16?\xa2\xdd\x16?\xa9\xeb\x16?\xae\xf9\x16?\xb2\x07\x17?\xb4\x15\x17?\xb4#\x17?\xb31\x17?\xb0?\x17?\xabM\x17?\xa5[\x17?\x9di\x17?\x94w\x17?\x88\x85\x17?{\x93\x17?m\xa1\x17?]\xaf\x17?K\xbd\x17?8\xcb\x17?#\xd9\x17?\x0c\xe7\x17?\xf4\xf4\x17?\xda\x02\x18?\xbe\x10\x18?\xa1\x1e\x18?\x82,\x18?a:\x18??H\x18?\x1cV\x18?\xf6c\x18?\xcfq\x18?\xa7\x7f\x18?}\x8d\x18?Q\x9b\x18?$\xa9\x18?\xf5\xb6\x18?\xc4\xc4\x18?\x92\xd2\x18?^\xe0\x18?)\xee\x18?\xf2\xfb\x18?\xb9\t\x19?\x7f\x17\x19?D%\x19?\x063\x19?\xc7@\x19?\x87N\x19?E\\\x19?\x01j\x19?\xbcw\x19?u\x85\x19?-\x93\x19?\xe3\xa0\x19?\x98\xae\x19?K\xbc\x19?\xfc\xc9\x19?\xac\xd7\x19?Z\xe5\x19?\x07\xf3\x19?\xb2\x00\x1a?[\x0e\x1a?\x03\x1c\x1a?\xaa)\x1a?O7\x1a?\xf2D\x1a?\x94R\x1a?4`\x1a?\xd3m\x1a?p{\x1a?\x0c\x89\x1a?\xa6\x96\x1a??\xa4\x1a?\xd6\xb1\x1a?k\xbf\x1a?\xff\xcc\x1a?\x92\xda\x1a?#\xe8\x1a?\xb2\xf5\x1a?@\x03\x1b?\xcd\x10\x1b?W\x1e\x1b?\xe1+\x1b?i9\x1b?\xefF\x1b?tT\x1b?\xf7a\x1b?yo\x1b?\xf9|\x1b?x\x8a\x1b?\xf5\x97\x1b?q\xa5\x1b?\xeb\xb2\x1b?d\xc0\x1b?\xdb\xcd\x1b?Q\xdb\x1b?\xc5\xe8\x1b?8\xf6\x1b?\xaa\x03\x1c?\x19\x11\x1c?\x88\x1e\x1c?\xf5+\x1c?`9\x1c?\xcaF\x1c?2T\x1c?\x99a\x1c?\xffn\x1c?c|\x1c?\xc5\x89\x1c?&\x97\x1c?\x86\xa4\x1c?\xe4\xb1\x1c?A\xbf\x1c?\x9c\xcc\x1c?\xf6\xd9\x1c?N\xe7\x1c?\xa5\xf4\x1c?\xfa\x01\x1d?N\x0f\x1d?\xa1\x1c\x1d?\xf2)\x1d?A7\x1d?\x8fD\x1d?\xdcQ\x1d?\'_\x1d?ql\x1d?\xb9y\x1d?\x00\x87\x1d?F\x94\x1d?\x8a\xa1\x1d?\xcc\xae\x1d?\r\xbc\x1d?M\xc9\x1d?\x8b\xd6\x1d?\xc8\xe3\x1d?\x04\xf1\x1d?>\xfe\x1d?v\x0b\x1e?\xae\x18\x1e?\xe3%\x1e?\x183\x1e?K@\x1e?|M\x1e?\xacZ\x1e?\xdbg\x1e?\x08u\x1e?4\x82\x1e?_\x8f\x1e?\x88\x9c\x1e?\xaf\xa9\x1e?\xd6\xb6\x1e?\xfa\xc3\x1e?\x1e\xd1\x1e?@\xde\x1e?a\xeb\x1e?\x80\xf8\x1e?\x9e\x05\x1f?\xba\x12\x1f?\xd5\x1f\x1f?\xef,\x1f?\x07:\x1f?\x1eG\x1f?4T\x1f?Ha\x1f?[n\x1f?l{\x1f?|\x88\x1f?\x8b\x95\x1f?\x98\xa2\x1f?\xa4\xaf\x1f?\xaf\xbc\x1f?\xb8\xc9\x1f?\xc0\xd6\x1f?\xc6\xe3\x1f?\xcb\xf0\x1f?\xcf\xfd\x1f?\xd1\n ?\xd2\x17 ?\xd2$ ?\xd01 ?\xcd> ?\xc9K ?\xc3X ?\xbce ?\xb3r ?\xa9\x7f ?\x9e\x8c ?\x92\x99 ?\x84\xa6 ?u\xb3 ?d\xc0 ?R\xcd ??\xda ?*\xe7 ?\x15\xf4 ?\xfd\x00!?\xe5\r!?\xcb\x1a!?\xb0\'!?\x934!?uA!?VN!?5[!?\x14h!?\xf0t!?\xcc\x81!?\xa6\x8e!?\x7f\x9b!?W\xa8!?-\xb5!?\x02\xc2!?\xd5\xce!?\xa8\xdb!?y\xe8!?H\xf5!?\x17\x02\"?\xe4\x0e\"?\xb0\x1b\"?z(\"?C5\"?\x0bB\"?\xd2N\"?\x97[\"?[h\"?\x1eu\"?\xdf\x81\"?\x9f\x8e\"?^\x9b\"?\x1c\xa8\"?\xd8\xb4\"?\x93\xc1\"?M\xce\"?\x05\xdb\"?\xbc\xe7\"?r\xf4\"?\'\x01#?\xda\r#?\x8c\x1a#?=\'#?\xed3#?\x9b@#?HM#?\xf3Y#?\x9ef#?Gs#?\xef\x7f#?\x96\x8c#?;\x99#?\xdf\xa5#?\x82\xb2#?$\xbf#?\xc4\xcb#?c\xd8#?\x01\xe5#?\x9d\xf1#?9\xfe#?\xd3\n$?l\x17$?\x03$$?\x9a0$?/=$?\xc3I$?UV$?\xe7b$?wo$?\x06|$?\x93\x88$? \x95$?\xab\xa1$?5\xae$?\xbe\xba$?E\xc7$?\xcb\xd3$?Q\xe0$?\xd4\xec$?W\xf9$?\xd8\x05%?X\x12%?\xd7\x1e%?U+%?\xd27%?MD%?\xc7P%?@]%?\xb7i%?.v%?\xa3\x82%?\x17\x8f%?\x8a\x9b%?\xfc\xa7%?l\xb4%?\xdb\xc0%?I\xcd%?\xb6\xd9%?!\xe6%?\x8c\xf2%?\xf5\xfe%?]\x0b&?\xc4\x17&?)$&?\x8e0&?\xf1<&?SI&?\xb4U&?\x13b&?rn&?\xcfz&?+\x87&?\x86\x93&?\xe0\x9f&?8\xac&?\x90\xb8&?\xe6\xc4&?;\xd1&?\x8f\xdd&?\xe1\xe9&?3\xf6&?\x83\x02\'?\xd2\x0e\'? \x1b\'?m\'\'?\xb93\'?\x03@\'?LL\'?\x94X\'?\xdbd\'?!q\'?f}\'?\xa9\x89\'?\xec\x95\'?-\xa2\'?m\xae\'?\xac\xba\'?\xe9\xc6\'?&\xd3\'?a\xdf\'?\x9c\xeb\'?\xd5\xf7\'?\r\x04(?C\x10(?y\x1c(?\xad((?\xe14(?\x13A(?DM(?tY(?\xa3e(?\xd0q(?\xfd}(?(\x8a(?S\x96(?|\xa2(?\xa4\xae(?\xcb\xba(?\xf0\xc6(?\x15\xd3(?8\xdf(?[\xeb(?|\xf7(?\x9c\x03)?\xbb\x0f)?\xd9\x1b)?\xf5\')?\x114)?+@)?EL)?]X)?td)?\x8ap)?\x9f|)?\xb3\x88)?\xc5\x94)?\xd7\xa0)?\xe7\xac)?\xf7\xb8)?\x05\xc5)?\x12\xd1)?\x1e\xdd)?)\xe9)?3\xf5)?<\x01*?C\r*?J\x19*?O%*?S1*?W=*?YI*?ZU*?Za*?Ym*?Vy*?S\x85*?O\x91*?I\x9d*?B\xa9*?;\xb5*?2\xc1*?(\xcd*?\x1d\xd9*?\x11\xe5*?\x04\xf1*?\xf6\xfc*?\xe7\x08+?\xd6\x14+?\xc5 +?\xb2,+?\x9f8+?\x8aD+?tP+?^\\+?Fh+?-t+?\x13\x80+?\xf8\x8b+?\xdc\x97+?\xbe\xa3+?\xa0\xaf+?\x81\xbb+?`\xc7+??\xd3+?\x1c\xdf+?\xf9\xea+?\xd4\xf6+?\xae\x02,?\x88\x0e,?`\x1a,?7&,?\r2,?\xe2=,?\xb6I,?\x89U,?[a,?,m,?\xfcx,?\xca\x84,?\x98\x90,?e\x9c,?0\xa8,?\xfb\xb3,?\xc4\xbf,?\x8d\xcb,?T\xd7,?\x1b\xe3,?\xe0\xee,?\xa4\xfa,?h\x06-?*\x12-?\xeb\x1d-?\xab)-?k5-?)A-?\xe6L-?\xa2X-?]d-?\x17p-?\xd0{-?\x88\x87-??\x93-?\xf5\x9e-?\xaa\xaa-?^\xb6-?\x10\xc2-?\xc2\xcd-?s\xd9-?#\xe5-?\xd2\xf0-?\x7f\xfc-?,\x08.?\xd8\x13.?\x83\x1f.?,+.?\xd56.?}B.?#N.?\xc9Y.?ne.?\x11q.?\xb4|.?V\x88.?\xf6\x93.?\x96\x9f.?5\xab.?\xd2\xb6.?o\xc2.?\n\xce.?\xa5\xd9.??\xe5.?\xd7\xf0.?o\xfc.?\x06\x08/?\x9b\x13/?0\x1f/?\xc4*/?V6/?\xe8A/?yM/?\x08Y/?\x97d/?%p/?\xb2{/?=\x87/?\xc8\x92/?R\x9e/?\xdb\xa9/?c\xb5/?\xea\xc0/?o\xcc/?\xf4\xd7/?x\xe3/?\xfb\xee/?}\xfa/?\xfe\x050?~\x110?\xfd\x1c0?{(0?\xf830?t?0?\xf0J0?jV0?\xe3a0?[m0?\xd3x0?I\x840?\xbe\x8f0?3\x9b0?\xa6\xa60?\x19\xb20?\x8a\xbd0?\xfb\xc80?j\xd40?\xd9\xdf0?G\xeb0?\xb3\xf60?\x1f\x021?\x8a\r1?\xf4\x181?]$1?\xc5/1?,;1?\x92F1?\xf7Q1?[]1?\xbeh1? t1?\x82\x7f1?\xe2\x8a1?A\x961?\xa0\xa11?\xfd\xac1?Z\xb81?\xb5\xc31?\x10\xcf1?j\xda1?\xc3\xe51?\x1b\xf11?r\xfc1?\xc8\x072?\x1d\x132?q\x1e2?\xc4)2?\x1652?h@2?\xb8K2?\x07W2?Vb2?\xa4m2?\xf0x2?<\x842?\x87\x8f2?\xd1\x9a2?\x1a\xa62?b\xb12?\xa9\xbc2?\xef\xc72?4\xd32?y\xde2?\xbc\xe92?\xff\xf42?@\x003?\x81\x0b3?\xc1\x163?\xff!3?=-3?z83?\xb6C3?\xf2N3?,Z3?ee3?\x9ep3?\xd5{3?\x0c\x873?B\x923?v\x9d3?\xaa\xa83?\xdd\xb33?\x0f\xbf3?A\xca3?q\xd53?\xa0\xe03?\xcf\xeb3?\xfc\xf63?)\x024?U\r4?\x80\x184?\xa9#4?\xd3.4?\xfb94?\"E4?HP4?n[4?\x92f4?\xb6q4?\xd9|4?\xfb\x874?\x1c\x934?<\x9e4?[\xa94?z\xb44?\x97\xbf4?\xb4\xca4?\xcf\xd54?\xea\xe04?\x04\xec4?\x1d\xf74?5\x025?M\r5?c\x185?x#5?\x8d.5?\xa195?\xb4D5?\xc6O5?\xd7Z5?\xe7e5?\xf6p5?\x05|5?\x12\x875?\x1f\x925?+\x9d5?6\xa85?@\xb35?I\xbe5?R\xc95?Y\xd45?`\xdf5?f\xea5?k\xf55?o\x006?r\x0b6?t\x166?u!6?v,6?v76?uB6?sM6?pX6?lc6?gn6?by6?\\\x846?T\x8f6?L\x9a6?C\xa56?:\xb06?/\xbb6?$\xc66?\x17\xd16?\n\xdc6?\xfc\xe66?\xed\xf16?\xde\xfc6?\xcd\x077?\xbc\x127?\xa9\x1d7?\x96(7?\x8237?n>7?XI7?AT7?*_7?\x12j7?\xf9t7?\xdf\x7f7?\xc4\x8a7?\xa9\x957?\x8c\xa07?o\xab7?Q\xb67?2\xc17?\x12\xcc7?\xf2\xd67?\xd0\xe17?\xae\xec7?\x8b\xf77?g\x028?B\r8?\x1d\x188?\xf6\"8?\xcf-8?\xa788?~C8?UN8?*Y8?\xffc8?\xd2n8?\xa5y8?w\x848?I\x8f8?\x19\x9a8?\xe9\xa48?\xb8\xaf8?\x86\xba8?S\xc58?\x1f\xd08?\xeb\xda8?\xb6\xe58?\x7f\xf08?I\xfb8?\x11\x069?\xd8\x109?\x9f\x1b9?e&9?*19?\xee;9?\xb1F9?tQ9?6\\9?\xf7f9?\xb7q9?v|9?5\x879?\xf2\x919?\xaf\x9c9?k\xa79?\'\xb29?\xe1\xbc9?\x9b\xc79?T\xd29?\x0c\xdd9?\xc3\xe79?y\xf29?/\xfd9?\xe4\x07:?\x98\x12:?K\x1d:?\xfd\':?\xaf2:?`=:?\x10H:?\xbfR:?n]:?\x1bh:?\xc8r:?t}:?\x1f\x88:?\xca\x92:?t\x9d:?\x1c\xa8:?\xc5\xb2:?l\xbd:?\x12\xc8:?\xb8\xd2:?]\xdd:?\x01\xe8:?\xa4\xf2:?G\xfd:?\xe9\x07;?\x8a\x12;?*\x1d;?\xc9\';?h2;?\x06=;?\xa3G;??R;?\xdb\\;?ug;?\x0fr;?\xa9|;?A\x87;?\xd9\x91;?o\x9c;?\x05\xa7;?\x9b\xb1;?/\xbc;?\xc3\xc6;?V\xd1;?\xe8\xdb;?y\xe6;?\n\xf1;?\x9a\xfb;?)\x06<?\xb7\x10<?E\x1b<?\xd1%<?]0<?\xe9:<?sE<?\xfdO<?\x86Z<?\x0ee<?\x95o<?\x1cz<?\xa1\x84<?\'\x8f<?\xab\x99<?.\xa4<?\xb1\xae<?3\xb9<?\xb4\xc3<?5\xce<?\xb5\xd8<?4\xe3<?\xb2\xed<?/\xf8<?\xac\x02=?(\r=?\xa3\x17=?\x1e\"=?\x97,=?\x107=?\x88A=?\x00L=?vV=?\xec`=?ak=?\xd6u=?I\x80=?\xbc\x8a=?.\x95=?\xa0\x9f=?\x10\xaa=?\x80\xb4=?\xef\xbe=?^\xc9=?\xcc\xd3=?8\xde=?\xa5\xe8=?\x10\xf3=?{\xfd=?\xe5\x07>?N\x12>?\xb6\x1c>?\x1e\'>?\x851>?\xeb;>?PF>?\xb5P>?\x19[>?|e>?\xdfo>?Az>?\xa2\x84>?\x02\x8f>?b\x99>?\xc0\xa3>?\x1f\xae>?|\xb8>?\xd9\xc2>?4\xcd>?\x90\xd7>?\xea\xe1>?D\xec>?\x9d\xf6>?\xf5\x00??L\x0b??\xa3\x15??\xf9\x1f??N*??\xa34??\xf7>??JI??\x9cS??\xee]???h??\x8fr??\xdf|??.\x87??|\x91??\xc9\x9b??\x16\xa6??a\xb0??\xad\xba??\xf7\xc4??A\xcf??\x8a\xd9??\xd2\xe3??\x1a\xee??`\xf8??\xa7\x02@?\xec\x0c@?1\x17@?u!@?\xb8+@?\xfb5@?<@@?}J@?\xbeT@?\xfe^@?=i@?{s@?\xb8}@?\xf5\x87@?1\x92@?m\x9c@?\xa8\xa6@?\xe2\xb0@?\x1b\xbb@?T\xc5@?\x8b\xcf@?\xc3\xd9@?\xf9\xe3@?/\xee@?d\xf8@?\x98\x02A?\xcc\x0cA?\xff\x16A?1!A?c+A?\x945A?\xc4?A?\xf3IA?\"TA?P^A?}hA?\xaarA?\xd6|A?\x01\x87A?,\x91A?V\x9bA?\x7f\xa5A?\xa7\xafA?\xcf\xb9A?\xf6\xc3A?\x1d\xceA?B\xd8A?g\xe2A?\x8c\xecA?\xaf\xf6A?\xd2\x00B?\xf4\nB?\x16\x15B?7\x1fB?W)B?v3B?\x95=B?\xb3GB?\xd1QB?\xed[B?\tfB?%pB??zB?Y\x84B?s\x8eB?\x8b\x98B?\xa3\xa2B?\xbb\xacB?\xd1\xb6B?\xe7\xc0B?\xfc\xcaB?\x11\xd5B?%\xdfB?8\xe9B?J\xf3B?\\\xfdB?m\x07C?~\x11C?\x8d\x1bC?\x9c%C?\xab/C?\xb99C?\xc6CC?\xd2MC?\xdeWC?\xe9aC?\xf3kC?\xfduC?\x06\x80C?\x0e\x8aC?\x16\x94C?\x1d\x9eC?#\xa8C?)\xb2C?.\xbcC?2\xc6C?6\xd0C?9\xdaC?;\xe4C?=\xeeC?>\xf8C?>\x02D?>\x0cD?<\x16D?; D?8*D?54D?2>D?-HD?(RD?#\\D?\x1cfD?\x15pD?\x0ezD?\x05\x84D?\xfc\x8dD?\xf3\x97D?\xe8\xa1D?\xde\xabD?\xd2\xb5D?\xc6\xbfD?\xb9\xc9D?\xab\xd3D?\x9d\xddD?\x8e\xe7D?~\xf1D?n\xfbD?]\x05E?L\x0fE?:\x19E?\'#E?\x13-E?\xff6E?\xea@E?\xd5JE?\xbfTE?\xa8^E?\x91hE?yrE?`|E?G\x86E?-\x90E?\x12\x9aE?\xf7\xa3E?\xdb\xadE?\xbe\xb7E?\xa1\xc1E?\x83\xcbE?e\xd5E?E\xdfE?&\xe9E?\x05\xf3E?\xe4\xfcE?\xc2\x06F?\xa0\x10F?}\x1aF?Y$F?5.F?\x108F?\xebAF?\xc4KF?\x9dUF?v_F?NiF?%sF?\xfc|F?\xd2\x86F?\xa7\x90F?|\x9aF?P\xa4F?#\xaeF?\xf6\xb7F?\xc8\xc1F?\x99\xcbF?j\xd5F?:\xdfF?\n\xe9F?\xd9\xf2F?\xa7\xfcF?u\x06G?B\x10G?\x0f\x1aG?\xda#G?\xa6-G?p7G?:AG?\x03KG?\xccTG?\x94^G?[hG?\"rG?\xe8{G?\xae\x85G?s\x8fG?7\x99G?\xfb\xa2G?\xbe\xacG?\x80\xb6G?B\xc0G?\x03\xcaG?\xc4\xd3G?\x84\xddG?C\xe7G?\x02\xf1G?\xc0\xfaG?}\x04H?:\x0eH?\xf6\x17H?\xb2!H?m+H?\'5H?\xe1>H?\x9aHH?RRH?\n\\H?\xc1eH?xoH?.yH?\xe3\x82H?\x98\x8cH?L\x96H?\x00\xa0H?\xb3\xa9H?e\xb3H?\x17\xbdH?\xc8\xc6H?x\xd0H?(\xdaH?\xd8\xe3H?\x86\xedH?4\xf7H?\xe2\x00I?\x8f\nI?;\x14I?\xe6\x1dI?\x91\'I?<1I?\xe6:I?\x8fDI?7NI?\xdfWI?\x87aI?-kI?\xd4tI?y~I?\x1e\x88I?\xc2\x91I?f\x9bI?\t\xa5I?\xac\xaeI?N\xb8I?\xef\xc1I?\x90\xcbI?0\xd5I?\xcf\xdeI?n\xe8I?\r\xf2I?\xaa\xfbI?H\x05J?\xe4\x0eJ?\x80\x18J?\x1b\"J?\xb6+J?P5J?\xea>J?\x83HJ?\x1bRJ?\xb3[J?JeJ?\xe1nJ?wxJ?\x0c\x82J?\xa1\x8bJ?5\x95J?\xc8\x9eJ?[\xa8J?\xee\xb1J?\x80\xbbJ?\x11\xc5J?\xa2\xceJ?2\xd8J?\xc1\xe1J?P\xebJ?\xde\xf4J?l\xfeJ?\xf9\x07K?\x86\x11K?\x11\x1bK?\x9d$K?(.K?\xb27K?;AK?\xc4JK?MTK?\xd4]K?\\gK?\xe2pK?hzK?\xee\x83K?s\x8dK?\xf7\x96K?{\xa0K?\xfe\xa9K?\x81\xb3K?\x03\xbdK?\x84\xc6K?\x05\xd0K?\x85\xd9K?\x05\xe3K?\x84\xecK?\x02\xf6K?\x80\xffK?\xfe\x08L?z\x12L?\xf7\x1bL?r%L?\xed.L?h8L?\xe2AL?[KL?\xd4TL?L^L?\xc3gL?;qL?\xb1zL?\'\x84L?\x9c\x8dL?\x11\x97L?\x85\xa0L?\xf9\xa9L?l\xb3L?\xde\xbcL?P\xc6L?\xc1\xcfL?2\xd9L?\xa2\xe2L?\x12\xecL?\x81\xf5L?\xef\xfeL?]\x08M?\xca\x11M?7\x1bM?\xa3$M?\x0f.M?z7M?\xe4@M?NJM?\xb8SM? ]M?\x89fM?\xf0oM?WyM?\xbe\x82M?$\x8cM?\x89\x95M?\xee\x9eM?R\xa8M?\xb6\xb1M?\x19\xbbM?|\xc4M?\xde\xcdM??\xd7M?\xa0\xe0M?\x01\xeaM?`\xf3M?\xc0\xfcM?\x1e\x06N?|\x0fN?\xda\x18N?7\"N?\x93+N?\xef4N?K>N?\xa5GN?\x00QN?YZN?\xb2cN?\x0bmN?cvN?\xba\x7fN?\x11\x89N?h\x92N?\xbd\x9bN?\x13\xa5N?g\xaeN?\xbc\xb7N?\x0f\xc1N?b\xcaN?\xb5\xd3N?\x07\xddN?X\xe6N?\xa9\xefN?\xf9\xf8N?I\x02O?\x98\x0bO?\xe7\x14O?5\x1eO?\x82\'O?\xcf0O?\x1c:O?gCO?\xb3LO?\xfeUO?H_O?\x92hO?\xdbqO?#{O?k\x84O?\xb3\x8dO?\xfa\x96O?@\xa0O?\x86\xa9O?\xcc\xb2O?\x10\xbcO?U\xc5O?\x98\xceO?\xdc\xd7O?\x1e\xe1O?`\xeaO?\xa2\xf3O?\xe3\xfcO?#\x06P?c\x0fP?\xa3\x18P?\xe2!P? +P?^4P?\x9b=P?\xd8FP?\x14PP?OYP?\x8bbP?\xc5kP?\xfftP?9~P?r\x87P?\xaa\x90P?\xe2\x99P?\x19\xa3P?P\xacP?\x86\xb5P?\xbc\xbeP?\xf1\xc7P?&\xd1P?Z\xdaP?\x8e\xe3P?\xc1\xecP?\xf4\xf5P?&\xffP?W\x08Q?\x88\x11Q?\xb9\x1aQ?\xe8#Q?\x18-Q?G6Q?u?Q?\xa3HQ?\xd0QQ?\xfdZQ?)dQ?UmQ?\x80vQ?\xab\x7fQ?\xd5\x88Q?\xfe\x91Q?\'\x9bQ?P\xa4Q?x\xadQ?\x9f\xb6Q?\xc6\xbfQ?\xed\xc8Q?\x13\xd2Q?8\xdbQ?]\xe4Q?\x81\xedQ?\xa5\xf6Q?\xc8\xffQ?\xeb\x08R?\r\x12R?/\x1bR?P$R?q-R?\x916R?\xb1?R?\xd0HR?\xefQR?\r[R?*dR?GmR?dvR?\x80\x7fR?\x9b\x88R?\xb6\x91R?\xd1\x9aR?\xeb\xa3R?\x04\xadR?\x1d\xb6R?6\xbfR?N\xc8R?e\xd1R?|\xdaR?\x92\xe3R?\xa8\xecR?\xbd\xf5R?\xd2\xfeR?\xe6\x07S?\xfa\x10S?\x0e\x1aS? #S?3,S?D5S?V>S?fGS?vPS?\x86YS?\x95bS?\xa4kS?\xb2tS?\xc0}S?\xcd\x86S?\xda\x8fS?\xe6\x98S?\xf1\xa1S?\xfd\xaaS?\x07\xb4S?\x11\xbdS?\x1b\xc6S?$\xcfS?-\xd8S?5\xe1S?<\xeaS?C\xf3S?J\xfcS?P\x05T?V\x0eT?[\x17T?_ T?c)T?g2T?j;T?lDT?nMT?pVT?q_T?qhT?rqT?qzT?p\x83T?o\x8cT?m\x95T?j\x9eT?g\xa7T?d\xb0T?`\xb9T?[\xc2T?V\xcbT?Q\xd4T?K\xddT?E\xe6T?>\xefT?6\xf8T?.\x01U?&\nU?\x1d\x13U?\x13\x1cU?\n%U?\xff-U?\xf46U?\xe9?U?\xddHU?\xd1QU?\xc4ZU?\xb6cU?\xa8lU?\x9auU?\x8b~U?|\x87U?l\x90U?\\\x99U?K\xa2U?:\xabU?(\xb4U?\x15\xbdU?\x03\xc6U?\xef\xceU?\xdc\xd7U?\xc7\xe0U?\xb3\xe9U?\x9d\xf2U?\x88\xfbU?r\x04V?[\rV?D\x16V?,\x1fV?\x14(V?\xfb0V?\xe29V?\xc9BV?\xaeKV?\x94TV?y]V?]fV?AoV?%xV?\x08\x81V?\xea\x89V?\xcc\x92V?\xae\x9bV?\x8f\xa4V?o\xadV?P\xb6V?/\xbfV?\x0e\xc8V?\xed\xd0V?\xcb\xd9V?\xa9\xe2V?\x86\xebV?c\xf4V??\xfdV?\x1b\x06W?\xf6\x0eW?\xd1\x17W?\xab W?\x85)W?_2W?8;W?\x10DW?\xe8LW?\xbfUW?\x96^W?mgW?CpW?\x18yW?\xee\x81W?\xc2\x8aW?\x96\x93W?j\x9cW?=\xa5W?\x10\xaeW?\xe2\xb6W?\xb4\xbfW?\x85\xc8W?V\xd1W?&\xdaW?\xf6\xe2W?\xc6\xebW?\x95\xf4W?c\xfdW?1\x06X?\xff\x0eX?\xcc\x17X?\x98 X?d)X?02X?\xfb:X?\xc6CX?\x90LX?ZUX?#^X?\xecfX?\xb4oX?|xX?C\x81X?\n\x8aX?\xd1\x92X?\x97\x9bX?\\\xa4X?!\xadX?\xe6\xb5X?\xaa\xbeX?n\xc7X?1\xd0X?\xf4\xd8X?\xb6\xe1X?x\xeaX?9\xf3X?\xfa\xfbX?\xbb\x04Y?z\rY?:\x16Y?\xf9\x1eY?\xb8\'Y?v0Y?39Y?\xf0AY?\xadJY?iSY?%\\Y?\xe1dY?\x9bmY?VvY?\x10\x7fY?\xc9\x87Y?\x82\x90Y?;\x99Y?\xf3\xa1Y?\xab\xaaY?b\xb3Y?\x19\xbcY?\xcf\xc4Y?\x85\xcdY?:\xd6Y?\xef\xdeY?\xa3\xe7Y?W\xf0Y?\x0b\xf9Y?\xbe\x01Z?q\nZ?#\x13Z?\xd4\x1bZ?\x86$Z?6-Z?\xe75Z?\x97>Z?FGZ?\xf5OZ?\xa4XZ?RaZ?\xffiZ?\xacrZ?Y{Z?\x05\x84Z?\xb1\x8cZ?\\\x95Z?\x07\x9eZ?\xb2\xa6Z?\\\xafZ?\x05\xb8Z?\xae\xc0Z?W\xc9Z?\xff\xd1Z?\xa7\xdaZ?N\xe3Z?\xf5\xebZ?\x9b\xf4Z?A\xfdZ?\xe7\x05[?\x8c\x0e[?1\x17[?\xd5\x1f[?x([?\x1c1[?\xbe9[?aB[?\x03K[?\xa4S[?E\\[?\xe6d[?\x86m[?%v[?\xc5~[?c\x87[?\x02\x90[?\xa0\x98[?=\xa1[?\xda\xa9[?w\xb2[?\x13\xbb[?\xae\xc3[?J\xcc[?\xe4\xd4[?\x7f\xdd[?\x19\xe6[?\xb2\xee[?K\xf7[?\xe4\xff[?|\x08\\?\x14\x11\\?\xab\x19\\?B\"\\?\xd8*\\?n3\\?\x03<\\?\x98D\\?-M\\?\xc1U\\?U^\\?\xe8f\\?{o\\?\rx\\?\x9f\x80\\?1\x89\\?\xc2\x91\\?S\x9a\\?\xe3\xa2\\?s\xab\\?\x02\xb4\\?\x91\xbc\\?\x1f\xc5\\?\xad\xcd\\?;\xd6\\?\xc8\xde\\?U\xe7\\?\xe1\xef\\?m\xf8\\?\xf8\x00]?\x83\t]?\x0e\x12]?\x98\x1a]?\"#]?\xab+]?44]?\xbc<]?DE]?\xccM]?SV]?\xd9^]?`g]?\xe5o]?kx]?\xf0\x80]?t\x89]?\xf8\x91]?|\x9a]?\xff\xa2]?\x82\xab]?\x04\xb4]?\x86\xbc]?\x07\xc5]?\x89\xcd]?\t\xd6]?\x89\xde]?\t\xe7]?\x88\xef]?\x07\xf8]?\x86\x00^?\x04\t^?\x81\x11^?\xff\x19^?{\"^?\xf8*^?t3^?\xef;^?jD^?\xe5L^?_U^?\xd9]^?Rf^?\xcbn^?Dw^?\xbc\x7f^?3\x88^?\xab\x90^?!\x99^?\x98\xa1^?\x0e\xaa^?\x83\xb2^?\xf8\xba^?m\xc3^?\xe1\xcb^?U\xd4^?\xc9\xdc^?<\xe5^?\xae\xed^? \xf6^?\x92\xfe^?\x04\x07_?t\x0f_?\xe5\x17_?U _?\xc5(_?41_?\xa39_?\x11B_?\x7fJ_?\xedR_?Z[_?\xc6c_?3l_?\x9ft_?\n}_?u\x85_?\xe0\x8d_?J\x96_?\xb4\x9e_?\x1d\xa7_?\x86\xaf_?\xee\xb7_?W\xc0_?\xbe\xc8_?&\xd1_?\x8c\xd9_?\xf3\xe1_?Y\xea_?\xbf\xf2_?$\xfb_?\x89\x03`?\xed\x0b`?Q\x14`?\xb4\x1c`?\x18%`?z-`?\xdd5`?>>`?\xa0F`?\x01O`?bW`?\xc2_`?\"h`?\x81p`?\xe0x`??\x81`?\x9d\x89`?\xfb\x91`?X\x9a`?\xb5\xa2`?\x12\xab`?n\xb3`?\xca\xbb`?%\xc4`?\x80\xcc`?\xda\xd4`?4\xdd`?\x8e\xe5`?\xe7\xed`?@\xf6`?\x99\xfe`?\xf1\x06a?H\x0fa?\xa0\x17a?\xf6\x1fa?M(a?\xa30a?\xf98a?NAa?\xa3Ia?\xf7Qa?KZa?\x9eba?\xf2ja?Dsa?\x97{a?\xe9\x83a?:\x8ca?\x8b\x94a?\xdc\x9ca?-\xa5a?}\xada?\xcc\xb5a?\x1b\xbea?j\xc6a?\xb8\xcea?\x06\xd7a?T\xdfa?\xa1\xe7a?\xee\xefa?:\xf8a?\x86\x00b?\xd1\x08b?\x1c\x11b?g\x19b?\xb1!b?\xfb)b?E2b?\x8e:b?\xd7Bb?\x1fKb?gSb?\xae[b?\xf6cb?<lb?\x83tb?\xc9|b?\x0e\x85b?S\x8db?\x98\x95b?\xdc\x9db? \xa6b?d\xaeb?\xa7\xb6b?\xea\xbeb?,\xc7b?n\xcfb?\xaf\xd7b?\xf1\xdfb?1\xe8b?r\xf0b?\xb2\xf8b?\xf1\x00c?0\tc?o\x11c?\xae\x19c?\xec!c?)*c?f2c?\xa3:c?\xe0Bc?\x1cKc?WSc?\x92[c?\xcdcc?\x08lc?Btc?{|c?\xb5\x84c?\xee\x8cc?&\x95c?^\x9dc?\x96\xa5c?\xcd\xadc?\x04\xb6c?;\xbec?q\xc6c?\xa7\xcec?\xdc\xd6c?\x11\xdfc?F\xe7c?z\xefc?\xae\xf7c?\xe1\xffc?\x14\x08d?G\x10d?y\x18d?\xab d?\xdc(d?\r1d?>9d?nAd?\x9eId?\xceQd?\xfdYd?,bd?Zjd?\x88rd?\xb6zd?\xe3\x82d?\x10\x8bd?<\x93d?h\x9bd?\x94\xa3d?\xbf\xabd?\xea\xb3d?\x15\xbcd??\xc4d?i\xccd?\x92\xd4d?\xbb\xdcd?\xe4\xe4d?\x0c\xedd?4\xf5d?[\xfdd?\x82\x05e?\xa9\re?\xcf\x15e?\xf5\x1de?\x1b&e?@.e?e6e?\x89>e?\xadFe?\xd1Ne?\xf4Ve?\x17_e?9ge?[oe?}we?\x9e\x7fe?\xbf\x87e?\xe0\x8fe?\x00\x98e? \xa0e??\xa8e?_\xb0e?}\xb8e?\x9c\xc0e?\xba\xc8e?\xd7\xd0e?\xf4\xd8e?\x11\xe1e?.\xe9e?J\xf1e?e\xf9e?\x81\x01f?\x9b\tf?\xb6\x11f?\xd0\x19f?\xea!f?\x03*f?\x1c2f?5:f?MBf?eJf?}Rf?\x94Zf?\xabbf?\xc1jf?\xd7rf?\xedzf?\x02\x83f?\x17\x8bf?,\x93f?@\x9bf?T\xa3f?g\xabf?z\xb3f?\x8d\xbbf?\x9f\xc3f?\xb1\xcbf?\xc2\xd3f?\xd4\xdbf?\xe4\xe3f?\xf5\xebf?\x05\xf4f?\x15\xfcf?$\x04g?3\x0cg?B\x14g?P\x1cg?^$g?k,g?x4g?\x85<g?\x91Dg?\x9dLg?\xa9Tg?\xb4\\g?\xbfdg?\xcalg?\xd4tg?\xde|g?\xe7\x84g?\xf0\x8cg?\xf9\x94g?\x01\x9dg?\t\xa5g?\x10\xadg?\x18\xb5g?\x1f\xbdg?%\xc5g?+\xcdg?1\xd5g?6\xddg?;\xe5g?@\xedg?D\xf5g?H\xfdg?L\x05h?O\rh?R\x15h?T\x1dh?V%h?X-h?Y5h?Z=h?[Eh?[Mh?[Uh?[]h?Zeh?Ymh?Wuh?U}h?S\x85h?P\x8dh?M\x95h?J\x9dh?F\xa5h?B\xadh?>\xb5h?9\xbdh?4\xc5h?.\xcdh?(\xd5h?\"\xddh?\x1b\xe5h?\x14\xedh?\r\xf5h?\x05\xfdh?\xfd\x04i?\xf5\x0ci?\xec\x14i?\xe3\x1ci?\xd9$i?\xcf,i?\xc54i?\xbb<i?\xb0Di?\xa5Li?\x99Ti?\x8d\\i?\x80di?tli?gti?Y|i?K\x84i?=\x8ci?/\x94i? \x9ci?\x11\xa4i?\x01\xaci?\xf1\xb3i?\xe1\xbbi?\xd0\xc3i?\xbf\xcbi?\xae\xd3i?\x9c\xdbi?\x8a\xe3i?x\xebi?e\xf3i?R\xfbi?>\x03j?+\x0bj?\x16\x13j?\x02\x1bj?\xed\"j?\xd8*j?\xc22j?\xac:j?\x96Bj?\x7fJj?hRj?QZj?9bj?!jj?\trj?\xf0yj?\xd7\x81j?\xbd\x89j?\xa3\x91j?\x89\x99j?o\xa1j?T\xa9j?8\xb1j?\x1d\xb9j?\x01\xc1j?\xe5\xc8j?\xc8\xd0j?\xab\xd8j?\x8e\xe0j?p\xe8j?R\xf0j?4\xf8j?\x15\x00k?\xf6\x07k?\xd6\x0fk?\xb7\x17k?\x96\x1fk?v\'k?U/k?47k?\x12?k?\xf1Fk?\xceNk?\xacVk?\x89^k?ffk?Bnk?\x1evk?\xfa}k?\xd5\x85k?\xb0\x8dk?\x8b\x95k?e\x9dk??\xa5k?\x19\xadk?\xf2\xb4k?\xcb\xbck?\xa3\xc4k?|\xcck?T\xd4k?+\xdck?\x02\xe4k?\xd9\xebk?\xb0\xf3k?\x86\xfbk?\\\x03l?1\x0bl?\x06\x13l?\xdb\x1al?\xb0\"l?\x84*l?X2l?+:l?\xfeAl?\xd1Il?\xa3Ql?uYl?Gal?\x18il?\xe9pl?\xbaxl?\x8b\x80l?[\x88l?*\x90l?\xfa\x97l?\xc9\x9fl?\x97\xa7l?f\xafl?3\xb7l?\x01\xbfl?\xce\xc6l?\x9b\xcel?h\xd6l?4\xdel?\x00\xe6l?\xcc\xedl?\x97\xf5l?b\xfdl?-\x05m?\xf7\x0cm?\xc1\x14m?\x8a\x1cm?S$m?\x1c,m?\xe53m?\xad;m?uCm?=Km?\x04Sm?\xcbZm?\x91bm?Wjm?\x1drm?\xe3ym?\xa8\x81m?m\x89m?1\x91m?\xf5\x98m?\xb9\xa0m?}\xa8m?@\xb0m?\x03\xb8m?\xc5\xbfm?\x87\xc7m?I\xcfm?\x0b\xd7m?\xcc\xdem?\x8d\xe6m?M\xeem?\r\xf6m?\xcd\xfdm?\x8d\x05n?L\rn?\x0b\x15n?\xc9\x1cn?\x87$n?E,n?\x034n?\xc0;n?}Cn?9Kn?\xf5Rn?\xb1Zn?mbn?(jn?\xe3qn?\x9dyn?W\x81n?\x11\x89n?\xcb\x90n?\x84\x98n?=\xa0n?\xf5\xa7n?\xae\xafn?e\xb7n?\x1d\xbfn?\xd4\xc6n?\x8b\xcen?B\xd6n?\xf8\xddn?\xae\xe5n?c\xedn?\x19\xf5n?\xcd\xfcn?\x82\x04o?6\x0co?\xea\x13o?\x9e\x1bo?Q#o?\x04+o?\xb72o?i:o?\x1bBo?\xccIo?~Qo?/Yo?\xdf`o?\x90ho?@po?\xefwo?\x9f\x7fo?N\x87o?\xfc\x8eo?\xab\x96o?Y\x9eo?\x06\xa6o?\xb4\xado?a\xb5o?\x0e\xbdo?\xba\xc4o?f\xcco?\x12\xd4o?\xbd\xdbo?h\xe3o?\x13\xebo?\xbe\xf2o?h\xfao?\x12\x02p?\xbb\tp?d\x11p?\r\x19p?\xb6 p?^(p?\x060p?\xad7p?U?p?\xfbFp?\xa2Np?HVp?\xee]p?\x94ep?9mp?\xdetp?\x83|p?\'\x84p?\xcb\x8bp?o\x93p?\x12\x9bp?\xb5\xa2p?X\xaap?\xfa\xb1p?\x9d\xb9p?>\xc1p?\xe0\xc8p?\x81\xd0p?\"\xd8p?\xc2\xdfp?b\xe7p?\x02\xefp?\xa2\xf6p?A\xfep?\xe0\x05q?~\rq?\x1d\x15q?\xbb\x1cq?X$q?\xf6+q?\x933q?/;q?\xccBq?hJq?\x03Rq?\x9fYq?:aq?\xd5hq?opq?\txq?\xa3\x7fq?<\x87q?\xd6\x8eq?o\x96q?\x07\x9eq?\x9f\xa5q?7\xadq?\xcf\xb4q?f\xbcq?\xfd\xc3q?\x94\xcbq?*\xd3q?\xc0\xdaq?V\xe2q?\xeb\xe9q?\x80\xf1q?\x15\xf9q?\xaa\x00r?>\x08r?\xd2\x0fr?e\x17r?\xf8\x1er?\x8b&r?\x1e.r?\xb05r?B=r?\xd4Dr?eLr?\xf6Sr?\x87[r?\x17cr?\xa7jr?7rr?\xc6yr?U\x81r?\xe4\x88r?s\x90r?\x01\x98r?\x8f\x9fr?\x1c\xa7r?\xaa\xaer?7\xb6r?\xc3\xbdr?P\xc5r?\xdc\xccr?g\xd4r?\xf3\xdbr?~\xe3r?\t\xebr?\x93\xf2r?\x1d\xfar?\xa7\x01s?1\ts?\xba\x10s?C\x18s?\xcb\x1fs?T\'s?\xdc.s?c6s?\xeb=s?rEs?\xf9Ls?\x7fTs?\x05\\s?\x8bcs?\x11ks?\x96rs?\x1bzs?\x9f\x81s?$\x89s?\xa8\x90s?+\x98s?\xaf\x9fs?2\xa7s?\xb5\xaes?7\xb6s?\xb9\xbds?;\xc5s?\xbd\xccs?>\xd4s?\xbf\xdbs??\xe3s?\xc0\xeas?@\xf2s?\xc0\xf9s??\x01t?\xbe\x08t?=\x10t?\xbb\x17t?:\x1ft?\xb7&t?5.t?\xb25t?/=t?\xacDt?(Lt?\xa4St? [t?\x9cbt?\x17jt?\x92qt?\x0cyt?\x86\x80t?\x00\x88t?z\x8ft?\xf3\x96t?l\x9et?\xe5\xa5t?]\xadt?\xd6\xb4t?M\xbct?\xc5\xc3t?<\xcbt?\xb3\xd2t?*\xdat?\xa0\xe1t?\x16\xe9t?\x8c\xf0t?\x01\xf8t?v\xfft?\xeb\x06u?`\x0eu?\xd4\x15u?H\x1du?\xbb$u?/,u?\xa23u?\x14;u?\x87Bu?\xf9Iu?kQu?\xdcXu?N`u?\xbegu?/ou?\x9fvu?\x0f~u?\x7f\x85u?\xef\x8cu?^\x94u?\xcd\x9bu?;\xa3u?\xa9\xaau?\x17\xb2u?\x85\xb9u?\xf2\xc0u?_\xc8u?\xcc\xcfu?8\xd7u?\xa5\xdeu?\x11\xe6u?|\xedu?\xe7\xf4u?R\xfcu?\xbd\x03v?\'\x0bv?\x91\x12v?\xfb\x19v?e!v?\xce(v?70v?\x9f7v?\x08?v?pFv?\xd7Mv??Uv?\xa6\\v?\rdv?skv?\xdarv?@zv?\xa5\x81v?\x0b\x89v?p\x90v?\xd4\x97v?9\x9fv?\x9d\xa6v?\x01\xaev?e\xb5v?\xc8\xbcv?+\xc4v?\x8e\xcbv?\xf0\xd2v?R\xdav?\xb4\xe1v?\x16\xe9v?w\xf0v?\xd8\xf7v?9\xffv?\x99\x06w?\xf9\rw?Y\x15w?\xb8\x1cw?\x18$w?w+w?\xd52w?4:w?\x92Aw?\xefHw?MPw?\xaaWw?\x07_w?dfw?\xc0mw?\x1cuw?x|w?\xd3\x83w?.\x8bw?\x89\x92w?\xe4\x99w?>\xa1w?\x98\xa8w?\xf2\xafw?K\xb7w?\xa4\xbew?\xfd\xc5w?V\xcdw?\xae\xd4w?\x06\xdcw?^\xe3w?\xb5\xeaw?\x0c\xf2w?c\xf9w?\xb9\x00x?\x10\x08x?f\x0fx?\xbb\x16x?\x11\x1ex?f%x?\xbb,x?\x0f4x?c;x?\xb7Bx?\x0bJx?^Qx?\xb2Xx?\x04`x?Wgx?\xa9nx?\xfbux?M}x?\x9e\x84x?\xef\x8bx?@\x93x?\x91\x9ax?\xe1\xa1x?1\xa9x?\x80\xb0x?\xd0\xb7x?\x1f\xbfx?n\xc6x?\xbc\xcdx?\n\xd5x?X\xdcx?\xa6\xe3x?\xf3\xeax?A\xf2x?\x8d\xf9x?\xda\x00y?&\x08y?r\x0fy?\xbe\x16y?\t\x1ey?T%y?\x9f,y?\xea3y?4;y?~By?\xc8Iy?\x11Qy?ZXy?\xa3_y?\xecfy?4ny?|uy?\xc4|y?\x0b\x84y?S\x8by?\x99\x92y?\xe0\x99y?&\xa1y?l\xa8y?\xb2\xafy?\xf8\xb6y?=\xbey?\x82\xc5y?\xc7\xccy?\x0b\xd4y?O\xdby?\x93\xe2y?\xd6\xe9y?\x1a\xf1y?]\xf8y?\x9f\xffy?\xe2\x06z?$\x0ez?f\x15z?\xa7\x1cz?\xe8#z?*+z?j2z?\xab9z?\xeb@z?+Hz?jOz?\xaaVz?\xe9]z?(ez?flz?\xa4sz?\xe2zz? \x82z?]\x89z?\x9b\x90z?\xd7\x97z?\x14\x9fz?P\xa6z?\x8c\xadz?\xc8\xb4z?\x03\xbcz??\xc3z?z\xcaz?\xb4\xd1z?\xef\xd8z?)\xe0z?b\xe7z?\x9c\xeez?\xd5\xf5z?\x0e\xfdz?G\x04{?\x7f\x0b{?\xb7\x12{?\xef\x19{?\'!{?^({?\x95/{?\xcc6{?\x03>{?9E{?oL{?\xa4S{?\xdaZ{?\x0fb{?Di{?xp{?\xadw{?\xe1~{?\x15\x86{?H\x8d{?{\x94{?\xae\x9b{?\xe1\xa2{?\x13\xaa{?E\xb1{?w\xb8{?\xa9\xbf{?\xda\xc6{?\x0b\xce{?<\xd5{?l\xdc{?\x9d\xe3{?\xcd\xea{?\xfc\xf1{?,\xf9{?[\x00|?\x8a\x07|?\xb8\x0e|?\xe7\x15|?\x15\x1d|?B$|?p+|?\x9d2|?\xca9|?\xf7@|?#H|?OO|?{V|?\xa7]|?\xd2d|?\xfdk|?(s|?Sz|?}\x81|?\xa7\x88|?\xd1\x8f|?\xfa\x96|?#\x9e|?L\xa5|?u\xac|?\x9d\xb3|?\xc5\xba|?\xed\xc1|?\x15\xc9|?<\xd0|?c\xd7|?\x8a\xde|?\xb0\xe5|?\xd6\xec|?\xfc\xf3|?\"\xfb|?H\x02}?m\t}?\x92\x10}?\xb6\x17}?\xdb\x1e}?\xff%}?\"-}?F4}?i;}?\x8cB}?\xafI}?\xd2P}?\xf4W}?\x16_}?7f}?Ym}?zt}?\x9b{}?\xbc\x82}?\xdc\x89}?\xfc\x90}?\x1c\x98}?;\x9f}?[\xa6}?z\xad}?\x98\xb4}?\xb7\xbb}?\xd5\xc2}?\xf3\xc9}?\x11\xd1}?.\xd8}?K\xdf}?h\xe6}?\x85\xed}?\xa1\xf4}?\xbe\xfb}?\xd9\x02~?\xf5\t~?\x10\x11~?+\x18~?F\x1f~?a&~?{-~?\x954~?\xaf;~?\xc8B~?\xe1I~?\xfaP~?\x13X~?,_~?Df~?\\m~?st~?\x8b{~?\xa2\x82~?\xb9\x89~?\xcf\x90~?\xe6\x97~?\xfc\x9e~?\x11\xa6~?\'\xad~?<\xb4~?Q\xbb~?f\xc2~?{\xc9~?\x8f\xd0~?\xa3\xd7~?\xb7\xde~?\xca\xe5~?\xdd\xec~?\xf0\xf3~?\x03\xfb~?\x15\x02\x7f?\'\t\x7f?9\x10\x7f?K\x17\x7f?\\\x1e\x7f?n%\x7f?~,\x7f?\x8f3\x7f?\x9f:\x7f?\xafA\x7f?\xbfH\x7f?\xcfO\x7f?\xdeV\x7f?\xed]\x7f?\xfcd\x7f?\nl\x7f?\x19s\x7f?\'z\x7f?4\x81\x7f?B\x88\x7f?O\x8f\x7f?\\\x96\x7f?i\x9d\x7f?u\xa4\x7f?\x82\xab\x7f?\x8e\xb2\x7f?\x99\xb9\x7f?\xa5\xc0\x7f?\xb0\xc7\x7f?\xbb\xce\x7f?\xc5\xd5\x7f?\xd0\xdc\x7f?\xda\xe3\x7f?\xe4\xea\x7f?\xee\xf1\x7f?\xf7\xf8\x7f?\x00\x00\x80?",
    );
    static LINEAR_TO_SRGB_U8: [u8; 4096] = *b"\x00\x01\x02\x02\x03\x04\x05\x06\x06\x07\x08\t\n\n\x0b\x0c\r\r\x0e\x0f\x0f\x10\x10\x11\x12\x12\x13\x13\x14\x14\x15\x15\x16\x16\x17\x17\x17\x18\x18\x19\x19\x19\x1a\x1a\x1b\x1b\x1b\x1c\x1c\x1d\x1d\x1d\x1e\x1e\x1e\x1f\x1f\x1f   !!!\"\"\"\"###$$$%%%%&&&&\'\'\'(((())))****+++++,,,,----.....////0000011111222223333344444555556666677777788888999999::::::;;;;;;<<<<<<======>>>>>>??????@@@@@@@AAAAAABBBBBBBCCCCCCCDDDDDDDEEEEEEEFFFFFFFGGGGGGGHHHHHHHHIIIIIIIJJJJJJJJKKKKKKKKLLLLLLLMMMMMMMMNNNNNNNNNOOOOOOOOPPPPPPPPQQQQQQQQQRRRRRRRRSSSSSSSSSTTTTTTTTTUUUUUUUUUVVVVVVVVVWWWWWWWWWXXXXXXXXXXYYYYYYYYYZZZZZZZZZZ[[[[[[[[[[\\\\\\\\\\\\\\\\\\\\]]]]]]]]]]^^^^^^^^^^__________```````````aaaaaaaaaabbbbbbbbbbbcccccccccccdddddddddddeeeeeeeeeeefffffffffffgggggggggggghhhhhhhhhhhiiiiiiiiiiiijjjjjjjjjjjjkkkkkkkkkkkkllllllllllllmmmmmmmmmmmmnnnnnnnnnnnnoooooooooooooppppppppppppqqqqqqqqqqqqqrrrrrrrrrrrrrssssssssssssstttttttttttttuuuuuuuuuuuuuuvvvvvvvvvvvvvwwwwwwwwwwwwwwxxxxxxxxxxxxxxyyyyyyyyyyyyyzzzzzzzzzzzzzzz{{{{{{{{{{{{{{||||||||||||||}}}}}}}}}}}}}}}~~~~~~~~~~~~~~\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x81\x81\x81\x81\x81\x81\x81\x81\x81\x81\x81\x81\x81\x81\x81\x82\x82\x82\x82\x82\x82\x82\x82\x82\x82\x82\x82\x82\x82\x82\x83\x83\x83\x83\x83\x83\x83\x83\x83\x83\x83\x83\x83\x83\x83\x83\x84\x84\x84\x84\x84\x84\x84\x84\x84\x84\x84\x84\x84\x84\x84\x85\x85\x85\x85\x85\x85\x85\x85\x85\x85\x85\x85\x85\x85\x85\x85\x86\x86\x86\x86\x86\x86\x86\x86\x86\x86\x86\x86\x86\x86\x86\x86\x87\x87\x87\x87\x87\x87\x87\x87\x87\x87\x87\x87\x87\x87\x87\x87\x88\x88\x88\x88\x88\x88\x88\x88\x88\x88\x88\x88\x88\x88\x88\x88\x89\x89\x89\x89\x89\x89\x89\x89\x89\x89\x89\x89\x89\x89\x89\x89\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8a\x8b\x8b\x8b\x8b\x8b\x8b\x8b\x8b\x8b\x8b\x8b\x8b\x8b\x8b\x8b\x8b\x8c\x8c\x8c\x8c\x8c\x8c\x8c\x8c\x8c\x8c\x8c\x8c\x8c\x8c\x8c\x8c\x8c\x8d\x8d\x8d\x8d\x8d\x8d\x8d\x8d\x8d\x8d\x8d\x8d\x8d\x8d\x8d\x8d\x8d\x8e\x8e\x8e\x8e\x8e\x8e\x8e\x8e\x8e\x8e\x8e\x8e\x8e\x8e\x8e\x8e\x8e\x8f\x8f\x8f\x8f\x8f\x8f\x8f\x8f\x8f\x8f\x8f\x8f\x8f\x8f\x8f\x8f\x8f\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x90\x91\x91\x91\x91\x91\x91\x91\x91\x91\x91\x91\x91\x91\x91\x91\x91\x91\x91\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x92\x93\x93\x93\x93\x93\x93\x93\x93\x93\x93\x93\x93\x93\x93\x93\x93\x93\x93\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x94\x95\x95\x95\x95\x95\x95\x95\x95\x95\x95\x95\x95\x95\x95\x95\x95\x95\x95\x96\x96\x96\x96\x96\x96\x96\x96\x96\x96\x96\x96\x96\x96\x96\x96\x96\x96\x96\x97\x97\x97\x97\x97\x97\x97\x97\x97\x97\x97\x97\x97\x97\x97\x97\x97\x97\x98\x98\x98\x98\x98\x98\x98\x98\x98\x98\x98\x98\x98\x98\x98\x98\x98\x98\x98\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x9a\x9a\x9a\x9a\x9a\x9a\x9a\x9a\x9a\x9a\x9a\x9a\x9a\x9a\x9a\x9a\x9a\x9a\x9a\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9c\x9c\x9c\x9c\x9c\x9c\x9c\x9c\x9c\x9c\x9c\x9c\x9c\x9c\x9c\x9c\x9c\x9c\x9c\x9c\x9d\x9d\x9d\x9d\x9d\x9d\x9d\x9d\x9d\x9d\x9d\x9d\x9d\x9d\x9d\x9d\x9d\x9d\x9d\x9e\x9e\x9e\x9e\x9e\x9e\x9e\x9e\x9e\x9e\x9e\x9e\x9e\x9e\x9e\x9e\x9e\x9e\x9e\x9f\x9f\x9f\x9f\x9f\x9f\x9f\x9f\x9f\x9f\x9f\x9f\x9f\x9f\x9f\x9f\x9f\x9f\x9f\x9f\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa1\xa1\xa1\xa1\xa1\xa1\xa1\xa1\xa1\xa1\xa1\xa1\xa1\xa1\xa1\xa1\xa1\xa1\xa1\xa1\xa2\xa2\xa2\xa2\xa2\xa2\xa2\xa2\xa2\xa2\xa2\xa2\xa2\xa2\xa2\xa2\xa2\xa2\xa2\xa2\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa3\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa4\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa5\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa6\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa7\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa8\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xa9\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xab\xab\xab\xab\xab\xab\xab\xab\xab\xab\xab\xab\xab\xab\xab\xab\xab\xab\xab\xab\xab\xab\xac\xac\xac\xac\xac\xac\xac\xac\xac\xac\xac\xac\xac\xac\xac\xac\xac\xac\xac\xac\xac\xac\xad\xad\xad\xad\xad\xad\xad\xad\xad\xad\xad\xad\xad\xad\xad\xad\xad\xad\xad\xad\xad\xad\xae\xae\xae\xae\xae\xae\xae\xae\xae\xae\xae\xae\xae\xae\xae\xae\xae\xae\xae\xae\xae\xae\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xaf\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb0\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb1\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb2\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb3\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb4\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb5\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb6\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb7\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb8\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xb9\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xba\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbb\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbc\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbe\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xbf\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc0\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc1\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc2\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc3\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc4\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc5\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc7\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc8\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xc9\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xca\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcb\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xce\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xcf\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd0\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd1\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd2\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd3\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd4\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd5\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd6\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd7\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd8\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xd9\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xda\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdb\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdc\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xdd\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xdf\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe0\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe1\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe2\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe3\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe4\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe5\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe6\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe7\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe8\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xe9\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xea\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xeb\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xec\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xed\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xee\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xef\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf1\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf2\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf3\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf4\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf5\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf6\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf7\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf8\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xf9\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfa\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfb\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfc\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfd\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff";
    /// sRGB u8 → linear f32 lookup table (256 entries).
    #[inline]
    pub(crate) fn linear_table_8() -> &'static [f32; 256] {
        let slice: &[f32] = bytemuck::cast_slice(&LINEAR_TABLE_8.0);
        slice.try_into().unwrap()
    }
    /// Linear → sRGB f32 lookup table (4096 entries, 12-bit resolution).
    #[inline]
    pub(crate) fn encode_table_12() -> &'static [f32; 4096] {
        let slice: &[f32] = bytemuck::cast_slice(&ENCODE_TABLE_12.0);
        slice.try_into().unwrap()
    }
    /// Linear (12-bit index) → sRGB u8 lookup table (4096 entries).
    #[inline]
    pub(crate) fn linear_to_srgb_u8() -> &'static [u8; 4096] {
        &LINEAR_TO_SRGB_U8
    }
}
#[doc(hidden)]
pub mod u16_lut {
    //! Lazily-initialized u16 sRGB lookup tables.
    //!
    //! Tables are generated on first use via `OnceLock` — no binary bloat,
    //! no compile-time cost. Only allocated if the u16 API is actually called.
    //!
    //! - **Decode** (sRGB u16 → linear f32): 65536-entry direct lookup, 256KB.
    //!   Exact — each u16 value has its own f32 entry.
    //!
    //! - **Encode** (linear f32 → sRGB u16): 65537-entry sqrt-indexed LUT, 128KB.
    //!   Indexed by `sqrt(linear) * 65536`, which concentrates resolution where
    //!   the sRGB curve is steepest (near black). Max roundtrip error ±1 u16 level,
    //!   94.2% exact roundtrip (vs 71.3% / ±6 with uniform indexing).
    //!
    //! Generation uses the scalar f64-intermediate polynomial for platform-
    //! independent precision (see `generate_decode_lut` for rationale).
    use std::sync::OnceLock;
    static DECODE_LUT: OnceLock<Box<[f32; 65536]>> = OnceLock::new();
    /// Generate the decode LUT using the scalar f64-intermediate polynomial.
    ///
    /// Uses `rational_poly::srgb_to_linear_fast` (f64 Horner evaluation) instead
    /// of SIMD f32 dispatch to guarantee exact roundtrip with the scalar encode
    /// path (`linear_to_srgb_u16`). The f64 intermediate eliminates platform-
    /// dependent rounding differences — WASM SIMD128 lacks fused multiply-add,
    /// so its f32 `mul_add` (two roundings) can diverge from f64 by enough ULPs
    /// to break u16 roundtrip at boundary values.
    ///
    /// LUT generation is one-time (`OnceLock`), so the cost is negligible.
    #[doc(hidden)]
    pub fn generate_decode_lut() -> Box<[f32; 65536]> {
        let v: Vec<f32> = (0..65536u32)
            .map(|i| crate::rational_poly::srgb_to_linear_fast(
                i as f32 * (1.0 / 65535.0),
            ))
            .collect();
        v.into_boxed_slice().try_into().ok().unwrap()
    }
    /// Get the decode LUT, initializing on first call.
    #[inline]
    pub(crate) fn decode_lut() -> &'static [f32; 65536] {
        DECODE_LUT.get_or_init(generate_decode_lut)
    }
    static ENCODE_LUT: OnceLock<Box<[u16; 65537]>> = OnceLock::new();
    /// Number of entries in the sqrt-indexed encode LUT.
    pub(crate) const ENCODE_LUT_N: usize = 65537;
    /// Scale factor for sqrt index: `idx = (sqrt(linear) * ENCODE_SQRT_SCALE + 0.5) as usize`
    pub(crate) const ENCODE_SQRT_SCALE: f32 = (ENCODE_LUT_N - 1) as f32;
    /// Generate the sqrt-indexed encode LUT using the scalar f64-intermediate polynomial.
    ///
    /// Entry `i` stores the sRGB u16 value for `linear = (i / 65536)²`.
    /// Lookup uses `idx = (sqrt(linear) * 65536 + 0.5)`.
    ///
    /// Uses `rational_poly::linear_to_srgb_fast` for platform-independent precision
    /// (see `generate_decode_lut` for rationale).
    #[doc(hidden)]
    pub fn generate_encode_lut() -> Box<[u16; 65537]> {
        let lut: Vec<u16> = (0..ENCODE_LUT_N as u32)
            .map(|i| {
                let t = i as f32 * (1.0 / ENCODE_SQRT_SCALE);
                let linear = t * t;
                let srgb = crate::rational_poly::linear_to_srgb_fast(linear);
                (srgb * 65535.0 + 0.5) as u16
            })
            .collect();
        lut.into_boxed_slice().try_into().ok().unwrap()
    }
    /// Get the sqrt-indexed encode LUT, initializing on first call.
    #[inline]
    pub(crate) fn encode_lut() -> &'static [u16; 65537] {
        ENCODE_LUT.get_or_init(generate_encode_lut)
    }
}
#[doc(hidden)]
pub mod alt {
    //! Alternative/experimental implementations for benchmarking.
    //!
    //! This module contains various pow approximations and conversion methods
    //! for performance comparison. Not part of the stable API.
    //!
    //! Enable with the `alt` feature flag.
    #![allow(missing_docs, dead_code)]
    pub mod accuracy {
        //! Accuracy analysis and ULP comparison utilities.
        /// Compute ULP (units in last place) distance between two f32 values.
        ///
        /// Returns the number of representable floats between `a` and `b`.
        /// Sign-magnitude floats are handled correctly.
        #[inline]
        pub fn ulp_distance_f32(a: f32, b: f32) -> u32 {
            if a.is_nan() || b.is_nan() {
                return u32::MAX;
            }
            if a == b {
                return 0;
            }
            fn to_lexical(x: f32) -> i32 {
                let bits = x.to_bits() as i32;
                if bits < 0 { i32::MIN - bits } else { bits }
            }
            let a_lex = to_lexical(a);
            let b_lex = to_lexical(b);
            (a_lex.wrapping_sub(b_lex)).unsigned_abs()
        }
        /// Compute ULP distance between two f64 values.
        #[inline]
        pub fn ulp_distance_f64(a: f64, b: f64) -> u64 {
            if a.is_nan() || b.is_nan() {
                return u64::MAX;
            }
            if a == b {
                return 0;
            }
            fn to_lexical(x: f64) -> i64 {
                let bits = x.to_bits() as i64;
                if bits < 0 { i64::MIN - bits } else { bits }
            }
            let a_lex = to_lexical(a);
            let b_lex = to_lexical(b);
            (a_lex.wrapping_sub(b_lex)).unsigned_abs()
        }
        /// Naive sRGB to linear conversion using only std library.
        /// This serves as the "textbook" reference implementation.
        #[inline]
        pub fn naive_srgb_to_linear_f64(gamma: f64) -> f64 {
            if gamma <= 0.04045 {
                gamma / 12.92
            } else {
                ((gamma + 0.055) / 1.055).powf(2.4)
            }
        }
        /// Naive linear to sRGB conversion using only std library.
        #[inline]
        pub fn naive_linear_to_srgb_f64(linear: f64) -> f64 {
            if linear <= 0.0031308 {
                linear * 12.92
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            }
        }
        /// Naive sRGB to linear conversion (f32).
        #[inline]
        pub fn naive_srgb_to_linear_f32(gamma: f32) -> f32 {
            if gamma <= 0.04045 {
                gamma / 12.92
            } else {
                ((gamma + 0.055) / 1.055).powf(2.4)
            }
        }
        /// Naive linear to sRGB conversion (f32).
        #[inline]
        pub fn naive_linear_to_srgb_f32(linear: f32) -> f32 {
            if linear <= 0.0031308 {
                linear * 12.92
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            }
        }
        /// Statistics about ULP differences.
        pub struct UlpStats {
            pub max_ulp: u64,
            pub total_ulp: u128,
            pub count: u64,
            pub max_ulp_input: f64,
            pub max_ulp_expected: f64,
            pub max_ulp_actual: f64,
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for UlpStats {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "max_ulp",
                    "total_ulp",
                    "count",
                    "max_ulp_input",
                    "max_ulp_expected",
                    "max_ulp_actual",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.max_ulp,
                    &self.total_ulp,
                    &self.count,
                    &self.max_ulp_input,
                    &self.max_ulp_expected,
                    &&self.max_ulp_actual,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "UlpStats",
                    names,
                    values,
                )
            }
        }
        #[automatically_derived]
        impl ::core::clone::Clone for UlpStats {
            #[inline]
            fn clone(&self) -> UlpStats {
                UlpStats {
                    max_ulp: ::core::clone::Clone::clone(&self.max_ulp),
                    total_ulp: ::core::clone::Clone::clone(&self.total_ulp),
                    count: ::core::clone::Clone::clone(&self.count),
                    max_ulp_input: ::core::clone::Clone::clone(&self.max_ulp_input),
                    max_ulp_expected: ::core::clone::Clone::clone(
                        &self.max_ulp_expected,
                    ),
                    max_ulp_actual: ::core::clone::Clone::clone(&self.max_ulp_actual),
                }
            }
        }
        #[automatically_derived]
        impl ::core::default::Default for UlpStats {
            #[inline]
            fn default() -> UlpStats {
                UlpStats {
                    max_ulp: ::core::default::Default::default(),
                    total_ulp: ::core::default::Default::default(),
                    count: ::core::default::Default::default(),
                    max_ulp_input: ::core::default::Default::default(),
                    max_ulp_expected: ::core::default::Default::default(),
                    max_ulp_actual: ::core::default::Default::default(),
                }
            }
        }
        impl UlpStats {
            /// Compute average ULP difference.
            #[inline]
            pub fn avg_ulp(&self) -> f64 {
                if self.count == 0 {
                    0.0
                } else {
                    self.total_ulp as f64 / self.count as f64
                }
            }
        }
        /// Compare an f32 implementation against f64 reference across a range.
        pub fn compare_f32_to_f64_reference<F, G>(
            f32_impl: F,
            f64_reference: G,
            start: f32,
            end: f32,
            steps: u32,
        ) -> UlpStats
        where
            F: Fn(f32) -> f32,
            G: Fn(f64) -> f64,
        {
            let mut stats = UlpStats::default();
            for i in 0..=steps {
                let t = i as f32 / steps as f32;
                let input = start + t * (end - start);
                let f64_result = f64_reference(input as f64);
                let f32_result = f32_impl(input);
                let expected_f32 = f64_result as f32;
                let ulp = ulp_distance_f32(f32_result, expected_f32) as u64;
                stats.total_ulp += ulp as u128;
                stats.count += 1;
                if ulp > stats.max_ulp {
                    stats.max_ulp = ulp;
                    stats.max_ulp_input = input as f64;
                    stats.max_ulp_expected = f64_result;
                    stats.max_ulp_actual = f32_result as f64;
                }
            }
            stats
        }
    }
    pub mod imageflow {
        //! Imageflow's sRGB conversion algorithms for benchmarking comparison.
        //!
        //! This module contains conversions equivalent to imageflow_core/src/graphics/color.rs
        //! and math.rs, using safe Rust bit manipulation (`f32::to_bits`/`f32::from_bits`).
        //!
        //! # Differences from crate primary curves
        //!
        //! Imageflow uses the IEC 61966-2-1 textbook constants (threshold=0.04045, a=0.055),
        //! while this crate's `scalar` functions use C0-continuous constants
        //! (threshold≈0.03929, a≈0.05501) that eliminate a ~2.3e-9 discontinuity at the
        //! piecewise boundary.
        //!
        //! ## f32→u8 agreement
        //!
        //! At u8 precision the two pipelines never differ by more than ±1, and agree on
        //! 99.99% of all f32 inputs in \[0, 1\] (121,729 disagreements out of 1,065,353,217
        //! values). Disagreements concentrate in the low sRGB range near the threshold:
        //!
        //! | sRGB output range | disagreements per value |
        //! |-------------------|------------------------|
        //! | 10–21             | 3,300–7,200            |
        //! | 22–49             | 1,300–4,100            |
        //! | 50–98             | 560–1,120              |
        //! | 99–137            | 320–840                |
        //! | 138–187           | 130–325                |
        //! | 188–255           | 2–210                  |
        //!
        //! For u8 roundtrips (sRGB u8 → linear f32 → sRGB u8), the two pipelines produce
        //! identical results for all 256 input values.
        use std::sync::LazyLock;
        mod safe_impl {
            /// Fast approximate 2^p using bit manipulation.
            /// Safe version using f32::from_bits.
            #[inline]
            pub fn fastpow2(p: f32) -> f32 {
                let offset: f32 = if p < 0.0 { 1.0 } else { 0.0 };
                let clipp: f32 = if p < -126.0 { -126.0 } else { p };
                let z: f32 = clipp - (clipp as i32) as f32 + offset;
                let bits = ((1_i32 << 23) as f32
                    * (clipp + 121.274_055_f32 + 27.728_024_f32 / (4.842_525_5_f32 - z)
                        - 1.490_129_1_f32 * z)) as u32;
                f32::from_bits(bits)
            }
            /// Fast approximate log2(x) using bit manipulation.
            /// Safe version using f32::to_bits/from_bits.
            #[inline]
            pub fn fastlog2(x: f32) -> f32 {
                let vx_bits = x.to_bits();
                let mx_bits = (vx_bits & 0x007f_ffff) | 0x3f00_0000;
                let mx = f32::from_bits(mx_bits);
                let mut y: f32 = vx_bits as f32;
                y *= 1.192_092_9e-7_f32;
                y - 124.225_52_f32 - 1.498_030_3_f32 * mx
                    - 1.725_88_f32 / (0.352_088_72_f32 + mx)
            }
        }
        use safe_impl::{fastlog2, fastpow2};
        /// Fast approximate power function using bit manipulation.
        /// From imageflow_core/src/graphics/math.rs
        #[inline]
        pub fn fastpow(x: f32, p: f32) -> f32 {
            fastpow2(p * fastlog2(x))
        }
        /// sRGB to linear using imageflow's constants (standard 0.04045 threshold).
        #[inline]
        pub fn srgb_to_linear(s: f32) -> f32 {
            if s <= 0.04045f32 {
                s / 12.92f32
            } else {
                f32::powf((s + 0.055f32) / (1_f32 + 0.055f32), 2.4f32)
            }
        }
        /// Linear to sRGB using imageflow's formula with fastpow.
        /// Returns value in [0, 255] range (not normalized).
        #[inline]
        pub fn linear_to_srgb_raw(clr: f32) -> f32 {
            if clr <= 0.0031308f32 {
                12.92f32 * clr * 255.0f32
            } else {
                1.055f32 * 255.0f32 * fastpow(clr, 0.41666666f32) - 14.025f32
            }
        }
        /// Linear to sRGB normalized (0 to 1) output using fastpow.
        #[inline]
        pub fn linear_to_srgb(clr: f32) -> f32 {
            if clr <= 0.0031308f32 {
                12.92f32 * clr
            } else {
                1.055f32 * fastpow(clr, 0.41666666f32) - 0.055f32
            }
        }
        /// Clamp float to u8 using imageflow's method.
        #[inline]
        pub fn uchar_clamp_ff(clr: f32) -> u8 {
            let mut result: u16;
            result = (clr as f64 + 0.5f64) as i16 as u16;
            if result as i32 > 255_i32 {
                result = if clr < 0_i32 as f32 { 0_i32 } else { 255_i32 } as u16;
            }
            result as u8
        }
        /// Convert linear f32 to sRGB u8 using fastpow formula.
        #[inline]
        pub fn linear_to_srgb_u8_fastpow(linear: f32) -> u8 {
            uchar_clamp_ff(linear_to_srgb_raw(linear))
        }
        /// Fast linear→sRGB using precomputed 16K LUT.
        #[inline]
        pub fn linear_to_srgb_lut(linear: f32) -> u8 {
            let idx = (linear * 16383.0).clamp(0.0, 16383.0) as usize;
            LINEAR_TO_SRGB_LUT[idx]
        }
        /// Convert sRGB u8 to linear f32 using standard formula.
        #[inline]
        pub fn srgb_u8_to_linear(value: u8) -> f32 {
            srgb_to_linear(value as f32 / 255.0)
        }
        /// Pre-computed sRGB u8→linear f32 lookup table (256 entries).
        pub struct SrgbToLinearLut {
            table: [f32; 256],
        }
        impl SrgbToLinearLut {
            /// Create the LUT using imageflow's srgb_to_linear formula.
            pub fn new() -> Self {
                let mut table = [0.0f32; 256];
                for (i, entry) in table.iter_mut().enumerate() {
                    *entry = srgb_to_linear(i as f32 / 255.0);
                }
                Self { table }
            }
            /// Lookup using bounds-checked indexing.
            #[inline]
            pub fn lookup(&self, value: u8) -> f32 {
                self.table[value as usize]
            }
        }
        impl Default for SrgbToLinearLut {
            fn default() -> Self {
                Self::new()
            }
        }
        /// Generate the linear→sRGB LUT using imageflow's exact formula.
        /// Returns 16384 entries matching imageflow's static table.
        fn generate_linear_to_srgb_lut() -> [u8; 16384] {
            let mut lut = [0u8; 16384];
            for (i, entry) in lut.iter_mut().enumerate() {
                let linear = i as f32 / 16383.0;
                let srgb_raw = if linear <= 0.0031308f32 {
                    12.92f32 * linear * 255.0f32
                } else {
                    1.055f32 * 255.0f32 * linear.powf(0.41666666f32) - 14.025f32
                };
                *entry = uchar_clamp_ff(srgb_raw);
            }
            lut
        }
        /// Precomputed linear→sRGB LUT (16384 entries)
        /// Generated using imageflow's exact formula
        pub static LINEAR_TO_SRGB_LUT: LazyLock<[u8; 16384]> = LazyLock::new(
            generate_linear_to_srgb_lut,
        );
    }
}
