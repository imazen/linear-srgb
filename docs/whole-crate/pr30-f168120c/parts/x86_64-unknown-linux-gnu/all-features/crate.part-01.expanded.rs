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
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::srgb_to_linear_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_to_linear_v3(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::srgb_to_linear_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::srgb_to_linear_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::x4::srgb_to_linear_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_srgb_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_v3(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::linear_to_srgb_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_srgb_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_slice_v3(values: &mut [f32]) {
            crate::kernels::x4::linear_to_srgb_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::srgb_to_linear_extended_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_to_linear_extended_v3(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::srgb_to_linear_extended_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_srgb_extended_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_extended_v3(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::linear_to_srgb_extended_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::gamma_to_linear_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn gamma_to_linear_v3(values: [f32; 4], gamma: f32) -> [f32; 4] {
            crate::kernels::x4::gamma_to_linear_v3(
                <archmage::X64V3Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::gamma_to_linear_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn gamma_to_linear_slice_v3(values: &mut [f32], gamma: f32) {
            crate::kernels::x4::gamma_to_linear_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_gamma_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_gamma_v3(values: [f32; 4], gamma: f32) -> [f32; 4] {
            crate::kernels::x4::linear_to_gamma_v3(
                <archmage::X64V3Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_gamma_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_gamma_slice_v3(values: &mut [f32], gamma: f32) {
            crate::kernels::x4::linear_to_gamma_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::tf_srgb_to_linear_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_srgb_to_linear_v3(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::tf_srgb_to_linear_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::tf_srgb_to_linear_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_srgb_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::x4::tf_srgb_to_linear_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::tf_linear_to_srgb_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_linear_to_srgb_v3(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::tf_linear_to_srgb_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::tf_linear_to_srgb_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_linear_to_srgb_slice_v3(values: &mut [f32]) {
            crate::kernels::x4::tf_linear_to_srgb_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::bt709_to_linear_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn bt709_to_linear_v3(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::bt709_to_linear_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::bt709_to_linear_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn bt709_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::x4::bt709_to_linear_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_bt709_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_bt709_v3(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::linear_to_bt709_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_bt709_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_bt709_slice_v3(values: &mut [f32]) {
            crate::kernels::x4::linear_to_bt709_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::pq_to_linear_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn pq_to_linear_v3(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::pq_to_linear_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::pq_to_linear_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn pq_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::x4::pq_to_linear_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_pq_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_pq_v3(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::linear_to_pq_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_pq_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_pq_slice_v3(values: &mut [f32]) {
            crate::kernels::x4::linear_to_pq_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::hlg_to_linear_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn hlg_to_linear_v3(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::hlg_to_linear_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::hlg_to_linear_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn hlg_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::x4::hlg_to_linear_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_hlg_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_hlg_v3(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::x4::linear_to_hlg_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x4::linear_to_hlg_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_hlg_slice_v3(values: &mut [f32]) {
            crate::kernels::x4::linear_to_hlg_slice_v3(
                <archmage::X64V3Token>::from_context(),
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
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn srgb_to_linear_extended_slice_v3(values: &mut [f32]) {
            crate::kernels::map_slice::<
                4,
            >(values, |chunk| srgb_to_linear_extended_v3(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_srgb_extended_slice_v3(values: &mut [f32]) {
            crate::kernels::map_slice::<
                4,
            >(values, |chunk| linear_to_srgb_extended_v3(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 4 values; see the module calling-context requirements.
        pub(crate) fn srgb_to_linear_v4(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::float::x4::srgb_to_linear(
                archmage::X64V4Token::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn srgb_to_linear_slice_v4(values: &mut [f32]) {
            crate::kernels::map_slice::<4>(values, |chunk| srgb_to_linear_v4(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 4 values; see the module calling-context requirements.
        pub(crate) fn linear_to_srgb_v4(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::float::x4::linear_to_srgb(
                archmage::X64V4Token::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_srgb_slice_v4(values: &mut [f32]) {
            crate::kernels::map_slice::<4>(values, |chunk| linear_to_srgb_v4(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 4 values; see the module calling-context requirements.
        pub(crate) fn srgb_to_linear_extended_v4(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::float::x4::srgb_to_linear_extended(
                archmage::X64V4Token::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn srgb_to_linear_extended_slice_v4(values: &mut [f32]) {
            crate::kernels::map_slice::<
                4,
            >(values, |chunk| srgb_to_linear_extended_v4(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 4 values; see the module calling-context requirements.
        pub(crate) fn linear_to_srgb_extended_v4(values: [f32; 4]) -> [f32; 4] {
            crate::kernels::float::x4::linear_to_srgb_extended(
                archmage::X64V4Token::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_srgb_extended_slice_v4(values: &mut [f32]) {
            crate::kernels::map_slice::<
                4,
            >(values, |chunk| linear_to_srgb_extended_v4(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 4 values; see the module calling-context requirements.
        pub(crate) fn gamma_to_linear_v4(values: [f32; 4], gamma: f32) -> [f32; 4] {
            crate::kernels::float::x4::gamma_to_linear(
                archmage::X64V4Token::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn gamma_to_linear_slice_v4(values: &mut [f32], gamma: f32) {
            crate::kernels::map_slice::<
                4,
            >(values, |chunk| gamma_to_linear_v4(chunk, gamma));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 4 values; see the module calling-context requirements.
        pub(crate) fn linear_to_gamma_v4(values: [f32; 4], gamma: f32) -> [f32; 4] {
            crate::kernels::float::x4::linear_to_gamma(
                archmage::X64V4Token::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_gamma_slice_v4(values: &mut [f32], gamma: f32) {
            crate::kernels::map_slice::<
                4,
            >(values, |chunk| linear_to_gamma_v4(chunk, gamma));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 4 values; see the module calling-context requirements.
        pub(crate) fn tf_srgb_to_linear_v4(values: [f32; 4]) -> [f32; 4] {
            let token = archmage::X64V4Token::from_context();
            crate::tf::srgb::srgb_to_linear_x4(
                    token,
                    magetypes::simd::generic::f32x4::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn tf_srgb_to_linear_slice_v4(values: &mut [f32]) {
            crate::kernels::map_slice::<4>(values, |chunk| tf_srgb_to_linear_v4(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 4 values; see the module calling-context requirements.
        pub(crate) fn tf_linear_to_srgb_v4(values: [f32; 4]) -> [f32; 4] {
            let token = archmage::X64V4Token::from_context();
            crate::tf::srgb::linear_to_srgb_x4(
                    token,
                    magetypes::simd::generic::f32x4::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn tf_linear_to_srgb_slice_v4(values: &mut [f32]) {
            crate::kernels::map_slice::<4>(values, |chunk| tf_linear_to_srgb_v4(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 4 values; see the module calling-context requirements.
        pub(crate) fn bt709_to_linear_v4(values: [f32; 4]) -> [f32; 4] {
            let token = archmage::X64V4Token::from_context();
            crate::tf::bt709::bt709_to_linear_x4(
                    token,
                    magetypes::simd::generic::f32x4::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn bt709_to_linear_slice_v4(values: &mut [f32]) {
            crate::kernels::map_slice::<4>(values, |chunk| bt709_to_linear_v4(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 4 values; see the module calling-context requirements.
        pub(crate) fn linear_to_bt709_v4(values: [f32; 4]) -> [f32; 4] {
            let token = archmage::X64V4Token::from_context();
            crate::tf::bt709::linear_to_bt709_x4(
                    token,
                    magetypes::simd::generic::f32x4::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_bt709_slice_v4(values: &mut [f32]) {
            crate::kernels::map_slice::<4>(values, |chunk| linear_to_bt709_v4(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 4 values; see the module calling-context requirements.
        pub(crate) fn pq_to_linear_v4(values: [f32; 4]) -> [f32; 4] {
            let token = archmage::X64V4Token::from_context();
            crate::tf::pq::pq_to_linear_x4(
                    token,
                    magetypes::simd::generic::f32x4::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn pq_to_linear_slice_v4(values: &mut [f32]) {
            crate::kernels::map_slice::<4>(values, |chunk| pq_to_linear_v4(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 4 values; see the module calling-context requirements.
        pub(crate) fn linear_to_pq_v4(values: [f32; 4]) -> [f32; 4] {
            let token = archmage::X64V4Token::from_context();
            crate::tf::pq::linear_to_pq_x4(
                    token,
                    magetypes::simd::generic::f32x4::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_pq_slice_v4(values: &mut [f32]) {
            crate::kernels::map_slice::<4>(values, |chunk| linear_to_pq_v4(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 4 values; see the module calling-context requirements.
        pub(crate) fn hlg_to_linear_v4(values: [f32; 4]) -> [f32; 4] {
            let token = archmage::X64V4Token::from_context();
            crate::tf::hlg::hlg_to_linear_x4(
                    token,
                    magetypes::simd::generic::f32x4::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn hlg_to_linear_slice_v4(values: &mut [f32]) {
            crate::kernels::map_slice::<4>(values, |chunk| hlg_to_linear_v4(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 4 values; see the module calling-context requirements.
        pub(crate) fn linear_to_hlg_v4(values: [f32; 4]) -> [f32; 4] {
            let token = archmage::X64V4Token::from_context();
            crate::tf::hlg::linear_to_hlg_x4(
                    token,
                    magetypes::simd::generic::f32x4::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_hlg_slice_v4(values: &mut [f32]) {
            crate::kernels::map_slice::<4>(values, |chunk| linear_to_hlg_v4(chunk));
        }
    }
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
    pub mod x8 {
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::srgb_to_linear_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_to_linear_v3(values: [f32; 8]) -> [f32; 8] {
            crate::kernels::x8::srgb_to_linear_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::srgb_to_linear_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::x8::srgb_to_linear_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_srgb_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_v3(values: [f32; 8]) -> [f32; 8] {
            crate::kernels::x8::linear_to_srgb_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_srgb_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_slice_v3(values: &mut [f32]) {
            crate::kernels::x8::linear_to_srgb_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::srgb_to_linear_extended_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_to_linear_extended_v3(values: [f32; 8]) -> [f32; 8] {
            crate::kernels::x8::srgb_to_linear_extended_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::srgb_to_linear_extended_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_to_linear_extended_slice_v3(values: &mut [f32]) {
            crate::kernels::x8::srgb_to_linear_extended_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_srgb_extended_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_extended_v3(values: [f32; 8]) -> [f32; 8] {
            crate::kernels::x8::linear_to_srgb_extended_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_srgb_extended_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_extended_slice_v3(values: &mut [f32]) {
            crate::kernels::x8::linear_to_srgb_extended_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::gamma_to_linear_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn gamma_to_linear_v3(values: [f32; 8], gamma: f32) -> [f32; 8] {
            crate::kernels::x8::gamma_to_linear_v3(
                <archmage::X64V3Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::gamma_to_linear_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn gamma_to_linear_slice_v3(values: &mut [f32], gamma: f32) {
            crate::kernels::x8::gamma_to_linear_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_gamma_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_gamma_v3(values: [f32; 8], gamma: f32) -> [f32; 8] {
            crate::kernels::x8::linear_to_gamma_v3(
                <archmage::X64V3Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_gamma_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_gamma_slice_v3(values: &mut [f32], gamma: f32) {
            crate::kernels::x8::linear_to_gamma_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::srgb_u8_to_linear_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_u8_to_linear_v3(values: [u8; 8]) -> [f32; 8] {
            crate::kernels::x8::srgb_u8_to_linear_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_srgb_u8_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_u8_v3(values: [f32; 8]) -> [u8; 8] {
            crate::kernels::x8::linear_to_srgb_u8_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::srgb_u8_to_linear_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_u8_to_linear_slice_v3(input: &[u8], output: &mut [f32]) {
            crate::kernels::x8::srgb_u8_to_linear_slice_v3(
                <archmage::X64V3Token>::from_context(),
                input,
                output,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_srgb_u8_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_u8_slice_v3(input: &[f32], output: &mut [u8]) {
            crate::kernels::x8::linear_to_srgb_u8_slice_v3(
                <archmage::X64V3Token>::from_context(),
                input,
                output,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::srgb_u16_to_linear_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_u16_to_linear_v3(values: [u16; 8]) -> [f32; 8] {
            crate::kernels::x8::srgb_u16_to_linear_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_srgb_u16_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_u16_v3(values: [f32; 8]) -> [u16; 8] {
            crate::kernels::x8::linear_to_srgb_u16_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::tf_srgb_to_linear_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_srgb_to_linear_v3(values: [f32; 8]) -> [f32; 8] {
            crate::kernels::x8::tf_srgb_to_linear_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::tf_srgb_to_linear_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_srgb_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::x8::tf_srgb_to_linear_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::tf_linear_to_srgb_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_linear_to_srgb_v3(values: [f32; 8]) -> [f32; 8] {
            crate::kernels::x8::tf_linear_to_srgb_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::tf_linear_to_srgb_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_linear_to_srgb_slice_v3(values: &mut [f32]) {
            crate::kernels::x8::tf_linear_to_srgb_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::bt709_to_linear_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn bt709_to_linear_v3(values: [f32; 8]) -> [f32; 8] {
            crate::kernels::x8::bt709_to_linear_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::bt709_to_linear_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn bt709_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::x8::bt709_to_linear_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_bt709_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_bt709_v3(values: [f32; 8]) -> [f32; 8] {
            crate::kernels::x8::linear_to_bt709_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_bt709_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_bt709_slice_v3(values: &mut [f32]) {
            crate::kernels::x8::linear_to_bt709_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::pq_to_linear_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn pq_to_linear_v3(values: [f32; 8]) -> [f32; 8] {
            crate::kernels::x8::pq_to_linear_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::pq_to_linear_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn pq_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::x8::pq_to_linear_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_pq_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_pq_v3(values: [f32; 8]) -> [f32; 8] {
            crate::kernels::x8::linear_to_pq_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_pq_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_pq_slice_v3(values: &mut [f32]) {
            crate::kernels::x8::linear_to_pq_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::hlg_to_linear_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn hlg_to_linear_v3(values: [f32; 8]) -> [f32; 8] {
            crate::kernels::x8::hlg_to_linear_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::hlg_to_linear_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn hlg_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::x8::hlg_to_linear_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_hlg_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_hlg_v3(values: [f32; 8]) -> [f32; 8] {
            crate::kernels::x8::linear_to_hlg_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_hlg_slice_v3`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_hlg_slice_v3(values: &mut [f32]) {
            crate::kernels::x8::linear_to_hlg_slice_v3(
                <archmage::X64V3Token>::from_context(),
                values,
            )
        }
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::srgb_u16_to_linear_scalar`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
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
        pub fn srgb_u16_to_linear_scalar(values: [u16; 8]) -> [f32; 8] {
            crate::kernels::x8::srgb_u16_to_linear_scalar(
                <archmage::ScalarToken>::from_context(),
                values,
            )
        }
        ///CPU-feature entry point corresponding to [`crate::tokens::x8::linear_to_srgb_u16_scalar`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
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
        pub fn linear_to_srgb_u16_scalar(values: [f32; 8]) -> [u16; 8] {
            crate::kernels::x8::linear_to_srgb_u16_scalar(
                <archmage::ScalarToken>::from_context(),
                values,
            )
        }
    }
    /// Internal wide dispatch and compatibility entry points.
    pub(crate) mod x16 {
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::srgb_to_linear_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_to_linear_v4(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::x16::srgb_to_linear_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::srgb_to_linear_slice_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_to_linear_slice_v4(values: &mut [f32]) {
            crate::kernels::x16::srgb_to_linear_slice_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::linear_to_srgb_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_v4(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::x16::linear_to_srgb_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::linear_to_srgb_slice_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_slice_v4(values: &mut [f32]) {
            crate::kernels::x16::linear_to_srgb_slice_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::gamma_to_linear_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn gamma_to_linear_v4(values: [f32; 16], gamma: f32) -> [f32; 16] {
            crate::kernels::x16::gamma_to_linear_v4(
                <archmage::X64V4Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::gamma_to_linear_slice_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn gamma_to_linear_slice_v4(values: &mut [f32], gamma: f32) {
            crate::kernels::x16::gamma_to_linear_slice_v4(
                <archmage::X64V4Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::linear_to_gamma_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_gamma_v4(values: [f32; 16], gamma: f32) -> [f32; 16] {
            crate::kernels::x16::linear_to_gamma_v4(
                <archmage::X64V4Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::linear_to_gamma_slice_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_gamma_slice_v4(values: &mut [f32], gamma: f32) {
            crate::kernels::x16::linear_to_gamma_slice_v4(
                <archmage::X64V4Token>::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::srgb_u8_to_linear_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_u8_to_linear_v4(values: [u8; 16]) -> [f32; 16] {
            crate::kernels::x16::srgb_u8_to_linear_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::linear_to_srgb_u8_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_u8_v4(values: [f32; 16]) -> [u8; 16] {
            crate::kernels::x16::linear_to_srgb_u8_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::srgb_u8_to_linear_slice_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn srgb_u8_to_linear_slice_v4(input: &[u8], output: &mut [f32]) {
            crate::kernels::x16::srgb_u8_to_linear_slice_v4(
                <archmage::X64V4Token>::from_context(),
                input,
                output,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::linear_to_srgb_u8_slice_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_srgb_u8_slice_v4(input: &[f32], output: &mut [u8]) {
            crate::kernels::x16::linear_to_srgb_u8_slice_v4(
                <archmage::X64V4Token>::from_context(),
                input,
                output,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::tf_srgb_to_linear_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_srgb_to_linear_v4(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::x16::tf_srgb_to_linear_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::tf_srgb_to_linear_slice_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_srgb_to_linear_slice_v4(values: &mut [f32]) {
            crate::kernels::x16::tf_srgb_to_linear_slice_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::tf_linear_to_srgb_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_linear_to_srgb_v4(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::x16::tf_linear_to_srgb_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::tf_linear_to_srgb_slice_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn tf_linear_to_srgb_slice_v4(values: &mut [f32]) {
            crate::kernels::x16::tf_linear_to_srgb_slice_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::bt709_to_linear_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn bt709_to_linear_v4(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::x16::bt709_to_linear_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::bt709_to_linear_slice_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn bt709_to_linear_slice_v4(values: &mut [f32]) {
            crate::kernels::x16::bt709_to_linear_slice_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::linear_to_bt709_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_bt709_v4(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::x16::linear_to_bt709_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::linear_to_bt709_slice_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_bt709_slice_v4(values: &mut [f32]) {
            crate::kernels::x16::linear_to_bt709_slice_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::pq_to_linear_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn pq_to_linear_v4(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::x16::pq_to_linear_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::pq_to_linear_slice_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn pq_to_linear_slice_v4(values: &mut [f32]) {
            crate::kernels::x16::pq_to_linear_slice_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::linear_to_pq_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_pq_v4(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::x16::linear_to_pq_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::linear_to_pq_slice_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_pq_slice_v4(values: &mut [f32]) {
            crate::kernels::x16::linear_to_pq_slice_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::hlg_to_linear_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn hlg_to_linear_v4(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::x16::hlg_to_linear_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::hlg_to_linear_slice_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn hlg_to_linear_slice_v4(values: &mut [f32]) {
            crate::kernels::x16::hlg_to_linear_slice_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::linear_to_hlg_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_hlg_v4(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::x16::linear_to_hlg_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        ///CPU-feature entry point corresponding to [`crate::tokens::x16::linear_to_hlg_slice_v4`] without its token argument. See [calling context](crate::target_features#calling-context) for required CPU features.
        ///
        /// # Safety
        ///
        /// On native targets rustc permits a safe call only from a function
        /// enabling all required CPU features (see the module's feature table).
        /// Use a matching feature function; ordinary callers should use
        /// [`crate::default`]. No token argument or `unsafe` block is needed
        /// inside that matching context. WASM validates SIMD at module load;
        /// scalar variants have no CPU requirements.
        pub fn linear_to_hlg_slice_v4(values: &mut [f32]) {
            crate::kernels::x16::linear_to_hlg_slice_v4(
                <archmage::X64V4Token>::from_context(),
                values,
            )
        }
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
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn srgb_to_linear_v3(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::float::x16::srgb_to_linear(
                archmage::X64V3Token::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn srgb_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, |chunk| srgb_to_linear_v3(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn linear_to_srgb_v3(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::float::x16::linear_to_srgb(
                archmage::X64V3Token::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_srgb_slice_v3(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, |chunk| linear_to_srgb_v3(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn srgb_to_linear_extended_v3(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::float::x16::srgb_to_linear_extended(
                archmage::X64V3Token::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn srgb_to_linear_extended_slice_v3(values: &mut [f32]) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| srgb_to_linear_extended_v3(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn linear_to_srgb_extended_v3(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::float::x16::linear_to_srgb_extended(
                archmage::X64V3Token::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_srgb_extended_slice_v3(values: &mut [f32]) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| linear_to_srgb_extended_v3(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn gamma_to_linear_v3(values: [f32; 16], gamma: f32) -> [f32; 16] {
            crate::kernels::float::x16::gamma_to_linear(
                archmage::X64V3Token::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn gamma_to_linear_slice_v3(values: &mut [f32], gamma: f32) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| gamma_to_linear_v3(chunk, gamma));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn linear_to_gamma_v3(values: [f32; 16], gamma: f32) -> [f32; 16] {
            crate::kernels::float::x16::linear_to_gamma(
                archmage::X64V3Token::from_context(),
                values,
                gamma,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_gamma_slice_v3(values: &mut [f32], gamma: f32) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| linear_to_gamma_v3(chunk, gamma));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn tf_srgb_to_linear_v3(values: [f32; 16]) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::tf_srgb_to_linear_v3(*src);
                dst.copy_from_slice(&converted);
            }
            output
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn tf_srgb_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, |chunk| tf_srgb_to_linear_v3(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn tf_linear_to_srgb_v3(values: [f32; 16]) -> [f32; 16] {
            let mut output = [0.0; 16];
            for (src, dst) in values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(output.as_chunks_mut::<4>().0)
            {
                let converted = super::x4::tf_linear_to_srgb_v3(*src);
                dst.copy_from_slice(&converted);
            }
            output
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn tf_linear_to_srgb_slice_v3(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, |chunk| tf_linear_to_srgb_v3(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn bt709_to_linear_v3(values: [f32; 16]) -> [f32; 16] {
            let token = archmage::X64V3Token::from_context();
            crate::tf::bt709::bt709_to_linear_x16(
                    token,
                    magetypes::simd::generic::f32x16::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn bt709_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, |chunk| bt709_to_linear_v3(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn linear_to_bt709_v3(values: [f32; 16]) -> [f32; 16] {
            let token = archmage::X64V3Token::from_context();
            crate::tf::bt709::linear_to_bt709_x16(
                    token,
                    magetypes::simd::generic::f32x16::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_bt709_slice_v3(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, |chunk| linear_to_bt709_v3(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn pq_to_linear_v3(values: [f32; 16]) -> [f32; 16] {
            let token = archmage::X64V3Token::from_context();
            crate::tf::pq::pq_to_linear_x16(
                    token,
                    magetypes::simd::generic::f32x16::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn pq_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, |chunk| pq_to_linear_v3(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn linear_to_pq_v3(values: [f32; 16]) -> [f32; 16] {
            let token = archmage::X64V3Token::from_context();
            crate::tf::pq::linear_to_pq_x16(
                    token,
                    magetypes::simd::generic::f32x16::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_pq_slice_v3(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, |chunk| linear_to_pq_v3(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn hlg_to_linear_v3(values: [f32; 16]) -> [f32; 16] {
            let token = archmage::X64V3Token::from_context();
            crate::tf::hlg::hlg_to_linear_x16(
                    token,
                    magetypes::simd::generic::f32x16::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn hlg_to_linear_slice_v3(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, |chunk| hlg_to_linear_v3(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn linear_to_hlg_v3(values: [f32; 16]) -> [f32; 16] {
            let token = archmage::X64V3Token::from_context();
            crate::tf::hlg::linear_to_hlg_x16(
                    token,
                    magetypes::simd::generic::f32x16::from_array_t(token, values),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_hlg_slice_v3(values: &mut [f32]) {
            crate::kernels::map_slice::<16>(values, |chunk| linear_to_hlg_v3(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn srgb_to_linear_extended_v4(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::float::x16::srgb_to_linear_extended(
                archmage::X64V4Token::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn srgb_to_linear_extended_slice_v4(values: &mut [f32]) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| srgb_to_linear_extended_v4(chunk));
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 values; see the module calling-context requirements.
        pub(crate) fn linear_to_srgb_extended_v4(values: [f32; 16]) -> [f32; 16] {
            crate::kernels::float::x16::linear_to_srgb_extended(
                archmage::X64V4Token::from_context(),
                values,
            )
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert all values, including a final partial array.
        pub(crate) fn linear_to_srgb_extended_slice_v4(values: &mut [f32]) {
            crate::kernels::map_slice::<
                16,
            >(values, |chunk| linear_to_srgb_extended_v4(chunk));
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
        pub use archmage::X64V3Token;
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 4 sRGB values to linear. Input clamped to \[0, 1\].
        ///
        /// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
        /// Uses 128-bit SSE registers (VEX-encoded) within the AVX2+FMA context.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub fn srgb_to_linear_v3(_token: X64V3Token, srgb: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::srgb_to_linear_v3(srgb)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_srgb_v3(_token: X64V3Token, linear: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::linear_to_srgb_v3(linear)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 4 sRGB values to linear without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted to \[0, 8\]. sampled max 8 ULP in \[0,1\], u16-safe to ~4.2.
        fn __attune_srgb_to_linear_extended_v3(
            _token: ::archmage::X64V3Token,
            srgb: [f32; 4],
        ) -> [f32; 4] {
            crate::target_features::x4::srgb_to_linear_extended_v3(srgb)
        }
        /// Convert 4 sRGB values to linear without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted to \[0, 8\]. sampled max 8 ULP in \[0,1\], u16-safe to ~4.2.
        #[inline(always)]
        #[allow(clippy::inline_always)]
        pub fn srgb_to_linear_extended_v3(
            __archmage_token: ::archmage::X64V3Token,
            srgb: [f32; 4],
        ) -> [f32; 4] {
            unsafe { __attune_srgb_to_linear_extended_v3(__archmage_token, srgb) }
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
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 4 linear values to sRGB without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted on √x to \[0, 64\]. sampled max 7 ULP in \[0,1\], u16-safe to 64.
        fn __attune_linear_to_srgb_extended_v3(
            _token: ::archmage::X64V3Token,
            linear: [f32; 4],
        ) -> [f32; 4] {
            crate::target_features::x4::linear_to_srgb_extended_v3(linear)
        }
        /// Convert 4 linear values to sRGB without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted on √x to \[0, 64\]. sampled max 7 ULP in \[0,1\], u16-safe to 64.
        #[inline(always)]
        #[allow(clippy::inline_always)]
        pub fn linear_to_srgb_extended_v3(
            __archmage_token: ::archmage::X64V3Token,
            linear: [f32; 4],
        ) -> [f32; 4] {
            unsafe { __attune_linear_to_srgb_extended_v3(__archmage_token, linear) }
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
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn gamma_to_linear_v3(
            _token: X64V3Token,
            encoded: [f32; 4],
            gamma: f32,
        ) -> [f32; 4] {
            crate::target_features::x4::gamma_to_linear_v3(encoded, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_gamma_v3(
            _token: X64V3Token,
            linear: [f32; 4],
            gamma: f32,
        ) -> [f32; 4] {
            crate::target_features::x4::linear_to_gamma_v3(linear, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert sRGB f32 values to linear in-place using 4-wide SSE SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn srgb_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x4::srgb_to_linear_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to sRGB in-place using 4-wide SSE SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_srgb_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x4::linear_to_srgb_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert gamma-encoded f32 values to linear in-place using 4-wide SSE SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn gamma_to_linear_slice_v3(
            _token: X64V3Token,
            values: &mut [f32],
            gamma: f32,
        ) {
            crate::target_features::x4::gamma_to_linear_slice_v3(values, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to gamma-encoded in-place using 4-wide SSE SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_gamma_slice_v3(
            _token: X64V3Token,
            values: &mut [f32],
            gamma: f32,
        ) {
            crate::target_features::x4::linear_to_gamma_slice_v3(values, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn tf_srgb_to_linear_v3(_token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::tf_srgb_to_linear_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn tf_linear_to_srgb_v3(_token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::tf_linear_to_srgb_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn bt709_to_linear_v3(_token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::bt709_to_linear_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn linear_to_bt709_v3(_token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::linear_to_bt709_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn pq_to_linear_v3(_token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::pq_to_linear_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn linear_to_pq_v3(_token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::linear_to_pq_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn hlg_to_linear_v3(_token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::hlg_to_linear_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn linear_to_hlg_v3(_token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::target_features::x4::linear_to_hlg_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn tf_srgb_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x4::tf_srgb_to_linear_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn tf_linear_to_srgb_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x4::tf_linear_to_srgb_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn bt709_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x4::bt709_to_linear_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn linear_to_bt709_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x4::linear_to_bt709_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn pq_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x4::pq_to_linear_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn linear_to_pq_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x4::linear_to_pq_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn hlg_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x4::hlg_to_linear_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub fn linear_to_hlg_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x4::linear_to_hlg_slice_v3(values)
        }
    }
    pub mod x8 {
        //! Compatibility adapters for the deprecated 8-lane token API.
        //! Numerical implementation lives behind `crate::target_features::x8`.
        use archmage::attune;
        pub use archmage::X64V3Token;
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 sRGB values to linear. Input clamped to \[0, 1\].
        ///
        /// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub fn srgb_to_linear_v3(_token: X64V3Token, srgb: [f32; 8]) -> [f32; 8] {
            crate::target_features::x8::srgb_to_linear_v3(srgb)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear values to sRGB. Input clamped to \[0, 1\].
        ///
        /// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub fn linear_to_srgb_v3(_token: X64V3Token, linear: [f32; 8]) -> [f32; 8] {
            crate::target_features::x8::linear_to_srgb_v3(linear)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 sRGB values to linear without clamping (extended range).
        ///
        /// Uses abs+sign with a 6/6 rational polynomial fitted to \[0, 8\].
        /// Sampled max 8 ULP in \[0,1\], u16-safe to |encoded| ≤ 6.18, u8-safe to 8.0.
        /// Pure SIMD — no per-lane branching or scalar fallback.
        pub fn srgb_to_linear_extended_v3(
            _token: X64V3Token,
            srgb: [f32; 8],
        ) -> [f32; 8] {
            crate::target_features::x8::srgb_to_linear_extended_v3(srgb)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear values to sRGB without clamping (extended range).
        ///
        /// Uses abs+sign with a 6/6 rational polynomial fitted on √x to \[0, 64\].
        /// Sampled max 7 ULP in \[0,1\], u16-safe across the full \[0, 64\] domain.
        /// Pure SIMD — no per-lane branching or scalar fallback.
        pub fn linear_to_srgb_extended_v3(
            _token: X64V3Token,
            linear: [f32; 8],
        ) -> [f32; 8] {
            crate::target_features::x8::linear_to_srgb_extended_v3(linear)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 gamma-encoded values to linear. Input clamped to \[0, 1\].
        ///
        /// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub fn gamma_to_linear_v3(
            _token: X64V3Token,
            encoded: [f32; 8],
            gamma: f32,
        ) -> [f32; 8] {
            crate::target_features::x8::gamma_to_linear_v3(encoded, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear values to gamma-encoded. Input clamped to \[0, 1\].
        ///
        /// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub fn linear_to_gamma_v3(
            _token: X64V3Token,
            linear: [f32; 8],
            gamma: f32,
        ) -> [f32; 8] {
            crate::target_features::x8::linear_to_gamma_v3(linear, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 sRGB u8 values to linear f32 via 256-entry LUT.
        ///
        /// Pure table lookup — no math. The 1KB table fits in L1 cache.
        /// The token is accepted for API consistency; the operation itself
        /// is scalar lookups assembled into an array.
        ///
        /// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub fn srgb_u8_to_linear_v3(_token: X64V3Token, srgb: [u8; 8]) -> [f32; 8] {
            crate::target_features::x8::srgb_u8_to_linear_v3(srgb)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert sRGB u8 values to linear f32 using 8-wide LUT lookup.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn srgb_u8_to_linear_slice_v3(
            _token: X64V3Token,
            input: &[u8],
            output: &mut [f32],
        ) {
            crate::target_features::x8::srgb_u8_to_linear_slice_v3(input, output)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 sRGB u16 values to linear f32 via rational polynomial.
        ///
        /// Pure SIMD — no LUT, no cache footprint. Normalizes u16 to [0, 1] via
        /// `*(1.0/65535.0)` and applies the C0-continuous `S2L_P/Q` rational
        /// polynomial. Boundary values (0, 65535) map exactly to (0.0, 1.0).
        ///
        /// The `attune` proof-output family generates `_v3`, `_neon`, `_wasm128`
        /// variants from this single body; f32x8 is native on V3 and polyfills
        /// to 2×f32x4 on NEON / WASM128 (fully inlined). Call from inside an
        /// `#[arcane]` function for zero-overhead inlining.
        fn __attune_srgb_u16_to_linear_v3(
            _token: ::archmage::X64V3Token,
            srgb: [u16; 8],
        ) -> [f32; 8] {
            crate::target_features::x8::srgb_u16_to_linear_v3(srgb)
        }
        /// Convert 8 sRGB u16 values to linear f32 via rational polynomial.
        ///
        /// Pure SIMD — no LUT, no cache footprint. Normalizes u16 to [0, 1] via
        /// `*(1.0/65535.0)` and applies the C0-continuous `S2L_P/Q` rational
        /// polynomial. Boundary values (0, 65535) map exactly to (0.0, 1.0).
        ///
        /// The `attune` proof-output family generates `_v3`, `_neon`, `_wasm128`
        /// variants from this single body; f32x8 is native on V3 and polyfills
        /// to 2×f32x4 on NEON / WASM128 (fully inlined). Call from inside an
        /// `#[arcane]` function for zero-overhead inlining.
        #[inline(always)]
        #[allow(clippy::inline_always)]
        pub fn srgb_u16_to_linear_v3(
            __archmage_token: ::archmage::X64V3Token,
            srgb: [u16; 8],
        ) -> [f32; 8] {
            unsafe { __attune_srgb_u16_to_linear_v3(__archmage_token, srgb) }
        }
        #[inline]
        /// Convert 8 sRGB u16 values to linear f32 via rational polynomial.
        ///
        /// Pure SIMD — no LUT, no cache footprint. Normalizes u16 to [0, 1] via
        /// `*(1.0/65535.0)` and applies the C0-continuous `S2L_P/Q` rational
        /// polynomial. Boundary values (0, 65535) map exactly to (0.0, 1.0).
        ///
        /// The `attune` proof-output family generates `_v3`, `_neon`, `_wasm128`
        /// variants from this single body; f32x8 is native on V3 and polyfills
        /// to 2×f32x4 on NEON / WASM128 (fully inlined). Call from inside an
        /// `#[arcane]` function for zero-overhead inlining.
        fn __attune_srgb_u16_to_linear_scalar(
            _token: ::archmage::ScalarToken,
            srgb: [u16; 8],
        ) -> [f32; 8] {
            crate::target_features::x8::srgb_u16_to_linear_scalar(srgb)
        }
        /// Convert 8 sRGB u16 values to linear f32 via rational polynomial.
        ///
        /// Pure SIMD — no LUT, no cache footprint. Normalizes u16 to [0, 1] via
        /// `*(1.0/65535.0)` and applies the C0-continuous `S2L_P/Q` rational
        /// polynomial. Boundary values (0, 65535) map exactly to (0.0, 1.0).
        ///
        /// The `attune` proof-output family generates `_v3`, `_neon`, `_wasm128`
        /// variants from this single body; f32x8 is native on V3 and polyfills
        /// to 2×f32x4 on NEON / WASM128 (fully inlined). Call from inside an
        /// `#[arcane]` function for zero-overhead inlining.
        #[inline(always)]
        #[allow(clippy::inline_always)]
        pub fn srgb_u16_to_linear_scalar(
            __archmage_token: ::archmage::ScalarToken,
            srgb: [u16; 8],
        ) -> [f32; 8] {
            __attune_srgb_u16_to_linear_scalar(__archmage_token, srgb)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear f32 values to sRGB u8 via LUT. Input clamped to \[0, 1\].
        ///
        /// Uses a 4096-entry const LUT with bitmask indexing (`& 0xFFF`) for
        /// provably safe bounds. SIMD accelerates the clamp and scale; lookups
        /// are scalar from an L1-resident 4KB table.
        ///
        /// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub fn linear_to_srgb_u8_v3(_token: X64V3Token, linear: [f32; 8]) -> [u8; 8] {
            crate::target_features::x8::linear_to_srgb_u8_v3(linear)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear f32 values to sRGB u16 via rational polynomial.
        ///
        /// Pure SIMD — no LUT, no cache footprint. Clamps to [0, 1], applies the
        /// C0-continuous `L2S_P/Q` rational polynomial on √x, then scales by
        /// 65535 + 0.5 and truncates to u16. Perfect roundtrip with the decode
        /// polynomial. Inputs ≥ 1.0 map to 65535 bit-exact; inputs ≤ 0.0 map to 0.
        ///
        /// The `attune` proof-output family generates `_v3`, `_neon`, `_wasm128`
        /// variants from this single body; f32x8 is native on V3 and polyfills
        /// to 2×f32x4 on NEON / WASM128 (fully inlined). Call from inside an
        /// `#[arcane]` function for zero-overhead inlining.
        fn __attune_linear_to_srgb_u16_v3(
            _token: ::archmage::X64V3Token,
            linear: [f32; 8],
        ) -> [u16; 8] {
            crate::target_features::x8::linear_to_srgb_u16_v3(linear)
        }
        /// Convert 8 linear f32 values to sRGB u16 via rational polynomial.
        ///
        /// Pure SIMD — no LUT, no cache footprint. Clamps to [0, 1], applies the
        /// C0-continuous `L2S_P/Q` rational polynomial on √x, then scales by
        /// 65535 + 0.5 and truncates to u16. Perfect roundtrip with the decode
        /// polynomial. Inputs ≥ 1.0 map to 65535 bit-exact; inputs ≤ 0.0 map to 0.
        ///
        /// The `attune` proof-output family generates `_v3`, `_neon`, `_wasm128`
        /// variants from this single body; f32x8 is native on V3 and polyfills
        /// to 2×f32x4 on NEON / WASM128 (fully inlined). Call from inside an
        /// `#[arcane]` function for zero-overhead inlining.
        #[inline(always)]
        #[allow(clippy::inline_always)]
        pub fn linear_to_srgb_u16_v3(
            __archmage_token: ::archmage::X64V3Token,
            linear: [f32; 8],
        ) -> [u16; 8] {
            unsafe { __attune_linear_to_srgb_u16_v3(__archmage_token, linear) }
        }
        #[inline]
        /// Convert 8 linear f32 values to sRGB u16 via rational polynomial.
        ///
        /// Pure SIMD — no LUT, no cache footprint. Clamps to [0, 1], applies the
        /// C0-continuous `L2S_P/Q` rational polynomial on √x, then scales by
        /// 65535 + 0.5 and truncates to u16. Perfect roundtrip with the decode
        /// polynomial. Inputs ≥ 1.0 map to 65535 bit-exact; inputs ≤ 0.0 map to 0.
        ///
        /// The `attune` proof-output family generates `_v3`, `_neon`, `_wasm128`
        /// variants from this single body; f32x8 is native on V3 and polyfills
        /// to 2×f32x4 on NEON / WASM128 (fully inlined). Call from inside an
        /// `#[arcane]` function for zero-overhead inlining.
        fn __attune_linear_to_srgb_u16_scalar(
            _token: ::archmage::ScalarToken,
            linear: [f32; 8],
        ) -> [u16; 8] {
            crate::target_features::x8::linear_to_srgb_u16_scalar(linear)
        }
        /// Convert 8 linear f32 values to sRGB u16 via rational polynomial.
        ///
        /// Pure SIMD — no LUT, no cache footprint. Clamps to [0, 1], applies the
        /// C0-continuous `L2S_P/Q` rational polynomial on √x, then scales by
        /// 65535 + 0.5 and truncates to u16. Perfect roundtrip with the decode
        /// polynomial. Inputs ≥ 1.0 map to 65535 bit-exact; inputs ≤ 0.0 map to 0.
        ///
        /// The `attune` proof-output family generates `_v3`, `_neon`, `_wasm128`
        /// variants from this single body; f32x8 is native on V3 and polyfills
        /// to 2×f32x4 on NEON / WASM128 (fully inlined). Call from inside an
        /// `#[arcane]` function for zero-overhead inlining.
        #[inline(always)]
        #[allow(clippy::inline_always)]
        pub fn linear_to_srgb_u16_scalar(
            __archmage_token: ::archmage::ScalarToken,
            linear: [f32; 8],
        ) -> [u16; 8] {
            __attune_linear_to_srgb_u16_scalar(__archmage_token, linear)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert sRGB f32 values to linear in-place using 8-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn srgb_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x8::srgb_to_linear_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to sRGB in-place using 8-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_srgb_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x8::linear_to_srgb_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert sRGB f32 values to linear in-place (extended range, no clamping).
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn srgb_to_linear_extended_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x8::srgb_to_linear_extended_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to sRGB in-place (extended range, no clamping).
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_srgb_extended_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x8::linear_to_srgb_extended_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert gamma-encoded f32 values to linear in-place using 8-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn gamma_to_linear_slice_v3(
            _token: X64V3Token,
            values: &mut [f32],
            gamma: f32,
        ) {
            crate::target_features::x8::gamma_to_linear_slice_v3(values, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to gamma-encoded in-place using 8-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_gamma_slice_v3(
            _token: X64V3Token,
            values: &mut [f32],
            gamma: f32,
        ) {
            crate::target_features::x8::linear_to_gamma_slice_v3(values, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to sRGB u8 using 8-wide SIMD + LUT.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_srgb_u8_slice_v3(
            _token: X64V3Token,
            input: &[f32],
            output: &mut [u8],
        ) {
            crate::target_features::x8::linear_to_srgb_u8_slice_v3(input, output)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 sRGB values to linear (rational polynomial, no powf).
        pub fn tf_srgb_to_linear_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::target_features::x8::tf_srgb_to_linear_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear values to sRGB (rational polynomial, no powf).
        pub fn tf_linear_to_srgb_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::target_features::x8::tf_linear_to_srgb_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 BT.709 encoded values to linear.
        pub fn bt709_to_linear_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::target_features::x8::bt709_to_linear_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear values to BT.709 encoded.
        pub fn linear_to_bt709_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::target_features::x8::linear_to_bt709_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 PQ signal values to linear.
        pub fn pq_to_linear_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::target_features::x8::pq_to_linear_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear values to PQ signal.
        pub fn linear_to_pq_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::target_features::x8::linear_to_pq_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 HLG signal values to linear.
        pub fn hlg_to_linear_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::target_features::x8::hlg_to_linear_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear values to HLG signal.
        pub fn linear_to_hlg_v3(_token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::target_features::x8::linear_to_hlg_v3(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert sRGB f32 values to linear in-place, 8-wide (TF module version).
        pub fn tf_srgb_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x8::tf_srgb_to_linear_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to sRGB in-place, 8-wide (TF module version).
        pub fn tf_linear_to_srgb_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x8::tf_linear_to_srgb_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert BT.709 f32 values to linear in-place, 8-wide.
        pub fn bt709_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x8::bt709_to_linear_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to BT.709 in-place, 8-wide.
        pub fn linear_to_bt709_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x8::linear_to_bt709_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert PQ f32 values to linear in-place, 8-wide.
        pub fn pq_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x8::pq_to_linear_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to PQ in-place, 8-wide.
        pub fn linear_to_pq_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x8::linear_to_pq_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert HLG f32 values to linear in-place, 8-wide.
        pub fn hlg_to_linear_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x8::hlg_to_linear_slice_v3(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to HLG in-place, 8-wide.
        pub fn linear_to_hlg_slice_v3(_token: X64V3Token, values: &mut [f32]) {
            crate::target_features::x8::linear_to_hlg_slice_v3(values)
        }
    }
    #[deprecated(
        note = "the x16 token API is deprecated without a direct replacement; use crate::default for ordinary slice conversion"
    )]
    pub mod x16 {
        //! Deprecated 16-lane token API, retained without a direct public replacement.
        //! Use `crate::default` for ordinary dispatched slice conversion.
        //! Numerical implementation lives behind `crate::target_features::x16`.
        use archmage::attune;
        pub use archmage::X64V4Token;
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 sRGB values to linear. Input clamped to \[0, 1\].
        ///
        /// The `X64V4Token` parameter proves AVX-512 support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub fn srgb_to_linear_v4(_token: X64V4Token, srgb: [f32; 16]) -> [f32; 16] {
            crate::target_features::x16::srgb_to_linear_v4(srgb)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 linear values to sRGB. Input clamped to \[0, 1\].
        ///
        /// The `X64V4Token` parameter proves AVX-512 support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub fn linear_to_srgb_v4(_token: X64V4Token, linear: [f32; 16]) -> [f32; 16] {
            crate::target_features::x16::linear_to_srgb_v4(linear)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 gamma-encoded values to linear. Input clamped to \[0, 1\].
        ///
        /// The `X64V4Token` parameter proves AVX-512 support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub fn gamma_to_linear_v4(
            _token: X64V4Token,
            encoded: [f32; 16],
            gamma: f32,
        ) -> [f32; 16] {
            crate::target_features::x16::gamma_to_linear_v4(encoded, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 linear values to gamma-encoded. Input clamped to \[0, 1\].
        ///
        /// The `X64V4Token` parameter proves AVX-512 support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub fn linear_to_gamma_v4(
            _token: X64V4Token,
            linear: [f32; 16],
            gamma: f32,
        ) -> [f32; 16] {
            crate::target_features::x16::linear_to_gamma_v4(linear, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 sRGB u8 values to linear f32 via 256-entry LUT.
        ///
        /// Pure table lookup — no math. The 1KB table fits in L1 cache.
        ///
        /// The `X64V4Token` parameter proves AVX-512 support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub fn srgb_u8_to_linear_v4(_token: X64V4Token, srgb: [u8; 16]) -> [f32; 16] {
            crate::target_features::x16::srgb_u8_to_linear_v4(srgb)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert sRGB u8 values to linear f32 using 16-wide LUT lookup.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn srgb_u8_to_linear_slice_v4(
            _token: X64V4Token,
            input: &[u8],
            output: &mut [f32],
        ) {
            crate::target_features::x16::srgb_u8_to_linear_slice_v4(input, output)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 linear f32 values to sRGB u8 via LUT. Input clamped to \[0, 1\].
        ///
        /// Uses a 4096-entry const LUT with bitmask indexing (`& 0xFFF`) for
        /// provably safe bounds. SIMD accelerates the clamp and scale; lookups
        /// are scalar from an L1-resident 4KB table.
        ///
        /// The `X64V4Token` parameter proves AVX-512 support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub fn linear_to_srgb_u8_v4(_token: X64V4Token, linear: [f32; 16]) -> [u8; 16] {
            crate::target_features::x16::linear_to_srgb_u8_v4(linear)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert sRGB f32 values to linear in-place using 16-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn srgb_to_linear_slice_v4(_token: X64V4Token, values: &mut [f32]) {
            crate::target_features::x16::srgb_to_linear_slice_v4(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert linear f32 values to sRGB in-place using 16-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_srgb_slice_v4(_token: X64V4Token, values: &mut [f32]) {
            crate::target_features::x16::linear_to_srgb_slice_v4(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert gamma-encoded f32 values to linear in-place using 16-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn gamma_to_linear_slice_v4(
            _token: X64V4Token,
            values: &mut [f32],
            gamma: f32,
        ) {
            crate::target_features::x16::gamma_to_linear_slice_v4(values, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert linear f32 values to gamma-encoded in-place using 16-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_gamma_slice_v4(
            _token: X64V4Token,
            values: &mut [f32],
            gamma: f32,
        ) {
            crate::target_features::x16::linear_to_gamma_slice_v4(values, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert linear f32 values to sRGB u8 using 16-wide SIMD + LUT.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub fn linear_to_srgb_u8_slice_v4(
            _token: X64V4Token,
            input: &[f32],
            output: &mut [u8],
        ) {
            crate::target_features::x16::linear_to_srgb_u8_slice_v4(input, output)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 sRGB values to linear (rational polynomial, no powf).
        pub fn tf_srgb_to_linear_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
            crate::target_features::x16::tf_srgb_to_linear_v4(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 linear values to sRGB (rational polynomial, no powf).
        pub fn tf_linear_to_srgb_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
            crate::target_features::x16::tf_linear_to_srgb_v4(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 PQ signal values to linear.
        pub fn pq_to_linear_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
            crate::target_features::x16::pq_to_linear_v4(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 linear values to PQ signal.
        pub fn linear_to_pq_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
            crate::target_features::x16::linear_to_pq_v4(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 BT.709 encoded values to linear (2×x8 via tokens::x8).
        pub fn bt709_to_linear_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
            crate::target_features::x16::bt709_to_linear_v4(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 linear values to BT.709 encoded (2×x8 via tokens::x8).
        pub fn linear_to_bt709_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
            crate::target_features::x16::linear_to_bt709_v4(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 HLG signal values to linear (2×x8 via tokens::x8).
        pub fn hlg_to_linear_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
            crate::target_features::x16::hlg_to_linear_v4(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 linear values to HLG signal (2×x8 via tokens::x8).
        pub fn linear_to_hlg_v4(_token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
            crate::target_features::x16::linear_to_hlg_v4(v)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub fn tf_srgb_to_linear_slice_v4(_token: X64V4Token, values: &mut [f32]) {
            crate::target_features::x16::tf_srgb_to_linear_slice_v4(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub fn tf_linear_to_srgb_slice_v4(_token: X64V4Token, values: &mut [f32]) {
            crate::target_features::x16::tf_linear_to_srgb_slice_v4(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub fn bt709_to_linear_slice_v4(_token: X64V4Token, values: &mut [f32]) {
            crate::target_features::x16::bt709_to_linear_slice_v4(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub fn linear_to_bt709_slice_v4(_token: X64V4Token, values: &mut [f32]) {
            crate::target_features::x16::linear_to_bt709_slice_v4(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub fn pq_to_linear_slice_v4(_token: X64V4Token, values: &mut [f32]) {
            crate::target_features::x16::pq_to_linear_slice_v4(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub fn linear_to_pq_slice_v4(_token: X64V4Token, values: &mut [f32]) {
            crate::target_features::x16::linear_to_pq_slice_v4(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub fn hlg_to_linear_slice_v4(_token: X64V4Token, values: &mut [f32]) {
            crate::target_features::x16::hlg_to_linear_slice_v4(values)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub fn linear_to_hlg_slice_v4(_token: X64V4Token, values: &mut [f32]) {
            crate::target_features::x16::linear_to_hlg_slice_v4(values)
        }
    }
    pub use archmage::X64V3Token;
    pub use archmage::X64V4Token;
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
        pub(crate) mod x8 {
            use super::*;
            #[inline(always)]
            pub(crate) fn srgb_to_linear<T: F32x8Backend>(
                token: T,
                srgb: [f32; 8],
            ) -> [f32; 8] {
                use crate::rational_poly::{S2L_P, S2L_Q};
                let zero = f32x8::<T>::zero_t(token);
                let one = f32x8::<T>::splat_t(token, 1.0);
                let v = f32x8::<T>::from_array_t(token, srgb);
                let clamped = v.max(zero).min(one);
                let linear_result = clamped * f32x8::<T>::splat_t(token, LINEAR_SCALE);
                let x = clamped;
                let yp = f32x8::<T>::splat_t(token, S2L_P[4])
                    .mla(x, f32x8::<T>::splat_t(token, S2L_P[3]));
                let yp = yp.mla(x, f32x8::<T>::splat_t(token, S2L_P[2]));
                let yp = yp.mla(x, f32x8::<T>::splat_t(token, S2L_P[1]));
                let yp = yp.mla(x, f32x8::<T>::splat_t(token, S2L_P[0]));
                let yq = f32x8::<T>::splat_t(token, S2L_Q[4])
                    .mla(x, f32x8::<T>::splat_t(token, S2L_Q[3]));
                let yq = yq.mla(x, f32x8::<T>::splat_t(token, S2L_Q[2]));
                let yq = yq.mla(x, f32x8::<T>::splat_t(token, S2L_Q[1]));
                let yq = yq.mla(x, f32x8::<T>::splat_t(token, S2L_Q[0]));
                let power_result = (yp / yq).min(one);
                let mask = clamped
                    .simd_lt(f32x8::<T>::splat_t(token, SRGB_LINEAR_THRESHOLD));
                let result = f32x8::<T>::blend(mask, linear_result, power_result);
                let ge_one = v.simd_ge(one);
                f32x8::<T>::blend(ge_one, one, result).to_array()
            }
            #[inline(always)]
            pub(crate) fn linear_to_srgb<T: F32x8Backend>(
                token: T,
                linear: [f32; 8],
            ) -> [f32; 8] {
                use crate::rational_poly::{L2S_P, L2S_Q};
                let zero = f32x8::<T>::zero_t(token);
                let one = f32x8::<T>::splat_t(token, 1.0);
                let v = f32x8::<T>::from_array_t(token, linear);
                let clamped = v.max(zero).min(one);
                let linear_result = clamped * f32x8::<T>::splat_t(token, TWELVE_92);
                let x = clamped.sqrt();
                let yp = f32x8::<T>::splat_t(token, L2S_P[4])
                    .mla(x, f32x8::<T>::splat_t(token, L2S_P[3]));
                let yp = yp.mla(x, f32x8::<T>::splat_t(token, L2S_P[2]));
                let yp = yp.mla(x, f32x8::<T>::splat_t(token, L2S_P[1]));
                let yp = yp.mla(x, f32x8::<T>::splat_t(token, L2S_P[0]));
                let yq = f32x8::<T>::splat_t(token, L2S_Q[4])
                    .mla(x, f32x8::<T>::splat_t(token, L2S_Q[3]));
                let yq = yq.mla(x, f32x8::<T>::splat_t(token, L2S_Q[2]));
                let yq = yq.mla(x, f32x8::<T>::splat_t(token, L2S_Q[1]));
                let yq = yq.mla(x, f32x8::<T>::splat_t(token, L2S_Q[0]));
                let power_result = (yp / yq).min(one);
                let mask = clamped.simd_lt(f32x8::<T>::splat_t(token, LINEAR_THRESHOLD));
                let result = f32x8::<T>::blend(mask, linear_result, power_result);
                let ge_one = v.simd_ge(one);
                f32x8::<T>::blend(ge_one, one, result).to_array()
            }
            #[inline(always)]
            pub(crate) fn gamma_to_linear<T: F32x8Convert>(
                token: T,
                encoded: [f32; 8],
                gamma: f32,
            ) -> [f32; 8] {
                let zero = f32x8::<T>::zero_t(token);
                let one = f32x8::<T>::splat_t(token, 1.0);
                let v = f32x8::<T>::from_array_t(token, encoded).max(zero).min(one);
                crate::tf::fast_math::pow_x8(token, v, gamma).to_array()
            }
            #[inline(always)]
            pub(crate) fn linear_to_gamma<T: F32x8Convert>(
                token: T,
                linear: [f32; 8],
                gamma: f32,
            ) -> [f32; 8] {
                let zero = f32x8::<T>::zero_t(token);
                let one = f32x8::<T>::splat_t(token, 1.0);
                let v = f32x8::<T>::from_array_t(token, linear).max(zero).min(one);
                crate::tf::fast_math::pow_x8(token, v, 1.0 / gamma).to_array()
            }
            #[inline(always)]
            pub(crate) fn srgb_to_linear_extended<T: F32x8Backend>(
                token: T,
                srgb: [f32; 8],
            ) -> [f32; 8] {
                use crate::rational_poly::{EXT_S2L_P as P, EXT_S2L_Q as Q};
                let zero = f32x8::zero_t(token);
                let v = f32x8::from_array_t(token, srgb);
                let neg_mask = v.simd_lt(zero);
                let abs_v = v.abs();
                let linear_result = abs_v * f32x8::splat_t(token, LINEAR_SCALE);
                let x = abs_v;
                let yp = f32x8::splat_t(token, P[6]).mla(x, f32x8::splat_t(token, P[5]));
                let yp = yp.mla(x, f32x8::splat_t(token, P[4]));
                let yp = yp.mla(x, f32x8::splat_t(token, P[3]));
                let yp = yp.mla(x, f32x8::splat_t(token, P[2]));
                let yp = yp.mla(x, f32x8::splat_t(token, P[1]));
                let yp = yp.mla(x, f32x8::splat_t(token, P[0]));
                let yq = f32x8::splat_t(token, Q[6]).mla(x, f32x8::splat_t(token, Q[5]));
                let yq = yq.mla(x, f32x8::splat_t(token, Q[4]));
                let yq = yq.mla(x, f32x8::splat_t(token, Q[3]));
                let yq = yq.mla(x, f32x8::splat_t(token, Q[2]));
                let yq = yq.mla(x, f32x8::splat_t(token, Q[1]));
                let yq = yq.mla(x, f32x8::splat_t(token, Q[0]));
                let power_result = yp / yq;
                let thresh_mask = abs_v
                    .simd_lt(f32x8::splat_t(token, SRGB_LINEAR_THRESHOLD));
                let result = f32x8::blend(thresh_mask, linear_result, power_result);
                let result = f32x8::blend(neg_mask, -result, result);
                result.to_array()
            }
            #[inline(always)]
            pub(crate) fn linear_to_srgb_extended<T: F32x8Backend>(
                token: T,
                linear: [f32; 8],
            ) -> [f32; 8] {
                use crate::rational_poly::{EXT_L2S_P as P, EXT_L2S_Q as Q};
                let zero = f32x8::zero_t(token);
                let v = f32x8::from_array_t(token, linear);
                let neg_mask = v.simd_lt(zero);
                let abs_v = v.abs();
                let linear_result = abs_v * f32x8::splat_t(token, TWELVE_92);
                let x = abs_v.sqrt();
                let yp = f32x8::splat_t(token, P[6]).mla(x, f32x8::splat_t(token, P[5]));
                let yp = yp.mla(x, f32x8::splat_t(token, P[4]));
                let yp = yp.mla(x, f32x8::splat_t(token, P[3]));
                let yp = yp.mla(x, f32x8::splat_t(token, P[2]));
                let yp = yp.mla(x, f32x8::splat_t(token, P[1]));
                let yp = yp.mla(x, f32x8::splat_t(token, P[0]));
                let yq = f32x8::splat_t(token, Q[6]).mla(x, f32x8::splat_t(token, Q[5]));
                let yq = yq.mla(x, f32x8::splat_t(token, Q[4]));
                let yq = yq.mla(x, f32x8::splat_t(token, Q[3]));
                let yq = yq.mla(x, f32x8::splat_t(token, Q[2]));
                let yq = yq.mla(x, f32x8::splat_t(token, Q[1]));
                let yq = yq.mla(x, f32x8::splat_t(token, Q[0]));
                let power_result = yp / yq;
                let thresh_mask = abs_v.simd_lt(f32x8::splat_t(token, LINEAR_THRESHOLD));
                let result = f32x8::blend(thresh_mask, linear_result, power_result);
                let result = f32x8::blend(neg_mask, -result, result);
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
    pub(crate) mod x16 {
        //! Private 16-lane kernels used by the canonical CPU-feature entry points.
        use crate::tf::fast_math::Mla;
        use archmage::attune;
        use archmage::X64V4Token;
        use magetypes::simd::v4::f32x16 as mt_f32x16;
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 sRGB values to linear. Input clamped to \[0, 1\].
        ///
        /// The `X64V4Token` parameter proves AVX-512 support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn srgb_to_linear_v4(
            token: X64V4Token,
            srgb: [f32; 16],
        ) -> [f32; 16] {
            super::float::x16::srgb_to_linear(token, srgb)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 linear values to sRGB. Input clamped to \[0, 1\].
        ///
        /// The `X64V4Token` parameter proves AVX-512 support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn linear_to_srgb_v4(
            token: X64V4Token,
            linear: [f32; 16],
        ) -> [f32; 16] {
            super::float::x16::linear_to_srgb(token, linear)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 gamma-encoded values to linear. Input clamped to \[0, 1\].
        ///
        /// The `X64V4Token` parameter proves AVX-512 support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn gamma_to_linear_v4(
            token: X64V4Token,
            encoded: [f32; 16],
            gamma: f32,
        ) -> [f32; 16] {
            super::float::x16::gamma_to_linear(token, encoded, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 linear values to gamma-encoded. Input clamped to \[0, 1\].
        ///
        /// The `X64V4Token` parameter proves AVX-512 support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn linear_to_gamma_v4(
            token: X64V4Token,
            linear: [f32; 16],
            gamma: f32,
        ) -> [f32; 16] {
            super::float::x16::linear_to_gamma(token, linear, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 sRGB u8 values to linear f32 via 256-entry LUT.
        ///
        /// Pure table lookup — no math. The 1KB table fits in L1 cache.
        ///
        /// The `X64V4Token` parameter proves AVX-512 support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn srgb_u8_to_linear_v4(
            _token: X64V4Token,
            srgb: [u8; 16],
        ) -> [f32; 16] {
            let lut = crate::const_luts::linear_table_8();
            [
                lut[srgb[0] as usize],
                lut[srgb[1] as usize],
                lut[srgb[2] as usize],
                lut[srgb[3] as usize],
                lut[srgb[4] as usize],
                lut[srgb[5] as usize],
                lut[srgb[6] as usize],
                lut[srgb[7] as usize],
                lut[srgb[8] as usize],
                lut[srgb[9] as usize],
                lut[srgb[10] as usize],
                lut[srgb[11] as usize],
                lut[srgb[12] as usize],
                lut[srgb[13] as usize],
                lut[srgb[14] as usize],
                lut[srgb[15] as usize],
            ]
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert sRGB u8 values to linear f32 using 16-wide LUT lookup.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn srgb_u8_to_linear_slice_v4(
            _token: X64V4Token,
            input: &[u8],
            output: &mut [f32],
        ) {
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
            let lut = crate::const_luts::linear_table_8();
            let (in_chunks, in_remainder) = input.as_chunks::<16>();
            let (out_chunks, out_remainder) = output.as_chunks_mut::<16>();
            for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
                *out = srgb_u8_to_linear_v4(_token, *inp);
            }
            for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
                *out = lut[*inp as usize];
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 linear f32 values to sRGB u8 via LUT. Input clamped to \[0, 1\].
        ///
        /// Uses a 4096-entry const LUT with bitmask indexing (`& 0xFFF`) for
        /// provably safe bounds. SIMD accelerates the clamp and scale; lookups
        /// are scalar from an L1-resident 4KB table.
        ///
        /// The `X64V4Token` parameter proves AVX-512 support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn linear_to_srgb_u8_v4(
            token: X64V4Token,
            linear: [f32; 16],
        ) -> [u8; 16] {
            let zero = mt_f32x16::zero_t(token);
            let one = mt_f32x16::splat_t(token, 1.0);
            let linear = mt_f32x16::from_array_t(token, linear).max(zero).min(one);
            let scaled = linear * mt_f32x16::splat_t(token, 4095.0)
                + mt_f32x16::splat_t(token, 0.5);
            let arr = scaled.to_array();
            let lut = crate::const_luts::linear_to_srgb_u8();
            [
                lut[arr[0] as usize & 0xFFF],
                lut[arr[1] as usize & 0xFFF],
                lut[arr[2] as usize & 0xFFF],
                lut[arr[3] as usize & 0xFFF],
                lut[arr[4] as usize & 0xFFF],
                lut[arr[5] as usize & 0xFFF],
                lut[arr[6] as usize & 0xFFF],
                lut[arr[7] as usize & 0xFFF],
                lut[arr[8] as usize & 0xFFF],
                lut[arr[9] as usize & 0xFFF],
                lut[arr[10] as usize & 0xFFF],
                lut[arr[11] as usize & 0xFFF],
                lut[arr[12] as usize & 0xFFF],
                lut[arr[13] as usize & 0xFFF],
                lut[arr[14] as usize & 0xFFF],
                lut[arr[15] as usize & 0xFFF],
            ]
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert sRGB f32 values to linear in-place using 16-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn srgb_to_linear_slice_v4(token: X64V4Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                *chunk = srgb_to_linear_v4(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 16];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = srgb_to_linear_v4(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert linear f32 values to sRGB in-place using 16-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_srgb_slice_v4(token: X64V4Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                *chunk = linear_to_srgb_v4(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 16];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_srgb_v4(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert gamma-encoded f32 values to linear in-place using 16-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn gamma_to_linear_slice_v4(
            token: X64V4Token,
            values: &mut [f32],
            gamma: f32,
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                *chunk = gamma_to_linear_v4(token, *chunk, gamma);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 16];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = gamma_to_linear_v4(token, padded, gamma);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert linear f32 values to gamma-encoded in-place using 16-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_gamma_slice_v4(
            token: X64V4Token,
            values: &mut [f32],
            gamma: f32,
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                *chunk = linear_to_gamma_v4(token, *chunk, gamma);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 16];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_gamma_v4(token, padded, gamma);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert linear f32 values to sRGB u8 using 16-wide SIMD + LUT.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_srgb_u8_slice_v4(
            token: X64V4Token,
            input: &[f32],
            output: &mut [u8],
        ) {
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
            let (in_chunks, in_remainder) = input.as_chunks::<16>();
            let (out_chunks, out_remainder) = output.as_chunks_mut::<16>();
            for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
                *out = linear_to_srgb_u8_v4(token, *inp);
            }
            for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
                *out = crate::scalar::linear_to_srgb_u8(*inp);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 sRGB values to linear (rational polynomial, no powf).
        pub(crate) fn tf_srgb_to_linear_v4(
            token: X64V4Token,
            v: [f32; 16],
        ) -> [f32; 16] {
            use crate::rational_poly::{LINEAR_SCALE, S2L_P, S2L_Q, SRGB_THRESHOLD};
            let v = mt_f32x16::from_array_t(token, v);
            let one = mt_f32x16::splat_t(token, 1.0);
            let threshold = mt_f32x16::splat_t(token, SRGB_THRESHOLD);
            let inv_12_92 = mt_f32x16::splat_t(token, LINEAR_SCALE);
            let linear = v * inv_12_92;
            let poly = eval_rational_poly_x16(token, v, S2L_P, S2L_Q).min(one);
            let mask = v.simd_le(threshold);
            mt_f32x16::blend(mask, linear, poly).to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 linear values to sRGB (rational polynomial, no powf).
        pub(crate) fn tf_linear_to_srgb_v4(
            token: X64V4Token,
            v: [f32; 16],
        ) -> [f32; 16] {
            use crate::rational_poly::{L2S_P, L2S_Q, LINEAR_THRESHOLD, TWELVE_92};
            let v = mt_f32x16::from_array_t(token, v);
            let one = mt_f32x16::splat_t(token, 1.0);
            let threshold = mt_f32x16::splat_t(token, LINEAR_THRESHOLD);
            let scale = mt_f32x16::splat_t(token, TWELVE_92);
            let linear = v * scale;
            let s = v.sqrt();
            let poly = eval_rational_poly_x16(token, s, L2S_P, L2S_Q).min(one);
            let mask = v.simd_le(threshold);
            mt_f32x16::blend(mask, linear, poly).to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 PQ signal values to linear.
        pub(crate) fn pq_to_linear_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
            crate::tf::pq::pq_to_linear_x16(token, mt_f32x16::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 linear values to PQ signal.
        pub(crate) fn linear_to_pq_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
            crate::tf::pq::linear_to_pq_x16(token, mt_f32x16::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 BT.709 encoded values to linear (2×x8 via tokens::x8).
        pub(crate) fn bt709_to_linear_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
            crate::tf::bt709::bt709_to_linear_x16(
                    token,
                    mt_f32x16::from_array_t(token, v),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 linear values to BT.709 encoded (2×x8 via tokens::x8).
        pub(crate) fn linear_to_bt709_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
            crate::tf::bt709::linear_to_bt709_x16(
                    token,
                    mt_f32x16::from_array_t(token, v),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 HLG signal values to linear (2×x8 via tokens::x8).
        pub(crate) fn hlg_to_linear_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
            crate::tf::hlg::hlg_to_linear_x16(token, mt_f32x16::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Convert 16 linear values to HLG signal (2×x8 via tokens::x8).
        pub(crate) fn linear_to_hlg_v4(token: X64V4Token, v: [f32; 16]) -> [f32; 16] {
            crate::tf::hlg::linear_to_hlg_x16(token, mt_f32x16::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub(crate) fn tf_srgb_to_linear_slice_v4(token: X64V4Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                *chunk = tf_srgb_to_linear_v4(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 16];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = tf_srgb_to_linear_v4(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub(crate) fn tf_linear_to_srgb_slice_v4(token: X64V4Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                *chunk = tf_linear_to_srgb_v4(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 16];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = tf_linear_to_srgb_v4(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub(crate) fn bt709_to_linear_slice_v4(token: X64V4Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                *chunk = bt709_to_linear_v4(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 16];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = bt709_to_linear_v4(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub(crate) fn linear_to_bt709_slice_v4(token: X64V4Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                *chunk = linear_to_bt709_v4(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 16];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_bt709_v4(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub(crate) fn pq_to_linear_slice_v4(token: X64V4Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                *chunk = pq_to_linear_v4(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 16];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = pq_to_linear_v4(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub(crate) fn linear_to_pq_slice_v4(token: X64V4Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                *chunk = linear_to_pq_v4(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 16];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_pq_v4(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub(crate) fn hlg_to_linear_slice_v4(token: X64V4Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                *chunk = hlg_to_linear_v4(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 16];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = hlg_to_linear_v4(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        /// Apply transfer function to a slice using AVX-512.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V4Token`.
        pub(crate) fn linear_to_hlg_slice_v4(token: X64V4Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                *chunk = linear_to_hlg_v4(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 16];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_hlg_v4(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[inline(always)]
        fn eval_rational_poly_x16(
            t: X64V4Token,
            x: magetypes::simd::v4::f32x16,
            p: [f32; 5],
            q: [f32; 5],
        ) -> magetypes::simd::v4::f32x16 {
            let mut yp = mt_f32x16::splat_t(t, p[4]);
            yp = yp.mla(x, mt_f32x16::splat_t(t, p[3]));
            yp = yp.mla(x, mt_f32x16::splat_t(t, p[2]));
            yp = yp.mla(x, mt_f32x16::splat_t(t, p[1]));
            yp = yp.mla(x, mt_f32x16::splat_t(t, p[0]));
            let mut yq = mt_f32x16::splat_t(t, q[4]);
            yq = yq.mla(x, mt_f32x16::splat_t(t, q[3]));
            yq = yq.mla(x, mt_f32x16::splat_t(t, q[2]));
            yq = yq.mla(x, mt_f32x16::splat_t(t, q[1]));
            yq = yq.mla(x, mt_f32x16::splat_t(t, q[0]));
            yp / yq
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
        use archmage::X64V3Token;
        use magetypes::simd::generic::f32x4 as gen_f32x4;
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 4 sRGB values to linear. Input clamped to \[0, 1\].
        ///
        /// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
        /// Uses 128-bit SSE registers (VEX-encoded) within the AVX2+FMA context.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn srgb_to_linear_v3(token: X64V3Token, srgb: [f32; 4]) -> [f32; 4] {
            srgb_to_linear_core(token, srgb)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 4 linear values to sRGB. Input clamped to \[0, 1\].
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_srgb_v3(
            token: X64V3Token,
            linear: [f32; 4],
        ) -> [f32; 4] {
            linear_to_srgb_core(token, linear)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 4 sRGB values to linear without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted to \[0, 8\]. sampled max 8 ULP in \[0,1\], u16-safe to ~4.2.
        pub(crate) fn srgb_to_linear_extended_v3(
            token: ::archmage::X64V3Token,
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
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 4 linear values to sRGB without clamping (extended range).
        ///
        /// 6/6 rational polynomial fitted on √x to \[0, 64\]. sampled max 7 ULP in \[0,1\], u16-safe to 64.
        pub(crate) fn linear_to_srgb_extended_v3(
            token: ::archmage::X64V3Token,
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
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 4 gamma-encoded values to linear. Input clamped to \[0, 1\].
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn gamma_to_linear_v3(
            token: X64V3Token,
            encoded: [f32; 4],
            gamma: f32,
        ) -> [f32; 4] {
            gamma_to_linear_core(token, encoded, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 4 linear values to gamma-encoded. Input clamped to \[0, 1\].
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_gamma_v3(
            token: X64V3Token,
            linear: [f32; 4],
            gamma: f32,
        ) -> [f32; 4] {
            linear_to_gamma_core(token, linear, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert sRGB f32 values to linear in-place using 4-wide SSE SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn srgb_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
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
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to sRGB in-place using 4-wide SSE SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_srgb_slice_v3(token: X64V3Token, values: &mut [f32]) {
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
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert gamma-encoded f32 values to linear in-place using 4-wide SSE SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn gamma_to_linear_slice_v3(
            token: X64V3Token,
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
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to gamma-encoded in-place using 4-wide SSE SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_gamma_slice_v3(
            token: X64V3Token,
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
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn tf_srgb_to_linear_v3(token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::tf::srgb::srgb_to_linear_x4(token, gen_f32x4::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn tf_linear_to_srgb_v3(token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::tf::srgb::linear_to_srgb_x4(token, gen_f32x4::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn bt709_to_linear_v3(token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::tf::bt709::bt709_to_linear_x4(
                    token,
                    gen_f32x4::from_array_t(token, v),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn linear_to_bt709_v3(token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::tf::bt709::linear_to_bt709_x4(
                    token,
                    gen_f32x4::from_array_t(token, v),
                )
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn pq_to_linear_v3(token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::tf::pq::pq_to_linear_x4(token, gen_f32x4::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn linear_to_pq_v3(token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::tf::pq::linear_to_pq_x4(token, gen_f32x4::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn hlg_to_linear_v3(token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::tf::hlg::hlg_to_linear_x4(token, gen_f32x4::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn linear_to_hlg_v3(token: X64V3Token, v: [f32; 4]) -> [f32; 4] {
            crate::tf::hlg::linear_to_hlg_x4(token, gen_f32x4::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn tf_srgb_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = tf_srgb_to_linear_v3(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = tf_srgb_to_linear_v3(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn tf_linear_to_srgb_slice_v3(token: X64V3Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = tf_linear_to_srgb_v3(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = tf_linear_to_srgb_v3(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn bt709_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = bt709_to_linear_v3(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = bt709_to_linear_v3(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn linear_to_bt709_slice_v3(token: X64V3Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = linear_to_bt709_v3(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_bt709_v3(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn pq_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = pq_to_linear_v3(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = pq_to_linear_v3(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn linear_to_pq_slice_v3(token: X64V3Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = linear_to_pq_v3(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_pq_v3(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn hlg_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = hlg_to_linear_v3(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = hlg_to_linear_v3(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Transfer function slice rite (4×f32, x86-64 V3). Requires `transfer` feature.
        ///
        /// # Safety
        ///
        /// Call from an `#[arcane]` context with a valid `X64V3Token`.
        pub(crate) fn linear_to_hlg_slice_v3(token: X64V3Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = linear_to_hlg_v3(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 4];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_hlg_v3(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
    }
    pub(crate) mod x8 {
        //! Private 8-lane kernels used by the canonical CPU-feature entry points.
        use crate::tf::fast_math::Mla;
        use archmage::attune;
        use archmage::X64V3Token;
        use magetypes::simd::f32x8 as mt_f32x8;
        use magetypes::simd::generic::f32x8 as gen_f32x8;
        const SRGB_LINEAR_THRESHOLD: f32 = 0.039_293_37;
        const LINEAR_THRESHOLD: f32 = 0.003_041_282_6;
        const LINEAR_SCALE: f32 = 1.0 / 12.92;
        const TWELVE_92: f32 = 12.92;
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 sRGB values to linear. Input clamped to \[0, 1\].
        ///
        /// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn srgb_to_linear_v3(token: X64V3Token, srgb: [f32; 8]) -> [f32; 8] {
            super::float::x8::srgb_to_linear(token, srgb)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear values to sRGB. Input clamped to \[0, 1\].
        ///
        /// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn linear_to_srgb_v3(
            token: X64V3Token,
            linear: [f32; 8],
        ) -> [f32; 8] {
            super::float::x8::linear_to_srgb(token, linear)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 sRGB values to linear without clamping (extended range).
        ///
        /// Uses abs+sign with a 6/6 rational polynomial fitted to \[0, 8\].
        /// Sampled max 8 ULP in \[0,1\], u16-safe to |encoded| ≤ 6.18, u8-safe to 8.0.
        /// Pure SIMD — no per-lane branching or scalar fallback.
        pub(crate) fn srgb_to_linear_extended_v3(
            token: X64V3Token,
            srgb: [f32; 8],
        ) -> [f32; 8] {
            super::float::x8::srgb_to_linear_extended(token, srgb)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear values to sRGB without clamping (extended range).
        ///
        /// Uses abs+sign with a 6/6 rational polynomial fitted on √x to \[0, 64\].
        /// Sampled max 7 ULP in \[0,1\], u16-safe across the full \[0, 64\] domain.
        /// Pure SIMD — no per-lane branching or scalar fallback.
        pub(crate) fn linear_to_srgb_extended_v3(
            token: X64V3Token,
            linear: [f32; 8],
        ) -> [f32; 8] {
            super::float::x8::linear_to_srgb_extended(token, linear)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 gamma-encoded values to linear. Input clamped to \[0, 1\].
        ///
        /// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn gamma_to_linear_v3(
            token: X64V3Token,
            encoded: [f32; 8],
            gamma: f32,
        ) -> [f32; 8] {
            super::float::x8::gamma_to_linear(token, encoded, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear values to gamma-encoded. Input clamped to \[0, 1\].
        ///
        /// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn linear_to_gamma_v3(
            token: X64V3Token,
            linear: [f32; 8],
            gamma: f32,
        ) -> [f32; 8] {
            super::float::x8::linear_to_gamma(token, linear, gamma)
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 sRGB u8 values to linear f32 via 256-entry LUT.
        ///
        /// Pure table lookup — no math. The 1KB table fits in L1 cache.
        /// The token is accepted for API consistency; the operation itself
        /// is scalar lookups assembled into an array.
        ///
        /// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn srgb_u8_to_linear_v3(
            _token: X64V3Token,
            srgb: [u8; 8],
        ) -> [f32; 8] {
            let lut = crate::const_luts::linear_table_8();
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
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert sRGB u8 values to linear f32 using 8-wide LUT lookup.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn srgb_u8_to_linear_slice_v3(
            _token: X64V3Token,
            input: &[u8],
            output: &mut [f32],
        ) {
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
            let lut = crate::const_luts::linear_table_8();
            let (in_chunks, in_remainder) = input.as_chunks::<8>();
            let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();
            for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
                *out = [
                    lut[inp[0] as usize],
                    lut[inp[1] as usize],
                    lut[inp[2] as usize],
                    lut[inp[3] as usize],
                    lut[inp[4] as usize],
                    lut[inp[5] as usize],
                    lut[inp[6] as usize],
                    lut[inp[7] as usize],
                ];
            }
            for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
                *out = lut[*inp as usize];
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 sRGB u16 values to linear f32 via rational polynomial.
        ///
        /// Pure SIMD — no LUT, no cache footprint. Normalizes u16 to [0, 1] via
        /// `*(1.0/65535.0)` and applies the C0-continuous `S2L_P/Q` rational
        /// polynomial. Boundary values (0, 65535) map exactly to (0.0, 1.0).
        ///
        /// The `#[magetypes(...)]` macro generates `_v3`, `_neon`, `_wasm128`
        /// variants from this single body; f32x8 is native on V3 and polyfills
        /// to 2×f32x4 on NEON / WASM128 (fully inlined). Call from inside an
        /// `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn srgb_u16_to_linear_v3(
            token: ::archmage::X64V3Token,
            srgb: [u16; 8],
        ) -> [f32; 8] {
            use crate::rational_poly::{S2L_P, S2L_Q};
            #[allow(non_camel_case_types)]
            type f32x8 = gen_f32x8<::archmage::X64V3Token>;
            let mut f = [0.0f32; 8];
            for i in 0..8 {
                f[i] = srgb[i] as f32;
            }
            let v = f32x8::from_array_t(token, f) * f32x8::splat_t(token, 1.0 / 65535.0);
            let one = f32x8::splat_t(token, 1.0);
            let linear_result = v * f32x8::splat_t(token, LINEAR_SCALE);
            let yp = f32x8::splat_t(token, S2L_P[4])
                .mla(v, f32x8::splat_t(token, S2L_P[3]));
            let yp = yp.mla(v, f32x8::splat_t(token, S2L_P[2]));
            let yp = yp.mla(v, f32x8::splat_t(token, S2L_P[1]));
            let yp = yp.mla(v, f32x8::splat_t(token, S2L_P[0]));
            let yq = f32x8::splat_t(token, S2L_Q[4])
                .mla(v, f32x8::splat_t(token, S2L_Q[3]));
            let yq = yq.mla(v, f32x8::splat_t(token, S2L_Q[2]));
            let yq = yq.mla(v, f32x8::splat_t(token, S2L_Q[1]));
            let yq = yq.mla(v, f32x8::splat_t(token, S2L_Q[0]));
            let power_result = (yp / yq).min(one);
            let mask = v.simd_lt(f32x8::splat_t(token, SRGB_LINEAR_THRESHOLD));
            let result = f32x8::blend(mask, linear_result, power_result);
            let ge_one = v.simd_ge(one);
            f32x8::blend(ge_one, one, result).to_array()
        }
        #[inline]
        /// Convert 8 sRGB u16 values to linear f32 via rational polynomial.
        ///
        /// Pure SIMD — no LUT, no cache footprint. Normalizes u16 to [0, 1] via
        /// `*(1.0/65535.0)` and applies the C0-continuous `S2L_P/Q` rational
        /// polynomial. Boundary values (0, 65535) map exactly to (0.0, 1.0).
        ///
        /// The `#[magetypes(...)]` macro generates `_v3`, `_neon`, `_wasm128`
        /// variants from this single body; f32x8 is native on V3 and polyfills
        /// to 2×f32x4 on NEON / WASM128 (fully inlined). Call from inside an
        /// `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn srgb_u16_to_linear_scalar(
            token: ::archmage::ScalarToken,
            srgb: [u16; 8],
        ) -> [f32; 8] {
            use crate::rational_poly::{S2L_P, S2L_Q};
            #[allow(non_camel_case_types)]
            type f32x8 = gen_f32x8<::archmage::ScalarToken>;
            let mut f = [0.0f32; 8];
            for i in 0..8 {
                f[i] = srgb[i] as f32;
            }
            let v = f32x8::from_array_t(token, f) * f32x8::splat_t(token, 1.0 / 65535.0);
            let one = f32x8::splat_t(token, 1.0);
            let linear_result = v * f32x8::splat_t(token, LINEAR_SCALE);
            let yp = f32x8::splat_t(token, S2L_P[4])
                .mla(v, f32x8::splat_t(token, S2L_P[3]));
            let yp = yp.mla(v, f32x8::splat_t(token, S2L_P[2]));
            let yp = yp.mla(v, f32x8::splat_t(token, S2L_P[1]));
            let yp = yp.mla(v, f32x8::splat_t(token, S2L_P[0]));
            let yq = f32x8::splat_t(token, S2L_Q[4])
                .mla(v, f32x8::splat_t(token, S2L_Q[3]));
            let yq = yq.mla(v, f32x8::splat_t(token, S2L_Q[2]));
            let yq = yq.mla(v, f32x8::splat_t(token, S2L_Q[1]));
            let yq = yq.mla(v, f32x8::splat_t(token, S2L_Q[0]));
            let power_result = (yp / yq).min(one);
            let mask = v.simd_lt(f32x8::splat_t(token, SRGB_LINEAR_THRESHOLD));
            let result = f32x8::blend(mask, linear_result, power_result);
            let ge_one = v.simd_ge(one);
            f32x8::blend(ge_one, one, result).to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear f32 values to sRGB u8 via LUT. Input clamped to \[0, 1\].
        ///
        /// Uses a 4096-entry const LUT with bitmask indexing (`& 0xFFF`) for
        /// provably safe bounds. SIMD accelerates the clamp and scale; lookups
        /// are scalar from an L1-resident 4KB table.
        ///
        /// The `X64V3Token` parameter proves AVX2+FMA support at compile time.
        /// Call from inside an `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn linear_to_srgb_u8_v3(
            token: X64V3Token,
            linear: [f32; 8],
        ) -> [u8; 8] {
            let zero = mt_f32x8::zero_t(token);
            let one = mt_f32x8::splat_t(token, 1.0);
            let linear = mt_f32x8::from_array_t(token, linear).max(zero).min(one);
            let scaled = linear * mt_f32x8::splat_t(token, 4095.0)
                + mt_f32x8::splat_t(token, 0.5);
            let arr = scaled.to_array();
            let lut = crate::const_luts::linear_to_srgb_u8();
            [
                lut[arr[0] as usize & 0xFFF],
                lut[arr[1] as usize & 0xFFF],
                lut[arr[2] as usize & 0xFFF],
                lut[arr[3] as usize & 0xFFF],
                lut[arr[4] as usize & 0xFFF],
                lut[arr[5] as usize & 0xFFF],
                lut[arr[6] as usize & 0xFFF],
                lut[arr[7] as usize & 0xFFF],
            ]
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear f32 values to sRGB u16 via rational polynomial.
        ///
        /// Pure SIMD — no LUT, no cache footprint. Clamps to [0, 1], applies the
        /// C0-continuous `L2S_P/Q` rational polynomial on √x, then scales by
        /// 65535 + 0.5 and truncates to u16. Perfect roundtrip with the decode
        /// polynomial. Inputs ≥ 1.0 map to 65535 bit-exact; inputs ≤ 0.0 map to 0.
        ///
        /// The `#[magetypes(...)]` macro generates `_v3`, `_neon`, `_wasm128`
        /// variants from this single body; f32x8 is native on V3 and polyfills
        /// to 2×f32x4 on NEON / WASM128 (fully inlined). Call from inside an
        /// `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn linear_to_srgb_u16_v3(
            token: ::archmage::X64V3Token,
            linear: [f32; 8],
        ) -> [u16; 8] {
            use crate::rational_poly::{L2S_P, L2S_Q};
            #[allow(non_camel_case_types)]
            type f32x8 = gen_f32x8<::archmage::X64V3Token>;
            let v = f32x8::from_array_t(token, linear);
            let zero = f32x8::zero_t(token);
            let one = f32x8::splat_t(token, 1.0);
            let clamped = v.max(zero).min(one);
            let linear_result = clamped * f32x8::splat_t(token, TWELVE_92);
            let x = clamped.sqrt();
            let yp = f32x8::splat_t(token, L2S_P[4])
                .mla(x, f32x8::splat_t(token, L2S_P[3]));
            let yp = yp.mla(x, f32x8::splat_t(token, L2S_P[2]));
            let yp = yp.mla(x, f32x8::splat_t(token, L2S_P[1]));
            let yp = yp.mla(x, f32x8::splat_t(token, L2S_P[0]));
            let yq = f32x8::splat_t(token, L2S_Q[4])
                .mla(x, f32x8::splat_t(token, L2S_Q[3]));
            let yq = yq.mla(x, f32x8::splat_t(token, L2S_Q[2]));
            let yq = yq.mla(x, f32x8::splat_t(token, L2S_Q[1]));
            let yq = yq.mla(x, f32x8::splat_t(token, L2S_Q[0]));
            let power_result = (yp / yq).min(one);
            let thresh_mask = clamped.simd_lt(f32x8::splat_t(token, LINEAR_THRESHOLD));
            let srgb = f32x8::blend(thresh_mask, linear_result, power_result);
            let ge_one = v.simd_ge(one);
            let srgb = f32x8::blend(ge_one, one, srgb);
            let scaled = srgb
                .mla(f32x8::splat_t(token, 65535.0), f32x8::splat_t(token, 0.5));
            let idx = scaled.to_i32().to_array();
            [
                idx[0] as u16,
                idx[1] as u16,
                idx[2] as u16,
                idx[3] as u16,
                idx[4] as u16,
                idx[5] as u16,
                idx[6] as u16,
                idx[7] as u16,
            ]
        }
        #[inline]
        /// Convert 8 linear f32 values to sRGB u16 via rational polynomial.
        ///
        /// Pure SIMD — no LUT, no cache footprint. Clamps to [0, 1], applies the
        /// C0-continuous `L2S_P/Q` rational polynomial on √x, then scales by
        /// 65535 + 0.5 and truncates to u16. Perfect roundtrip with the decode
        /// polynomial. Inputs ≥ 1.0 map to 65535 bit-exact; inputs ≤ 0.0 map to 0.
        ///
        /// The `#[magetypes(...)]` macro generates `_v3`, `_neon`, `_wasm128`
        /// variants from this single body; f32x8 is native on V3 and polyfills
        /// to 2×f32x4 on NEON / WASM128 (fully inlined). Call from inside an
        /// `#[arcane]` function for zero-overhead inlining.
        pub(crate) fn linear_to_srgb_u16_scalar(
            token: ::archmage::ScalarToken,
            linear: [f32; 8],
        ) -> [u16; 8] {
            use crate::rational_poly::{L2S_P, L2S_Q};
            #[allow(non_camel_case_types)]
            type f32x8 = gen_f32x8<::archmage::ScalarToken>;
            let v = f32x8::from_array_t(token, linear);
            let zero = f32x8::zero_t(token);
            let one = f32x8::splat_t(token, 1.0);
            let clamped = v.max(zero).min(one);
            let linear_result = clamped * f32x8::splat_t(token, TWELVE_92);
            let x = clamped.sqrt();
            let yp = f32x8::splat_t(token, L2S_P[4])
                .mla(x, f32x8::splat_t(token, L2S_P[3]));
            let yp = yp.mla(x, f32x8::splat_t(token, L2S_P[2]));
            let yp = yp.mla(x, f32x8::splat_t(token, L2S_P[1]));
            let yp = yp.mla(x, f32x8::splat_t(token, L2S_P[0]));
            let yq = f32x8::splat_t(token, L2S_Q[4])
                .mla(x, f32x8::splat_t(token, L2S_Q[3]));
            let yq = yq.mla(x, f32x8::splat_t(token, L2S_Q[2]));
            let yq = yq.mla(x, f32x8::splat_t(token, L2S_Q[1]));
            let yq = yq.mla(x, f32x8::splat_t(token, L2S_Q[0]));
            let power_result = (yp / yq).min(one);
            let thresh_mask = clamped.simd_lt(f32x8::splat_t(token, LINEAR_THRESHOLD));
            let srgb = f32x8::blend(thresh_mask, linear_result, power_result);
            let ge_one = v.simd_ge(one);
            let srgb = f32x8::blend(ge_one, one, srgb);
            let scaled = srgb
                .mla(f32x8::splat_t(token, 65535.0), f32x8::splat_t(token, 0.5));
            let idx = scaled.to_i32().to_array();
            [
                idx[0] as u16,
                idx[1] as u16,
                idx[2] as u16,
                idx[3] as u16,
                idx[4] as u16,
                idx[5] as u16,
                idx[6] as u16,
                idx[7] as u16,
            ]
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert sRGB f32 values to linear in-place using 8-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn srgb_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<8>();
            for chunk in chunks {
                *chunk = srgb_to_linear_v3(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 8];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = srgb_to_linear_v3(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to sRGB in-place using 8-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_srgb_slice_v3(token: X64V3Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<8>();
            for chunk in chunks {
                *chunk = linear_to_srgb_v3(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 8];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_srgb_v3(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert sRGB f32 values to linear in-place (extended range, no clamping).
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn srgb_to_linear_extended_slice_v3(
            token: X64V3Token,
            values: &mut [f32],
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<8>();
            for chunk in chunks {
                *chunk = srgb_to_linear_extended_v3(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 8];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = srgb_to_linear_extended_v3(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to sRGB in-place (extended range, no clamping).
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_srgb_extended_slice_v3(
            token: X64V3Token,
            values: &mut [f32],
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<8>();
            for chunk in chunks {
                *chunk = linear_to_srgb_extended_v3(token, *chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 8];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_srgb_extended_v3(token, padded);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert gamma-encoded f32 values to linear in-place using 8-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn gamma_to_linear_slice_v3(
            token: X64V3Token,
            values: &mut [f32],
            gamma: f32,
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<8>();
            for chunk in chunks {
                *chunk = gamma_to_linear_v3(token, *chunk, gamma);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 8];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = gamma_to_linear_v3(token, padded, gamma);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to gamma-encoded in-place using 8-wide SIMD.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_gamma_slice_v3(
            token: X64V3Token,
            values: &mut [f32],
            gamma: f32,
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<8>();
            for chunk in chunks {
                *chunk = linear_to_gamma_v3(token, *chunk, gamma);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 8];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = linear_to_gamma_v3(token, padded, gamma);
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to sRGB u8 using 8-wide SIMD + LUT.
        ///
        /// Token parameter proves CPU support. Call from `#[arcane]` context.
        pub(crate) fn linear_to_srgb_u8_slice_v3(
            token: X64V3Token,
            input: &[f32],
            output: &mut [u8],
        ) {
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
                *out = linear_to_srgb_u8_v3(token, *inp);
            }
            for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
                *out = crate::scalar::linear_to_srgb_u8(*inp);
            }
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 sRGB values to linear (rational polynomial, no powf).
        pub(crate) fn tf_srgb_to_linear_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::tf::srgb::srgb_to_linear_x8(token, mt_f32x8::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear values to sRGB (rational polynomial, no powf).
        pub(crate) fn tf_linear_to_srgb_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::tf::srgb::linear_to_srgb_x8(token, mt_f32x8::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 BT.709 encoded values to linear.
        pub(crate) fn bt709_to_linear_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::tf::bt709::bt709_to_linear_x8(token, mt_f32x8::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear values to BT.709 encoded.
        pub(crate) fn linear_to_bt709_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::tf::bt709::linear_to_bt709_x8(token, mt_f32x8::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 PQ signal values to linear.
        pub(crate) fn pq_to_linear_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::tf::pq::pq_to_linear_x8(token, mt_f32x8::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear values to PQ signal.
        pub(crate) fn linear_to_pq_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::tf::pq::linear_to_pq_x8(token, mt_f32x8::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 HLG signal values to linear.
        pub(crate) fn hlg_to_linear_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::tf::hlg::hlg_to_linear_x8(token, mt_f32x8::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert 8 linear values to HLG signal.
        pub(crate) fn linear_to_hlg_v3(token: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            crate::tf::hlg::linear_to_hlg_x8(token, mt_f32x8::from_array_t(token, v))
                .to_array()
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert sRGB f32 values to linear in-place, 8-wide (TF module version).
        pub(crate) fn tf_srgb_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
            tf_slice_x8(
                values,
                |v| tf_srgb_to_linear_v3(token, v),
                crate::tf::srgb_to_linear,
            );
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to sRGB in-place, 8-wide (TF module version).
        pub(crate) fn tf_linear_to_srgb_slice_v3(token: X64V3Token, values: &mut [f32]) {
            tf_slice_x8(
                values,
                |v| tf_linear_to_srgb_v3(token, v),
                crate::tf::linear_to_srgb,
            );
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert BT.709 f32 values to linear in-place, 8-wide.
        pub(crate) fn bt709_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
            tf_slice_x8(
                values,
                |v| bt709_to_linear_v3(token, v),
                crate::tf::bt709_to_linear,
            );
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to BT.709 in-place, 8-wide.
        pub(crate) fn linear_to_bt709_slice_v3(token: X64V3Token, values: &mut [f32]) {
            tf_slice_x8(
                values,
                |v| linear_to_bt709_v3(token, v),
                crate::tf::linear_to_bt709,
            );
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert PQ f32 values to linear in-place, 8-wide.
        pub(crate) fn pq_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
            tf_slice_x8(values, |v| pq_to_linear_v3(token, v), crate::tf::pq_to_linear);
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to PQ in-place, 8-wide.
        pub(crate) fn linear_to_pq_slice_v3(token: X64V3Token, values: &mut [f32]) {
            tf_slice_x8(values, |v| linear_to_pq_v3(token, v), crate::tf::linear_to_pq);
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert HLG f32 values to linear in-place, 8-wide.
        pub(crate) fn hlg_to_linear_slice_v3(token: X64V3Token, values: &mut [f32]) {
            tf_slice_x8(
                values,
                |v| hlg_to_linear_v3(token, v),
                crate::tf::hlg_to_linear,
            );
        }
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        /// Convert linear f32 values to HLG in-place, 8-wide.
        pub(crate) fn linear_to_hlg_slice_v3(token: X64V3Token, values: &mut [f32]) {
            tf_slice_x8(
                values,
                |v| linear_to_hlg_v3(token, v),
                crate::tf::linear_to_hlg,
            );
        }
        #[inline(always)]
        fn tf_slice_x8(
            values: &mut [f32],
            tf_x8: impl Fn([f32; 8]) -> [f32; 8],
            _tf_scalar: fn(f32) -> f32,
        ) {
            let (chunks, remainder) = values.as_chunks_mut::<8>();
            for chunk in chunks {
                *chunk = tf_x8(*chunk);
            }
            if !remainder.is_empty() {
                let mut padded = [0.0; 8];
                padded[..remainder.len()].copy_from_slice(remainder);
                let converted = tf_x8(padded);
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
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    fn __attune_srgb_to_linear_slice_tier_v4x(
        _token: ::archmage::X64V4xToken,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::X64V4xToken>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::srgb_to_linear_slice_v4(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn srgb_to_linear_slice_tier_v4x_t(
        __archmage_token: ::archmage::X64V4xToken,
        values: &mut [f32],
    ) {
        unsafe { __attune_srgb_to_linear_slice_tier_v4x(__archmage_token, values) }
    }
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __attune_srgb_to_linear_slice_tier_v3(
        _token: ::archmage::X64V3Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::X64V3Token>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::srgb_to_linear_slice_v3(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn srgb_to_linear_slice_tier_v3_t(
        __archmage_token: ::archmage::X64V3Token,
        values: &mut [f32],
    ) {
        unsafe { __attune_srgb_to_linear_slice_tier_v3(__archmage_token, values) }
    }
    fn srgb_to_linear_slice_tier_scalar_t(_token: ScalarToken, values: &mut [f32]) {
        crate::target_features::x16::srgb_to_linear_slice_scalar(values);
    }
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    fn __arcane_srgb_to_linear_rgba_slice_tier_v4(
        _token: archmage::X64V4Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V4Token>;
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
    #[inline(always)]
    fn srgb_to_linear_rgba_slice_tier_v4(
        __archmage_token: archmage::X64V4Token,
        values: &mut [f32],
    ) {
        let _: () = <archmage::X64V4Token>::__ARCHMAGE_ASSERT_TIER_FE1B900C;
        unsafe { __arcane_srgb_to_linear_rgba_slice_tier_v4(__archmage_token, values) }
    }
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __arcane_srgb_to_linear_rgba_slice_tier_v3(
        _token: archmage::X64V3Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V3Token>;
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
    #[inline(always)]
    fn srgb_to_linear_rgba_slice_tier_v3(
        __archmage_token: archmage::X64V3Token,
        values: &mut [f32],
    ) {
        let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
        unsafe { __arcane_srgb_to_linear_rgba_slice_tier_v3(__archmage_token, values) }
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
                    if let Some(__attune_proof) = <::archmage::X64V4xToken as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 srgb_to_linear_slice_tier_v4x_t(
                            __attune_proof,
                            values,
                        );
                    }
                }
                {
                    if let Some(__attune_proof) = <::archmage::X64V3Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 srgb_to_linear_slice_tier_v3_t(
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
                #[allow(unexpected_cfgs)]
                {
                    if let Some(__t) = archmage::X64V4Token::summon() {
                        break '__incant srgb_to_linear_rgba_slice_tier_v4(__t, values);
                    }
                }
                if let Some(__t) = archmage::X64V3Token::summon() {
                    break '__incant srgb_to_linear_rgba_slice_tier_v3(__t, values);
                }
            }
            srgb_to_linear_rgba_slice_tier_scalar(archmage::ScalarToken, values)
        }
    }
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    fn __attune_linear_to_srgb_slice_tier_v4x(
        _token: ::archmage::X64V4xToken,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::X64V4xToken>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::linear_to_srgb_slice_v4(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn linear_to_srgb_slice_tier_v4x_t(
        __archmage_token: ::archmage::X64V4xToken,
        values: &mut [f32],
    ) {
        unsafe { __attune_linear_to_srgb_slice_tier_v4x(__archmage_token, values) }
    }
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __attune_linear_to_srgb_slice_tier_v3(
        _token: ::archmage::X64V3Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::X64V3Token>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::linear_to_srgb_slice_v3(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn linear_to_srgb_slice_tier_v3_t(
        __archmage_token: ::archmage::X64V3Token,
        values: &mut [f32],
    ) {
        unsafe { __attune_linear_to_srgb_slice_tier_v3(__archmage_token, values) }
    }
    fn linear_to_srgb_slice_tier_scalar_t(_token: ScalarToken, values: &mut [f32]) {
        crate::target_features::x16::linear_to_srgb_slice_scalar(values);
    }
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    fn __arcane_linear_to_srgb_rgba_slice_tier_v4(
        _token: archmage::X64V4Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V4Token>;
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
    #[inline(always)]
    fn linear_to_srgb_rgba_slice_tier_v4(
        __archmage_token: archmage::X64V4Token,
        values: &mut [f32],
    ) {
        let _: () = <archmage::X64V4Token>::__ARCHMAGE_ASSERT_TIER_FE1B900C;
        unsafe { __arcane_linear_to_srgb_rgba_slice_tier_v4(__archmage_token, values) }
    }
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __arcane_linear_to_srgb_rgba_slice_tier_v3(
        _token: archmage::X64V3Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V3Token>;
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
    #[inline(always)]
    fn linear_to_srgb_rgba_slice_tier_v3(
        __archmage_token: archmage::X64V3Token,
        values: &mut [f32],
    ) {
        let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
        unsafe { __arcane_linear_to_srgb_rgba_slice_tier_v3(__archmage_token, values) }
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
                    if let Some(__attune_proof) = <::archmage::X64V4xToken as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 linear_to_srgb_slice_tier_v4x_t(
                            __attune_proof,
                            values,
                        );
                    }
                }
                {
                    if let Some(__attune_proof) = <::archmage::X64V3Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 linear_to_srgb_slice_tier_v3_t(
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
                #[allow(unexpected_cfgs)]
                {
                    if let Some(__t) = archmage::X64V4Token::summon() {
                        break '__incant linear_to_srgb_rgba_slice_tier_v4(__t, values);
                    }
                }
                if let Some(__t) = archmage::X64V3Token::summon() {
                    break '__incant linear_to_srgb_rgba_slice_tier_v3(__t, values);
                }
            }
            linear_to_srgb_rgba_slice_tier_scalar(archmage::ScalarToken, values)
        }
    }
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    fn __attune_srgb_to_linear_extended_slice_tier_v4x(
        _token: ::archmage::X64V4xToken,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::X64V4xToken>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::srgb_to_linear_extended_slice_v4(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn srgb_to_linear_extended_slice_tier_v4x_t(
        __archmage_token: ::archmage::X64V4xToken,
        values: &mut [f32],
    ) {
        unsafe {
            __attune_srgb_to_linear_extended_slice_tier_v4x(__archmage_token, values)
        }
    }
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __attune_srgb_to_linear_extended_slice_tier_v3(
        _token: ::archmage::X64V3Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::X64V3Token>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::srgb_to_linear_extended_slice_v3(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn srgb_to_linear_extended_slice_tier_v3_t(
        __archmage_token: ::archmage::X64V3Token,
        values: &mut [f32],
    ) {
        unsafe {
            __attune_srgb_to_linear_extended_slice_tier_v3(__archmage_token, values)
        }
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
                    if let Some(__attune_proof) = <::archmage::X64V4xToken as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 srgb_to_linear_extended_slice_tier_v4x_t(
                            __attune_proof,
                            values,
                        );
                    }
                }
                {
                    if let Some(__attune_proof) = <::archmage::X64V3Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 srgb_to_linear_extended_slice_tier_v3_t(
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
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    fn __attune_linear_to_srgb_extended_slice_tier_v4x(
        _token: ::archmage::X64V4xToken,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::X64V4xToken>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::linear_to_srgb_extended_slice_v4(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn linear_to_srgb_extended_slice_tier_v4x_t(
        __archmage_token: ::archmage::X64V4xToken,
        values: &mut [f32],
    ) {
        unsafe {
            __attune_linear_to_srgb_extended_slice_tier_v4x(__archmage_token, values)
        }
    }
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __attune_linear_to_srgb_extended_slice_tier_v3(
        _token: ::archmage::X64V3Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<::archmage::X64V3Token>;
        {
            '__attune_call_0: {
                { crate::target_features::x16::linear_to_srgb_extended_slice_v3(values) }
            }
        };
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    fn linear_to_srgb_extended_slice_tier_v3_t(
        __archmage_token: ::archmage::X64V3Token,
        values: &mut [f32],
    ) {
        unsafe {
            __attune_linear_to_srgb_extended_slice_tier_v3(__archmage_token, values)
        }
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
                    if let Some(__attune_proof) = <::archmage::X64V4xToken as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 linear_to_srgb_extended_slice_tier_v4x_t(
                            __attune_proof,
                            values,
                        );
                    }
                }
                {
                    if let Some(__attune_proof) = <::archmage::X64V3Token as ::archmage::SimdToken>::summon() {
                        break '__attune_call_0 linear_to_srgb_extended_slice_tier_v3_t(
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
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    fn __arcane_srgb_to_linear_extended_rgba_slice_tier_v4(
        token: archmage::X64V4Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V4Token>;
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
    #[inline(always)]
    fn srgb_to_linear_extended_rgba_slice_tier_v4(
        token: archmage::X64V4Token,
        values: &mut [f32],
    ) {
        let _: () = <archmage::X64V4Token>::__ARCHMAGE_ASSERT_TIER_FE1B900C;
        unsafe { __arcane_srgb_to_linear_extended_rgba_slice_tier_v4(token, values) }
    }
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __arcane_srgb_to_linear_extended_rgba_slice_tier_v3(
        token: archmage::X64V3Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V3Token>;
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
    #[inline(always)]
    fn srgb_to_linear_extended_rgba_slice_tier_v3(
        token: archmage::X64V3Token,
        values: &mut [f32],
    ) {
        let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
        unsafe { __arcane_srgb_to_linear_extended_rgba_slice_tier_v3(token, values) }
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
                #[allow(unexpected_cfgs)]
                {
                    if let Some(__t) = archmage::X64V4Token::summon() {
                        break '__incant srgb_to_linear_extended_rgba_slice_tier_v4(
                            __t,
                            values,
                        );
                    }
                }
                if let Some(__t) = archmage::X64V3Token::summon() {
                    break '__incant srgb_to_linear_extended_rgba_slice_tier_v3(
                        __t,
                        values,
                    );
                }
            }
            srgb_to_linear_extended_rgba_slice_tier_scalar(archmage::ScalarToken, values)
        }
    }
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    fn __arcane_linear_to_srgb_extended_rgba_slice_tier_v4(
        token: archmage::X64V4Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V4Token>;
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
    #[inline(always)]
    fn linear_to_srgb_extended_rgba_slice_tier_v4(
        token: archmage::X64V4Token,
        values: &mut [f32],
    ) {
        let _: () = <archmage::X64V4Token>::__ARCHMAGE_ASSERT_TIER_FE1B900C;
        unsafe { __arcane_linear_to_srgb_extended_rgba_slice_tier_v4(token, values) }
    }
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __arcane_linear_to_srgb_extended_rgba_slice_tier_v3(
        token: archmage::X64V3Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V3Token>;
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
    #[inline(always)]
    fn linear_to_srgb_extended_rgba_slice_tier_v3(
        token: archmage::X64V3Token,
        values: &mut [f32],
    ) {
        let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
        unsafe { __arcane_linear_to_srgb_extended_rgba_slice_tier_v3(token, values) }
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
                #[allow(unexpected_cfgs)]
                {
                    if let Some(__t) = archmage::X64V4Token::summon() {
                        break '__incant linear_to_srgb_extended_rgba_slice_tier_v4(
                            __t,
                            values,
                        );
                    }
                }
                if let Some(__t) = archmage::X64V3Token::summon() {
                    break '__incant linear_to_srgb_extended_rgba_slice_tier_v3(
                        __t,
                        values,
                    );
                }
            }
            linear_to_srgb_extended_rgba_slice_tier_scalar(archmage::ScalarToken, values)
        }
    }
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    fn __arcane_srgb_to_linear_premultiply_rgba_slice_tier_v4(
        token: archmage::X64V4Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V4Token>;
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
    #[inline(always)]
    fn srgb_to_linear_premultiply_rgba_slice_tier_v4(
        token: archmage::X64V4Token,
        values: &mut [f32],
    ) {
        let _: () = <archmage::X64V4Token>::__ARCHMAGE_ASSERT_TIER_FE1B900C;
        unsafe { __arcane_srgb_to_linear_premultiply_rgba_slice_tier_v4(token, values) }
    }
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __arcane_srgb_to_linear_premultiply_rgba_slice_tier_v3(
        token: archmage::X64V3Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V3Token>;
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
    #[inline(always)]
    fn srgb_to_linear_premultiply_rgba_slice_tier_v3(
        token: archmage::X64V3Token,
        values: &mut [f32],
    ) {
        let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
        unsafe { __arcane_srgb_to_linear_premultiply_rgba_slice_tier_v3(token, values) }
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
                #[allow(unexpected_cfgs)]
                {
                    if let Some(__t) = archmage::X64V4Token::summon() {
                        break '__incant srgb_to_linear_premultiply_rgba_slice_tier_v4(
                            __t,
                            values,
                        );
                    }
                }
                if let Some(__t) = archmage::X64V3Token::summon() {
                    break '__incant srgb_to_linear_premultiply_rgba_slice_tier_v3(
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
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    fn __arcane_unpremultiply_linear_to_srgb_rgba_slice_tier_v4(
        token: archmage::X64V4Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V4Token>;
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
    #[inline(always)]
    fn unpremultiply_linear_to_srgb_rgba_slice_tier_v4(
        token: archmage::X64V4Token,
        values: &mut [f32],
    ) {
        let _: () = <archmage::X64V4Token>::__ARCHMAGE_ASSERT_TIER_FE1B900C;
        unsafe {
            __arcane_unpremultiply_linear_to_srgb_rgba_slice_tier_v4(token, values)
        }
    }
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __arcane_unpremultiply_linear_to_srgb_rgba_slice_tier_v3(
        token: archmage::X64V3Token,
        values: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V3Token>;
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
    #[inline(always)]
    fn unpremultiply_linear_to_srgb_rgba_slice_tier_v3(
        token: archmage::X64V3Token,
        values: &mut [f32],
    ) {
        let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
        unsafe {
            __arcane_unpremultiply_linear_to_srgb_rgba_slice_tier_v3(token, values)
        }
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
                #[allow(unexpected_cfgs)]
                {
                    if let Some(__t) = archmage::X64V4Token::summon() {
                        break '__incant unpremultiply_linear_to_srgb_rgba_slice_tier_v4(
                            __t,
                            values,
                        );
                    }
                }
                if let Some(__t) = archmage::X64V3Token::summon() {
                    break '__incant unpremultiply_linear_to_srgb_rgba_slice_tier_v3(
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
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    fn __arcane_linear_to_srgb_u8_slice_tier_v4(
        token: archmage::X64V4Token,
        input: &[f32],
        output: &mut [u8],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V4Token>;
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
    #[inline(always)]
    fn linear_to_srgb_u8_slice_tier_v4(
        token: archmage::X64V4Token,
        input: &[f32],
        output: &mut [u8],
    ) {
        let _: () = <archmage::X64V4Token>::__ARCHMAGE_ASSERT_TIER_FE1B900C;
        unsafe { __arcane_linear_to_srgb_u8_slice_tier_v4(token, input, output) }
    }
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __arcane_linear_to_srgb_u8_slice_tier_v3(
        token: archmage::X64V3Token,
        input: &[f32],
        output: &mut [u8],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V3Token>;
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
    #[inline(always)]
    fn linear_to_srgb_u8_slice_tier_v3(
        token: archmage::X64V3Token,
        input: &[f32],
        output: &mut [u8],
    ) {
        let _: () = <archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
        unsafe { __arcane_linear_to_srgb_u8_slice_tier_v3(token, input, output) }
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
                #[allow(unexpected_cfgs)]
                {
                    if let Some(__t) = archmage::X64V4Token::summon() {
                        break '__incant linear_to_srgb_u8_slice_tier_v4(
                            __t,
                            input,
                            output,
                        );
                    }
                }
                if let Some(__t) = archmage::X64V3Token::summon() {
                    break '__incant linear_to_srgb_u8_slice_tier_v3(__t, input, output);
                }
            }
            linear_to_srgb_u8_slice_tier_scalar(archmage::ScalarToken, input, output)
        }
    }
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
    )]
    #[inline]
    fn __arcane_srgb_u16_to_linear_slice_tier_v4(
        token: archmage::X64V4Token,
        input: &[u16],
        output: &mut [f32],
    ) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x16 = ::magetypes::simd::generic::f32x16<archmage::X64V4Token>;
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
