//! Fast SIMD math approximations for pow, exp2, and log2.
//!
//! These are "dirty" approximations optimized for speed over precision.
//! Suitable for sRGB transfer functions where ~1e-5 error is acceptable.
//!
//! All functions are `_inline` only (no dispatch wrappers). Used by tests
//! and available for benchmarking.

use bytemuck::cast;
use wide::{CmpLt, f32x8, i32x8, u32x8};

// Constants for dirty_log2f_x8
const SQRT2_OVER_2_BITS: u32 = 0x3f3504f3; // sqrt(2)/2 ~ 0.7071
const ONE_BITS: u32 = 0x3f800000; // 1.0

// Polynomial coefficients for log2 approximation
// log2((1+x)/(1-x)) ≈ 2x * (1 + x²/3 + x⁴/5 + ...)
// Rearranged for (a-1)/(a+1) form
const LOG2_C0: f32 = 0.412_198_57;
const LOG2_C1: f32 = 0.577_078_04;
const LOG2_C2: f32 = 0.961_796_7;
const LOG2_SCALE: f32 = 2.885_39; // 2/ln(2)

// 64-entry exp2 lookup table (same as pxfm's EXP2FT)
// Each entry is 2^(k/64) for k in 0..64
#[rustfmt::skip]
static EXP2_TABLE: [u32; 64] = [
    0x3F3504F3, 0x3F36FD92, 0x3F38FBAF, 0x3F3AFF5B, 0x3F3D08A4, 0x3F3F179A, 0x3F412C4D, 0x3F4346CD,
    0x3F45672A, 0x3F478D75, 0x3F49B9BE, 0x3F4BEC15, 0x3F4E248C, 0x3F506334, 0x3F52A81E, 0x3F54F35B,
    0x3F5744FD, 0x3F599D16, 0x3F5BFBB8, 0x3F5E60F5, 0x3F60CCDF, 0x3F633F89, 0x3F65B907, 0x3F68396A,
    0x3F6AC0C7, 0x3F6D4F30, 0x3F6FE4BA, 0x3F728177, 0x3F75257D, 0x3F77D0DF, 0x3F7A83B3, 0x3F7D3E0C,
    0x3F800000, 0x3F8164D2, 0x3F82CD87, 0x3F843A29, 0x3F85AAC3, 0x3F871F62, 0x3F88980F, 0x3F8A14D5,
    0x3F8B95C2, 0x3F8D1ADF, 0x3F8EA43A, 0x3F9031DC, 0x3F91C3D3, 0x3F935A2B, 0x3F94F4F0, 0x3F96942D,
    0x3F9837F0, 0x3F99E046, 0x3F9B8D3A, 0x3F9D3EDA, 0x3F9EF532, 0x3FA0B051, 0x3FA27043, 0x3FA43516,
    0x3FA5FED7, 0x3FA7CD94, 0x3FA9A15B, 0x3FAB7A3A, 0x3FAD583F, 0x3FAF3B79, 0x3FB123F6, 0x3FB311C4,
];

// exp2 polynomial coefficients (from pxfm)
const EXP2_C0: f32 = 0.240_226_5;
#[allow(clippy::approx_constant)] // This is a polynomial coefficient, not meant to be LN_2
const EXP2_C1: f32 = 0.693_147_2;

// Table size for exp2
const TBLSIZE: usize = 64;

/// Helper: reinterpret f32x8 bits as u32x8 (zero-cost cast).
#[inline(always)]
fn f32x8_to_bits(v: f32x8) -> u32x8 {
    cast(v)
}

/// Helper: reinterpret u32x8 bits as f32x8 (zero-cost cast).
#[inline(always)]
fn f32x8_from_bits(v: u32x8) -> f32x8 {
    cast(v)
}

/// FMA: a * b + c using wide's native mul_add
#[inline(always)]
fn f32x8_fma(a: f32x8, b: f32x8, c: f32x8) -> f32x8 {
    a.mul_add(b, c)
}

/// Fast approximate log2 for 8 f32 values (always inlined).
///
/// Accuracy: ~1e-5 relative error for inputs in [0.001, 1000].
/// Not suitable for values near 0 or negative values.
#[inline(always)]
fn dirty_log2f_x8_inline(d: f32x8) -> f32x8 {
    // Extract bits
    let bits = f32x8_to_bits(d);

    // Normalize: add offset to handle exponent extraction
    // ix = ix + (0x3f800000 - 0x3f3504f3) to reduce x into [sqrt(2)/2, sqrt(2)]
    let offset = u32x8::splat(ONE_BITS - SQRT2_OVER_2_BITS);
    let adjusted = bits + offset;

    // Extract exponent: n = (ix >> 23) - 127
    let exponent_raw: u32x8 = adjusted >> 23;
    // Cast to i32x8, subtract bias, convert to f32x8 (all SIMD)
    let exponent_i32: i32x8 = cast(exponent_raw);
    let n = f32x8::from_i32x8(exponent_i32 - i32x8::splat(0x7f));

    // Reconstruct mantissa with exponent = 0 (biased 127)
    // ix = (ix & 0x007fffff) + 0x3f3504f3
    let mantissa_mask = u32x8::splat(0x007fffff);
    let mantissa_bits = (adjusted & mantissa_mask) + u32x8::splat(SQRT2_OVER_2_BITS);
    let a = f32x8_from_bits(mantissa_bits);

    // x = (a - 1) / (a + 1), range [-0.17, 0.17]
    let one = f32x8::splat(1.0);
    let x = (a - one) / (a + one);

    let x2 = x * x;

    // Polynomial: log2((1+x)/(1-x)) ≈ 2x * P(x²)
    // P(x²) = c2 + c1*x² + c0*x⁴
    let mut u = f32x8::splat(LOG2_C0);
    u = f32x8_fma(u, x2, f32x8::splat(LOG2_C1));
    u = f32x8_fma(u, x2, f32x8::splat(LOG2_C2));

    // Result: n + 2*x*P(x²)/ln(2) = n + x*u*scale
    f32x8_fma(x2 * x, u, f32x8_fma(x, f32x8::splat(LOG2_SCALE), n))
}

/// Fast approximate exp2 (2^x) for 8 f32 values (always inlined).
///
/// Accuracy: ~1e-5 relative error for inputs in [-10, 10].
/// Uses 64-entry LUT with polynomial refinement.
#[inline(always)]
fn dirty_exp2f_x8_inline(d: f32x8) -> f32x8 {
    // Redux constant for extracting integer part
    // redux = 0x4b400000 / 64 = 12582912 / 64
    let redux = f32x8::splat(f32::from_bits(0x4b400000) / TBLSIZE as f32);

    // Add redux to get the integer index bits
    let sum = d + redux;
    let ui = f32x8_to_bits(sum);

    // Extract table index: (ui + 32) & 63
    let i0 = (ui + u32x8::splat(TBLSIZE as u32 / 2)) & u32x8::splat(TBLSIZE as u32 - 1);

    // Extract k for 2^k scaling: k = (ui + 32) / 64
    let k: u32x8 = (ui + u32x8::splat(TBLSIZE as u32 / 2)) >> 6;

    // Fractional part: f = d - floor(d)
    let uf = sum - redux;
    let f = d - uf;

    // LUT lookup - scalar gather (SIMD gather not universally available)
    let i0_arr: [u32; 8] = i0.into();
    let z0 = f32x8::from([
        f32::from_bits(EXP2_TABLE[i0_arr[0] as usize]),
        f32::from_bits(EXP2_TABLE[i0_arr[1] as usize]),
        f32::from_bits(EXP2_TABLE[i0_arr[2] as usize]),
        f32::from_bits(EXP2_TABLE[i0_arr[3] as usize]),
        f32::from_bits(EXP2_TABLE[i0_arr[4] as usize]),
        f32::from_bits(EXP2_TABLE[i0_arr[5] as usize]),
        f32::from_bits(EXP2_TABLE[i0_arr[6] as usize]),
        f32::from_bits(EXP2_TABLE[i0_arr[7] as usize]),
    ]);

    // Polynomial refinement: u = c0 + c1*f, then u *= f
    let mut u = f32x8::splat(EXP2_C0);
    u = f32x8_fma(u, f, f32x8::splat(EXP2_C1));
    u *= f;

    // Result before scaling: (1 + u) * z0 = z0 + u*z0
    let result_unscaled = f32x8_fma(u, z0, z0);

    // Scale by 2^k using bit manipulation (SIMD)
    // pow2i(k) = float from bits ((k + 127) << 23)
    let k_i32: i32x8 = cast(k);
    let scale_bits: u32x8 = cast((k_i32 + i32x8::splat(0x7f)) << 23);
    let scale: f32x8 = cast(scale_bits);

    result_unscaled * scale
}

/// Fast approximate pow(x, n) for 8 f32 values (always inlined).
///
/// Computes x^n = exp2(n * log2(x)).
/// Accuracy: ~1e-4 relative error for typical sRGB range.
///
/// Note: Only handles positive x values. For negative x, behavior is undefined.
#[inline(always)]
fn dirty_pow_x8_inline(x: f32x8, n: f32x8) -> f32x8 {
    let lg = dirty_log2f_x8_inline(x);
    dirty_exp2f_x8_inline(n * lg)
}

// ============================================================================
// Imageflow-style fast math (no LUT, uses division-based polynomial)
// ============================================================================

// Imageflow constants for fastlog2/fastpow2
const IF_MANTISSA_MASK: u32 = 0x007FFFFF; // 23 mantissa bits
const IF_HALF_BIAS: u32 = 0x3F000000; // exponent for 0.5

/// Imageflow-style fast log2 for 8 f32 values (always inlined).
///
/// Uses bit manipulation to extract exponent and polynomial approximation.
/// Simpler than dirty_log2f_x8 but slightly less accurate.
#[inline(always)]
fn imageflow_log2_x8_inline(x: f32x8) -> f32x8 {
    // Extract bits
    let vx = f32x8_to_bits(x);

    // Create mantissa float in [0.5, 1) range by setting exponent to -1
    let mx_bits = (vx & u32x8::splat(IF_MANTISSA_MASK)) | u32x8::splat(IF_HALF_BIAS);
    let mx = f32x8_from_bits(mx_bits);

    // Convert raw bits to float and scale by 2^-23
    // This extracts: exponent + mantissa_fraction
    let vx_i32: i32x8 = cast(vx);
    let y = f32x8::from_i32x8(vx_i32) * f32x8::splat(1.192_092_9e-7);

    // Polynomial approximation:
    // log2(x) ≈ y - 124.22552 - 1.4980303 * mx - 1.72588 / (0.35208872 + mx)
    y - f32x8::splat(124.225_52)
        - f32x8::splat(1.498_030_3) * mx
        - f32x8::splat(1.725_88) / (f32x8::splat(0.352_088_72) + mx)
}

/// Imageflow-style fast pow2 (2^x) for 8 f32 values (always inlined).
///
/// Directly constructs IEEE 754 bits without LUT lookup.
/// For sRGB range inputs, this is faster than LUT-based dirty_exp2f_x8.
#[inline(always)]
fn imageflow_pow2_x8_inline(p: f32x8) -> f32x8 {
    // Handle offset for negative values
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);

    // offset = 1.0 if p < 0, else 0.0
    // Using simd_lt from CmpLt trait, then blend
    let is_negative = p.simd_lt(zero);
    let offset = is_negative.blend(one, zero);

    // Clamp to prevent overflow (min exponent is -126)
    let min_val = f32x8::splat(-126.0);
    let clipp = p.max(min_val);

    // Extract integer and fractional parts
    // w = trunc(clipp), z = frac(clipp) + offset
    // Use trunc_int and convert back to float
    let w_i32 = clipp.trunc_int();
    let w_f32 = f32x8::from_i32x8(w_i32);
    let z = clipp - w_f32 + offset;

    // Construct IEEE 754 bits using polynomial:
    // bits = 2^23 * (clipp + 121.274055 + 27.728024 / (4.8425255 - z) - 1.4901291 * z)
    let scale = f32x8::splat((1_i32 << 23) as f32);
    let inner = clipp
        + f32x8::splat(121.274_055)
        + f32x8::splat(27.728_024) / (f32x8::splat(4.842_525_5) - z)
        - f32x8::splat(1.490_129_1) * z;

    let bits_f32 = scale * inner;

    // Convert to integer bits and reinterpret as float
    // Use trunc_int for the conversion
    let bits_i32 = bits_f32.trunc_int();
    let bits_u32: u32x8 = cast(bits_i32);
    f32x8_from_bits(bits_u32)
}

/// Imageflow-style fast pow(x, n) for 8 f32 values (always inlined).
#[inline(always)]
fn imageflow_pow_x8_inline(x: f32x8, n: f32x8) -> f32x8 {
    let lg = imageflow_log2_x8_inline(x);
    imageflow_pow2_x8_inline(n * lg)
}

// ============================================================================
// Optimized fastpow: imageflow algorithm with RCPPS + Newton-Raphson
// ============================================================================

/// Fast reciprocal approximation with one Newton-Raphson refinement.
///
/// Uses RCPPS for ~12-bit approximation, then refines to ~23-bit accuracy.
/// Much faster than VDIVPS for computing 1/x or c/x.
#[inline(always)]
fn fast_recip_x8(x: f32x8) -> f32x8 {
    // Initial approximation (RCPPS gives ~12 bits of precision)
    let approx = x.recip();

    // One Newton-Raphson iteration: y' = y * (2 - x * y)
    // This gives ~23 bits of precision
    let two = f32x8::splat(2.0);
    approx * (two - x * approx)
}

/// Fast division c / x using reciprocal approximation.
#[inline(always)]
fn fast_div_x8(c: f32x8, x: f32x8) -> f32x8 {
    c * fast_recip_x8(x)
}

/// Optimized fast log2 using reciprocal approximation (always inlined).
#[inline(always)]
fn fastpow_log2_x8_inline(x: f32x8) -> f32x8 {
    // Extract bits
    let vx = f32x8_to_bits(x);

    // Create mantissa float in [0.5, 1) range
    let mx_bits = (vx & u32x8::splat(IF_MANTISSA_MASK)) | u32x8::splat(IF_HALF_BIAS);
    let mx = f32x8_from_bits(mx_bits);

    // Convert raw bits to float and scale by 2^-23
    let vx_i32: i32x8 = cast(vx);
    let y = f32x8::from_i32x8(vx_i32) * f32x8::splat(1.192_092_9e-7);

    // Use fast reciprocal instead of division
    // log2(x) ≈ y - 124.22552 - 1.4980303 * mx - 1.72588 / (0.35208872 + mx)
    let denom = f32x8::splat(0.352_088_72) + mx;
    let div_term = fast_div_x8(f32x8::splat(1.725_88), denom);

    y - f32x8::splat(124.225_52) - f32x8::splat(1.498_030_3) * mx - div_term
}

/// Optimized fast pow2 using reciprocal approximation (always inlined).
#[inline(always)]
fn fastpow_pow2_x8_inline(p: f32x8) -> f32x8 {
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);

    // offset = 1.0 if p < 0, else 0.0
    let is_negative = p.simd_lt(zero);
    let offset = is_negative.blend(one, zero);

    // Clamp to prevent overflow
    let min_val = f32x8::splat(-126.0);
    let clipp = p.max(min_val);

    // Extract integer and fractional parts
    let w_i32 = clipp.trunc_int();
    let w_f32 = f32x8::from_i32x8(w_i32);
    let z = clipp - w_f32 + offset;

    // Use fast reciprocal instead of division
    // bits = 2^23 * (clipp + 121.274055 + 27.728024 / (4.8425255 - z) - 1.4901291 * z)
    let denom = f32x8::splat(4.842_525_5) - z;
    let div_term = fast_div_x8(f32x8::splat(27.728_024), denom);

    let scale = f32x8::splat((1_i32 << 23) as f32);
    let inner = clipp + f32x8::splat(121.274_055) + div_term - f32x8::splat(1.490_129_1) * z;

    let bits_f32 = scale * inner;
    let bits_i32 = bits_f32.trunc_int();
    let bits_u32: u32x8 = cast(bits_i32);
    f32x8_from_bits(bits_u32)
}

/// Optimized fast pow using reciprocal approximation (always inlined).
///
/// This is the fastest SIMD pow approximation - uses RCPPS + Newton-Raphson
/// instead of VDIVPS for divisions.
#[inline(always)]
fn fastpow_x8_inline(x: f32x8, n: f32x8) -> f32x8 {
    let lg = fastpow_log2_x8_inline(x);
    fastpow_pow2_x8_inline(n * lg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirty_log2f_x8() {
        let input = f32x8::from([0.5, 1.0, 2.0, 4.0, 0.25, 8.0, 0.125, 16.0]);
        let result = dirty_log2f_x8_inline(input);
        let result_arr: [f32; 8] = result.into();

        let expected = [-1.0, 0.0, 1.0, 2.0, -2.0, 3.0, -3.0, 4.0];
        for (i, (&r, &e)) in result_arr.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "log2 mismatch at {}: got {}, expected {}",
                i,
                r,
                e
            );
        }
    }

    #[test]
    fn test_dirty_exp2f_x8() {
        // Test values in typical sRGB range (avoiding 0 which is an edge case)
        let input = f32x8::from([-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]);
        let result = dirty_exp2f_x8_inline(input);
        let result_arr: [f32; 8] = result.into();
        let input_arr: [f32; 8] = input.into();

        for (i, (&r, &inp)) in result_arr.iter().zip(input_arr.iter()).enumerate() {
            let expected = inp.exp2();
            assert!(
                (r - expected).abs() / expected.abs().max(1e-10) < 1e-4,
                "exp2 mismatch at {}: got {}, expected {}",
                i,
                r,
                expected
            );
        }
    }

    #[test]
    fn test_dirty_pow_x8() {
        // Test values relevant to sRGB conversion (avoiding 0)
        let x = f32x8::from([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]);
        let n = f32x8::splat(2.4);
        let result = dirty_pow_x8_inline(x, n);
        let result_arr: [f32; 8] = result.into();
        let x_arr: [f32; 8] = x.into();

        for (i, (&r, &inp)) in result_arr.iter().zip(x_arr.iter()).enumerate() {
            let expected = inp.powf(2.4);
            // Allow slightly larger tolerance for pow
            assert!(
                (r - expected).abs() / expected.abs().max(1e-10) < 5e-4,
                "pow mismatch at {}: got {}, expected {}",
                i,
                r,
                expected
            );
        }
    }

    #[test]
    fn test_dirty_pow_srgb_gamma() {
        // Test with sRGB gamma values (2.4 and 1/2.4)
        let x = f32x8::from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);

        // Test x^2.4 (sRGB decode)
        let gamma = f32x8::splat(2.4);
        let result = dirty_pow_x8_inline(x, gamma);
        let result_arr: [f32; 8] = result.into();
        let x_arr: [f32; 8] = x.into();

        for (i, (&r, &inp)) in result_arr.iter().zip(x_arr.iter()).enumerate() {
            let expected = inp.powf(2.4);
            assert!(
                (r - expected).abs() < 1e-4,
                "pow(x, 2.4) mismatch at {}: got {}, expected {}",
                i,
                r,
                expected
            );
        }

        // Test x^(1/2.4) (sRGB encode)
        let inv_gamma = f32x8::splat(1.0 / 2.4);
        let result = dirty_pow_x8_inline(x, inv_gamma);
        let result_arr: [f32; 8] = result.into();

        for (i, (&r, &inp)) in result_arr.iter().zip(x_arr.iter()).enumerate() {
            let expected = inp.powf(1.0 / 2.4);
            assert!(
                (r - expected).abs() < 1e-4,
                "pow(x, 1/2.4) mismatch at {}: got {}, expected {}",
                i,
                r,
                expected
            );
        }
    }

    #[test]
    fn test_imageflow_log2_x8() {
        let input = f32x8::from([0.5, 1.0, 2.0, 4.0, 0.25, 8.0, 0.125, 16.0]);
        let result = imageflow_log2_x8_inline(input);
        let result_arr: [f32; 8] = result.into();

        let expected = [-1.0, 0.0, 1.0, 2.0, -2.0, 3.0, -3.0, 4.0];
        for (i, (&r, &e)) in result_arr.iter().zip(expected.iter()).enumerate() {
            // Imageflow's approximation is less accurate (~1%)
            assert!(
                (r - e).abs() < 0.05,
                "imageflow log2 mismatch at {}: got {}, expected {}",
                i,
                r,
                e
            );
        }
    }

    #[test]
    fn test_imageflow_pow2_x8() {
        let input = f32x8::from([-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]);
        let result = imageflow_pow2_x8_inline(input);
        let result_arr: [f32; 8] = result.into();
        let input_arr: [f32; 8] = input.into();

        for (i, (&r, &inp)) in result_arr.iter().zip(input_arr.iter()).enumerate() {
            let expected = inp.exp2();
            // Imageflow's approximation allows ~5% error
            assert!(
                (r - expected).abs() / expected.abs().max(1e-10) < 0.05,
                "imageflow pow2 mismatch at {}: got {}, expected {}",
                i,
                r,
                expected
            );
        }
    }

    #[test]
    fn test_imageflow_pow_srgb_gamma() {
        // Test with sRGB gamma values
        let x = f32x8::from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);

        // Test x^(1/2.4) which is used in linear→sRGB
        let inv_gamma = f32x8::splat(1.0 / 2.4);
        let result = imageflow_pow_x8_inline(x, inv_gamma);
        let result_arr: [f32; 8] = result.into();
        let x_arr: [f32; 8] = x.into();

        for (i, (&r, &inp)) in result_arr.iter().zip(x_arr.iter()).enumerate() {
            let expected = inp.powf(1.0 / 2.4);
            // Imageflow's approximation allows ~5% relative error
            assert!(
                (r - expected).abs() / expected.max(1e-10) < 0.05,
                "imageflow pow(x, 1/2.4) mismatch at {}: got {}, expected {}",
                i,
                r,
                expected
            );
        }
    }

    #[test]
    fn test_fastpow_log2_x8() {
        let input = f32x8::from([0.5, 1.0, 2.0, 4.0, 0.25, 8.0, 0.125, 16.0]);
        let result = fastpow_log2_x8_inline(input);
        let result_arr: [f32; 8] = result.into();

        let expected = [-1.0, 0.0, 1.0, 2.0, -2.0, 3.0, -3.0, 4.0];
        for (i, (&r, &e)) in result_arr.iter().zip(expected.iter()).enumerate() {
            // Same tolerance as imageflow (should be similar accuracy)
            assert!(
                (r - e).abs() < 0.05,
                "fastpow log2 mismatch at {}: got {}, expected {}",
                i,
                r,
                e
            );
        }
    }

    #[test]
    fn test_fastpow_pow2_x8() {
        let input = f32x8::from([-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]);
        let result = fastpow_pow2_x8_inline(input);
        let result_arr: [f32; 8] = result.into();
        let input_arr: [f32; 8] = input.into();

        for (i, (&r, &inp)) in result_arr.iter().zip(input_arr.iter()).enumerate() {
            let expected = inp.exp2();
            // Same tolerance as imageflow
            assert!(
                (r - expected).abs() / expected.abs().max(1e-10) < 0.05,
                "fastpow pow2 mismatch at {}: got {}, expected {}",
                i,
                r,
                expected
            );
        }
    }

    #[test]
    fn test_fastpow_srgb_gamma() {
        // Test with sRGB gamma values
        let x = f32x8::from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);

        // Test x^(1/2.4) which is used in linear→sRGB
        let inv_gamma = f32x8::splat(1.0 / 2.4);
        let result = fastpow_x8_inline(x, inv_gamma);
        let result_arr: [f32; 8] = result.into();
        let x_arr: [f32; 8] = x.into();

        for (i, (&r, &inp)) in result_arr.iter().zip(x_arr.iter()).enumerate() {
            let expected = inp.powf(1.0 / 2.4);
            // Same tolerance as imageflow
            assert!(
                (r - expected).abs() / expected.max(1e-10) < 0.05,
                "fastpow pow(x, 1/2.4) mismatch at {}: got {}, expected {}",
                i,
                r,
                expected
            );
        }

        // Test x^2.4 which is used in sRGB→linear
        let gamma = f32x8::splat(2.4);
        let result = fastpow_x8_inline(x, gamma);
        let result_arr: [f32; 8] = result.into();

        for (i, (&r, &inp)) in result_arr.iter().zip(x_arr.iter()).enumerate() {
            let expected = inp.powf(2.4);
            assert!(
                (r - expected).abs() / expected.max(1e-10) < 0.05,
                "fastpow pow(x, 2.4) mismatch at {}: got {}, expected {}",
                i,
                r,
                expected
            );
        }
    }
}
