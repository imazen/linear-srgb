//! Fast SIMD math approximations for pow, exp2, and log2.
//!
//! Internal module providing SIMD pow approximation for sRGB conversion.
//! Uses polynomial approximation (no table lookups) for best performance.
//!
//! Implementation derived from archmage's mid-precision transcendental functions.

use bytemuck::cast;
use wide::{f32x8, i32x8};

// ============================================================================
// Mid-precision log2 (~3 ULP max error)
// Uses (a-1)/(a+1) transform with degree-6 odd polynomial
// ============================================================================

/// Fast approximate log2 for 8 f32 values.
///
/// Uses (a-1)/(a+1) transform with odd polynomial.
/// Max error ~3 ULP (sufficient for 8-bit to 12-bit color).
#[inline(always)]
pub(crate) fn log2_x8(x: f32x8) -> f32x8 {
    // Constants for range reduction
    const SQRT2_OVER_2: u32 = 0x3f3504f3; // sqrt(2)/2 in f32 bits
    const ONE: u32 = 0x3f800000; // 1.0 in f32 bits
    const MANTISSA_MASK: u32 = 0x007fffff;
    const EXPONENT_BIAS: i32 = 127;

    // Coefficients for odd polynomial on y = (a-1)/(a+1)
    const C0: f32 = 2.885_39; // 2/ln(2)
    const C1: f32 = 0.961_800_8; // y^2 coefficient
    const C2: f32 = 0.576_974_5; // y^4 coefficient
    const C3: f32 = 0.434_412; // y^6 coefficient

    let x_bits: i32x8 = cast(x);

    // Normalize mantissa to [sqrt(2)/2, sqrt(2)]
    let offset = i32x8::splat((ONE - SQRT2_OVER_2) as i32);
    let adjusted = x_bits + offset;

    // Extract exponent
    let exp_raw = adjusted >> 23;
    let exp_biased = exp_raw - i32x8::splat(EXPONENT_BIAS);
    let n = f32x8::from_i32x8(exp_biased);

    // Reconstruct normalized mantissa
    let mantissa_bits = adjusted & i32x8::splat(MANTISSA_MASK as i32);
    let a_bits = mantissa_bits + i32x8::splat(SQRT2_OVER_2 as i32);
    let a: f32x8 = cast(a_bits);

    // y = (a - 1) / (a + 1) using fast reciprocal + Newton-Raphson
    let one = f32x8::splat(1.0);
    let two = f32x8::splat(2.0);
    let ap1 = a + one;
    let r = ap1.recip(); // RCPPS: ~12-bit precision
    let r = r * (two - ap1 * r); // Newton-Raphson: ~24-bit precision
    let y = (a - one) * r;

    // y^2
    let y2 = y * y;

    // Polynomial: c0 + y^2*(c1 + y^2*(c2 + y^2*c3))
    let poly = y2.mul_add(f32x8::splat(C3), f32x8::splat(C2));
    let poly = y2.mul_add(poly, f32x8::splat(C1));
    let poly = y2.mul_add(poly, f32x8::splat(C0));

    // Result: y * poly + n
    y.mul_add(poly, n)
}

// ============================================================================
// Mid-precision exp2 (~140 ULP max error)
// Uses degree-6 minimax polynomial
// ============================================================================

/// Fast approximate 2^x for 8 f32 values.
///
/// Uses a degree-6 minimax polynomial.
/// Max error ~140 ULP (~8e-6 relative error).
#[inline(always)]
pub(crate) fn exp2_x8(x: f32x8) -> f32x8 {
    // Degree-6 minimax polynomial coefficients for 2^x on [0, 1]
    // (truncated to f32 precision)
    const C0: f32 = 1.0;
    const C1: f32 = core::f32::consts::LN_2; // 0.693_147_18
    const C2: f32 = 0.240_226_5;
    const C3: f32 = 0.055_504_11;
    const C4: f32 = 0.009_618_129;
    const C5: f32 = 0.001_333_355_8;
    const C6: f32 = 0.000_154_035_3;

    // Clamp to safe range to avoid overflow/underflow
    let x = x.max(f32x8::splat(-126.0)).min(f32x8::splat(126.0));

    // Split into integer and fractional parts
    let xi = x.floor();
    let xf = x - xi;

    // Horner's method for the polynomial
    let poly = xf.mul_add(f32x8::splat(C6), f32x8::splat(C5));
    let poly = xf.mul_add(poly, f32x8::splat(C4));
    let poly = xf.mul_add(poly, f32x8::splat(C3));
    let poly = xf.mul_add(poly, f32x8::splat(C2));
    let poly = xf.mul_add(poly, f32x8::splat(C1));
    let poly = xf.mul_add(poly, f32x8::splat(C0));

    // Scale by 2^integer: construct float with exponent = xi + 127
    let xi_i32 = xi.round_int();
    let bias = i32x8::splat(127);
    let scale_bits: i32x8 = (xi_i32 + bias) << 23;
    let scale: f32x8 = cast(scale_bits);

    poly * scale
}

// ============================================================================
// pow(x, n) = 2^(n * log2(x))
// ============================================================================

/// Fast approximate pow(x, n) for 8 f32 values.
///
/// Computed as `2^(n * log2(x))`.
/// Combined error sufficient for sRGB 8-bit/10-bit/12-bit work.
#[inline(always)]
pub(crate) fn pow_x8(x: f32x8, n: f32) -> f32x8 {
    let lg = log2_x8(x);
    exp2_x8(f32x8::splat(n) * lg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2_x8() {
        let input = f32x8::from([0.5, 1.0, 2.0, 4.0, 0.25, 8.0, 0.125, 16.0]);
        let result = log2_x8(input);
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
    fn test_exp2_x8() {
        let input = f32x8::from([-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]);
        let result = exp2_x8(input);
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
    fn test_pow_x8_srgb_gamma() {
        let x = f32x8::from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);

        // Test x^2.4 (sRGB decode)
        let result = pow_x8(x, 2.4);
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
        let result = pow_x8(x, 1.0 / 2.4);
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
}
