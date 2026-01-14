//! Fast SIMD math approximations for pow, exp2, and log2.
//!
//! Internal module providing SIMD pow approximation for sRGB conversion.
//! Uses LUT-based exp2 with polynomial log2 for best performance.

use bytemuck::cast;
use multiversed::multiversed;
use wide::{f32x8, i32x8, u32x8};

use crate::mlaf::mlaf;

// Constants for log2 approximation
const SQRT2_OVER_2_BITS: u32 = 0x3f3504f3; // sqrt(2)/2 ~ 0.7071
const ONE_BITS: u32 = 0x3f800000; // 1.0

// Polynomial coefficients for log2
const LOG2_C0: f32 = 0.412_198_57;
const LOG2_C1: f32 = 0.577_078_04;
const LOG2_C2: f32 = 0.961_796_7;
const LOG2_SCALE: f32 = 2.885_39; // 2/ln(2)

// 64-entry exp2 lookup table
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

// exp2 polynomial coefficients
const EXP2_C0: f32 = 0.240_226_5;
#[allow(clippy::approx_constant)]
const EXP2_C1: f32 = 0.693_147_2;
const TBLSIZE: usize = 64;

#[inline(always)]
fn f32x8_to_bits(v: f32x8) -> u32x8 {
    cast(v)
}

#[inline(always)]
fn f32x8_from_bits(v: u32x8) -> f32x8 {
    cast(v)
}

#[inline(always)]
fn f32x8_fma(a: f32x8, b: f32x8, c: f32x8) -> f32x8 {
    mlaf(c, a, b)
}

/// Fast approximate log2 for 8 f32 values.
#[multiversed]
#[inline]
pub(crate) fn log2_x8(d: f32x8) -> f32x8 {
    let bits = f32x8_to_bits(d);
    let offset = u32x8::splat(ONE_BITS - SQRT2_OVER_2_BITS);
    let adjusted = bits + offset;

    let exponent_raw: u32x8 = adjusted >> 23;
    let exponent_i32: i32x8 = cast(exponent_raw);
    let n = f32x8::from_i32x8(exponent_i32 - i32x8::splat(0x7f));

    let mantissa_mask = u32x8::splat(0x007fffff);
    let mantissa_bits = (adjusted & mantissa_mask) + u32x8::splat(SQRT2_OVER_2_BITS);
    let a = f32x8_from_bits(mantissa_bits);

    let one = f32x8::splat(1.0);
    let x = (a - one) / (a + one);
    let x2 = x * x;

    let mut u = f32x8::splat(LOG2_C0);
    u = f32x8_fma(u, x2, f32x8::splat(LOG2_C1));
    u = f32x8_fma(u, x2, f32x8::splat(LOG2_C2));

    f32x8_fma(x2 * x, u, f32x8_fma(x, f32x8::splat(LOG2_SCALE), n))
}

/// Fast approximate exp2 (2^x) for 8 f32 values.
#[multiversed]
#[inline]
pub(crate) fn exp2_x8(d: f32x8) -> f32x8 {
    let redux = f32x8::splat(f32::from_bits(0x4b400000) / TBLSIZE as f32);
    let sum = d + redux;
    let ui = f32x8_to_bits(sum);

    let i0 = (ui + u32x8::splat(TBLSIZE as u32 / 2)) & u32x8::splat(TBLSIZE as u32 - 1);
    let k: u32x8 = (ui + u32x8::splat(TBLSIZE as u32 / 2)) >> 6;

    let uf = sum - redux;
    let f = d - uf;

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

    let mut u = f32x8::splat(EXP2_C0);
    u = f32x8_fma(u, f, f32x8::splat(EXP2_C1));
    u *= f;

    let result_unscaled = f32x8_fma(u, z0, z0);

    let k_i32: i32x8 = cast(k);
    let scale_bits: u32x8 = cast((k_i32 + i32x8::splat(0x7f)) << 23);
    let scale: f32x8 = cast(scale_bits);

    result_unscaled * scale
}

/// Fast approximate pow(x, n) for 8 f32 values.
#[multiversed]
#[inline]
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
