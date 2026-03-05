//! Fast transfer function math using rational polynomials and bit tricks.
//!
//! Replaces libm `powf()` (~40-100ns, non-vectorizable) with rational polynomial
//! approximations (~5ns, SIMD-vectorizable). Coefficients from libjxl (BSD licensed).

// =============================================================================
// Rational polynomial evaluator — Horner's method for P(x)/Q(x)
// =============================================================================

/// Evaluate a rational polynomial P(x)/Q(x) using Horner's method.
///
/// Coefficients are stored lowest-degree-first: `p[0] + p[1]*x + p[2]*x^2 + ...`
#[inline(always)]
pub fn eval_rational_poly<const P: usize, const Q: usize>(x: f32, p: [f32; P], q: [f32; Q]) -> f32 {
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

// =============================================================================
// fast_log2f / fast_pow2f / fast_powf — bit-manipulation transcendentals
// =============================================================================

/// Fast base-2 logarithm (~3e-6 relative error).
///
/// Uses integer bit extraction for exponent + rational polynomial on mantissa.
/// Coefficients from libjxl (via jxl crate).
#[inline(always)]
pub fn fast_log2f(x: f32) -> f32 {
    let x_bits = x.to_bits() as i32;
    let exp_bits = x_bits.wrapping_sub(0x3f2aaaab);
    let exp_shifted = exp_bits >> 23;
    let mantissa = f32::from_bits((x_bits.wrapping_sub(exp_shifted << 23)) as u32);
    let exp_val = exp_shifted as f32;
    eval_rational_poly(mantissa - 1.0, LOG2_P, LOG2_Q) + exp_val
}

/// Fast base-2 exponentiation (~3e-6 relative error).
///
/// Splits into integer exponent (bit shift) + fractional part (rational polynomial).
/// Coefficients from libjxl (via jxl crate).
#[inline(always)]
pub fn fast_pow2f(x: f32) -> f32 {
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
pub fn fast_powf(base: f32, exp: f32) -> f32 {
    fast_pow2f(fast_log2f(base) * exp)
}

// =============================================================================
// Coefficients
// =============================================================================

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
pub(crate) const POW2_DEN: [f32; 4] = [2.10242958e-1, -2.22328856e-2, -1.94414990e1, 9.85506633e1];

// =============================================================================
// Generic SIMD versions
// =============================================================================

use magetypes::simd::backends::F32x4Convert;
use magetypes::simd::generic::{f32x4, i32x4};

/// Evaluate degree-4 rational polynomial P(x)/Q(x) on 4 f32 values via FMA.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn eval_rational_poly_x4<T: F32x4Convert>(
    t: T,
    x: f32x4<T>,
    p: [f32; 5],
    q: [f32; 5],
) -> f32x4<T> {
    let mut yp = f32x4::splat(t, p[4]);
    yp = yp.mul_add(x, f32x4::splat(t, p[3]));
    yp = yp.mul_add(x, f32x4::splat(t, p[2]));
    yp = yp.mul_add(x, f32x4::splat(t, p[1]));
    yp = yp.mul_add(x, f32x4::splat(t, p[0]));

    let mut yq = f32x4::splat(t, q[4]);
    yq = yq.mul_add(x, f32x4::splat(t, q[3]));
    yq = yq.mul_add(x, f32x4::splat(t, q[2]));
    yq = yq.mul_add(x, f32x4::splat(t, q[1]));
    yq = yq.mul_add(x, f32x4::splat(t, q[0]));

    yp / yq
}

/// fast_log2f on 4 f32 values — bit manipulation + rational polynomial.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn fast_log2f_x4<T: F32x4Convert>(t: T, x: f32x4<T>) -> f32x4<T> {
    let x_bits = x.bitcast_to_i32();
    let magic = i32x4::splat(t, 0x3f2aaaab_u32 as i32);
    let exp_bits = x_bits - magic;
    let exp_shifted = exp_bits.shr_arithmetic_const::<23>();

    let shifted_back = exp_shifted.shl_const::<23>();
    let mantissa_bits = x_bits - shifted_back;
    let mantissa = f32x4::from_i32_bitcast(t, mantissa_bits);

    let exp_f = f32x4::from_i32(t, exp_shifted);
    let one = f32x4::splat(t, 1.0);
    let m = mantissa - one;

    let mut yp = f32x4::splat(t, LOG2_P[2]);
    yp = yp.mul_add(m, f32x4::splat(t, LOG2_P[1]));
    yp = yp.mul_add(m, f32x4::splat(t, LOG2_P[0]));

    let mut yq = f32x4::splat(t, LOG2_Q[2]);
    yq = yq.mul_add(m, f32x4::splat(t, LOG2_Q[1]));
    yq = yq.mul_add(m, f32x4::splat(t, LOG2_Q[0]));

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
    let bias = i32x4::splat(t, 127);
    let exp_bits = (x_floor_i + bias).shl_const::<23>();
    let exp = f32x4::from_i32_bitcast(t, exp_bits);

    let mut num = frac + f32x4::splat(t, POW2_NUM[0]);
    num = num.mul_add(frac, f32x4::splat(t, POW2_NUM[1]));
    num = num.mul_add(frac, f32x4::splat(t, POW2_NUM[2]));
    num = num * exp;

    let mut den = f32x4::splat(t, POW2_DEN[0]).mul_add(frac, f32x4::splat(t, POW2_DEN[1]));
    den = den.mul_add(frac, f32x4::splat(t, POW2_DEN[2]));
    den = den.mul_add(frac, f32x4::splat(t, POW2_DEN[3]));

    num / den
}

/// fast_powf(base, exp) on 4 f32 values: pow2f(exp * log2f(base)).
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn fast_powf_x4<T: F32x4Convert>(t: T, base: f32x4<T>, exponent: f32) -> f32x4<T> {
    let log2 = fast_log2f_x4(t, base);
    let scaled = log2 * f32x4::splat(t, exponent);
    fast_pow2f_x4(t, scaled)
}

// --- x8 generics ---

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
    let mut yp = f32x8::splat(t, p[4]);
    yp = yp.mul_add(x, f32x8::splat(t, p[3]));
    yp = yp.mul_add(x, f32x8::splat(t, p[2]));
    yp = yp.mul_add(x, f32x8::splat(t, p[1]));
    yp = yp.mul_add(x, f32x8::splat(t, p[0]));

    let mut yq = f32x8::splat(t, q[4]);
    yq = yq.mul_add(x, f32x8::splat(t, q[3]));
    yq = yq.mul_add(x, f32x8::splat(t, q[2]));
    yq = yq.mul_add(x, f32x8::splat(t, q[1]));
    yq = yq.mul_add(x, f32x8::splat(t, q[0]));

    yp / yq
}

/// fast_log2f on 8 f32 values — bit manipulation + rational polynomial.
#[inline(always)]
pub(crate) fn fast_log2f_x8<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
    let x_bits = x.bitcast_to_i32();
    let magic = i32x8::splat(t, 0x3f2aaaab_u32 as i32);
    let exp_bits = x_bits - magic;
    let exp_shifted = exp_bits.shr_arithmetic_const::<23>();

    let shifted_back = exp_shifted.shl_const::<23>();
    let mantissa_bits = x_bits - shifted_back;
    let mantissa = f32x8::from_i32_bitcast(t, mantissa_bits);

    let exp_f = f32x8::from_i32(t, exp_shifted);
    let one = f32x8::splat(t, 1.0);
    let m = mantissa - one;

    let mut yp = f32x8::splat(t, LOG2_P[2]);
    yp = yp.mul_add(m, f32x8::splat(t, LOG2_P[1]));
    yp = yp.mul_add(m, f32x8::splat(t, LOG2_P[0]));

    let mut yq = f32x8::splat(t, LOG2_Q[2]);
    yq = yq.mul_add(m, f32x8::splat(t, LOG2_Q[1]));
    yq = yq.mul_add(m, f32x8::splat(t, LOG2_Q[0]));

    let poly = yp / yq;
    poly + exp_f
}

/// fast_pow2f on 8 f32 values — integer bit manipulation + polynomial.
#[inline(always)]
pub(crate) fn fast_pow2f_x8<T: F32x8Convert>(t: T, x: f32x8<T>) -> f32x8<T> {
    let x_floor = x.floor();
    let frac = x - x_floor;

    let x_floor_i = x_floor.to_i32();
    let bias = i32x8::splat(t, 127);
    let exp_bits = (x_floor_i + bias).shl_const::<23>();
    let exp = f32x8::from_i32_bitcast(t, exp_bits);

    let mut num = frac + f32x8::splat(t, POW2_NUM[0]);
    num = num.mul_add(frac, f32x8::splat(t, POW2_NUM[1]));
    num = num.mul_add(frac, f32x8::splat(t, POW2_NUM[2]));
    num = num * exp;

    let mut den = f32x8::splat(t, POW2_DEN[0]).mul_add(frac, f32x8::splat(t, POW2_DEN[1]));
    den = den.mul_add(frac, f32x8::splat(t, POW2_DEN[2]));
    den = den.mul_add(frac, f32x8::splat(t, POW2_DEN[3]));

    num / den
}

/// fast_powf(base, exp) on 8 f32 values: pow2f(exp * log2f(base)).
#[inline(always)]
pub(crate) fn fast_powf_x8<T: F32x8Convert>(t: T, base: f32x8<T>, exponent: f32) -> f32x8<T> {
    let log2 = fast_log2f_x8(t, base);
    let scaled = log2 * f32x8::splat(t, exponent);
    fast_pow2f_x8(t, scaled)
}
