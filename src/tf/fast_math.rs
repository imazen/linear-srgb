//! Fast transfer function math using rational polynomials and bit tricks.
//!
//! Replaces libm `powf()` (~40-100ns, non-vectorizable) with rational polynomial
//! approximations (~5ns, SIMD-vectorizable). Coefficients from libjxl (BSD licensed).

#![allow(clippy::excessive_precision, clippy::assign_op_pattern)]

#[allow(unused_imports)]
use num_traits::Float; // provides mul_add/floor via libm in no_std

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
    // Clamp to valid IEEE 754 exponent range to avoid i32 overflow in debug builds.
    // Values outside [-126, 128] produce 0.0 or infinity anyway.
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

use magetypes::simd::backends::{F32x4Backend, F32x4Convert, F32x8Backend, F32x16Backend};
use magetypes::simd::generic::{f32x4, i32x4};

/// Multiply-add in the kernels. Fused wherever that costs nothing or the scalar tier needs it, so
/// every native tier gives the same bits. On WASM, `mul_add_fused` is software unless the engine
/// fuses, so it stays relaxed there unless the `precise-portable` feature asks for the same bits.
pub(crate) trait Mla: Sized {
    fn mla(self, a: Self, b: Self) -> Self;
}

macro_rules! mla_for {
    ($vector:ident, $backend:ident) => {
        impl<T: $backend> Mla for $vector<T> {
            #[inline(always)]
            fn mla(self, a: Self, b: Self) -> Self {
                #[cfg(all(target_arch = "wasm32", not(feature = "precise-portable")))]
                {
                    self.mul_add_relaxed(a, b)
                }
                #[cfg(not(all(target_arch = "wasm32", not(feature = "precise-portable"))))]
                {
                    self.mul_add_fused(a, b)
                }
            }
        }
    };
}
mla_for!(f32x4, F32x4Backend);
mla_for!(f32x8, F32x8Backend);
mla_for!(f32x16, F32x16Backend);

// `pow` for the gamma and near-black PQ paths, ours so that its rounding is under the same rule as
// the other kernels. The same coefficients and operations as magetypes' `pow_midp`.
pub(crate) const POW_SQRT1_2: u32 = 0x3f35_04f3;
pub(crate) const POW_LOG2: [f32; 4] = [2.885_39, 0.961_800_76, 0.576_974_45, 0.434_411_97];
pub(crate) const POW_EXP2: [f32; 7] = [
    1.0,
    core::f32::consts::LN_2,
    0.240_226_46,
    0.055_504_545,
    0.009_618_055,
    0.001_333_37,
    0.000_154_47,
];

macro_rules! pow_for {
    ($name:ident, $float:ident, $int:ident, $convert:ident) => {
        #[inline(always)]
        pub(crate) fn $name<T: $convert>(t: T, x: $float<T>, n: f32) -> $float<T> {
            let f = |v: f32| $float::splat_t(t, v);
            let i = |v: i32| $int::splat_t(t, v);

            // log2
            let adjusted = x.bitcast_to_i32() + i((0x3f80_0000u32 - POW_SQRT1_2) as i32);
            let exponent = $float::from_i32_t(t, adjusted.shr_arithmetic_const::<23>() - i(127));
            let a =
                $float::from_i32_bitcast_t(t, (adjusted & i(0x007f_ffff)) + i(POW_SQRT1_2 as i32));
            let y = (a - f(1.0)) / (a + f(1.0));
            let y2 = y * y;
            let poly = f(POW_LOG2[3]).mla(y2, f(POW_LOG2[2]));
            let poly = poly.mla(y2, f(POW_LOG2[1]));
            let poly = poly.mla(y2, f(POW_LOG2[0]));
            let log = poly.mla(y, exponent);
            let log = $float::blend(x.simd_eq(f(0.0)), f(f32::NEG_INFINITY), log);
            let log = $float::blend(x.simd_lt(f(0.0)), f(f32::NAN), log);
            let log = $float::blend(x.simd_eq(f(f32::INFINITY)), f(f32::INFINITY), log);

            // exp2
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
            let result = poly * $float::from_i32_bitcast_t(t, scale);
            let result = $float::blend(e.simd_lt(f(-126.0)), f(0.0), result);
            $float::blend(e.simd_ge(f(128.0)), f(f32::INFINITY), result)
        }
    };
}

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

    let mut den = f32x4::splat_t(t, POW2_DEN[0]).mla(frac, f32x4::splat_t(t, POW2_DEN[1]));
    den = den.mla(frac, f32x4::splat_t(t, POW2_DEN[2]));
    den = den.mla(frac, f32x4::splat_t(t, POW2_DEN[3]));

    num / den
}

/// fast_powf(base, exp) on 4 f32 values: pow2f(exp * log2f(base)).
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn fast_powf_x4<T: F32x4Convert>(t: T, base: f32x4<T>, exponent: f32) -> f32x4<T> {
    let log2 = fast_log2f_x4(t, base);
    let scaled = log2 * f32x4::splat_t(t, exponent);
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

    let mut den = f32x8::splat_t(t, POW2_DEN[0]).mla(frac, f32x8::splat_t(t, POW2_DEN[1]));
    den = den.mla(frac, f32x8::splat_t(t, POW2_DEN[2]));
    den = den.mla(frac, f32x8::splat_t(t, POW2_DEN[3]));

    num / den
}

/// fast_powf(base, exp) on 8 f32 values: pow2f(exp * log2f(base)).
#[inline(always)]
pub(crate) fn fast_powf_x8<T: F32x8Convert>(t: T, base: f32x8<T>, exponent: f32) -> f32x8<T> {
    let log2 = fast_log2f_x8(t, base);
    let scaled = log2 * f32x8::splat_t(t, exponent);
    fast_pow2f_x8(t, scaled)
}

// --- x16 generics (polyfilled to 2×x8 on V3, 4×x4 on NEON/WASM; native on V4 AVX-512) ---

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

    let mut den = f32x16::splat_t(t, POW2_DEN[0]).mla(frac, f32x16::splat_t(t, POW2_DEN[1]));
    den = den.mla(frac, f32x16::splat_t(t, POW2_DEN[2]));
    den = den.mla(frac, f32x16::splat_t(t, POW2_DEN[3]));

    num / den
}

/// fast_powf(base, exp) on 16 f32 values: pow2f(exp * log2f(base)).
#[inline(always)]
pub(crate) fn fast_powf_x16<T: F32x16Convert>(t: T, base: f32x16<T>, exponent: f32) -> f32x16<T> {
    let log2 = fast_log2f_x16(t, base);
    let scaled = log2 * f32x16::splat_t(t, exponent);
    fast_pow2f_x16(t, scaled)
}

pow_for!(pow_x4, f32x4, i32x4, F32x4Convert);
pow_for!(pow_x8, f32x8, i32x8, F32x8Convert);
pow_for!(pow_x16, f32x16, i32x16, F32x16Convert);
