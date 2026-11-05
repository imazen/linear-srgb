//! One sample through the same operations as the vector kernels, with every multiply-add fused.
//!
//! The scalar tiers use these, so a value converts to the same bits at every tier. They are
//! crate-private kernels written out per lane. Keep their operations paired with
//! the private vector copies; `tests/tier_parity.rs` holds them together.

#[allow(unused_imports)]
use num_traits::Float;

use crate::rational_poly::{
    EXT_L2S_P, EXT_L2S_Q, EXT_S2L_P, EXT_S2L_Q, L2S_P, L2S_Q, LINEAR_SCALE, LINEAR_THRESHOLD,
    S2L_P, S2L_Q, SRGB_THRESHOLD, TWELVE_92,
};
#[cfg(feature = "transfer")]
use crate::tf::fast_math::fast_log2f;
use crate::tf::fast_math::{self, eval_rational_poly};

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
#[cfg(feature = "transfer")]
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
    let adjusted =
        (x.to_bits() as i32).wrapping_add((0x3f80_0000u32 - fast_math::POW_SQRT1_2) as i32);
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

#[cfg(feature = "transfer")]
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

#[cfg(feature = "transfer")]
pub(crate) fn linear_to_bt709(v: f32) -> f32 {
    use crate::tf::bt709::{BT709_ALPHA, BT709_BETA};
    if v < BT709_BETA {
        return v * 4.5;
    }
    let power = pow2(fast_log2f(v) * 0.45);
    (1.0 + BT709_ALPHA).mul_add(power, -BT709_ALPHA)
}

#[cfg(feature = "transfer")]
pub(crate) fn hlg_to_linear(v: f32) -> f32 {
    use crate::tf::hlg::{HLG_B, HLG_C, HLG_INV_A_LOG2E};
    let a = non_negative(v);
    if a <= 0.5 {
        return (a * a) * (1.0 / 3.0);
    }
    (pow2((a - HLG_C) * HLG_INV_A_LOG2E) + HLG_B) * (1.0 / 12.0)
}

#[cfg(feature = "transfer")]
pub(crate) fn linear_to_hlg(v: f32) -> f32 {
    use crate::tf::hlg::{HLG_A_INV_LOG2E, HLG_B, HLG_C};
    let a = non_negative(v);
    if a <= 1.0 / 12.0 {
        return (3.0 * a).sqrt();
    }
    let high = HLG_A_INV_LOG2E.mul_add(fast_log2f(12.0 * a - HLG_B), HLG_C);
    if high > 0.5 { high } else { 0.5 }
}

#[cfg(feature = "transfer")]
pub(crate) fn pq_to_linear(v: f32) -> f32 {
    use crate::tf::pq::{
        C1, C2, C3, EOTF_POLY_FLOOR, M1, M2, PQ_EOTF_P_LARGE, PQ_EOTF_P_SMALL, PQ_EOTF_Q_LARGE,
        PQ_EOTF_Q_SMALL,
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

#[cfg(feature = "transfer")]
pub(crate) fn linear_to_pq(v: f32) -> f32 {
    use crate::tf::pq::{
        C1, C2, C3, M1, M2, OETF_POLY_FLOOR, PQ_INV_P_LARGE, PQ_INV_P_SMALL, PQ_INV_Q_LARGE,
        PQ_INV_Q_SMALL,
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
