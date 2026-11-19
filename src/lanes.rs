//! One sample through the same operations as the vector kernels, with every multiply-add fused.
//!
//! Native and precise-portable scalar tiers use these. Default WASM uses the relaxed
//! generic vector kernels for float entry points and slice tails. These are
//! crate-private kernels written out per lane. Keep their operations paired with
//! the private vector copies; `tests/tier_parity.rs` holds them together.

#[allow(unused_imports)]
use num_traits::Float;

use crate::rational_poly::{
    EXT_L2S_P, EXT_L2S_Q, EXT_S2L_P, EXT_S2L_Q, LINEAR_SCALE, S2L_P, S2L_Q, SRGB_THRESHOLD,
};
use crate::rational_poly::{L2S_P, L2S_Q, LINEAR_THRESHOLD, TWELVE_92};
use crate::tf::polynomial::eval_rational_poly;

#[inline(always)]
fn unit(v: f32) -> f32 {
    if v > 0.0 { v.min(1.0) } else { 0.0 }
}

pub(crate) fn srgb_to_linear(v: f32) -> f32 {
    let clamped = unit(v);
    if v >= 1.0 {
        return 1.0;
    }
    if clamped <= SRGB_THRESHOLD {
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
    if clamped <= LINEAR_THRESHOLD {
        return clamped * TWELVE_92;
    }
    let power = eval_rational_poly(clamped.sqrt(), L2S_P, L2S_Q);
    if power < 1.0 { power } else { 1.0 }
}

pub(crate) fn linear_to_srgb_u16(v: f32) -> u16 {
    linear_to_srgb(v).mul_add(65535.0, 0.5) as i32 as u16
}

#[cfg(feature = "transfer")]
pub(crate) use crate::tf::{
    bt709_to_linear, hlg_to_linear, linear_to_bt709, linear_to_hlg, linear_to_pq, pq_to_linear,
};

pub(crate) fn srgb_to_linear_extended(v: f32) -> f32 {
    let magnitude = v.abs();
    if magnitude > 8.0 {
        return crate::scalar::srgb_to_linear_extended(v);
    }
    let result = if magnitude < SRGB_THRESHOLD {
        magnitude * LINEAR_SCALE
    } else {
        eval_rational_poly(magnitude, EXT_S2L_P, EXT_S2L_Q)
    };
    result.copysign(v)
}

pub(crate) fn linear_to_srgb_extended(v: f32) -> f32 {
    let magnitude = v.abs();
    if magnitude > 64.0 {
        return crate::scalar::linear_to_srgb_extended(v);
    }
    let result = if magnitude < LINEAR_THRESHOLD {
        magnitude * TWELVE_92
    } else {
        eval_rational_poly(magnitude.sqrt(), EXT_L2S_P, EXT_L2S_Q)
    };
    result.copysign(v)
}

pub(crate) fn gamma_to_linear(v: f32, gamma: f32) -> f32 {
    crate::tf::math::gamma(v, gamma)
}

pub(crate) fn linear_to_gamma(v: f32, gamma: f32) -> f32 {
    crate::tf::math::gamma(v, 1.0 / gamma)
}
