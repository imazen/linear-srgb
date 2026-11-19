//! One sample through the same operations as the vector kernels, with every multiply-add fused.
//!
//! Native and precise-portable scalar tiers use these. Default WASM uses the relaxed
//! generic vector kernels for float entry points and slice tails. These are
//! crate-private kernels written out per lane. Keep their operations paired with
//! the private vector copies; `tests/tier_parity.rs` holds them together.

use archmage::ScalarToken;
use magetypes::scalar::{F32ScalarBackend, sqrt_f32};

use crate::rational_poly::{
    EXT_L2S_P, EXT_L2S_Q, EXT_S2L_P, EXT_S2L_Q, LINEAR_SCALE, S2L_P, S2L_Q, SRGB_THRESHOLD,
};
use crate::rational_poly::{L2S_P, L2S_Q, LINEAR_THRESHOLD, TWELVE_92};
use crate::tf::polynomial::{eval_rational_poly, mla_scalar};

#[inline(always)]
fn unit(v: f32) -> f32 {
    if v > 0.0 { v.min(1.0) } else { 0.0 }
}

pub(crate) fn srgb_to_linear(v: f32) -> f32 {
    srgb_to_linear_with_token(ScalarToken, v)
}

#[inline(always)]
pub(crate) fn srgb_to_linear_with_token<T: F32ScalarBackend>(token: T, v: f32) -> f32 {
    let clamped = unit(v);
    if v >= 1.0 {
        return 1.0;
    }
    if clamped <= SRGB_THRESHOLD {
        return clamped * LINEAR_SCALE;
    }
    let power = eval_rational_poly(token, clamped, S2L_P, S2L_Q);
    if power < 1.0 { power } else { 1.0 }
}

pub(crate) fn linear_to_srgb(v: f32) -> f32 {
    linear_to_srgb_with_token(ScalarToken, v)
}

#[inline(always)]
fn linear_to_srgb_with_token<T: F32ScalarBackend>(token: T, v: f32) -> f32 {
    let clamped = unit(v);
    if v >= 1.0 {
        return 1.0;
    }
    if clamped <= LINEAR_THRESHOLD {
        return clamped * TWELVE_92;
    }
    let power = eval_rational_poly(token, sqrt_f32(clamped), L2S_P, L2S_Q);
    if power < 1.0 { power } else { 1.0 }
}

#[inline(always)]
pub(crate) fn linear_to_srgb_u16_with_token<T: F32ScalarBackend>(token: T, v: f32) -> u16 {
    mla_scalar(token, linear_to_srgb_with_token(token, v), 65535.0, 0.5) as u16
}

pub(crate) fn srgb_to_linear_extended(v: f32) -> f32 {
    let magnitude = v.abs();
    if magnitude > 8.0 {
        return crate::scalar::srgb_to_linear_extended(v);
    }
    let result = if magnitude < SRGB_THRESHOLD {
        magnitude * LINEAR_SCALE
    } else {
        eval_rational_poly(ScalarToken, magnitude, EXT_S2L_P, EXT_S2L_Q)
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
        eval_rational_poly(ScalarToken, sqrt_f32(magnitude), EXT_L2S_P, EXT_L2S_Q)
    };
    result.copysign(v)
}

pub(crate) fn gamma_to_linear(v: f32, gamma: f32) -> f32 {
    crate::tf::math::gamma(v, gamma)
}

pub(crate) fn linear_to_gamma(v: f32, gamma: f32) -> f32 {
    crate::tf::math::gamma(v, 1.0 / gamma)
}

// The explicit scalar tiers must retain their proof when evaluating HDR arrays.
#[cfg(feature = "transfer")]
macro_rules! scalar_hdr {
    ($module:ident, $($name:ident => $kernel:ident),+ $(,)?) => {
        $(
            #[inline(always)]
            pub(crate) fn $name(v: f32) -> f32 {
                crate::tf::$module::$kernel(ScalarToken, v)
            }
        )+
    };
}

#[cfg(feature = "transfer")]
scalar_hdr!(bt709, bt709_to_linear => bt709_to_linear_with_token, linear_to_bt709 => linear_to_bt709_with_token);
#[cfg(feature = "transfer")]
scalar_hdr!(pq, pq_to_linear => pq_to_linear_with_token, linear_to_pq => linear_to_pq_with_token);
#[cfg(feature = "transfer")]
scalar_hdr!(hlg, hlg_to_linear => hlg_to_linear_with_token, linear_to_hlg => linear_to_hlg_with_token);
