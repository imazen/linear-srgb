//! Transfer profiles and their exceptional-lane fallbacks.
use archmage::ScalarToken;
use magemath::SimdMath;
use magemath::powf::{Powf, PowfDomain, PowfOptions};
use magetypes::simd::{backends::*, generic::*};

const GAMMA_OPTIONS: PowfOptions = PowfOptions::new()
    .domain(PowfDomain::Unit)
    .exponent_range(0.0, 4.0)
    .clamp_finite()
    .caller();
const GAMMA: u128 = GAMMA_OPTIONS.bits();
const FULL_POW: u128 = PowfOptions::new().full_domain().highp().caller().bits();
const SRGB_DECODE_POWER: u128 = GAMMA_OPTIONS.const_exponent(2.4).bits();
const SRGB_ENCODE_POWER: u128 = GAMMA_OPTIONS.const_exponent(1.0 / 2.4).bits();

#[cfg(feature = "transfer")]
pub(crate) const BT709_DECODE_POWER: u128 = PowfOptions::new()
    .positive_gamma()
    .const_exponent(1.0 / 0.45)
    .caller()
    .bits();
#[cfg(feature = "transfer")]
pub(crate) const BT709_ENCODE_POWER: u128 = PowfOptions::new()
    .positive_gamma()
    .const_exponent(0.45)
    .caller()
    .bits();

#[cfg(feature = "transfer")]
pub(crate) const EXP: u128 = magemath::expf::ExpfOptions::new()
    .full_domain()
    .caller()
    .bits();
#[cfg(feature = "transfer")]
pub(crate) const HLG_EXP: u128 = magemath::expf::ExpfOptions::new()
    .unclamped_exp2_range()
    .caller()
    .bits();
#[cfg(feature = "transfer")]
pub(crate) const LOG: u128 = magemath::logf::LogfOptions::new()
    .full_domain()
    .caller()
    .bits();
#[cfg(feature = "transfer")]
pub(crate) const HLG_LOG: u128 = magemath::logf::LogfOptions::new()
    .reject_subnormals()
    .caller()
    .bits();

#[cfg(all(test, feature = "transfer"))]
pub(crate) fn exp2(x: f32) -> f32 {
    exp2_with_token(ScalarToken, x)
}

#[cfg(feature = "transfer")]
#[inline(always)]
pub(crate) fn exp2_with_token<T: F32x4Convert>(token: T, x: f32) -> f32 {
    // Expf's Midp graph flushes subnormal outputs. HLG's vector argument
    // stays above this range, but callers of this scalar helper may enter it.
    if x < -126.0 {
        return num_traits::Float::exp2(x);
    }
    magemath::expf::Expf::<EXP>::exp2_with_token(token, x).expect("full exp2 domain")
}

#[cfg(all(test, feature = "transfer"))]
pub(crate) fn log2(x: f32) -> f32 {
    log2_with_token(ScalarToken, x)
}

#[cfg(feature = "transfer")]
#[inline(always)]
pub(crate) fn log2_with_token<T: F32x4Convert>(token: T, x: f32) -> f32 {
    magemath::logf::Logf::<LOG>::log2_with_token(token, x).expect("full log2 domain")
}

#[inline(always)]
fn full_pow(base: f32, n: f32) -> f32 {
    Powf::<FULL_POW>::scalar_with_token(ScalarToken, base, n).expect("full power domain")
}

#[cfg(feature = "transfer")]
#[inline(always)]
pub(crate) fn fixed_pow<const CFG: u128, T: F32x4Convert>(token: T, x: f32, exponent: f32) -> f32 {
    Powf::<CFG>::scalar_const_with_token(token, x).unwrap_or_else(|_| {
        Powf::<FULL_POW>::scalar_with_token(token, x, exponent).expect("full power domain")
    })
}

#[inline(always)]
pub(crate) fn gamma(x: f32, n: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if x.is_nan() {
        return x;
    }
    if n == 2.4 {
        Powf::<SRGB_DECODE_POWER>::scalar_const_infallible(x)
    } else if n == 1.0 / 2.4 {
        Powf::<SRGB_ENCODE_POWER>::scalar_const_infallible(x)
    } else if (0.0..=4.0).contains(&n) {
        Powf::<GAMMA>::scalar_infallible(x, n)
    } else {
        full_pow(x, n)
    }
}

#[cold]
#[inline(never)]
fn gamma_outside_fit<const N: usize>(values: [f32; N], exponent: f32) -> [f32; N] {
    values.map(|x| gamma(x, exponent))
}

macro_rules! gamma_for {
    ($name:ident, $float:ident, $backend:ident) => {
        #[inline(always)]
        pub(crate) fn $name<T: $backend>(t: T, v: $float<T>, n: f32) -> $float<T> {
            if !(0.0..=4.0).contains(&n) {
                return $float::from_array_t(t, gamma_outside_fit(v.to_array(), n));
            }
            let zero = $float::zero_t(t);
            let result = if n == 2.4 {
                v.powf_const::<SRGB_DECODE_POWER>()
            } else if n == 1.0 / 2.4 {
                v.powf_const::<SRGB_ENCODE_POWER>()
            } else if n == 0.0 {
                v.simd_gt(zero) & $float::splat_t(t, 1.0)
            } else {
                v.powf::<GAMMA>(n)
            };
            let nan = v.simd_ne(v);
            if nan.bitcast_to_i32().any_true() {
                $float::blend(nan, v, result)
            } else {
                result
            }
        }
    };
}

gamma_for!(gamma_x4, f32x4, F32x4Convert);
gamma_for!(gamma_x8, f32x8, F32x8Convert);
gamma_for!(gamma_x16, f32x16, F32x16Convert);
