//! Transfer profiles and their exceptional-lane fallbacks.
use archmage::ScalarToken;
use magemath::SimdMath;
use magemath::powf::{Powf, PowfOptions};
use magetypes::simd::{backends::*, generic::*};

const GAMMA: u128 = PowfOptions::new().unit_gamma_domain().caller().bits();
const FULL_POW: u128 = PowfOptions::new().full_domain().highp().caller().bits();
const SRGB_DECODE_POWER: u128 = PowfOptions::new()
    .unit_gamma_domain()
    .const_exponent(2.4)
    .caller()
    .bits();
const SRGB_ENCODE_POWER: u128 = PowfOptions::new()
    .unit_gamma_domain()
    .const_exponent(1.0 / 2.4)
    .caller()
    .bits();

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
pub(crate) const LOG: u128 = magemath::logf::LogfOptions::new()
    .full_domain()
    .caller()
    .bits();

#[cfg(feature = "transfer")]
#[inline(always)]
pub(crate) fn exp2(x: f32) -> f32 {
    // Expf's Midp graph flushes subnormal outputs. HLG's vector argument
    // stays above this range, but callers of this scalar helper may enter it.
    if x < -126.0 {
        return num_traits::Float::exp2(x);
    }
    magemath::expf::Expf::<EXP>::exp2_with_token(ScalarToken, x).expect("full exp2 domain")
}

#[cfg(feature = "transfer")]
#[inline(always)]
pub(crate) fn log2(x: f32) -> f32 {
    magemath::logf::Logf::<LOG>::log2_with_token(ScalarToken, x).expect("full log2 domain")
}

#[inline(always)]
fn full_pow(base: f32, n: f32) -> f32 {
    Powf::<FULL_POW>::scalar_with_token(ScalarToken, base, n).expect("full power domain")
}

#[cfg(feature = "transfer")]
#[inline(always)]
pub(crate) fn fixed_pow<const CFG: u128>(x: f32, exponent: f32) -> f32 {
    Powf::<CFG>::scalar_const_with_token(ScalarToken, x).unwrap_or_else(|_| full_pow(x, exponent))
}

#[inline(always)]
pub(crate) fn gamma(x: f32, n: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let fast = if n == 2.4 {
        Powf::<SRGB_DECODE_POWER>::scalar_const_with_token(ScalarToken, x)
    } else if n == 1.0 / 2.4 {
        Powf::<SRGB_ENCODE_POWER>::scalar_const_with_token(ScalarToken, x)
    } else {
        Powf::<GAMMA>::scalar_with_token(ScalarToken, x, n)
    };
    fast.unwrap_or_else(|_| full_pow(x, n))
}

macro_rules! gamma_for {
    ($name:ident, $float:ident, $backend:ident) => {
        #[inline(always)]
        pub(crate) fn $name<T: $backend>(t: T, v: $float<T>, n: f32) -> $float<T> {
            let zero = $float::zero_t(t);
            let one = $float::splat_t(t, 1.0);
            let positive = v.simd_gt(zero);
            let white = v.simd_ge(one);
            let nan = v.simd_ne(v);
            let inside = positive & v.simd_lt(one);
            // Evaluate endpoint lanes at 1, so invalid 0^n never enters the profile.
            let bases = $float::blend(inside, v, one);
            let fast = if n == 2.4 {
                bases.try_powf_const::<SRGB_DECODE_POWER>()
            } else if n == 1.0 / 2.4 {
                bases.try_powf_const::<SRGB_ENCODE_POWER>()
            } else {
                bases.try_powf_with::<GAMMA>(n)
            };
            let values = fast
                .unwrap_or_else(|_| $float::from_array_t(t, bases.to_array().map(|x| gamma(x, n))));
            let values = $float::blend(positive, values, zero);
            let values = $float::blend(white, one, values);
            $float::blend(nan, v, values)
        }
    };
}

gamma_for!(gamma_x4, f32x4, F32x4Convert);
gamma_for!(gamma_x8, f32x8, F32x8Convert);
gamma_for!(gamma_x16, f32x16, F32x16Convert);
