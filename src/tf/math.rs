//! Transfer profiles, const-generic rail ablation, and exceptional-lane fallbacks.
use archmage::ScalarToken;
use magemath::SimdMath;
use magemath::powf::{Powf, PowfDomain, PowfOptions};
use magetypes::simd::{backends::*, generic::*};

pub(crate) const MATH_PORTABILITY: magemath::Portability = if super::polynomial::USE_FUSED_MATH {
    magemath::Portability::BitExact
} else {
    magemath::Portability::BackendDependent
};

const GAMMA_OPTIONS: PowfOptions = PowfOptions::new()
    .domain(PowfDomain::Unit)
    .exponent_range(0.0, 4.0)
    .clamp_finite()
    .portability(MATH_PORTABILITY)
    .caller();
const GAMMA: u128 = GAMMA_OPTIONS.bits();

/// Unchecked custom-gamma profile for verified finite inputs in `[1e-9, 1.0]`
/// and exponents in `(0.0, 4.0]`. Statically proves `|n * log2(x)| <= 120 <= 125`,
/// eliding all input clamps, NaN masks, subnormal checks, and `exp2` output saturation.
const GAMMA_UNCHECKED: u128 = PowfOptions::new()
    .domain(PowfDomain::PositiveUnit)
    .base_range(1.0e-9, 1.0)
    .exponent_range(0.0, 4.0)
    .reject_subnormals()
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();

const FULL_POW: u128 = PowfOptions::new().full_domain().highp().caller().bits();

#[cfg(feature = "transfer")]
pub(crate) const BT709_DECODE_POWER: u128 = PowfOptions::new()
    .positive_gamma()
    .const_exponent(1.0 / 0.45)
    .portability(MATH_PORTABILITY)
    .caller()
    .bits();
#[cfg(feature = "transfer")]
pub(crate) const BT709_ENCODE_POWER: u128 = PowfOptions::new()
    .positive_gamma()
    .const_exponent(0.45)
    .portability(MATH_PORTABILITY)
    .caller()
    .bits();
#[cfg(feature = "transfer")]
pub(crate) const BT709_DECODE_FAST: u128 = PowfOptions::new()
    .positive_gamma()
    .reject_subnormals()
    .const_exponent(1.0 / 0.45)
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();
#[cfg(feature = "transfer")]
pub(crate) const BT709_ENCODE_FAST: u128 = PowfOptions::new()
    .positive_gamma()
    .reject_subnormals()
    .const_exponent(0.45)
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();

#[cfg(feature = "transfer")]
pub(crate) const EXP: u128 = magemath::expf::ExpfOptions::new()
    .full_domain()
    .portability(MATH_PORTABILITY)
    .caller()
    .bits();
#[cfg(feature = "transfer")]
pub(crate) const HLG_EXP_FAST: u128 = magemath::expf::ExpfOptions::new()
    .unclamped_exp2_range()
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();
#[cfg(feature = "transfer")]
pub(crate) const LOG: u128 = magemath::logf::LogfOptions::new()
    .full_domain()
    .portability(MATH_PORTABILITY)
    .caller()
    .bits();
#[cfg(feature = "transfer")]
pub(crate) const HLG_LOG_FAST: u128 = magemath::logf::LogfOptions::new()
    .reject_subnormals()
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();

/// PQ EOTF dark-toe `ln(1 + dm1)` for `dm1 = fma(v, 1/C1^M2, -1) in [2.5e-4, 32768]`.
#[cfg(feature = "transfer")]
pub(crate) const PQ_EOTF_LOG1P: u128 = magemath::log1pf::Log1pfOptions::new()
    .input_range(2.5e-4, 32_768.0)
    .reject_subnormals()
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();

/// PQ EOTF dark-toe `base^(1/M1)` for `base = num / den in [1.25e-6, 0.125]`.
/// `|(1/M1) * log2(1.25e-6)| = 123.1 <= 125`, eliding all output range checks.
#[cfg(feature = "transfer")]
pub(crate) const PQ_EOTF_POW: u128 = PowfOptions::new()
    .domain(PowfDomain::PositiveUnit)
    .base_range(1.25e-6, 0.125)
    .const_exponent(16384.0 / 2610.0)
    .reject_subnormals()
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();

/// PQ OETF dark-toe `v^M1` for `v in (0, 1e-7)` (including positive subnormals).
#[cfg(feature = "transfer")]
pub(crate) const PQ_OETF_POW1: u128 = PowfOptions::new()
    .domain(PowfDomain::PositiveUnit)
    .const_exponent(2610.0 / 16384.0)
    .admit_subnormals()
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
    .caller()
    .bits();

/// PQ OETF dark-toe `ratio^M2` over `PowfDomain::PqRatio` (`[107/128, 1.0]`).
#[cfg(feature = "transfer")]
pub(crate) const PQ_OETF_POW2: u128 = PowfOptions::new()
    .domain(PowfDomain::PqRatio)
    .const_exponent(2523.0 / 32.0)
    .reject_subnormals()
    .portability(MATH_PORTABILITY)
    .assume_in_bounds()
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
        Powf::<FULL_POW>::scalar_with_token(ScalarToken, x, exponent).expect("full power domain")
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
    if (0.0..=4.0).contains(&n) {
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "relaxed-simd",
            not(feature = "precise-portable")
        ))]
        {
            use archmage::SimdToken;
            if let Some(w) = archmage::Wasm128Token::summon() {
                return Powf::<GAMMA>::scalar_with_token(w, x, n).expect("infallible gamma");
            }
        }
        Powf::<GAMMA>::scalar_infallible(x, n)
    } else {
        full_pow(x, n)
    }
}

#[cold]
#[inline(never)]
pub(crate) fn gamma_outside_fit<const N: usize>(values: [f32; N], exponent: f32) -> [f32; N] {
    values.map(|x| gamma(x, exponent))
}

#[cold]
#[inline(never)]
pub(crate) fn gamma_outside_fit_slice(values: &mut [f32], exponent: f32) {
    for x in values {
        *x = gamma(*x, exponent);
    }
}

#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
fn preserve_nans_patch<const N: usize>(src: &[f32; N], out: &mut [f32; N]) {
    for (&s, o) in src.iter().zip(out.iter_mut()) {
        if s.is_nan() {
            *o = s;
        }
    }
}

macro_rules! gamma_for {
    ($name:ident, $name_cfg:ident, $in_range:ident, $zero_exp:ident, $slice:ident, $slice_cfg:ident, $float:ident, $backend:ident, $width:literal) => {
        #[inline(always)]
        pub(crate) fn $zero_exp<T: $backend>(t: T, v: $float<T>) -> $float<T> {
            let zero = $float::zero_t(t);
            let result = v.simd_gt(zero) & $float::splat_t(t, 1.0);
            $float::blend(v.simd_ne(v), v, result)
        }

        #[inline(always)]
        pub(crate) fn $in_range<T: $backend>(t: T, v: $float<T>, n: f32) -> $float<T> {
            let _ = t;
            let result = v.powf::<GAMMA>(n);
            $float::blend(v.simd_ne(v), v, result)
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $name_cfg<T: $backend, const RAILS: bool>(
            t: T,
            v: $float<T>,
            n: f32,
        ) -> $float<T> {
            if !RAILS {
                return v.powf::<GAMMA_UNCHECKED>(n);
            }
            if !(0.0..=4.0).contains(&n) {
                return $float::from_array_t(t, gamma_outside_fit(v.to_array(), n));
            }
            if n == 0.0 {
                return $zero_exp(t, v);
            }
            $in_range(t, v, n)
        }

        #[inline(always)]
        pub(crate) fn $name<T: $backend>(t: T, v: $float<T>, n: f32) -> $float<T> {
            $name_cfg::<T, true>(t, v, n)
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $slice_cfg<T: $backend, const RAILS: bool>(t: T, values: &mut [f32], n: f32) {
            if !RAILS {
                #[cfg(not(target_arch = "x86_64"))]
                fn tail_unchecked<T: $backend>(
                    (t, n): (T, f32),
                    chunk: [f32; $width],
                ) -> [f32; $width] {
                    $float::from_array_t(t, chunk)
                        .powf::<GAMMA_UNCHECKED>(n)
                        .to_array()
                }
                crate::kernels::map_slice_inline!(
                    values,
                    $width,
                    (t, n),
                    |chunk| $float::from_array_t(t, chunk)
                        .powf::<GAMMA_UNCHECKED>(n)
                        .to_array(),
                    tail_unchecked::<T>
                );
                return;
            }
            if !(0.0..=4.0).contains(&n) {
                gamma_outside_fit_slice(values, n);
                return;
            }
            if n == 0.0 {
                // On x86-64 AVX2 ($width == 8), in-place `blend(isnan(v), v, result)` lowers
                // to microcoded `vmaskmovps`. Branching on NaN with a cold pointer patch keeps
                // an unconditional `vmovups` store on the hot path. AVX-512 ($width == 16) uses
                // fast hardware opmask `vmovups {k1}` and needs no branch.
                #[cfg(target_arch = "x86_64")]
                if $width == 8 {
                    let (chunks, remainder) = values.as_chunks_mut::<$width>();
                    let zero = $float::zero_t(t);
                    let one = $float::splat_t(t, 1.0);
                    for slot in chunks {
                        let v = $float::load_t(t, slot);
                        let result = v.simd_gt(zero) & one;
                        if v.simd_ne(v).bitcast_to_i32().any_true() {
                            let mut out = result.to_array();
                            preserve_nans_patch(slot, &mut out);
                            *slot = out;
                        } else {
                            result.store(slot);
                        }
                    }
                    if !remainder.is_empty() {
                        let mut block = [0.0; $width];
                        block[..remainder.len()].copy_from_slice(remainder);
                        let v = $float::load_t(t, &block);
                        let converted = $zero_exp(t, v).to_array();
                        remainder.copy_from_slice(&converted[..remainder.len()]);
                    }
                    return;
                }
                #[cfg(not(target_arch = "x86_64"))]
                fn tail_zero<T: $backend>(t: T, chunk: [f32; $width]) -> [f32; $width] {
                    $zero_exp(t, $float::from_array_t(t, chunk)).to_array()
                }
                crate::kernels::map_slice_inline!(
                    values,
                    $width,
                    t,
                    |chunk| $zero_exp(t, $float::from_array_t(t, chunk)).to_array(),
                    tail_zero::<T>
                );
                return;
            }
            #[cfg(target_arch = "x86_64")]
            if $width == 8 {
                let (chunks, remainder) = values.as_chunks_mut::<$width>();
                for slot in chunks {
                    let v = $float::load_t(t, slot);
                    let result = v.powf::<GAMMA>(n);
                    if v.simd_ne(v).bitcast_to_i32().any_true() {
                        let mut out = result.to_array();
                        preserve_nans_patch(slot, &mut out);
                        *slot = out;
                    } else {
                        result.store(slot);
                    }
                }
                if !remainder.is_empty() {
                    let mut block = [0.0; $width];
                    block[..remainder.len()].copy_from_slice(remainder);
                    let v = $float::load_t(t, &block);
                    let converted = $in_range(t, v, n).to_array();
                    remainder.copy_from_slice(&converted[..remainder.len()]);
                }
                return;
            }
            #[cfg(not(target_arch = "x86_64"))]
            fn tail_in_range<T: $backend>((t, n): (T, f32), chunk: [f32; $width]) -> [f32; $width] {
                $in_range(t, $float::from_array_t(t, chunk), n).to_array()
            }
            crate::kernels::map_slice_inline!(
                values,
                $width,
                (t, n),
                |chunk| $in_range(t, $float::from_array_t(t, chunk), n).to_array(),
                tail_in_range::<T>
            );
        }

        #[inline(always)]
        pub(crate) fn $slice<T: $backend>(t: T, values: &mut [f32], n: f32) {
            $slice_cfg::<T, true>(t, values, n);
        }
    };
}

gamma_for!(
    gamma_x4,
    gamma_x4_cfg,
    gamma_in_range_x4,
    gamma_zero_x4,
    gamma_slice_x4,
    gamma_slice_x4_cfg,
    f32x4,
    F32x4Convert,
    4
);
gamma_for!(
    gamma_x8,
    gamma_x8_cfg,
    gamma_in_range_x8,
    gamma_zero_x8,
    gamma_slice_x8,
    gamma_slice_x8_cfg,
    f32x8,
    F32x8Convert,
    8
);
gamma_for!(
    gamma_x16,
    gamma_x16_cfg,
    gamma_in_range_x16,
    gamma_zero_x16,
    gamma_slice_x16,
    gamma_slice_x16_cfg,
    f32x16,
    F32x16Convert,
    16
);
