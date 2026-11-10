//! Shared sRGB and custom-gamma kernels. Public adapters own no math.
use crate::rational_poly::{
    LINEAR_SCALE, LINEAR_THRESHOLD, SRGB_THRESHOLD as SRGB_LINEAR_THRESHOLD, TWELVE_92,
};
use crate::tf::fast_math::Mla;
use magetypes::simd::{backends::*, generic::*};

macro_rules! width {
    ($module:ident, $float:ident, $backend:ident, $convert:ident, $width:literal, $pow:ident) => {
        pub(crate) mod $module {
            use super::*;
            #[inline(always)]
            pub(crate) fn srgb_to_linear<T: $backend>(
                token: T,
                srgb: [f32; $width],
            ) -> [f32; $width] {
                use crate::rational_poly::{S2L_P, S2L_Q};
                let zero = $float::<T>::zero_t(token);
                let one = $float::<T>::splat_t(token, 1.0);
                let v = $float::<T>::from_array_t(token, srgb);
                let clamped = v.max(zero).min(one);
                let linear_result = clamped * $float::<T>::splat_t(token, LINEAR_SCALE);
                // Rational polynomial P(x)/Q(x) via Horner's method
                let x = clamped;
                let yp = $float::<T>::splat_t(token, S2L_P[4])
                    .mla(x, $float::<T>::splat_t(token, S2L_P[3]));
                let yp = yp.mla(x, $float::<T>::splat_t(token, S2L_P[2]));
                let yp = yp.mla(x, $float::<T>::splat_t(token, S2L_P[1]));
                let yp = yp.mla(x, $float::<T>::splat_t(token, S2L_P[0]));
                let yq = $float::<T>::splat_t(token, S2L_Q[4])
                    .mla(x, $float::<T>::splat_t(token, S2L_Q[3]));
                let yq = yq.mla(x, $float::<T>::splat_t(token, S2L_Q[2]));
                let yq = yq.mla(x, $float::<T>::splat_t(token, S2L_Q[1]));
                let yq = yq.mla(x, $float::<T>::splat_t(token, S2L_Q[0]));
                let power_result = (yp / yq).min(one);
                let mask = clamped.simd_lt($float::<T>::splat_t(token, SRGB_LINEAR_THRESHOLD));
                let result = $float::<T>::blend(mask, linear_result, power_result);
                let ge_one = v.simd_ge(one);
                $float::<T>::blend(ge_one, one, result).to_array()
            }
            #[inline(always)]
            pub(crate) fn linear_to_srgb<T: $backend>(
                token: T,
                linear: [f32; $width],
            ) -> [f32; $width] {
                use crate::rational_poly::{L2S_P, L2S_Q};
                let zero = $float::<T>::zero_t(token);
                let one = $float::<T>::splat_t(token, 1.0);
                let v = $float::<T>::from_array_t(token, linear);
                let clamped = v.max(zero).min(one);
                let linear_result = clamped * $float::<T>::splat_t(token, TWELVE_92);
                // sqrt transform + rational polynomial P(√x)/Q(√x) via Horner's method
                let x = clamped.sqrt();
                let yp = $float::<T>::splat_t(token, L2S_P[4])
                    .mla(x, $float::<T>::splat_t(token, L2S_P[3]));
                let yp = yp.mla(x, $float::<T>::splat_t(token, L2S_P[2]));
                let yp = yp.mla(x, $float::<T>::splat_t(token, L2S_P[1]));
                let yp = yp.mla(x, $float::<T>::splat_t(token, L2S_P[0]));
                let yq = $float::<T>::splat_t(token, L2S_Q[4])
                    .mla(x, $float::<T>::splat_t(token, L2S_Q[3]));
                let yq = yq.mla(x, $float::<T>::splat_t(token, L2S_Q[2]));
                let yq = yq.mla(x, $float::<T>::splat_t(token, L2S_Q[1]));
                let yq = yq.mla(x, $float::<T>::splat_t(token, L2S_Q[0]));
                let power_result = (yp / yq).min(one);
                let mask = clamped.simd_lt($float::<T>::splat_t(token, LINEAR_THRESHOLD));
                let result = $float::<T>::blend(mask, linear_result, power_result);
                let ge_one = v.simd_ge(one);
                $float::<T>::blend(ge_one, one, result).to_array()
            }
            #[inline(always)]
            pub(crate) fn gamma_to_linear<T: $convert>(
                token: T,
                encoded: [f32; $width],
                gamma: f32,
            ) -> [f32; $width] {
                let zero = $float::<T>::zero_t(token);
                let one = $float::<T>::splat_t(token, 1.0);
                let v = $float::<T>::from_array_t(token, encoded).max(zero).min(one);
                crate::tf::fast_math::$pow(token, v, gamma).to_array()
            }
            #[inline(always)]
            pub(crate) fn linear_to_gamma<T: $convert>(
                token: T,
                linear: [f32; $width],
                gamma: f32,
            ) -> [f32; $width] {
                let zero = $float::<T>::zero_t(token);
                let one = $float::<T>::splat_t(token, 1.0);
                let v = $float::<T>::from_array_t(token, linear).max(zero).min(one);
                crate::tf::fast_math::$pow(token, v, 1.0 / gamma).to_array()
            }
            #[inline(always)]
            pub(crate) fn srgb_to_linear_extended<T: $backend>(
                token: T,
                srgb: [f32; $width],
            ) -> [f32; $width] {
                use crate::rational_poly::{EXT_S2L_P as P, EXT_S2L_Q as Q};

                let zero = $float::zero_t(token);
                let v = $float::from_array_t(token, srgb);
                let neg_mask = v.simd_lt(zero);
                let abs_v = v.abs();

                let linear_result = abs_v * $float::splat_t(token, LINEAR_SCALE);

                let x = abs_v;
                let yp = $float::splat_t(token, P[6]).mla(x, $float::splat_t(token, P[5]));
                let yp = yp.mla(x, $float::splat_t(token, P[4]));
                let yp = yp.mla(x, $float::splat_t(token, P[3]));
                let yp = yp.mla(x, $float::splat_t(token, P[2]));
                let yp = yp.mla(x, $float::splat_t(token, P[1]));
                let yp = yp.mla(x, $float::splat_t(token, P[0]));

                let yq = $float::splat_t(token, Q[6]).mla(x, $float::splat_t(token, Q[5]));
                let yq = yq.mla(x, $float::splat_t(token, Q[4]));
                let yq = yq.mla(x, $float::splat_t(token, Q[3]));
                let yq = yq.mla(x, $float::splat_t(token, Q[2]));
                let yq = yq.mla(x, $float::splat_t(token, Q[1]));
                let yq = yq.mla(x, $float::splat_t(token, Q[0]));

                let power_result = yp / yq;

                let thresh_mask = abs_v.simd_lt($float::splat_t(token, SRGB_LINEAR_THRESHOLD));
                let result = $float::blend(thresh_mask, linear_result, power_result);
                let result = $float::blend(neg_mask, -result, result);
                result.to_array()
            }
            #[inline(always)]
            pub(crate) fn linear_to_srgb_extended<T: $backend>(
                token: T,
                linear: [f32; $width],
            ) -> [f32; $width] {
                use crate::rational_poly::{EXT_L2S_P as P, EXT_L2S_Q as Q};

                let zero = $float::zero_t(token);
                let v = $float::from_array_t(token, linear);
                let neg_mask = v.simd_lt(zero);
                let abs_v = v.abs();

                let linear_result = abs_v * $float::splat_t(token, TWELVE_92);

                let x = abs_v.sqrt();
                let yp = $float::splat_t(token, P[6]).mla(x, $float::splat_t(token, P[5]));
                let yp = yp.mla(x, $float::splat_t(token, P[4]));
                let yp = yp.mla(x, $float::splat_t(token, P[3]));
                let yp = yp.mla(x, $float::splat_t(token, P[2]));
                let yp = yp.mla(x, $float::splat_t(token, P[1]));
                let yp = yp.mla(x, $float::splat_t(token, P[0]));

                let yq = $float::splat_t(token, Q[6]).mla(x, $float::splat_t(token, Q[5]));
                let yq = yq.mla(x, $float::splat_t(token, Q[4]));
                let yq = yq.mla(x, $float::splat_t(token, Q[3]));
                let yq = yq.mla(x, $float::splat_t(token, Q[2]));
                let yq = yq.mla(x, $float::splat_t(token, Q[1]));
                let yq = yq.mla(x, $float::splat_t(token, Q[0]));

                let power_result = yp / yq;

                let thresh_mask = abs_v.simd_lt($float::splat_t(token, LINEAR_THRESHOLD));
                let result = $float::blend(thresh_mask, linear_result, power_result);
                let result = $float::blend(neg_mask, -result, result);
                result.to_array()
            }
        }
    };
}
width!(x4, f32x4, F32x4Backend, F32x4Convert, 4, pow_x4);
#[cfg(any(target_arch = "x86_64", test))]
width!(x8, f32x8, F32x8Backend, F32x8Convert, 8, pow_x8);
width!(x16, f32x16, F32x16Backend, F32x16Convert, 16, pow_x16);
