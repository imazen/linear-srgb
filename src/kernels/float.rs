//! Shared sRGB and custom-gamma kernels parameterized by `const` rail flags.
//! Public adapters own no math.

use crate::rational_poly::{
    LINEAR_SCALE, LINEAR_THRESHOLD, SRGB_THRESHOLD as SRGB_LINEAR_THRESHOLD, TWELVE_92,
};
use crate::tf::polynomial::Mla;
use magetypes::{
    scalar::F32ScalarBackend,
    simd::{backends::*, generic::*},
};

/// Compile-time rail flags for const-generic sRGB vector evaluation.
///
/// Specializing at compile time ablates redundant input/output clamps when a
/// caller's domain is already narrower (e.g. `u16 * (1/65535)` in `[0, 1]`),
/// and enables direct codegen/latency ablation benchmarks without duplicating
/// polynomial bodies.
pub(crate) mod cfg {
    /// Clamp negative and unordered (`NaN`) inputs to `+0.0`.
    pub(crate) const CLAMP_LO: u32 = 1 << 0;
    /// Clamp inputs `> 1.0` to `1.0` before polynomial evaluation.
    pub(crate) const CLAMP_HI: u32 = 1 << 1;
    /// Force exact `1.0` for inputs `>= 1.0` (`P(1)/Q(1)` lands 6–7 ULPs below `1.0`).
    pub(crate) const SNAP_ONE: u32 = 1 << 2;
    /// Legacy post-division `.min(1.0)` clamp (proven redundant over all `2^32`
    /// `f32` inputs for the current C0 coefficients; retained for ablation tests).
    #[allow(dead_code)]
    pub(crate) const POST_DIV_MIN_ONE: u32 = 1 << 3;

    /// Full public clamped `f32` contract (`[0, 1]` clip, `NaN -> 0`, exact `1.0`).
    pub(crate) const SRGB_CLAMPED: u32 = CLAMP_LO | CLAMP_HI | SNAP_ONE;
    /// Normalized non-negative `[0, 1]` integer decode (`u16 / 65535`): elides
    /// lower/upper pre-clamps while preserving exact `1.0` at `65535`.
    pub(crate) const SRGB_UNIT_NONNEG: u32 = SNAP_ONE;
    /// Pure in-domain `[0, 1)` evaluation with all boundary clamps ablated.
    #[allow(dead_code)]
    pub(crate) const SRGB_IN_OPEN_UNIT: u32 = 0;
    /// Legacy 0.6.13-pre configuration including the redundant post-div `.min(1.0)`.
    #[allow(dead_code)]
    pub(crate) const SRGB_LEGACY_POST_MIN: u32 = SRGB_CLAMPED | POST_DIV_MIN_ONE;
}

#[cold]
#[inline(never)]
pub(crate) fn srgb_to_linear_extended_patch<const N: usize>(
    source: &[f32; N],
    result: &mut [f32; N],
) {
    for (&s, o) in source.iter().zip(result.iter_mut()) {
        if s.abs() > 8.0 {
            *o = crate::scalar::srgb_to_linear_extended(s);
        }
    }
}

#[cold]
#[inline(never)]
pub(crate) fn srgb_to_linear_extended_fallback<const N: usize>(
    source: [f32; N],
    mut result: [f32; N],
) -> [f32; N] {
    srgb_to_linear_extended_patch(&source, &mut result);
    result
}

#[cold]
#[inline(never)]
pub(crate) fn linear_to_srgb_extended_patch<T: F32ScalarBackend, const N: usize>(
    token: T,
    source: &[f32; N],
    result: &mut [f32; N],
) {
    for (&s, o) in source.iter().zip(result.iter_mut()) {
        if s.abs() > 64.0 {
            *o = crate::scalar::linear_to_srgb_extended_with_token(token, s);
        }
    }
}

#[cold]
#[inline(never)]
pub(crate) fn linear_to_srgb_extended_fallback<T: F32ScalarBackend, const N: usize>(
    token: T,
    source: [f32; N],
    mut result: [f32; N],
) -> [f32; N] {
    linear_to_srgb_extended_patch(token, &source, &mut result);
    result
}

macro_rules! width {
    ($(#[$attr:meta])* $module:ident, $float:ident, $backend:ident, $convert:ident, $width:literal, $pow:ident, $pow_cfg:ident, $pow_slice:ident, $pow_slice_cfg:ident; $(#[$array_only:meta])*) => {
        $(#[$attr])*
        pub(crate) mod $module {
            use super::*;

            $(#[$array_only])*
            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn srgb_to_linear<T: $backend>(
                token: T,
                srgb: [f32; $width],
            ) -> [f32; $width] {
                srgb_to_linear_vector(token, $float::from_array_t(token, srgb)).to_array()
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn srgb_to_linear_cfg<T: $backend, const CFG: u32>(
                token: T,
                srgb: [f32; $width],
            ) -> [f32; $width] {
                srgb_to_linear_vector_cfg::<T, CFG>(token, $float::from_array_t(token, srgb))
                    .to_array()
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn srgb_to_linear_slice_cfg<T: $backend, const CFG: u32>(
                token: T,
                values: &mut [f32],
            ) {
                #[cfg(not(target_arch = "x86_64"))]
                fn tail<T: $backend, const CFG: u32>(
                    token: T,
                    chunk: [f32; $width],
                ) -> [f32; $width] {
                    srgb_to_linear_cfg::<T, CFG>(token, chunk)
                }
                crate::kernels::map_slice_inline!(
                    values,
                    $width,
                    token,
                    |chunk| srgb_to_linear_cfg::<T, CFG>(token, chunk),
                    tail::<T, CFG>
                );
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn srgb_to_linear_slice<T: $backend>(token: T, values: &mut [f32]) {
                srgb_to_linear_slice_cfg::<T, { cfg::SRGB_CLAMPED }>(token, values);
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn srgb_to_linear_slice_into<T: $backend>(
                token: T,
                input: &[f32],
                output: &mut [f32],
            ) {
                crate::kernels::map_slice_into_inline!(input, output, $width, |chunk| {
                    srgb_to_linear_cfg::<T, { cfg::SRGB_CLAMPED }>(token, chunk)
                });
            }

            #[inline(always)]
            pub(crate) fn srgb_to_linear_vector<T: $backend>(token: T, v: $float<T>) -> $float<T> {
                srgb_to_linear_vector_cfg::<T, { cfg::SRGB_CLAMPED }>(token, v)
            }

            #[inline(always)]
            pub(crate) fn srgb_to_linear_vector_cfg<T: $backend, const CFG: u32>(
                token: T,
                v: $float<T>,
            ) -> $float<T> {
                use crate::rational_poly::{S2L_P, S2L_Q};
                let one = $float::<T>::splat_t(token, 1.0);
                let x = if (CFG & cfg::CLAMP_LO) != 0 {
                    let zero = $float::<T>::zero_t(token);
                    let lo = crate::tf::polynomial::floor_zero!(v, zero);
                    if (CFG & cfg::CLAMP_HI) != 0 {
                        lo.min_fast(one)
                    } else {
                        lo
                    }
                } else if (CFG & cfg::CLAMP_HI) != 0 {
                    v.min_fast(one)
                } else {
                    v
                };
                let linear_result = x * $float::<T>::splat_t(token, LINEAR_SCALE);
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
                let div = yp / yq;
                let power_result = if (CFG & cfg::POST_DIV_MIN_ONE) != 0 {
                    div.min(one)
                } else {
                    div
                };
                let mask = x.simd_le($float::<T>::splat_t(token, SRGB_LINEAR_THRESHOLD));
                let result = $float::<T>::blend(mask, linear_result, power_result);
                if (CFG & cfg::SNAP_ONE) != 0 {
                    $float::<T>::blend(v.simd_ge(one), one, result)
                } else {
                    result
                }
            }

            $(#[$array_only])*
            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn linear_to_srgb<T: $backend>(
                token: T,
                linear: [f32; $width],
            ) -> [f32; $width] {
                linear_to_srgb_vector(token, $float::from_array_t(token, linear)).to_array()
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn linear_to_srgb_cfg<T: $backend, const CFG: u32>(
                token: T,
                linear: [f32; $width],
            ) -> [f32; $width] {
                linear_to_srgb_vector_cfg::<T, CFG>(token, $float::from_array_t(token, linear))
                    .to_array()
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn linear_to_srgb_slice_cfg<T: $backend, const CFG: u32>(
                token: T,
                values: &mut [f32],
            ) {
                #[cfg(not(target_arch = "x86_64"))]
                fn tail<T: $backend, const CFG: u32>(
                    token: T,
                    chunk: [f32; $width],
                ) -> [f32; $width] {
                    linear_to_srgb_cfg::<T, CFG>(token, chunk)
                }
                crate::kernels::map_slice_inline!(
                    values,
                    $width,
                    token,
                    |chunk| linear_to_srgb_cfg::<T, CFG>(token, chunk),
                    tail::<T, CFG>
                );
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn linear_to_srgb_slice<T: $backend>(token: T, values: &mut [f32]) {
                linear_to_srgb_slice_cfg::<T, { cfg::SRGB_CLAMPED }>(token, values);
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn linear_to_srgb_slice_into<T: $backend>(
                token: T,
                input: &[f32],
                output: &mut [f32],
            ) {
                crate::kernels::map_slice_into_inline!(input, output, $width, |chunk| {
                    linear_to_srgb_cfg::<T, { cfg::SRGB_CLAMPED }>(token, chunk)
                });
            }

            #[inline(always)]
            pub(crate) fn linear_to_srgb_vector<T: $backend>(token: T, v: $float<T>) -> $float<T> {
                linear_to_srgb_vector_cfg::<T, { cfg::SRGB_CLAMPED }>(token, v)
            }

            #[inline(always)]
            pub(crate) fn linear_to_srgb_vector_cfg<T: $backend, const CFG: u32>(
                token: T,
                v: $float<T>,
            ) -> $float<T> {
                use crate::rational_poly::{L2S_P, L2S_Q};
                let one = $float::<T>::splat_t(token, 1.0);
                let clamped = if (CFG & cfg::CLAMP_LO) != 0 {
                    let zero = $float::<T>::zero_t(token);
                    let lo = crate::tf::polynomial::floor_zero!(v, zero);
                    if (CFG & cfg::CLAMP_HI) != 0 {
                        lo.min_fast(one)
                    } else {
                        lo
                    }
                } else if (CFG & cfg::CLAMP_HI) != 0 {
                    v.min_fast(one)
                } else {
                    v
                };
                let linear_result = clamped * $float::<T>::splat_t(token, TWELVE_92);
                let x = clamped.fast_sqrt();
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
                let div = yp / yq;
                let power_result = if (CFG & cfg::POST_DIV_MIN_ONE) != 0 {
                    div.min(one)
                } else {
                    div
                };
                let mask = clamped.simd_le($float::<T>::splat_t(token, LINEAR_THRESHOLD));
                let result = $float::<T>::blend(mask, linear_result, power_result);
                if (CFG & cfg::SNAP_ONE) != 0 {
                    $float::<T>::blend(v.simd_ge(one), one, result)
                } else {
                    result
                }
            }

            $(#[$array_only])*
            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn gamma_to_linear<T: $convert>(
                token: T,
                encoded: [f32; $width],
                gamma: f32,
            ) -> [f32; $width] {
                let v = $float::<T>::from_array_t(token, encoded);
                crate::tf::math::$pow(token, v, gamma).to_array()
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn gamma_to_linear_cfg<T: $convert, const RAILS: bool>(
                token: T,
                encoded: [f32; $width],
                gamma: f32,
            ) -> [f32; $width] {
                let v = $float::<T>::from_array_t(token, encoded);
                crate::tf::math::$pow_cfg::<T, RAILS>(token, v, gamma).to_array()
            }

            $(#[$array_only])*
            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn linear_to_gamma<T: $convert>(
                token: T,
                linear: [f32; $width],
                gamma: f32,
            ) -> [f32; $width] {
                let v = $float::<T>::from_array_t(token, linear);
                crate::tf::math::$pow(token, v, 1.0 / gamma).to_array()
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn linear_to_gamma_cfg<T: $convert, const RAILS: bool>(
                token: T,
                linear: [f32; $width],
                gamma: f32,
            ) -> [f32; $width] {
                let v = $float::<T>::from_array_t(token, linear);
                crate::tf::math::$pow_cfg::<T, RAILS>(token, v, 1.0 / gamma).to_array()
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn gamma_to_linear_slice<T: $convert>(
                token: T,
                values: &mut [f32],
                gamma: f32,
            ) {
                crate::tf::math::$pow_slice(token, values, gamma);
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn gamma_to_linear_slice_cfg<T: $convert, const RAILS: bool>(
                token: T,
                values: &mut [f32],
                gamma: f32,
            ) {
                crate::tf::math::$pow_slice_cfg::<T, RAILS>(token, values, gamma);
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn linear_to_gamma_slice<T: $convert>(
                token: T,
                values: &mut [f32],
                gamma: f32,
            ) {
                crate::tf::math::$pow_slice(token, values, 1.0 / gamma);
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn linear_to_gamma_slice_cfg<T: $convert, const RAILS: bool>(
                token: T,
                values: &mut [f32],
                gamma: f32,
            ) {
                crate::tf::math::$pow_slice_cfg::<T, RAILS>(token, values, 1.0 / gamma);
            }

            #[inline(always)]
            pub(crate) fn srgb_to_linear_extended_poly<T: $convert>(
                token: T,
                v: $float<T>,
            ) -> ($float<T>, $float<T>) {
                use crate::rational_poly::{EXT_S2L_P as P, EXT_S2L_Q as Q};
                let sign = $float::splat_t(token, -0.0);
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
                (abs_v, result.abs() | (v & sign))
            }

            $(#[$array_only])*
            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn srgb_to_linear_extended<T: $convert>(
                token: T,
                srgb: [f32; $width],
            ) -> [f32; $width] {
                srgb_to_linear_extended_vector(token, $float::from_array_t(token, srgb)).to_array()
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn srgb_to_linear_extended_vector<T: $convert>(
                token: T,
                v: $float<T>,
            ) -> $float<T> {
                srgb_to_linear_extended_vector_cfg::<T, true>(token, v)
            }

            #[inline(always)]
            pub(crate) fn srgb_to_linear_extended_vector_cfg<T: $convert, const FALLBACK: bool>(
                token: T,
                v: $float<T>,
            ) -> $float<T> {
                let (abs_v, result) = srgb_to_linear_extended_poly(token, v);
                if FALLBACK
                    && abs_v
                        .simd_gt($float::splat_t(token, 8.0))
                        .bitcast_to_i32()
                        .any_true()
                {
                    $float::from_array_t(
                        token,
                        srgb_to_linear_extended_fallback(v.to_array(), result.to_array()),
                    )
                } else {
                    result
                }
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn srgb_to_linear_extended_slice_cfg<T: $convert, const FALLBACK: bool>(
                token: T,
                values: &mut [f32],
            ) {
                if !FALLBACK {
                    #[cfg(not(target_arch = "x86_64"))]
                    fn tail<T: $convert>(token: T, chunk: [f32; $width]) -> [f32; $width] {
                        srgb_to_linear_extended_vector_cfg::<T, false>(
                            token,
                            $float::from_array_t(token, chunk),
                        )
                        .to_array()
                    }
                    crate::kernels::map_slice_inline!(
                        values,
                        $width,
                        token,
                        |chunk| {
                            srgb_to_linear_extended_vector_cfg::<T, false>(
                                token,
                                $float::from_array_t(token, chunk),
                            )
                            .to_array()
                        },
                        tail::<T>
                    );
                    return;
                }
                let (chunks, remainder) = values.as_chunks_mut::<$width>();
                let limit = $float::splat_t(token, 8.0);
                for slot in chunks {
                    let (abs_v, result) =
                        srgb_to_linear_extended_poly(token, $float::load_t(token, slot));
                    if abs_v.simd_gt(limit).bitcast_to_i32().any_true() {
                        let mut out = result.to_array();
                        srgb_to_linear_extended_patch(slot, &mut out);
                        *slot = out;
                    } else {
                        result.store(slot);
                    }
                }
                if !remainder.is_empty() {
                    #[cfg(target_arch = "x86_64")]
                    {
                        let mut block = [0.0; $width];
                        block[..remainder.len()].copy_from_slice(remainder);
                        let (abs_v, result) =
                            srgb_to_linear_extended_poly(token, $float::load_t(token, &block));
                        let mut out = result.to_array();
                        if abs_v.simd_gt(limit).bitcast_to_i32().any_true() {
                            srgb_to_linear_extended_patch(&block, &mut out);
                        }
                        remainder.copy_from_slice(&out[..remainder.len()]);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        fn tail<T: $convert>(token: T, chunk: [f32; $width]) -> [f32; $width] {
                            srgb_to_linear_extended_vector(token, $float::from_array_t(token, chunk))
                                .to_array()
                        }
                        crate::kernels::finish_tail_cold(remainder, token, tail::<T>);
                    }
                }
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn srgb_to_linear_extended_slice<T: $convert>(
                token: T,
                values: &mut [f32],
            ) {
                srgb_to_linear_extended_slice_cfg::<T, true>(token, values);
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn srgb_to_linear_extended_slice_into<T: $convert>(
                token: T,
                input: &[f32],
                output: &mut [f32],
            ) {
                assert_eq!(input.len(), output.len());
                let (in_chunks, in_rem) = input.as_chunks::<$width>();
                let (out_chunks, out_rem) = output.as_chunks_mut::<$width>();
                let limit = $float::splat_t(token, 8.0);
                for (src_slot, dst_slot) in in_chunks.iter().zip(out_chunks.iter_mut()) {
                    let (abs_v, result) =
                        srgb_to_linear_extended_poly(token, $float::load_t(token, src_slot));
                    result.store(dst_slot);
                    if abs_v.simd_gt(limit).bitcast_to_i32().any_true() {
                        srgb_to_linear_extended_patch(src_slot, dst_slot);
                    }
                }
                if !in_rem.is_empty() {
                    let mut src_block = [0.0; $width];
                    src_block[..in_rem.len()].copy_from_slice(in_rem);
                    let (abs_v, result) =
                        srgb_to_linear_extended_poly(token, $float::load_t(token, &src_block));
                    let mut out = result.to_array();
                    if abs_v.simd_gt(limit).bitcast_to_i32().any_true() {
                        srgb_to_linear_extended_patch(&src_block, &mut out);
                    }
                    out_rem.copy_from_slice(&out[..out_rem.len()]);
                }
            }

            #[inline(always)]
            pub(crate) fn linear_to_srgb_extended_poly<T: $convert>(
                token: T,
                v: $float<T>,
            ) -> ($float<T>, $float<T>) {
                use crate::rational_poly::{EXT_L2S_P as P, EXT_L2S_Q as Q};
                let sign = $float::splat_t(token, -0.0);
                let abs_v = v.abs();
                let linear_result = abs_v * $float::splat_t(token, TWELVE_92);
                let x = abs_v.fast_sqrt();
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
                (abs_v, result.abs() | (v & sign))
            }

            $(#[$array_only])*
            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn linear_to_srgb_extended<T: $convert>(
                token: T,
                linear: [f32; $width],
            ) -> [f32; $width] {
                linear_to_srgb_extended_vector(token, $float::from_array_t(token, linear))
                    .to_array()
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn linear_to_srgb_extended_vector<T: $convert>(
                token: T,
                v: $float<T>,
            ) -> $float<T> {
                linear_to_srgb_extended_vector_cfg::<T, true>(token, v)
            }

            #[inline(always)]
            pub(crate) fn linear_to_srgb_extended_vector_cfg<T: $convert, const FALLBACK: bool>(
                token: T,
                v: $float<T>,
            ) -> $float<T> {
                let (abs_v, result) = linear_to_srgb_extended_poly(token, v);
                if FALLBACK
                    && abs_v
                        .simd_gt($float::splat_t(token, 64.0))
                        .bitcast_to_i32()
                        .any_true()
                {
                    $float::from_array_t(
                        token,
                        linear_to_srgb_extended_fallback(token, v.to_array(), result.to_array()),
                    )
                } else {
                    result
                }
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn linear_to_srgb_extended_slice_cfg<T: $convert, const FALLBACK: bool>(
                token: T,
                values: &mut [f32],
            ) {
                if !FALLBACK {
                    #[cfg(not(target_arch = "x86_64"))]
                    fn tail<T: $convert>(token: T, chunk: [f32; $width]) -> [f32; $width] {
                        linear_to_srgb_extended_vector_cfg::<T, false>(
                            token,
                            $float::from_array_t(token, chunk),
                        )
                        .to_array()
                    }
                    crate::kernels::map_slice_inline!(
                        values,
                        $width,
                        token,
                        |chunk| {
                            linear_to_srgb_extended_vector_cfg::<T, false>(
                                token,
                                $float::from_array_t(token, chunk),
                            )
                            .to_array()
                        },
                        tail::<T>
                    );
                    return;
                }
                let (chunks, remainder) = values.as_chunks_mut::<$width>();
                let limit = $float::splat_t(token, 64.0);
                for slot in chunks {
                    let (abs_v, result) =
                        linear_to_srgb_extended_poly(token, $float::load_t(token, slot));
                    if abs_v.simd_gt(limit).bitcast_to_i32().any_true() {
                        let mut out = result.to_array();
                        linear_to_srgb_extended_patch(token, slot, &mut out);
                        *slot = out;
                    } else {
                        result.store(slot);
                    }
                }
                if !remainder.is_empty() {
                    #[cfg(target_arch = "x86_64")]
                    {
                        let mut block = [0.0; $width];
                        block[..remainder.len()].copy_from_slice(remainder);
                        let (abs_v, result) =
                            linear_to_srgb_extended_poly(token, $float::load_t(token, &block));
                        let mut out = result.to_array();
                        if abs_v.simd_gt(limit).bitcast_to_i32().any_true() {
                            linear_to_srgb_extended_patch(token, &block, &mut out);
                        }
                        remainder.copy_from_slice(&out[..remainder.len()]);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        fn tail<T: $convert>(token: T, chunk: [f32; $width]) -> [f32; $width] {
                            linear_to_srgb_extended_vector(token, $float::from_array_t(token, chunk))
                                .to_array()
                        }
                        crate::kernels::finish_tail_cold(remainder, token, tail::<T>);
                    }
                }
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn linear_to_srgb_extended_slice<T: $convert>(
                token: T,
                values: &mut [f32],
            ) {
                linear_to_srgb_extended_slice_cfg::<T, true>(token, values);
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub(crate) fn linear_to_srgb_extended_slice_into<T: $convert>(
                token: T,
                input: &[f32],
                output: &mut [f32],
            ) {
                assert_eq!(input.len(), output.len());
                let (in_chunks, in_rem) = input.as_chunks::<$width>();
                let (out_chunks, out_rem) = output.as_chunks_mut::<$width>();
                let limit = $float::splat_t(token, 64.0);
                for (src_slot, dst_slot) in in_chunks.iter().zip(out_chunks.iter_mut()) {
                    let (abs_v, result) =
                        linear_to_srgb_extended_poly(token, $float::load_t(token, src_slot));
                    result.store(dst_slot);
                    if abs_v.simd_gt(limit).bitcast_to_i32().any_true() {
                        linear_to_srgb_extended_patch(token, src_slot, dst_slot);
                    }
                }
                if !in_rem.is_empty() {
                    let mut src_block = [0.0; $width];
                    src_block[..in_rem.len()].copy_from_slice(in_rem);
                    let (abs_v, result) =
                        linear_to_srgb_extended_poly(token, $float::load_t(token, &src_block));
                    let mut out = result.to_array();
                    if abs_v.simd_gt(limit).bitcast_to_i32().any_true() {
                        linear_to_srgb_extended_patch(token, &src_block, &mut out);
                    }
                    out_rem.copy_from_slice(&out[..out_rem.len()]);
                }
            }
        }
    };
}

width!(
    x4,
    f32x4,
    F32x4Backend,
    F32x4Convert,
    4,
    gamma_x4,
    gamma_x4_cfg,
    gamma_slice_x4,
    gamma_slice_x4_cfg;
);
width!(
    #[cfg_attr(not(any(all(target_arch = "x86_64", target_feature = "sse2"), test)), allow(dead_code))]
    x8,
    f32x8,
    F32x8Backend,
    F32x8Convert,
    8,
    gamma_x8,
    gamma_x8_cfg,
    gamma_slice_x8,
    gamma_slice_x8_cfg;
);
// The x16 array adapters are used only by native entry points on x86_64.
width!(
    x16,
    f32x16,
    F32x16Backend,
    F32x16Convert,
    16,
    gamma_x16,
    gamma_x16_cfg,
    gamma_slice_x16,
    gamma_slice_x16_cfg;
    #[cfg(not(all(target_arch = "x86_64", not(target_feature = "sse2"))))]
);
