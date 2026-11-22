//! Internal benchmark & codegen-audit entry points exposing the crate-private
//! const-generic rail-ablated (`RAILS = false` / `CFG = 0`) kernels.
//!
//! Compiled **only** when `--cfg linear_srgb_bench_internal` is passed (`just asm-*`,
//! `just bench-rail-ablation`). Absent from all consumer builds and public API snapshots.

#![allow(dead_code)]

use crate::kernels::float::cfg;
use archmage::{attune, incant, magetypes};

macro_rules! unclamped_slice_dispatcher {
    ($pub_name:ident, $tier_name:ident, $x4_call:path, $x8_call:path, $x16_call:path) => {
        #[magetypes(v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        fn $tier_name(token: Token, values: &mut [f32]) {
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
            if let Some(t) = archmage::IntoConcreteToken::as_x64v4(token) {
                $x16_call(t, values);
                return;
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            if let Some(t) = archmage::IntoConcreteToken::as_x64v3(token) {
                $x8_call(t, values);
                return;
            }
            $x4_call(token, values);
        }

        #[inline]
        pub fn $pub_name(values: &mut [f32]) {
            incant!($tier_name(values), [v4, v3, neon, wasm128, scalar])
        }
    };
}

unclamped_slice_dispatcher!(
    srgb_to_linear_slice_unclamped,
    srgb_to_linear_slice_unclamped_tier,
    crate::kernels::float::x4::srgb_to_linear_slice_cfg::<_, { cfg::SRGB_IN_OPEN_UNIT }>,
    crate::kernels::float::x8::srgb_to_linear_slice_cfg::<_, { cfg::SRGB_IN_OPEN_UNIT }>,
    crate::kernels::float::x16::srgb_to_linear_slice_cfg::<_, { cfg::SRGB_IN_OPEN_UNIT }>
);

unclamped_slice_dispatcher!(
    srgb_to_linear_slice_unit_nonneg,
    srgb_to_linear_slice_unit_nonneg_tier,
    crate::kernels::float::x4::srgb_to_linear_slice_cfg::<_, { cfg::SRGB_UNIT_NONNEG }>,
    crate::kernels::float::x8::srgb_to_linear_slice_cfg::<_, { cfg::SRGB_UNIT_NONNEG }>,
    crate::kernels::float::x16::srgb_to_linear_slice_cfg::<_, { cfg::SRGB_UNIT_NONNEG }>
);

unclamped_slice_dispatcher!(
    linear_to_srgb_slice_unclamped,
    linear_to_srgb_slice_unclamped_tier,
    crate::kernels::float::x4::linear_to_srgb_slice_cfg::<_, { cfg::SRGB_IN_OPEN_UNIT }>,
    crate::kernels::float::x8::linear_to_srgb_slice_cfg::<_, { cfg::SRGB_IN_OPEN_UNIT }>,
    crate::kernels::float::x16::linear_to_srgb_slice_cfg::<_, { cfg::SRGB_IN_OPEN_UNIT }>
);

unclamped_slice_dispatcher!(
    srgb_to_linear_extended_slice_unclamped,
    srgb_to_linear_extended_slice_unclamped_tier,
    crate::kernels::float::x4::srgb_to_linear_extended_slice_cfg::<_, false>,
    crate::kernels::float::x8::srgb_to_linear_extended_slice_cfg::<_, false>,
    crate::kernels::float::x16::srgb_to_linear_extended_slice_cfg::<_, false>
);

unclamped_slice_dispatcher!(
    linear_to_srgb_extended_slice_unclamped,
    linear_to_srgb_extended_slice_unclamped_tier,
    crate::kernels::float::x4::linear_to_srgb_extended_slice_cfg::<_, false>,
    crate::kernels::float::x8::linear_to_srgb_extended_slice_cfg::<_, false>,
    crate::kernels::float::x16::linear_to_srgb_extended_slice_cfg::<_, false>
);

#[magetypes(v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn gamma_to_linear_slice_unclamped_tier(token: Token, values: &mut [f32], gamma: f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
    if let Some(t) = archmage::IntoConcreteToken::as_x64v4(token) {
        crate::kernels::float::x16::gamma_to_linear_slice_cfg::<_, false>(t, values, gamma);
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if let Some(t) = archmage::IntoConcreteToken::as_x64v3(token) {
        crate::kernels::float::x8::gamma_to_linear_slice_cfg::<_, false>(t, values, gamma);
        return;
    }
    crate::kernels::float::x4::gamma_to_linear_slice_cfg::<_, false>(token, values, gamma);
}

#[inline]
pub fn gamma_to_linear_slice_unclamped(values: &mut [f32], gamma: f32) {
    incant!(
        gamma_to_linear_slice_unclamped_tier(values, gamma),
        [v4, v3, neon, wasm128, scalar]
    )
}

#[magetypes(v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn linear_to_gamma_slice_unclamped_tier(token: Token, values: &mut [f32], gamma: f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
    if let Some(t) = archmage::IntoConcreteToken::as_x64v4(token) {
        crate::kernels::float::x16::linear_to_gamma_slice_cfg::<_, false>(t, values, gamma);
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if let Some(t) = archmage::IntoConcreteToken::as_x64v3(token) {
        crate::kernels::float::x8::linear_to_gamma_slice_cfg::<_, false>(t, values, gamma);
        return;
    }
    crate::kernels::float::x4::linear_to_gamma_slice_cfg::<_, false>(token, values, gamma);
}

#[inline]
pub fn linear_to_gamma_slice_unclamped(values: &mut [f32], gamma: f32) {
    incant!(
        linear_to_gamma_slice_unclamped_tier(values, gamma),
        [v4, v3, neon, wasm128, scalar]
    )
}

#[cfg(feature = "transfer")]
unclamped_slice_dispatcher!(
    bt709_to_linear_slice_unclamped,
    bt709_to_linear_slice_unclamped_tier,
    crate::tf::bt709::bt709_to_linear_slice_x4_cfg::<_, false>,
    crate::tf::bt709::bt709_to_linear_slice_x8_cfg::<_, false>,
    crate::tf::bt709::bt709_to_linear_slice_x16_cfg::<_, false>
);

#[cfg(feature = "transfer")]
unclamped_slice_dispatcher!(
    linear_to_bt709_slice_unclamped,
    linear_to_bt709_slice_unclamped_tier,
    crate::tf::bt709::linear_to_bt709_slice_x4_cfg::<_, false>,
    crate::tf::bt709::linear_to_bt709_slice_x8_cfg::<_, false>,
    crate::tf::bt709::linear_to_bt709_slice_x16_cfg::<_, false>
);

#[cfg(feature = "transfer")]
unclamped_slice_dispatcher!(
    pq_to_linear_slice_unclamped,
    pq_to_linear_slice_unclamped_tier,
    crate::tf::pq::pq_to_linear_slice_x4_cfg::<_, false>,
    crate::tf::pq::pq_to_linear_slice_x8_cfg::<_, false>,
    crate::tf::pq::pq_to_linear_slice_x16_cfg::<_, false>
);

#[cfg(feature = "transfer")]
unclamped_slice_dispatcher!(
    linear_to_pq_slice_unclamped,
    linear_to_pq_slice_unclamped_tier,
    crate::tf::pq::linear_to_pq_slice_x4_cfg::<_, false>,
    crate::tf::pq::linear_to_pq_slice_x8_cfg::<_, false>,
    crate::tf::pq::linear_to_pq_slice_x16_cfg::<_, false>
);

#[cfg(feature = "transfer")]
unclamped_slice_dispatcher!(
    hlg_to_linear_slice_unclamped,
    hlg_to_linear_slice_unclamped_tier,
    crate::tf::hlg::hlg_to_linear_slice_x4_cfg::<_, false>,
    crate::tf::hlg::hlg_to_linear_slice_x8_cfg::<_, false>,
    crate::tf::hlg::hlg_to_linear_slice_x16_cfg::<_, false>
);

#[cfg(feature = "transfer")]
unclamped_slice_dispatcher!(
    linear_to_hlg_slice_unclamped,
    linear_to_hlg_slice_unclamped_tier,
    crate::tf::hlg::linear_to_hlg_slice_x4_cfg::<_, false>,
    crate::tf::hlg::linear_to_hlg_slice_x8_cfg::<_, false>,
    crate::tf::hlg::linear_to_hlg_slice_x16_cfg::<_, false>
);

// ============================================================================
// In-register SIMD array kernels (`RAILS = false` / `SRGB_IN_OPEN_UNIT`)
// ============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
pub mod x8 {
    use super::*;
    use archmage::X64V3Token;
    use magetypes::simd::generic::f32x8;

    #[attune(v3, inline(hint))]
    pub fn srgb_to_linear_v3_unclamped(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        crate::kernels::float::x8::srgb_to_linear_cfg::<_, { cfg::SRGB_IN_OPEN_UNIT }>(t, v)
    }

    #[attune(v3, inline(hint))]
    pub fn srgb_to_linear_v3_unit_nonneg(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        crate::kernels::float::x8::srgb_to_linear_cfg::<_, { cfg::SRGB_UNIT_NONNEG }>(t, v)
    }

    #[attune(v3, inline(hint))]
    pub fn linear_to_srgb_v3_unclamped(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        crate::kernels::float::x8::linear_to_srgb_cfg::<_, { cfg::SRGB_IN_OPEN_UNIT }>(t, v)
    }

    #[attune(v3, inline(hint))]
    pub fn gamma_to_linear_v3_unclamped(t: X64V3Token, v: [f32; 8], g: f32) -> [f32; 8] {
        crate::kernels::float::x8::gamma_to_linear_cfg::<_, false>(t, v, g)
    }

    #[attune(v3, inline(hint))]
    pub fn linear_to_gamma_v3_unclamped(t: X64V3Token, v: [f32; 8], g: f32) -> [f32; 8] {
        crate::kernels::float::x8::linear_to_gamma_cfg::<_, false>(t, v, g)
    }

    #[cfg(feature = "transfer")]
    #[attune(v3, inline(hint))]
    pub fn bt709_to_linear_v3_unclamped(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        crate::tf::bt709::bt709_to_linear_x8_cfg::<_, false>(t, f32x8::from_array_t(t, v))
            .to_array()
    }

    #[cfg(feature = "transfer")]
    #[attune(v3, inline(hint))]
    pub fn linear_to_bt709_v3_unclamped(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        crate::tf::bt709::linear_to_bt709_x8_cfg::<_, false>(t, f32x8::from_array_t(t, v))
            .to_array()
    }

    #[cfg(feature = "transfer")]
    #[attune(v3, inline(hint))]
    pub fn pq_to_linear_v3_unclamped(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        crate::tf::pq::pq_to_linear_x8_cfg::<_, false>(t, f32x8::from_array_t(t, v)).to_array()
    }

    #[cfg(feature = "transfer")]
    #[attune(v3, inline(hint))]
    pub fn linear_to_pq_v3_unclamped(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        crate::tf::pq::linear_to_pq_x8_cfg::<_, false>(t, f32x8::from_array_t(t, v)).to_array()
    }

    #[cfg(feature = "transfer")]
    #[attune(v3, inline(hint))]
    pub fn hlg_to_linear_v3_unclamped(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        crate::tf::hlg::hlg_to_linear_x8_cfg::<_, false>(t, f32x8::from_array_t(t, v)).to_array()
    }

    #[cfg(feature = "transfer")]
    #[attune(v3, inline(hint))]
    pub fn linear_to_hlg_v3_unclamped(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        crate::tf::hlg::linear_to_hlg_x8_cfg::<_, false>(t, f32x8::from_array_t(t, v)).to_array()
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
pub mod x16 {
    use super::*;
    use archmage::X64V4Token;
    use magetypes::simd::generic::f32x16;

    #[attune(v4, inline(hint))]
    pub fn srgb_to_linear_v4_unclamped(t: X64V4Token, v: [f32; 16]) -> [f32; 16] {
        crate::kernels::float::x16::srgb_to_linear_cfg::<_, { cfg::SRGB_IN_OPEN_UNIT }>(t, v)
    }

    #[attune(v4, inline(hint))]
    pub fn linear_to_srgb_v4_unclamped(t: X64V4Token, v: [f32; 16]) -> [f32; 16] {
        crate::kernels::float::x16::linear_to_srgb_cfg::<_, { cfg::SRGB_IN_OPEN_UNIT }>(t, v)
    }

    #[attune(v4, inline(hint))]
    pub fn gamma_to_linear_v4_unclamped(t: X64V4Token, v: [f32; 16], g: f32) -> [f32; 16] {
        crate::kernels::float::x16::gamma_to_linear_cfg::<_, false>(t, v, g)
    }

    #[cfg(feature = "transfer")]
    #[attune(v4, inline(hint))]
    pub fn bt709_to_linear_v4_unclamped(t: X64V4Token, v: [f32; 16]) -> [f32; 16] {
        crate::tf::bt709::bt709_to_linear_x16_cfg::<_, false>(t, f32x16::from_array_t(t, v))
            .to_array()
    }

    #[cfg(feature = "transfer")]
    #[attune(v4, inline(hint))]
    pub fn linear_to_bt709_v4_unclamped(t: X64V4Token, v: [f32; 16]) -> [f32; 16] {
        crate::tf::bt709::linear_to_bt709_x16_cfg::<_, false>(t, f32x16::from_array_t(t, v))
            .to_array()
    }

    #[cfg(feature = "transfer")]
    #[attune(v4, inline(hint))]
    pub fn pq_to_linear_v4_unclamped(t: X64V4Token, v: [f32; 16]) -> [f32; 16] {
        crate::tf::pq::pq_to_linear_x16_cfg::<_, false>(t, f32x16::from_array_t(t, v)).to_array()
    }

    #[cfg(feature = "transfer")]
    #[attune(v4, inline(hint))]
    pub fn linear_to_pq_v4_unclamped(t: X64V4Token, v: [f32; 16]) -> [f32; 16] {
        crate::tf::pq::linear_to_pq_x16_cfg::<_, false>(t, f32x16::from_array_t(t, v)).to_array()
    }

    #[cfg(feature = "transfer")]
    #[attune(v4, inline(hint))]
    pub fn hlg_to_linear_v4_unclamped(t: X64V4Token, v: [f32; 16]) -> [f32; 16] {
        crate::tf::hlg::hlg_to_linear_x16_cfg::<_, false>(t, f32x16::from_array_t(t, v)).to_array()
    }

    #[cfg(feature = "transfer")]
    #[attune(v4, inline(hint))]
    pub fn linear_to_hlg_v4_unclamped(t: X64V4Token, v: [f32; 16]) -> [f32; 16] {
        crate::tf::hlg::linear_to_hlg_x16_cfg::<_, false>(t, f32x16::from_array_t(t, v)).to_array()
    }
}

pub mod x4 {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use magetypes::simd::generic::f32x4;

    #[cfg(target_arch = "aarch64")]
    #[attune(neon, inline(hint))]
    pub fn srgb_to_linear_neon_unclamped(t: archmage::NeonToken, v: [f32; 4]) -> [f32; 4] {
        crate::kernels::float::x4::srgb_to_linear_cfg::<_, { cfg::SRGB_IN_OPEN_UNIT }>(t, v)
    }
    #[cfg(target_arch = "aarch64")]
    #[attune(neon, inline(hint))]
    pub fn linear_to_srgb_neon_unclamped(t: archmage::NeonToken, v: [f32; 4]) -> [f32; 4] {
        crate::kernels::float::x4::linear_to_srgb_cfg::<_, { cfg::SRGB_IN_OPEN_UNIT }>(t, v)
    }
    #[cfg(target_arch = "aarch64")]
    #[attune(neon, inline(hint))]
    pub fn gamma_to_linear_neon_unclamped(t: archmage::NeonToken, v: [f32; 4], g: f32) -> [f32; 4] {
        crate::kernels::float::x4::gamma_to_linear_cfg::<_, false>(t, v, g)
    }
    #[cfg(all(target_arch = "aarch64", feature = "transfer"))]
    #[attune(neon, inline(hint))]
    pub fn bt709_to_linear_neon_unclamped(t: archmage::NeonToken, v: [f32; 4]) -> [f32; 4] {
        crate::tf::bt709::bt709_to_linear_x4_cfg::<_, false>(t, f32x4::from_array_t(t, v))
            .to_array()
    }
    #[cfg(all(target_arch = "aarch64", feature = "transfer"))]
    #[attune(neon, inline(hint))]
    pub fn linear_to_bt709_neon_unclamped(t: archmage::NeonToken, v: [f32; 4]) -> [f32; 4] {
        crate::tf::bt709::linear_to_bt709_x4_cfg::<_, false>(t, f32x4::from_array_t(t, v))
            .to_array()
    }
    #[cfg(all(target_arch = "aarch64", feature = "transfer"))]
    #[attune(neon, inline(hint))]
    pub fn pq_to_linear_neon_unclamped(t: archmage::NeonToken, v: [f32; 4]) -> [f32; 4] {
        crate::tf::pq::pq_to_linear_x4_cfg::<_, false>(t, f32x4::from_array_t(t, v)).to_array()
    }
    #[cfg(all(target_arch = "aarch64", feature = "transfer"))]
    #[attune(neon, inline(hint))]
    pub fn linear_to_pq_neon_unclamped(t: archmage::NeonToken, v: [f32; 4]) -> [f32; 4] {
        crate::tf::pq::linear_to_pq_x4_cfg::<_, false>(t, f32x4::from_array_t(t, v)).to_array()
    }
    #[cfg(all(target_arch = "aarch64", feature = "transfer"))]
    #[attune(neon, inline(hint))]
    pub fn hlg_to_linear_neon_unclamped(t: archmage::NeonToken, v: [f32; 4]) -> [f32; 4] {
        crate::tf::hlg::hlg_to_linear_x4_cfg::<_, false>(t, f32x4::from_array_t(t, v)).to_array()
    }
    #[cfg(all(target_arch = "aarch64", feature = "transfer"))]
    #[attune(neon, inline(hint))]
    pub fn linear_to_hlg_neon_unclamped(t: archmage::NeonToken, v: [f32; 4]) -> [f32; 4] {
        crate::tf::hlg::linear_to_hlg_x4_cfg::<_, false>(t, f32x4::from_array_t(t, v)).to_array()
    }

    #[cfg(target_arch = "wasm32")]
    #[attune(wasm128, inline(hint))]
    pub fn srgb_to_linear_wasm128_unclamped(t: archmage::Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        crate::kernels::float::x4::srgb_to_linear_cfg::<_, { cfg::SRGB_IN_OPEN_UNIT }>(t, v)
    }
    #[cfg(target_arch = "wasm32")]
    #[attune(wasm128, inline(hint))]
    pub fn linear_to_srgb_wasm128_unclamped(t: archmage::Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        crate::kernels::float::x4::linear_to_srgb_cfg::<_, { cfg::SRGB_IN_OPEN_UNIT }>(t, v)
    }
    #[cfg(target_arch = "wasm32")]
    #[attune(wasm128, inline(hint))]
    pub fn gamma_to_linear_wasm128_unclamped(
        t: archmage::Wasm128Token,
        v: [f32; 4],
        g: f32,
    ) -> [f32; 4] {
        crate::kernels::float::x4::gamma_to_linear_cfg::<_, false>(t, v, g)
    }
    #[cfg(all(target_arch = "wasm32", feature = "transfer"))]
    #[attune(wasm128, inline(hint))]
    pub fn bt709_to_linear_wasm128_unclamped(t: archmage::Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        crate::tf::bt709::bt709_to_linear_x4_cfg::<_, false>(t, f32x4::from_array_t(t, v))
            .to_array()
    }
    #[cfg(all(target_arch = "wasm32", feature = "transfer"))]
    #[attune(wasm128, inline(hint))]
    pub fn linear_to_bt709_wasm128_unclamped(t: archmage::Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        crate::tf::bt709::linear_to_bt709_x4_cfg::<_, false>(t, f32x4::from_array_t(t, v))
            .to_array()
    }
    #[cfg(all(target_arch = "wasm32", feature = "transfer"))]
    #[attune(wasm128, inline(hint))]
    pub fn pq_to_linear_wasm128_unclamped(t: archmage::Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        crate::tf::pq::pq_to_linear_x4_cfg::<_, false>(t, f32x4::from_array_t(t, v)).to_array()
    }
    #[cfg(all(target_arch = "wasm32", feature = "transfer"))]
    #[attune(wasm128, inline(hint))]
    pub fn linear_to_pq_wasm128_unclamped(t: archmage::Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        crate::tf::pq::linear_to_pq_x4_cfg::<_, false>(t, f32x4::from_array_t(t, v)).to_array()
    }
    #[cfg(all(target_arch = "wasm32", feature = "transfer"))]
    #[attune(wasm128, inline(hint))]
    pub fn hlg_to_linear_wasm128_unclamped(t: archmage::Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        crate::tf::hlg::hlg_to_linear_x4_cfg::<_, false>(t, f32x4::from_array_t(t, v)).to_array()
    }
    #[cfg(all(target_arch = "wasm32", feature = "transfer"))]
    #[attune(wasm128, inline(hint))]
    pub fn linear_to_hlg_wasm128_unclamped(t: archmage::Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        crate::tf::hlg::linear_to_hlg_x4_cfg::<_, false>(t, f32x4::from_array_t(t, v)).to_array()
    }
}
