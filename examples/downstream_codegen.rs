//! Versioned codegen probe exercising every `linear-srgb` function used by
//! downstream crates across the zen workspace (`zenpixels-convert`, `zenresize`,
//! `zenpng`, `zenjpeg`, `zenavif`, `zengif`, `zentone`, `zenanalyze`, `zensim`,
//! and `imageflow`).
//!
//! Inspect assembly across all 5 supported SIMD tiers via `just`:
//! - `just asm-v3 ds_srgb_to_linear_slice`
//! - `just asm-v4 ds_srgb_to_linear_slice`
//! - `just asm-neon ds_srgb_to_linear_slice`
//! - `just asm-wasm128 ds_srgb_to_linear_slice`
//! - `just asm-wasm128-relaxed ds_srgb_to_linear_slice`
//! - `just asm-all ds_srgb_to_linear_slice`

#![allow(deprecated)]

use core::hint::black_box;
use linear_srgb::{default, precise};

// ============================================================================
// 1. f32 sRGB slices (clamped + extended + RGBA + premultiply)
// ============================================================================

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_srgb_to_linear_slice(buf: &mut [f32]) {
    default::srgb_to_linear_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_linear_to_srgb_slice(buf: &mut [f32]) {
    default::linear_to_srgb_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_srgb_to_linear_rgba_slice(buf: &mut [f32]) {
    default::srgb_to_linear_rgba_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_linear_to_srgb_rgba_slice(buf: &mut [f32]) {
    default::linear_to_srgb_rgba_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_srgb_to_linear_extended_slice(buf: &mut [f32]) {
    default::srgb_to_linear_extended_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_linear_to_srgb_extended_slice(buf: &mut [f32]) {
    default::linear_to_srgb_extended_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_srgb_to_linear_premultiply_rgba_slice(buf: &mut [f32]) {
    default::srgb_to_linear_premultiply_rgba_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_unpremultiply_linear_to_srgb_rgba_slice(buf: &mut [f32]) {
    default::unpremultiply_linear_to_srgb_rgba_slice(buf);
}

// ============================================================================
// 2. Integer u8 / u16 sRGB slices + fused u8 premultiply
// ============================================================================

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_srgb_u8_to_linear_slice(src: &[u8], dst: &mut [f32]) {
    default::srgb_u8_to_linear_slice(src, dst);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_srgb_u8_to_linear_rgba_slice(src: &[u8], dst: &mut [f32]) {
    default::srgb_u8_to_linear_rgba_slice(src, dst);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_linear_to_srgb_u8_slice(src: &[f32], dst: &mut [u8]) {
    default::linear_to_srgb_u8_slice(src, dst);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_linear_to_srgb_u8_rgba_slice(src: &[f32], dst: &mut [u8]) {
    default::linear_to_srgb_u8_rgba_slice(src, dst);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_srgb_u8_to_linear_premultiply_rgba_slice(src: &[u8], dst: &mut [f32]) {
    default::srgb_u8_to_linear_premultiply_rgba_slice(src, dst);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_unpremultiply_linear_to_srgb_u8_rgba_slice(src: &[f32], dst: &mut [u8]) {
    default::unpremultiply_linear_to_srgb_u8_rgba_slice(src, dst);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_srgb_u16_to_linear_slice(src: &[u16], dst: &mut [f32]) {
    default::srgb_u16_to_linear_slice(src, dst);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_srgb_u16_to_linear_rgba_slice(src: &[u16], dst: &mut [f32]) {
    default::srgb_u16_to_linear_rgba_slice(src, dst);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_linear_to_srgb_u16_slice(src: &[f32], dst: &mut [u16]) {
    default::linear_to_srgb_u16_slice(src, dst);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_linear_to_srgb_u16_slice_fast(src: &[f32], dst: &mut [u16]) {
    default::linear_to_srgb_u16_slice_fast(src, dst);
}

// ============================================================================
// 3. Custom gamma slices (runtime gamma + constant 2.2)
// ============================================================================

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_gamma_to_linear_slice(buf: &mut [f32], gamma: f32) {
    default::gamma_to_linear_slice(buf, gamma);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_linear_to_gamma_slice(buf: &mut [f32], gamma: f32) {
    default::linear_to_gamma_slice(buf, gamma);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_gamma22_to_linear_slice(buf: &mut [f32]) {
    default::gamma_to_linear_slice(buf, 2.2);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_linear_to_gamma22_slice(buf: &mut [f32]) {
    default::linear_to_gamma_slice(buf, 2.2);
}

// ============================================================================
// 4. HDR / broadcast transfer slices (BT.709, PQ, HLG)
// ============================================================================

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_bt709_to_linear_slice(buf: &mut [f32]) {
    default::bt709_to_linear_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_linear_to_bt709_slice(buf: &mut [f32]) {
    default::linear_to_bt709_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_pq_to_linear_slice(buf: &mut [f32]) {
    default::pq_to_linear_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_linear_to_pq_slice(buf: &mut [f32]) {
    default::linear_to_pq_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_hlg_to_linear_slice(buf: &mut [f32]) {
    default::hlg_to_linear_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_linear_to_hlg_slice(buf: &mut [f32]) {
    default::linear_to_hlg_slice(buf);
}

// ============================================================================
// 5. In-register SIMD array kernels (`zenpixels-convert::fast_gamut`, `zenjpeg`)
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod arr_impl {
    use archmage::{SimdToken, X64V3Token, arcane};
    use linear_srgb::tokens::x8;

    #[arcane]
    fn i_s2l(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        x8::srgb_to_linear_v3(t, v)
    }
    #[arcane]
    fn i_l2s(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        x8::linear_to_srgb_v3(t, v)
    }
    #[arcane]
    fn i_u16_s2l(t: X64V3Token, v: [u16; 8]) -> [f32; 8] {
        x8::srgb_u16_to_linear_v3(t, v)
    }
    #[arcane]
    fn i_l2s_u16(t: X64V3Token, v: [f32; 8]) -> [u16; 8] {
        x8::linear_to_srgb_u16_v3(t, v)
    }
    #[arcane]
    fn i_g2l(t: X64V3Token, v: [f32; 8], g: f32) -> [f32; 8] {
        x8::gamma_to_linear_v3(t, v, g)
    }
    #[arcane]
    fn i_bt_d(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        x8::bt709_to_linear_v3(t, v)
    }
    #[arcane]
    fn i_bt_e(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        x8::linear_to_bt709_v3(t, v)
    }
    #[arcane]
    fn i_pq_d(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        x8::pq_to_linear_v3(t, v)
    }
    #[arcane]
    fn i_pq_e(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        x8::linear_to_pq_v3(t, v)
    }
    #[arcane]
    fn i_hlg_d(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        x8::hlg_to_linear_v3(t, v)
    }
    #[arcane]
    fn i_hlg_e(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
        x8::linear_to_hlg_v3(t, v)
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_srgb_to_linear(v: [f32; 8]) -> [f32; 8] {
        i_s2l(X64V3Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_linear_to_srgb(v: [f32; 8]) -> [f32; 8] {
        i_l2s(X64V3Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_srgb_u16_to_linear(v: [u16; 8]) -> [f32; 8] {
        i_u16_s2l(X64V3Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_linear_to_srgb_u16(v: [f32; 8]) -> [u16; 8] {
        i_l2s_u16(X64V3Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_gamma_to_linear(v: [f32; 8], g: f32) -> [f32; 8] {
        i_g2l(X64V3Token::summon().unwrap(), v, g)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_bt709_to_linear(v: [f32; 8]) -> [f32; 8] {
        i_bt_d(X64V3Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_linear_to_bt709(v: [f32; 8]) -> [f32; 8] {
        i_bt_e(X64V3Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_pq_to_linear(v: [f32; 8]) -> [f32; 8] {
        i_pq_d(X64V3Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_linear_to_pq(v: [f32; 8]) -> [f32; 8] {
        i_pq_e(X64V3Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_hlg_to_linear(v: [f32; 8]) -> [f32; 8] {
        i_hlg_d(X64V3Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_linear_to_hlg(v: [f32; 8]) -> [f32; 8] {
        i_hlg_e(X64V3Token::summon().unwrap(), v)
    }
}

#[cfg(target_arch = "aarch64")]
mod arr_impl {
    use archmage::{NeonToken, SimdToken, arcane};
    use linear_srgb::tokens::x4;

    #[arcane]
    fn i_s2l(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
        x4::srgb_to_linear_neon(t, v)
    }
    #[arcane]
    fn i_l2s(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
        x4::linear_to_srgb_neon(t, v)
    }
    #[arcane]
    fn i_g2l(t: NeonToken, v: [f32; 4], g: f32) -> [f32; 4] {
        x4::gamma_to_linear_neon(t, v, g)
    }
    #[arcane]
    fn i_bt_d(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
        x4::bt709_to_linear_neon(t, v)
    }
    #[arcane]
    fn i_bt_e(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
        x4::linear_to_bt709_neon(t, v)
    }
    #[arcane]
    fn i_pq_d(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
        x4::pq_to_linear_neon(t, v)
    }
    #[arcane]
    fn i_pq_e(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
        x4::linear_to_pq_neon(t, v)
    }
    #[arcane]
    fn i_hlg_d(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
        x4::hlg_to_linear_neon(t, v)
    }
    #[arcane]
    fn i_hlg_e(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
        x4::linear_to_hlg_neon(t, v)
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_srgb_to_linear(v: [f32; 4]) -> [f32; 4] {
        i_s2l(NeonToken::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_linear_to_srgb(v: [f32; 4]) -> [f32; 4] {
        i_l2s(NeonToken::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_gamma_to_linear(v: [f32; 4], g: f32) -> [f32; 4] {
        i_g2l(NeonToken::summon().unwrap(), v, g)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_bt709_to_linear(v: [f32; 4]) -> [f32; 4] {
        i_bt_d(NeonToken::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_linear_to_bt709(v: [f32; 4]) -> [f32; 4] {
        i_bt_e(NeonToken::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_pq_to_linear(v: [f32; 4]) -> [f32; 4] {
        i_pq_d(NeonToken::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_linear_to_pq(v: [f32; 4]) -> [f32; 4] {
        i_pq_e(NeonToken::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_hlg_to_linear(v: [f32; 4]) -> [f32; 4] {
        i_hlg_d(NeonToken::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_linear_to_hlg(v: [f32; 4]) -> [f32; 4] {
        i_hlg_e(NeonToken::summon().unwrap(), v)
    }
}

#[cfg(target_arch = "wasm32")]
mod arr_impl {
    use archmage::{SimdToken, Wasm128Token, arcane};
    use linear_srgb::tokens::x4;

    #[arcane]
    fn i_s2l(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        x4::srgb_to_linear_wasm128(t, v)
    }
    #[arcane]
    fn i_l2s(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        x4::linear_to_srgb_wasm128(t, v)
    }
    #[arcane]
    fn i_g2l(t: Wasm128Token, v: [f32; 4], g: f32) -> [f32; 4] {
        x4::gamma_to_linear_wasm128(t, v, g)
    }
    #[arcane]
    fn i_bt_d(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        x4::bt709_to_linear_wasm128(t, v)
    }
    #[arcane]
    fn i_bt_e(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        x4::linear_to_bt709_wasm128(t, v)
    }
    #[arcane]
    fn i_pq_d(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        x4::pq_to_linear_wasm128(t, v)
    }
    #[arcane]
    fn i_pq_e(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        x4::linear_to_pq_wasm128(t, v)
    }
    #[arcane]
    fn i_hlg_d(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        x4::hlg_to_linear_wasm128(t, v)
    }
    #[arcane]
    fn i_hlg_e(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
        x4::linear_to_hlg_wasm128(t, v)
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_srgb_to_linear(v: [f32; 4]) -> [f32; 4] {
        i_s2l(Wasm128Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_linear_to_srgb(v: [f32; 4]) -> [f32; 4] {
        i_l2s(Wasm128Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_gamma_to_linear(v: [f32; 4], g: f32) -> [f32; 4] {
        i_g2l(Wasm128Token::summon().unwrap(), v, g)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_bt709_to_linear(v: [f32; 4]) -> [f32; 4] {
        i_bt_d(Wasm128Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_linear_to_bt709(v: [f32; 4]) -> [f32; 4] {
        i_bt_e(Wasm128Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_pq_to_linear(v: [f32; 4]) -> [f32; 4] {
        i_pq_d(Wasm128Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_linear_to_pq(v: [f32; 4]) -> [f32; 4] {
        i_pq_e(Wasm128Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_hlg_to_linear(v: [f32; 4]) -> [f32; 4] {
        i_hlg_d(Wasm128Token::summon().unwrap(), v)
    }
    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_arr_linear_to_hlg(v: [f32; 4]) -> [f32; 4] {
        i_hlg_e(Wasm128Token::summon().unwrap(), v)
    }
}

// ============================================================================
// 6. Scalar entry points
// ============================================================================

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_scalar_srgb_to_linear(v: f32) -> f32 {
    default::srgb_to_linear(v)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_scalar_linear_to_srgb(v: f32) -> f32 {
    default::linear_to_srgb(v)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_scalar_bt709_to_linear(v: f32) -> f32 {
    default::bt709_to_linear(v)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_scalar_linear_to_bt709(v: f32) -> f32 {
    default::linear_to_bt709(v)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_scalar_pq_to_linear(v: f32) -> f32 {
    default::pq_to_linear(v)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_scalar_linear_to_pq(v: f32) -> f32 {
    default::linear_to_pq(v)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_scalar_hlg_to_linear(v: f32) -> f32 {
    default::hlg_to_linear(v)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_scalar_linear_to_hlg(v: f32) -> f32 {
    default::linear_to_hlg(v)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_scalar_gamma_to_linear(v: f32, g: f32) -> f32 {
    default::gamma_to_linear(v, g)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn ds_scalar_precise_srgb_to_linear(v: f32) -> f32 {
    precise::srgb_to_linear(v)
}

// ============================================================================
// 7. Const-generic rail-ablated (`RAILS = false` / `CFG = 0`) slices & arrays
// ============================================================================

#[cfg(linear_srgb_bench_internal)]
mod unclamped_impl {
    use core::hint::black_box;
    use linear_srgb::__bench_rails as br;

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_unclamped_srgb_to_linear_slice(buf: &mut [f32]) {
        br::srgb_to_linear_slice_unclamped(buf);
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_unclamped_linear_to_srgb_slice(buf: &mut [f32]) {
        br::linear_to_srgb_slice_unclamped(buf);
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_unclamped_srgb_to_linear_extended_slice(buf: &mut [f32]) {
        br::srgb_to_linear_extended_slice_unclamped(buf);
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_unclamped_linear_to_srgb_extended_slice(buf: &mut [f32]) {
        br::linear_to_srgb_extended_slice_unclamped(buf);
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_unclamped_gamma_to_linear_slice(buf: &mut [f32], gamma: f32) {
        br::gamma_to_linear_slice_unclamped(buf, gamma);
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_unclamped_linear_to_gamma_slice(buf: &mut [f32], gamma: f32) {
        br::linear_to_gamma_slice_unclamped(buf, gamma);
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_unclamped_bt709_to_linear_slice(buf: &mut [f32]) {
        br::bt709_to_linear_slice_unclamped(buf);
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_unclamped_linear_to_bt709_slice(buf: &mut [f32]) {
        br::linear_to_bt709_slice_unclamped(buf);
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_unclamped_pq_to_linear_slice(buf: &mut [f32]) {
        br::pq_to_linear_slice_unclamped(buf);
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_unclamped_linear_to_pq_slice(buf: &mut [f32]) {
        br::linear_to_pq_slice_unclamped(buf);
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_unclamped_hlg_to_linear_slice(buf: &mut [f32]) {
        br::hlg_to_linear_slice_unclamped(buf);
    }

    #[inline(never)]
    #[unsafe(no_mangle)]
    pub fn ds_unclamped_linear_to_hlg_slice(buf: &mut [f32]) {
        br::linear_to_hlg_slice_unclamped(buf);
    }

    #[cfg(target_arch = "x86_64")]
    pub mod arr {
        use archmage::{SimdToken, X64V3Token, arcane};
        use linear_srgb::__bench_rails::x8;

        #[arcane]
        fn i_s2l(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            x8::srgb_to_linear_v3_unclamped(t, v)
        }
        #[arcane]
        fn i_l2s(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            x8::linear_to_srgb_v3_unclamped(t, v)
        }
        #[arcane]
        fn i_g2l(t: X64V3Token, v: [f32; 8], g: f32) -> [f32; 8] {
            x8::gamma_to_linear_v3_unclamped(t, v, g)
        }
        #[arcane]
        fn i_bt_d(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            x8::bt709_to_linear_v3_unclamped(t, v)
        }
        #[arcane]
        fn i_bt_e(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            x8::linear_to_bt709_v3_unclamped(t, v)
        }
        #[arcane]
        fn i_pq_d(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            x8::pq_to_linear_v3_unclamped(t, v)
        }
        #[arcane]
        fn i_pq_e(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            x8::linear_to_pq_v3_unclamped(t, v)
        }
        #[arcane]
        fn i_hlg_d(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            x8::hlg_to_linear_v3_unclamped(t, v)
        }
        #[arcane]
        fn i_hlg_e(t: X64V3Token, v: [f32; 8]) -> [f32; 8] {
            x8::linear_to_hlg_v3_unclamped(t, v)
        }

        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_srgb_to_linear(v: [f32; 8]) -> [f32; 8] {
            i_s2l(X64V3Token::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_linear_to_srgb(v: [f32; 8]) -> [f32; 8] {
            i_l2s(X64V3Token::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_gamma_to_linear(v: [f32; 8], g: f32) -> [f32; 8] {
            i_g2l(X64V3Token::summon().unwrap(), v, g)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_bt709_to_linear(v: [f32; 8]) -> [f32; 8] {
            i_bt_d(X64V3Token::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_linear_to_bt709(v: [f32; 8]) -> [f32; 8] {
            i_bt_e(X64V3Token::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_pq_to_linear(v: [f32; 8]) -> [f32; 8] {
            i_pq_d(X64V3Token::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_linear_to_pq(v: [f32; 8]) -> [f32; 8] {
            i_pq_e(X64V3Token::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_hlg_to_linear(v: [f32; 8]) -> [f32; 8] {
            i_hlg_d(X64V3Token::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_linear_to_hlg(v: [f32; 8]) -> [f32; 8] {
            i_hlg_e(X64V3Token::summon().unwrap(), v)
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub mod arr {
        use archmage::{NeonToken, SimdToken, arcane};
        use linear_srgb::__bench_rails::x4;

        #[arcane]
        fn i_s2l(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
            x4::srgb_to_linear_neon_unclamped(t, v)
        }
        #[arcane]
        fn i_l2s(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
            x4::linear_to_srgb_neon_unclamped(t, v)
        }
        #[arcane]
        fn i_g2l(t: NeonToken, v: [f32; 4], g: f32) -> [f32; 4] {
            x4::gamma_to_linear_neon_unclamped(t, v, g)
        }
        #[arcane]
        fn i_bt_d(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
            x4::bt709_to_linear_neon_unclamped(t, v)
        }
        #[arcane]
        fn i_bt_e(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
            x4::linear_to_bt709_neon_unclamped(t, v)
        }
        #[arcane]
        fn i_pq_d(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
            x4::pq_to_linear_neon_unclamped(t, v)
        }
        #[arcane]
        fn i_pq_e(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
            x4::linear_to_pq_neon_unclamped(t, v)
        }
        #[arcane]
        fn i_hlg_d(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
            x4::hlg_to_linear_neon_unclamped(t, v)
        }
        #[arcane]
        fn i_hlg_e(t: NeonToken, v: [f32; 4]) -> [f32; 4] {
            x4::linear_to_hlg_neon_unclamped(t, v)
        }

        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_srgb_to_linear(v: [f32; 4]) -> [f32; 4] {
            i_s2l(NeonToken::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_linear_to_srgb(v: [f32; 4]) -> [f32; 4] {
            i_l2s(NeonToken::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_gamma_to_linear(v: [f32; 4], g: f32) -> [f32; 4] {
            i_g2l(NeonToken::summon().unwrap(), v, g)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_bt709_to_linear(v: [f32; 4]) -> [f32; 4] {
            i_bt_d(NeonToken::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_linear_to_bt709(v: [f32; 4]) -> [f32; 4] {
            i_bt_e(NeonToken::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_pq_to_linear(v: [f32; 4]) -> [f32; 4] {
            i_pq_d(NeonToken::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_linear_to_pq(v: [f32; 4]) -> [f32; 4] {
            i_pq_e(NeonToken::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_hlg_to_linear(v: [f32; 4]) -> [f32; 4] {
            i_hlg_d(NeonToken::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_linear_to_hlg(v: [f32; 4]) -> [f32; 4] {
            i_hlg_e(NeonToken::summon().unwrap(), v)
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub mod arr {
        use archmage::{SimdToken, Wasm128Token, arcane};
        use linear_srgb::__bench_rails::x4;

        #[arcane]
        fn i_s2l(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            x4::srgb_to_linear_wasm128_unclamped(t, v)
        }
        #[arcane]
        fn i_l2s(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            x4::linear_to_srgb_wasm128_unclamped(t, v)
        }
        #[arcane]
        fn i_g2l(t: Wasm128Token, v: [f32; 4], g: f32) -> [f32; 4] {
            x4::gamma_to_linear_wasm128_unclamped(t, v, g)
        }
        #[arcane]
        fn i_bt_d(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            x4::bt709_to_linear_wasm128_unclamped(t, v)
        }
        #[arcane]
        fn i_bt_e(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            x4::linear_to_bt709_wasm128_unclamped(t, v)
        }
        #[arcane]
        fn i_pq_d(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            x4::pq_to_linear_wasm128_unclamped(t, v)
        }
        #[arcane]
        fn i_pq_e(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            x4::linear_to_pq_wasm128_unclamped(t, v)
        }
        #[arcane]
        fn i_hlg_d(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            x4::hlg_to_linear_wasm128_unclamped(t, v)
        }
        #[arcane]
        fn i_hlg_e(t: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            x4::linear_to_hlg_wasm128_unclamped(t, v)
        }

        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_srgb_to_linear(v: [f32; 4]) -> [f32; 4] {
            i_s2l(Wasm128Token::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_linear_to_srgb(v: [f32; 4]) -> [f32; 4] {
            i_l2s(Wasm128Token::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_gamma_to_linear(v: [f32; 4], g: f32) -> [f32; 4] {
            i_g2l(Wasm128Token::summon().unwrap(), v, g)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_bt709_to_linear(v: [f32; 4]) -> [f32; 4] {
            i_bt_d(Wasm128Token::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_linear_to_bt709(v: [f32; 4]) -> [f32; 4] {
            i_bt_e(Wasm128Token::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_pq_to_linear(v: [f32; 4]) -> [f32; 4] {
            i_pq_d(Wasm128Token::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_linear_to_pq(v: [f32; 4]) -> [f32; 4] {
            i_pq_e(Wasm128Token::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_hlg_to_linear(v: [f32; 4]) -> [f32; 4] {
            i_hlg_d(Wasm128Token::summon().unwrap(), v)
        }
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn ds_arr_unclamped_linear_to_hlg(v: [f32; 4]) -> [f32; 4] {
            i_hlg_e(Wasm128Token::summon().unwrap(), v)
        }
    }

    pub fn exercise(f: &mut [f32]) {
        ds_unclamped_srgb_to_linear_slice(black_box(f));
        ds_unclamped_linear_to_srgb_slice(black_box(f));
        ds_unclamped_srgb_to_linear_extended_slice(black_box(f));
        ds_unclamped_linear_to_srgb_extended_slice(black_box(f));
        ds_unclamped_gamma_to_linear_slice(black_box(f), black_box(2.2));
        ds_unclamped_linear_to_gamma_slice(black_box(f), black_box(2.2));
        ds_unclamped_bt709_to_linear_slice(black_box(f));
        ds_unclamped_linear_to_bt709_slice(black_box(f));
        ds_unclamped_pq_to_linear_slice(black_box(f));
        ds_unclamped_linear_to_pq_slice(black_box(f));
        ds_unclamped_hlg_to_linear_slice(black_box(f));
        ds_unclamped_linear_to_hlg_slice(black_box(f));
        #[cfg(target_arch = "x86_64")]
        {
            black_box(arr::ds_arr_unclamped_srgb_to_linear(black_box([0.5; 8])));
            black_box(arr::ds_arr_unclamped_linear_to_srgb(black_box([0.5; 8])));
            black_box(arr::ds_arr_unclamped_gamma_to_linear(
                black_box([0.5; 8]),
                black_box(2.2),
            ));
            black_box(arr::ds_arr_unclamped_bt709_to_linear(black_box([0.5; 8])));
            black_box(arr::ds_arr_unclamped_linear_to_bt709(black_box([0.5; 8])));
            black_box(arr::ds_arr_unclamped_pq_to_linear(black_box([0.5; 8])));
            black_box(arr::ds_arr_unclamped_linear_to_pq(black_box([0.5; 8])));
            black_box(arr::ds_arr_unclamped_hlg_to_linear(black_box([0.5; 8])));
            black_box(arr::ds_arr_unclamped_linear_to_hlg(black_box([0.5; 8])));
        }
        #[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
        {
            black_box(arr::ds_arr_unclamped_srgb_to_linear(black_box([0.5; 4])));
            black_box(arr::ds_arr_unclamped_linear_to_srgb(black_box([0.5; 4])));
            black_box(arr::ds_arr_unclamped_gamma_to_linear(
                black_box([0.5; 4]),
                black_box(2.2),
            ));
            black_box(arr::ds_arr_unclamped_bt709_to_linear(black_box([0.5; 4])));
            black_box(arr::ds_arr_unclamped_linear_to_bt709(black_box([0.5; 4])));
            black_box(arr::ds_arr_unclamped_pq_to_linear(black_box([0.5; 4])));
            black_box(arr::ds_arr_unclamped_linear_to_pq(black_box([0.5; 4])));
            black_box(arr::ds_arr_unclamped_hlg_to_linear(black_box([0.5; 4])));
            black_box(arr::ds_arr_unclamped_linear_to_hlg(black_box([0.5; 4])));
        }
    }
}

fn main() {
    let mut f = vec![0.5f32; 64];
    let mut u8b = vec![128u8; 64];
    let mut u16b = vec![32768u16; 64];
    ds_srgb_to_linear_slice(black_box(&mut f));
    ds_linear_to_srgb_slice(black_box(&mut f));
    ds_srgb_to_linear_rgba_slice(black_box(&mut f));
    ds_linear_to_srgb_rgba_slice(black_box(&mut f));
    ds_srgb_to_linear_extended_slice(black_box(&mut f));
    ds_linear_to_srgb_extended_slice(black_box(&mut f));
    ds_srgb_to_linear_premultiply_rgba_slice(black_box(&mut f));
    ds_unpremultiply_linear_to_srgb_rgba_slice(black_box(&mut f));
    ds_srgb_u8_to_linear_slice(black_box(&u8b), black_box(&mut f));
    ds_srgb_u8_to_linear_rgba_slice(black_box(&u8b), black_box(&mut f));
    ds_linear_to_srgb_u8_slice(black_box(&f), black_box(&mut u8b));
    ds_linear_to_srgb_u8_rgba_slice(black_box(&f), black_box(&mut u8b));
    ds_srgb_u8_to_linear_premultiply_rgba_slice(black_box(&u8b), black_box(&mut f));
    ds_unpremultiply_linear_to_srgb_u8_rgba_slice(black_box(&f), black_box(&mut u8b));
    ds_srgb_u16_to_linear_slice(black_box(&u16b), black_box(&mut f));
    ds_srgb_u16_to_linear_rgba_slice(black_box(&u16b), black_box(&mut f));
    ds_linear_to_srgb_u16_slice(black_box(&f), black_box(&mut u16b));
    ds_linear_to_srgb_u16_slice_fast(black_box(&f), black_box(&mut u16b));
    ds_gamma_to_linear_slice(black_box(&mut f), black_box(2.2));
    ds_linear_to_gamma_slice(black_box(&mut f), black_box(2.2));
    ds_gamma22_to_linear_slice(black_box(&mut f));
    ds_linear_to_gamma22_slice(black_box(&mut f));
    ds_bt709_to_linear_slice(black_box(&mut f));
    ds_linear_to_bt709_slice(black_box(&mut f));
    ds_pq_to_linear_slice(black_box(&mut f));
    ds_linear_to_pq_slice(black_box(&mut f));
    ds_hlg_to_linear_slice(black_box(&mut f));
    ds_linear_to_hlg_slice(black_box(&mut f));

    #[cfg(target_arch = "x86_64")]
    {
        black_box(arr_impl::ds_arr_srgb_to_linear(black_box([0.5; 8])));
        black_box(arr_impl::ds_arr_linear_to_srgb(black_box([0.5; 8])));
        black_box(arr_impl::ds_arr_srgb_u16_to_linear(black_box([32768; 8])));
        black_box(arr_impl::ds_arr_linear_to_srgb_u16(black_box([0.5; 8])));
        black_box(arr_impl::ds_arr_gamma_to_linear(
            black_box([0.5; 8]),
            black_box(2.2),
        ));
        black_box(arr_impl::ds_arr_bt709_to_linear(black_box([0.5; 8])));
        black_box(arr_impl::ds_arr_linear_to_bt709(black_box([0.5; 8])));
        black_box(arr_impl::ds_arr_pq_to_linear(black_box([0.5; 8])));
        black_box(arr_impl::ds_arr_linear_to_pq(black_box([0.5; 8])));
        black_box(arr_impl::ds_arr_hlg_to_linear(black_box([0.5; 8])));
        black_box(arr_impl::ds_arr_linear_to_hlg(black_box([0.5; 8])));
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
    {
        black_box(arr_impl::ds_arr_srgb_to_linear(black_box([0.5; 4])));
        black_box(arr_impl::ds_arr_linear_to_srgb(black_box([0.5; 4])));
        black_box(arr_impl::ds_arr_gamma_to_linear(
            black_box([0.5; 4]),
            black_box(2.2),
        ));
        black_box(arr_impl::ds_arr_bt709_to_linear(black_box([0.5; 4])));
        black_box(arr_impl::ds_arr_linear_to_bt709(black_box([0.5; 4])));
        black_box(arr_impl::ds_arr_pq_to_linear(black_box([0.5; 4])));
        black_box(arr_impl::ds_arr_linear_to_pq(black_box([0.5; 4])));
        black_box(arr_impl::ds_arr_hlg_to_linear(black_box([0.5; 4])));
        black_box(arr_impl::ds_arr_linear_to_hlg(black_box([0.5; 4])));
    }

    black_box(ds_scalar_srgb_to_linear(black_box(0.5)));
    black_box(ds_scalar_linear_to_srgb(black_box(0.5)));
    black_box(ds_scalar_bt709_to_linear(black_box(0.5)));
    black_box(ds_scalar_linear_to_bt709(black_box(0.5)));
    black_box(ds_scalar_pq_to_linear(black_box(0.5)));
    black_box(ds_scalar_linear_to_pq(black_box(0.5)));
    black_box(ds_scalar_hlg_to_linear(black_box(0.5)));
    black_box(ds_scalar_linear_to_hlg(black_box(0.5)));
    black_box(ds_scalar_gamma_to_linear(black_box(0.5), black_box(2.2)));
    black_box(ds_scalar_precise_srgb_to_linear(black_box(0.5)));

    #[cfg(linear_srgb_bench_internal)]
    unclamped_impl::exercise(&mut f);
}
