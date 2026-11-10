//! Capture numerical output before an entry-point refactor; compare the text afterwards.
#![forbid(unsafe_code)]

use archmage::{SimdToken, arcane};
use std::{println, vec::Vec};

fn sample(i: usize) -> f32 {
    const EDGES: [f32; 19] = [
        -4.0,
        -0.1,
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        1e-8,
        1e-7,
        0.0030412826,
        0.02,
        0.03929337,
        0.25,
        0.5,
        1.0,
        1.0001,
        4.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    if i < EDGES.len() {
        EDGES[i]
    } else {
        (i - EDGES.len()) as f32 / 65535.0
    }
}

fn fold(hash: &mut u64, values: &[f32]) {
    for &v in values {
        // NaN payload/sign are outside the cross-tier contract.
        let bits = if v.is_nan() {
            f32::NAN.to_bits()
        } else {
            v.to_bits()
        };
        *hash = (*hash ^ u64::from(bits)).wrapping_mul(0x100000001b3);
    }
}

macro_rules! capture {
    ($width:ident, $n:literal, $tier:ident; [$($plain:ident),*]; $decode:ident, $encode:ident, $decode_slice:ident, $encode_slice:ident) => {{
        use crate::target_features::$width as api;
        let mut hash = 0xcbf29ce484222325;
        macro_rules! plain {
            ($name:ident) => {
                for offset in (0..65555).step_by($n) {
                    let input = core::array::from_fn(|i| sample(offset + i));
                    fold(&mut hash, &api::$name(input));
                }
            };
        }
        $(plain!($plain);)*
        for gamma in [0.5, 1.0, 1.8, 2.2, 2.4] {
            for offset in (0..65555).step_by($n) {
                let input = core::array::from_fn(|i| sample(offset + i));
                fold(&mut hash, &api::$decode(input, gamma));
                fold(&mut hash, &api::$encode(input, gamma));
            }
        }
        for len in 0..=65 {
            let mut values: Vec<_> = (0..len).map(sample).collect();
            api::$decode_slice(&mut values);
            fold(&mut hash, &values);
            api::$encode_slice(&mut values);
            fold(&mut hash, &values);
        }
        println!("{} {} {hash:016x}", stringify!($width), stringify!($tier));
    }};
}

#[arcane]
fn capture_tier_scalar(_token: archmage::ScalarToken) {
    capture!(x4, 4, scalar; [srgb_to_linear_scalar, linear_to_srgb_scalar, srgb_to_linear_extended_scalar, linear_to_srgb_extended_scalar]; gamma_to_linear_scalar, linear_to_gamma_scalar, srgb_to_linear_slice_scalar, linear_to_srgb_slice_scalar);
    capture!(x8, 8, scalar; [srgb_to_linear_scalar, linear_to_srgb_scalar, srgb_to_linear_extended_scalar, linear_to_srgb_extended_scalar]; gamma_to_linear_scalar, linear_to_gamma_scalar, srgb_to_linear_slice_scalar, linear_to_srgb_slice_scalar);
    capture!(x16, 16, scalar; [srgb_to_linear_scalar, linear_to_srgb_scalar, srgb_to_linear_extended_scalar, linear_to_srgb_extended_scalar]; gamma_to_linear_scalar, linear_to_gamma_scalar, srgb_to_linear_slice_scalar, linear_to_srgb_slice_scalar);
    #[cfg(feature = "transfer")]
    capture!(x4, 4, scalar_transfer; [tf_srgb_to_linear_scalar, tf_linear_to_srgb_scalar, bt709_to_linear_scalar, linear_to_bt709_scalar, pq_to_linear_scalar, linear_to_pq_scalar, hlg_to_linear_scalar, linear_to_hlg_scalar]; gamma_to_linear_scalar, linear_to_gamma_scalar, srgb_to_linear_slice_scalar, linear_to_srgb_slice_scalar);
    #[cfg(feature = "transfer")]
    capture!(x8, 8, scalar_transfer; [tf_srgb_to_linear_scalar, tf_linear_to_srgb_scalar, bt709_to_linear_scalar, linear_to_bt709_scalar, pq_to_linear_scalar, linear_to_pq_scalar, hlg_to_linear_scalar, linear_to_hlg_scalar]; gamma_to_linear_scalar, linear_to_gamma_scalar, srgb_to_linear_slice_scalar, linear_to_srgb_slice_scalar);
    #[cfg(feature = "transfer")]
    capture!(x16, 16, scalar_transfer; [tf_srgb_to_linear_scalar, tf_linear_to_srgb_scalar, bt709_to_linear_scalar, linear_to_bt709_scalar, pq_to_linear_scalar, linear_to_pq_scalar, hlg_to_linear_scalar, linear_to_hlg_scalar]; gamma_to_linear_scalar, linear_to_gamma_scalar, srgb_to_linear_slice_scalar, linear_to_srgb_slice_scalar);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn capture_tier_v3(_token: archmage::X64V3Token) {
    capture!(x4, 4, v3; [srgb_to_linear_v3, linear_to_srgb_v3, srgb_to_linear_extended_v3, linear_to_srgb_extended_v3]; gamma_to_linear_v3, linear_to_gamma_v3, srgb_to_linear_slice_v3, linear_to_srgb_slice_v3);
    capture!(x8, 8, v3; [srgb_to_linear_v3, linear_to_srgb_v3, srgb_to_linear_extended_v3, linear_to_srgb_extended_v3]; gamma_to_linear_v3, linear_to_gamma_v3, srgb_to_linear_slice_v3, linear_to_srgb_slice_v3);
    capture!(x16, 16, v3; [srgb_to_linear_v3, linear_to_srgb_v3, srgb_to_linear_extended_v3, linear_to_srgb_extended_v3]; gamma_to_linear_v3, linear_to_gamma_v3, srgb_to_linear_slice_v3, linear_to_srgb_slice_v3);
    #[cfg(feature = "transfer")]
    capture!(x4, 4, v3_transfer; [tf_srgb_to_linear_v3, tf_linear_to_srgb_v3, bt709_to_linear_v3, linear_to_bt709_v3, pq_to_linear_v3, linear_to_pq_v3, hlg_to_linear_v3, linear_to_hlg_v3]; gamma_to_linear_v3, linear_to_gamma_v3, srgb_to_linear_slice_v3, linear_to_srgb_slice_v3);
    #[cfg(feature = "transfer")]
    capture!(x8, 8, v3_transfer; [tf_srgb_to_linear_v3, tf_linear_to_srgb_v3, bt709_to_linear_v3, linear_to_bt709_v3, pq_to_linear_v3, linear_to_pq_v3, hlg_to_linear_v3, linear_to_hlg_v3]; gamma_to_linear_v3, linear_to_gamma_v3, srgb_to_linear_slice_v3, linear_to_srgb_slice_v3);
    #[cfg(feature = "transfer")]
    capture!(x16, 16, v3_transfer; [tf_srgb_to_linear_v3, tf_linear_to_srgb_v3, bt709_to_linear_v3, linear_to_bt709_v3, pq_to_linear_v3, linear_to_pq_v3, hlg_to_linear_v3, linear_to_hlg_v3]; gamma_to_linear_v3, linear_to_gamma_v3, srgb_to_linear_slice_v3, linear_to_srgb_slice_v3);
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn capture_tier_v4(_token: archmage::X64V4Token) {
    capture!(x4, 4, v4; [srgb_to_linear_v4, linear_to_srgb_v4, srgb_to_linear_extended_v4, linear_to_srgb_extended_v4]; gamma_to_linear_v4, linear_to_gamma_v4, srgb_to_linear_slice_v4, linear_to_srgb_slice_v4);
    capture!(x8, 8, v4; [srgb_to_linear_v4, linear_to_srgb_v4, srgb_to_linear_extended_v4, linear_to_srgb_extended_v4]; gamma_to_linear_v4, linear_to_gamma_v4, srgb_to_linear_slice_v4, linear_to_srgb_slice_v4);
    capture!(x16, 16, v4; [srgb_to_linear_v4, linear_to_srgb_v4, srgb_to_linear_extended_v4, linear_to_srgb_extended_v4]; gamma_to_linear_v4, linear_to_gamma_v4, srgb_to_linear_slice_v4, linear_to_srgb_slice_v4);
    #[cfg(feature = "transfer")]
    capture!(x4, 4, v4_transfer; [tf_srgb_to_linear_v4, tf_linear_to_srgb_v4, bt709_to_linear_v4, linear_to_bt709_v4, pq_to_linear_v4, linear_to_pq_v4, hlg_to_linear_v4, linear_to_hlg_v4]; gamma_to_linear_v4, linear_to_gamma_v4, srgb_to_linear_slice_v4, linear_to_srgb_slice_v4);
    #[cfg(feature = "transfer")]
    capture!(x8, 8, v4_transfer; [tf_srgb_to_linear_v4, tf_linear_to_srgb_v4, bt709_to_linear_v4, linear_to_bt709_v4, pq_to_linear_v4, linear_to_pq_v4, hlg_to_linear_v4, linear_to_hlg_v4]; gamma_to_linear_v4, linear_to_gamma_v4, srgb_to_linear_slice_v4, linear_to_srgb_slice_v4);
    #[cfg(feature = "transfer")]
    capture!(x16, 16, v4_transfer; [tf_srgb_to_linear_v4, tf_linear_to_srgb_v4, bt709_to_linear_v4, linear_to_bt709_v4, pq_to_linear_v4, linear_to_pq_v4, hlg_to_linear_v4, linear_to_hlg_v4]; gamma_to_linear_v4, linear_to_gamma_v4, srgb_to_linear_slice_v4, linear_to_srgb_slice_v4);
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn capture_tier_neon(_token: archmage::NeonToken) {
    capture!(x4, 4, neon; [srgb_to_linear_neon, linear_to_srgb_neon, srgb_to_linear_extended_neon, linear_to_srgb_extended_neon]; gamma_to_linear_neon, linear_to_gamma_neon, srgb_to_linear_slice_neon, linear_to_srgb_slice_neon);
    capture!(x8, 8, neon; [srgb_to_linear_neon, linear_to_srgb_neon, srgb_to_linear_extended_neon, linear_to_srgb_extended_neon]; gamma_to_linear_neon, linear_to_gamma_neon, srgb_to_linear_slice_neon, linear_to_srgb_slice_neon);
    capture!(x16, 16, neon; [srgb_to_linear_neon, linear_to_srgb_neon, srgb_to_linear_extended_neon, linear_to_srgb_extended_neon]; gamma_to_linear_neon, linear_to_gamma_neon, srgb_to_linear_slice_neon, linear_to_srgb_slice_neon);
    #[cfg(feature = "transfer")]
    capture!(x4, 4, neon_transfer; [tf_srgb_to_linear_neon, tf_linear_to_srgb_neon, bt709_to_linear_neon, linear_to_bt709_neon, pq_to_linear_neon, linear_to_pq_neon, hlg_to_linear_neon, linear_to_hlg_neon]; gamma_to_linear_neon, linear_to_gamma_neon, srgb_to_linear_slice_neon, linear_to_srgb_slice_neon);
    #[cfg(feature = "transfer")]
    capture!(x8, 8, neon_transfer; [tf_srgb_to_linear_neon, tf_linear_to_srgb_neon, bt709_to_linear_neon, linear_to_bt709_neon, pq_to_linear_neon, linear_to_pq_neon, hlg_to_linear_neon, linear_to_hlg_neon]; gamma_to_linear_neon, linear_to_gamma_neon, srgb_to_linear_slice_neon, linear_to_srgb_slice_neon);
    #[cfg(feature = "transfer")]
    capture!(x16, 16, neon_transfer; [tf_srgb_to_linear_neon, tf_linear_to_srgb_neon, bt709_to_linear_neon, linear_to_bt709_neon, pq_to_linear_neon, linear_to_pq_neon, hlg_to_linear_neon, linear_to_hlg_neon]; gamma_to_linear_neon, linear_to_gamma_neon, srgb_to_linear_slice_neon, linear_to_srgb_slice_neon);
}

#[cfg(target_arch = "wasm32")]
#[arcane]
fn capture_tier_wasm128(_token: archmage::Wasm128Token) {
    capture!(x4, 4, wasm128; [srgb_to_linear_wasm128, linear_to_srgb_wasm128, srgb_to_linear_extended_wasm128, linear_to_srgb_extended_wasm128]; gamma_to_linear_wasm128, linear_to_gamma_wasm128, srgb_to_linear_slice_wasm128, linear_to_srgb_slice_wasm128);
    capture!(x8, 8, wasm128; [srgb_to_linear_wasm128, linear_to_srgb_wasm128, srgb_to_linear_extended_wasm128, linear_to_srgb_extended_wasm128]; gamma_to_linear_wasm128, linear_to_gamma_wasm128, srgb_to_linear_slice_wasm128, linear_to_srgb_slice_wasm128);
    capture!(x16, 16, wasm128; [srgb_to_linear_wasm128, linear_to_srgb_wasm128, srgb_to_linear_extended_wasm128, linear_to_srgb_extended_wasm128]; gamma_to_linear_wasm128, linear_to_gamma_wasm128, srgb_to_linear_slice_wasm128, linear_to_srgb_slice_wasm128);
    #[cfg(feature = "transfer")]
    capture!(x4, 4, wasm128_transfer; [tf_srgb_to_linear_wasm128, tf_linear_to_srgb_wasm128, bt709_to_linear_wasm128, linear_to_bt709_wasm128, pq_to_linear_wasm128, linear_to_pq_wasm128, hlg_to_linear_wasm128, linear_to_hlg_wasm128]; gamma_to_linear_wasm128, linear_to_gamma_wasm128, srgb_to_linear_slice_wasm128, linear_to_srgb_slice_wasm128);
    #[cfg(feature = "transfer")]
    capture!(x8, 8, wasm128_transfer; [tf_srgb_to_linear_wasm128, tf_linear_to_srgb_wasm128, bt709_to_linear_wasm128, linear_to_bt709_wasm128, pq_to_linear_wasm128, linear_to_pq_wasm128, hlg_to_linear_wasm128, linear_to_hlg_wasm128]; gamma_to_linear_wasm128, linear_to_gamma_wasm128, srgb_to_linear_slice_wasm128, linear_to_srgb_slice_wasm128);
    #[cfg(feature = "transfer")]
    capture!(x16, 16, wasm128_transfer; [tf_srgb_to_linear_wasm128, tf_linear_to_srgb_wasm128, bt709_to_linear_wasm128, linear_to_bt709_wasm128, pq_to_linear_wasm128, linear_to_pq_wasm128, hlg_to_linear_wasm128, linear_to_hlg_wasm128]; gamma_to_linear_wasm128, linear_to_gamma_wasm128, srgb_to_linear_slice_wasm128, linear_to_srgb_slice_wasm128);
}
#[test]
fn entry_point_fingerprint() {
    println!();
    capture_tier_scalar(archmage::ScalarToken);
    #[cfg(target_arch = "x86_64")]
    {
        let token = archmage::X64V3Token::summon();
        let required = std::env::var("LINEAR_SRGB_REQUIRE_TIERS").unwrap_or_default();
        if required.split(',').any(|tier| tier == "X64V3Token") {
            assert!(token.is_some(), "V3 required for this native probe");
        }
        if let Some(token) = token {
            capture_tier_v3(token);
        }
        #[cfg(feature = "avx512")]
        if let Some(token) = archmage::X64V4Token::summon() {
            capture_tier_v4(token);
        }
    }
    #[cfg(target_arch = "aarch64")]
    capture_tier_neon(archmage::NeonToken::summon().expect("NEON required"));
    #[cfg(target_arch = "wasm32")]
    capture_tier_wasm128(archmage::Wasm128Token::summon().expect("SIMD128 required"));
}
