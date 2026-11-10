//! Consumer-side compatibility of the public x4/x8 replacements.
#![forbid(unsafe_code)]
#![allow(deprecated)]

use archmage::{SimdToken, arcane};

macro_rules! check {
    ($test:ident, $token:ident, $width:ident, $n:literal, $name:ident) => {
        #[test]
        fn $test() {
            #[arcane]
            fn run(token: archmage::$token) {
                for offset in 0..256 {
                    let values: [f32; $n] = core::array::from_fn(|i| (offset + i) as f32 / 255.0);
                    assert_eq!(
                        linear_srgb::target_features::$width::$name(values).map(f32::to_bits),
                        linear_srgb::tokens::$width::$name(token, values).map(f32::to_bits)
                    );
                }
            }
            let token = archmage::$token::summon();
            let required = std::env::var("LINEAR_SRGB_REQUIRE_TIERS").unwrap_or_default();
            if required.split(',').any(|name| name == stringify!($token)) {
                assert!(
                    token.is_some(),
                    "required CPU tier {} unavailable",
                    stringify!($token)
                );
            }
            if let Some(token) = token {
                run(token);
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
check!(x4_v3, X64V3Token, x4, 4, srgb_to_linear_v3);
#[cfg(target_arch = "x86_64")]
check!(x8_v3, X64V3Token, x8, 8, srgb_to_linear_v3);
#[cfg(target_arch = "aarch64")]
check!(x4_neon, NeonToken, x4, 4, srgb_to_linear_neon);
#[cfg(target_arch = "wasm32")]
check!(x4_wasm128, Wasm128Token, x4, 4, srgb_to_linear_wasm128);
