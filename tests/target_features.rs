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

// These compatibility functions historically accepted a token from ordinary
// code and implemented Fn traits. They must not become direct feature functions.
macro_rules! proof_callbacks {
    ($test:ident, $token:ident, $decode:ident, $encode:ident) => {
        #[test]
        fn $test() {
            let _: fn(archmage::$token, [f32; 4]) -> [f32; 4] = linear_srgb::tokens::x4::$decode;
            let _: fn(archmage::$token, [f32; 4]) -> [f32; 4] = linear_srgb::tokens::x4::$encode;
        }
    };
}

#[cfg(target_arch = "x86_64")]
proof_callbacks!(
    x4_proof_callbacks_v3,
    X64V3Token,
    srgb_to_linear_extended_v3,
    linear_to_srgb_extended_v3
);
#[cfg(target_arch = "aarch64")]
proof_callbacks!(
    x4_proof_callbacks_neon,
    NeonToken,
    srgb_to_linear_extended_neon,
    linear_to_srgb_extended_neon
);
#[cfg(target_arch = "wasm32")]
proof_callbacks!(
    x4_proof_callbacks_wasm128,
    Wasm128Token,
    srgb_to_linear_extended_wasm128,
    linear_to_srgb_extended_wasm128
);

#[cfg(target_arch = "x86_64")]
#[test]
fn x8_u16_proof_callbacks_v3() {
    let _: fn(archmage::X64V3Token, [u16; 8]) -> [f32; 8] =
        linear_srgb::tokens::x8::srgb_u16_to_linear_v3;
    let _: fn(archmage::X64V3Token, [f32; 8]) -> [u16; 8] =
        linear_srgb::tokens::x8::linear_to_srgb_u16_v3;
}
