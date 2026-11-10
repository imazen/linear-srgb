//! A caller owns dispatch and uses only the direct public replacements.
#![forbid(unsafe_code)]
use archmage::{ScalarToken, arcane, incant};
use linear_srgb::{default, target_features::x4};

macro_rules! feature_caller {
    ($name:ident, $token:ident, $convert:ident) => {
        #[arcane]
        fn $name(_token: archmage::$token, values: [f32; 4]) -> [f32; 4] {
            x4::$convert(values)
        }
    };
}

#[cfg(target_arch = "x86_64")]
feature_caller!(convert4_v3, X64V3Token, srgb_to_linear_v3);
#[cfg(target_arch = "aarch64")]
feature_caller!(convert4_neon, NeonToken, srgb_to_linear_neon);
#[cfg(target_arch = "wasm32")]
feature_caller!(convert4_wasm128, Wasm128Token, srgb_to_linear_wasm128);

fn convert4_scalar(_token: ScalarToken, mut values: [f32; 4]) -> [f32; 4] {
    default::srgb_to_linear_slice(&mut values);
    values
}

#[test]
fn caller_owned_dispatch_with_public_replacements() {
    for offset in 0..256 {
        let values = core::array::from_fn(|i| (offset + i) as f32 / 255.0);
        let converted = incant!(convert4(values), [v3, neon, wasm128, scalar]);
        let mut expected = values;
        default::srgb_to_linear_slice(&mut expected);
        assert_eq!(converted.map(f32::to_bits), expected.map(f32::to_bits));
    }
}
