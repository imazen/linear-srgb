//! Public fixed-array families compose through the caller's own archmage version.
#![forbid(unsafe_code)]
use archmage::{incant, magetypes};
use linear_srgb::target_features::{x8, x16};

#[magetypes(v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn convert8(_token: Token, values: [f32; 8]) -> [f32; 8] {
    incant!(x8::srgb_to_linear(values) without token)
}
#[magetypes(v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn convert16(_token: Token, values: [f32; 16]) -> [f32; 16] {
    incant!(x16::srgb_to_linear(values) without token)
}
#[magetypes(v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn gamma16(_token: Token, values: [f32; 16], gamma: f32) -> [f32; 16] {
    incant!(x16::gamma_to_linear(values, gamma) without token)
}
#[magetypes(v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn slice16(_token: Token, values: &mut [f32]) {
    incant!(x16::srgb_to_linear_slice(values) without token)
}
#[cfg(feature = "transfer")]
#[magetypes(v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn pq16(_token: Token, values: [f32; 16]) -> [f32; 16] {
    incant!(x16::pq_to_linear(values) without token)
}

#[test]
fn complete_dispatch_and_scalar_fallbacks() {
    for offset in 0..256 {
        let values = core::array::from_fn(|i| (offset + i) as f32 / 255.0);
        let scalar = incant!(convert16(values), [scalar]);
        assert_eq!(
            scalar.map(f32::to_bits),
            x16::srgb_to_linear_scalar(values).map(f32::to_bits)
        );
        #[cfg(target_arch = "x86_64")]
        assert_eq!(
            incant!(convert16(values), [v3, scalar]).map(f32::to_bits),
            scalar.map(f32::to_bits)
        );
        let got = incant!(
            convert16(values),
            [v4(cfg(avx512)), v3, neon, wasm128, scalar]
        );
        assert_eq!(
            got.map(f32::to_bits),
            x16::srgb_to_linear_scalar(values).map(f32::to_bits)
        );
        let values8 = core::array::from_fn(|i| values[i]);
        assert_eq!(
            incant!(
                convert8(values8),
                [v4(cfg(avx512)), v3, neon, wasm128, scalar]
            )
            .map(f32::to_bits),
            x8::srgb_to_linear_scalar(values8).map(f32::to_bits)
        );
        for gamma in [0.5, 1.0, 1.8, 2.2] {
            assert_eq!(
                incant!(
                    gamma16(values, gamma),
                    [v4(cfg(avx512)), v3, neon, wasm128, scalar]
                )
                .map(f32::to_bits),
                x16::gamma_to_linear_scalar(values, gamma).map(f32::to_bits)
            );
        }
        #[cfg(feature = "transfer")]
        assert_eq!(
            incant!(pq16(values), [v4(cfg(avx512)), v3, neon, wasm128, scalar]).map(f32::to_bits),
            x16::pq_to_linear_scalar(values).map(f32::to_bits)
        );
    }
    for len in 0..65 {
        let mut values: Vec<_> = (0..len).map(|i| i as f32 / 64.0).collect();
        let mut expected = values.clone();
        incant!(
            slice16(&mut values),
            [v4(cfg(avx512)), v3, neon, wasm128, scalar]
        );
        x16::srgb_to_linear_slice_scalar(&mut expected);
        assert_eq!(
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
