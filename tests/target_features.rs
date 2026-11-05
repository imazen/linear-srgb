//! Bit-exact compatibility and safe composition of the array/slice entry points.
#![forbid(unsafe_code)]
#![allow(deprecated)] // The old kernels are the compatibility oracle.

use archmage::{SimdToken, arcane};

fn sample(i: usize) -> f32 {
    const VALUES: [f32; 20] = [
        -1.0,
        -0.0,
        0.0,
        f32::MIN_POSITIVE,
        0.00001,
        0.003,
        0.0030412826,
        0.03929337,
        0.04045,
        0.1,
        0.5,
        0.9,
        1.0,
        1.0001,
        2.0,
        4.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(1),
    ];
    VALUES[i % VALUES.len()]
}

macro_rules! arrays {
    ($token:ident, $n:literal; $($name:ident),* $(,)?) => {
        for offset in 0..40 {
            let input: [f32; $n] = core::array::from_fn(|i| sample(offset + i));
            $(assert_eq!(new::$name(input).map(f32::to_bits),
                         old::$name($token, input).map(f32::to_bits), stringify!($name));)*
        }
    };
}
macro_rules! slices {
    ($token:ident; $($name:ident),* $(,)?) => {
        for len in 0..40 {
            $(
                let mut a: Vec<_> = (0..len).map(sample).collect();
                let mut b = a.clone();
                new::$name(&mut a);
                old::$name($token, &mut b);
                assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                           b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                           "{}, len={len}", stringify!($name));
            )*
        }
    };
}
macro_rules! gamma {
    ($token:ident, $n:literal; $decode:ident, $encode:ident, $decode_slice:ident, $encode_slice:ident) => {
        for gamma in [0.5, 1.0, 1.8, 2.2] {
            for offset in 0..40 {
                let input: [f32; $n] = core::array::from_fn(|i| sample(offset + i));
                for (a, b) in [
                    (
                        new::$decode(input, gamma),
                        old::$decode($token, input, gamma),
                    ),
                    (
                        new::$encode(input, gamma),
                        old::$encode($token, input, gamma),
                    ),
                ] {
                    assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
                }
                for (new_fn, old_fn) in [
                    (
                        new::$decode_slice as fn(&mut [f32], f32),
                        old::$decode_slice as fn(_, &mut [f32], f32),
                    ),
                    (
                        new::$encode_slice as fn(&mut [f32], f32),
                        old::$encode_slice as fn(_, &mut [f32], f32),
                    ),
                ] {
                    let mut a: Vec<_> = (0..offset).map(sample).collect();
                    let mut b = a.clone();
                    new_fn(&mut a, gamma);
                    old_fn($token, &mut b, gamma);
                    assert_eq!(
                        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
    };
}
macro_rules! integers {
    ($token:ident, $n:literal; $decode:ident, $encode:ident, $decode_slice:ident, $encode_slice:ident) => {
        for offset in 0..256 {
            let bytes: [u8; $n] = core::array::from_fn(|i| (offset + i) as u8);
            let floats: [f32; $n] = core::array::from_fn(|i| sample(offset + i));
            assert_eq!(
                new::$decode(bytes).map(f32::to_bits),
                old::$decode($token, bytes).map(f32::to_bits)
            );
            assert_eq!(new::$encode(floats), old::$encode($token, floats));
        }
        for len in 0..40 {
            let bytes: Vec<_> = (0..len).map(|i| (i * 37) as u8).collect();
            let mut a = vec![0.0; len];
            let mut b = a.clone();
            new::$decode_slice(&bytes, &mut a);
            old::$decode_slice($token, &bytes, &mut b);
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            let floats: Vec<_> = (0..len).map(sample).collect();
            let mut a = vec![0; len];
            let mut b = a.clone();
            new::$encode_slice(&floats, &mut a);
            old::$encode_slice($token, &floats, &mut b);
            assert_eq!(a, b);
        }
    };
}
macro_rules! u16_arrays {
    ($token:ident; $decode:ident, $encode:ident) => {
        for offset in (0..=u16::MAX).step_by(8) {
            let input = core::array::from_fn(|i| offset + i as u16);
            assert_eq!(
                new::$decode(input).map(f32::to_bits),
                old::$decode($token, input).map(f32::to_bits)
            );
        }
        for offset in 0..40 {
            let input = core::array::from_fn(|i| sample(offset + i));
            assert_eq!(new::$encode(input), old::$encode($token, input));
        }
    };
}

// The caller owns the capability proof; none crosses the new API boundary.
macro_rules! check {
    ($test:ident, $token:ident, $width:ident, $n:literal, $proof:ident; $($body:tt)*) => {
        #[test]
        fn $test() {
            #[arcane]
            fn run($proof: archmage::$token) {
                use linear_srgb::target_features::$width as new;
                use linear_srgb::tokens::$width as old;
                $($body)*
            }
            if let Some(token) = archmage::$token::summon() { run(token); }
        }
    };
}

#[cfg(target_arch = "x86_64")]
check!(x4_v3, X64V3Token, x4, 4, token;
    arrays!(token, 4; srgb_to_linear_v3, linear_to_srgb_v3, srgb_to_linear_extended_v3, linear_to_srgb_extended_v3);
    slices!(token; srgb_to_linear_slice_v3, linear_to_srgb_slice_v3);
    #[cfg(feature = "transfer")]
    {
    arrays!(token, 4; tf_srgb_to_linear_v3, tf_linear_to_srgb_v3, bt709_to_linear_v3, linear_to_bt709_v3, pq_to_linear_v3, linear_to_pq_v3, hlg_to_linear_v3, linear_to_hlg_v3);
    slices!(token; tf_srgb_to_linear_slice_v3, tf_linear_to_srgb_slice_v3, bt709_to_linear_slice_v3, linear_to_bt709_slice_v3, pq_to_linear_slice_v3, linear_to_pq_slice_v3, hlg_to_linear_slice_v3, linear_to_hlg_slice_v3);
    }
    gamma!(token, 4; gamma_to_linear_v3, linear_to_gamma_v3, gamma_to_linear_slice_v3, linear_to_gamma_slice_v3);
);

#[cfg(target_arch = "aarch64")]
check!(x4_neon, NeonToken, x4, 4, token;
    arrays!(token, 4; srgb_to_linear_neon, linear_to_srgb_neon, srgb_to_linear_extended_neon, linear_to_srgb_extended_neon);
    slices!(token; srgb_to_linear_slice_neon, linear_to_srgb_slice_neon);
    #[cfg(feature = "transfer")]
    {
    arrays!(token, 4; tf_srgb_to_linear_neon, tf_linear_to_srgb_neon, bt709_to_linear_neon, linear_to_bt709_neon, pq_to_linear_neon, linear_to_pq_neon, hlg_to_linear_neon, linear_to_hlg_neon);
    slices!(token; tf_srgb_to_linear_slice_neon, tf_linear_to_srgb_slice_neon, bt709_to_linear_slice_neon, linear_to_bt709_slice_neon, pq_to_linear_slice_neon, linear_to_pq_slice_neon, hlg_to_linear_slice_neon, linear_to_hlg_slice_neon);
    }
    gamma!(token, 4; gamma_to_linear_neon, linear_to_gamma_neon, gamma_to_linear_slice_neon, linear_to_gamma_slice_neon);
);

#[cfg(target_arch = "wasm32")]
check!(x4_wasm128, Wasm128Token, x4, 4, token;
    arrays!(token, 4; srgb_to_linear_wasm128, linear_to_srgb_wasm128, srgb_to_linear_extended_wasm128, linear_to_srgb_extended_wasm128);
    slices!(token; srgb_to_linear_slice_wasm128, linear_to_srgb_slice_wasm128);
    #[cfg(feature = "transfer")]
    {
    arrays!(token, 4; tf_srgb_to_linear_wasm128, tf_linear_to_srgb_wasm128, bt709_to_linear_wasm128, linear_to_bt709_wasm128, pq_to_linear_wasm128, linear_to_pq_wasm128, hlg_to_linear_wasm128, linear_to_hlg_wasm128);
    slices!(token; tf_srgb_to_linear_slice_wasm128, tf_linear_to_srgb_slice_wasm128, bt709_to_linear_slice_wasm128, linear_to_bt709_slice_wasm128, pq_to_linear_slice_wasm128, linear_to_pq_slice_wasm128, hlg_to_linear_slice_wasm128, linear_to_hlg_slice_wasm128);
    }
    gamma!(token, 4; gamma_to_linear_wasm128, linear_to_gamma_wasm128, gamma_to_linear_slice_wasm128, linear_to_gamma_slice_wasm128);
);

check!(x4_scalar, ScalarToken, x4, 4, token;
    arrays!(token, 4; srgb_to_linear_extended_scalar, linear_to_srgb_extended_scalar);
);

#[cfg(target_arch = "x86_64")]
check!(x8_v3, X64V3Token, x8, 8, token;
    arrays!(token, 8; srgb_to_linear_v3, linear_to_srgb_v3, srgb_to_linear_extended_v3, linear_to_srgb_extended_v3);
    slices!(token; srgb_to_linear_slice_v3, linear_to_srgb_slice_v3, srgb_to_linear_extended_slice_v3, linear_to_srgb_extended_slice_v3);
    #[cfg(feature = "transfer")]
    {
    arrays!(token, 8; tf_srgb_to_linear_v3, tf_linear_to_srgb_v3, bt709_to_linear_v3, linear_to_bt709_v3, pq_to_linear_v3, linear_to_pq_v3, hlg_to_linear_v3, linear_to_hlg_v3);
    slices!(token; tf_srgb_to_linear_slice_v3, tf_linear_to_srgb_slice_v3, bt709_to_linear_slice_v3, linear_to_bt709_slice_v3, pq_to_linear_slice_v3, linear_to_pq_slice_v3, hlg_to_linear_slice_v3, linear_to_hlg_slice_v3);
    }
    gamma!(token, 8; gamma_to_linear_v3, linear_to_gamma_v3, gamma_to_linear_slice_v3, linear_to_gamma_slice_v3);
    integers!(token, 8; srgb_u8_to_linear_v3, linear_to_srgb_u8_v3, srgb_u8_to_linear_slice_v3, linear_to_srgb_u8_slice_v3);
    u16_arrays!(token; srgb_u16_to_linear_v3, linear_to_srgb_u16_v3);
);

#[cfg(target_arch = "x86_64")]
check!(x8_scalar, ScalarToken, x8, 8, token;
    u16_arrays!(token; srgb_u16_to_linear_scalar, linear_to_srgb_u16_scalar);
);

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
check!(x16_v4, X64V4Token, x16, 16, token;
    arrays!(token, 16; srgb_to_linear_v4, linear_to_srgb_v4);
    slices!(token; srgb_to_linear_slice_v4, linear_to_srgb_slice_v4);
    #[cfg(feature = "transfer")]
    {
    arrays!(token, 16; tf_srgb_to_linear_v4, tf_linear_to_srgb_v4, bt709_to_linear_v4, linear_to_bt709_v4, pq_to_linear_v4, linear_to_pq_v4, hlg_to_linear_v4, linear_to_hlg_v4);
    slices!(token; tf_srgb_to_linear_slice_v4, tf_linear_to_srgb_slice_v4, bt709_to_linear_slice_v4, linear_to_bt709_slice_v4, pq_to_linear_slice_v4, linear_to_pq_slice_v4, hlg_to_linear_slice_v4, linear_to_hlg_slice_v4);
    }
    gamma!(token, 16; gamma_to_linear_v4, linear_to_gamma_v4, gamma_to_linear_slice_v4, linear_to_gamma_slice_v4);
    integers!(token, 16; srgb_u8_to_linear_v4, linear_to_srgb_u8_v4, srgb_u8_to_linear_slice_v4, linear_to_srgb_u8_slice_v4);
);
