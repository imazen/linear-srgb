//! Direct token hooks have the same row-length contract as dispatched slices.
#[cfg(target_arch = "aarch64")]
use archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
use archmage::X64V3Token;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use archmage::X64V4Token;
use archmage::{SimdToken, arcane};

macro_rules! hooks {
    ($entry:ident, $token:ident, $module:path, [$($fun:ident),+]) => {
        #[arcane]
        fn $entry(token: $token, input: &[f32]) {
            use $module as m;
            $(
                let mut whole = input.to_vec();
                m::$fun(token, &mut whole);
                for width in [1, 3, 7, 17] {
                    let mut split = input.to_vec();
                    for row in split.chunks_mut(width) { m::$fun(token, row); }
                    for (i,(a,b)) in whole.iter().zip(&split).enumerate() {
                        assert_eq!(a.to_bits(),b.to_bits(), "{} index={i} width={width}", stringify!($fun));
                    }
                }
            )+
        }
    };
}

#[cfg(target_arch = "x86_64")]
hooks!(
    v3,
    X64V3Token,
    linear_srgb::tokens::x8,
    [
        srgb_to_linear_slice_v3,
        linear_to_srgb_slice_v3,
        srgb_to_linear_extended_slice_v3,
        linear_to_srgb_extended_slice_v3
    ]
);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
hooks!(
    v4,
    X64V4Token,
    linear_srgb::tokens::x16,
    [srgb_to_linear_slice_v4, linear_to_srgb_slice_v4]
);
#[cfg(all(target_arch = "x86_64", feature = "transfer"))]
hooks!(
    tf_v3,
    X64V3Token,
    linear_srgb::tokens::x4,
    [
        bt709_to_linear_slice_v3,
        linear_to_bt709_slice_v3,
        pq_to_linear_slice_v3,
        linear_to_pq_slice_v3,
        hlg_to_linear_slice_v3,
        linear_to_hlg_slice_v3
    ]
);
#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "transfer"))]
hooks!(
    tf_v4,
    X64V4Token,
    linear_srgb::tokens::x16,
    [
        bt709_to_linear_slice_v4,
        linear_to_bt709_slice_v4,
        pq_to_linear_slice_v4,
        linear_to_pq_slice_v4,
        hlg_to_linear_slice_v4,
        linear_to_hlg_slice_v4
    ]
);
#[cfg(target_arch = "aarch64")]
hooks!(
    neon,
    NeonToken,
    linear_srgb::tokens::x4,
    [srgb_to_linear_slice_neon, linear_to_srgb_slice_neon]
);
#[cfg(all(target_arch = "aarch64", feature = "transfer"))]
hooks!(
    tf_neon,
    NeonToken,
    linear_srgb::tokens::x4,
    [
        bt709_to_linear_slice_neon,
        linear_to_bt709_slice_neon,
        pq_to_linear_slice_neon,
        linear_to_pq_slice_neon,
        hlg_to_linear_slice_neon,
        linear_to_hlg_slice_neon
    ]
);

#[test]
fn token_slice_rows_match() {
    let input: Vec<f32> = (0..1600).map(|i| (i as f32 / 1600.0).powi(4)).collect();
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = archmage::X64V3Token::summon() {
        v3(t, &input);
        #[cfg(feature = "transfer")]
        tf_v3(t, &input);
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = archmage::X64V4Token::summon() {
        v4(t, &input);
        #[cfg(feature = "transfer")]
        tf_v4(t, &input);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        neon(t, &input);
        #[cfg(feature = "transfer")]
        tf_neon(t, &input);
    }
}
