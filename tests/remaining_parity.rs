use linear_srgb::default as d;

#[test]
fn u16_decode_and_rgba_encode_are_independent_of_row_length() {
    let input: Vec<u16> = (0..=65535).collect();
    let mut whole = vec![0.0f32; input.len()];
    d::srgb_u16_to_linear_slice(&input, &mut whole);
    for width in [1, 7, 8, 17] {
        let mut split = vec![0.0; input.len()];
        for (s, o) in input.chunks(width).zip(split.chunks_mut(width)) {
            d::srgb_u16_to_linear_slice(s, o);
        }
        assert_eq!(whole, split, "decode width={width}");
    }
    let input: Vec<f32> = (0..65536).map(|i| i as f32 / 65535.0).collect();
    let mut whole = vec![0u16; input.len()];
    d::linear_to_srgb_u16_rgba_slice(&input, &mut whole);
    for width in [4, 28, 32, 68] {
        let mut split = vec![0; input.len()];
        for (s, o) in input.chunks(width).zip(split.chunks_mut(width)) {
            d::linear_to_srgb_u16_rgba_slice(s, o);
        }
        assert_eq!(whole, split, "encode width={width}");
    }
}

#[cfg(feature = "transfer")]
#[test]
fn hdr_rgba_uses_the_plain_kernel_at_every_tier() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
    type Pair = (fn(&mut [f32]), fn(&mut [f32]));
    let pairs: [Pair; 6] = [
        (d::bt709_to_linear_slice, d::bt709_to_linear_rgba_slice),
        (d::linear_to_bt709_slice, d::linear_to_bt709_rgba_slice),
        (d::pq_to_linear_slice, d::pq_to_linear_rgba_slice),
        (d::linear_to_pq_slice, d::linear_to_pq_rgba_slice),
        (d::hlg_to_linear_slice, d::hlg_to_linear_rgba_slice),
        (d::linear_to_hlg_slice, d::linear_to_hlg_rgba_slice),
    ];
    let input: Vec<f32> = (0..65536).map(|i| (i as f32 / 65535.0).powi(4)).collect();
    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        for (plain, rgba) in pairs {
            let mut want = input.clone();
            plain(&mut want);
            let mut actual = input.clone();
            rgba(&mut actual);
            for (i, (a, b)) in actual.iter().zip(&want).enumerate() {
                let expected = if i % 4 == 3 { input[i] } else { *b };
                assert_eq!(a.to_bits(), expected.to_bits(), "index={i} tier={perm}");
            }
        }
    });
}
