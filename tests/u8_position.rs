use linear_srgb::default as d;

#[test]
fn u8_index_boundaries_match_scalar_and_row_splits() {
    let mut input = vec![f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -0.0, 0.0, 1.0];
    for i in 0..4095 {
        let center = ((i as f64 + 0.5) / 4095.0) as f32;
        input.extend((0..=8).map(|k| f32::from_bits(center.to_bits() - 4 + k)));
    }
    let want: Vec<u8> = input.iter().map(|&x| d::linear_to_srgb_u8(x)).collect();
    for width in [1, 7, 8, 16, 31, input.len()] {
        let mut out = vec![0; input.len()];
        for (s, o) in input.chunks(width).zip(out.chunks_mut(width)) {
            d::linear_to_srgb_u8_slice(s, o);
        }
        for (i, (&a, &b)) in out.iter().zip(&want).enumerate() {
            assert_eq!(a, b, "input={} index={i} width={width}", input[i]);
        }
    }
}
