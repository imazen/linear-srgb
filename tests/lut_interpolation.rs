use linear_srgb::lut::lut_interp_linear_float;

#[test]
fn interpolation_preserves_small_values_and_knots() {
    for bits in (1..=1.0f32.to_bits()).step_by(8191) {
        let x = f32::from_bits(bits);
        assert_eq!(lut_interp_linear_float(x, &[0.0, 1.0]).to_bits(), bits);
    }
    for table in [[0.0, 0.25, 1.0], [0.25, 0.5, 0.75]] {
        for (i, &value) in table.iter().enumerate() {
            assert_eq!(lut_interp_linear_float(i as f32 / 2.0, &table), value);
        }
    }
}
