use linear_srgb::default::*;

#[test]
fn s_rgb_clamps_white_and_keeps_extended_zero_and_tails() {
    let clamped = [-2.0, 0.0, 1.0, 1.1, f32::INFINITY];
    for convert in [srgb_to_linear_slice, linear_to_srgb_slice] {
        let mut row = clamped;
        convert(&mut row);
        assert_eq!(
            row.map(f32::to_bits),
            [
                0,
                0,
                1.0_f32.to_bits(),
                1.0_f32.to_bits(),
                1.0_f32.to_bits()
            ]
        );
    }

    let samples = [-0.0, -9.0, 9.0, -65.0, 65.0, f32::INFINITY];
    let original_alpha = f32::from_bits(0xff80_1234);
    for (plain, rgba, precise, limit) in [
        (
            srgb_to_linear_extended_slice as fn(&mut [f32]),
            srgb_to_linear_extended_rgba_slice as fn(&mut [f32]),
            linear_srgb::precise::srgb_to_linear_extended as fn(f32) -> f32,
            8.0,
        ),
        (
            linear_to_srgb_extended_slice as fn(&mut [f32]),
            linear_to_srgb_extended_rgba_slice as fn(&mut [f32]),
            linear_srgb::precise::linear_to_srgb_extended as fn(f32) -> f32,
            64.0,
        ),
    ] {
        let mut expected = samples;
        plain(&mut expected);
        assert_eq!(expected[0].to_bits(), (-0.0_f32).to_bits());
        for (input, output) in samples.into_iter().zip(expected) {
            if input.abs() > limit {
                assert_eq!(output.to_bits(), precise(input).to_bits());
            }
            let mut row = [input, input, input, original_alpha].repeat(5);
            rgba(&mut row);
            for pixel in row.as_chunks::<4>().0 {
                assert_eq!(pixel[0].to_bits(), output.to_bits());
                assert_eq!(pixel[1].to_bits(), output.to_bits());
                assert_eq!(pixel[2].to_bits(), output.to_bits());
                assert_eq!(pixel[3].to_bits(), original_alpha.to_bits());
            }
        }
    }
}

#[test]
fn gamma_preserves_rails_and_row_position() {
    let values = [0.0, -0.0, 0.02, 1e-40, 0.123, 1.0, 2.0, f32::NAN];
    for exponent in [0.0, -0.25, 2.4, 1.0 / 2.4, 5.0] {
        let mut full = values.repeat(11);
        gamma_to_linear_slice(&mut full, exponent);
        for (i, input) in values.into_iter().enumerate() {
            let mut single = [input];
            gamma_to_linear_slice(&mut single, exponent);
            assert_eq!(
                single[0].to_bits(),
                full[i].to_bits(),
                "{input}, {exponent}"
            );
            assert_eq!(
                single[0].to_bits(),
                full[80 + i].to_bits(),
                "{input}, {exponent}"
            );
        }
        assert_eq!(full[0].to_bits(), 0);
        assert_eq!(full[1].to_bits(), 0);
        assert_eq!(full[5], 1.0);
        assert_eq!(full[6], 1.0);
        assert!(full[7].is_nan());
    }
}

#[cfg(feature = "transfer")]
#[test]
fn pq_batches_handle_dark_toes_and_nan_at_every_position() {
    let nan = f32::from_bits(0xffc0_3210);
    for (fast, precise, samples) in [
        (
            pq_to_linear_slice as fn(&mut [f32]),
            linear_srgb::tf::pq_to_linear_precise_slice as fn(&mut [f32]),
            [0.0, 0.005, 0.0052051, 0.0199, 1.0, f32::INFINITY, nan],
        ),
        (
            linear_to_pq_slice as fn(&mut [f32]),
            linear_srgb::tf::linear_to_pq_precise_slice as fn(&mut [f32]),
            [0.0, 1e-20, 1e-8, 9e-5, 1.0, f32::INFINITY, nan],
        ),
    ] {
        let mut expected = samples;
        precise(&mut expected);
        assert_eq!(expected[0].to_bits(), 0);
        assert!(expected[1] > 0.0);
        assert_eq!(expected[4..6], [1.0; 2]);
        assert_eq!(expected[6].to_bits(), nan.to_bits());

        let mut row = samples.repeat(13);
        fast(&mut row);
        for chunk in row.chunks_exact(samples.len()) {
            assert!(
                chunk
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
        for (input, want) in samples.into_iter().zip(expected) {
            let mut single = [input];
            fast(&mut single);
            assert_eq!(single[0].to_bits(), want.to_bits(), "input {input}");
        }
    }
}
