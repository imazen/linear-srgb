use linear_srgb::default as d;

#[test]
fn rgba_alpha_quantization_rounds_the_input_to_nearest() {
    for scale in [255.0f64, 65535.0] {
        let mut alpha = vec![
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -0.0,
            0.0,
            1.0,
            2.0,
        ];
        for code in 0..scale as u32 {
            let center = ((code as f64 + 0.5) / scale) as f32;
            alpha.extend([center.next_down(), center, center.next_up()]);
        }
        let input: Vec<f32> = alpha.iter().flat_map(|&a| [0.0, 0.0, 0.0, a]).collect();
        if scale == 255.0 {
            for f in [
                d::linear_to_srgb_u8_rgba_slice,
                d::unpremultiply_linear_to_srgb_u8_rgba_slice,
            ] {
                let mut output = vec![0; input.len()];
                f(&input, &mut output);
                for (&a, p) in alpha.iter().zip(output.as_chunks::<4>().0) {
                    assert_eq!(
                        p[3],
                        (f64::from(a).clamp(0.0, 1.0) * scale + 0.5) as u8,
                        "alpha={a}"
                    );
                }
            }
        } else {
            for f in [
                d::linear_to_srgb_u16_rgba_slice,
                d::linear_to_srgb_u16_rgba_slice_fast,
            ] {
                let mut output = vec![0; input.len()];
                f(&input, &mut output);
                for (&a, p) in alpha.iter().zip(output.as_chunks::<4>().0) {
                    assert_eq!(
                        p[3],
                        (f64::from(a).clamp(0.0, 1.0) * scale + 0.5) as u16,
                        "alpha={a}"
                    );
                }
            }
        }
    }
}
