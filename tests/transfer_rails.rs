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

#[test]
fn issue_32_chunk_and_tail_match() {
    let mut row = [0.125_f32; 17];
    srgb_to_linear_slice(&mut row);
    assert_eq!(row[0].to_bits(), row[16].to_bits());
}

#[test]
fn gamma_rails_and_subnormal_policy_agree() {
    let inputs = [
        f32::NAN,
        f32::from_bits(0x7f80_0001),
        -0.0,
        0.0,
        1.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::from_bits(128),
    ];
    for (scalar, slice) in [
        (
            gamma_to_linear as fn(f32, f32) -> f32,
            gamma_to_linear_slice as fn(&mut [f32], f32),
        ),
        (linear_to_gamma, linear_to_gamma_slice),
    ] {
        for exponent in [0.0, 2.4, 0.5, -0.5, 5.0] {
            let mut row = inputs.repeat(7);
            slice(&mut row, exponent);
            for (&input, output) in inputs.iter().cycle().zip(row) {
                let expected = scalar(input, exponent);
                if input.is_nan() {
                    assert!(output.is_nan() && expected.is_nan());
                } else {
                    assert_eq!(output.to_bits(), expected.to_bits(), "{input}, {exponent}");
                }
            }
        }
    }
}

#[test]
fn float_rgba_keeps_alpha_bits_and_the_unpremultiply_threshold() {
    let threshold = linear_srgb::UNPREMUL_ALPHA_THRESHOLD;
    let alphas = [
        -0.0,
        0.0,
        threshold,
        threshold.next_up(),
        0.5,
        f32::from_bits(0x7f80_0001),
    ];
    let pixels: Vec<_> = alphas
        .into_iter()
        .flat_map(|a| [0.25, 0.5, 1.0, a])
        .collect();
    for convert in [
        srgb_to_linear_premultiply_rgba_slice,
        unpremultiply_linear_to_srgb_rgba_slice,
    ] {
        let mut row = pixels.clone();
        convert(&mut row);
        for (input, output) in pixels.as_chunks::<4>().0.iter().zip(row.as_chunks::<4>().0) {
            assert_eq!(input[3].to_bits(), output[3].to_bits());
            let mut single = *input;
            convert(&mut single);
            for (&a, &b) in single[..3].iter().zip(&output[..3]) {
                assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
            }
        }
    }
    let mut row = pixels;
    unpremultiply_linear_to_srgb_rgba_slice(&mut row);
    for (pixel, alpha) in row.as_chunks::<4>().0.iter().zip(alphas) {
        let expected = if alpha > threshold { 1.0_f32 } else { 0.0 };
        assert_eq!(pixel[2].to_bits(), expected.to_bits(), "alpha {alpha}");
    }
}

#[test]
fn clamped_srgb_rails_agree_on_zero_and_white() {
    let rails = [
        f32::NAN,
        f32::from_bits(0x7f80_0001),
        -0.0,
        0.0,
        1.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
    ];
    let expected = [0.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0].map(f32::to_bits);
    for scalar in [
        srgb_to_linear,
        linear_to_srgb,
        linear_srgb::precise::srgb_to_linear,
        linear_srgb::precise::linear_to_srgb,
    ] {
        assert_eq!(rails.map(scalar).map(f32::to_bits), expected);
    }
    for scalar in [
        linear_srgb::precise::srgb_to_linear_f64,
        linear_srgb::precise::linear_to_srgb_f64,
    ] {
        assert_eq!(
            rails.map(|x| scalar(f64::from(x)) as f32).map(f32::to_bits),
            expected
        );
    }
    #[cfg(feature = "iec")]
    {
        for scalar in [
            linear_srgb::iec::srgb_to_linear,
            linear_srgb::iec::linear_to_srgb,
        ] {
            assert_eq!(rails.map(scalar).map(f32::to_bits), expected);
        }
        for scalar in [
            linear_srgb::iec::srgb_to_linear_f64,
            linear_srgb::iec::linear_to_srgb_f64,
        ] {
            assert_eq!(
                rails.map(|x| scalar(f64::from(x)) as f32).map(f32::to_bits),
                expected
            );
        }
    }
    for slice in [srgb_to_linear_slice, linear_to_srgb_slice] {
        let mut row = rails.repeat(7);
        slice(&mut row);
        assert!(
            row.iter()
                .map(|v| v.to_bits())
                .eq(expected.into_iter().cycle().take(row.len()))
        );
    }

    // The positive subnormal toe is still evaluated, rather than flushed.
    let tiny = f32::from_bits(128);
    for slice in [srgb_to_linear_slice, linear_to_srgb_slice] {
        let mut row = [tiny; 17];
        slice(&mut row);
        assert!(row[0].is_subnormal());
        assert_eq!(row[0].to_bits(), row[16].to_bits());
    }
}

#[test]
fn signed_extended_srgb_rails() {
    for (scalar, slice) in [
        (
            linear_srgb::precise::srgb_to_linear_extended as fn(f32) -> f32,
            srgb_to_linear_extended_slice as fn(&mut [f32]),
        ),
        (
            linear_srgb::precise::linear_to_srgb_extended as fn(f32) -> f32,
            linear_to_srgb_extended_slice as fn(&mut [f32]),
        ),
    ] {
        let rails = [f32::NAN, -0.0, 0.0, f32::NEG_INFINITY, f32::INFINITY];
        let mut row = rails.repeat(7);
        slice(&mut row);
        for (&input, output) in rails.iter().cycle().zip(row) {
            let want = scalar(input);
            if input.is_nan() {
                assert!(want.is_nan() && output.is_nan());
            } else {
                assert_eq!(output.to_bits(), want.to_bits(), "extended rail {input}");
            }
        }
    }
}

#[cfg(feature = "transfer")]
#[test]
fn hdr_scalar_and_slice_match_all_positions() {
    use linear_srgb::default as tf;
    let rails = [
        f32::NAN,
        f32::from_bits(0xff80_0001),
        -0.0,
        0.0,
        1.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::from_bits(128),
        -f32::from_bits(128),
        1e-8,
        1e-7,
        1e-6,
        9e-5,
        0.018_053_97,
        0.08124286,
        1.0 / 12.0,
        0.02,
        0.125,
        0.25,
        0.5,
        0.9,
        f32::MAX,
    ];
    for (name, scalar, slice, nan_becomes_zero) in [
        (
            "BT.709 decode",
            tf::bt709_to_linear as fn(f32) -> f32,
            bt709_to_linear_slice as fn(&mut [f32]),
            false,
        ),
        (
            "BT.709 encode",
            tf::linear_to_bt709,
            linear_to_bt709_slice,
            false,
        ),
        ("PQ decode", tf::pq_to_linear, pq_to_linear_slice, false),
        ("PQ encode", tf::linear_to_pq, linear_to_pq_slice, false),
        ("HLG decode", tf::hlg_to_linear, hlg_to_linear_slice, true),
        ("HLG encode", tf::linear_to_hlg, linear_to_hlg_slice, true),
    ] {
        // Include mixed exceptional and normal lanes and a partial final block.
        let mut input = rails.repeat(7);
        input.extend((0..=1024).map(|i| i as f32 / 1024.0));
        let mut row = input.clone();
        slice(&mut row);
        for (input, output) in input.into_iter().zip(row) {
            let expect = scalar(input);
            let mut single = [input];
            slice(&mut single);
            if input.is_nan() && !nan_becomes_zero {
                assert!(
                    expect.is_nan() && output.is_nan() && single[0].is_nan(),
                    "{name}"
                );
            } else {
                if input.is_nan() {
                    assert_eq!(expect.to_bits(), 0, "{name}");
                }
                assert_eq!(
                    single[0].to_bits(),
                    expect.to_bits(),
                    "{name} one sample, input {input}"
                );
                assert_eq!(
                    output.to_bits(),
                    expect.to_bits(),
                    "{name} mixed row, input {input}"
                );
            }
        }
    }
}

#[cfg(feature = "transfer")]
#[test]
fn pq_batches_handle_dark_toes_and_nan_at_every_position() {
    let nan = f32::from_bits(0xffc0_3210);
    for (name, fast, scalar, precise, samples, max_rel, max_abs) in [
        (
            "pq_to_linear",
            pq_to_linear_slice as fn(&mut [f32]),
            linear_srgb::default::pq_to_linear as fn(f32) -> f32,
            linear_srgb::precise::pq_to_linear_precise_slice as fn(&mut [f32]),
            [0.0, 0.005, 0.0052051, 0.0199, 1.0, f32::INFINITY, nan],
            2.5e-5_f64,
            1e-10_f64,
        ),
        (
            "linear_to_pq",
            linear_to_pq_slice as fn(&mut [f32]),
            linear_srgb::default::linear_to_pq as fn(f32) -> f32,
            linear_srgb::precise::linear_to_pq_precise_slice as fn(&mut [f32]),
            [
                0.0,
                1e-20,
                1e-8,
                (1e-7_f32).next_down(),
                1.0,
                f32::INFINITY,
                nan,
            ],
            5e-5_f64,
            1e-7_f64,
        ),
    ] {
        let mut reference = samples;
        precise(&mut reference);
        assert_eq!(reference[0].to_bits(), 0);
        assert!(reference[1] > 0.0);
        assert_eq!(reference[4..6], [1.0; 2]);
        assert_eq!(reference[6].to_bits(), nan.to_bits());

        let expected: [f32; 7] = samples.map(scalar);
        assert_eq!(expected[0].to_bits(), 0, "{name}: zero");
        assert!(expected[1] > 0.0, "{name}: dark toe positive");
        assert_eq!(expected[4..6], [1.0; 2], "{name}: clip high");
        assert_eq!(expected[6].to_bits(), nan.to_bits(), "{name}: NaN payload");

        for (i, ((&inp, &got), &want)) in samples
            .iter()
            .zip(expected.iter())
            .zip(reference.iter())
            .enumerate()
        {
            if inp.is_nan() {
                assert_eq!(got.to_bits(), nan.to_bits());
            } else {
                let err = (f64::from(got) - f64::from(want)).abs();
                let rel = if want > 0.0 {
                    err / f64::from(want)
                } else {
                    err
                };
                assert!(
                    err <= max_abs && rel <= max_rel,
                    "{name}[{i}] input={inp:?}: got={got:?} ref={want:?} abs={err:e} rel={rel:e}"
                );
            }
        }

        let mut row = samples.repeat(13);
        fast(&mut row);
        for chunk in row.chunks_exact(samples.len()) {
            assert!(
                chunk
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name}: slice disagrees with scalar"
            );
        }
        for (input, want) in samples.into_iter().zip(expected) {
            let mut single = [input];
            fast(&mut single);
            assert_eq!(single[0].to_bits(), want.to_bits(), "{name} input {input}");
        }
    }
}

#[cfg(feature = "transfer")]
#[test]
fn pq_oetf_uses_the_small_fit_within_one_code() {
    let floor = 1e-7_f32;
    let inputs = [
        floor.next_down(),
        floor,
        floor.next_up(),
        1e-6,
        1e-5,
        9e-5,
        1e-4,
    ];
    let mut reference = inputs;
    linear_srgb::default::linear_to_pq_precise_slice(&mut reference);
    let mut reference_precise = inputs;
    linear_srgb::precise::linear_to_pq_precise_slice(&mut reference_precise);
    assert_eq!(
        reference.map(f32::to_bits),
        reference_precise.map(f32::to_bits)
    );
    let mut actual = inputs.repeat(5);
    linear_to_pq_slice(&mut actual);
    let code = |x: f32| (f64::from(x) * 65535.0 + 0.5) as i32;

    assert!(
        (f64::from(actual[0]) - f64::from(reference[0])).abs() <= 1e-7,
        "strictly below the floor stays accurate vs f64 reference"
    );
    assert_ne!(
        actual[1].to_bits(),
        reference[1].to_bits(),
        "the floor enters the fit"
    );
    for (i, actual) in actual.into_iter().enumerate() {
        let lane = i % inputs.len();
        assert!(
            (code(actual) - code(reference[lane])).abs() <= 1,
            "PQ OETF input {}",
            inputs[lane]
        );
        assert_eq!(
            actual.to_bits(),
            linear_srgb::default::linear_to_pq(inputs[lane]).to_bits()
        );
    }
}

#[cfg(feature = "transfer")]
#[test]
fn pq_mixed_dark_toe_and_bright_chunks_match_scalar_and_preserve_alpha() {
    let nan = f32::from_bits(0xffc0_4321);
    let mut base = Vec::with_capacity(131);
    // Chunk 0 triggers the dark-toe fallback while subsequent chunks stay in the fast polynomial domain.
    base.extend_from_slice(&[0.0, 0.005, 0.015, 0.5]);
    for i in 4..131 {
        base.push(0.1 + (i as f32) * 0.0065);
    }
    base[67] = nan;
    base[96] = 1e-9;

    for (slice_fn, rgba_fn, scalar_fn) in [
        (
            pq_to_linear_slice as fn(&mut [f32]),
            pq_to_linear_rgba_slice as fn(&mut [f32]),
            linear_srgb::default::pq_to_linear as fn(f32) -> f32,
        ),
        (
            linear_to_pq_slice as fn(&mut [f32]),
            linear_to_pq_rgba_slice as fn(&mut [f32]),
            linear_srgb::default::linear_to_pq as fn(f32) -> f32,
        ),
    ] {
        let mut flat = base.clone();
        slice_fn(&mut flat);
        for (i, (&inp, &got)) in base.iter().zip(flat.iter()).enumerate() {
            let want = scalar_fn(inp);
            assert_eq!(got.to_bits(), want.to_bits(), "flat lane {i} input {inp}");
        }

        // RGBA: length 132 (33 pixels = 8 full 16-float chunks + 1 pixel tail).
        let mut rgba = Vec::with_capacity(132);
        for px in 0..33 {
            let a = (px as f32) * 0.03 + 0.01;
            let r = base[(px * 3) % base.len()];
            let g = base[(px * 3 + 1) % base.len()];
            let b = base[(px * 3 + 2) % base.len()];
            rgba.extend_from_slice(&[r, g, b, a]);
        }
        let orig_rgba = rgba.clone();
        rgba_fn(&mut rgba);
        for (px, (orig, got)) in orig_rgba
            .as_chunks::<4>()
            .0
            .iter()
            .zip(rgba.as_chunks::<4>().0)
            .enumerate()
        {
            for c in 0..3 {
                let want = scalar_fn(orig[c]);
                assert_eq!(
                    got[c].to_bits(),
                    want.to_bits(),
                    "rgba px {px} ch {c} input {}",
                    orig[c]
                );
            }
            assert_eq!(
                got[3].to_bits(),
                orig[3].to_bits(),
                "rgba px {px} alpha altered"
            );
        }
    }
}

#[test]
fn srgb_slice_into_matches_in_place_across_lengths_and_rails() {
    let rails = [
        -65.0,
        -9.0,
        -0.1,
        -0.0,
        0.0,
        f32::from_bits(128),
        0.003,
        0.04,
        0.125,
        0.5,
        0.999,
        1.0,
        1.5,
        9.0,
        65.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0xff80_1234),
    ];

    for len in 0..=67 {
        let input: Vec<f32> = (0..len).map(|i| rails[i % rails.len()]).collect();
        for (name, inplace, into) in [
            (
                "srgb_to_linear",
                srgb_to_linear_slice as fn(&mut [f32]),
                srgb_to_linear_slice_into as fn(&[f32], &mut [f32]),
            ),
            (
                "linear_to_srgb",
                linear_to_srgb_slice,
                linear_to_srgb_slice_into,
            ),
            (
                "srgb_to_linear_extended",
                srgb_to_linear_extended_slice,
                srgb_to_linear_extended_slice_into,
            ),
            (
                "linear_to_srgb_extended",
                linear_to_srgb_extended_slice,
                linear_to_srgb_extended_slice_into,
            ),
        ] {
            let mut expected = input.clone();
            inplace(&mut expected);
            let mut actual = vec![f32::from_bits(0xdead_beef); len];
            into(&input, &mut actual);
            for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
                if a.is_nan() && b.is_nan() {
                    continue;
                }
                assert_eq!(a.to_bits(), b.to_bits(), "{name} len={len} idx={i}");
            }
        }

        let sentinel = f32::from_bits(0x7fc0_beef);
        for (name, inplace, into) in [
            (
                "srgb_to_linear_rgba",
                srgb_to_linear_rgba_slice as fn(&mut [f32]),
                srgb_to_linear_rgba_slice_into as fn(&[f32], &mut [f32]),
            ),
            (
                "linear_to_srgb_rgba",
                linear_to_srgb_rgba_slice,
                linear_to_srgb_rgba_slice_into,
            ),
            (
                "srgb_to_linear_extended_rgba",
                srgb_to_linear_extended_rgba_slice,
                srgb_to_linear_extended_rgba_slice_into,
            ),
            (
                "linear_to_srgb_extended_rgba",
                linear_to_srgb_extended_rgba_slice,
                linear_to_srgb_extended_rgba_slice_into,
            ),
        ] {
            let mut expected = input.clone();
            inplace(&mut expected);
            let mut actual = vec![sentinel; len];
            into(&input, &mut actual);
            let complete = len / 4 * 4;
            for i in 0..complete {
                if i % 4 == 3 {
                    // Alpha must be preserved bit-for-bit including NaN payloads.
                    assert_eq!(
                        actual[i].to_bits(),
                        input[i].to_bits(),
                        "{name} alpha len={len} idx={i}"
                    );
                } else if actual[i].is_nan() && expected[i].is_nan() {
                    continue;
                } else {
                    assert_eq!(
                        actual[i].to_bits(),
                        expected[i].to_bits(),
                        "{name} len={len} idx={i}"
                    );
                }
            }
            for (i, &v) in actual.iter().enumerate().skip(complete) {
                assert_eq!(
                    v.to_bits(),
                    sentinel.to_bits(),
                    "{name} trailing lane touched len={len} idx={i}"
                );
            }
        }
    }
}

#[test]
#[should_panic]
fn srgb_to_linear_slice_into_panics_on_length_mismatch() {
    let mut dst = [0.0f32; 3];
    srgb_to_linear_slice_into(&[0.5f32; 4], &mut dst);
}

#[test]
#[should_panic]
fn srgb_to_linear_rgba_slice_into_panics_on_length_mismatch() {
    let mut dst = [0.0f32; 4];
    srgb_to_linear_rgba_slice_into(&[0.5f32; 8], &mut dst);
}

#[cfg(feature = "transfer")]
#[test]
fn pq_dark_toe_f32_magemath_accuracy_scalar_parity_and_seam_monotonicity() {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 32.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 128.0;
    const C3: f64 = 2392.0 / 128.0;
    let ref_eotf = |v: f32| -> f32 {
        if v.is_nan() {
            return v;
        }
        if v <= 0.0 {
            return 0.0;
        }
        if v >= 1.0 {
            return 1.0;
        }
        let vp = f64::from(v).powf(1.0 / M2);
        (((vp - C1).max(0.0) / (C2 - C3 * vp)).powf(1.0 / M1)) as f32
    };
    let ref_oetf = |v: f32| -> f32 {
        if v.is_nan() {
            return v;
        }
        if v <= 0.0 {
            return 0.0;
        }
        if v >= 1.0 {
            return 1.0;
        }
        let vp = f64::from(v).powf(M1);
        ((C1 + C2 * vp) / (1.0 + C3 * vp)).powf(M2) as f32
    };
    let ref_oetf_f64 = |y: f64| -> f64 {
        if y <= 0.0 {
            return 0.0;
        }
        let vp = y.powf(M1);
        ((C1 + C2 * vp) / (1.0 + C3 * vp)).powf(M2)
    };

    // 1. EOTF dark toe: 65,536 samples across [7.313e-7, 0.02) + rails & NaN payload.
    let mut dark = Vec::with_capacity(65_546);
    dark.extend_from_slice(&[
        -1.0,
        -0.0,
        0.0,
        f32::from_bits(1),
        7.309_559e-7,
        7.313e-7,
        7.313e-7_f32.next_up(),
        7.4e-7,
        1e-6,
        f32::from_bits(0xffc0_1234),
    ]);
    let lo = 7.313e-7_f32.next_up().to_bits();
    let hi = 0.02_f32.to_bits();
    let step = (hi - lo) / 65_536;
    for k in 0..65_536u32 {
        dark.push(f32::from_bits(lo + k * step));
    }

    let mut via_precise = dark.clone();
    linear_srgb::precise::pq_to_linear_precise_slice(&mut via_precise);
    let mut via_fast = dark.clone();
    pq_to_linear_slice(&mut via_fast);

    let mut max_eotf_rel = 0.0_f64;
    let mut max_eotf_abs = 0.0_f64;
    let mut max_eotf_rt_u16 = 0.0_f64;
    for (i, ((&inp, &p), &f)) in dark
        .iter()
        .zip(via_precise.iter())
        .zip(via_fast.iter())
        .enumerate()
    {
        let want = ref_eotf(inp);
        assert_eq!(
            p.to_bits(),
            want.to_bits(),
            "pq_to_linear_precise_slice mismatch at {i} (input={inp:?})"
        );
        assert_eq!(
            f.to_bits(),
            linear_srgb::default::pq_to_linear(inp).to_bits(),
            "pq_to_linear_slice != scalar at {i} (input={inp:?})"
        );
        if inp.is_nan() {
            assert_eq!(f.to_bits(), inp.to_bits());
        } else if inp <= 7.313e-7 {
            assert_eq!(f.to_bits(), 0);
        } else {
            let err = (f64::from(f) - f64::from(want)).abs();
            max_eotf_abs = max_eotf_abs.max(err);
            if want > 0.0 {
                max_eotf_rel = max_eotf_rel.max(err / f64::from(want));
            }
            let rt_u16 = (ref_oetf_f64(f64::from(f)) - f64::from(inp)).abs() * 65535.0;
            max_eotf_rt_u16 = max_eotf_rt_u16.max(rt_u16);
        }
    }
    assert!(
        max_eotf_rel <= 1e-4 && max_eotf_abs <= 1e-10 && max_eotf_rt_u16 <= 0.02,
        "PQ EOTF dark toe error too large: max_rel={max_eotf_rel:e} max_abs={max_eotf_abs:e} max_rt_u16={max_eotf_rt_u16}"
    );

    // 2. OETF dark toe: 65,536 samples across (0, 1e-7) including subnormals + rails.
    let mut oetf_dark = Vec::with_capacity(65_542);
    oetf_dark.extend_from_slice(&[
        -1.0,
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::from_bits(0xffc0_5678),
    ]);
    let o_lo = 1u32;
    let o_hi = 1e-7_f32.to_bits();
    let o_step = (o_hi - o_lo) / 65_536;
    for k in 0..65_536u32 {
        oetf_dark.push(f32::from_bits(o_lo + k * o_step));
    }

    let mut o_precise = oetf_dark.clone();
    linear_srgb::precise::linear_to_pq_precise_slice(&mut o_precise);
    let mut o_fast = oetf_dark.clone();
    linear_to_pq_slice(&mut o_fast);

    let mut max_oetf_abs = 0.0_f64;
    let mut max_oetf_rel = 0.0_f64;
    for (i, ((&inp, &p), &f)) in oetf_dark
        .iter()
        .zip(o_precise.iter())
        .zip(o_fast.iter())
        .enumerate()
    {
        let want = ref_oetf(inp);
        assert_eq!(
            p.to_bits(),
            want.to_bits(),
            "linear_to_pq_precise_slice mismatch at {i} (input={inp:?})"
        );
        assert_eq!(
            f.to_bits(),
            linear_srgb::default::linear_to_pq(inp).to_bits(),
            "linear_to_pq_slice != scalar at {i} (input={inp:?})"
        );
        if inp.is_nan() {
            assert_eq!(f.to_bits(), inp.to_bits());
        } else if inp <= 0.0 {
            assert_eq!(f.to_bits(), 0);
        } else {
            let err = (f64::from(f) - f64::from(want)).abs();
            max_oetf_abs = max_oetf_abs.max(err);
            max_oetf_rel = max_oetf_rel.max(err / f64::from(want));
        }
    }
    assert!(
        max_oetf_abs <= 1e-7 && max_oetf_rel <= 1e-4,
        "PQ OETF dark toe error too large: max_abs={max_oetf_abs:e} max_rel={max_oetf_rel:e}"
    );

    // 3. Seam continuity at 0.02 (EOTF) and 1e-7 (OETF): no big step-down across the split.
    for (name, f, seam) in [
        (
            "pq_to_linear",
            pq_to_linear_slice as fn(&mut [f32]),
            0.02_f32,
        ),
        (
            "linear_to_pq",
            linear_to_pq_slice as fn(&mut [f32]),
            1e-7_f32,
        ),
    ] {
        let center = seam.to_bits();
        let mut out: Vec<f32> = (0..=2048u32)
            .map(|i| f32::from_bits(center - 1024 + i))
            .collect();
        f(&mut out);
        let seam_drop = out[1023] - out[1024];
        assert!(
            seam_drop <= out[1024] * 1e-4,
            "{name} stepped down across seam {seam}: {} -> {}",
            out[1023],
            out[1024]
        );
    }

    // 4. Full [0, 1] sweep for `pq_to_linear_precise_slice`.
    let full: Vec<f32> = (0..=65_535).map(|i| i as f32 / 65_535.0).collect();
    let mut got = full.clone();
    linear_srgb::precise::pq_to_linear_precise_slice(&mut got);
    for (&inp, &out) in full.iter().zip(got.iter()) {
        assert_eq!(
            out.to_bits(),
            ref_eotf(inp).to_bits(),
            "pq_to_linear_precise_slice full-range mismatch (input={inp:?})"
        );
    }
}
