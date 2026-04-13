//! Analysis of practical value ranges during BT.2020 PQ → sRGB conversion,
//! and evaluation of rational polynomial extrapolation beyond [0, 1].

use linear_srgb::precise::{linear_to_srgb_extended, srgb_to_linear_extended};

// BT.2020 → BT.709 (sRGB primaries) gamut matrix.
// Source: zenpixels registry, Bradford-adapted, matches CSS Color 4 spec.
const BT2020_TO_SRGB: [[f64; 3]; 3] = [
    [1.6604910, -0.5876411, -0.0728499],
    [-0.1245505, 1.1328999, -0.0083494],
    [-0.0181508, -0.1005789, 1.1187297],
];

/// PQ EOTF (ST 2084): PQ signal [0,1] → linear light [0,1] (normalized to 10000 nits).
fn pq_eotf(v: f64) -> f64 {
    if v <= 0.0 {
        return 0.0;
    }
    const M1: f64 = 0.1593017578125;
    const M2: f64 = 78.84375;
    const C1: f64 = 0.8359375;
    const C2: f64 = 18.8515625;
    const C3: f64 = 18.6875;

    let vp = v.powf(1.0 / M2);
    let num = (vp - C1).max(0.0);
    let den = C2 - C3 * vp;
    if den <= 0.0 {
        return 1.0;
    }
    (num / den).powf(1.0 / M1)
}

/// sRGB OETF (C0-continuous): linear light → sRGB encoded.
fn srgb_oetf(linear: f64) -> f64 {
    const A: f64 = 0.055_010_718_947_586_6;
    const A_PLUS_1: f64 = 1.055_010_718_947_586_6;
    const LINEAR_THRESHOLD: f64 = 0.003_041_282_560_127_521;

    if linear < 0.0 {
        // Mirror for negative (extended sRGB)
        -srgb_oetf(-linear)
    } else if linear <= LINEAR_THRESHOLD {
        linear * 12.92
    } else {
        A_PLUS_1 * linear.powf(1.0 / 2.4) - A
    }
}

/// sRGB EOTF: sRGB encoded → linear light (extended range).
fn srgb_eotf(v: f64) -> f64 {
    const A: f64 = 0.055_010_718_947_586_6;
    const A_PLUS_1: f64 = 1.055_010_718_947_586_6;
    const SRGB_THRESHOLD: f64 = 12.92 * 0.003_041_282_560_127_521;

    if v < 0.0 {
        -srgb_eotf(-v)
    } else if v <= SRGB_THRESHOLD {
        v / 12.92
    } else {
        ((v + A) / A_PLUS_1).powf(2.4)
    }
}

fn mul_mv(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// ============================================================================
// Step 1: Determine the actual value ranges
// ============================================================================

#[test]
fn step1_bt2020_pq_to_srgb_value_ranges() {
    let mut linear_min = f64::MAX;
    let mut linear_max = f64::MIN;
    let mut encoded_min = f64::MAX;
    let mut encoded_max = f64::MIN;

    // Test all PQ levels for neutral (equal-energy) and pure primaries
    let test_colors: &[([f64; 3], &str)] = &[
        ([1.0, 0.0, 0.0], "red"),
        ([0.0, 1.0, 0.0], "green"),
        ([0.0, 0.0, 1.0], "blue"),
        ([1.0, 1.0, 0.0], "yellow"),
        ([1.0, 0.0, 1.0], "magenta"),
        ([0.0, 1.0, 1.0], "cyan"),
        ([1.0, 1.0, 1.0], "white"),
    ];

    println!("\n=== BT.2020 PQ → sRGB: Value range analysis ===\n");
    println!(
        "{:<10} {:<10} {:<14} {:<14} {:<14} {:<14}",
        "Color", "PQ level", "Lin sRGB min", "Lin sRGB max", "Enc sRGB min", "Enc sRGB max"
    );
    println!("{}", "-".repeat(80));

    for (base_color, name) in test_colors {
        let mut color_lin_min = f64::MAX;
        let mut color_lin_max = f64::MIN;
        let mut color_enc_min = f64::MAX;
        let mut color_enc_max = f64::MIN;

        // PQ signal from 0.0 to 1.0 in fine steps
        for i in 0..=1000 {
            let pq_signal = i as f64 / 1000.0;
            let linear_bt2020 = pq_eotf(pq_signal);

            // Scale the base color by the decoded linear value
            let rgb_2020 = [
                base_color[0] * linear_bt2020,
                base_color[1] * linear_bt2020,
                base_color[2] * linear_bt2020,
            ];

            // Apply BT.2020→sRGB gamut matrix in linear space
            let rgb_srgb_linear = mul_mv(&BT2020_TO_SRGB, rgb_2020);

            // Apply sRGB OETF to each channel
            for ch in 0..3 {
                let lin = rgb_srgb_linear[ch];
                let enc = srgb_oetf(lin);

                if lin < color_lin_min {
                    color_lin_min = lin;
                }
                if lin > color_lin_max {
                    color_lin_max = lin;
                }
                if enc < color_enc_min {
                    color_enc_min = enc;
                }
                if enc > color_enc_max {
                    color_enc_max = enc;
                }
            }
        }

        println!(
            "{:<10} {:<10} {:<14.6} {:<14.6} {:<14.6} {:<14.6}",
            name, "0..1", color_lin_min, color_lin_max, color_enc_min, color_enc_max
        );

        if color_lin_min < linear_min {
            linear_min = color_lin_min;
        }
        if color_lin_max > linear_max {
            linear_max = color_lin_max;
        }
        if color_enc_min < encoded_min {
            encoded_min = color_enc_min;
        }
        if color_enc_max > encoded_max {
            encoded_max = color_enc_max;
        }
    }

    println!("\n=== Summary ===");
    println!("Linear sRGB range: [{:.6}, {:.6}]", linear_min, linear_max);
    println!(
        "Encoded sRGB range: [{:.6}, {:.6}]",
        encoded_min, encoded_max
    );

    // Also report the specific worst-case colors
    println!("\n=== Worst-case per-channel at full PQ=1.0 ===\n");
    for (base_color, name) in test_colors {
        let linear_bt2020 = pq_eotf(1.0);
        let rgb_2020 = [
            base_color[0] * linear_bt2020,
            base_color[1] * linear_bt2020,
            base_color[2] * linear_bt2020,
        ];
        let rgb_srgb_linear = mul_mv(&BT2020_TO_SRGB, rgb_2020);
        let rgb_srgb_encoded = [
            srgb_oetf(rgb_srgb_linear[0]),
            srgb_oetf(rgb_srgb_linear[1]),
            srgb_oetf(rgb_srgb_linear[2]),
        ];
        println!(
            "{:<10} linear=({:>10.4}, {:>10.4}, {:>10.4})  encoded=({:>10.4}, {:>10.4}, {:>10.4})",
            name,
            rgb_srgb_linear[0],
            rgb_srgb_linear[1],
            rgb_srgb_linear[2],
            rgb_srgb_encoded[0],
            rgb_srgb_encoded[1],
            rgb_srgb_encoded[2]
        );
    }

    // Practical range that actually matters: what's the realistic range
    // for Display P3 content (phones) vs full BT.2020?
    println!("\n=== Display P3 → sRGB ranges (more common in practice) ===\n");
    // P3→sRGB matrix (D65 white, no chromatic adaptation needed)
    let p3_to_srgb: [[f64; 3]; 3] = [
        [1.2249402, -0.2249402, 0.0],
        [-0.0420570, 1.0420570, 0.0],
        [-0.0196376, -0.0786360, 1.0982736],
    ];
    for (base_color, name) in test_colors {
        // P3 doesn't use PQ normally, but test the pure gamut range
        // at linear=1.0 (fully saturated P3 primaries)
        let rgb_p3 = *base_color;
        let rgb_srgb_linear = mul_mv(&p3_to_srgb, rgb_p3);
        let rgb_srgb_encoded = [
            srgb_oetf(rgb_srgb_linear[0]),
            srgb_oetf(rgb_srgb_linear[1]),
            srgb_oetf(rgb_srgb_linear[2]),
        ];
        println!(
            "{:<10} linear=({:>10.6}, {:>10.6}, {:>10.6})  encoded=({:>10.6}, {:>10.6}, {:>10.6})",
            name,
            rgb_srgb_linear[0],
            rgb_srgb_linear[1],
            rgb_srgb_linear[2],
            rgb_srgb_encoded[0],
            rgb_srgb_encoded[1],
            rgb_srgb_encoded[2]
        );
    }
}

// ============================================================================
// Step 2: Evaluate polynomial fit feasibility
// ============================================================================

/// The existing rational polynomial coefficients and evaluator from src/rational_poly.rs,
/// but WITHOUT the clamp so we can evaluate outside [0, 1].
fn eval_rational_poly_5_unclamped(x: f32, p: [f32; 5], q: [f32; 5]) -> f32 {
    let x = x as f64;
    let yp = (p[4] as f64).mul_add(x, p[3] as f64);
    let yp = yp.mul_add(x, p[2] as f64);
    let yp = yp.mul_add(x, p[1] as f64);
    let yp = yp.mul_add(x, p[0] as f64);

    let yq = (q[4] as f64).mul_add(x, q[3] as f64);
    let yq = yq.mul_add(x, q[2] as f64);
    let yq = yq.mul_add(x, q[1] as f64);
    let yq = yq.mul_add(x, q[0] as f64);

    (yp / yq) as f32
}

// Copy the coefficients from src/rational_poly.rs
const S2L_P: [f32; 5] = [
    1.724_942_4e-2,
    8.335_514_7e-1,
    1.326_215_8e1,
    7.033_073_4e1,
    8.387_046e1,
];
const S2L_Q: [f32; 5] = [2.066_183e1, 9.917_607e1, 5.466_011e1, -7.183_806, 1.0];

const L2S_P: [f32; 5] = [
    -1.513_885e-2,
    1.167_372_8e-1,
    1.257_921_2e1,
    5.259_309_8e1,
    2.852_907_6e1,
];
const L2S_Q: [f32; 5] = [2.943_901_4e-1, 9.779_103, 4.726_487_7e1, 3.546_463_8e1, 1.0];

#[test]
fn step2_polynomial_extrapolation_error() {
    println!("\n=== sRGB→linear polynomial extrapolation (input = encoded sRGB) ===\n");
    println!(
        "{:<12} {:<18} {:<18} {:<18}",
        "Input", "Poly result", "Exact (powf)", "Abs error"
    );
    println!("{}", "-".repeat(70));

    // Test the sRGB→linear polynomial beyond [0, 1]
    let test_inputs_s2l: &[f32] = &[
        -0.5, -0.3, -0.1, 0.0, 0.5, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0,
    ];

    for &x in test_inputs_s2l {
        let poly = eval_rational_poly_5_unclamped(x, S2L_P, S2L_Q);
        let exact = srgb_eotf(x as f64) as f32;
        let err = (poly - exact).abs();
        println!(
            "{:<12.4} {:<18.10} {:<18.10} {:<18.10}",
            x, poly, exact, err
        );
    }

    println!("\n=== linear→sRGB polynomial extrapolation (input = sqrt(linear)) ===\n");
    println!(
        "{:<12} {:<12} {:<18} {:<18} {:<18}",
        "Linear", "sqrt(lin)", "Poly result", "Exact (powf)", "Abs error"
    );
    println!("{}", "-".repeat(80));

    // The L2S polynomial is evaluated on sqrt(linear), so test various linear values
    let test_inputs_l2s: &[f64] = &[0.0, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0];

    for &linear in test_inputs_l2s {
        let s = (linear as f32).sqrt();
        let poly = eval_rational_poly_5_unclamped(s, L2S_P, L2S_Q);
        let exact = srgb_oetf(linear) as f32;
        let err = (poly - exact).abs();
        println!(
            "{:<12.4} {:<12.6} {:<18.10} {:<18.10} {:<18.10}",
            linear, s, poly, exact, err
        );
    }

    // Detailed error sweep for values just above 1.0
    println!("\n=== S2L polynomial error near and beyond 1.0 (fine sweep) ===\n");
    println!(
        "{:<12} {:<18} {:<18} {:<12} {:<12}",
        "Input", "Poly", "Exact", "Abs err", "Rel err"
    );
    println!("{}", "-".repeat(70));

    for i in 90..=200 {
        let x = i as f32 / 100.0;
        let poly = eval_rational_poly_5_unclamped(x, S2L_P, S2L_Q);
        let exact = srgb_eotf(x as f64) as f32;
        let err = (poly - exact).abs();
        let rel = if exact.abs() > 1e-10 {
            err / exact.abs()
        } else {
            0.0
        };
        if i % 5 == 0 || i <= 105 {
            println!(
                "{:<12.4} {:<18.10} {:<18.10} {:<12.6e} {:<12.6e}",
                x, poly, exact, err, rel
            );
        }
    }

    // L2S detailed sweep
    println!("\n=== L2S polynomial error near and beyond 1.0 (fine sweep) ===\n");
    println!(
        "{:<12} {:<18} {:<18} {:<12} {:<12}",
        "Linear in", "Poly", "Exact", "Abs err", "Rel err"
    );
    println!("{}", "-".repeat(70));

    for i in 90..=300 {
        let linear = i as f64 / 100.0;
        let s = (linear as f32).sqrt();
        let poly = eval_rational_poly_5_unclamped(s, L2S_P, L2S_Q);
        let exact = srgb_oetf(linear) as f32;
        let err = (poly - exact).abs();
        let rel = if exact.abs() > 1e-10 {
            err / exact.abs()
        } else {
            0.0
        };
        if i % 10 == 0 || i <= 105 {
            println!(
                "{:<12.4} {:<18.10} {:<18.10} {:<12.6e} {:<12.6e}",
                linear, poly, exact, err, rel
            );
        }
    }
}

#[test]
fn step2_polynomial_denominator_zeros() {
    // Check if the S2L or L2S denominator crosses zero in the extended range,
    // which would cause poles (singularities).
    println!("\n=== Denominator behavior for S2L (sRGB→linear) ===\n");

    let mut s2l_denom_sign_changes = 0;
    let mut prev_sign = true;
    for i in -2000..=5000i32 {
        let x = i as f64 / 1000.0;
        let yq = (S2L_Q[4] as f64).mul_add(x, S2L_Q[3] as f64);
        let yq = yq.mul_add(x, S2L_Q[2] as f64);
        let yq = yq.mul_add(x, S2L_Q[1] as f64);
        let yq = yq.mul_add(x, S2L_Q[0] as f64);
        let sign = yq >= 0.0;
        if i > -2000 && sign != prev_sign {
            s2l_denom_sign_changes += 1;
            println!(
                "  S2L denom sign change near x={:.3}, Q(x)={:.6}",
                x as f64 / 1.0,
                yq
            );
        }
        prev_sign = sign;
    }
    println!(
        "S2L denominator sign changes in [-2, 5]: {}",
        s2l_denom_sign_changes
    );

    println!("\n=== Denominator behavior for L2S (linear→sRGB, input=sqrt(linear)) ===\n");
    let mut l2s_denom_sign_changes = 0;
    let mut prev_sign = true;
    for i in 0..=10000i32 {
        let x = i as f64 / 1000.0;
        let yq = (L2S_Q[4] as f64).mul_add(x, L2S_Q[3] as f64);
        let yq = yq.mul_add(x, L2S_Q[2] as f64);
        let yq = yq.mul_add(x, L2S_Q[1] as f64);
        let yq = yq.mul_add(x, L2S_Q[0] as f64);
        let sign = yq >= 0.0;
        if i > 0 && sign != prev_sign {
            l2s_denom_sign_changes += 1;
            println!("  L2S denom sign change near sqrt={:.3}, Q(s)={:.6}", x, yq);
        }
        prev_sign = sign;
    }
    println!(
        "L2S denominator sign changes in [0, 10] (sqrt domain): {}",
        l2s_denom_sign_changes
    );
}

#[test]
fn step2_verify_extended_functions_match_ground_truth() {
    // Verify that the existing precise::*_extended functions produce correct results
    // at various out-of-range values, confirming our ground truth is good.
    println!("\n=== Verify precise::*_extended against f64 ground truth ===\n");

    let test_values = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, -0.1, -0.5];
    for &v in &test_values {
        let f32_result = srgb_to_linear_extended(v);
        let f64_result = srgb_eotf(v as f64);
        let err = (f32_result as f64 - f64_result).abs();
        println!(
            "srgb_to_linear_extended({:>6.2}) = {:>12.8}  f64={:>12.8}  err={:.2e}",
            v, f32_result, f64_result, err
        );
    }

    let test_linear = [0.0f32, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0];
    for &v in &test_linear {
        let f32_result = linear_to_srgb_extended(v);
        let f64_result = srgb_oetf(v as f64);
        let err = (f32_result as f64 - f64_result).abs();
        println!(
            "linear_to_srgb_extended({:>6.2}) = {:>12.8}  f64={:>12.8}  err={:.2e}",
            v, f32_result, f64_result, err
        );
    }
}
