//! Comprehensive analysis of value ranges across all plausible gamut conversions,
//! and evaluation of polynomial accuracy at those ranges.
//!
//! This test determines whether the abs+sign SIMD approach (polynomial on |x|,
//! restore sign) produces acceptable error for real-world gamut conversions.

use linear_srgb::precise::{linear_to_srgb_extended, srgb_to_linear_extended};

// ============================================================================
// Gamut matrices (Bradford-adapted where white points differ)
// ============================================================================

/// BT.2020 → BT.709/sRGB (D65→D65, no chromatic adaptation).
const BT2020_TO_SRGB: [[f64; 3]; 3] = [
    [1.6604910021, -0.5876411388, -0.0728498633],
    [-0.1245504745, 1.1328998971, -0.0083494226],
    [-0.0181507634, -0.1005788980, 1.1187296614],
];

/// Display P3 → BT.709/sRGB (D65→D65, no chromatic adaptation).
const P3_TO_SRGB: [[f64; 3]; 3] = [
    [1.2249401763, -0.2249401763, 0.0000000000],
    [-0.0420569547, 1.0420569547, -0.0000000000],
    [-0.0196375546, -0.0786360456, 1.0982736001],
];

/// ProPhoto RGB → BT.709/sRGB (D50→D65, Bradford adaptation).
const PROPHOTO_TO_SRGB: [[f64; 3]; 3] = [
    [2.0343675435, -0.7276344742, -0.3067330693],
    [-0.2288267982, 1.2317533962, -0.0029265980],
    [-0.0085584243, -0.1532682035, 1.1618266279],
];

/// ACES AP0 → BT.709/sRGB (ACES D60→D65, Bradford adaptation).
/// AP0 contains imaginary colors; this is the widest possible gamut.
const ACES_AP0_TO_SRGB: [[f64; 3]; 3] = [
    [2.5216861867, -1.1341309882, -0.3875551985],
    [-0.2764799142, 1.3727190877, -0.0962391734],
    [-0.0153780650, -0.1529753359, 1.1683534008],
];

// ============================================================================
// Helper functions
// ============================================================================

fn mul_mv(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

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

/// sRGB OETF with sign extension: sign(v) * oetf(|v|).
fn srgb_oetf_extended(linear: f64) -> f64 {
    const A: f64 = 0.055_010_718_947_586_6;
    const A_PLUS_1: f64 = 1.055_010_718_947_586_6;
    const LINEAR_THRESHOLD: f64 = 0.003_041_282_560_127_521;

    if linear < 0.0 {
        -srgb_oetf_extended(-linear)
    } else if linear <= LINEAR_THRESHOLD {
        linear * 12.92
    } else {
        A_PLUS_1 * linear.powf(1.0 / 2.4) - A
    }
}

/// sRGB EOTF with sign extension: sign(v) * eotf(|v|).
fn srgb_eotf_extended(v: f64) -> f64 {
    const A: f64 = 0.055_010_718_947_586_6;
    const A_PLUS_1: f64 = 1.055_010_718_947_586_6;
    const SRGB_THRESHOLD: f64 = 12.92 * 0.003_041_282_560_127_521;

    if v < 0.0 {
        -srgb_eotf_extended(-v)
    } else if v <= SRGB_THRESHOLD {
        v / 12.92
    } else {
        ((v + A) / A_PLUS_1).powf(2.4)
    }
}

// Rational polynomial coefficients from src/rational_poly.rs
const S2L_P: [f64; 5] = [
    1.724_942_4e-2,
    8.335_514_7e-1,
    1.326_215_8e1,
    7.033_073_4e1,
    8.387_046e1,
];
const S2L_Q: [f64; 5] = [2.066_183e1, 9.917_607e1, 5.466_011e1, -7.183_806, 1.0];

const L2S_P: [f64; 5] = [
    -1.513_885e-2,
    1.167_372_8e-1,
    1.257_921_2e1,
    5.259_309_8e1,
    2.852_907_6e1,
];
const L2S_Q: [f64; 5] = [2.943_901_4e-1, 9.779_103, 4.726_487_7e1, 3.546_463_8e1, 1.0];

/// Evaluate degree-4 rational polynomial P(x)/Q(x) via Horner's method.
fn eval_rational_poly(x: f64, p: &[f64; 5], q: &[f64; 5]) -> f64 {
    let yp = p[4].mul_add(x, p[3]);
    let yp = yp.mul_add(x, p[2]);
    let yp = yp.mul_add(x, p[1]);
    let yp = yp.mul_add(x, p[0]);

    let yq = q[4].mul_add(x, q[3]);
    let yq = yq.mul_add(x, q[2]);
    let yq = yq.mul_add(x, q[1]);
    let yq = yq.mul_add(x, q[0]);

    yp / yq
}

// ============================================================================
// Test colors: primaries + secondaries (the gamut boundary)
// ============================================================================

const TEST_COLORS: [([f64; 3], &str); 7] = [
    ([1.0, 0.0, 0.0], "red"),
    ([0.0, 1.0, 0.0], "green"),
    ([0.0, 0.0, 1.0], "blue"),
    ([1.0, 1.0, 0.0], "yellow"),
    ([1.0, 0.0, 1.0], "magenta"),
    ([0.0, 1.0, 1.0], "cyan"),
    ([1.0, 1.0, 1.0], "white"),
];

/// Compute the min/max linear and encoded sRGB values for a gamut matrix
/// applied to test colors at a given linear scale.
fn compute_range(matrix: &[[f64; 3]; 3], scale: f64) -> (f64, f64, f64, f64) {
    let mut lin_min = f64::MAX;
    let mut lin_max = f64::MIN;
    let mut enc_min = f64::MAX;
    let mut enc_max = f64::MIN;

    for (color, _name) in &TEST_COLORS {
        let rgb_src = [color[0] * scale, color[1] * scale, color[2] * scale];
        let rgb_srgb_lin = mul_mv(matrix, rgb_src);
        for ch in 0..3 {
            let lin = rgb_srgb_lin[ch];
            let enc = srgb_oetf_extended(lin);
            lin_min = lin_min.min(lin);
            lin_max = lin_max.max(lin);
            enc_min = enc_min.min(enc);
            enc_max = enc_max.max(enc);
        }
    }
    (lin_min, lin_max, enc_min, enc_max)
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn comprehensive_gamut_ranges() {
    println!("\n=== Comprehensive gamut range analysis ===\n");
    println!(
        "{:<20} {:<6} {:>12} {:>12} {:>12} {:>12}",
        "Source", "Scale", "Lin min", "Lin max", "Enc min", "Enc max"
    );
    println!("{}", "-".repeat(80));

    let scenarios: &[(&str, &[[f64; 3]; 3])] = &[
        ("BT.2020", &BT2020_TO_SRGB),
        ("Display P3", &P3_TO_SRGB),
        ("ProPhoto", &PROPHOTO_TO_SRGB),
        ("ACES AP0", &ACES_AP0_TO_SRGB),
    ];

    let scales = [1.0, 2.0, 5.0, 10.0];

    let mut global_lin_min = f64::MAX;
    let mut global_lin_max = f64::MIN;
    let mut global_enc_min = f64::MAX;
    let mut global_enc_max = f64::MIN;

    // At-1.0 tracking for the "practical" range
    let mut practical_lin_min = f64::MAX;
    let mut practical_lin_max = f64::MIN;
    let mut practical_enc_min = f64::MAX;
    let mut practical_enc_max = f64::MIN;

    for (name, matrix) in scenarios {
        for &scale in &scales {
            let (lmin, lmax, emin, emax) = compute_range(matrix, scale);
            println!(
                "{:<20} {:<6.1} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
                name, scale, lmin, lmax, emin, emax
            );

            global_lin_min = global_lin_min.min(lmin);
            global_lin_max = global_lin_max.max(lmax);
            global_enc_min = global_enc_min.min(emin);
            global_enc_max = global_enc_max.max(emax);

            if scale == 1.0 {
                practical_lin_min = practical_lin_min.min(lmin);
                practical_lin_max = practical_lin_max.max(lmax);
                practical_enc_min = practical_enc_min.min(emin);
                practical_enc_max = practical_enc_max.max(emax);
            }
        }
        println!();
    }

    // BT.2020 PQ at PQ signal 1.0 (same as BT.2020 at linear 1.0 since PQ(1.0)=1.0)
    let pq_linear = pq_eotf(1.0);
    let (lmin, lmax, emin, emax) = compute_range(&BT2020_TO_SRGB, pq_linear);
    println!(
        "{:<20} {:<6} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
        "BT.2020 PQ@1.0",
        format!("{:.4}", pq_linear),
        lmin,
        lmax,
        emin,
        emax
    );
    practical_lin_min = practical_lin_min.min(lmin);
    practical_lin_max = practical_lin_max.max(lmax);
    practical_enc_min = practical_enc_min.min(emin);
    practical_enc_max = practical_enc_max.max(emax);

    println!("\n=== Summary ===");
    println!("Practical range (all sources at linear 1.0 + BT.2020 PQ):");
    println!(
        "  Linear sRGB: [{:.6}, {:.6}]",
        practical_lin_min, practical_lin_max
    );
    println!(
        "  Encoded sRGB: [{:.6}, {:.6}]",
        practical_enc_min, practical_enc_max
    );
    println!("\nScene-referred range (all sources up to 10x):");
    println!(
        "  Linear sRGB: [{:.6}, {:.6}]",
        global_lin_min, global_lin_max
    );
    println!(
        "  Encoded sRGB: [{:.6}, {:.6}]",
        global_enc_min, global_enc_max
    );

    // The polynomial must handle abs values up to these maximums
    let max_abs_encoded = practical_enc_min.abs().max(practical_enc_max.abs());
    let max_abs_linear = practical_lin_min.abs().max(practical_lin_max.abs());
    println!("\nFor abs+sign approach (practical range):");
    println!(
        "  S2L polynomial max |encoded input|: {:.6}",
        max_abs_encoded
    );
    println!("  L2S polynomial max |linear input|: {:.6}", max_abs_linear);
    println!(
        "  L2S polynomial max sqrt(|linear|): {:.6}",
        max_abs_linear.sqrt()
    );
}

#[test]
fn polynomial_accuracy_at_practical_ranges() {
    // The polynomial fitted for [0, 1]. Test extrapolation accuracy
    // at the ranges found in comprehensive_gamut_ranges.
    //
    // For the abs+sign approach, the polynomial receives abs(value),
    // so we only test non-negative values here.

    const SRGB_THRESHOLD: f64 = 0.039_293_37;
    const LINEAR_THRESHOLD: f64 = 0.003_041_282_6;

    struct RangeTest {
        name: &'static str,
        max_s2l: f64, // max abs encoded sRGB value for S2L polynomial
        max_l2s: f64, // max abs linear sRGB value for L2S polynomial
    }

    let tests = [
        RangeTest {
            name: "Display P3 @ 1.0",
            max_s2l: 1.10,
            max_l2s: 1.25,
        },
        RangeTest {
            name: "BT.2020 @ 1.0",
            max_s2l: 1.25,
            max_l2s: 1.70,
        },
        RangeTest {
            name: "ProPhoto @ 1.0",
            max_s2l: 1.40,
            max_l2s: 2.05,
        },
        RangeTest {
            name: "ACES AP0 @ 1.0",
            max_s2l: 1.50,
            max_l2s: 2.55,
        },
        RangeTest {
            name: "BT.2020 @ 2.0",
            max_s2l: 1.70,
            max_l2s: 3.40,
        },
        RangeTest {
            name: "ACES AP0 @ 2.0",
            max_s2l: 2.00,
            max_l2s: 5.10,
        },
    ];

    println!("\n=== Polynomial extrapolation accuracy (abs+sign approach) ===\n");
    println!(
        "{:<25} {:>10} {:>12} {:>10} {:>12}",
        "Scenario", "S2L max", "S2L err", "L2S max", "L2S err"
    );
    println!("{}", "-".repeat(75));

    let mut any_exceeds = false;

    for test in &tests {
        // S2L polynomial error
        let mut s2l_max_err = 0.0_f64;
        let steps = ((test.max_s2l - SRGB_THRESHOLD) * 10000.0) as usize + 1;
        for i in 0..=steps {
            let x = SRGB_THRESHOLD + (test.max_s2l - SRGB_THRESHOLD) * i as f64 / steps as f64;
            let poly = eval_rational_poly(x, &S2L_P, &S2L_Q);
            let exact = srgb_eotf_extended(x);
            s2l_max_err = s2l_max_err.max((poly - exact).abs());
        }

        // L2S polynomial error
        let mut l2s_max_err = 0.0_f64;
        let steps = ((test.max_l2s - LINEAR_THRESHOLD) * 10000.0) as usize + 1;
        for i in 0..=steps {
            let lin =
                LINEAR_THRESHOLD + (test.max_l2s - LINEAR_THRESHOLD) * i as f64 / steps as f64;
            let s = lin.sqrt();
            let poly = eval_rational_poly(s, &L2S_P, &L2S_Q);
            let exact = srgb_oetf_extended(lin);
            l2s_max_err = l2s_max_err.max((poly - exact).abs());
        }

        let s2l_flag = if s2l_max_err > 1e-3 { " ***" } else { "" };
        let l2s_flag = if l2s_max_err > 1e-3 { " ***" } else { "" };
        if s2l_max_err > 1e-3 || l2s_max_err > 1e-3 {
            any_exceeds = true;
        }

        println!(
            "{:<25} {:>10.2} {:>12.6e}{} {:>10.2} {:>12.6e}{}",
            test.name, test.max_s2l, s2l_max_err, s2l_flag, test.max_l2s, l2s_max_err, l2s_flag
        );
    }

    // Find where errors cross 1e-3
    println!("\n=== Error threshold crossings ===");
    let mut prev = 0.0_f64;
    for i in (SRGB_THRESHOLD * 10000.0) as i64..50000 {
        let x = i as f64 / 10000.0;
        let poly = eval_rational_poly(x, &S2L_P, &S2L_Q);
        let exact = srgb_eotf_extended(x);
        let err = (poly - exact).abs();
        if prev < 1e-3 && err >= 1e-3 {
            println!("  S2L error crosses 1e-3 at encoded = {:.4}", x);
        }
        prev = err;
    }

    let mut prev = 0.0_f64;
    for i in (LINEAR_THRESHOLD * 10000.0) as i64..500000 {
        let lin = i as f64 / 10000.0;
        let s = lin.sqrt();
        let poly = eval_rational_poly(s, &L2S_P, &L2S_Q);
        let exact = srgb_oetf_extended(lin);
        let err = (poly - exact).abs();
        if prev < 1e-3 && err >= 1e-3 {
            println!("  L2S error crosses 1e-3 at linear = {:.4}", lin);
        }
        prev = err;
    }

    if any_exceeds {
        println!("\n*** Some scenarios exceed 1e-3 error. These are scene-referred workflows");
        println!(
            "    at 2x+ intensity. All standard gamut-mapping at linear 1.0 is within bounds."
        );
    }
}

#[test]
fn negative_values_after_gamut_matrix() {
    // ACES AP0 has imaginary primaries that produce the largest negatives.
    // Test what happens with negative linear values after matrix application.

    println!("\n=== Negative sRGB values from ACES AP0 primaries ===\n");
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Color", "Lin R", "Lin G", "Lin B", "Enc R", "Enc G", "Enc B"
    );
    println!("{}", "-".repeat(85));

    let mut max_negative_linear = 0.0_f64;
    let mut max_negative_encoded = 0.0_f64;

    for (color, name) in &TEST_COLORS {
        let rgb_lin = mul_mv(&ACES_AP0_TO_SRGB, *color);
        let rgb_enc = [
            srgb_oetf_extended(rgb_lin[0]),
            srgb_oetf_extended(rgb_lin[1]),
            srgb_oetf_extended(rgb_lin[2]),
        ];

        println!(
            "{:<10} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            name, rgb_lin[0], rgb_lin[1], rgb_lin[2], rgb_enc[0], rgb_enc[1], rgb_enc[2]
        );

        for ch in 0..3 {
            if rgb_lin[ch] < max_negative_linear {
                max_negative_linear = rgb_lin[ch];
            }
            if rgb_enc[ch] < max_negative_encoded {
                max_negative_encoded = rgb_enc[ch];
            }
        }
    }

    println!("\nMost negative linear value: {:.6}", max_negative_linear);
    println!("Most negative encoded value: {:.6}", max_negative_encoded);

    // Verify the f32 scalar extended functions handle these correctly
    println!("\n=== f32 scalar extended vs f64 ground truth for negatives ===\n");
    let test_neg = [
        max_negative_linear as f32,
        max_negative_encoded as f32,
        -0.5_f32,
        -1.0,
        -1.5,
        -2.0,
    ];
    for &v in &test_neg {
        let scalar_l2s = linear_to_srgb_extended(v);
        let exact_l2s = srgb_oetf_extended(v as f64) as f32;

        let scalar_s2l = srgb_to_linear_extended(v);
        let exact_s2l = srgb_eotf_extended(v as f64) as f32;

        println!(
            "  v={:>8.4}: L2S scalar={:>12.6} exact={:>12.6} err={:.2e}  |  S2L scalar={:>12.6} exact={:>12.6} err={:.2e}",
            v,
            scalar_l2s,
            exact_l2s,
            (scalar_l2s - exact_l2s).abs(),
            scalar_s2l,
            exact_s2l,
            (scalar_s2l - exact_s2l).abs(),
        );
    }
}

#[test]
fn s2l_denominator_safety() {
    // The S2L polynomial has denominator zeros near x=-0.24 and x=-1.27.
    // With abs+sign, the polynomial only sees non-negative inputs, so
    // denominator zeros in the negative domain are irrelevant.

    println!("\n=== S2L denominator Q(x) for non-negative x ===\n");
    let mut min_denom = f64::MAX;
    let mut min_denom_x = 0.0;

    // Sweep the full range the polynomial might see (up to ~4.0 for extreme cases)
    for i in 0..=40000 {
        let x = i as f64 / 10000.0;
        let yq = S2L_Q[4].mul_add(x, S2L_Q[3]);
        let yq = yq.mul_add(x, S2L_Q[2]);
        let yq = yq.mul_add(x, S2L_Q[1]);
        let yq = yq.mul_add(x, S2L_Q[0]);

        if yq.abs() < min_denom.abs() {
            min_denom = yq;
            min_denom_x = x;
        }
    }

    println!(
        "Min |Q(x)| in [0, 4]: {:.6} at x = {:.4}",
        min_denom.abs(),
        min_denom_x
    );
    // The denominator should never be near zero for non-negative x
    assert!(
        min_denom.abs() > 1.0,
        "S2L denominator dangerously close to zero at x={}: Q={}",
        min_denom_x,
        min_denom,
    );

    // L2S denominator (evaluated on sqrt(linear), always non-negative)
    let mut min_denom = f64::MAX;
    let mut min_denom_x = 0.0;
    for i in 0..=50000 {
        let x = i as f64 / 10000.0;
        let yq = L2S_Q[4].mul_add(x, L2S_Q[3]);
        let yq = yq.mul_add(x, L2S_Q[2]);
        let yq = yq.mul_add(x, L2S_Q[1]);
        let yq = yq.mul_add(x, L2S_Q[0]);

        if yq.abs() < min_denom.abs() {
            min_denom = yq;
            min_denom_x = x;
        }
    }

    println!(
        "Min |Q(x)| in [0, 5] for L2S: {:.6} at x = {:.4}",
        min_denom.abs(),
        min_denom_x
    );
    assert!(
        min_denom.abs() > 0.1,
        "L2S denominator dangerously close to zero at x={}: Q={}",
        min_denom_x,
        min_denom,
    );
    println!("\nBoth denominators are safe for all non-negative inputs.");
}

#[test]
fn worst_case_summary() {
    // Compute and assert the absolute worst-case range for each scenario.
    // This is the primary reference for deciding polynomial validity.

    let scenarios: &[(&str, &[[f64; 3]; 3], f64)] = &[
        ("BT.2020 @ 1.0", &BT2020_TO_SRGB, 1.0),
        ("Display P3 @ 1.0", &P3_TO_SRGB, 1.0),
        ("ProPhoto @ 1.0", &PROPHOTO_TO_SRGB, 1.0),
        ("ACES AP0 @ 1.0", &ACES_AP0_TO_SRGB, 1.0),
        ("BT.2020 PQ @ 1.0", &BT2020_TO_SRGB, pq_eotf(1.0)),
    ];

    println!("\n=== Worst-case absolute value per scenario ===\n");
    println!("{:<25} {:>12} {:>12}", "Scenario", "max |lin|", "max |enc|");
    println!("{}", "-".repeat(50));

    let mut overall_max_lin = 0.0_f64;
    let mut overall_max_enc = 0.0_f64;

    for (name, matrix, scale) in scenarios {
        let (lmin, lmax, emin, emax) = compute_range(matrix, *scale);
        let max_lin = lmin.abs().max(lmax.abs());
        let max_enc = emin.abs().max(emax.abs());

        println!("{:<25} {:>12.6} {:>12.6}", name, max_lin, max_enc);

        overall_max_lin = overall_max_lin.max(max_lin);
        overall_max_enc = overall_max_enc.max(max_enc);
    }

    println!("{}", "-".repeat(50));
    println!(
        "{:<25} {:>12.6} {:>12.6}",
        "OVERALL", overall_max_lin, overall_max_enc
    );

    // The S2L polynomial error stays under 1e-3 for encoded inputs up to ~1.59.
    // The worst-case encoded value at linear 1.0 is ~1.50 (ACES AP0).
    // This means the abs+sign approach is valid for all standard gamut mapping.
    assert!(
        overall_max_enc < 1.59,
        "Worst-case encoded sRGB ({:.4}) exceeds S2L polynomial 1e-3 boundary (1.59)",
        overall_max_enc,
    );

    // The L2S polynomial error stays under 1e-3 for linear inputs up to ~4.33.
    // The worst-case linear value at linear 1.0 is ~2.52 (ACES AP0).
    assert!(
        overall_max_lin < 4.33,
        "Worst-case linear sRGB ({:.4}) exceeds L2S polynomial 1e-3 boundary (4.33)",
        overall_max_lin,
    );

    println!("\nAll scenarios within polynomial accuracy bounds (< 1e-3 error).");
    println!(
        "S2L polynomial valid up to |encoded| = 1.59, worst case = {:.4}",
        overall_max_enc
    );
    println!(
        "L2S polynomial valid up to |linear| = 4.33, worst case = {:.4}",
        overall_max_lin
    );
}
