//! Test what happens if we use the moxcms threshold with the rational polynomial.
//!
//! Run with: cargo run --release --example threshold_experiment --features "std"

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Threshold experiment: which threshold is best for rational poly?");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Test s2l with different thresholds
    let _thresholds: &[(&str, f32)] = &[
        ("IEC (0.04045)", 0.04045),
        (
            "moxcms (0.039293...)",
            (12.92 * 0.003_041_282_560_127_521_f64) as f32,
        ),
        ("optimal (see below)", 0.0), // placeholder
    ];

    // First, find the optimal threshold: where does |linear_seg - rat_poly| cross zero?
    println!("  Finding optimal threshold for s2l (where linear_seg == rational_poly):\n");
    let mut best_thresh = 0.0_f32;
    let mut best_diff = f32::MAX;
    let mut v = 0.035_f32;
    while v <= 0.045 {
        let lin_seg = v / 12.92;
        let poly = s2l_rational_poly(v);
        let diff = (poly - lin_seg).abs();
        if diff < best_diff {
            best_diff = diff;
            best_thresh = v;
        }
        v = next_f32(v);
    }
    println!(
        "    Optimal s2l threshold: {:.15} (diff={:.6e})",
        best_thresh, best_diff
    );

    // Similarly for l2s
    let mut best_l2s_thresh = 0.0_f32;
    let mut best_l2s_diff = f32::MAX;
    let mut v = 0.002_f32;
    while v <= 0.004 {
        let lin_seg = v * 12.92;
        let poly = l2s_rational_poly(v);
        let diff = (poly - lin_seg).abs();
        if diff < best_l2s_diff {
            best_l2s_diff = diff;
            best_l2s_thresh = v;
        }
        v = next_f32(v);
    }
    println!(
        "    Optimal l2s threshold: {:.15} (diff={:.6e})\n",
        best_l2s_thresh, best_l2s_diff
    );

    // Now sweep with each threshold choice
    println!("───────────────────────────────────────────────────────────────────────");
    println!("  sRGB → Linear: full [0,1] sweep with different thresholds");
    println!("───────────────────────────────────────────────────────────────────────\n");

    let s2l_thresholds = [
        ("IEC (0.04045)", 0.04045_f32),
        ("moxcms", (12.92 * 0.003_041_282_560_127_521_f64) as f32),
        ("optimal", best_thresh),
    ];

    for (name, thresh) in &s2l_thresholds {
        let mut max_ulp = 0u32;
        let mut max_ulp_at = 0.0_f32;
        let mut total_ulp = 0u64;
        let mut count = 0u64;
        let mut max_abs_err = 0.0_f64;
        let mut disc_below = 0.0_f32;
        let mut disc_above = 0.0_f32;
        let mut found_above = false;

        let mut v = 0.0_f32;
        while v <= 1.0 {
            let result = if v < *thresh {
                v / 12.92
            } else {
                s2l_rational_poly(v)
            };

            let ref_val = s2l_iec_f64(v as f64);
            let expected = ref_val as f32;
            let ulp = ulp_distance(result, expected);
            let abs_err = (result as f64 - ref_val).abs();

            total_ulp += ulp as u64;
            count += 1;

            if ulp > max_ulp {
                max_ulp = ulp;
                max_ulp_at = v;
            }
            if abs_err > max_abs_err {
                max_abs_err = abs_err;
            }

            // Track discontinuity at this threshold
            if v < *thresh {
                disc_below = result;
            } else if !found_above {
                disc_above = result;
                found_above = true;
            }

            v = next_f32(v);
        }

        let avg_ulp = total_ulp as f64 / count as f64;
        let disc_ulp = ulp_distance(disc_below, disc_above);
        let disc_abs = (disc_above - disc_below).abs();
        println!(
            "  threshold={:20}: max={:4} ULP, avg={:.2}, disc={:4} ULP ({:.2e}), max_abs={:.6e} at {:.10}",
            name, max_ulp, avg_ulp, disc_ulp, disc_abs, max_abs_err, max_ulp_at
        );
    }

    println!("\n───────────────────────────────────────────────────────────────────────");
    println!("  Linear → sRGB: full [0,1] sweep with different thresholds");
    println!("───────────────────────────────────────────────────────────────────────\n");

    let l2s_thresholds = [
        ("IEC (0.0031308)", 0.003_130_8_f32),
        ("moxcms", 0.003_041_282_560_127_521_f64 as f32),
        ("optimal", best_l2s_thresh),
    ];

    for (name, thresh) in &l2s_thresholds {
        let mut max_ulp = 0u32;
        let mut max_ulp_at = 0.0_f32;
        let mut total_ulp = 0u64;
        let mut count = 0u64;
        let mut max_abs_err = 0.0_f64;
        let mut disc_below = 0.0_f32;
        let mut disc_above = 0.0_f32;
        let mut found_above = false;

        let mut v = 0.0_f32;
        while v <= 1.0 {
            let result = if v < *thresh {
                v * 12.92
            } else {
                l2s_rational_poly(v)
            };

            let ref_val = l2s_iec_f64(v as f64);
            let expected = ref_val as f32;
            let ulp = ulp_distance(result, expected);
            let abs_err = (result as f64 - ref_val).abs();

            total_ulp += ulp as u64;
            count += 1;

            if ulp > max_ulp {
                max_ulp = ulp;
                max_ulp_at = v;
            }
            if abs_err > max_abs_err {
                max_abs_err = abs_err;
            }

            if v < *thresh {
                disc_below = result;
            } else if !found_above {
                disc_above = result;
                found_above = true;
            }

            v = next_f32(v);
        }

        let avg_ulp = total_ulp as f64 / count as f64;
        let disc_ulp = ulp_distance(disc_below, disc_above);
        let disc_abs = (disc_above - disc_below).abs();
        println!(
            "  threshold={:20}: max={:4} ULP, avg={:.2}, disc={:4} ULP ({:.2e}), max_abs={:.6e} at {:.10}",
            name, max_ulp, avg_ulp, disc_ulp, disc_abs, max_abs_err, max_ulp_at
        );
    }

    println!();
}

fn s2l_iec_f64(gamma: f64) -> f64 {
    if gamma <= 0.0 {
        0.0
    } else if gamma < 0.04045 {
        gamma / 12.92
    } else if gamma < 1.0 {
        ((gamma + 0.055) / 1.055).powf(2.4)
    } else {
        1.0
    }
}

fn l2s_iec_f64(linear: f64) -> f64 {
    if linear <= 0.0 {
        0.0
    } else if linear < 0.003_130_8 {
        linear * 12.92
    } else if linear < 1.0 {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    } else {
        1.0
    }
}

fn s2l_rational_poly(gamma: f32) -> f32 {
    const P: [f32; 5] = [
        2.200_248_3e-4,
        1.043_637_6e-2,
        1.624_820_4e-1,
        7.961_565e-1,
        8.210_153e-1,
    ];
    const Q: [f32; 5] = [
        2.631_847e-1,
        1.076_976_5,
        4.987_528_3e-1,
        -5.512_498_3e-2,
        6.521_209e-3,
    ];
    let x = gamma;
    let yp = P[4]
        .mul_add(x, P[3])
        .mul_add(x, P[2])
        .mul_add(x, P[1])
        .mul_add(x, P[0]);
    let yq = Q[4]
        .mul_add(x, Q[3])
        .mul_add(x, Q[2])
        .mul_add(x, Q[1])
        .mul_add(x, Q[0]);
    yp / yq
}

fn l2s_rational_poly(linear: f32) -> f32 {
    const P: [f32; 5] = [
        -5.135_152_6e-4,
        5.287_254_7e-3,
        3.903_843e-1,
        1.474_205_3,
        7.352_63e-1,
    ];
    const Q: [f32; 5] = [
        1.004_519_6e-2,
        3.036_675_5e-1,
        1.340_817,
        9.258_482e-1,
        2.424_867_8e-2,
    ];
    let x = linear.sqrt();
    let yp = P[4]
        .mul_add(x, P[3])
        .mul_add(x, P[2])
        .mul_add(x, P[1])
        .mul_add(x, P[0]);
    let yq = Q[4]
        .mul_add(x, Q[3])
        .mul_add(x, Q[2])
        .mul_add(x, Q[1])
        .mul_add(x, Q[0]);
    yp / yq
}

fn next_f32(v: f32) -> f32 {
    f32::from_bits(v.to_bits() + 1)
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    let a_bits = a.to_bits() as i32;
    let b_bits = b.to_bits() as i32;
    (a_bits - b_bits).unsigned_abs()
}
