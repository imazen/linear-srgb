#![allow(clippy::needless_range_loop, clippy::manual_range_contains)]
//! Generate polynomial coefficients for sRGB transfer function approximation.
//!
//! Uses Chebyshev interpolation for near-minimax polynomial approximation.
//! Outputs coefficients for both centered (numerically stable) and direct
//! (no transform overhead) polynomial evaluation.

use std::f64::consts::PI;

// sRGB constants (C0-continuous, moxcms-derived)
const SRGB_OFFSET: f64 = 0.055_010_718_947_586_6;
const SRGB_SCALE: f64 = 1.055_010_718_947_586_6;
const INV_SRGB_SCALE: f64 = 1.0 / SRGB_SCALE;
const SRGB_LINEAR_THRESHOLD: f64 = 12.92 * 0.003_041_282_560_127_521;
const LINEAR_THRESHOLD: f64 = 0.003_041_282_560_127_521;
const GAMMA: f64 = 2.4;
const INV_GAMMA: f64 = 1.0 / GAMMA;

/// sRGB → linear power segment: ((s + offset) / scale)^2.4
fn srgb_to_linear_power(s: f64) -> f64 {
    ((s + SRGB_OFFSET) * INV_SRGB_SCALE).powf(GAMMA)
}

/// linear → sRGB power segment: scale * l^(1/2.4) - offset
fn linear_to_srgb_power(l: f64) -> f64 {
    SRGB_SCALE * l.powf(INV_GAMMA) - SRGB_OFFSET
}

/// Generate N Chebyshev nodes on [a, b]
fn chebyshev_nodes(n: usize, a: f64, b: f64) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let theta = PI * (2.0 * k as f64 + 1.0) / (2.0 * n as f64);
            (a + b) / 2.0 + (b - a) / 2.0 * theta.cos()
        })
        .collect()
}

/// Solve Ax = b using Gaussian elimination with partial pivoting
fn solve(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let n = b.len();

    for i in 0..n {
        let mut max_val = a[i][i].abs();
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[k][i].abs() > max_val {
                max_val = a[k][i].abs();
                max_row = k;
            }
        }
        a.swap(i, max_row);
        b.swap(i, max_row);

        let pivot = a[i][i];
        for k in (i + 1)..n {
            let factor = a[k][i] / pivot;
            for j in i..n {
                a[k][j] -= factor * a[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in (i + 1)..n {
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }
    x
}

/// Fit polynomial in centered variable u = (x - center) / half_width
fn fit_polynomial_centered(f: impl Fn(f64) -> f64, a: f64, b: f64, degree: usize) -> Vec<f64> {
    let n = degree + 1;
    let center = (a + b) / 2.0;
    let half_width = (b - a) / 2.0;

    let nodes = chebyshev_nodes(n, a, b);

    let mut matrix = vec![vec![0.0f64; n]; n];
    let mut rhs = vec![0.0f64; n];

    for i in 0..n {
        let u = (nodes[i] - center) / half_width; // u ∈ [-1, 1]
        let mut uk = 1.0;
        for j in 0..n {
            matrix[i][j] = uk;
            uk *= u;
        }
        rhs[i] = f(nodes[i]);
    }

    solve(&mut matrix, &mut rhs)
}

/// Convert polynomial from centered variable u = (x - center)/hw to direct x
fn uncenter_polynomial(centered: &[f64], center: f64, half_width: f64) -> Vec<f64> {
    let n = centered.len();
    let mut result = vec![0.0f64; n];

    // Track (x - center)^k as polynomial in x, built incrementally
    // power_of_xmc[j] = coefficient of x^j in (x - center)^k
    let mut power_of_xmc = vec![0.0f64; n];
    power_of_xmc[0] = 1.0; // (x - center)^0 = 1

    let inv_hw = 1.0 / half_width;
    let mut scale = 1.0; // (1/half_width)^k

    // k = 0: add centered[0] * scale * power_of_xmc
    result[0] += centered[0] * scale * power_of_xmc[0];

    for k in 1..n {
        // Multiply power_of_xmc by (x - center)
        let mut new_power = vec![0.0f64; n];
        for j in 0..n {
            if j > 0 {
                new_power[j] += power_of_xmc[j - 1]; // * x term
            }
            new_power[j] -= center * power_of_xmc[j]; // * (-center) term
        }
        power_of_xmc = new_power;
        scale *= inv_hw;

        for j in 0..n {
            result[j] += centered[k] * scale * power_of_xmc[j];
        }
    }

    result
}

/// Evaluate polynomial using Horner's method (f64)
fn eval_poly(coeffs: &[f64], x: f64) -> f64 {
    let mut result = coeffs[coeffs.len() - 1];
    for i in (0..coeffs.len() - 1).rev() {
        result = result * x + coeffs[i];
    }
    result
}

/// Evaluate centered polynomial: p(u) where u = (x - center) / half_width
fn eval_poly_centered(coeffs: &[f64], x: f64, center: f64, half_width: f64) -> f64 {
    let u = (x - center) / half_width;
    eval_poly(coeffs, u)
}

/// Simulate f32 evaluation of centered polynomial (Horner's)
fn eval_poly_centered_f32(coeffs: &[f64], x: f32, center: f32, half_width: f32) -> f32 {
    let u = (x - center) / half_width;
    let mut result = coeffs[coeffs.len() - 1] as f32;
    for i in (0..coeffs.len() - 1).rev() {
        result = result.mul_add(u, coeffs[i] as f32);
    }
    result
}

/// Simulate f32 evaluation of direct polynomial (Horner's)
fn eval_poly_direct_f32(coeffs: &[f64], x: f32) -> f32 {
    let mut result = coeffs[coeffs.len() - 1] as f32;
    for i in (0..coeffs.len() - 1).rev() {
        result = result.mul_add(x, coeffs[i] as f32);
    }
    result
}

/// Check error stats over many sample points
fn check_errors(
    f: impl Fn(f64) -> f64,
    eval: impl Fn(f64) -> (f64, f32),
    a: f64,
    b: f64,
) -> (f64, f64, f64) {
    let n = 1_000_000;
    let mut max_abs_f64 = 0.0f64;
    let mut max_rel_f64 = 0.0f64;
    let mut max_abs_f32 = 0.0f64;

    for i in 0..=n {
        let x = a + (b - a) * i as f64 / n as f64;
        let exact = f(x);
        let (approx_f64, approx_f32) = eval(x);

        let abs_f64 = (exact - approx_f64).abs();
        let rel_f64 = if exact.abs() > 1e-15 {
            abs_f64 / exact.abs()
        } else {
            abs_f64
        };
        max_abs_f64 = max_abs_f64.max(abs_f64);
        max_rel_f64 = max_rel_f64.max(rel_f64);

        let abs_f32 = (exact - approx_f32 as f64).abs();
        max_abs_f32 = max_abs_f32.max(abs_f32);
    }

    (max_abs_f64, max_rel_f64, max_abs_f32)
}

fn print_section(title: &str) {
    println!("\n{}", "=".repeat(60));
    println!("  {title}");
    println!("{}\n", "=".repeat(60));
}

fn main() {
    // ========================================================================
    // sRGB → Linear
    // ========================================================================
    print_section("sRGB → Linear: ((s + offset) / scale)^2.4");

    let s2l_a = SRGB_LINEAR_THRESHOLD;
    let s2l_b = 1.0;
    let s2l_center = (s2l_a + s2l_b) / 2.0;
    let s2l_hw = (s2l_b - s2l_a) / 2.0;

    println!("Domain: [{:.10}, {:.10}]", s2l_a, s2l_b);
    println!("Center: {:.10}, HalfWidth: {:.10}\n", s2l_center, s2l_hw);

    for degree in [5, 7, 9, 11, 13] {
        let coeffs_c = fit_polynomial_centered(srgb_to_linear_power, s2l_a, s2l_b, degree);
        let coeffs_d = uncenter_polynomial(&coeffs_c, s2l_center, s2l_hw);

        // Centered evaluation
        let (abs64_c, rel64_c, abs32_c) = check_errors(
            srgb_to_linear_power,
            |x| {
                let f64_val = eval_poly_centered(&coeffs_c, x, s2l_center, s2l_hw);
                let f32_val =
                    eval_poly_centered_f32(&coeffs_c, x as f32, s2l_center as f32, s2l_hw as f32);
                (f64_val, f32_val)
            },
            s2l_a,
            s2l_b,
        );

        // Direct evaluation
        let (abs64_d, rel64_d, abs32_d) = check_errors(
            srgb_to_linear_power,
            |x| {
                let f64_val = eval_poly(&coeffs_d, x);
                let f32_val = eval_poly_direct_f32(&coeffs_d, x as f32);
                (f64_val, f32_val)
            },
            s2l_a,
            s2l_b,
        );

        let bits_c = -(rel64_c.log2());
        let bits_d = -(rel64_d.log2());

        println!("Degree {degree}:");
        println!(
            "  Centered: f64 abs={abs64_c:.2e} rel={rel64_c:.2e} ({bits_c:.1} bits) | f32 abs={abs32_c:.2e}"
        );
        println!(
            "  Direct:   f64 abs={abs64_d:.2e} rel={rel64_d:.2e} ({bits_d:.1} bits) | f32 abs={abs32_d:.2e}"
        );

        // Print coefficients for promising degrees
        if degree >= 7 {
            println!("  --- Centered coefficients (u = (s - {s2l_center:.8}) / {s2l_hw:.8}) ---");
            for (i, c) in coeffs_c.iter().enumerate() {
                let f32_c = *c as f32;
                println!("    C{i}: {c:>22.15e}  (f32: {f32_c:e})");
            }
            println!("  --- Direct coefficients (polynomial in s) ---");
            for (i, c) in coeffs_d.iter().enumerate() {
                let f32_c = *c as f32;
                println!("    C{i}: {c:>22.15e}  (f32: {f32_c:e})");
            }
        }
        println!();
    }

    // ========================================================================
    // Linear → sRGB
    // ========================================================================
    print_section("Linear → sRGB: scale * l^(1/2.4) - offset");

    let l2s_a = LINEAR_THRESHOLD;
    let l2s_b = 1.0;
    let l2s_center = (l2s_a + l2s_b) / 2.0;
    let l2s_hw = (l2s_b - l2s_a) / 2.0;

    println!("Domain: [{:.10}, {:.10}]", l2s_a, l2s_b);
    println!("Center: {:.10}, HalfWidth: {:.10}\n", l2s_center, l2s_hw);

    for degree in [7, 9, 11, 13, 15, 17] {
        let coeffs_c = fit_polynomial_centered(linear_to_srgb_power, l2s_a, l2s_b, degree);

        let (abs64_c, rel64_c, abs32_c) = check_errors(
            linear_to_srgb_power,
            |x| {
                let f64_val = eval_poly_centered(&coeffs_c, x, l2s_center, l2s_hw);
                let f32_val =
                    eval_poly_centered_f32(&coeffs_c, x as f32, l2s_center as f32, l2s_hw as f32);
                (f64_val, f32_val)
            },
            l2s_a,
            l2s_b,
        );

        let bits_c = -(rel64_c.log2());

        println!("Degree {degree}:");
        println!(
            "  Centered: f64 abs={abs64_c:.2e} rel={rel64_c:.2e} ({bits_c:.1} bits) | f32 abs={abs32_c:.2e}"
        );

        if degree >= 11 && degree <= 15 {
            println!("  --- Centered coefficients ---");
            for (i, c) in coeffs_c.iter().enumerate() {
                let f32_c = *c as f32;
                println!("    C{i}: {c:>22.15e}  (f32: {f32_c:e})");
            }
        }
        println!();
    }

    // ========================================================================
    // Linear → sRGB with sqrt transform
    // ========================================================================
    print_section("Linear → sRGB via sqrt: scale * (√l)^(2/2.4) - offset");

    // Approximate g(s) where s = √l, l = s²
    // g(s) = scale * s^(2/2.4) - offset = scale * s^(5/6) - offset
    // Wait, 2 * (1/2.4) = 2/2.4 = 5/6... no.
    // l^(1/2.4) where l = s² → s^(2/2.4) = s^(5/6)
    // Hmm, 2/2.4 = 0.8333...

    let sqrt_lo = l2s_a.sqrt();
    let sqrt_hi = l2s_b.sqrt(); // = 1.0
    let sqrt_center = (sqrt_lo + sqrt_hi) / 2.0;
    let sqrt_hw = (sqrt_hi - sqrt_lo) / 2.0;

    let sqrt_power = |s: f64| -> f64 { SRGB_SCALE * s.powf(2.0 * INV_GAMMA) - SRGB_OFFSET };

    println!("Domain (sqrt): [{:.10}, {:.10}]", sqrt_lo, sqrt_hi);

    for degree in [5, 7, 9, 11, 13, 15, 17] {
        let coeffs_c = fit_polynomial_centered(sqrt_power, sqrt_lo, sqrt_hi, degree);

        let (abs64_c, rel64_c, abs32_c) = check_errors(
            sqrt_power,
            |s| {
                let f64_val = eval_poly_centered(&coeffs_c, s, sqrt_center, sqrt_hw);
                let f32_val =
                    eval_poly_centered_f32(&coeffs_c, s as f32, sqrt_center as f32, sqrt_hw as f32);
                (f64_val, f32_val)
            },
            sqrt_lo,
            sqrt_hi,
        );

        let bits_c = -(rel64_c.log2());

        println!("Degree {degree}:");
        println!(
            "  Centered: f64 abs={abs64_c:.2e} rel={rel64_c:.2e} ({bits_c:.1} bits) | f32 abs={abs32_c:.2e}"
        );

        if degree >= 7 {
            println!(
                "  --- Centered coefficients (u = (√l - {sqrt_center:.8}) / {sqrt_hw:.8}) ---"
            );
            for (i, c) in coeffs_c.iter().enumerate() {
                let f32_c = *c as f32;
                println!("    C{i}: {c:>22.15e}  (f32: {f32_c:e})");
            }
        }
        println!();
    }
}
