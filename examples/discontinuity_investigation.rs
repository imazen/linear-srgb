//! Investigate sRGB transfer function discontinuity at the linear/power segment boundary.
//!
//! Run with: cargo run --release --example discontinuity_investigation --features "std alt"

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  sRGB Transfer Function Discontinuity Investigation");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Constants
    // =========================================================================

    // IEC 61966-2-1 textbook constants
    let iec_gamma_thresh: f64 = 0.04045;
    let iec_linear_thresh: f64 = 0.0031308;
    let iec_a: f64 = 0.055;
    let iec_scale: f64 = 1.055;

    // moxcms C0-continuous constants (from scalar.rs)
    let mox_a: f64 = 0.055_010_718_947_586_6;
    let mox_scale: f64 = 1.055_010_718_947_586_6;
    let mox_linear_thresh: f64 = 0.003_041_282_560_127_521;
    let mox_gamma_thresh: f64 = 12.92 * mox_linear_thresh;

    println!("Constant comparison:");
    println!("  {:30} {:>22} {:>22}", "", "IEC textbook", "moxcms C0");
    println!("  {:30} {:>22.15} {:>22.15}", "a (offset)", iec_a, mox_a);
    println!("  {:30} {:>22.15} {:>22.15}", "1+a (scale)", iec_scale, mox_scale);
    println!(
        "  {:30} {:>22.15} {:>22.15}",
        "gamma threshold", iec_gamma_thresh, mox_gamma_thresh
    );
    println!(
        "  {:30} {:>22.15} {:>22.15}",
        "linear threshold", iec_linear_thresh, mox_linear_thresh
    );
    println!();

    // =========================================================================
    // sRGB → Linear discontinuity (at gamma threshold)
    // =========================================================================

    println!("───────────────────────────────────────────────────────────────────────");
    println!("  sRGB → Linear: discontinuity at gamma threshold");
    println!("───────────────────────────────────────────────────────────────────────\n");

    // IEC textbook
    let iec_s2l_linear = iec_gamma_thresh / 12.92;
    let iec_s2l_power = ((iec_gamma_thresh + iec_a) / iec_scale).powf(2.4);
    let iec_s2l_disc = (iec_s2l_linear - iec_s2l_power).abs();

    // moxcms
    let mox_s2l_linear = mox_gamma_thresh / 12.92;
    let mox_s2l_power = ((mox_gamma_thresh + mox_a) / mox_scale).powf(2.4);
    let mox_s2l_disc = (mox_s2l_linear - mox_s2l_power).abs();

    println!("  IEC textbook at gamma = {:.15}:", iec_gamma_thresh);
    println!("    linear segment: {:.15e}", iec_s2l_linear);
    println!("    power  segment: {:.15e}", iec_s2l_power);
    println!("    discontinuity:  {:.6e}", iec_s2l_disc);

    println!("  moxcms C0 at gamma = {:.15}:", mox_gamma_thresh);
    println!("    linear segment: {:.15e}", mox_s2l_linear);
    println!("    power  segment: {:.15e}", mox_s2l_power);
    println!("    discontinuity:  {:.6e}", mox_s2l_disc);

    // Rational polynomial at the IEC threshold
    let rp_s2l = eval_s2l_rational_poly(iec_gamma_thresh as f32);
    let rp_s2l_linear = (iec_gamma_thresh as f32) / 12.92_f32;
    let rp_s2l_disc = (rp_s2l - rp_s2l_linear).abs();
    let rp_s2l_vs_iec = (rp_s2l as f64 - iec_s2l_power).abs();

    println!("\n  Rational polynomial at gamma = 0.04045 (f32):");
    println!("    linear seg (f32): {:.10e}", rp_s2l_linear);
    println!("    rat.poly (f32):   {:.10e}", rp_s2l);
    println!("    disc (poly vs linear seg): {:.6e}", rp_s2l_disc);
    println!("    poly vs IEC powf:          {:.6e}", rp_s2l_vs_iec);

    // =========================================================================
    // Linear → sRGB discontinuity (at linear threshold)
    // =========================================================================

    println!("\n───────────────────────────────────────────────────────────────────────");
    println!("  Linear → sRGB: discontinuity at linear threshold");
    println!("───────────────────────────────────────────────────────────────────────\n");

    // IEC textbook
    let iec_l2s_linear = iec_linear_thresh * 12.92;
    let iec_l2s_power = iec_scale * iec_linear_thresh.powf(1.0 / 2.4) - iec_a;
    let iec_l2s_disc = (iec_l2s_linear - iec_l2s_power).abs();

    // moxcms
    let mox_l2s_linear = mox_linear_thresh * 12.92;
    let mox_l2s_power = mox_scale * mox_linear_thresh.powf(1.0 / 2.4) - mox_a;
    let mox_l2s_disc = (mox_l2s_linear - mox_l2s_power).abs();

    println!("  IEC textbook at linear = {:.15}:", iec_linear_thresh);
    println!("    linear segment: {:.15e}", iec_l2s_linear);
    println!("    power  segment: {:.15e}", iec_l2s_power);
    println!("    discontinuity:  {:.6e}", iec_l2s_disc);

    println!("  moxcms C0 at linear = {:.15}:", mox_linear_thresh);
    println!("    linear segment: {:.15e}", mox_l2s_linear);
    println!("    power  segment: {:.15e}", mox_l2s_power);
    println!("    discontinuity:  {:.6e}", mox_l2s_disc);

    // Rational polynomial at the IEC threshold
    let rp_l2s = eval_l2s_rational_poly(iec_linear_thresh as f32);
    let rp_l2s_linear = (iec_linear_thresh as f32) * 12.92_f32;
    let rp_l2s_disc = (rp_l2s - rp_l2s_linear).abs();

    println!("\n  Rational polynomial at linear = 0.0031308 (f32):");
    println!("    linear seg (f32): {:.10e}", rp_l2s_linear);
    println!("    rat.poly (f32):   {:.10e}", rp_l2s);
    println!("    disc (poly vs linear seg): {:.6e}", rp_l2s_disc);

    // =========================================================================
    // f32 neighborhood sweep: how many values are affected?
    // =========================================================================

    println!("\n───────────────────────────────────────────────────────────────────────");
    println!("  f32 neighborhood sweep around thresholds");
    println!("───────────────────────────────────────────────────────────────────────\n");

    // sRGB → Linear: sweep around gamma = 0.04045
    println!("  sRGB → Linear around gamma threshold (0.04045):");
    println!("  {:>10} {:>14} {:>14} {:>14} {:>14}",
        "offset", "gamma", "linear_seg", "rat_poly", "delta");
    let base = 0.04045_f32;
    for offset in -5..=5 {
        let gamma = f32_offset(base, offset);
        let lin = gamma / 12.92_f32;
        let poly = eval_s2l_rational_poly(gamma);
        let delta = poly - lin;
        let which = if gamma <= base { "L" } else { "P" };
        println!("  {:>+10} {:>14.10} {:>14.10e} {:>14.10e} {:>+14.6e} {}",
            offset, gamma, lin, poly, delta, which);
    }

    // Linear → sRGB: sweep around linear = 0.0031308
    println!("\n  Linear → sRGB around linear threshold (0.0031308):");
    println!("  {:>10} {:>14} {:>14} {:>14} {:>14}",
        "offset", "linear", "linear_seg", "rat_poly", "delta");
    let base = 0.003_130_8_f32;
    for offset in -5..=5 {
        let linear = f32_offset(base, offset);
        let lin_seg = linear * 12.92_f32;
        let poly = eval_l2s_rational_poly(linear);
        let delta = poly - lin_seg;
        let which = if linear <= base { "L" } else { "P" };
        println!("  {:>+10} {:>14.10} {:>14.10e} {:>14.10e} {:>+14.6e} {}",
            offset, linear, lin_seg, poly, delta, which);
    }

    // =========================================================================
    // ULP span: how many f32 values between the two thresholds?
    // =========================================================================

    println!("\n───────────────────────────────────────────────────────────────────────");
    println!("  Distance between IEC and moxcms thresholds");
    println!("───────────────────────────────────────────────────────────────────────\n");

    let iec_gamma_f32 = 0.04045_f32;
    let mox_gamma_f32 = mox_gamma_thresh as f32;
    let gamma_ulps = ulp_distance(iec_gamma_f32, mox_gamma_f32);
    println!("  gamma threshold: IEC={:.10}, moxcms={:.10}, distance={} ULP ({:.6e} abs)",
        iec_gamma_f32, mox_gamma_f32, gamma_ulps, (iec_gamma_f32 - mox_gamma_f32).abs());

    let iec_linear_f32 = 0.003_130_8_f32;
    let mox_linear_f32 = mox_linear_thresh as f32;
    let linear_ulps = ulp_distance(iec_linear_f32, mox_linear_f32);
    println!("  linear threshold: IEC={:.10}, moxcms={:.10}, distance={} ULP ({:.6e} abs)",
        iec_linear_f32, mox_linear_f32, linear_ulps, (iec_linear_f32 - mox_linear_f32).abs());

    // =========================================================================
    // Practical impact: u8 and u16 quantization
    // =========================================================================

    println!("\n───────────────────────────────────────────────────────────────────────");
    println!("  Practical impact at u8/u16 quantization");
    println!("───────────────────────────────────────────────────────────────────────\n");

    // What sRGB u8 values are near the threshold?
    // threshold ~= 0.04045, * 255 = 10.3
    // So u8 values 10 and 11 straddle the threshold
    println!("  sRGB u8 values near gamma threshold (0.04045 * 255 = {:.1}):", 0.04045 * 255.0);
    for u8val in 9..=12 {
        let gamma = u8val as f64 / 255.0;
        let iec_result = if gamma <= iec_gamma_thresh {
            gamma / 12.92
        } else {
            ((gamma + iec_a) / iec_scale).powf(2.4)
        };
        let mox_result = if gamma <= mox_gamma_thresh {
            gamma / 12.92
        } else {
            ((gamma + mox_a) / mox_scale).powf(2.4)
        };
        let diff = (iec_result - mox_result).abs();
        let iec_path = if gamma <= iec_gamma_thresh { "linear" } else { "power" };
        let mox_path = if gamma <= mox_gamma_thresh { "linear" } else { "power" };
        println!("    u8={:3} (gamma={:.6}): IEC={:.10e} [{}], mox={:.10e} [{}], diff={:.2e}",
            u8val, gamma, iec_result, iec_path, mox_result, mox_path, diff);
    }

    // Linear threshold in u16 space
    println!("\n  Linear values near encoding threshold (0.0031308 * 65535 = {:.1}):",
        0.0031308 * 65535.0);
    for u16val in [204u16, 205, 206, 207] {
        let linear = u16val as f64 / 65535.0;
        let iec_result = if linear <= iec_linear_thresh {
            linear * 12.92
        } else {
            iec_scale * linear.powf(1.0 / 2.4) - iec_a
        };
        let mox_result = if linear <= mox_linear_thresh {
            linear * 12.92
        } else {
            mox_scale * linear.powf(1.0 / 2.4) - mox_a
        };
        let diff = (iec_result - mox_result).abs();
        println!("    u16={:5} (linear={:.8}): IEC={:.10e}, mox={:.10e}, diff={:.2e}",
            u16val, linear, iec_result, mox_result, diff);
    }

    // =========================================================================
    // Summary: which code paths use which constants
    // =========================================================================

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("  Summary: constants used by each code path");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:40} {:>15} {:>15}", "Code path", "Threshold", "Offset (a)");
    println!("  {:40} {:>15} {:>15}", "─".repeat(40), "─".repeat(15), "─".repeat(15));
    println!("  {:40} {:>15} {:>15}", "precise:: (scalar powf)", "moxcms C0", "0.055011...");
    println!("  {:40} {:>15} {:>15}", "default:: (rational poly)", "IEC 0.04045", "(no powf)");
    println!("  {:40} {:>15} {:>15}", "tokens::x8 (SIMD rational poly)", "IEC 0.04045", "(no powf)");
    println!("  {:40} {:>15} {:>15}", "tokens::x4 (SIMD rational poly)", "IEC 0.04045", "(no powf)");
    println!("  {:40} {:>15} {:>15}", "tokens::x16 (SIMD rational poly)", "IEC 0.04045", "(no powf)");
    println!("  {:40} {:>15} {:>15}", "tf::srgb (TF module scalar)", "IEC 0.04045", "(no powf)");
    println!("  {:40} {:>15} {:>15}", "tf::srgb SIMD rites", "IEC 0.04045", "(no powf)");
    println!("  {:40} {:>15} {:>15}", "LUT tables (const_luts)", "moxcms C0", "0.055011...");

    println!("\n  Note: The rational polynomial replaces the power segment entirely,");
    println!("  so the 'a' constant (0.055 vs 0.055011) is irrelevant for those paths.");
    println!("  Only the threshold matters — it determines where the linear/poly split is.\n");

    // =========================================================================
    // Cross-path comparison: does the choice actually matter?
    // =========================================================================

    println!("───────────────────────────────────────────────────────────────────────");
    println!("  Cross-path max error in the threshold region [0.035, 0.050]");
    println!("───────────────────────────────────────────────────────────────────────\n");

    let mut max_diff_default_vs_precise = 0.0_f64;
    let mut max_diff_at = 0.0_f32;
    let mut count = 0u64;

    // Sweep every f32 in [0.035, 0.050]
    let mut v = 0.035_f32;
    while v <= 0.050_f32 {
        let default_result = eval_s2l_full(v); // rational poly path
        let precise_result = eval_s2l_precise(v); // powf path (moxcms)
        let diff = (default_result as f64 - precise_result as f64).abs();
        if diff > max_diff_default_vs_precise {
            max_diff_default_vs_precise = diff;
            max_diff_at = v;
        }
        count += 1;
        v = next_f32(v);
    }

    println!("  Swept {} f32 values in [0.035, 0.050]", count);
    println!("  Max |default - precise| = {:.6e} at gamma = {:.10}",
        max_diff_default_vs_precise, max_diff_at);

    let default_at = eval_s2l_full(max_diff_at);
    let precise_at = eval_s2l_precise(max_diff_at);
    println!("    default({}): {:.10e}", max_diff_at, default_at);
    println!("    precise({}): {:.10e}", max_diff_at, precise_at);
    let ulp_diff = ulp_distance(default_at, precise_at);
    println!("    ULP distance: {}", ulp_diff);

    // Same for l2s
    let mut max_diff_l2s = 0.0_f64;
    let mut max_diff_l2s_at = 0.0_f32;
    let mut count_l2s = 0u64;

    let mut v = 0.002_f32;
    while v <= 0.004_f32 {
        let default_result = eval_l2s_full(v);
        let precise_result = eval_l2s_precise(v);
        let diff = (default_result as f64 - precise_result as f64).abs();
        if diff > max_diff_l2s {
            max_diff_l2s = diff;
            max_diff_l2s_at = v;
        }
        count_l2s += 1;
        v = next_f32(v);
    }

    println!("\n  Swept {} f32 values in [0.002, 0.004] for l2s", count_l2s);
    println!("  Max |default - precise| = {:.6e} at linear = {:.10}",
        max_diff_l2s, max_diff_l2s_at);

    let default_at = eval_l2s_full(max_diff_l2s_at);
    let precise_at = eval_l2s_precise(max_diff_l2s_at);
    println!("    default({}): {:.10e}", max_diff_l2s_at, default_at);
    println!("    precise({}): {:.10e}", max_diff_l2s_at, precise_at);
    let ulp_diff = ulp_distance(default_at, precise_at);
    println!("    ULP distance: {}", ulp_diff);

    println!();
}

// =============================================================================
// Helpers
// =============================================================================

fn eval_s2l_rational_poly(gamma: f32) -> f32 {
    const S2L_P: [f32; 5] = [
        2.200_248_3e-4, 1.043_637_6e-2, 1.624_820_4e-1, 7.961_565e-1, 8.210_153e-1,
    ];
    const S2L_Q: [f32; 5] = [
        2.631_847e-1, 1.076_976_5, 4.987_528_3e-1, -5.512_498_3e-2, 6.521_209e-3,
    ];
    let x = gamma;
    let yp = S2L_P[4].mul_add(x, S2L_P[3]);
    let yp = yp.mul_add(x, S2L_P[2]);
    let yp = yp.mul_add(x, S2L_P[1]);
    let yp = yp.mul_add(x, S2L_P[0]);

    let yq = S2L_Q[4].mul_add(x, S2L_Q[3]);
    let yq = yq.mul_add(x, S2L_Q[2]);
    let yq = yq.mul_add(x, S2L_Q[1]);
    let yq = yq.mul_add(x, S2L_Q[0]);

    yp / yq
}

fn eval_l2s_rational_poly(linear: f32) -> f32 {
    const L2S_P: [f32; 5] = [
        -5.135_152_6e-4, 5.287_254_7e-3, 3.903_843e-1, 1.474_205_3, 7.352_63e-1,
    ];
    const L2S_Q: [f32; 5] = [
        1.004_519_6e-2, 3.036_675_5e-1, 1.340_817, 9.258_482e-1, 2.424_867_8e-2,
    ];
    let x = linear.sqrt();
    let yp = L2S_P[4].mul_add(x, L2S_P[3]);
    let yp = yp.mul_add(x, L2S_P[2]);
    let yp = yp.mul_add(x, L2S_P[1]);
    let yp = yp.mul_add(x, L2S_P[0]);

    let yq = L2S_Q[4].mul_add(x, L2S_Q[3]);
    let yq = yq.mul_add(x, L2S_Q[2]);
    let yq = yq.mul_add(x, L2S_Q[1]);
    let yq = yq.mul_add(x, L2S_Q[0]);

    yp / yq
}

/// Full srgb_to_linear as `default::` does it (IEC threshold + rational poly)
fn eval_s2l_full(gamma: f32) -> f32 {
    if gamma <= 0.04045 {
        gamma / 12.92
    } else {
        eval_s2l_rational_poly(gamma)
    }
}

/// Full srgb_to_linear as `precise::` does it (moxcms threshold + powf)
fn eval_s2l_precise(gamma: f32) -> f32 {
    let mox_thresh: f32 = (12.92 * 0.003_041_282_560_127_521_f64) as f32;
    let mox_a: f32 = 0.055_010_718_947_586_6_f64 as f32;
    let mox_scale: f32 = 1.055_010_718_947_586_6_f64 as f32;
    if gamma <= mox_thresh {
        gamma / 12.92
    } else {
        ((gamma + mox_a) / mox_scale).powf(2.4)
    }
}

/// Full linear_to_srgb as `default::` does it (IEC threshold + rational poly)
fn eval_l2s_full(linear: f32) -> f32 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        eval_l2s_rational_poly(linear)
    }
}

/// Full linear_to_srgb as `precise::` does it (moxcms threshold + powf)
fn eval_l2s_precise(linear: f32) -> f32 {
    let mox_thresh: f32 = 0.003_041_282_560_127_521_f64 as f32;
    let mox_a: f32 = 0.055_010_718_947_586_6_f64 as f32;
    let mox_scale: f32 = 1.055_010_718_947_586_6_f64 as f32;
    if linear <= mox_thresh {
        linear * 12.92
    } else {
        mox_scale * linear.powf(1.0 / 2.4) - mox_a
    }
}

fn f32_offset(base: f32, offset: i32) -> f32 {
    let bits = base.to_bits() as i32;
    f32::from_bits((bits + offset) as u32)
}

fn next_f32(v: f32) -> f32 {
    f32::from_bits(v.to_bits() + 1)
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    let a_bits = a.to_bits() as i32;
    let b_bits = b.to_bits() as i32;
    (a_bits - b_bits).unsigned_abs()
}
