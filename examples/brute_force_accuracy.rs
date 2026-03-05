//! Brute-force accuracy sweep of every f32 in [0, 1] across all code paths.
//!
//! Tests against both IEC f64 reference and moxcms C0 f64 reference.
//! Run with: cargo run --release --example brute_force_accuracy --features "std alt"

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Brute-force accuracy: every f32 in [0, 1]");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // sRGB → Linear sweep
    // =========================================================================

    println!("───────────────────────────────────────────────────────────────────────");
    println!("  sRGB → Linear: every f32 in [0.0, 1.0]");
    println!("───────────────────────────────────────────────────────────────────────\n");

    let mut s2l_stats = SweepStats::new("default (rat.poly + IEC thresh)");
    let mut s2l_precise_stats = SweepStats::new("precise (powf + moxcms thresh)");
    let mut s2l_default_vs_iec = SweepStats::new("default vs IEC f64 ref");
    let mut s2l_default_vs_mox = SweepStats::new("default vs moxcms f64 ref");
    let mut s2l_precise_vs_iec = SweepStats::new("precise vs IEC f64 ref");
    let mut s2l_precise_vs_mox = SweepStats::new("precise vs moxcms f64 ref");

    // Also: what happens at the moxcms threshold for the default path?
    let mut s2l_default_disc_at_iec = DiscontinuityTracker::new("default at IEC threshold");
    let mut s2l_default_disc_at_mox = DiscontinuityTracker::new("default at moxcms threshold");
    let mut s2l_precise_disc_at_iec = DiscontinuityTracker::new("precise at IEC threshold");
    let mut s2l_precise_disc_at_mox = DiscontinuityTracker::new("precise at moxcms threshold");

    let iec_gamma_thresh: f32 = 0.04045;
    let mox_gamma_thresh: f32 = (12.92 * 0.003_041_282_560_127_521_f64) as f32;

    let mut v = 0.0_f32;
    let mut count = 0u64;
    while v <= 1.0 {
        let vf64 = v as f64;

        // Reference values
        let iec_ref = s2l_iec_f64(vf64);
        let mox_ref = s2l_mox_f64(vf64);

        // Code path values
        let default_val = s2l_default(v);
        let precise_val = s2l_precise(v);

        // Track accuracy vs both references
        s2l_default_vs_iec.update(v, default_val, iec_ref);
        s2l_default_vs_mox.update(v, default_val, mox_ref);
        s2l_precise_vs_iec.update(v, precise_val, iec_ref);
        s2l_precise_vs_mox.update(v, precise_val, mox_ref);

        // Track discontinuity at thresholds
        s2l_default_disc_at_iec.update(v, default_val, iec_gamma_thresh);
        s2l_default_disc_at_mox.update(v, default_val, mox_gamma_thresh);
        s2l_precise_disc_at_iec.update(v, precise_val, iec_gamma_thresh);
        s2l_precise_disc_at_mox.update(v, precise_val, mox_gamma_thresh);

        // Self-consistency
        s2l_stats.update(v, default_val, precise_val as f64);
        s2l_precise_stats.update(v, precise_val, iec_ref);

        count += 1;
        v = next_f32(v);
    }

    println!("  Swept {} f32 values.\n", count);

    println!("  Accuracy vs f64 reference (full [0,1] range):");
    println!("  {:50} {:>12} {:>12} {:>14}", "", "max ULP", "avg ULP", "max abs err");
    println!("  {:50} {:>12} {:>12} {:>14}", "─".repeat(50), "─".repeat(12), "─".repeat(12), "─".repeat(14));
    s2l_default_vs_iec.print_summary();
    s2l_default_vs_mox.print_summary();
    s2l_precise_vs_iec.print_summary();
    s2l_precise_vs_mox.print_summary();

    println!("\n  Where max errors occur:");
    s2l_default_vs_iec.print_worst();
    s2l_default_vs_mox.print_worst();
    s2l_precise_vs_iec.print_worst();
    s2l_precise_vs_mox.print_worst();

    println!("\n  Discontinuity at thresholds (jump between adjacent f32 values):");
    s2l_default_disc_at_iec.print();
    s2l_default_disc_at_mox.print();
    s2l_precise_disc_at_iec.print();
    s2l_precise_disc_at_mox.print();

    // =========================================================================
    // Linear → sRGB sweep
    // =========================================================================

    println!("\n───────────────────────────────────────────────────────────────────────");
    println!("  Linear → sRGB: every f32 in [0.0, 1.0]");
    println!("───────────────────────────────────────────────────────────────────────\n");

    let mut l2s_default_vs_iec = SweepStats::new("default vs IEC f64 ref");
    let mut l2s_default_vs_mox = SweepStats::new("default vs moxcms f64 ref");
    let mut l2s_precise_vs_iec = SweepStats::new("precise vs IEC f64 ref");
    let mut l2s_precise_vs_mox = SweepStats::new("precise vs moxcms f64 ref");

    let iec_linear_thresh: f32 = 0.003_130_8;
    let mox_linear_thresh: f32 = 0.003_041_282_560_127_521_f64 as f32;

    let mut l2s_default_disc_at_iec = DiscontinuityTracker::new("default at IEC threshold");
    let mut l2s_default_disc_at_mox = DiscontinuityTracker::new("default at moxcms threshold");
    let mut l2s_precise_disc_at_iec = DiscontinuityTracker::new("precise at IEC threshold");
    let mut l2s_precise_disc_at_mox = DiscontinuityTracker::new("precise at moxcms threshold");

    let mut v = 0.0_f32;
    count = 0;
    while v <= 1.0 {
        let vf64 = v as f64;

        let iec_ref = l2s_iec_f64(vf64);
        let mox_ref = l2s_mox_f64(vf64);

        let default_val = l2s_default(v);
        let precise_val = l2s_precise(v);

        l2s_default_vs_iec.update(v, default_val, iec_ref);
        l2s_default_vs_mox.update(v, default_val, mox_ref);
        l2s_precise_vs_iec.update(v, precise_val, iec_ref);
        l2s_precise_vs_mox.update(v, precise_val, mox_ref);

        l2s_default_disc_at_iec.update(v, default_val, iec_linear_thresh);
        l2s_default_disc_at_mox.update(v, default_val, mox_linear_thresh);
        l2s_precise_disc_at_iec.update(v, precise_val, iec_linear_thresh);
        l2s_precise_disc_at_mox.update(v, precise_val, mox_linear_thresh);

        count += 1;
        v = next_f32(v);
    }

    println!("  Swept {} f32 values.\n", count);

    println!("  Accuracy vs f64 reference (full [0,1] range):");
    println!("  {:50} {:>12} {:>12} {:>14}", "", "max ULP", "avg ULP", "max abs err");
    println!("  {:50} {:>12} {:>12} {:>14}", "─".repeat(50), "─".repeat(12), "─".repeat(12), "─".repeat(14));
    l2s_default_vs_iec.print_summary();
    l2s_default_vs_mox.print_summary();
    l2s_precise_vs_iec.print_summary();
    l2s_precise_vs_mox.print_summary();

    println!("\n  Where max errors occur:");
    l2s_default_vs_iec.print_worst();
    l2s_default_vs_mox.print_worst();
    l2s_precise_vs_iec.print_worst();
    l2s_precise_vs_mox.print_worst();

    println!("\n  Discontinuity at thresholds (jump between adjacent f32 values):");
    l2s_default_disc_at_iec.print();
    l2s_default_disc_at_mox.print();
    l2s_precise_disc_at_iec.print();
    l2s_precise_disc_at_mox.print();

    // =========================================================================
    // Verdict
    // =========================================================================

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("  Verdict");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  The rational polynomial was fitted to the IEC curve, so it should be");
    println!("  measured against the IEC f64 reference. The precise:: powf path uses");
    println!("  moxcms C0 constants, so it should be measured against the moxcms ref.");
    println!();
    println!("  Key numbers:");
    println!("    s2l default vs IEC:    max {} ULP  (correct reference)", s2l_default_vs_iec.max_ulp);
    println!("    s2l precise vs moxcms: max {} ULP  (correct reference)", s2l_precise_vs_mox.max_ulp);
    println!("    l2s default vs IEC:    max {} ULP  (correct reference)", l2s_default_vs_iec.max_ulp);
    println!("    l2s precise vs moxcms: max {} ULP  (correct reference)", l2s_precise_vs_mox.max_ulp);
    println!();
    println!("  Cross-reference (mismatched constants — not bugs, just different curves):");
    println!("    s2l default vs moxcms: max {} ULP", s2l_default_vs_mox.max_ulp);
    println!("    s2l precise vs IEC:    max {} ULP", s2l_precise_vs_iec.max_ulp);
    println!("    l2s default vs moxcms: max {} ULP", l2s_default_vs_mox.max_ulp);
    println!("    l2s precise vs IEC:    max {} ULP", l2s_precise_vs_iec.max_ulp);
    println!();
}

// =============================================================================
// f64 reference implementations
// =============================================================================

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

fn s2l_mox_f64(gamma: f64) -> f64 {
    const A: f64 = 0.055_010_718_947_586_6;
    const SCALE: f64 = 1.055_010_718_947_586_6;
    const THRESH: f64 = 12.92 * 0.003_041_282_560_127_521;
    if gamma <= 0.0 {
        0.0
    } else if gamma < THRESH {
        gamma / 12.92
    } else if gamma < 1.0 {
        ((gamma + A) / SCALE).powf(2.4)
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

fn l2s_mox_f64(linear: f64) -> f64 {
    const A: f64 = 0.055_010_718_947_586_6;
    const SCALE: f64 = 1.055_010_718_947_586_6;
    const THRESH: f64 = 0.003_041_282_560_127_521;
    if linear <= 0.0 {
        0.0
    } else if linear < THRESH {
        linear * 12.92
    } else if linear < 1.0 {
        SCALE * linear.powf(1.0 / 2.4) - A
    } else {
        1.0
    }
}

// =============================================================================
// f32 code path implementations
// =============================================================================

fn s2l_default(gamma: f32) -> f32 {
    if gamma <= 0.0 {
        0.0
    } else if gamma < 0.04045 {
        gamma / 12.92
    } else if gamma < 1.0 {
        s2l_rational_poly(gamma)
    } else {
        1.0
    }
}

fn s2l_precise(gamma: f32) -> f32 {
    const THRESH: f32 = (12.92 * 0.003_041_282_560_127_521_f64) as f32;
    const A: f32 = 0.055_010_718_947_586_6_f64 as f32;
    const SCALE: f32 = 1.055_010_718_947_586_6_f64 as f32;
    if gamma <= 0.0 {
        0.0
    } else if gamma < THRESH {
        gamma / 12.92
    } else if gamma < 1.0 {
        ((gamma + A) / SCALE).powf(2.4)
    } else {
        1.0
    }
}

fn l2s_default(linear: f32) -> f32 {
    if linear <= 0.0 {
        0.0
    } else if linear < 0.003_130_8 {
        linear * 12.92
    } else if linear < 1.0 {
        l2s_rational_poly(linear)
    } else {
        1.0
    }
}

fn l2s_precise(linear: f32) -> f32 {
    const THRESH: f32 = 0.003_041_282_560_127_521_f64 as f32;
    const A: f32 = 0.055_010_718_947_586_6_f64 as f32;
    const SCALE: f32 = 1.055_010_718_947_586_6_f64 as f32;
    if linear <= 0.0 {
        0.0
    } else if linear < THRESH {
        linear * 12.92
    } else if linear < 1.0 {
        SCALE * linear.powf(1.0 / 2.4) - A
    } else {
        1.0
    }
}

// =============================================================================
// Rational polynomial (from libjxl)
// =============================================================================

fn s2l_rational_poly(gamma: f32) -> f32 {
    const P: [f32; 5] = [
        2.200_248_3e-4, 1.043_637_6e-2, 1.624_820_4e-1, 7.961_565e-1, 8.210_153e-1,
    ];
    const Q: [f32; 5] = [
        2.631_847e-1, 1.076_976_5, 4.987_528_3e-1, -5.512_498_3e-2, 6.521_209e-3,
    ];
    let x = gamma;
    let yp = P[4].mul_add(x, P[3]).mul_add(x, P[2]).mul_add(x, P[1]).mul_add(x, P[0]);
    let yq = Q[4].mul_add(x, Q[3]).mul_add(x, Q[2]).mul_add(x, Q[1]).mul_add(x, Q[0]);
    yp / yq
}

fn l2s_rational_poly(linear: f32) -> f32 {
    const P: [f32; 5] = [
        -5.135_152_6e-4, 5.287_254_7e-3, 3.903_843e-1, 1.474_205_3, 7.352_63e-1,
    ];
    const Q: [f32; 5] = [
        1.004_519_6e-2, 3.036_675_5e-1, 1.340_817, 9.258_482e-1, 2.424_867_8e-2,
    ];
    let x = linear.sqrt();
    let yp = P[4].mul_add(x, P[3]).mul_add(x, P[2]).mul_add(x, P[1]).mul_add(x, P[0]);
    let yq = Q[4].mul_add(x, Q[3]).mul_add(x, Q[2]).mul_add(x, Q[1]).mul_add(x, Q[0]);
    yp / yq
}

// =============================================================================
// Stats tracking
// =============================================================================

struct SweepStats {
    name: &'static str,
    max_ulp: u32,
    max_ulp_at: f32,
    max_abs_err: f64,
    max_abs_at: f32,
    total_ulp: u64,
    count: u64,
    actual_at_worst: f32,
    ref_at_worst: f64,
}

impl SweepStats {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            max_ulp: 0,
            max_ulp_at: 0.0,
            max_abs_err: 0.0,
            max_abs_at: 0.0,
            total_ulp: 0,
            count: 0,
            actual_at_worst: 0.0,
            ref_at_worst: 0.0,
        }
    }

    fn update(&mut self, input: f32, actual: f32, reference: f64) {
        let expected = reference as f32;
        let ulp = ulp_distance(actual, expected);
        let abs_err = (actual as f64 - reference).abs();

        self.total_ulp += ulp as u64;
        self.count += 1;

        if ulp > self.max_ulp {
            self.max_ulp = ulp;
            self.max_ulp_at = input;
            self.actual_at_worst = actual;
            self.ref_at_worst = reference;
        }
        if abs_err > self.max_abs_err {
            self.max_abs_err = abs_err;
            self.max_abs_at = input;
        }
    }

    fn print_summary(&self) {
        let avg_ulp = if self.count > 0 {
            self.total_ulp as f64 / self.count as f64
        } else {
            0.0
        };
        println!(
            "  {:50} {:>12} {:>12.2} {:>14.6e}",
            self.name, self.max_ulp, avg_ulp, self.max_abs_err,
        );
    }

    fn print_worst(&self) {
        println!(
            "    {:48} at input={:.10}: actual={:.10e}, ref={:.15e}",
            self.name, self.max_ulp_at, self.actual_at_worst, self.ref_at_worst,
        );
    }
}

/// Track the jump across a threshold to detect discontinuity
struct DiscontinuityTracker {
    name: &'static str,
    threshold: f32,
    // Value just below threshold
    val_below: f32,
    input_below: f32,
    // Value at or just above threshold
    val_above: f32,
    input_above: f32,
    found_below: bool,
    found_above: bool,
}

impl DiscontinuityTracker {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            threshold: 0.0,
            val_below: 0.0,
            input_below: 0.0,
            val_above: 0.0,
            input_above: 0.0,
            found_below: false,
            found_above: false,
        }
    }

    fn update(&mut self, input: f32, output: f32, threshold: f32) {
        self.threshold = threshold;
        if input < threshold {
            // Keep updating - we want the last value below
            self.val_below = output;
            self.input_below = input;
            self.found_below = true;
        } else if !self.found_above {
            // First value at or above threshold
            self.val_above = output;
            self.input_above = input;
            self.found_above = true;
        }
    }

    fn print(&self) {
        if self.found_below && self.found_above {
            let jump = (self.val_above - self.val_below).abs();
            let input_step = (self.input_above - self.input_below).abs();
            // Expected step: if continuous, output should change by roughly
            // derivative * input_step. A jump >> expected indicates discontinuity.
            let ulps = ulp_distance(self.val_below, self.val_above);
            println!(
                "    {:48} below={:.10e} above={:.10e} jump={:.6e} ({} ULP) step={:.6e}",
                self.name, self.val_below, self.val_above, jump, ulps, input_step,
            );
        }
    }
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
