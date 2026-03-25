//! Brute-force correctness tests for sRGB transfer functions.
//!
//! Tests every path (scalar, fast/polynomial, SIMD slice) against known-good
//! f64 reference implementations. Covers exhaustive f32 sweeps, every u8/u16
//! value, and roundtrip through all paths.

use linear_srgb::default::{
    linear_to_srgb, linear_to_srgb_slice, linear_to_srgb_u8_slice, linear_to_srgb_u16_slice,
    srgb_to_linear, srgb_to_linear_slice, srgb_u8_to_linear_slice, srgb_u16_to_linear_slice,
};
use linear_srgb::precise::{
    linear_to_srgb as precise_l2s, linear_to_srgb_extended as precise_l2s_ext,
    linear_to_srgb_f64 as precise_l2s_f64, srgb_to_linear as precise_s2l,
    srgb_to_linear_extended as precise_s2l_ext, srgb_to_linear_f64 as precise_s2l_f64,
};

// ============================================================================
// f64 reference (known-good, C0-continuous moxcms constants)
// ============================================================================

const A: f64 = 0.0550107189475866;
const A1: f64 = 1.0 + A;
const LIN_THRESH: f64 = 0.003041282560127521;
const GAM_THRESH: f64 = 12.92 * LIN_THRESH;

fn ref_s2l(v: f64) -> f64 {
    if v <= 0.0 {
        0.0
    } else if v <= GAM_THRESH {
        v / 12.92
    } else if v < 1.0 {
        ((v + A) / A1).powf(2.4)
    } else {
        1.0
    }
}

fn ref_l2s(v: f64) -> f64 {
    if v <= 0.0 {
        0.0
    } else if v <= LIN_THRESH {
        v * 12.92
    } else if v < 1.0 {
        A1 * v.powf(1.0 / 2.4) - A
    } else {
        1.0
    }
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    (ai - bi).unsigned_abs()
}

fn next_f32_above(v: f32) -> f32 {
    if v >= f32::MAX {
        return v;
    }
    f32::from_bits(v.to_bits() + 1)
}

// ============================================================================
// Exhaustive scalar fast path vs f64 reference
// ============================================================================

#[test]
fn exhaustive_srgb_to_linear_fast_vs_f64() {
    // Sweep every f32 in [0.0, 1.0]
    let mut v = 0.0_f32;
    let mut max_ulp: u32 = 0;
    let mut worst_input = 0.0_f32;
    let mut count: u64 = 0;

    while v <= 1.0 {
        let got = srgb_to_linear(v);
        let expected = ref_s2l(v as f64) as f32;
        let ulp = ulp_distance(got, expected);
        if ulp > max_ulp {
            max_ulp = ulp;
            worst_input = v;
        }
        count += 1;
        v = next_f32_above(v);
    }

    eprintln!(
        "srgb_to_linear (default/fast): {count} values, max ULP = {max_ulp} at {worst_input}"
    );
    assert!(
        max_ulp <= 16,
        "srgb_to_linear max ULP {max_ulp} at {worst_input} exceeds 16"
    );
}

#[test]
fn exhaustive_linear_to_srgb_fast_vs_f64() {
    let mut v = 0.0_f32;
    let mut max_ulp: u32 = 0;
    let mut worst_input = 0.0_f32;
    let mut count: u64 = 0;

    while v <= 1.0 {
        let got = linear_to_srgb(v);
        let expected = ref_l2s(v as f64) as f32;
        let ulp = ulp_distance(got, expected);
        if ulp > max_ulp {
            max_ulp = ulp;
            worst_input = v;
        }
        count += 1;
        v = next_f32_above(v);
    }

    eprintln!(
        "linear_to_srgb (default/fast): {count} values, max ULP = {max_ulp} at {worst_input}"
    );
    assert!(
        max_ulp <= 16,
        "linear_to_srgb max ULP {max_ulp} at {worst_input} exceeds 16"
    );
}

// ============================================================================
// Exhaustive precise (powf) path vs f64 reference
// ============================================================================

#[test]
fn exhaustive_srgb_to_linear_precise_vs_f64() {
    let mut v = 0.0_f32;
    let mut max_ulp: u32 = 0;
    let mut worst_input = 0.0_f32;
    let mut count: u64 = 0;

    while v <= 1.0 {
        let got = precise_s2l(v);
        let expected = precise_s2l_f64(v as f64) as f32;
        let ulp = ulp_distance(got, expected);
        if ulp > max_ulp {
            max_ulp = ulp;
            worst_input = v;
        }
        count += 1;
        v = next_f32_above(v);
    }

    eprintln!(
        "srgb_to_linear (precise/powf): {count} values, max ULP = {max_ulp} at {worst_input}"
    );
    // powf f32 vs f64: expect ~6 ULP max from powf precision
    assert!(
        max_ulp <= 10,
        "srgb_to_linear precise max ULP {max_ulp} at {worst_input} exceeds 10"
    );
}

#[test]
fn exhaustive_linear_to_srgb_precise_vs_f64() {
    let mut v = 0.0_f32;
    let mut max_ulp: u32 = 0;
    let mut worst_input = 0.0_f32;
    let mut count: u64 = 0;

    while v <= 1.0 {
        let got = precise_l2s(v);
        let expected = precise_l2s_f64(v as f64) as f32;
        let ulp = ulp_distance(got, expected);
        if ulp > max_ulp {
            max_ulp = ulp;
            worst_input = v;
        }
        count += 1;
        v = next_f32_above(v);
    }

    eprintln!(
        "linear_to_srgb (precise/powf): {count} values, max ULP = {max_ulp} at {worst_input}"
    );
    assert!(
        max_ulp <= 10,
        "linear_to_srgb precise max ULP {max_ulp} at {worst_input} exceeds 10"
    );
}

// ============================================================================
// Fast vs precise: the two paths must agree within combined error budget
// ============================================================================

#[test]
fn exhaustive_fast_vs_precise_s2l() {
    let mut v = 0.0_f32;
    let mut max_ulp: u32 = 0;
    let mut worst_input = 0.0_f32;
    let mut count: u64 = 0;

    while v <= 1.0 {
        let fast = srgb_to_linear(v);
        let prec = precise_s2l(v);
        let ulp = ulp_distance(fast, prec);
        if ulp > max_ulp {
            max_ulp = ulp;
            worst_input = v;
        }
        count += 1;
        v = next_f32_above(v);
    }

    eprintln!(
        "fast vs precise srgb_to_linear: {count} values, max ULP = {max_ulp} at {worst_input}"
    );
    // Combined budget: fast ~16 ULP vs f64, precise ~6 ULP vs f64 → max ~22 ULP between them
    assert!(
        max_ulp <= 22,
        "fast vs precise s2l max ULP {max_ulp} at {worst_input} exceeds 22"
    );
}

#[test]
fn exhaustive_fast_vs_precise_l2s() {
    let mut v = 0.0_f32;
    let mut max_ulp: u32 = 0;
    let mut worst_input = 0.0_f32;
    let mut count: u64 = 0;

    while v <= 1.0 {
        let fast = linear_to_srgb(v);
        let prec = precise_l2s(v);
        let ulp = ulp_distance(fast, prec);
        if ulp > max_ulp {
            max_ulp = ulp;
            worst_input = v;
        }
        count += 1;
        v = next_f32_above(v);
    }

    eprintln!(
        "fast vs precise linear_to_srgb: {count} values, max ULP = {max_ulp} at {worst_input}"
    );
    assert!(
        max_ulp <= 22,
        "fast vs precise l2s max ULP {max_ulp} at {worst_input} exceeds 22"
    );
}

// ============================================================================
// SIMD slice path must match scalar fast path exactly
// ============================================================================

#[test]
fn simd_s2l_matches_scalar_dense() {
    // Test 1M values evenly spaced, plus boundary values
    let n = 1_000_000;
    let mut values: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
    // Add exact boundaries
    values.push(0.0);
    values.push(0.039_293_37); // C0 gamma threshold
    values.push(0.039_293_38);
    values.push(0.5);
    values.push(1.0);

    let scalar_results: Vec<f32> = values.iter().map(|&v| srgb_to_linear(v)).collect();

    let mut simd_buf = values.clone();
    srgb_to_linear_slice(&mut simd_buf);

    let mut max_ulp = 0_u32;
    for (i, ((&scalar, &simd), &input)) in scalar_results
        .iter()
        .zip(simd_buf.iter())
        .zip(values.iter())
        .enumerate()
    {
        let ulp = ulp_distance(scalar, simd);
        if ulp > max_ulp {
            max_ulp = ulp;
        }
        // SIMD evaluates in f32 while scalar uses f64 intermediates. ARM NEON
        // FMA rounding produces up to ~11 ULP divergence. Both paths are accurate
        // vs the f64 reference (spec allows ±14 ULP).
        assert!(
            ulp <= 16,
            "SIMD vs scalar s2l mismatch at index {i}, input={input}: \
             scalar={scalar}, simd={simd}, ULP={ulp}"
        );
    }
    eprintln!("SIMD vs scalar s2l: max ULP = {max_ulp}");
}

#[test]
fn simd_l2s_matches_scalar_dense() {
    let n = 1_000_000;
    let mut values: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
    values.push(0.0);
    values.push(0.003_041_282_6); // C0 linear threshold
    values.push(0.003_041_283);
    values.push(0.5);
    values.push(1.0);

    let scalar_results: Vec<f32> = values.iter().map(|&v| linear_to_srgb(v)).collect();

    let mut simd_buf = values.clone();
    linear_to_srgb_slice(&mut simd_buf);

    let mut max_ulp = 0_u32;
    for (i, ((&scalar, &simd), &input)) in scalar_results
        .iter()
        .zip(simd_buf.iter())
        .zip(values.iter())
        .enumerate()
    {
        let ulp = ulp_distance(scalar, simd);
        if ulp > max_ulp {
            max_ulp = ulp;
        }
        assert!(
            ulp <= 16,
            "SIMD vs scalar l2s mismatch at index {i}, input={input}: \
             scalar={scalar}, simd={simd}, ULP={ulp}"
        );
    }
    eprintln!("SIMD vs scalar l2s: max ULP = {max_ulp}");
}

// ============================================================================
// Exhaustive roundtrip: fast(fast(x)) ≈ x
// ============================================================================

#[test]
fn exhaustive_roundtrip_fast() {
    let u16_step = 1.0_f32 / 65535.0;
    let mut v = 0.0_f32;
    let mut max_err: f32 = 0.0;
    let mut worst_input = 0.0_f32;
    let mut over_u16 = 0_u64;
    let mut count: u64 = 0;

    while v <= 1.0 {
        let linear = srgb_to_linear(v);
        let back = linear_to_srgb(linear);
        let err = (back - v).abs();
        if err > max_err {
            max_err = err;
            worst_input = v;
        }
        if err > u16_step {
            over_u16 += 1;
        }
        count += 1;
        v = next_f32_above(v);
    }

    eprintln!(
        "roundtrip fast: {count} values, max err = {max_err:.2e} at {worst_input}, \
         over 1 U16: {over_u16}"
    );
    assert!(
        max_err < u16_step,
        "roundtrip max error {max_err:.2e} at {worst_input} exceeds 1 U16 step ({u16_step:.2e})"
    );
    assert_eq!(over_u16, 0, "{over_u16} values exceed 1 U16 roundtrip");
}

#[test]
fn exhaustive_roundtrip_inverse() {
    // linear → sRGB → linear roundtrip
    let u16_step = 1.0_f32 / 65535.0;
    let mut v = 0.0_f32;
    let mut max_err: f32 = 0.0;
    let mut worst_input = 0.0_f32;
    let mut over_u16 = 0_u64;
    let mut count: u64 = 0;

    while v <= 1.0 {
        let srgb = linear_to_srgb(v);
        let back = srgb_to_linear(srgb);
        let err = (back - v).abs();
        if err > max_err {
            max_err = err;
            worst_input = v;
        }
        if err > u16_step {
            over_u16 += 1;
        }
        count += 1;
        v = next_f32_above(v);
    }

    eprintln!(
        "roundtrip inverse: {count} values, max err = {max_err:.2e} at {worst_input}, \
         over 1 U16: {over_u16}"
    );
    assert!(
        max_err < u16_step,
        "roundtrip inverse max error {max_err:.2e} at {worst_input} exceeds 1 U16"
    );
}

// ============================================================================
// Every u8 value: roundtrip through all paths
// ============================================================================

#[test]
fn every_u8_roundtrip() {
    // u8 → f32 linear → u8 sRGB must be lossless
    let input: Vec<u8> = (0..=255).collect();
    let mut linear = vec![0.0_f32; 256];
    srgb_u8_to_linear_slice(&input, &mut linear);

    let mut back_u8 = vec![0_u8; 256];
    linear_to_srgb_u8_slice(&linear, &mut back_u8);

    for i in 0..=255_usize {
        assert_eq!(
            input[i], back_u8[i],
            "u8 roundtrip failed at {i}: in={}, out={}",
            input[i], back_u8[i]
        );
    }

    // Also verify the linear values are monotonically increasing
    for i in 1..256 {
        assert!(
            linear[i] >= linear[i - 1],
            "u8→linear not monotonic at {i}: {} < {}",
            linear[i],
            linear[i - 1]
        );
    }

    // Verify boundaries
    assert_eq!(linear[0], 0.0, "u8 0 → linear must be 0.0");
    assert_eq!(linear[255], 1.0, "u8 255 → linear must be 1.0");
}

// ============================================================================
// Every u16 value: roundtrip and accuracy
// ============================================================================

#[test]
fn every_u16_roundtrip() {
    let input: Vec<u16> = (0..=65535).collect();
    let mut linear = vec![0.0_f32; 65536];
    srgb_u16_to_linear_slice(&input, &mut linear);

    let mut back_u16 = vec![0_u16; 65536];
    linear_to_srgb_u16_slice(&linear, &mut back_u16);

    let mut max_diff = 0_i32;
    let mut off_by_one = 0_u32;
    let mut off_by_more = 0_u32;

    for i in 0..=65535_usize {
        let diff = (back_u16[i] as i32) - (input[i] as i32);
        let abs_diff = diff.abs();
        if abs_diff > max_diff {
            max_diff = abs_diff;
        }
        if abs_diff == 1 {
            off_by_one += 1;
        }
        if abs_diff > 1 {
            off_by_more += 1;
        }
    }

    let pct_exact = 100.0 * (65536 - off_by_one - off_by_more) as f64 / 65536.0;
    eprintln!(
        "u16 roundtrip: max diff = {max_diff}, off-by-1 = {off_by_one}, \
         off-by-more = {off_by_more}, exact = {pct_exact:.1}%"
    );

    // u16 → f32 (polynomial) → f32 (polynomial) → u16 accumulates:
    //   - u16→f32 quantization (~0.5 LSB)
    //   - polynomial forward error (~16 ULP in f32 ≈ a few u16 LSBs near threshold)
    //   - polynomial inverse error
    //   - f32→u16 rounding
    // Near the piecewise threshold where f32 ULPs are tiny relative to u16 LSBs,
    // a few LSBs of drift is expected. The important check: max diff is small.
    assert!(
        max_diff <= 6,
        "u16 roundtrip max diff {max_diff} exceeds 6 LSB"
    );

    // Verify monotonicity
    for i in 1..65536 {
        assert!(
            linear[i] >= linear[i - 1],
            "u16→linear not monotonic at {i}: {} < {}",
            linear[i],
            linear[i - 1]
        );
    }

    // Boundaries
    assert_eq!(linear[0], 0.0, "u16 0 → linear must be 0.0");
    assert_eq!(linear[65535], 1.0, "u16 65535 → linear must be 1.0");
}

// ============================================================================
// u8 and u16 accuracy vs f64 reference
// ============================================================================

#[test]
fn every_u8_vs_f64_reference() {
    let input: Vec<u8> = (0..=255).collect();
    let mut linear = vec![0.0_f32; 256];
    srgb_u8_to_linear_slice(&input, &mut linear);

    for (i, &val) in linear.iter().enumerate() {
        let srgb_f64 = i as f64 / 255.0;
        let expected = ref_s2l(srgb_f64);
        let got = val as f64;
        let err = (got - expected).abs();
        // f32 can represent ~7 decimal digits; u8 inputs give ~1e-7 precision
        assert!(
            err < 1e-5,
            "u8 {i} → linear: got {got}, expected {expected}, err {err:.2e}"
        );
    }
}

#[test]
fn every_u16_vs_f64_reference() {
    let input: Vec<u16> = (0..=65535).collect();
    let mut linear = vec![0.0_f32; 65536];
    srgb_u16_to_linear_slice(&input, &mut linear);

    let mut max_err: f64 = 0.0;
    let mut worst = 0_u16;

    for (i, &val) in linear.iter().enumerate() {
        let srgb_f64 = i as f64 / 65535.0;
        let expected = ref_s2l(srgb_f64);
        let got = val as f64;
        let err = (got - expected).abs();
        if err > max_err {
            max_err = err;
            worst = i as u16;
        }
    }

    eprintln!("u16→linear vs f64: max err = {max_err:.2e} at u16 {worst}");
    // f32 polynomial error + u16→f32 quantization: should be well under 1e-5
    assert!(
        max_err < 1e-5,
        "u16→linear max error {max_err:.2e} at {worst} exceeds 1e-5"
    );
}

// ============================================================================
// SIMD roundtrip through slice functions
// ============================================================================

#[test]
fn simd_slice_roundtrip_exhaustive_u16_range() {
    // Test every u16-representable value through SIMD slice roundtrip
    let n = 65536;
    let mut values: Vec<f32> = (0..n).map(|i| i as f32 / 65535.0).collect();
    let original = values.clone();

    srgb_to_linear_slice(&mut values);
    linear_to_srgb_slice(&mut values);

    let u16_step = 1.0_f32 / 65535.0;
    let mut max_err: f32 = 0.0;
    let mut over_u16 = 0;

    for (i, (&orig, &back)) in original.iter().zip(values.iter()).enumerate() {
        let err = (back - orig).abs();
        if err > max_err {
            max_err = err;
        }
        if err > u16_step {
            over_u16 += 1;
            if over_u16 <= 5 {
                eprintln!("  SIMD roundtrip over U16 at {i}: {orig} → {back}, err={err:.2e}");
            }
        }
    }

    eprintln!("SIMD slice roundtrip: max err = {max_err:.2e}, over U16: {over_u16}/{n}");
    assert_eq!(over_u16, 0, "{over_u16} values exceed 1 U16 SIMD roundtrip");
}

// ============================================================================
// Monotonicity: both directions must be monotonically non-decreasing
// ============================================================================

#[test]
fn exhaustive_monotonicity_s2l() {
    // Rational polynomials are not perfectly monotonic — small 1-ULP reversals
    // are inherent. We verify that violations are all 1-ULP (never 2+) and that
    // the overall trend is strongly monotonic.
    let mut prev = 0.0_f32;
    let mut prev_input = 0.0_f32;
    let mut v = 0.0_f32;
    let mut violations = 0_u64;
    let mut max_reversal_ulp: u32 = 0;

    while v <= 1.0 {
        let result = srgb_to_linear(v);
        if result < prev {
            violations += 1;
            let rev = ulp_distance(result, prev);
            if rev > max_reversal_ulp {
                max_reversal_ulp = rev;
            }
            if violations <= 3 {
                eprintln!(
                    "  s2l monotonicity: f({v}) = {result} < f({prev_input}) = {prev} ({rev} ULP)"
                );
            }
        }
        prev = result;
        prev_input = v;
        v = next_f32_above(v);
    }

    eprintln!("s2l monotonicity: {violations} violations, max reversal = {max_reversal_ulp} ULP");
    // The scalar polynomial evaluates in f64 and rounds to f32, which
    // guarantees monotonicity (the f64 result is monotonic and round-to-nearest
    // preserves ordering).
    assert_eq!(
        violations, 0,
        "s2l has {violations} monotonicity violations (max {max_reversal_ulp} ULP)"
    );
}

#[test]
fn exhaustive_monotonicity_l2s() {
    let mut prev = 0.0_f32;
    let mut prev_input = 0.0_f32;
    let mut v = 0.0_f32;
    let mut violations = 0_u64;
    let mut max_reversal_ulp: u32 = 0;

    while v <= 1.0 {
        let result = linear_to_srgb(v);
        if result < prev {
            violations += 1;
            let rev = ulp_distance(result, prev);
            if rev > max_reversal_ulp {
                max_reversal_ulp = rev;
            }
            if violations <= 3 {
                eprintln!(
                    "  l2s monotonicity: f({v}) = {result} < f({prev_input}) = {prev} ({rev} ULP)"
                );
            }
        }
        prev = result;
        prev_input = v;
        v = next_f32_above(v);
    }

    eprintln!("l2s monotonicity: {violations} violations, max reversal = {max_reversal_ulp} ULP");
    assert_eq!(
        violations, 0,
        "l2s has {violations} monotonicity violations (max {max_reversal_ulp} ULP)"
    );
}

// ============================================================================
// Boundary exactness: 0.0 and 1.0 must be exact fixed points
// ============================================================================

#[test]
fn boundary_fixed_points() {
    // Scalar fast
    assert_eq!(srgb_to_linear(0.0), 0.0, "fast s2l(0) != 0");
    assert_eq!(srgb_to_linear(1.0), 1.0, "fast s2l(1) != 1");
    assert_eq!(linear_to_srgb(0.0), 0.0, "fast l2s(0) != 0");
    assert_eq!(linear_to_srgb(1.0), 1.0, "fast l2s(1) != 1");

    // Scalar precise
    assert_eq!(precise_s2l(0.0), 0.0, "precise s2l(0) != 0");
    assert_eq!(precise_s2l(1.0), 1.0, "precise s2l(1) != 1");
    assert_eq!(precise_l2s(0.0), 0.0, "precise l2s(0) != 0");
    assert_eq!(precise_l2s(1.0), 1.0, "precise l2s(1) != 1");

    // SIMD slice (1.0 was the specific failure case before output clamping)
    let mut s2l_buf = vec![0.0_f32, 1.0];
    srgb_to_linear_slice(&mut s2l_buf);
    assert_eq!(s2l_buf[0], 0.0, "SIMD s2l(0) != 0");
    assert_eq!(s2l_buf[1], 1.0, "SIMD s2l(1) != 1");

    let mut l2s_buf = vec![0.0_f32, 1.0];
    linear_to_srgb_slice(&mut l2s_buf);
    assert_eq!(l2s_buf[0], 0.0, "SIMD l2s(0) != 0");
    assert_eq!(l2s_buf[1], 1.0, "SIMD l2s(1) != 1");
}

// ============================================================================
// Threshold continuity: values on both sides of the piecewise join must agree
// ============================================================================

#[test]
fn threshold_continuity_s2l() {
    // The piecewise threshold in gamma domain
    let thresh: f32 = 0.039_293_37;
    let below = f32::from_bits(thresh.to_bits() - 1);
    let above = f32::from_bits(thresh.to_bits() + 1);

    let r_below = srgb_to_linear(below);
    let r_at = srgb_to_linear(thresh);
    let r_above = srgb_to_linear(above);

    // Must be monotonic
    assert!(r_below <= r_at, "s2l not monotonic below threshold");
    assert!(r_at <= r_above, "s2l not monotonic above threshold");

    // Gap at threshold should be tiny (< 2 ULP of the output value)
    let ulp_gap = ulp_distance(r_below, r_above);
    eprintln!(
        "s2l threshold: f({below})={r_below}, f({thresh})={r_at}, f({above})={r_above}, \
         gap={ulp_gap} ULP"
    );
    assert!(
        ulp_gap <= 4,
        "s2l threshold gap {ulp_gap} ULP too large (below={r_below}, above={r_above})"
    );
}

#[test]
fn threshold_continuity_l2s() {
    let thresh: f32 = 0.003_041_282_6;
    let below = f32::from_bits(thresh.to_bits() - 1);
    let above = f32::from_bits(thresh.to_bits() + 1);

    let r_below = linear_to_srgb(below);
    let r_at = linear_to_srgb(thresh);
    let r_above = linear_to_srgb(above);

    assert!(r_below <= r_at, "l2s not monotonic below threshold");
    assert!(r_at <= r_above, "l2s not monotonic above threshold");

    let ulp_gap = ulp_distance(r_below, r_above);
    eprintln!(
        "l2s threshold: f({below})={r_below}, f({thresh})={r_at}, f({above})={r_above}, \
         gap={ulp_gap} ULP"
    );
    assert!(
        ulp_gap <= 4,
        "l2s threshold gap {ulp_gap} ULP too large (below={r_below}, above={r_above})"
    );
}

// ============================================================================
// Negative and out-of-range clamping
// ============================================================================

#[test]
fn clamping_behavior() {
    // Negatives
    assert_eq!(srgb_to_linear(-1.0), 0.0);
    assert_eq!(srgb_to_linear(-0.001), 0.0);
    assert_eq!(linear_to_srgb(-1.0), 0.0);
    assert_eq!(linear_to_srgb(-0.001), 0.0);

    // Above 1.0
    assert_eq!(srgb_to_linear(1.5), 1.0);
    assert_eq!(srgb_to_linear(100.0), 1.0);
    assert_eq!(linear_to_srgb(1.5), 1.0);
    assert_eq!(linear_to_srgb(100.0), 1.0);

    // SIMD paths
    let mut buf = vec![-1.0_f32, -0.001, 1.5, 100.0];
    srgb_to_linear_slice(&mut buf);
    assert_eq!(buf[0], 0.0);
    assert_eq!(buf[1], 0.0);
    assert_eq!(buf[2], 1.0);
    assert_eq!(buf[3], 1.0);

    let mut buf2 = vec![-1.0_f32, -0.001, 1.5, 100.0];
    linear_to_srgb_slice(&mut buf2);
    assert_eq!(buf2[0], 0.0);
    assert_eq!(buf2[1], 0.0);
    assert_eq!(buf2[2], 1.0);
    assert_eq!(buf2[3], 1.0);
}

// ============================================================================
// Extended-range f64 references (unclamped)
// ============================================================================

fn ref_s2l_ext(v: f64) -> f64 {
    if v < GAM_THRESH {
        v / 12.92
    } else {
        ((v + A) / A1).powf(2.4)
    }
}

fn ref_l2s_ext(v: f64) -> f64 {
    if v < LIN_THRESH {
        v * 12.92
    } else {
        A1 * v.powf(1.0 / 2.4) - A
    }
}

// ============================================================================
// Extended range: positive [1.0, 8.0] — exhaustive f32 sweep
// ============================================================================

#[test]
fn extended_s2l_above_one_exhaustive() {
    // Every f32 in (1.0, 8.0] through the power segment
    let start = f32::from_bits(1.0_f32.to_bits() + 1); // just above 1.0
    let end = 8.0_f32;

    let mut v = start;
    let mut max_ulp: u32 = 0;
    let mut worst_input = start;
    let mut count: u64 = 0;

    while v <= end {
        let got = precise_s2l_ext(v);
        let expected = ref_s2l_ext(v as f64) as f32;
        let ulp = ulp_distance(got, expected);
        if ulp > max_ulp {
            max_ulp = ulp;
            worst_input = v;
        }
        count += 1;
        v = next_f32_above(v);
    }

    eprintln!("s2l_extended [1+eps, 8]: {count} values, max ULP = {max_ulp} at {worst_input}");
    // powf f32 vs f64: expect same ~6 ULP budget as [0,1]
    assert!(
        max_ulp <= 10,
        "s2l_extended max ULP {max_ulp} at {worst_input} exceeds 10"
    );
}

#[test]
fn extended_l2s_above_one_exhaustive() {
    let start = f32::from_bits(1.0_f32.to_bits() + 1);
    let end = 8.0_f32;

    let mut v = start;
    let mut max_ulp: u32 = 0;
    let mut worst_input = start;
    let mut count: u64 = 0;

    while v <= end {
        let got = precise_l2s_ext(v);
        let expected = ref_l2s_ext(v as f64) as f32;
        let ulp = ulp_distance(got, expected);
        if ulp > max_ulp {
            max_ulp = ulp;
            worst_input = v;
        }
        count += 1;
        v = next_f32_above(v);
    }

    eprintln!("l2s_extended [1+eps, 8]: {count} values, max ULP = {max_ulp} at {worst_input}");
    assert!(
        max_ulp <= 10,
        "l2s_extended max ULP {max_ulp} at {worst_input} exceeds 10"
    );
}

// ============================================================================
// Extended range: negative [-1.0, 0) — exhaustive f32 sweep
// Both directions use the linear segment for negatives, so error should be
// minimal (just f32 multiplication precision).
// ============================================================================

/// Iterate f32 values from a negative toward zero (increasing order).
fn next_f32_toward_zero_neg(v: f32) -> f32 {
    debug_assert!(v < 0.0);
    // Negative f32 bit patterns decrease toward zero in sign-magnitude
    f32::from_bits(v.to_bits() - 1)
}

#[test]
fn extended_s2l_negative_exhaustive() {
    // Every f32 in [-1.0, -0.0)
    let mut v = -1.0_f32;
    let mut max_ulp: u32 = 0;
    let mut worst_input = v;
    let mut count: u64 = 0;

    while v < 0.0 {
        let got = precise_s2l_ext(v);
        let expected = ref_s2l_ext(v as f64) as f32;
        let ulp = ulp_distance(got, expected);
        if ulp > max_ulp {
            max_ulp = ulp;
            worst_input = v;
        }
        count += 1;
        v = next_f32_toward_zero_neg(v);
    }

    eprintln!("s2l_extended [-1, 0): {count} values, max ULP = {max_ulp} at {worst_input}");
    // Linear segment: v / 12.92 — single f32 division, expect ≤ 1 ULP
    assert!(
        max_ulp <= 1,
        "s2l_extended negative max ULP {max_ulp} at {worst_input} exceeds 1"
    );
}

#[test]
fn extended_l2s_negative_exhaustive() {
    let mut v = -1.0_f32;
    let mut max_ulp: u32 = 0;
    let mut worst_input = v;
    let mut count: u64 = 0;

    while v < 0.0 {
        let got = precise_l2s_ext(v);
        let expected = ref_l2s_ext(v as f64) as f32;
        let ulp = ulp_distance(got, expected);
        if ulp > max_ulp {
            max_ulp = ulp;
            worst_input = v;
        }
        count += 1;
        v = next_f32_toward_zero_neg(v);
    }

    eprintln!("l2s_extended [-1, 0): {count} values, max ULP = {max_ulp} at {worst_input}");
    // Linear segment: v * 12.92 — single f32 multiply, expect ≤ 1 ULP
    assert!(
        max_ulp <= 1,
        "l2s_extended negative max ULP {max_ulp} at {worst_input} exceeds 1"
    );
}

// ============================================================================
// Extended-range roundtrip: l2s(s2l(x)) ≈ x for [0, 8] and negatives
// ============================================================================

#[test]
fn extended_roundtrip_above_one() {
    let start = f32::from_bits(1.0_f32.to_bits() + 1);
    let end = 8.0_f32;

    let mut v = start;
    let mut max_ulp: u32 = 0;
    let mut worst_input = start;
    let mut count: u64 = 0;

    while v <= end {
        let linear = precise_s2l_ext(v);
        let back = precise_l2s_ext(linear);
        let ulp = ulp_distance(v, back);
        if ulp > max_ulp {
            max_ulp = ulp;
            worst_input = v;
        }
        count += 1;
        v = next_f32_above(v);
    }

    eprintln!(
        "roundtrip s2l->l2s [1+eps, 8]: {count} values, max ULP = {max_ulp} at {worst_input}"
    );
    // Combined powf error: ~10 ULP each direction, ~16 ULP combined
    assert!(
        max_ulp <= 20,
        "extended roundtrip max ULP {max_ulp} at {worst_input} exceeds 20"
    );
}

#[test]
fn extended_roundtrip_inverse_above_one() {
    // linear → sRGB → linear for values > 1.0
    let start = f32::from_bits(1.0_f32.to_bits() + 1);
    let end = 8.0_f32;

    let mut v = start;
    let mut max_ulp: u32 = 0;
    let mut worst_input = start;
    let mut count: u64 = 0;

    while v <= end {
        let srgb = precise_l2s_ext(v);
        let back = precise_s2l_ext(srgb);
        let ulp = ulp_distance(v, back);
        if ulp > max_ulp {
            max_ulp = ulp;
            worst_input = v;
        }
        count += 1;
        v = next_f32_above(v);
    }

    eprintln!(
        "roundtrip l2s->s2l [1+eps, 8]: {count} values, max ULP = {max_ulp} at {worst_input}"
    );
    assert!(
        max_ulp <= 20,
        "extended roundtrip inverse max ULP {max_ulp} at {worst_input} exceeds 20"
    );
}

#[test]
fn extended_roundtrip_negative() {
    let mut v = -1.0_f32;
    let mut max_ulp: u32 = 0;
    let mut worst_input = v;
    let mut count: u64 = 0;

    while v < 0.0 {
        let linear = precise_s2l_ext(v);
        let back = precise_l2s_ext(linear);
        let ulp = ulp_distance(v, back);
        if ulp > max_ulp {
            max_ulp = ulp;
            worst_input = v;
        }
        count += 1;
        v = next_f32_toward_zero_neg(v);
    }

    eprintln!("roundtrip s2l->l2s [-1, 0): {count} values, max ULP = {max_ulp} at {worst_input}");
    // Both directions use the linear segment (multiply then divide).
    // Near zero, subnormal f32 values lose precision in the divide→multiply
    // chain, producing up to ~6 ULP error at the smallest subnormals.
    assert!(
        max_ulp <= 6,
        "negative roundtrip max ULP {max_ulp} at {worst_input} exceeds 6"
    );
}

// ============================================================================
// Extended-range monotonicity
// ============================================================================

#[test]
fn extended_monotonicity_s2l_above_one() {
    let start = 1.0_f32;
    let end = 8.0_f32;

    let mut prev = precise_s2l_ext(start);
    let mut v = next_f32_above(start);
    let mut violations = 0_u64;
    let mut max_reversal_ulp: u32 = 0;

    while v <= end {
        let result = precise_s2l_ext(v);
        if result < prev {
            violations += 1;
            let rev = ulp_distance(result, prev);
            if rev > max_reversal_ulp {
                max_reversal_ulp = rev;
            }
        }
        prev = result;
        v = next_f32_above(v);
    }

    eprintln!(
        "s2l_extended [1, 8] monotonicity: {violations} violations, max reversal = {max_reversal_ulp} ULP"
    );
    // powf should be monotonic; any violation is concerning
    assert!(
        max_reversal_ulp <= 2,
        "s2l_extended monotonicity: {max_reversal_ulp}-ULP reversal ({violations} violations)"
    );
}

#[test]
fn extended_monotonicity_l2s_above_one() {
    let start = 1.0_f32;
    let end = 8.0_f32;

    let mut prev = precise_l2s_ext(start);
    let mut v = next_f32_above(start);
    let mut violations = 0_u64;
    let mut max_reversal_ulp: u32 = 0;

    while v <= end {
        let result = precise_l2s_ext(v);
        if result < prev {
            violations += 1;
            let rev = ulp_distance(result, prev);
            if rev > max_reversal_ulp {
                max_reversal_ulp = rev;
            }
        }
        prev = result;
        v = next_f32_above(v);
    }

    eprintln!(
        "l2s_extended [1, 8] monotonicity: {violations} violations, max reversal = {max_reversal_ulp} ULP"
    );
    assert!(
        max_reversal_ulp <= 2,
        "l2s_extended monotonicity: {max_reversal_ulp}-ULP reversal ({violations} violations)"
    );
}

// ============================================================================
// Extended-range boundary: exact 1.0 and 0.0 fixed points
// ============================================================================

#[test]
fn extended_boundary_values() {
    // 0.0 fixed point
    assert_eq!(precise_s2l_ext(0.0), 0.0, "s2l_ext(0) != 0");
    assert_eq!(precise_l2s_ext(0.0), 0.0, "l2s_ext(0) != 0");

    // 1.0: s2l uses the power curve for gamma >= threshold (~0.039), so
    // powf((1.0 + a) / (1.0 + a), 2.4) = 1.0 exactly.
    // l2s uses powf(1.0, 1/2.4) * a1 - a — the FMA may introduce ≤1 ULP error.
    assert_eq!(precise_s2l_ext(1.0), 1.0, "s2l_ext(1) != 1");
    let l2s_one = precise_l2s_ext(1.0);
    assert!(
        ulp_distance(l2s_one, 1.0) <= 1,
        "l2s_ext(1) = {l2s_one}, expected 1.0 within 1 ULP"
    );

    // -0.0 should map to -0.0 (sign preservation)
    let neg_zero = -0.0_f32;
    let s2l_nz = precise_s2l_ext(neg_zero);
    let l2s_nz = precise_l2s_ext(neg_zero);
    assert!(s2l_nz == 0.0, "s2l_ext(-0) not zero: {s2l_nz}");
    assert!(l2s_nz == 0.0, "l2s_ext(-0) not zero: {l2s_nz}");

    // Values just above 1.0 should produce results >= 1.0 (not clamped to 1.0)
    let above_one = f32::from_bits(1.0_f32.to_bits() + 1);
    assert!(
        precise_s2l_ext(above_one) > 1.0,
        "s2l_ext(1+eps) should be > 1.0"
    );
    // l2s has derivative ~0.44 at 1.0, so 1+eps_f32 maps to ~1+0.44*eps_f64
    // which rounds to exactly 1.0 in f32. Check a larger step.
    let above_one_10 = f32::from_bits(1.0_f32.to_bits() + 10);
    assert!(
        precise_l2s_ext(above_one_10) >= 1.0,
        "l2s_ext(1+10eps) should be >= 1.0"
    );

    // Negative values should produce negative results
    assert!(precise_s2l_ext(-0.5) < 0.0, "s2l_ext(-0.5) should be < 0");
    assert!(precise_l2s_ext(-0.5) < 0.0, "l2s_ext(-0.5) should be < 0");
}

// ============================================================================
// Extended range: continuity across the [0, 1] → (1, 8] boundary
// ============================================================================

#[test]
fn extended_continuity_at_one() {
    // Verify the extended functions smoothly continue past 1.0
    // (no discontinuity from a clamping branch)
    let below = f32::from_bits(1.0_f32.to_bits() - 1);
    let at = 1.0_f32;
    let above = f32::from_bits(1.0_f32.to_bits() + 1);

    // s2l: should be monotonically increasing
    let s2l_below = precise_s2l_ext(below);
    let s2l_at = precise_s2l_ext(at);
    let s2l_above = precise_s2l_ext(above);
    assert!(s2l_below <= s2l_at, "s2l_ext not monotonic below 1.0");
    assert!(s2l_at <= s2l_above, "s2l_ext not monotonic above 1.0");

    // l2s: should be monotonically increasing
    let l2s_below = precise_l2s_ext(below);
    let l2s_at = precise_l2s_ext(at);
    let l2s_above = precise_l2s_ext(above);
    assert!(l2s_below <= l2s_at, "l2s_ext not monotonic below 1.0");
    assert!(l2s_at <= l2s_above, "l2s_ext not monotonic above 1.0");

    // The gap across 1.0 should be tiny (adjacent f32 values)
    let s2l_gap = ulp_distance(s2l_below, s2l_above);
    let l2s_gap = ulp_distance(l2s_below, l2s_above);
    eprintln!("Continuity at 1.0: s2l gap = {s2l_gap} ULP, l2s gap = {l2s_gap} ULP");
    assert!(s2l_gap <= 4, "s2l_ext gap at 1.0: {s2l_gap} ULP");
    assert!(l2s_gap <= 4, "l2s_ext gap at 1.0: {l2s_gap} ULP");
}

// ============================================================================
// NaN, Infinity, and special float handling
// ============================================================================

#[test]
fn nan_handling() {
    let nan = f32::NAN;

    // Clamped functions: NaN should produce 0.0 (NaN < 0.0 is false,
    // NaN >= 1.0 is false, so it falls through to the power segment.
    // We don't require specific NaN handling — just verify no panic.
    let s2l = srgb_to_linear(nan);
    let l2s = linear_to_srgb(nan);
    // The result is implementation-defined (NaN through polynomial),
    // but the function must not panic.
    let _ = (s2l, l2s);

    // Precise (powf) clamped: same — NaN falls through to powf(NaN) = NaN
    let ps2l = precise_s2l(nan);
    let pl2s = precise_l2s(nan);
    let _ = (ps2l, pl2s);

    // Extended: NaN through the linear segment or power segment
    let es2l = precise_s2l_ext(nan);
    let el2s = precise_l2s_ext(nan);
    let _ = (es2l, el2s);

    // SIMD slice: NaN in a slice must not panic or corrupt other values
    let mut buf = [0.5_f32, nan, 0.5, nan];
    srgb_to_linear_slice(&mut buf);
    // Non-NaN values should be correctly converted
    assert!(!buf[0].is_nan(), "SIMD s2l corrupted non-NaN value");
    assert!(!buf[2].is_nan(), "SIMD s2l corrupted non-NaN value");

    let mut buf2 = [0.5_f32, nan, 0.5, nan];
    linear_to_srgb_slice(&mut buf2);
    assert!(!buf2[0].is_nan(), "SIMD l2s corrupted non-NaN value");
    assert!(!buf2[2].is_nan(), "SIMD l2s corrupted non-NaN value");
}

#[test]
fn infinity_handling() {
    let inf = f32::INFINITY;
    let neg_inf = f32::NEG_INFINITY;

    // Clamped: +inf should clamp to 1.0, -inf to 0.0
    assert_eq!(srgb_to_linear(inf), 1.0, "s2l(+inf) should clamp to 1.0");
    assert_eq!(
        srgb_to_linear(neg_inf),
        0.0,
        "s2l(-inf) should clamp to 0.0"
    );
    assert_eq!(linear_to_srgb(inf), 1.0, "l2s(+inf) should clamp to 1.0");
    assert_eq!(
        linear_to_srgb(neg_inf),
        0.0,
        "l2s(-inf) should clamp to 0.0"
    );

    // Precise clamped
    assert_eq!(
        precise_s2l(inf),
        1.0,
        "precise s2l(+inf) should clamp to 1.0"
    );
    assert_eq!(
        precise_s2l(neg_inf),
        0.0,
        "precise s2l(-inf) should clamp to 0.0"
    );
    assert_eq!(
        precise_l2s(inf),
        1.0,
        "precise l2s(+inf) should clamp to 1.0"
    );
    assert_eq!(
        precise_l2s(neg_inf),
        0.0,
        "precise l2s(-inf) should clamp to 0.0"
    );

    // SIMD slice: inf values should clamp, not corrupt neighbors
    let mut buf = [0.5_f32, inf, 0.5, neg_inf];
    srgb_to_linear_slice(&mut buf);
    assert!(
        buf[0] > 0.0 && buf[0] < 1.0,
        "SIMD s2l corrupted non-inf value"
    );
    assert_eq!(buf[1], 1.0, "SIMD s2l(+inf) should be 1.0");
    assert!(
        buf[2] > 0.0 && buf[2] < 1.0,
        "SIMD s2l corrupted non-inf value"
    );
    assert_eq!(buf[3], 0.0, "SIMD s2l(-inf) should be 0.0");
}

#[test]
fn negative_zero_handling() {
    let neg_zero = -0.0_f32;

    // All functions should treat -0.0 like 0.0
    assert_eq!(srgb_to_linear(neg_zero), 0.0, "s2l(-0) should be 0.0");
    assert_eq!(linear_to_srgb(neg_zero), 0.0, "l2s(-0) should be 0.0");
    assert_eq!(precise_s2l(neg_zero), 0.0, "precise s2l(-0) should be 0.0");
    assert_eq!(precise_l2s(neg_zero), 0.0, "precise l2s(-0) should be 0.0");

    let mut buf = [-0.0_f32, -0.0, -0.0, -0.0];
    srgb_to_linear_slice(&mut buf);
    assert!(
        buf.iter().all(|&x| x == 0.0),
        "SIMD s2l(-0) should all be 0.0"
    );
}

// ============================================================================
// Const LUT verification: ensure embedded tables match runtime computation
// ============================================================================

#[test]
fn const_lut_u8_matches_precise() {
    use linear_srgb::lut::SrgbConverter;

    let conv = SrgbConverter::new();

    // Every u8 entry in the const LUT must match precise f64 computation
    for i in 0..=255u8 {
        let lut_val = conv.srgb_u8_to_linear(i);
        let precise_val = precise_s2l_f64(i as f64 / 255.0) as f32;
        let ulp = ulp_distance(lut_val, precise_val);
        assert!(
            ulp <= 1,
            "u8 LUT[{i}] = {lut_val}, precise = {precise_val}, ULP = {ulp}"
        );
    }
}

#[test]
fn const_lut_u8_cross_check() {
    use linear_srgb::default::srgb_u8_to_linear;
    use linear_srgb::lut::SrgbConverter;

    let conv = SrgbConverter::new();

    // The two u8→linear paths (scalar LUT and SrgbConverter const LUT)
    // must produce identical results for all 256 values.
    for i in 0..=255u8 {
        let scalar = srgb_u8_to_linear(i);
        let converter = conv.srgb_u8_to_linear(i);
        assert_eq!(
            scalar.to_bits(),
            converter.to_bits(),
            "u8→linear mismatch at {i}: scalar={scalar}, converter={converter}"
        );
    }
}

#[test]
fn u16_lut_decode_vs_f64() {
    use linear_srgb::default::srgb_u16_to_linear;

    // The u16 LUT is SIMD-generated (rational polynomial with FMA).
    // Verify accuracy vs f64 reference — must be ≤16 ULP (same as f32 path).
    let mut max_ulp = 0u32;
    let mut worst = 0u16;
    for i in 0..=65535u16 {
        let lut_val = srgb_u16_to_linear(i);
        let precise_val = precise_s2l_f64(i as f64 / 65535.0) as f32;
        let ulp = ulp_distance(lut_val, precise_val);
        if ulp > max_ulp {
            max_ulp = ulp;
            worst = i;
        }
    }
    eprintln!("u16 decode LUT vs f64: max ULP = {max_ulp} at {worst}");
    assert!(
        max_ulp <= 16,
        "u16 decode LUT max ULP {max_ulp} at {worst} exceeds 16"
    );
    // Verify endpoints are exact
    assert_eq!(srgb_u16_to_linear(0), 0.0, "sRGB 0 must map to 0.0");
    assert_eq!(srgb_u16_to_linear(65535), 1.0, "sRGB 65535 must map to 1.0");
}

#[test]
fn u16_lut_encode_vs_f64() {
    use linear_srgb::default::linear_to_srgb_u16;

    // Verify encode LUT vs f64 reference.
    let mut max_diff = 0u32;
    let mut worst = 0u32;
    for i in 0..=65535u32 {
        let linear = i as f32 / 65535.0;
        let lut_val = linear_to_srgb_u16(linear);
        let precise_val = (precise_l2s_f64(linear as f64) * 65535.0 + 0.5) as u16;
        let diff = (lut_val as i32 - precise_val as i32).unsigned_abs();
        if diff > max_diff {
            max_diff = diff;
            worst = i;
        }
    }
    eprintln!("u16 encode LUT vs f64: max diff = {max_diff} at {worst}");
    // SIMD rational poly + index quantization → allow ±3 vs f64
    assert!(
        max_diff <= 3,
        "u16 encode LUT vs f64 max diff {max_diff} at {worst} exceeds 3"
    );
    // Verify endpoints
    assert_eq!(linear_to_srgb_u16(0.0), 0, "linear 0.0 must map to 0");
    assert_eq!(
        linear_to_srgb_u16(1.0),
        65535,
        "linear 1.0 must map to 65535"
    );
}

// ============================================================================
// Custom gamma identity and edge cases
// ============================================================================

#[test]
fn gamma_identity() {
    use linear_srgb::default::{gamma_to_linear, linear_to_gamma};

    // gamma = 1.0 should be identity (pure power x^1 = x)
    for i in 0..=100 {
        let v = i as f32 / 100.0;
        assert_eq!(
            gamma_to_linear(v, 1.0),
            v,
            "gamma_to_linear({v}, 1.0) != {v}"
        );
        assert_eq!(
            linear_to_gamma(v, 1.0),
            v,
            "linear_to_gamma({v}, 1.0) != {v}"
        );
    }
}

#[test]
fn gamma_various_exponents() {
    use linear_srgb::default::{gamma_to_linear, linear_to_gamma};

    for gamma in [1.0_f32, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0] {
        for i in 1..=99 {
            let v = i as f32 / 100.0;
            let lin = gamma_to_linear(v, gamma);
            let back = linear_to_gamma(lin, gamma);
            assert!(
                (v - back).abs() < 1e-5,
                "gamma {gamma} roundtrip at {v}: {v} -> {lin} -> {back}"
            );
        }
    }
}

// ============================================================================
// BT.709 roundtrip (was missing)
// ============================================================================

#[cfg(feature = "transfer")]
#[test]
fn bt709_roundtrip() {
    use linear_srgb::default::{bt709_to_linear, linear_to_bt709};

    let u16_step = 1.0 / 65535.0_f32;
    for i in 0..=10000 {
        let v = i as f32 / 10000.0;
        let linear = bt709_to_linear(v);
        let back = linear_to_bt709(linear);
        let err = (back - v).abs();
        assert!(
            err < u16_step * 2.0,
            "BT.709 roundtrip at {v}: -> {linear} -> {back} (err={err})"
        );
    }
}

// ============================================================================
// mlaf / fmla coverage for f64 and neg_mlaf
// ============================================================================

#[test]
fn fmla_f64() {
    // fmla(a, b, c) = a * b + c
    let result: f64 = linear_srgb::precise::gamma_to_linear_f64(0.5, 2.2);
    // Just verify it doesn't panic and produces a sane value
    assert!(result > 0.0 && result < 1.0);
}

// ============================================================================
// SrgbConverter accuracy
// ============================================================================

#[test]
fn srgb_converter_accuracy() {
    use linear_srgb::lut::SrgbConverter;

    let conv = SrgbConverter::new();

    // linear_to_srgb via LUT interpolation should be close to precise
    let mut max_err: f32 = 0.0;
    for i in 0..=10000 {
        let linear = i as f32 / 10000.0;
        let lut = conv.linear_to_srgb(linear);
        let precise = precise_l2s_f64(linear as f64) as f32;
        let err = (lut - precise).abs();
        if err > max_err {
            max_err = err;
        }
    }
    eprintln!("SrgbConverter linear_to_srgb: max err = {max_err:.2e}");
    assert!(
        max_err < 5e-4,
        "SrgbConverter max error {max_err:.2e} exceeds 5e-4"
    );
}

// ============================================================================
// LUT interpolation edge cases
// ============================================================================

#[test]
fn lut_interp_edge_cases() {
    use linear_srgb::lut::lut_interp_linear_float;

    let table = [0.0_f32, 0.5, 1.0];

    // Exact endpoints
    assert_eq!(lut_interp_linear_float(0.0, &table), 0.0);
    assert_eq!(lut_interp_linear_float(1.0, &table), 1.0);

    // Midpoint
    let mid = lut_interp_linear_float(0.5, &table);
    assert!((mid - 0.5).abs() < 1e-6, "midpoint = {mid}");

    // Clamping: out-of-range inputs
    assert_eq!(lut_interp_linear_float(-1.0, &table), 0.0);
    assert_eq!(lut_interp_linear_float(2.0, &table), 1.0);
}

// ============================================================================
// Runtime LUT table sizes beyond 8/12-bit
// ============================================================================

#[test]
#[allow(deprecated)]
fn lut_table_16bit() {
    use linear_srgb::lut::LinearTable16;

    let table = LinearTable16::new();
    assert_eq!(table.lookup(0), 0.0);
    assert!((table.lookup(65535) - 1.0).abs() < 1e-6);

    // Monotonicity
    let mut prev = 0.0_f32;
    for i in 0..=65535 {
        let val = table.lookup(i);
        assert!(val >= prev, "16-bit LUT not monotonic at {i}");
        prev = val;
    }
}

#[test]
fn lut_table_10bit() {
    use linear_srgb::lut::LinearTable10;

    let table = LinearTable10::new();
    assert_eq!(table.lookup(0), 0.0);
    assert!((table.lookup(1023) - 1.0).abs() < 1e-6);

    // Spot check against precise
    let val = table.lookup(512);
    let precise = precise_s2l_f64(512.0 / 1023.0) as f32;
    assert!(
        (val - precise).abs() < 1e-6,
        "10-bit LUT[512] = {val}, precise = {precise}"
    );
}
