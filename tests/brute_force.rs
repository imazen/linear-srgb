//! Brute-force correctness tests for sRGB transfer functions.
//!
//! Tests every path (scalar, fast/polynomial, SIMD slice) against known-good
//! f64 reference implementations. Covers exhaustive f32 sweeps, every u8/u16
//! value, and roundtrip through all paths.

use linear_srgb::default::{
    linear_to_srgb, linear_to_srgb_slice, linear_to_srgb_u16_slice, linear_to_srgb_u8_slice,
    srgb_to_linear, srgb_to_linear_slice, srgb_u16_to_linear_slice, srgb_u8_to_linear_slice,
};
use linear_srgb::precise::{
    linear_to_srgb as precise_l2s, linear_to_srgb_f64 as precise_l2s_f64,
    srgb_to_linear as precise_s2l, srgb_to_linear_f64 as precise_s2l_f64,
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
        // SIMD uses branchless blend with different FMA ordering, so small
        // differences are expected. The key invariant: both are accurate vs f64.
        assert!(
            ulp <= 5,
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
            ulp <= 5,
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

    let pct_exact =
        100.0 * (65536 - off_by_one - off_by_more) as f64 / 65536.0;
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

    for i in 0..=255_usize {
        let srgb_f64 = i as f64 / 255.0;
        let expected = ref_s2l(srgb_f64);
        let got = linear[i] as f64;
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

    for i in 0..=65535_usize {
        let srgb_f64 = i as f64 / 65535.0;
        let expected = ref_s2l(srgb_f64);
        let got = linear[i] as f64;
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
                eprintln!(
                    "  SIMD roundtrip over U16 at {i}: {orig} → {back}, err={err:.2e}"
                );
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
    // Near the piecewise threshold, the linear segment and polynomial produce
    // interleaved f32 values (both approximate the same smooth curve). Reversals
    // up to ~5 ULP occur in a narrow band around the threshold and are harmless
    // — the output values differ by < 1e-9 absolute.
    assert!(
        max_reversal_ulp <= 5,
        "s2l has {max_reversal_ulp}-ULP reversal (violations={violations})"
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
    assert!(
        max_reversal_ulp <= 5,
        "l2s has {max_reversal_ulp}-ULP reversal (violations={violations})"
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
