//! Accuracy analysis and ULP comparison utilities.

/// Compute ULP (units in last place) distance between two f32 values.
///
/// Returns the number of representable floats between `a` and `b`.
/// Sign-magnitude floats are handled correctly.
#[inline]
pub fn ulp_distance_f32(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    if a == b {
        return 0;
    }

    // Convert to lexicographically ordered integer representation
    fn to_lexical(x: f32) -> i32 {
        let bits = x.to_bits() as i32;
        if bits < 0 {
            // Negative float: flip all bits except sign to get proper ordering
            i32::MIN - bits
        } else {
            bits
        }
    }

    let a_lex = to_lexical(a);
    let b_lex = to_lexical(b);

    (a_lex.wrapping_sub(b_lex)).unsigned_abs()
}

/// Compute ULP distance between two f64 values.
#[inline]
pub fn ulp_distance_f64(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    if a == b {
        return 0;
    }

    fn to_lexical(x: f64) -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    }

    let a_lex = to_lexical(a);
    let b_lex = to_lexical(b);

    (a_lex.wrapping_sub(b_lex)).unsigned_abs()
}

/// Naive sRGB to linear conversion using only std library.
/// This serves as the "textbook" reference implementation.
#[inline]
pub fn naive_srgb_to_linear_f64(gamma: f64) -> f64 {
    if gamma <= 0.04045 {
        gamma / 12.92
    } else {
        ((gamma + 0.055) / 1.055).powf(2.4)
    }
}

/// Naive linear to sRGB conversion using only std library.
#[inline]
pub fn naive_linear_to_srgb_f64(linear: f64) -> f64 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// Naive sRGB to linear conversion (f32).
#[inline]
pub fn naive_srgb_to_linear_f32(gamma: f32) -> f32 {
    if gamma <= 0.04045 {
        gamma / 12.92
    } else {
        ((gamma + 0.055) / 1.055).powf(2.4)
    }
}

/// Naive linear to sRGB conversion (f32).
#[inline]
pub fn naive_linear_to_srgb_f32(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// Statistics about ULP differences.
#[derive(Debug, Clone, Default)]
pub struct UlpStats {
    pub max_ulp: u64,
    pub total_ulp: u128,
    pub count: u64,
    pub max_ulp_input: f64,
    pub max_ulp_expected: f64,
    pub max_ulp_actual: f64,
}

impl UlpStats {
    /// Compute average ULP difference.
    #[inline]
    pub fn avg_ulp(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_ulp as f64 / self.count as f64
        }
    }
}

/// Compare an f32 implementation against f64 reference across a range.
pub fn compare_f32_to_f64_reference<F, G>(
    f32_impl: F,
    f64_reference: G,
    start: f32,
    end: f32,
    steps: u32,
) -> UlpStats
where
    F: Fn(f32) -> f32,
    G: Fn(f64) -> f64,
{
    let mut stats = UlpStats::default();

    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let input = start + t * (end - start);

        let f64_result = f64_reference(input as f64);
        let f32_result = f32_impl(input);
        let expected_f32 = f64_result as f32;

        let ulp = ulp_distance_f32(f32_result, expected_f32) as u64;
        stats.total_ulp += ulp as u128;
        stats.count += 1;

        if ulp > stats.max_ulp {
            stats.max_ulp = ulp;
            stats.max_ulp_input = input as f64;
            stats.max_ulp_expected = f64_result;
            stats.max_ulp_actual = f32_result as f64;
        }
    }

    stats
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::scalar::{linear_to_srgb, linear_to_srgb_f64, srgb_to_linear, srgb_to_linear_f64};

    #[test]
    fn test_ulp_distance() {
        assert_eq!(ulp_distance_f32(1.0, 1.0), 0);
        assert_eq!(ulp_distance_f32(0.0, 0.0), 0);

        // Adjacent floats should have ULP distance of 1
        let x = 1.0f32;
        let next = f32::from_bits(x.to_bits() + 1);
        assert_eq!(ulp_distance_f32(x, next), 1);
    }

    #[test]
    fn test_srgb_to_linear_f32_vs_f64() {
        // Compare our f32 implementation against our f64 implementation
        // This measures precision loss from f32, not algorithm differences
        let stats =
            compare_f32_to_f64_reference(srgb_to_linear, srgb_to_linear_f64, 0.0, 1.0, 10000);

        println!("sRGB->Linear (f32 vs f64 same algorithm):");
        println!("  Max ULP: {}", stats.max_ulp);
        println!("  Avg ULP: {:.2}", stats.avg_ulp());
        println!(
            "  Worst case: input={:.6}, f64={:.10}, f32={:.10}",
            stats.max_ulp_input, stats.max_ulp_expected, stats.max_ulp_actual
        );

        // f32 pow() introduces some ULP error; 8 ULP is still excellent
        assert!(
            stats.max_ulp <= 8,
            "Max ULP {} exceeds threshold",
            stats.max_ulp
        );
    }

    #[test]
    fn test_linear_to_srgb_f32_vs_f64() {
        let stats =
            compare_f32_to_f64_reference(linear_to_srgb, linear_to_srgb_f64, 0.0, 1.0, 10000);

        println!("Linear->sRGB (f32 vs f64 same algorithm):");
        println!("  Max ULP: {}", stats.max_ulp);
        println!("  Avg ULP: {:.2}", stats.avg_ulp());
        println!(
            "  Worst case: input={:.6}, f64={:.10}, f32={:.10}",
            stats.max_ulp_input, stats.max_ulp_expected, stats.max_ulp_actual
        );

        assert!(
            stats.max_ulp <= 8,
            "Max ULP {} exceeds threshold",
            stats.max_ulp
        );
    }

    #[test]
    fn test_vs_naive_srgb_to_linear() {
        // Compare our implementation against naive textbook constants
        // Documents the difference due to using more precise C0-continuous constants
        let stats =
            compare_f32_to_f64_reference(srgb_to_linear, naive_srgb_to_linear_f64, 0.0, 1.0, 10000);

        println!("sRGB->Linear vs naive textbook:");
        println!("  Max ULP: {}", stats.max_ulp);
        println!("  Avg ULP: {:.2}", stats.avg_ulp());
        println!(
            "  Worst case: input={:.6}, naive={:.10}, ours={:.10}",
            stats.max_ulp_input, stats.max_ulp_expected, stats.max_ulp_actual
        );

        // The difference near threshold (~0.04) is larger due to different constants
        // This is expected and documented
    }

    #[test]
    fn test_vs_naive_linear_to_srgb() {
        let stats =
            compare_f32_to_f64_reference(linear_to_srgb, naive_linear_to_srgb_f64, 0.0, 1.0, 10000);

        println!("Linear->sRGB vs naive textbook:");
        println!("  Max ULP: {}", stats.max_ulp);
        println!("  Avg ULP: {:.2}", stats.avg_ulp());
        println!(
            "  Worst case: input={:.6}, naive={:.10}, ours={:.10}",
            stats.max_ulp_input, stats.max_ulp_expected, stats.max_ulp_actual
        );
    }

    /// Sweep every f32 bit pattern in [start, end], comparing f32_impl against
    /// f64 reference. Returns (max_ulp, avg_ulp, worst_input, count).
    fn exhaustive_ulp_sweep<F, G>(
        f32_impl: F,
        f64_reference: G,
        start: f32,
        end: f32,
    ) -> (u32, f64, f32, u64)
    where
        F: Fn(f32) -> f32,
        G: Fn(f64) -> f64,
    {
        let start_bits = start.to_bits();
        let end_bits = end.to_bits();
        let mut max_ulp: u32 = 0;
        let mut total_ulp: u128 = 0;
        let mut worst_input: f32 = start;
        let mut count: u64 = 0;

        let mut bits = start_bits;
        loop {
            let input = f32::from_bits(bits);
            let expected = f64_reference(input as f64) as f32;
            let actual = f32_impl(input);
            let ulp = ulp_distance_f32(actual, expected);
            total_ulp += ulp as u128;
            count += 1;
            if ulp > max_ulp {
                max_ulp = ulp;
                worst_input = input;
            }
            if bits == end_bits {
                break;
            }
            bits += 1;
        }

        let avg = total_ulp as f64 / count as f64;
        (max_ulp, avg, worst_input, count)
    }

    #[test]
    fn test_srgb_to_linear_fast_exhaustive_ulp() {
        use crate::scalar::{srgb_to_linear_f64, srgb_to_linear_fast};

        // Sweep every f32 in the power segment [threshold, 1.0]
        let threshold = 0.039_293_37_f32; // SRGB_LINEAR_THRESHOLD

        // Full range
        let (max_ulp, avg_ulp, worst, count) =
            exhaustive_ulp_sweep(srgb_to_linear_fast, srgb_to_linear_f64, threshold, 1.0);
        let f64_ref = srgb_to_linear_f64(worst as f64);
        let fast_val = srgb_to_linear_fast(worst);
        println!("srgb_to_linear_fast exhaustive ({count} values):");
        println!("  Max ULP: {max_ulp} at input {worst:.10}");
        println!("  f64 ref: {f64_ref:.12}, fast: {fast_val:.12}");
        println!("  Avg ULP: {avg_ulp:.4}");

        // Away from threshold: [0.05, 1.0]
        let (max_ulp_mid, avg_ulp_mid, worst_mid, count_mid) =
            exhaustive_ulp_sweep(srgb_to_linear_fast, srgb_to_linear_f64, 0.05, 1.0);
        println!(
            "  [0.05, 1.0] ({count_mid} values): max={max_ulp_mid}, avg={avg_ulp_mid:.4}, worst={worst_mid:.10}"
        );

        // Upper half: [0.5, 1.0]
        let (max_ulp_hi, avg_ulp_hi, _, count_hi) =
            exhaustive_ulp_sweep(srgb_to_linear_fast, srgb_to_linear_f64, 0.5, 1.0);
        println!("  [0.5, 1.0] ({count_hi} values): max={max_ulp_hi}, avg={avg_ulp_hi:.4}");

        // Measured: max 303 ULP at threshold, ~28 avg
        assert!(
            max_ulp <= 310,
            "srgb_to_linear_fast max ULP {max_ulp} exceeds 310"
        );
    }

    #[test]
    fn test_linear_to_srgb_fast_exhaustive_ulp() {
        use crate::scalar::{linear_to_srgb_f64, linear_to_srgb_fast};

        // Sweep every f32 in the power segment [threshold, 1.0]
        let threshold = 0.003_041_282_6_f32; // LINEAR_THRESHOLD

        // Full range
        let (max_ulp, avg_ulp, worst, count) =
            exhaustive_ulp_sweep(linear_to_srgb_fast, linear_to_srgb_f64, threshold, 1.0);
        let f64_ref = linear_to_srgb_f64(worst as f64);
        let fast_val = linear_to_srgb_fast(worst);
        println!("linear_to_srgb_fast exhaustive ({count} values):");
        println!("  Max ULP: {max_ulp} at input {worst:.10}");
        println!("  f64 ref: {f64_ref:.12}, fast: {fast_val:.12}");
        println!("  Avg ULP: {avg_ulp:.4}");

        // Away from threshold: [0.01, 1.0]
        let (max_ulp_mid, avg_ulp_mid, worst_mid, count_mid) =
            exhaustive_ulp_sweep(linear_to_srgb_fast, linear_to_srgb_f64, 0.01, 1.0);
        println!(
            "  [0.01, 1.0] ({count_mid} values): max={max_ulp_mid}, avg={avg_ulp_mid:.4}, worst={worst_mid:.10}"
        );

        // Upper half: [0.5, 1.0]
        let (max_ulp_hi, avg_ulp_hi, _, count_hi) =
            exhaustive_ulp_sweep(linear_to_srgb_fast, linear_to_srgb_f64, 0.5, 1.0);
        println!("  [0.5, 1.0] ({count_hi} values): max={max_ulp_hi}, avg={avg_ulp_hi:.4}");

        // Measured: max 300 ULP at threshold, ~31 avg
        assert!(
            max_ulp <= 310,
            "linear_to_srgb_fast max ULP {max_ulp} exceeds 310"
        );
    }

    #[test]
    fn test_constant_differences() {
        // Document the difference between naive textbook constants and
        // the more precise C0-continuous constants we use (from moxcms)

        // Naive uses 0.04045 for sRGB threshold
        // We use 12.92 * 0.0030412825601275209 ≈ 0.0393 for C0 continuity
        let naive_threshold = 0.04045f64;
        let optimized_threshold = 12.92 * 0.003_041_282_560_127_521;
        let threshold_diff = (naive_threshold - optimized_threshold).abs();

        println!("Linearization threshold comparison:");
        println!("  Naive textbook:     {:.10}", naive_threshold);
        println!("  C0-continuous:      {:.10}", optimized_threshold);
        println!("  Difference:         {:.2e}", threshold_diff);

        // The difference is about 1.2e-3 (not 2.6e-6 as I mistakenly wrote)
        // This is because the naive constants are rounded approximations
        assert!(threshold_diff < 2e-3);
    }
}
