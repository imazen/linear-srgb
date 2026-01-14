//! Comprehensive accuracy comparison of all sRGB conversion implementations.
//!
//! Run with: cargo run --release --example accuracy_comparison

use linear_srgb::accuracy::{naive_linear_to_srgb_f64, naive_srgb_to_linear_f64, ulp_distance_f32};
use linear_srgb::lut::{EncodeTable12, LinearTable8, SrgbConverter, lut_interp_linear_float};
use linear_srgb::transfer::{linear_to_srgb_f64, srgb_to_linear_f64};
use linear_srgb::{linear_to_srgb, simd, srgb_to_linear};
use wide::f32x8;

/// Statistics for error analysis
#[derive(Debug, Clone, Default)]
struct ErrorStats {
    name: String,
    max_abs_error: f64,
    max_rel_error: f64,
    avg_abs_error: f64,
    max_ulp: u32,
    avg_ulp: f64,
    count: u64,
    worst_input: f64,
    worst_expected: f64,
    worst_actual: f64,
}

impl ErrorStats {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn update(&mut self, input: f64, expected: f64, actual: f64) {
        let abs_error = (expected - actual).abs();
        let rel_error = if expected.abs() > 1e-10 {
            abs_error / expected.abs()
        } else {
            abs_error
        };
        let ulp = ulp_distance_f32(expected as f32, actual as f32);

        self.avg_abs_error = (self.avg_abs_error * self.count as f64 + abs_error)
            / (self.count as f64 + 1.0);
        self.avg_ulp =
            (self.avg_ulp * self.count as f64 + ulp as f64) / (self.count as f64 + 1.0);
        self.count += 1;

        if abs_error > self.max_abs_error {
            self.max_abs_error = abs_error;
            self.worst_input = input;
            self.worst_expected = expected;
            self.worst_actual = actual;
        }
        if rel_error > self.max_rel_error {
            self.max_rel_error = rel_error;
        }
        if ulp > self.max_ulp {
            self.max_ulp = ulp;
        }
    }

    fn print(&self) {
        println!("  {}", self.name);
        println!("    Max Abs Error: {:.2e}", self.max_abs_error);
        println!("    Max Rel Error: {:.2e}", self.max_rel_error);
        println!("    Avg Abs Error: {:.2e}", self.avg_abs_error);
        println!("    Max ULP:       {}", self.max_ulp);
        println!("    Avg ULP:       {:.2}", self.avg_ulp);
        if self.max_abs_error > 0.0 {
            println!(
                "    Worst case:    input={:.6}, expected={:.10}, actual={:.10}",
                self.worst_input, self.worst_expected, self.worst_actual
            );
        }
    }
}

fn main() {
    println!("=== sRGB Conversion Accuracy Comparison ===\n");
    println!("Reference: f64 implementation with precise constants\n");

    compare_srgb_to_linear();
    println!();
    compare_linear_to_srgb();
    println!();
    compare_u8_roundtrip();
}

fn compare_srgb_to_linear() {
    println!("--- sRGB → Linear Conversions ---\n");

    let lut8 = LinearTable8::new();

    // Test across all 8-bit values and interpolated points
    let test_values: Vec<f64> = (0..=10000).map(|i| i as f64 / 10000.0).collect();

    // 1. Scalar f32 (optimized constants)
    let mut stats_scalar = ErrorStats::new("Scalar f32 (optimized constants)");

    // 2. SIMD f32x8 (dirty pow)
    let mut stats_simd = ErrorStats::new("SIMD f32x8 (dirty pow approximation)");

    // 3. Naive f32 (textbook constants)
    let mut stats_naive = ErrorStats::new("Naive f32 (textbook powf)");

    // 4. LUT 8-bit (exact for u8 inputs)
    let mut stats_lut8 = ErrorStats::new("LUT 8-bit (256 entries)");

    for &input in &test_values {
        let reference = srgb_to_linear_f64(input);

        // Scalar f32
        let scalar_result = srgb_to_linear(input as f32) as f64;
        stats_scalar.update(input, reference, scalar_result);

        // SIMD f32x8
        let v = f32x8::splat(input as f32);
        let simd_result: [f32; 8] = simd::srgb_to_linear_x8(v).into();
        stats_simd.update(input, reference, simd_result[0] as f64);

        // Naive f32
        let naive_result = naive_srgb_to_linear_f64(input);
        stats_naive.update(input, reference, naive_result);

        // LUT 8-bit - interpolate between entries
        let u8_val = (input * 255.0).round() as usize;
        if u8_val <= 255 {
            let lut_result = lut8.lookup(u8_val) as f64;
            // Compare against f64 result for exact u8 input
            let exact_input = u8_val as f64 / 255.0;
            let exact_ref = srgb_to_linear_f64(exact_input);
            stats_lut8.update(exact_input, exact_ref, lut_result);
        }
    }

    stats_scalar.print();
    println!();
    stats_simd.print();
    println!();
    stats_naive.print();
    println!();
    stats_lut8.print();
}

fn compare_linear_to_srgb() {
    println!("--- Linear → sRGB Conversions ---\n");

    let encode_table = EncodeTable12::new();
    let converter = SrgbConverter::new();

    // Test across the full range
    let test_values: Vec<f64> = (0..=10000).map(|i| i as f64 / 10000.0).collect();

    // 1. Scalar f32 (optimized constants)
    let mut stats_scalar = ErrorStats::new("Scalar f32 (optimized constants)");

    // 2. SIMD f32x8 (dirty pow)
    let mut stats_simd = ErrorStats::new("SIMD f32x8 (dirty pow approximation)");

    // 3. Naive f32 (textbook constants)
    let mut stats_naive = ErrorStats::new("Naive f32 (textbook powf)");

    // 4. LUT 12-bit with interpolation
    let mut stats_lut12 = ErrorStats::new("LUT 12-bit (4096 entries, interpolated)");

    // 5. Converter (LUT-based)
    let mut stats_converter = ErrorStats::new("SrgbConverter (12-bit LUT interpolated)");

    for &input in &test_values {
        let reference = linear_to_srgb_f64(input);

        // Scalar f32
        let scalar_result = linear_to_srgb(input as f32) as f64;
        stats_scalar.update(input, reference, scalar_result);

        // SIMD f32x8
        let v = f32x8::splat(input as f32);
        let simd_result: [f32; 8] = simd::linear_to_srgb_x8(v).into();
        stats_simd.update(input, reference, simd_result[0] as f64);

        // Naive f32
        let naive_result = naive_linear_to_srgb_f64(input);
        stats_naive.update(input, reference, naive_result);

        // LUT 12-bit
        let lut_result = lut_interp_linear_float(input as f32, encode_table.as_slice()) as f64;
        stats_lut12.update(input, reference, lut_result);

        // Converter
        let conv_result = converter.linear_to_srgb(input as f32) as f64;
        stats_converter.update(input, reference, conv_result);
    }

    stats_scalar.print();
    println!();
    stats_simd.print();
    println!();
    stats_naive.print();
    println!();
    stats_lut12.print();
    println!();
    stats_converter.print();
}

fn compare_u8_roundtrip() {
    println!("--- u8 Roundtrip Accuracy ---\n");

    let lut = LinearTable8::new();

    // Test all 256 u8 values
    println!("  u8 sRGB → f32 Linear → u8 sRGB roundtrip:");

    let mut exact_matches = 0;
    let mut off_by_one = 0;
    let mut off_by_more = 0;
    let mut max_diff = 0i32;

    for i in 0..=255u8 {
        // u8 → f32 linear (LUT)
        let linear = lut.lookup(i as usize);

        // f32 linear → u8 sRGB (SIMD batch with scalar fallback)
        let v = f32x8::splat(linear);
        let srgb_arr = simd::linear_to_srgb_u8_x8(v);
        let back = srgb_arr[0];

        let diff = (i as i32 - back as i32).abs();
        max_diff = max_diff.max(diff);

        if diff == 0 {
            exact_matches += 1;
        } else if diff == 1 {
            off_by_one += 1;
        } else {
            off_by_more += 1;
            println!("    {} → {} → {} (diff={})", i, linear, back, diff);
        }
    }

    println!("    Exact matches: {}/256 ({:.1}%)", exact_matches, exact_matches as f64 / 256.0 * 100.0);
    println!("    Off by 1:      {}/256 ({:.1}%)", off_by_one, off_by_one as f64 / 256.0 * 100.0);
    println!("    Off by >1:     {}/256", off_by_more);
    println!("    Max diff:      {}", max_diff);

    println!();

    // Compare SIMD u8 conversion against scalar
    println!("  SIMD u8 vs Scalar u8 conversion accuracy:");

    let mut simd_vs_scalar_exact = 0;
    let mut simd_vs_scalar_diff1 = 0;
    let mut simd_vs_scalar_diff_more = 0;

    for i in 0..=255u8 {
        let linear = lut.lookup(i as usize);

        // SIMD path
        let v = f32x8::splat(linear);
        let simd_result = simd::linear_to_srgb_u8_x8(v)[0];

        // Scalar path
        let scalar_srgb = linear_to_srgb(linear);
        let scalar_result = (scalar_srgb * 255.0 + 0.5) as u8;

        let diff = (simd_result as i32 - scalar_result as i32).abs();
        if diff == 0 {
            simd_vs_scalar_exact += 1;
        } else if diff == 1 {
            simd_vs_scalar_diff1 += 1;
        } else {
            simd_vs_scalar_diff_more += 1;
            println!(
                "    input={}, linear={:.6}, SIMD={}, scalar={}",
                i, linear, simd_result, scalar_result
            );
        }
    }

    println!("    Exact matches: {}/256 ({:.1}%)", simd_vs_scalar_exact, simd_vs_scalar_exact as f64 / 256.0 * 100.0);
    println!("    Off by 1:      {}/256", simd_vs_scalar_diff1);
    println!("    Off by >1:     {}/256", simd_vs_scalar_diff_more);
}
