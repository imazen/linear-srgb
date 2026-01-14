//! Comprehensive accuracy comparison of all sRGB conversion implementations.
//!
//! Run with: cargo run --release --example accuracy_comparison

use linear_srgb::accuracy::{naive_linear_to_srgb_f64, naive_srgb_to_linear_f64, ulp_distance_f32};
use linear_srgb::lut::{EncodeTable12, LinearTable8, SrgbConverter, lut_interp_linear_float};
use linear_srgb::transfer::{linear_to_srgb_f64, srgb_to_linear_f64};
use linear_srgb::{linear_to_srgb, simd, srgb_to_linear};
use moxcms::{
    CicpColorPrimaries, CicpProfile, ColorProfile, Layout, MatrixCoefficients, RenderingIntent,
    TransferCharacteristics, TransformOptions,
};
use wide::f32x8;

/// Statistics for error analysis
#[derive(Debug, Clone)]
struct ErrorStats {
    name: String,
    max_abs_error: f64,
    sum_abs_error: f64,
    max_ulp: u32,
    sum_ulp: u64,
    count: u64,
}

impl ErrorStats {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            max_abs_error: 0.0,
            sum_abs_error: 0.0,
            max_ulp: 0,
            sum_ulp: 0,
            count: 0,
        }
    }

    fn update(&mut self, expected: f64, actual: f64) {
        let abs_error = (expected - actual).abs();
        let ulp = ulp_distance_f32(expected as f32, actual as f32);

        self.sum_abs_error += abs_error;
        self.sum_ulp += ulp as u64;
        self.count += 1;

        if abs_error > self.max_abs_error {
            self.max_abs_error = abs_error;
        }
        if ulp > self.max_ulp {
            self.max_ulp = ulp;
        }
    }

    fn avg_abs_error(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum_abs_error / self.count as f64 }
    }

    fn avg_ulp(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum_ulp as f64 / self.count as f64 }
    }
}

fn print_table_header() {
    println!(
        "{:<40} {:>10} {:>10} {:>12} {:>12}",
        "Implementation", "Max ULP", "Avg ULP", "Max Abs Err", "Avg Abs Err"
    );
    println!("{}", "-".repeat(86));
}

fn print_stats(stats: &ErrorStats) {
    println!(
        "{:<40} {:>10} {:>10.2} {:>12.2e} {:>12.2e}",
        stats.name,
        stats.max_ulp,
        stats.avg_ulp(),
        stats.max_abs_error,
        stats.avg_abs_error()
    );
}

fn main() {
    println!("=== sRGB Conversion Accuracy Comparison ===");
    println!("Reference: f64 implementation with precise IEC 61966-2-1 constants\n");

    compare_srgb_to_linear_f32();
    println!();
    compare_linear_to_srgb_f32();
    println!();
    compare_u8_to_linear();
    println!();
    compare_linear_to_u8();
    println!();
    compare_u8_roundtrip();
}

fn create_moxcms_srgb_to_linear_transform() -> std::sync::Arc<moxcms::TransformF32Executor> {
    let srgb = ColorProfile::new_srgb();
    let linear_srgb = ColorProfile::new_from_cicp(CicpProfile {
        color_primaries: CicpColorPrimaries::Bt709,
        transfer_characteristics: TransferCharacteristics::Linear,
        matrix_coefficients: MatrixCoefficients::Identity,
        full_range: true,
    });
    let options = TransformOptions {
        rendering_intent: RenderingIntent::Perceptual,
        ..Default::default()
    };
    srgb.create_transform_f32(Layout::Rgb, &linear_srgb, Layout::Rgb, options)
        .expect("Failed to create moxcms sRGB→linear transform")
}

fn create_moxcms_linear_to_srgb_transform() -> std::sync::Arc<moxcms::TransformF32Executor> {
    let linear_srgb = ColorProfile::new_from_cicp(CicpProfile {
        color_primaries: CicpColorPrimaries::Bt709,
        transfer_characteristics: TransferCharacteristics::Linear,
        matrix_coefficients: MatrixCoefficients::Identity,
        full_range: true,
    });
    let srgb = ColorProfile::new_srgb();
    let options = TransformOptions {
        rendering_intent: RenderingIntent::Perceptual,
        ..Default::default()
    };
    linear_srgb
        .create_transform_f32(Layout::Rgb, &srgb, Layout::Rgb, options)
        .expect("Failed to create moxcms linear→sRGB transform")
}

fn compare_srgb_to_linear_f32() {
    println!("--- sRGB → Linear (f32 → f32) ---\n");

    // Exclude very small values (< 0.01) where near-zero clipping causes ULP explosion
    let test_values: Vec<f64> = (100..=10000).map(|i| i as f64 / 10000.0).collect();

    let mut stats_scalar = ErrorStats::new("Scalar f32 (optimized constants)");
    let mut stats_simd = ErrorStats::new("SIMD f32x8 (dirty pow)");
    let mut stats_naive = ErrorStats::new("Naive f32 (textbook powf)");
    let mut stats_lut12 = ErrorStats::new("LUT 12-bit interpolated");
    let mut stats_moxcms = ErrorStats::new("moxcms");

    let lut12 = linear_srgb::lut::LinearTable12::new();
    let moxcms_transform = create_moxcms_srgb_to_linear_transform();

    for &input in &test_values {
        let reference = srgb_to_linear_f64(input);
        let input_f32 = input as f32;

        // Scalar f32
        let scalar_result = srgb_to_linear(input_f32) as f64;
        stats_scalar.update(reference, scalar_result);

        // SIMD f32x8
        let v = f32x8::splat(input_f32);
        let simd_result: [f32; 8] = simd::srgb_to_linear_x8(v).into();
        stats_simd.update(reference, simd_result[0] as f64);

        // Naive f32
        let naive_result = naive_srgb_to_linear_f64(input);
        stats_naive.update(reference, naive_result);

        // LUT 12-bit interpolated
        let lut_result = lut_interp_linear_float(input_f32, lut12.as_slice()) as f64;
        stats_lut12.update(reference, lut_result);

        // moxcms (process as RGB triple)
        let rgb_in = [input_f32, input_f32, input_f32];
        let mut rgb_out = [0.0f32; 3];
        let _ = moxcms_transform.transform(&rgb_in, &mut rgb_out);
        stats_moxcms.update(reference, rgb_out[0] as f64);
    }

    println!("(Testing range 0.01-1.0 to exclude near-zero ULP explosion)\n");
    print_table_header();
    print_stats(&stats_scalar);
    print_stats(&stats_simd);
    print_stats(&stats_lut12);
    print_stats(&stats_moxcms);
    print_stats(&stats_naive);

    println!("\nNote: moxcms clips near-zero values, causing huge ULP at very small inputs.");
}

fn compare_linear_to_srgb_f32() {
    println!("--- Linear → sRGB (f32 → f32) ---\n");

    // Full range test
    let test_values: Vec<f64> = (0..=10000).map(|i| i as f64 / 10000.0).collect();

    let mut stats_scalar = ErrorStats::new("Scalar f32 (optimized constants)");
    let mut stats_simd = ErrorStats::new("SIMD f32x8 (dirty pow)");
    let mut stats_naive = ErrorStats::new("Naive f32 (textbook powf)");
    let mut stats_lut12 = ErrorStats::new("LUT 12-bit interpolated");
    let mut stats_moxcms = ErrorStats::new("moxcms");

    let encode_table = EncodeTable12::new();
    let moxcms_transform = create_moxcms_linear_to_srgb_transform();

    for &input in &test_values {
        let reference = linear_to_srgb_f64(input);
        let input_f32 = input as f32;

        // Scalar f32
        let scalar_result = linear_to_srgb(input_f32) as f64;
        stats_scalar.update(reference, scalar_result);

        // SIMD f32x8
        let v = f32x8::splat(input_f32);
        let simd_result: [f32; 8] = simd::linear_to_srgb_x8(v).into();
        stats_simd.update(reference, simd_result[0] as f64);

        // Naive f32
        let naive_result = naive_linear_to_srgb_f64(input);
        stats_naive.update(reference, naive_result);

        // LUT 12-bit
        let lut_result = lut_interp_linear_float(input_f32, encode_table.as_slice()) as f64;
        stats_lut12.update(reference, lut_result);

        // moxcms
        let rgb_in = [input_f32, input_f32, input_f32];
        let mut rgb_out = [0.0f32; 3];
        let _ = moxcms_transform.transform(&rgb_in, &mut rgb_out);
        stats_moxcms.update(reference, rgb_out[0] as f64);
    }

    print_table_header();
    print_stats(&stats_scalar);
    print_stats(&stats_simd);
    print_stats(&stats_lut12);
    print_stats(&stats_moxcms);
    print_stats(&stats_naive);
}

fn compare_u8_to_linear() {
    println!("--- u8 sRGB → f32 Linear ---\n");

    let lut8 = LinearTable8::new();
    let converter = SrgbConverter::new();

    let mut stats_lut8 = ErrorStats::new("LUT 8-bit (direct lookup)");
    let mut stats_simd_batch = ErrorStats::new("SIMD batch (LUT-based)");
    let mut stats_converter = ErrorStats::new("SrgbConverter (LUT-based)");
    let mut stats_f32_clamp = ErrorStats::new("f32 scalar (from u8/255)");

    // Test all 256 u8 values
    for i in 0..=255u8 {
        let input_normalized = i as f64 / 255.0;
        let reference = srgb_to_linear_f64(input_normalized);

        // LUT 8-bit direct
        let lut_result = lut8.lookup(i as usize) as f64;
        stats_lut8.update(reference, lut_result);

        // SIMD batch (same as LUT for u8 input)
        let input_arr: [u8; 8] = [i; 8];
        let simd_result = simd::srgb_u8_to_linear_x8(&lut8, input_arr);
        let simd_arr: [f32; 8] = simd_result.into();
        stats_simd_batch.update(reference, simd_arr[0] as f64);

        // SrgbConverter
        let conv_result = converter.srgb_u8_to_linear(i) as f64;
        stats_converter.update(reference, conv_result);

        // f32 scalar with clamped input (simulating u8 precision)
        let f32_input = i as f32 / 255.0;
        let f32_result = srgb_to_linear(f32_input) as f64;
        stats_f32_clamp.update(reference, f32_result);
    }

    print_table_header();
    print_stats(&stats_lut8);
    print_stats(&stats_simd_batch);
    print_stats(&stats_converter);
    print_stats(&stats_f32_clamp);
}

fn compare_linear_to_u8() {
    println!("--- f32 Linear → u8 sRGB ---\n");

    let converter = SrgbConverter::new();

    println!(
        "{:<40} {:>10} {:>10} {:>12}",
        "Implementation", "Max Diff", "Avg Diff", "Exact Match"
    );
    println!("{}", "-".repeat(74));

    let mut simd_exact = 0;
    let mut conv_exact = 0;
    let mut scalar_exact = 0;
    let mut simd_max = 0i32;
    let mut conv_max = 0i32;
    let mut scalar_max = 0i32;
    let mut simd_sum = 0i32;
    let mut conv_sum = 0i32;
    let mut scalar_sum = 0i32;

    for i in 0..=255u8 {
        let srgb_normalized = i as f64 / 255.0;
        let linear_input = srgb_to_linear_f64(srgb_normalized);
        let linear_f32 = linear_input as f32;

        let v = f32x8::splat(linear_f32);
        let simd_result = simd::linear_to_srgb_u8_x8(v)[0];
        let conv_result = converter.linear_to_srgb_u8(linear_f32);
        let scalar_srgb = linear_to_srgb(linear_f32);
        let scalar_result = (scalar_srgb * 255.0 + 0.5) as u8;

        let simd_diff = (simd_result as i32 - i as i32).abs();
        let conv_diff = (conv_result as i32 - i as i32).abs();
        let scalar_diff = (scalar_result as i32 - i as i32).abs();

        if simd_diff == 0 { simd_exact += 1; }
        if conv_diff == 0 { conv_exact += 1; }
        if scalar_diff == 0 { scalar_exact += 1; }

        simd_max = simd_max.max(simd_diff);
        conv_max = conv_max.max(conv_diff);
        scalar_max = scalar_max.max(scalar_diff);

        simd_sum += simd_diff;
        conv_sum += conv_diff;
        scalar_sum += scalar_diff;
    }

    println!(
        "{:<40} {:>10} {:>10.2} {:>8}/256",
        "SIMD batch (dirty pow)", simd_max, simd_sum as f64 / 256.0, simd_exact
    );
    println!(
        "{:<40} {:>10} {:>10.2} {:>8}/256",
        "SrgbConverter (LUT interp)", conv_max, conv_sum as f64 / 256.0, conv_exact
    );
    println!(
        "{:<40} {:>10} {:>10.2} {:>8}/256",
        "Scalar f32 + round", scalar_max, scalar_sum as f64 / 256.0, scalar_exact
    );
}

fn compare_u8_roundtrip() {
    println!("\n--- u8 Roundtrip: sRGB → Linear → sRGB ---\n");

    let lut = LinearTable8::new();
    let converter = SrgbConverter::new();

    println!(
        "{:<40} {:>10} {:>10} {:>12}",
        "Implementation", "Max Diff", "Exact", "Off-by-1"
    );
    println!("{}", "-".repeat(74));

    let mut simd_exact = 0;
    let mut simd_off1 = 0;
    let mut simd_max = 0;

    let mut scalar_exact = 0;
    let mut scalar_off1 = 0;
    let mut scalar_max = 0;

    let mut lut_exact = 0;
    let mut lut_off1 = 0;
    let mut lut_max = 0;

    for i in 0..=255u8 {
        // SIMD: LUT lookup -> SIMD transfer -> u8
        let linear = lut.lookup(i as usize);
        let v = f32x8::splat(linear);
        let simd_back = simd::linear_to_srgb_u8_x8(v)[0];
        let simd_diff = (i as i32 - simd_back as i32).abs();
        simd_max = simd_max.max(simd_diff);
        if simd_diff == 0 { simd_exact += 1; }
        else if simd_diff == 1 { simd_off1 += 1; }

        // Scalar: LUT lookup -> scalar transfer -> u8
        let scalar_srgb = linear_to_srgb(linear);
        let scalar_back = (scalar_srgb * 255.0 + 0.5) as u8;
        let scalar_diff = (i as i32 - scalar_back as i32).abs();
        scalar_max = scalar_max.max(scalar_diff);
        if scalar_diff == 0 { scalar_exact += 1; }
        else if scalar_diff == 1 { scalar_off1 += 1; }

        // LUT: converter roundtrip
        let lut_linear = converter.srgb_u8_to_linear(i);
        let lut_back = converter.linear_to_srgb_u8(lut_linear);
        let lut_diff = (i as i32 - lut_back as i32).abs();
        lut_max = lut_max.max(lut_diff);
        if lut_diff == 0 { lut_exact += 1; }
        else if lut_diff == 1 { lut_off1 += 1; }
    }

    println!(
        "{:<40} {:>10} {:>6}/256 {:>8}/256",
        "SIMD (LUT → dirty pow → u8)", simd_max, simd_exact, simd_off1
    );
    println!(
        "{:<40} {:>10} {:>6}/256 {:>8}/256",
        "Scalar (LUT → f32 pow → u8)", scalar_max, scalar_exact, scalar_off1
    );
    println!(
        "{:<40} {:>10} {:>6}/256 {:>8}/256",
        "SrgbConverter (LUT → LUT interp)", lut_max, lut_exact, lut_off1
    );

    println!("\nNote: Off-by-1 errors are acceptable for u8 precision.");
}
