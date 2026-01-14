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

    #[allow(dead_code)]
    fn avg_abs_error(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum_abs_error / self.count as f64 }
    }

    fn avg_ulp(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum_ulp as f64 / self.count as f64 }
    }
}

fn print_table_header() {
    println!(
        "{:<55} {:>10} {:>10} {:>12}",
        "Implementation", "Max ULP", "Avg ULP", "Max Abs Err"
    );
    println!("{}", "-".repeat(89));
}

fn print_stats(stats: &ErrorStats) {
    println!(
        "{:<55} {:>10} {:>10.2} {:>12.2e}",
        stats.name,
        stats.max_ulp,
        stats.avg_ulp(),
        stats.max_abs_error,
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

fn create_moxcms_srgb_to_linear_transform(
    intent: RenderingIntent,
    use_cicp: bool,
) -> std::sync::Arc<moxcms::TransformF32Executor> {
    let srgb = ColorProfile::new_srgb();
    let linear_srgb = ColorProfile::new_from_cicp(CicpProfile {
        color_primaries: CicpColorPrimaries::Bt709,
        transfer_characteristics: TransferCharacteristics::Linear,
        matrix_coefficients: MatrixCoefficients::Identity,
        full_range: true,
    });
    let options = TransformOptions {
        rendering_intent: intent,
        allow_use_cicp_transfer: use_cicp,
        prefer_fixed_point: false,
        ..Default::default()
    };
    srgb.create_transform_f32(Layout::Rgb, &linear_srgb, Layout::Rgb, options)
        .expect("Failed to create moxcms sRGB→linear transform")
}

fn create_moxcms_linear_to_srgb_transform(
    intent: RenderingIntent,
    use_cicp: bool,
) -> std::sync::Arc<moxcms::TransformF32Executor> {
    let linear_srgb = ColorProfile::new_from_cicp(CicpProfile {
        color_primaries: CicpColorPrimaries::Bt709,
        transfer_characteristics: TransferCharacteristics::Linear,
        matrix_coefficients: MatrixCoefficients::Identity,
        full_range: true,
    });
    let srgb = ColorProfile::new_srgb();
    let options = TransformOptions {
        rendering_intent: intent,
        allow_use_cicp_transfer: use_cicp,
        prefer_fixed_point: false,
        ..Default::default()
    };
    linear_srgb
        .create_transform_f32(Layout::Rgb, &srgb, Layout::Rgb, options)
        .expect("Failed to create moxcms linear→sRGB transform")
}

fn compare_srgb_to_linear_f32() {
    println!("--- sRGB → Linear (f32 input → f32 output) ---\n");
    println!("(Testing range 0.01-1.0 to exclude near-zero ULP explosion)\n");

    let test_values: Vec<f64> = (100..=10000).map(|i| i as f64 / 10000.0).collect();

    let mut stats: Vec<ErrorStats> = vec![
        ErrorStats::new("f32→f32: Scalar powf (optimized constants)"),
        ErrorStats::new("f32→f32: SIMD dirty_pow approx"),
        ErrorStats::new("f32→f32: LUT 12-bit interp"),
        ErrorStats::new("f32→f32: moxcms (default)"),
        ErrorStats::new("f32→f32: moxcms (allow_use_cicp_transfer)"),
        ErrorStats::new("f32→f32: Naive powf (textbook constants)"),
    ];

    let lut12 = linear_srgb::lut::LinearTable12::new();
    let moxcms_default = create_moxcms_srgb_to_linear_transform(RenderingIntent::Perceptual, false);
    let moxcms_cicp = create_moxcms_srgb_to_linear_transform(RenderingIntent::Perceptual, true);

    for &input in &test_values {
        let reference = srgb_to_linear_f64(input);
        let input_f32 = input as f32;

        // Scalar f32
        let scalar_result = srgb_to_linear(input_f32) as f64;
        stats[0].update(reference, scalar_result);

        // SIMD f32x8
        let v = f32x8::splat(input_f32);
        let simd_result: [f32; 8] = simd::srgb_to_linear_x8(v).into();
        stats[1].update(reference, simd_result[0] as f64);

        // LUT 12-bit interpolated
        let lut_result = lut_interp_linear_float(input_f32, lut12.as_slice()) as f64;
        stats[2].update(reference, lut_result);

        // moxcms default
        let rgb_in = [input_f32, input_f32, input_f32];
        let mut rgb_out = [0.0f32; 3];
        let _ = moxcms_default.transform(&rgb_in, &mut rgb_out);
        stats[3].update(reference, rgb_out[0] as f64);

        // moxcms with allow_use_cicp_transfer
        let mut rgb_out2 = [0.0f32; 3];
        let _ = moxcms_cicp.transform(&rgb_in, &mut rgb_out2);
        stats[4].update(reference, rgb_out2[0] as f64);

        // Naive f32
        let naive_result = naive_srgb_to_linear_f64(input);
        stats[5].update(reference, naive_result);
    }

    print_table_header();
    for s in &stats {
        print_stats(s);
    }
}

fn compare_linear_to_srgb_f32() {
    println!("--- Linear → sRGB (f32 input → f32 output) ---\n");

    let test_values: Vec<f64> = (0..=10000).map(|i| i as f64 / 10000.0).collect();

    let mut stats: Vec<ErrorStats> = vec![
        ErrorStats::new("f32→f32: Scalar powf (optimized constants)"),
        ErrorStats::new("f32→f32: SIMD dirty_pow approx"),
        ErrorStats::new("f32→f32: LUT 12-bit interp"),
        ErrorStats::new("f32→f32: moxcms (default)"),
        ErrorStats::new("f32→f32: moxcms (allow_use_cicp_transfer)"),
        ErrorStats::new("f32→f32: Naive powf (textbook constants)"),
    ];

    let encode_table = EncodeTable12::new();
    let moxcms_default = create_moxcms_linear_to_srgb_transform(RenderingIntent::Perceptual, false);
    let moxcms_cicp = create_moxcms_linear_to_srgb_transform(RenderingIntent::Perceptual, true);

    for &input in &test_values {
        let reference = linear_to_srgb_f64(input);
        let input_f32 = input as f32;

        // Scalar f32
        let scalar_result = linear_to_srgb(input_f32) as f64;
        stats[0].update(reference, scalar_result);

        // SIMD f32x8
        let v = f32x8::splat(input_f32);
        let simd_result: [f32; 8] = simd::linear_to_srgb_x8(v).into();
        stats[1].update(reference, simd_result[0] as f64);

        // LUT 12-bit
        let lut_result = lut_interp_linear_float(input_f32, encode_table.as_slice()) as f64;
        stats[2].update(reference, lut_result);

        // moxcms default
        let rgb_in = [input_f32, input_f32, input_f32];
        let mut rgb_out = [0.0f32; 3];
        let _ = moxcms_default.transform(&rgb_in, &mut rgb_out);
        stats[3].update(reference, rgb_out[0] as f64);

        // moxcms with allow_use_cicp_transfer
        let mut rgb_out2 = [0.0f32; 3];
        let _ = moxcms_cicp.transform(&rgb_in, &mut rgb_out2);
        stats[4].update(reference, rgb_out2[0] as f64);

        // Naive f32
        let naive_result = naive_linear_to_srgb_f64(input);
        stats[5].update(reference, naive_result);
    }

    print_table_header();
    for s in &stats {
        print_stats(s);
    }
}

fn compare_u8_to_linear() {
    println!("--- u8 sRGB → f32 Linear ---\n");

    let lut8 = LinearTable8::new();
    let converter = SrgbConverter::new();

    let mut stats: Vec<ErrorStats> = vec![
        ErrorStats::new("u8→f32: LUT 8-bit direct lookup"),
        ErrorStats::new("u8→f32: SIMD batch (8x LUT lookup)"),
        ErrorStats::new("u8→f32: SrgbConverter (LUT lookup)"),
        ErrorStats::new("u8→f32: u8/255→f32, then scalar powf"),
    ];

    for i in 0..=255u8 {
        let input_normalized = i as f64 / 255.0;
        let reference = srgb_to_linear_f64(input_normalized);

        // LUT 8-bit direct
        let lut_result = lut8.lookup(i as usize) as f64;
        stats[0].update(reference, lut_result);

        // SIMD batch
        let input_arr: [u8; 8] = [i; 8];
        let simd_result = simd::srgb_u8_to_linear_x8(&lut8, input_arr);
        let simd_arr: [f32; 8] = simd_result.into();
        stats[1].update(reference, simd_arr[0] as f64);

        // SrgbConverter
        let conv_result = converter.srgb_u8_to_linear(i) as f64;
        stats[2].update(reference, conv_result);

        // f32 scalar (u8/255 then powf)
        let f32_input = i as f32 / 255.0;
        let f32_result = srgb_to_linear(f32_input) as f64;
        stats[3].update(reference, f32_result);
    }

    print_table_header();
    for s in &stats {
        print_stats(s);
    }
}

fn compare_linear_to_u8() {
    println!("--- f32 Linear → u8 sRGB ---\n");

    let converter = SrgbConverter::new();

    println!(
        "{:<55} {:>10} {:>10} {:>12}",
        "Implementation", "Max Diff", "Avg Diff", "Exact Match"
    );
    println!("{}", "-".repeat(89));

    struct U8Stats {
        name: String,
        exact: i32,
        max_diff: i32,
        sum_diff: i32,
    }

    let mut stats = vec![
        U8Stats { name: "f32→u8: SIMD dirty_pow, then *255+0.5→u8".into(), exact: 0, max_diff: 0, sum_diff: 0 },
        U8Stats { name: "f32→u8: LUT 12-bit interp, then *255+0.5→u8".into(), exact: 0, max_diff: 0, sum_diff: 0 },
        U8Stats { name: "f32→u8: Scalar powf, then *255+0.5→u8".into(), exact: 0, max_diff: 0, sum_diff: 0 },
    ];

    for i in 0..=255u8 {
        let srgb_normalized = i as f64 / 255.0;
        let linear_input = srgb_to_linear_f64(srgb_normalized);
        let linear_f32 = linear_input as f32;

        // SIMD dirty pow + round
        let v = f32x8::splat(linear_f32);
        let simd_result = simd::linear_to_srgb_u8_x8(v)[0];

        // LUT interp + round
        let conv_result = converter.linear_to_srgb_u8(linear_f32);

        // Scalar powf + round
        let scalar_srgb = linear_to_srgb(linear_f32);
        let scalar_result = (scalar_srgb * 255.0 + 0.5) as u8;

        let results = [simd_result, conv_result, scalar_result];
        for (idx, &result) in results.iter().enumerate() {
            let diff = (result as i32 - i as i32).abs();
            if diff == 0 { stats[idx].exact += 1; }
            stats[idx].max_diff = stats[idx].max_diff.max(diff);
            stats[idx].sum_diff += diff;
        }
    }

    for s in &stats {
        println!(
            "{:<55} {:>10} {:>10.2} {:>8}/256",
            s.name, s.max_diff, s.sum_diff as f64 / 256.0, s.exact
        );
    }
}

fn compare_u8_roundtrip() {
    println!("\n--- u8 Roundtrip: sRGB u8 → f32 Linear → sRGB u8 ---\n");

    let lut = LinearTable8::new();
    let converter = SrgbConverter::new();

    println!(
        "{:<55} {:>10} {:>10} {:>12}",
        "Implementation", "Max Diff", "Exact", "Off-by-1"
    );
    println!("{}", "-".repeat(89));

    struct RoundtripStats {
        name: String,
        exact: i32,
        off1: i32,
        max_diff: i32,
    }

    let mut stats = vec![
        RoundtripStats { name: "u8→LUT→f32→SIMD dirty_pow→*255+0.5→u8".into(), exact: 0, off1: 0, max_diff: 0 },
        RoundtripStats { name: "u8→LUT→f32→scalar powf→*255+0.5→u8".into(), exact: 0, off1: 0, max_diff: 0 },
        RoundtripStats { name: "u8→LUT→f32→LUT interp→*255+0.5→u8".into(), exact: 0, off1: 0, max_diff: 0 },
    ];

    for i in 0..=255u8 {
        let linear = lut.lookup(i as usize);

        // SIMD path
        let v = f32x8::splat(linear);
        let simd_back = simd::linear_to_srgb_u8_x8(v)[0];

        // Scalar path
        let scalar_srgb = linear_to_srgb(linear);
        let scalar_back = (scalar_srgb * 255.0 + 0.5) as u8;

        // LUT path
        let lut_linear = converter.srgb_u8_to_linear(i);
        let lut_back = converter.linear_to_srgb_u8(lut_linear);

        let results = [simd_back, scalar_back, lut_back];
        for (idx, &result) in results.iter().enumerate() {
            let diff = (i as i32 - result as i32).abs();
            stats[idx].max_diff = stats[idx].max_diff.max(diff);
            if diff == 0 { stats[idx].exact += 1; }
            else if diff == 1 { stats[idx].off1 += 1; }
        }
    }

    for s in &stats {
        println!(
            "{:<55} {:>10} {:>6}/256 {:>8}/256",
            s.name, s.max_diff, s.exact, s.off1
        );
    }

    println!("\nNote: Off-by-1 errors are acceptable for u8 precision.");
}
