//! Comprehensive accuracy comparison of all sRGB conversion implementations.
//!
//! Run with: cargo run --release --example accuracy_comparison --features alt
//!
//! Tests all combinations of:
//! - 2 directions: sRGB→Linear, Linear→sRGB
//! - Input types: f32, u8, u16
//! - Output types: f32, u8, u16
//! - Multiple implementations for each direction

#![allow(deprecated)]

#[cfg(feature = "alt")]
use linear_srgb::alt::accuracy::{
    naive_linear_to_srgb_f64, naive_srgb_to_linear_f64, ulp_distance_f32,
};
#[cfg(feature = "alt")]
use linear_srgb::alt::imageflow;
use linear_srgb::lut::{
    EncodeTable12, EncodeTable16, LinearTable8, LinearTable16, SrgbConverter,
    lut_interp_linear_float,
};
use linear_srgb::scalar::{linear_to_srgb, linear_to_srgb_f64, srgb_to_linear, srgb_to_linear_f64};
use linear_srgb::simd;
use std::sync::Arc;
use wide::f32x8;

// ============================================================================
// Converter Abstractions
// ============================================================================

/// A method for converting sRGB → Linear
struct SrgbToLinearMethod {
    name: &'static str,
    /// f32 [0,1] → f32 [0,1]
    f32_to_f32: Box<dyn Fn(f32) -> f32 + Sync>,
    /// Optional specialized u8 → f32 (if None, uses f32/255 → f32_to_f32)
    u8_to_f32: Option<Box<dyn Fn(u8) -> f32 + Sync>>,
    /// Optional specialized u16 → f32 (if None, uses f32/65535 → f32_to_f32)
    u16_to_f32: Option<Box<dyn Fn(u16) -> f32 + Sync>>,
}

impl SrgbToLinearMethod {
    fn convert_f32(&self, x: f32) -> f32 {
        (self.f32_to_f32)(x)
    }

    fn convert_u8(&self, x: u8) -> f32 {
        match &self.u8_to_f32 {
            Some(f) => f(x),
            None => (self.f32_to_f32)(x as f32 / 255.0),
        }
    }

    fn convert_u16(&self, x: u16) -> f32 {
        match &self.u16_to_f32 {
            Some(f) => f(x),
            None => (self.f32_to_f32)(x as f32 / 65535.0),
        }
    }
}

/// A method for converting Linear → sRGB
struct LinearToSrgbMethod {
    name: &'static str,
    /// f32 [0,1] → f32 [0,1]
    f32_to_f32: Box<dyn Fn(f32) -> f32 + Sync>,
    /// Optional specialized f32 → u8 (if None, uses f32_to_f32 * 255 + 0.5)
    f32_to_u8: Option<Box<dyn Fn(f32) -> u8 + Sync>>,
    /// Optional specialized f32 → u16 (if None, uses f32_to_f32 * 65535 + 0.5)
    f32_to_u16: Option<Box<dyn Fn(f32) -> u16 + Sync>>,
}

impl LinearToSrgbMethod {
    fn convert_f32(&self, x: f32) -> f32 {
        (self.f32_to_f32)(x)
    }

    fn convert_to_u8(&self, x: f32) -> u8 {
        match &self.f32_to_u8 {
            Some(f) => f(x),
            None => ((self.f32_to_f32)(x) * 255.0 + 0.5) as u8,
        }
    }

    fn convert_to_u16(&self, x: f32) -> u16 {
        match &self.f32_to_u16 {
            Some(f) => f(x),
            None => ((self.f32_to_f32)(x) * 65535.0 + 0.5) as u16,
        }
    }
}

// ============================================================================
// Method Factories
// ============================================================================

#[cfg(feature = "alt")]
fn create_srgb_to_linear_methods() -> Vec<SrgbToLinearMethod> {
    // Pre-create shared resources
    let lut8 = Arc::new(LinearTable8::new());
    let lut16 = Arc::new(LinearTable16::new());
    let imageflow_lut = Arc::new(imageflow::SrgbToLinearLut::new());

    let lut8_clone = lut8.clone();
    let lut16_clone = lut16.clone();
    let imageflow_lut_clone = imageflow_lut.clone();

    vec![
        SrgbToLinearMethod {
            name: "Scalar powf (optimized constants)",
            f32_to_f32: Box::new(srgb_to_linear),
            u8_to_f32: None,
            u16_to_f32: None,
        },
        SrgbToLinearMethod {
            name: "SIMD wide::f32x8 dirty_pow",
            f32_to_f32: Box::new(|x| {
                let v = f32x8::splat(x);
                let result: [f32; 8] = simd::srgb_to_linear_x8(v).into();
                result[0]
            }),
            u8_to_f32: None,
            u16_to_f32: None,
        },
        SrgbToLinearMethod {
            name: "LUT-8 direct lookup",
            f32_to_f32: Box::new(move |x| lut8.lookup((x * 255.0 + 0.5) as usize)),
            u8_to_f32: Some(Box::new(move |x| lut8_clone.lookup(x as usize))),
            u16_to_f32: None,
        },
        SrgbToLinearMethod {
            name: "LUT-16 direct lookup",
            f32_to_f32: Box::new(move |x| lut16.lookup((x * 65535.0 + 0.5) as usize)),
            u8_to_f32: None,
            u16_to_f32: Some(Box::new(move |x| lut16_clone.lookup(x as usize))),
        },
        SrgbToLinearMethod {
            name: "Imageflow scalar powf (textbook)",
            f32_to_f32: Box::new(imageflow::srgb_to_linear),
            u8_to_f32: None,
            u16_to_f32: None,
        },
        SrgbToLinearMethod {
            name: "Imageflow LUT-8 lookup",
            f32_to_f32: Box::new(move |x| imageflow_lut.lookup((x * 255.0 + 0.5) as u8)),
            u8_to_f32: Some(Box::new(move |x| imageflow_lut_clone.lookup(x))),
            u16_to_f32: None,
        },
        SrgbToLinearMethod {
            name: "Naive powf (textbook constants)",
            f32_to_f32: Box::new(|x| naive_srgb_to_linear_f64(x as f64) as f32),
            u8_to_f32: None,
            u16_to_f32: None,
        },
    ]
}

#[cfg(not(feature = "alt"))]
fn create_srgb_to_linear_methods() -> Vec<SrgbToLinearMethod> {
    let lut8 = Arc::new(LinearTable8::new());
    let lut16 = Arc::new(LinearTable16::new());

    let lut8_clone = lut8.clone();
    let lut16_clone = lut16.clone();

    vec![
        SrgbToLinearMethod {
            name: "Scalar powf (optimized constants)",
            f32_to_f32: Box::new(srgb_to_linear),
            u8_to_f32: None,
            u16_to_f32: None,
        },
        SrgbToLinearMethod {
            name: "SIMD wide::f32x8 dirty_pow",
            f32_to_f32: Box::new(|x| {
                let v = f32x8::splat(x);
                let result: [f32; 8] = simd::srgb_to_linear_x8(v).into();
                result[0]
            }),
            u8_to_f32: None,
            u16_to_f32: None,
        },
        SrgbToLinearMethod {
            name: "LUT-8 direct lookup",
            f32_to_f32: Box::new(move |x| lut8.lookup((x * 255.0 + 0.5) as usize)),
            u8_to_f32: Some(Box::new(move |x| lut8_clone.lookup(x as usize))),
            u16_to_f32: None,
        },
        SrgbToLinearMethod {
            name: "LUT-16 direct lookup",
            f32_to_f32: Box::new(move |x| lut16.lookup((x * 65535.0 + 0.5) as usize)),
            u8_to_f32: None,
            u16_to_f32: Some(Box::new(move |x| lut16_clone.lookup(x as usize))),
        },
    ]
}

#[cfg(feature = "alt")]
fn create_linear_to_srgb_methods() -> Vec<LinearToSrgbMethod> {
    // Pre-create shared resources
    let encode12 = Arc::new(EncodeTable12::new());
    let encode16 = Arc::new(EncodeTable16::new());
    let converter = Arc::new(SrgbConverter::new());

    let encode12_clone = encode12.clone();
    let encode12_clone2 = encode12.clone();
    let encode16_clone = encode16.clone();
    let converter_clone = converter.clone();
    let converter_clone2 = converter.clone();

    vec![
        LinearToSrgbMethod {
            name: "Scalar powf (optimized constants)",
            f32_to_f32: Box::new(linear_to_srgb),
            f32_to_u8: None,
            f32_to_u16: None,
        },
        LinearToSrgbMethod {
            name: "SIMD wide::f32x8 dirty_pow",
            f32_to_f32: Box::new(|x| {
                let v = f32x8::splat(x);
                let result: [f32; 8] = simd::linear_to_srgb_x8(v).into();
                result[0]
            }),
            f32_to_u8: Some(Box::new(|x| {
                let v = f32x8::splat(x);
                simd::linear_to_srgb_u8_x8(v)[0]
            })),
            f32_to_u16: None,
        },
        LinearToSrgbMethod {
            name: "LUT-12 interpolated",
            f32_to_f32: Box::new(move |x| lut_interp_linear_float(x, encode12.as_slice())),
            f32_to_u8: Some(Box::new(move |x| {
                (lut_interp_linear_float(x, encode12_clone.as_slice()) * 255.0 + 0.5) as u8
            })),
            f32_to_u16: Some(Box::new(move |x| {
                (lut_interp_linear_float(x, encode12_clone2.as_slice()) * 65535.0 + 0.5) as u16
            })),
        },
        LinearToSrgbMethod {
            name: "LUT-16 interpolated",
            f32_to_f32: Box::new(move |x| lut_interp_linear_float(x, encode16.as_slice())),
            f32_to_u8: None,
            f32_to_u16: Some(Box::new(move |x| {
                (lut_interp_linear_float(x, encode16_clone.as_slice()) * 65535.0 + 0.5) as u16
            })),
        },
        LinearToSrgbMethod {
            name: "SrgbConverter (LUT-12)",
            f32_to_f32: Box::new(move |x| converter.linear_to_srgb(x)),
            f32_to_u8: Some(Box::new(move |x| converter_clone.linear_to_srgb_u8(x))),
            f32_to_u16: Some(Box::new(move |x| {
                (converter_clone2.linear_to_srgb(x) * 65535.0 + 0.5) as u16
            })),
        },
        LinearToSrgbMethod {
            name: "Imageflow fastpow",
            f32_to_f32: Box::new(imageflow::linear_to_srgb),
            f32_to_u8: Some(Box::new(imageflow::linear_to_srgb_u8_fastpow)),
            f32_to_u16: None,
        },
        LinearToSrgbMethod {
            name: "Imageflow LUT-16K",
            f32_to_f32: Box::new(|x| imageflow::linear_to_srgb_lut(x) as f32 / 255.0),
            f32_to_u8: Some(Box::new(imageflow::linear_to_srgb_lut)),
            f32_to_u16: None,
        },
        LinearToSrgbMethod {
            name: "Naive powf (textbook constants)",
            f32_to_f32: Box::new(|x| naive_linear_to_srgb_f64(x as f64) as f32),
            f32_to_u8: None,
            f32_to_u16: None,
        },
    ]
}

#[cfg(not(feature = "alt"))]
fn create_linear_to_srgb_methods() -> Vec<LinearToSrgbMethod> {
    let encode12 = Arc::new(EncodeTable12::new());
    let encode16 = Arc::new(EncodeTable16::new());
    let converter = Arc::new(SrgbConverter::new());

    let encode12_clone = encode12.clone();
    let encode12_clone2 = encode12.clone();
    let encode16_clone = encode16.clone();
    let converter_clone = converter.clone();
    let converter_clone2 = converter.clone();

    vec![
        LinearToSrgbMethod {
            name: "Scalar powf (optimized constants)",
            f32_to_f32: Box::new(linear_to_srgb),
            f32_to_u8: None,
            f32_to_u16: None,
        },
        LinearToSrgbMethod {
            name: "SIMD wide::f32x8 dirty_pow",
            f32_to_f32: Box::new(|x| {
                let v = f32x8::splat(x);
                let result: [f32; 8] = simd::linear_to_srgb_x8(v).into();
                result[0]
            }),
            f32_to_u8: Some(Box::new(|x| {
                let v = f32x8::splat(x);
                simd::linear_to_srgb_u8_x8(v)[0]
            })),
            f32_to_u16: None,
        },
        LinearToSrgbMethod {
            name: "LUT-12 interpolated",
            f32_to_f32: Box::new(move |x| lut_interp_linear_float(x, encode12.as_slice())),
            f32_to_u8: Some(Box::new(move |x| {
                (lut_interp_linear_float(x, encode12_clone.as_slice()) * 255.0 + 0.5) as u8
            })),
            f32_to_u16: Some(Box::new(move |x| {
                (lut_interp_linear_float(x, encode12_clone2.as_slice()) * 65535.0 + 0.5) as u16
            })),
        },
        LinearToSrgbMethod {
            name: "LUT-16 interpolated",
            f32_to_f32: Box::new(move |x| lut_interp_linear_float(x, encode16.as_slice())),
            f32_to_u8: None,
            f32_to_u16: Some(Box::new(move |x| {
                (lut_interp_linear_float(x, encode16_clone.as_slice()) * 65535.0 + 0.5) as u16
            })),
        },
        LinearToSrgbMethod {
            name: "SrgbConverter (LUT-12)",
            f32_to_f32: Box::new(move |x| converter.linear_to_srgb(x)),
            f32_to_u8: Some(Box::new(move |x| converter_clone.linear_to_srgb_u8(x))),
            f32_to_u16: Some(Box::new(move |x| {
                (converter_clone2.linear_to_srgb(x) * 65535.0 + 0.5) as u16
            })),
        },
    ]
}

// ============================================================================
// Statistics
// ============================================================================

#[derive(Debug, Clone)]
struct F32Stats {
    name: String,
    max_abs_error: f64,
    sum_abs_error: f64,
    max_ulp: u32,
    sum_ulp: u64,
    count: u64,
}

impl F32Stats {
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

    #[cfg(feature = "alt")]
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

    #[cfg(not(feature = "alt"))]
    fn update(&mut self, expected: f64, actual: f64) {
        let abs_error = (expected - actual).abs();

        self.sum_abs_error += abs_error;
        self.count += 1;

        if abs_error > self.max_abs_error {
            self.max_abs_error = abs_error;
        }
    }

    fn avg_ulp(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_ulp as f64 / self.count as f64
        }
    }
}

#[derive(Debug, Clone)]
struct IntStats {
    name: String,
    exact: u32,
    off_by_1: u32,
    max_diff: u32,
    sum_diff: u64,
    count: u32,
}

impl IntStats {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            exact: 0,
            off_by_1: 0,
            max_diff: 0,
            sum_diff: 0,
            count: 0,
        }
    }

    fn update(&mut self, expected: u32, actual: u32) {
        let diff = (expected as i64 - actual as i64).unsigned_abs() as u32;
        self.count += 1;
        self.sum_diff += diff as u64;

        if diff == 0 {
            self.exact += 1;
        } else if diff == 1 {
            self.off_by_1 += 1;
        }
        if diff > self.max_diff {
            self.max_diff = diff;
        }
    }

    fn avg_diff(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_diff as f64 / self.count as f64
        }
    }
}

// ============================================================================
// Comparison Functions
// ============================================================================

fn print_f32_header() {
    println!(
        "{:<45} {:>10} {:>10} {:>12}",
        "Implementation", "Max ULP", "Avg ULP", "Max Abs Err"
    );
    println!("{}", "-".repeat(79));
}

fn print_f32_stats(stats: &F32Stats) {
    println!(
        "{:<45} {:>10} {:>10.2} {:>12.2e}",
        stats.name,
        stats.max_ulp,
        stats.avg_ulp(),
        stats.max_abs_error,
    );
}

fn print_int_header(bits: u32) {
    println!(
        "{:<45} {:>8} {:>10} {:>10} {:>10}",
        "Implementation",
        "Max Diff",
        format!("Exact/{}", 1 << bits),
        format!("±1/{}", 1 << bits),
        "Avg Diff"
    );
    println!("{}", "-".repeat(85));
}

fn print_int_stats(stats: &IntStats) {
    println!(
        "{:<45} {:>8} {:>10} {:>10} {:>10.3}",
        stats.name,
        stats.max_diff,
        stats.exact,
        stats.off_by_1,
        stats.avg_diff()
    );
}

fn compare_srgb_to_linear_f32(methods: &[SrgbToLinearMethod]) {
    println!("=== sRGB → Linear (f32 → f32) ===\n");
    println!("Reference: f64 implementation with IEC 61966-2-1 constants");
    println!("Test range: 0.01 to 1.0 (excluding near-zero ULP explosion)\n");

    let test_values: Vec<f64> = (100..=10000).map(|i| i as f64 / 10000.0).collect();

    let mut stats: Vec<F32Stats> = methods.iter().map(|m| F32Stats::new(m.name)).collect();

    for &input in &test_values {
        let reference = srgb_to_linear_f64(input);
        let input_f32 = input as f32;

        for (i, method) in methods.iter().enumerate() {
            let result = method.convert_f32(input_f32) as f64;
            stats[i].update(reference, result);
        }
    }

    print_f32_header();
    for s in &stats {
        print_f32_stats(s);
    }
}

fn compare_linear_to_srgb_f32(methods: &[LinearToSrgbMethod]) {
    println!("\n=== Linear → sRGB (f32 → f32) ===\n");
    println!("Reference: f64 implementation with IEC 61966-2-1 constants\n");

    let test_values: Vec<f64> = (0..=10000).map(|i| i as f64 / 10000.0).collect();

    let mut stats: Vec<F32Stats> = methods.iter().map(|m| F32Stats::new(m.name)).collect();

    for &input in &test_values {
        let reference = linear_to_srgb_f64(input);
        let input_f32 = input as f32;

        for (i, method) in methods.iter().enumerate() {
            let result = method.convert_f32(input_f32) as f64;
            stats[i].update(reference, result);
        }
    }

    print_f32_header();
    for s in &stats {
        print_f32_stats(s);
    }
}

fn compare_u8_to_linear(methods: &[SrgbToLinearMethod]) {
    println!("\n=== sRGB u8 → Linear f32 ===\n");
    println!("Reference: f64 srgb_to_linear(u8/255.0)\n");

    let mut stats: Vec<F32Stats> = methods.iter().map(|m| F32Stats::new(m.name)).collect();

    for i in 0..=255u8 {
        let reference = srgb_to_linear_f64(i as f64 / 255.0);

        for (idx, method) in methods.iter().enumerate() {
            let result = method.convert_u8(i) as f64;
            stats[idx].update(reference, result);
        }
    }

    print_f32_header();
    for s in &stats {
        print_f32_stats(s);
    }
}

fn compare_u16_to_linear(methods: &[SrgbToLinearMethod]) {
    println!("\n=== sRGB u16 → Linear f32 ===\n");
    println!("Reference: f64 srgb_to_linear(u16/65535.0)\n");

    let mut stats: Vec<F32Stats> = methods.iter().map(|m| F32Stats::new(m.name)).collect();

    // Test every 256th value (256 samples across the u16 range)
    for i in (0..=65535u16).step_by(256) {
        let reference = srgb_to_linear_f64(i as f64 / 65535.0);

        for (idx, method) in methods.iter().enumerate() {
            let result = method.convert_u16(i) as f64;
            stats[idx].update(reference, result);
        }
    }

    print_f32_header();
    for s in &stats {
        print_f32_stats(s);
    }
}

fn compare_linear_to_u8(methods: &[LinearToSrgbMethod]) {
    println!("\n=== Linear f32 → sRGB u8 ===\n");
    println!("Reference: round(f64_linear_to_srgb * 255)\n");

    let mut stats: Vec<IntStats> = methods.iter().map(|m| IntStats::new(m.name)).collect();

    for i in 0..=255u8 {
        // Use the exact linear value that should map back to this u8
        let srgb_normalized = i as f64 / 255.0;
        let linear_input = srgb_to_linear_f64(srgb_normalized);
        let reference = i as u32;

        for (idx, method) in methods.iter().enumerate() {
            let result = method.convert_to_u8(linear_input as f32) as u32;
            stats[idx].update(reference, result);
        }
    }

    print_int_header(8);
    for s in &stats {
        print_int_stats(s);
    }
}

fn compare_linear_to_u16(methods: &[LinearToSrgbMethod]) {
    println!("\n=== Linear f32 → sRGB u16 ===\n");
    println!("Reference: round(f64_linear_to_srgb * 65535)\n");

    let mut stats: Vec<IntStats> = methods.iter().map(|m| IntStats::new(m.name)).collect();

    // Test every 256th value
    for i in (0..=65535u16).step_by(256) {
        let srgb_normalized = i as f64 / 65535.0;
        let linear_input = srgb_to_linear_f64(srgb_normalized);
        let reference = i as u32;

        for (idx, method) in methods.iter().enumerate() {
            let result = method.convert_to_u16(linear_input as f32) as u32;
            stats[idx].update(reference, result);
        }
    }

    print_int_header(16);
    for s in &stats {
        print_int_stats(s);
    }
}

fn compare_u8_roundtrip(
    srgb_to_linear_methods: &[SrgbToLinearMethod],
    linear_to_srgb_methods: &[LinearToSrgbMethod],
) {
    println!("\n=== u8 Round-trip: sRGB u8 → Linear f32 → sRGB u8 ===\n");
    println!("Testing all pairs of (sRGB→Linear, Linear→sRGB) methods\n");

    println!(
        "{:<35} × {:<35} {:>6} {:>8} {:>6}",
        "sRGB→Linear", "Linear→sRGB", "Exact", "±1", "Max"
    );
    println!("{}", "-".repeat(95));

    for s2l in srgb_to_linear_methods {
        for l2s in linear_to_srgb_methods {
            let mut exact = 0u32;
            let mut off1 = 0u32;
            let mut max_diff = 0u32;

            for i in 0..=255u8 {
                let linear = s2l.convert_u8(i);
                let back = l2s.convert_to_u8(linear);
                let diff = (i as i32 - back as i32).unsigned_abs();

                if diff == 0 {
                    exact += 1;
                } else if diff == 1 {
                    off1 += 1;
                }
                max_diff = max_diff.max(diff);
            }

            println!(
                "{:<35} × {:<35} {:>3}/256 {:>5}/256 {:>6}",
                truncate(s2l.name, 35),
                truncate(l2s.name, 35),
                exact,
                off1,
                max_diff
            );
        }
    }
}

fn compare_u16_roundtrip(
    srgb_to_linear_methods: &[SrgbToLinearMethod],
    linear_to_srgb_methods: &[LinearToSrgbMethod],
) {
    println!("\n=== u16 Round-trip: sRGB u16 → Linear f32 → sRGB u16 ===\n");
    println!("Testing all 65536 values for each pair\n");

    println!(
        "{:<35} × {:<35} {:>8} {:>8} {:>6}",
        "sRGB→Linear", "Linear→sRGB", "Exact", "±1", "Max"
    );
    println!("{}", "-".repeat(97));

    for s2l in srgb_to_linear_methods {
        for l2s in linear_to_srgb_methods {
            let mut exact = 0u32;
            let mut off1 = 0u32;
            let mut max_diff = 0u32;

            for i in 0..=65535u16 {
                let linear = s2l.convert_u16(i);
                let back = l2s.convert_to_u16(linear);
                let diff = (i as i32 - back as i32).unsigned_abs();

                if diff == 0 {
                    exact += 1;
                } else if diff == 1 {
                    off1 += 1;
                }
                max_diff = max_diff.max(diff);
            }

            println!(
                "{:<35} × {:<35} {:>5}/65536 {:>5}/65536 {:>6}",
                truncate(s2l.name, 35),
                truncate(l2s.name, 35),
                exact,
                off1,
                max_diff
            );
        }
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len - 1])
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║           sRGB Conversion Accuracy Comparison - All Implementations          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "alt"))]
    println!("Note: Run with --features alt to include imageflow and naive implementations\n");

    let s2l = create_srgb_to_linear_methods();
    let l2s = create_linear_to_srgb_methods();

    compare_srgb_to_linear_f32(&s2l);
    compare_linear_to_srgb_f32(&l2s);

    compare_u8_to_linear(&s2l);
    compare_u16_to_linear(&s2l);

    compare_linear_to_u8(&l2s);
    compare_linear_to_u16(&l2s);

    compare_u8_roundtrip(&s2l, &l2s);
    compare_u16_roundtrip(&s2l, &l2s);

    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!("Notes:");
    println!("  - ULP = Units in Last Place (floating point precision measure)");
    println!("  - Off-by-1 errors are acceptable for u8 precision");
    println!("  - For u16, off-by-1 represents ~0.0015% error");
    println!("  - 'textbook constants' = naive 0.04045/0.0031308 thresholds");
    println!("  - 'optimized constants' = IEC 61966-2-1 continuous piecewise values");
}
