use criterion::{Criterion, black_box, criterion_group, criterion_main};
use linear_srgb::accuracy::{naive_linear_to_srgb_f32, naive_srgb_to_linear_f32};
use linear_srgb::lut::{EncodeTable12, LinearTable12, lut_interp_linear_float};
use linear_srgb::{simd, srgb_to_linear};
use moxcms::{
    CicpColorPrimaries, CicpProfile, ColorProfile, Layout, MatrixCoefficients, RenderingIntent,
    TransferCharacteristics, TransformOptions,
};
use wide::f32x8;

// 10000 values = 1250 f32x8 vectors
const NUM_VECTORS: usize = 1250;
const BATCH_SIZE: usize = NUM_VECTORS * 8;

fn create_test_vectors() -> Vec<f32x8> {
    (0..NUM_VECTORS)
        .map(|i| {
            let base = (i * 8) as f32 / BATCH_SIZE as f32;
            let step = 1.0 / BATCH_SIZE as f32;
            f32x8::from([
                base,
                base + step,
                base + 2.0 * step,
                base + 3.0 * step,
                base + 4.0 * step,
                base + 5.0 * step,
                base + 6.0 * step,
                base + 7.0 * step,
            ])
        })
        .collect()
}

fn create_test_f32() -> Vec<f32> {
    (0..BATCH_SIZE)
        .map(|i| i as f32 / BATCH_SIZE as f32)
        .collect()
}

// For moxcms RGB transform (3 channels)
fn create_test_rgb() -> Vec<f32> {
    (0..BATCH_SIZE)
        .flat_map(|i| {
            let v = i as f32 / BATCH_SIZE as f32;
            [v, v, v] // RGB with same value
        })
        .collect()
}

fn bench_srgb_to_linear_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group("srgb_to_linear_10k");
    let vectors = create_test_vectors();
    let f32_data = create_test_f32();

    // Native f32x8 slice (best case - data already in SIMD format)
    group.bench_function("native_f32x8", |b| {
        let mut output = vectors.clone();
        b.iter(|| {
            simd::srgb_to_linear_x8_slice(&mut output);
            black_box(&output);
        })
    });

    // f32 slice with chunks (typical use case)
    group.bench_function("f32_slice", |b| {
        let mut values = f32_data.clone();
        b.iter(|| {
            simd::srgb_to_linear_slice(&mut values);
            black_box(&values);
        })
    });

    // Scalar loop (baseline)
    group.bench_function("scalar", |b| {
        let mut values = f32_data.clone();
        b.iter(|| {
            values.copy_from_slice(&f32_data);
            for v in values.iter_mut() {
                *v = srgb_to_linear(*v);
            }
            black_box(&values);
        })
    });

    // Naive scalar (textbook implementation)
    group.bench_function("naive", |b| {
        let mut values = f32_data.clone();
        b.iter(|| {
            values.copy_from_slice(&f32_data);
            for v in values.iter_mut() {
                *v = naive_srgb_to_linear_f32(*v);
            }
            black_box(&values);
        })
    });

    // LUT-based (12-bit table with interpolation)
    group.bench_function("lut_12bit", |b| {
        let table = LinearTable12::new();
        let mut values = f32_data.clone();
        b.iter(|| {
            for v in values.iter_mut() {
                *v = lut_interp_linear_float(*v, table.as_slice());
            }
            black_box(&values);
        })
    });

    // moxcms transform API (sRGB → Linear)
    group.bench_function("moxcms_transform", |b| {
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
        let transform = srgb
            .create_transform_f32(Layout::Rgb, &linear_srgb, Layout::Rgb, options)
            .expect("Failed to create sRGB→linear transform");

        let rgb_input = create_test_rgb();
        let mut rgb_output = vec![0.0f32; rgb_input.len()];

        b.iter(|| {
            let _ = transform.transform(black_box(&rgb_input), &mut rgb_output);
            black_box(&rgb_output);
        })
    });

    group.finish();
}

fn bench_linear_to_srgb_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_to_srgb_10k");

    // Create linear input
    let srgb_vectors = create_test_vectors();
    let linear_vectors: Vec<f32x8> = srgb_vectors
        .iter()
        .map(|v| simd::srgb_to_linear_x8(*v))
        .collect();

    let f32_data = create_test_f32();
    let linear_f32: Vec<f32> = f32_data.iter().map(|&v| srgb_to_linear(v)).collect();

    // Native f32x8 slice (best case)
    group.bench_function("native_f32x8", |b| {
        let mut output = linear_vectors.clone();
        b.iter(|| {
            simd::linear_to_srgb_x8_slice(&mut output);
            black_box(&output);
        })
    });

    // f32 slice with chunks
    group.bench_function("f32_slice", |b| {
        let mut values = linear_f32.clone();
        b.iter(|| {
            simd::linear_to_srgb_slice(&mut values);
            black_box(&values);
        })
    });

    // Scalar loop
    group.bench_function("scalar", |b| {
        let mut values = linear_f32.clone();
        b.iter(|| {
            values.copy_from_slice(&linear_f32);
            for v in values.iter_mut() {
                *v = linear_srgb::linear_to_srgb(*v);
            }
            black_box(&values);
        })
    });

    // Naive scalar
    group.bench_function("naive", |b| {
        let mut values = linear_f32.clone();
        b.iter(|| {
            values.copy_from_slice(&linear_f32);
            for v in values.iter_mut() {
                *v = naive_linear_to_srgb_f32(*v);
            }
            black_box(&values);
        })
    });

    // LUT-based (12-bit table with interpolation)
    group.bench_function("lut_12bit", |b| {
        let table = EncodeTable12::new();
        let mut values = linear_f32.clone();
        b.iter(|| {
            for v in values.iter_mut() {
                *v = lut_interp_linear_float(*v, table.as_slice());
            }
            black_box(&values);
        })
    });

    // moxcms transform API (Linear → sRGB)
    group.bench_function("moxcms_transform", |b| {
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
        let transform = linear_srgb
            .create_transform_f32(Layout::Rgb, &srgb, Layout::Rgb, options)
            .expect("Failed to create linear→sRGB transform");

        // Create linear RGB input
        let linear_rgb: Vec<f32> = linear_f32.iter().flat_map(|&v| [v, v, v]).collect();
        let mut rgb_output = vec![0.0f32; linear_rgb.len()];

        b.iter(|| {
            let _ = transform.transform(black_box(&linear_rgb), &mut rgb_output);
            black_box(&rgb_output);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_srgb_to_linear_10k, bench_linear_to_srgb_10k,);

criterion_main!(benches);
