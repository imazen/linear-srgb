use criterion::{Criterion, black_box, criterion_group, criterion_main};
use linear_srgb::accuracy::{naive_linear_to_srgb_f32, naive_srgb_to_linear_f32};
use linear_srgb::{SrgbConverter, simd, srgb_to_linear};
use moxcms::TransferCharacteristics;

const BATCH_SIZE: usize = 10000;

fn create_test_data() -> Vec<f32> {
    (0..BATCH_SIZE)
        .map(|i| (i as f32 / BATCH_SIZE as f32))
        .collect()
}

fn bench_srgb_to_linear_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group("srgb_to_linear_10k");
    let input = create_test_data();

    // SIMD (wide crate)
    group.bench_function("simd_wide", |b| {
        let mut values = input.clone();
        b.iter(|| {
            values.copy_from_slice(&input);
            simd::srgb_to_linear_slice(black_box(&mut values));
            black_box(&values);
        })
    });

    // Scalar loop
    group.bench_function("scalar", |b| {
        let mut values = input.clone();
        b.iter(|| {
            values.copy_from_slice(&input);
            for v in values.iter_mut() {
                *v = srgb_to_linear(*v);
            }
            black_box(&values);
        })
    });

    // Naive scalar
    group.bench_function("naive", |b| {
        let mut values = input.clone();
        b.iter(|| {
            values.copy_from_slice(&input);
            for v in values.iter_mut() {
                *v = naive_srgb_to_linear_f32(*v);
            }
            black_box(&values);
        })
    });

    // moxcms f64 (reference)
    group.bench_function("moxcms_f64", |b| {
        let tc = TransferCharacteristics::Srgb;
        let mut output = vec![0.0f64; BATCH_SIZE];
        b.iter(|| {
            for (i, &inp) in input.iter().enumerate() {
                output[i] = tc.linearize(inp as f64);
            }
            black_box(&output);
        })
    });

    group.finish();
}

fn bench_linear_to_srgb_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_to_srgb_10k");

    // Create linear input (convert sRGB to linear first)
    let srgb_input = create_test_data();
    let input: Vec<f32> = srgb_input.iter().map(|&v| srgb_to_linear(v)).collect();

    // SIMD (wide crate)
    group.bench_function("simd_wide", |b| {
        let mut values = input.clone();
        b.iter(|| {
            values.copy_from_slice(&input);
            simd::linear_to_srgb_slice(black_box(&mut values));
            black_box(&values);
        })
    });

    // Scalar loop
    group.bench_function("scalar", |b| {
        let mut values = input.clone();
        b.iter(|| {
            values.copy_from_slice(&input);
            for v in values.iter_mut() {
                *v = linear_srgb::linear_to_srgb(*v);
            }
            black_box(&values);
        })
    });

    // Naive scalar
    group.bench_function("naive", |b| {
        let mut values = input.clone();
        b.iter(|| {
            values.copy_from_slice(&input);
            for v in values.iter_mut() {
                *v = naive_linear_to_srgb_f32(*v);
            }
            black_box(&values);
        })
    });

    // moxcms f64
    group.bench_function("moxcms_f64", |b| {
        let tc = TransferCharacteristics::Srgb;
        let mut output = vec![0.0f64; BATCH_SIZE];
        b.iter(|| {
            for (i, &inp) in input.iter().enumerate() {
                output[i] = tc.gamma(inp as f64);
            }
            black_box(&output);
        })
    });

    group.finish();
}

fn bench_lut_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group("lut_10k");
    let conv = SrgbConverter::new();

    // For LUT, we need u8 input
    let u8_input: Vec<u8> = (0..BATCH_SIZE).map(|i| (i % 256) as u8).collect();
    let mut output = vec![0.0f32; BATCH_SIZE];

    group.bench_function("srgb_u8_to_linear", |b| {
        b.iter(|| {
            conv.batch_srgb_to_linear(black_box(&u8_input), &mut output);
            black_box(&output);
        })
    });

    // For linear_to_srgb, use f32 input
    let f32_input = create_test_data();
    let linear_input: Vec<f32> = f32_input.iter().map(|&v| srgb_to_linear(v)).collect();
    let mut u8_output = vec![0u8; BATCH_SIZE];

    group.bench_function("linear_to_srgb_u8", |b| {
        b.iter(|| {
            conv.batch_linear_to_srgb(black_box(&linear_input), &mut u8_output);
            black_box(&u8_output);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_srgb_to_linear_10k,
    bench_linear_to_srgb_10k,
    bench_lut_10k,
);

criterion_main!(benches);
