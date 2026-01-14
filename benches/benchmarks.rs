use criterion::{Criterion, black_box, criterion_group, criterion_main};
use linear_srgb::{
    SrgbConverter, linear_to_srgb, linear_to_srgb_slice, srgb_to_linear, srgb_to_linear_slice,
};

fn bench_direct_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct");

    group.bench_function("srgb_to_linear", |b| {
        b.iter(|| {
            for i in 0..=255u8 {
                black_box(srgb_to_linear(black_box(i as f32 / 255.0)));
            }
        })
    });

    group.bench_function("linear_to_srgb", |b| {
        b.iter(|| {
            for i in 0..=255u8 {
                black_box(linear_to_srgb(black_box(i as f32 / 255.0)));
            }
        })
    });

    group.finish();
}

fn bench_lut_conversions(c: &mut Criterion) {
    let conv = SrgbConverter::new();
    let mut group = c.benchmark_group("lut");

    group.bench_function("srgb_u8_to_linear", |b| {
        b.iter(|| {
            for i in 0..=255u8 {
                black_box(conv.srgb_u8_to_linear(black_box(i)));
            }
        })
    });

    group.bench_function("linear_to_srgb", |b| {
        b.iter(|| {
            for i in 0..=255u8 {
                black_box(conv.linear_to_srgb(black_box(i as f32 / 255.0)));
            }
        })
    });

    group.finish();
}

fn bench_batch_conversions(c: &mut Criterion) {
    let conv = SrgbConverter::new();
    let mut group = c.benchmark_group("batch");

    let srgb_input: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let linear_input: Vec<f32> = (0..1024).map(|i| (i % 256) as f32 / 255.0).collect();
    let mut linear_output = vec![0.0f32; 1024];
    let mut srgb_output = vec![0u8; 1024];

    group.bench_function("srgb_to_linear_1024", |b| {
        b.iter(|| {
            conv.batch_srgb_to_linear(black_box(&srgb_input), &mut linear_output);
            black_box(&linear_output);
        })
    });

    group.bench_function("linear_to_srgb_1024", |b| {
        b.iter(|| {
            conv.batch_linear_to_srgb(black_box(&linear_input), &mut srgb_output);
            black_box(&srgb_output);
        })
    });

    group.finish();
}

fn bench_slice_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice");

    let mut values: Vec<f32> = (0..1024).map(|i| (i % 256) as f32 / 255.0).collect();

    group.bench_function("srgb_to_linear_slice_1024", |b| {
        b.iter(|| {
            srgb_to_linear_slice(black_box(&mut values));
        })
    });

    group.bench_function("linear_to_srgb_slice_1024", |b| {
        b.iter(|| {
            linear_to_srgb_slice(black_box(&mut values));
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_direct_conversions,
    bench_lut_conversions,
    bench_batch_conversions,
    bench_slice_conversions,
);

criterion_main!(benches);
