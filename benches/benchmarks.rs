use criterion::{Criterion, black_box, criterion_group, criterion_main};
use linear_srgb::accuracy::{naive_linear_to_srgb_f32, naive_srgb_to_linear_f32};
use linear_srgb::{
    SrgbConverter, linear_to_srgb, linear_to_srgb_f64, srgb_to_linear, srgb_to_linear_f64,
    srgb_to_linear_slice,
};
use moxcms::TransferCharacteristics;

fn bench_srgb_to_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("srgb_to_linear");

    // linear-srgb f32
    group.bench_function("linear-srgb_f32", |b| {
        b.iter(|| {
            for i in 0..=255u8 {
                black_box(srgb_to_linear(black_box(i as f32 / 255.0)));
            }
        })
    });

    // linear-srgb f64
    group.bench_function("linear-srgb_f64", |b| {
        b.iter(|| {
            for i in 0..=255u8 {
                black_box(srgb_to_linear_f64(black_box(i as f64 / 255.0)));
            }
        })
    });

    // moxcms f64 (via TransferCharacteristics)
    group.bench_function("moxcms_f64", |b| {
        let tc = TransferCharacteristics::Srgb;
        b.iter(|| {
            for i in 0..=255u8 {
                black_box(tc.linearize(black_box(i as f64 / 255.0)));
            }
        })
    });

    // Naive f32
    group.bench_function("naive_f32", |b| {
        b.iter(|| {
            for i in 0..=255u8 {
                black_box(naive_srgb_to_linear_f32(black_box(i as f32 / 255.0)));
            }
        })
    });

    group.finish();
}

fn bench_linear_to_srgb(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_to_srgb");

    // linear-srgb f32
    group.bench_function("linear-srgb_f32", |b| {
        b.iter(|| {
            for i in 0..=255u8 {
                black_box(linear_to_srgb(black_box(i as f32 / 255.0)));
            }
        })
    });

    // linear-srgb f64
    group.bench_function("linear-srgb_f64", |b| {
        b.iter(|| {
            for i in 0..=255u8 {
                black_box(linear_to_srgb_f64(black_box(i as f64 / 255.0)));
            }
        })
    });

    // moxcms f64 (via TransferCharacteristics)
    group.bench_function("moxcms_f64", |b| {
        let tc = TransferCharacteristics::Srgb;
        b.iter(|| {
            for i in 0..=255u8 {
                black_box(tc.gamma(black_box(i as f64 / 255.0)));
            }
        })
    });

    // Naive f32
    group.bench_function("naive_f32", |b| {
        b.iter(|| {
            for i in 0..=255u8 {
                black_box(naive_linear_to_srgb_f32(black_box(i as f32 / 255.0)));
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

fn bench_batch_1024(c: &mut Criterion) {
    let conv = SrgbConverter::new();
    let mut group = c.benchmark_group("batch_1024");

    let input_f32: Vec<f32> = (0..1024).map(|i| (i % 256) as f32 / 255.0).collect();

    // LUT batch
    let srgb_input: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let mut linear_output = vec![0.0f32; 1024];

    group.bench_function("lut_srgb_to_linear", |b| {
        b.iter(|| {
            conv.batch_srgb_to_linear(black_box(&srgb_input), &mut linear_output);
            black_box(&linear_output);
        })
    });

    // Direct slice
    let mut values = input_f32.clone();
    group.bench_function("direct_srgb_to_linear", |b| {
        b.iter(|| {
            values.copy_from_slice(&input_f32);
            srgb_to_linear_slice(black_box(&mut values));
            black_box(&values);
        })
    });

    // Naive slice
    group.bench_function("naive_srgb_to_linear", |b| {
        b.iter(|| {
            values.copy_from_slice(&input_f32);
            for v in values.iter_mut() {
                *v = naive_srgb_to_linear_f32(*v);
            }
            black_box(&values);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_srgb_to_linear,
    bench_linear_to_srgb,
    bench_lut_conversions,
    bench_batch_1024,
);

criterion_main!(benches);
