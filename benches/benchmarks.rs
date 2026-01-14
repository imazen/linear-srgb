//! Comprehensive benchmarks for sRGB conversion methods.
//!
//! Tests all combinations of input/output types (u8, u16, f32) across implementations.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use linear_srgb::imageflow;
use linear_srgb::lut::{
    EncodeTable12, EncodeTable16, LinearTable8, LinearTable16, SrgbConverter,
    lut_interp_linear_float,
};
use linear_srgb::{linear_to_srgb, simd, srgb_to_linear};
use wide::f32x8;

const BATCH_SIZE: usize = 10000;

// ============================================================================
// Test Data Generation
// ============================================================================

fn create_f32_srgb() -> Vec<f32> {
    (0..BATCH_SIZE)
        .map(|i| i as f32 / BATCH_SIZE as f32)
        .collect()
}

fn create_f32_linear() -> Vec<f32> {
    create_f32_srgb()
        .iter()
        .map(|&v| srgb_to_linear(v))
        .collect()
}

fn create_u8_srgb() -> Vec<u8> {
    (0..BATCH_SIZE).map(|i| (i % 256) as u8).collect()
}

fn create_u16_srgb() -> Vec<u16> {
    (0..BATCH_SIZE)
        .map(|i| ((i * 65535) / BATCH_SIZE) as u16)
        .collect()
}

fn create_f32x8_srgb() -> Vec<f32x8> {
    let f32_data = create_f32_srgb();
    f32_data
        .chunks(8)
        .map(|chunk| {
            let mut arr = [0.0f32; 8];
            arr[..chunk.len()].copy_from_slice(chunk);
            f32x8::from(arr)
        })
        .collect()
}

fn create_f32x8_linear() -> Vec<f32x8> {
    create_f32x8_srgb()
        .iter()
        .map(|&v| simd::srgb_to_linear_x8(v))
        .collect()
}

// ============================================================================
// sRGB → Linear Benchmarks
// ============================================================================

fn bench_srgb_to_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("srgb_to_linear");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    // Shared resources
    let lut8 = LinearTable8::new();
    let lut16 = LinearTable16::new();
    let imageflow_lut = imageflow::SrgbToLinearLut::new();
    let f32_data = create_f32_srgb();
    let u8_data = create_u8_srgb();
    let u16_data = create_u16_srgb();
    let f32x8_data = create_f32x8_srgb();

    // === f32 → f32 ===

    group.bench_function("f32_f32/simd_dirty_pow", |b| {
        let mut output = f32x8_data.clone();
        b.iter(|| {
            simd::srgb_to_linear_x8_slice(&mut output);
            black_box(&output);
        })
    });

    group.bench_function("f32_f32/scalar_powf", |b| {
        let mut output = f32_data.clone();
        b.iter(|| {
            for v in output.iter_mut() {
                *v = srgb_to_linear(*v);
            }
            black_box(&output);
        })
    });

    group.bench_function("f32_f32/lut12_interp", |b| {
        let table = linear_srgb::lut::LinearTable12::new();
        let mut output = f32_data.clone();
        b.iter(|| {
            for v in output.iter_mut() {
                *v = lut_interp_linear_float(*v, table.as_slice());
            }
            black_box(&output);
        })
    });

    group.bench_function("f32_f32/imageflow_powf", |b| {
        let mut output = f32_data.clone();
        b.iter(|| {
            for v in output.iter_mut() {
                *v = imageflow::srgb_to_linear(*v);
            }
            black_box(&output);
        })
    });

    // === u8 → f32 ===

    group.bench_function("u8_f32/lut8_direct", |b| {
        let mut output = vec![0.0f32; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in u8_data.iter().zip(output.iter_mut()) {
                *o = lut8.lookup(*i as usize);
            }
            black_box(&output);
        })
    });

    group.bench_function("u8_f32/simd_lut8_batch", |b| {
        let mut output = vec![0.0f32; BATCH_SIZE];
        b.iter(|| {
            simd::srgb_u8_to_linear_batch(&lut8, black_box(&u8_data), &mut output);
            black_box(&output);
        })
    });

    group.bench_function("u8_f32/imageflow_lut8", |b| {
        let mut output = vec![0.0f32; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in u8_data.iter().zip(output.iter_mut()) {
                *o = imageflow_lut.lookup(*i);
            }
            black_box(&output);
        })
    });

    group.bench_function("u8_f32/scalar_powf", |b| {
        let mut output = vec![0.0f32; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in u8_data.iter().zip(output.iter_mut()) {
                *o = srgb_to_linear(*i as f32 / 255.0);
            }
            black_box(&output);
        })
    });

    // === u16 → f32 ===

    group.bench_function("u16_f32/lut16_direct", |b| {
        let mut output = vec![0.0f32; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in u16_data.iter().zip(output.iter_mut()) {
                *o = lut16.lookup(*i as usize);
            }
            black_box(&output);
        })
    });

    group.bench_function("u16_f32/scalar_powf", |b| {
        let mut output = vec![0.0f32; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in u16_data.iter().zip(output.iter_mut()) {
                *o = srgb_to_linear(*i as f32 / 65535.0);
            }
            black_box(&output);
        })
    });

    group.bench_function("u16_f32/lut8_quantized", |b| {
        let mut output = vec![0.0f32; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in u16_data.iter().zip(output.iter_mut()) {
                *o = lut8.lookup((*i >> 8) as usize);
            }
            black_box(&output);
        })
    });

    group.finish();
}

// ============================================================================
// Linear → sRGB Benchmarks
// ============================================================================

fn bench_linear_to_srgb(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_to_srgb");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    // Shared resources
    let encode12 = EncodeTable12::new();
    let encode16 = EncodeTable16::new();
    let converter = SrgbConverter::new();
    let lut8 = LinearTable8::new();

    let f32_linear = create_f32_linear();
    let f32x8_linear = create_f32x8_linear();

    // Create u8-derived linear values for fair u8 output comparison
    let u8_srgb = create_u8_srgb();
    let linear_from_u8: Vec<f32> = u8_srgb.iter().map(|&v| lut8.lookup(v as usize)).collect();

    // === f32 → f32 ===

    group.bench_function("f32_f32/simd_dirty_pow", |b| {
        let mut output = f32x8_linear.clone();
        b.iter(|| {
            simd::linear_to_srgb_x8_slice(&mut output);
            black_box(&output);
        })
    });

    group.bench_function("f32_f32/scalar_powf", |b| {
        let mut output = f32_linear.clone();
        b.iter(|| {
            for v in output.iter_mut() {
                *v = linear_to_srgb(*v);
            }
            black_box(&output);
        })
    });

    group.bench_function("f32_f32/lut12_interp", |b| {
        let mut output = f32_linear.clone();
        b.iter(|| {
            for v in output.iter_mut() {
                *v = lut_interp_linear_float(*v, encode12.as_slice());
            }
            black_box(&output);
        })
    });

    group.bench_function("f32_f32/lut16_interp", |b| {
        let mut output = f32_linear.clone();
        b.iter(|| {
            for v in output.iter_mut() {
                *v = lut_interp_linear_float(*v, encode16.as_slice());
            }
            black_box(&output);
        })
    });

    group.bench_function("f32_f32/imageflow_fastpow", |b| {
        let mut output = f32_linear.clone();
        b.iter(|| {
            for v in output.iter_mut() {
                *v = imageflow::linear_to_srgb(*v);
            }
            black_box(&output);
        })
    });

    // === f32 → u8 ===

    group.bench_function("f32_u8/simd_dirty_pow", |b| {
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            simd::linear_to_srgb_u8_batch(black_box(&linear_from_u8), &mut output);
            black_box(&output);
        })
    });

    group.bench_function("f32_u8/scalar_powf", |b| {
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in linear_from_u8.iter().zip(output.iter_mut()) {
                *o = (linear_to_srgb(*i) * 255.0 + 0.5) as u8;
            }
            black_box(&output);
        })
    });

    group.bench_function("f32_u8/lut12_interp", |b| {
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in linear_from_u8.iter().zip(output.iter_mut()) {
                *o = (lut_interp_linear_float(*i, encode12.as_slice()) * 255.0 + 0.5) as u8;
            }
            black_box(&output);
        })
    });

    group.bench_function("f32_u8/converter_lut12", |b| {
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            converter.batch_linear_to_srgb(black_box(&linear_from_u8), &mut output);
            black_box(&output);
        })
    });

    group.bench_function("f32_u8/imageflow_fastpow", |b| {
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in linear_from_u8.iter().zip(output.iter_mut()) {
                *o = imageflow::linear_to_srgb_u8_fastpow(*i);
            }
            black_box(&output);
        })
    });

    group.bench_function("f32_u8/imageflow_lut16k", |b| {
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in linear_from_u8.iter().zip(output.iter_mut()) {
                *o = imageflow::linear_to_srgb_lut(*i);
            }
            black_box(&output);
        })
    });

    // === f32 → u16 ===

    group.bench_function("f32_u16/scalar_powf", |b| {
        let mut output = vec![0u16; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in f32_linear.iter().zip(output.iter_mut()) {
                *o = (linear_to_srgb(*i) * 65535.0 + 0.5) as u16;
            }
            black_box(&output);
        })
    });

    group.bench_function("f32_u16/simd_dirty_pow", |b| {
        let mut output = vec![0u16; BATCH_SIZE];
        b.iter(|| {
            for (chunk, out_chunk) in f32_linear.chunks(8).zip(output.chunks_mut(8)) {
                let mut arr = [0.0f32; 8];
                arr[..chunk.len()].copy_from_slice(chunk);
                let v = f32x8::from(arr);
                let srgb = simd::linear_to_srgb_x8(v);
                let srgb_arr: [f32; 8] = srgb.into();
                for (s, o) in srgb_arr.iter().zip(out_chunk.iter_mut()) {
                    *o = (*s * 65535.0 + 0.5) as u16;
                }
            }
            black_box(&output);
        })
    });

    group.bench_function("f32_u16/lut16_interp", |b| {
        let mut output = vec![0u16; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in f32_linear.iter().zip(output.iter_mut()) {
                *o = (lut_interp_linear_float(*i, encode16.as_slice()) * 65535.0 + 0.5) as u16;
            }
            black_box(&output);
        })
    });

    group.finish();
}

// ============================================================================
// Roundtrip Benchmarks (measures full pipeline)
// ============================================================================

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let lut8 = LinearTable8::new();
    let lut16 = LinearTable16::new();
    let encode12 = EncodeTable12::new();
    let encode16 = EncodeTable16::new();

    let u8_data = create_u8_srgb();
    let u16_data = create_u16_srgb();

    // === u8 → f32 → u8 ===

    group.bench_function("u8_f32_u8/lut8_simd", |b| {
        let mut linear = vec![0.0f32; BATCH_SIZE];
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            // u8 → f32
            simd::srgb_u8_to_linear_batch(&lut8, &u8_data, &mut linear);
            // f32 → u8
            simd::linear_to_srgb_u8_batch(&linear, &mut output);
            black_box(&output);
        })
    });

    group.bench_function("u8_f32_u8/lut8_scalar_powf", |b| {
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in u8_data.iter().zip(output.iter_mut()) {
                let linear = lut8.lookup(*i as usize);
                *o = (linear_to_srgb(linear) * 255.0 + 0.5) as u8;
            }
            black_box(&output);
        })
    });

    group.bench_function("u8_f32_u8/lut8_lut12", |b| {
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in u8_data.iter().zip(output.iter_mut()) {
                let linear = lut8.lookup(*i as usize);
                *o = (lut_interp_linear_float(linear, encode12.as_slice()) * 255.0 + 0.5) as u8;
            }
            black_box(&output);
        })
    });

    group.bench_function("u8_f32_u8/imageflow_lut_fastpow", |b| {
        let iflow_lut = imageflow::SrgbToLinearLut::new();
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in u8_data.iter().zip(output.iter_mut()) {
                let linear = iflow_lut.lookup(*i);
                *o = imageflow::linear_to_srgb_u8_fastpow(linear);
            }
            black_box(&output);
        })
    });

    group.bench_function("u8_f32_u8/imageflow_lut_lut16k", |b| {
        let iflow_lut = imageflow::SrgbToLinearLut::new();
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in u8_data.iter().zip(output.iter_mut()) {
                let linear = iflow_lut.lookup(*i);
                *o = imageflow::linear_to_srgb_lut(linear);
            }
            black_box(&output);
        })
    });

    // === u16 → f32 → u16 ===

    group.bench_function("u16_f32_u16/lut16_scalar_powf", |b| {
        let mut output = vec![0u16; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in u16_data.iter().zip(output.iter_mut()) {
                let linear = lut16.lookup(*i as usize);
                *o = (linear_to_srgb(linear) * 65535.0 + 0.5) as u16;
            }
            black_box(&output);
        })
    });

    group.bench_function("u16_f32_u16/lut16_lut16", |b| {
        let mut output = vec![0u16; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in u16_data.iter().zip(output.iter_mut()) {
                let linear = lut16.lookup(*i as usize);
                *o = (lut_interp_linear_float(linear, encode16.as_slice()) * 65535.0 + 0.5) as u16;
            }
            black_box(&output);
        })
    });

    group.bench_function("u16_f32_u16/scalar_powf_both", |b| {
        let mut output = vec![0u16; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in u16_data.iter().zip(output.iter_mut()) {
                let linear = srgb_to_linear(*i as f32 / 65535.0);
                *o = (linear_to_srgb(linear) * 65535.0 + 0.5) as u16;
            }
            black_box(&output);
        })
    });

    group.finish();
}

// ============================================================================
// Scaling Benchmarks (different batch sizes)
// ============================================================================

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");

    let sizes = [100, 1000, 10000, 100000];

    for size in sizes {
        let f32_data: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
        let f32x8_data: Vec<f32x8> = f32_data
            .chunks(8)
            .map(|chunk| {
                let mut arr = [0.0f32; 8];
                arr[..chunk.len()].copy_from_slice(chunk);
                f32x8::from(arr)
            })
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("simd_s2l", size),
            &f32x8_data,
            |b, data| {
                let mut output = data.clone();
                b.iter(|| {
                    simd::srgb_to_linear_x8_slice(&mut output);
                    black_box(&output);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar_s2l", size),
            &f32_data,
            |b, data| {
                let mut output = data.clone();
                b.iter(|| {
                    for v in output.iter_mut() {
                        *v = srgb_to_linear(*v);
                    }
                    black_box(&output);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_srgb_to_linear,
    bench_linear_to_srgb,
    bench_roundtrip,
    bench_scaling,
);

criterion_main!(benches);
