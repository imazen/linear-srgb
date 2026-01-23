//! Comprehensive benchmarks for sRGB conversion methods.
//!
//! Tests all combinations of input/output types (u8, u16, f32) across implementations.

#![allow(deprecated)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
#[cfg(feature = "alt")]
use linear_srgb::alt::imageflow;
use linear_srgb::lut::{
    EncodeTable12, EncodeTable16, LinearTable8, LinearTable12, LinearTable16, SrgbConverter,
    lut_interp_linear_float,
};
#[cfg(feature = "mage")]
use linear_srgb::mage;
use linear_srgb::scalar::{linear_to_srgb, srgb_to_linear};
use linear_srgb::simd;
use std::hint::black_box;
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
    #[cfg(feature = "alt")]
    let imageflow_lut = imageflow::SrgbToLinearLut::new();
    let f32_data = create_f32_srgb();
    let u8_data = create_u8_srgb();
    let u16_data = create_u16_srgb();
    let _f32x8_data = create_f32x8_srgb();

    // === f32 → f32 ===

    group.bench_function("f32_f32/simd_dirty_pow", |b| {
        let mut output = f32_data.clone();
        b.iter(|| {
            simd::srgb_to_linear_slice(&mut output);
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

    #[cfg(feature = "alt")]
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

    group.bench_function("u8_f32/simd_lut8_slice", |b| {
        let mut output = vec![0.0f32; BATCH_SIZE];
        b.iter(|| {
            simd::srgb_u8_to_linear_slice(black_box(&u8_data), &mut output);
            black_box(&output);
        })
    });

    #[cfg(feature = "alt")]
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
    let _f32x8_linear = create_f32x8_linear();

    // Create u8-derived linear values for fair u8 output comparison
    let u8_srgb = create_u8_srgb();
    let linear_from_u8: Vec<f32> = u8_srgb.iter().map(|&v| lut8.lookup(v as usize)).collect();

    // === f32 → f32 ===

    group.bench_function("f32_f32/simd_dirty_pow", |b| {
        let mut output = f32_linear.clone();
        b.iter(|| {
            simd::linear_to_srgb_slice(&mut output);
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

    #[cfg(feature = "alt")]
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
            simd::linear_to_srgb_u8_slice(black_box(&linear_from_u8), &mut output);
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

    #[cfg(feature = "alt")]
    group.bench_function("f32_u8/imageflow_fastpow", |b| {
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            for (i, o) in linear_from_u8.iter().zip(output.iter_mut()) {
                *o = imageflow::linear_to_srgb_u8_fastpow(*i);
            }
            black_box(&output);
        })
    });

    #[cfg(feature = "alt")]
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

    group.bench_function("u8_f32_u8/simd_lut_dirty_pow", |b| {
        let mut linear = vec![0.0f32; BATCH_SIZE];
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            // u8 → f32
            simd::srgb_u8_to_linear_slice(&u8_data, &mut linear);
            // f32 → u8
            simd::linear_to_srgb_u8_slice(&linear, &mut output);
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

    #[cfg(feature = "alt")]
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

    #[cfg(feature = "alt")]
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

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("simd_s2l", size), &f32_data, |b, data| {
            let mut output = data.clone();
            b.iter(|| {
                simd::srgb_to_linear_slice(&mut output);
                black_box(&output);
            })
        });

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

// ============================================================================
// Dispatch Overhead Benchmarks (small sizes to measure dispatch cost)
// ============================================================================

fn bench_dispatch_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("dispatch_overhead");

    // Small sizes where dispatch overhead matters most
    let sizes = [8, 16, 32, 64, 128, 256, 512, 1024];

    for size in sizes {
        let f32_data: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();

        group.throughput(Throughput::Elements(size as u64));

        // === sRGB → Linear (LLVM optimizes powf(2.4) well, so scalar often wins) ===

        // Slice function: dispatch once, inline x8 inside
        group.bench_with_input(
            BenchmarkId::new("s2l_slice_dispatch_once", size),
            &f32_data,
            |b, data| {
                let mut output = data.clone();
                b.iter(|| {
                    simd::srgb_to_linear_slice(&mut output);
                    black_box(&output);
                })
            },
        );

        // Dispatch per chunk: worst case, dispatch every 8 elements
        group.bench_with_input(
            BenchmarkId::new("s2l_dispatch_per_chunk", size),
            &f32_data,
            |b, data| {
                let mut output = data.clone();
                b.iter(|| {
                    for chunk in output.chunks_exact_mut(8) {
                        let v = f32x8::from([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        let result = simd::srgb_to_linear_x8_dispatch(v);
                        let arr: [f32; 8] = result.into();
                        chunk.copy_from_slice(&arr);
                    }
                    black_box(&output);
                })
            },
        );

        // Inline x8 (no dispatch, called directly - baseline SIMD cost)
        group.bench_with_input(
            BenchmarkId::new("s2l_inline_no_dispatch", size),
            &f32_data,
            |b, data| {
                let mut output = data.clone();
                b.iter(|| {
                    for chunk in output.chunks_exact_mut(8) {
                        let v = f32x8::from([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        let result = simd::srgb_to_linear_x8_inline(v);
                        let arr: [f32; 8] = result.into();
                        chunk.copy_from_slice(&arr);
                    }
                    black_box(&output);
                })
            },
        );

        // Pure scalar (no SIMD, no dispatch)
        group.bench_with_input(
            BenchmarkId::new("s2l_scalar", size),
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

        // === Linear → sRGB (SIMD typically wins here) ===

        // Convert to linear first for l2s tests
        let linear_data: Vec<f32> = f32_data.iter().map(|&v| srgb_to_linear(v)).collect();

        // Slice function: dispatch once
        group.bench_with_input(
            BenchmarkId::new("l2s_slice_dispatch_once", size),
            &linear_data,
            |b, data| {
                let mut output = data.clone();
                b.iter(|| {
                    simd::linear_to_srgb_slice(&mut output);
                    black_box(&output);
                })
            },
        );

        // Dispatch per chunk
        group.bench_with_input(
            BenchmarkId::new("l2s_dispatch_per_chunk", size),
            &linear_data,
            |b, data| {
                let mut output = data.clone();
                b.iter(|| {
                    for chunk in output.chunks_exact_mut(8) {
                        let v = f32x8::from([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        let result = simd::linear_to_srgb_x8_dispatch(v);
                        let arr: [f32; 8] = result.into();
                        chunk.copy_from_slice(&arr);
                    }
                    black_box(&output);
                })
            },
        );

        // Inline x8 (no dispatch)
        group.bench_with_input(
            BenchmarkId::new("l2s_inline_no_dispatch", size),
            &linear_data,
            |b, data| {
                let mut output = data.clone();
                b.iter(|| {
                    for chunk in output.chunks_exact_mut(8) {
                        let v = f32x8::from([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        let result = simd::linear_to_srgb_x8_inline(v);
                        let arr: [f32; 8] = result.into();
                        chunk.copy_from_slice(&arr);
                    }
                    black_box(&output);
                })
            },
        );

        // Pure scalar
        group.bench_with_input(
            BenchmarkId::new("l2s_scalar", size),
            &linear_data,
            |b, data| {
                let mut output = data.clone();
                b.iter(|| {
                    for v in output.iter_mut() {
                        *v = linear_to_srgb(*v);
                    }
                    black_box(&output);
                })
            },
        );

        // === LUT-based approaches (interpolated for f32→f32) ===

        // LUT12 for sRGB → Linear (interpolated)
        group.bench_with_input(
            BenchmarkId::new("s2l_lut12_interp", size),
            &f32_data,
            |b, data| {
                let lut = LinearTable12::new();
                let mut output = data.clone();
                b.iter(|| {
                    for v in output.iter_mut() {
                        *v = lut_interp_linear_float(*v, lut.as_slice());
                    }
                    black_box(&output);
                })
            },
        );

        // LUT12 for Linear → sRGB (interpolated)
        group.bench_with_input(
            BenchmarkId::new("l2s_lut12_interp", size),
            &linear_data,
            |b, data| {
                let lut = EncodeTable12::new();
                let mut output = data.clone();
                b.iter(|| {
                    for v in output.iter_mut() {
                        *v = lut_interp_linear_float(*v, lut.as_slice());
                    }
                    black_box(&output);
                })
            },
        );

        // === u8 input (direct LUT lookup - no interpolation needed) ===

        let u8_data: Vec<u8> = (0..size).map(|i| (i * 255 / size) as u8).collect();
        let converter = SrgbConverter::new();

        // u8→f32 via direct LUT lookup (const table)
        group.bench_with_input(
            BenchmarkId::new("s2l_u8_lut8_direct", size),
            &u8_data,
            |b, data| {
                let mut output = vec![0.0f32; data.len()];
                b.iter(|| {
                    for (i, o) in data.iter().zip(output.iter_mut()) {
                        *o = converter.srgb_u8_to_linear(*i);
                    }
                    black_box(&output);
                })
            },
        );

        // u8→f32 via scalar powf
        group.bench_with_input(
            BenchmarkId::new("s2l_u8_scalar", size),
            &u8_data,
            |b, data| {
                let mut output = vec![0.0f32; data.len()];
                b.iter(|| {
                    for (i, o) in data.iter().zip(output.iter_mut()) {
                        *o = srgb_to_linear(*i as f32 / 255.0);
                    }
                    black_box(&output);
                })
            },
        );

        // u8→f32 via slice function (uses static LUT, no SIMD dispatch)
        // Note: srgb_u8_to_linear_slice uses LUT lookups, NOT SIMD powf
        group.bench_with_input(
            BenchmarkId::new("s2l_u8_lut_slice", size),
            &u8_data,
            |b, data| {
                let mut output = vec![0.0f32; data.len()];
                b.iter(|| {
                    simd::srgb_u8_to_linear_slice(data, &mut output);
                    black_box(&output);
                })
            },
        );

        // === f32→u8 output ===

        // f32→u8 via LUT interp + quantize (SrgbConverter)
        group.bench_with_input(
            BenchmarkId::new("l2s_u8_lut12", size),
            &linear_data,
            |b, data| {
                let mut output = vec![0u8; data.len()];
                b.iter(|| {
                    for (i, o) in data.iter().zip(output.iter_mut()) {
                        *o = converter.linear_to_srgb_u8(*i);
                    }
                    black_box(&output);
                })
            },
        );

        // f32→u8 via scalar powf
        group.bench_with_input(
            BenchmarkId::new("l2s_u8_scalar", size),
            &linear_data,
            |b, data| {
                let mut output = vec![0u8; data.len()];
                b.iter(|| {
                    for (i, o) in data.iter().zip(output.iter_mut()) {
                        *o = (linear_to_srgb(*i) * 255.0 + 0.5) as u8;
                    }
                    black_box(&output);
                })
            },
        );

        // f32→u8 via SIMD slice with dispatch (multiversed)
        group.bench_with_input(
            BenchmarkId::new("l2s_u8_simd_dispatch_slice", size),
            &linear_data,
            |b, data| {
                let mut output = vec![0u8; data.len()];
                b.iter(|| {
                    simd::linear_to_srgb_u8_slice(data, &mut output);
                    black_box(&output);
                })
            },
        );

        // f32→u8 via x8 dispatch per chunk (worst case dispatch overhead)
        group.bench_with_input(
            BenchmarkId::new("l2s_u8_x8_dispatch_per_chunk", size),
            &linear_data,
            |b, data| {
                let mut output = vec![0u8; data.len()];
                b.iter(|| {
                    for (inp, out) in data.chunks_exact(8).zip(output.chunks_exact_mut(8)) {
                        let v = f32x8::from([
                            inp[0], inp[1], inp[2], inp[3], inp[4], inp[5], inp[6], inp[7],
                        ]);
                        let result = simd::linear_to_srgb_u8_x8_dispatch(v);
                        out.copy_from_slice(&result);
                    }
                    black_box(&output);
                })
            },
        );

        // f32→u8 via x8 inline per chunk (no dispatch - baseline SIMD)
        group.bench_with_input(
            BenchmarkId::new("l2s_u8_x8_inline_per_chunk", size),
            &linear_data,
            |b, data| {
                let mut output = vec![0u8; data.len()];
                b.iter(|| {
                    for (inp, out) in data.chunks_exact(8).zip(output.chunks_exact_mut(8)) {
                        let v = f32x8::from([
                            inp[0], inp[1], inp[2], inp[3], inp[4], inp[5], inp[6], inp[7],
                        ]);
                        let result = simd::linear_to_srgb_u8_x8_inline(v);
                        out.copy_from_slice(&result);
                    }
                    black_box(&output);
                })
            },
        );

        // === u16 input/output ===

        let u16_data: Vec<u16> = (0..size).map(|i| (i * 65535 / size) as u16).collect();
        let lut16 = LinearTable16::new();
        let encode16 = EncodeTable16::new();

        // u16→f32 via direct LUT16 lookup
        group.bench_with_input(
            BenchmarkId::new("s2l_u16_lut16_direct", size),
            &u16_data,
            |b, data| {
                let mut output = vec![0.0f32; data.len()];
                b.iter(|| {
                    for (i, o) in data.iter().zip(output.iter_mut()) {
                        *o = lut16.lookup(*i as usize);
                    }
                    black_box(&output);
                })
            },
        );

        // u16→f32 via scalar powf
        group.bench_with_input(
            BenchmarkId::new("s2l_u16_scalar", size),
            &u16_data,
            |b, data| {
                let mut output = vec![0.0f32; data.len()];
                b.iter(|| {
                    for (i, o) in data.iter().zip(output.iter_mut()) {
                        *o = srgb_to_linear(*i as f32 / 65535.0);
                    }
                    black_box(&output);
                })
            },
        );

        // Linear f32 data for u16 output tests
        let linear_from_u16: Vec<f32> = u16_data
            .iter()
            .map(|&v| srgb_to_linear(v as f32 / 65535.0))
            .collect();

        // f32→u16 via LUT16 interp
        group.bench_with_input(
            BenchmarkId::new("l2s_u16_lut16_interp", size),
            &linear_from_u16,
            |b, data| {
                let mut output = vec![0u16; data.len()];
                b.iter(|| {
                    for (i, o) in data.iter().zip(output.iter_mut()) {
                        *o = (lut_interp_linear_float(*i, encode16.as_slice()) * 65535.0 + 0.5)
                            as u16;
                    }
                    black_box(&output);
                })
            },
        );

        // f32→u16 via scalar powf
        group.bench_with_input(
            BenchmarkId::new("l2s_u16_scalar", size),
            &linear_from_u16,
            |b, data| {
                let mut output = vec![0u16; data.len()];
                b.iter(|| {
                    for (i, o) in data.iter().zip(output.iter_mut()) {
                        *o = (linear_to_srgb(*i) * 65535.0 + 0.5) as u16;
                    }
                    black_box(&output);
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Mage Module Benchmarks (archmage native SIMD with true FMA)
// ============================================================================

#[cfg(feature = "mage")]
fn bench_mage(c: &mut Criterion) {
    use mage::{SimdToken, Token};

    let Some(token) = Token::try_new() else {
        eprintln!("Skipping mage benchmarks: AVX2+FMA not available");
        return;
    };

    let mut group = c.benchmark_group("mage");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let f32_srgb = create_f32_srgb();
    let f32_linear = create_f32_linear();

    // === f32 slice conversion (primary use case) ===

    group.bench_function("f32_f32/srgb_to_linear_slice", |b| {
        let mut output = f32_srgb.clone();
        b.iter(|| {
            mage::srgb_to_linear_slice(token, &mut output);
            black_box(&output);
        })
    });

    group.bench_function("f32_f32/linear_to_srgb_slice", |b| {
        let mut output = f32_linear.clone();
        b.iter(|| {
            mage::linear_to_srgb_slice(token, &mut output);
            black_box(&output);
        })
    });

    // === Compare with simd module (wide-based) ===

    group.bench_function("f32_f32/simd_srgb_to_linear_slice", |b| {
        let mut output = f32_srgb.clone();
        b.iter(|| {
            simd::srgb_to_linear_slice(&mut output);
            black_box(&output);
        })
    });

    group.bench_function("f32_f32/simd_linear_to_srgb_slice", |b| {
        let mut output = f32_linear.clone();
        b.iter(|| {
            simd::linear_to_srgb_slice(&mut output);
            black_box(&output);
        })
    });

    // === f32 → u8 ===

    let lut8 = LinearTable8::new();
    let u8_srgb = create_u8_srgb();
    let linear_from_u8: Vec<f32> = u8_srgb.iter().map(|&v| lut8.lookup(v as usize)).collect();

    group.bench_function("f32_u8/mage_linear_to_srgb_u8_slice", |b| {
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            mage::linear_to_srgb_u8_slice(token, black_box(&linear_from_u8), &mut output);
            black_box(&output);
        })
    });

    group.bench_function("f32_u8/simd_linear_to_srgb_u8_slice", |b| {
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            simd::linear_to_srgb_u8_slice(black_box(&linear_from_u8), &mut output);
            black_box(&output);
        })
    });

    // === Gamma conversion ===

    group.bench_function("gamma/mage_to_linear_2.2", |b| {
        let mut output = f32_srgb.clone();
        b.iter(|| {
            mage::gamma_to_linear_slice(token, &mut output, 2.2);
            black_box(&output);
        })
    });

    group.bench_function("gamma/simd_to_linear_2.2", |b| {
        let mut output = f32_srgb.clone();
        b.iter(|| {
            simd::gamma_to_linear_slice(&mut output, 2.2);
            black_box(&output);
        })
    });

    group.finish();
}

#[cfg(not(feature = "mage"))]
fn bench_mage(_c: &mut Criterion) {
    // No-op when mage feature is disabled
}

criterion_group!(
    benches,
    bench_srgb_to_linear,
    bench_linear_to_srgb,
    bench_roundtrip,
    bench_scaling,
    bench_dispatch_overhead,
    bench_mage,
);

criterion_main!(benches);
