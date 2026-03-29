#![allow(deprecated)] // LinearTable16/EncodeTable16 benchmarks kept for comparison
//! Comprehensive benchmarks for sRGB conversion methods.
//!
//! Tests all combinations of input/output types (u8, u16, f32) across implementations.

#[cfg(feature = "alt")]
use linear_srgb::alt::imageflow;
use linear_srgb::default;
use linear_srgb::lut::{
    EncodeTable12, EncodeTable16, LinearTable8, LinearTable12, LinearTable16, SrgbConverter,
    lut_interp_linear_float,
};
use linear_srgb::precise::{linear_to_srgb, srgb_to_linear};
use std::hint::black_box;
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

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

    // === f32 → f32 ===

    group.bench_function("f32_f32/simd_slice", |b| {
        let mut output = f32_data.clone();
        b.iter(|| {
            default::srgb_to_linear_slice(&mut output);
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
            default::srgb_u8_to_linear_slice(black_box(&u8_data), &mut output);
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

    // Create u8-derived linear values for fair u8 output comparison
    let u8_srgb = create_u8_srgb();
    let linear_from_u8: Vec<f32> = u8_srgb.iter().map(|&v| lut8.lookup(v as usize)).collect();

    // === f32 → f32 ===

    group.bench_function("f32_f32/simd_slice", |b| {
        let mut output = f32_linear.clone();
        b.iter(|| {
            default::linear_to_srgb_slice(&mut output);
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

    group.bench_function("f32_u8/simd_slice", |b| {
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            default::linear_to_srgb_u8_slice(black_box(&linear_from_u8), &mut output);
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

    group.bench_function("f32_u16/simd_slice", |b| {
        let mut output = vec![0u16; BATCH_SIZE];
        b.iter(|| {
            default::linear_to_srgb_u16_slice(&f32_linear, &mut output);
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

    group.bench_function("u8_f32_u8/simd_lut_slice", |b| {
        let mut linear = vec![0.0f32; BATCH_SIZE];
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            // u8 → f32
            default::srgb_u8_to_linear_slice(&u8_data, &mut linear);
            // f32 → u8
            default::linear_to_srgb_u8_slice(&linear, &mut output);
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
                default::srgb_to_linear_slice(&mut output);
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

        // === sRGB → Linear ===

        // Slice function: dispatch once, process entire slice
        group.bench_with_input(BenchmarkId::new("s2l_slice", size), &f32_data, |b, data| {
            let mut output = data.clone();
            b.iter(|| {
                default::srgb_to_linear_slice(&mut output);
                black_box(&output);
            })
        });

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

        // === Linear → sRGB ===

        let linear_data: Vec<f32> = f32_data.iter().map(|&v| srgb_to_linear(v)).collect();

        // Slice function: dispatch once
        group.bench_with_input(
            BenchmarkId::new("l2s_slice", size),
            &linear_data,
            |b, data| {
                let mut output = data.clone();
                b.iter(|| {
                    default::linear_to_srgb_slice(&mut output);
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

        // === LUT-based approaches ===

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

        // === u8 input ===

        let u8_data: Vec<u8> = (0..size).map(|i| (i * 255 / size) as u8).collect();
        let converter = SrgbConverter::new();

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

        group.bench_with_input(
            BenchmarkId::new("s2l_u8_lut_slice", size),
            &u8_data,
            |b, data| {
                let mut output = vec![0.0f32; data.len()];
                b.iter(|| {
                    default::srgb_u8_to_linear_slice(data, &mut output);
                    black_box(&output);
                })
            },
        );

        // === f32→u8 output ===

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

        group.bench_with_input(
            BenchmarkId::new("l2s_u8_simd_slice", size),
            &linear_data,
            |b, data| {
                let mut output = vec![0u8; data.len()];
                b.iter(|| {
                    default::linear_to_srgb_u8_slice(data, &mut output);
                    black_box(&output);
                })
            },
        );

        // === u16 input/output ===

        let u16_data: Vec<u16> = (0..size).map(|i| (i * 65535 / size) as u16).collect();
        let lut16 = LinearTable16::new();
        let encode16 = EncodeTable16::new();

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

        let linear_from_u16: Vec<f32> = u16_data
            .iter()
            .map(|&v| srgb_to_linear(v as f32 / 65535.0))
            .collect();

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
// Tier Isolation Benchmarks
// ============================================================================
//
// Tests all incant!-dispatched slice functions at each token tier by using
// dangerously_disable_token_process_wide to force dispatch fallback.
// This verifies real-world perf through the actual dispatch path, not just
// direct rite calls.

fn bench_dispatched_at_tier(c: &mut Criterion, tier: &str) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        // First restore all tokens to a clean state
        let _ = archmage::X64V3Token::dangerously_disable_token_process_wide(false);
        let _ = archmage::X64V4Token::dangerously_disable_token_process_wide(false);

        match tier {
            "scalar" => {
                // Disable V3 (cascades to V4/V4x) to force scalar fallback
                if archmage::X64V3Token::dangerously_disable_token_process_wide(true).is_err() {
                    eprintln!(
                        "Cannot disable V3 (compile-time guaranteed). \
                         Build without -Ctarget-cpu=native or enable testable_dispatch."
                    );
                    return;
                }
            }
            "v3" => {
                // Disable V4 to isolate V3 (AVX2+FMA only)
                let _ = archmage::X64V4Token::dangerously_disable_token_process_wide(true);
                if archmage::X64V3Token::summon().is_none() {
                    eprintln!("V3 (AVX2+FMA) not available on this CPU. Skipping.");
                    return;
                }
            }
            "v4" => {
                // Leave everything enabled — V4 (AVX-512) takes priority
                if archmage::X64V4Token::summon().is_none() {
                    eprintln!("V4 (AVX-512) not available on this CPU. Skipping.");
                    return;
                }
            }
            _ => {}
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    if tier != "scalar" {
        eprintln!("Non-x86_64: only scalar tier available. Skipping {tier}.");
        return;
    }

    let group_name = format!("tier_{tier}");
    let mut group = c.benchmark_group(&group_name);
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let f32_srgb = create_f32_srgb();
    let f32_linear = create_f32_linear();
    let u8_data = create_u8_srgb();
    let u16_data = create_u16_srgb();
    let lut8 = LinearTable8::new();
    let linear_from_u8: Vec<f32> = u8_data.iter().map(|&v| lut8.lookup(v as usize)).collect();

    // --- Dispatched f32 in-place (incant! → v3 or scalar) ---

    group.bench_function("srgb_to_linear_slice", |b| {
        let mut buf = f32_srgb.clone();
        b.iter(|| {
            buf.copy_from_slice(&f32_srgb);
            default::srgb_to_linear_slice(black_box(&mut buf));
        })
    });

    group.bench_function("linear_to_srgb_slice", |b| {
        let mut buf = f32_linear.clone();
        b.iter(|| {
            buf.copy_from_slice(&f32_linear);
            default::linear_to_srgb_slice(black_box(&mut buf));
        })
    });

    group.bench_function("gamma_to_linear_slice_2.2", |b| {
        let mut buf = f32_srgb.clone();
        b.iter(|| {
            buf.copy_from_slice(&f32_srgb);
            default::gamma_to_linear_slice(black_box(&mut buf), 2.2);
        })
    });

    group.bench_function("linear_to_gamma_slice_2.2", |b| {
        let mut buf = f32_linear.clone();
        b.iter(|| {
            buf.copy_from_slice(&f32_linear);
            default::linear_to_gamma_slice(black_box(&mut buf), 2.2);
        })
    });

    // --- LUT-based (no dispatch, should be identical across tiers) ---

    group.bench_function("srgb_u8_to_linear_slice", |b| {
        let mut output = vec![0.0f32; BATCH_SIZE];
        b.iter(|| {
            default::srgb_u8_to_linear_slice(black_box(&u8_data), &mut output);
            black_box(&output);
        })
    });

    group.bench_function("linear_to_srgb_u8_slice", |b| {
        let mut output = vec![0u8; BATCH_SIZE];
        b.iter(|| {
            default::linear_to_srgb_u8_slice(black_box(&linear_from_u8), &mut output);
            black_box(&output);
        })
    });

    group.bench_function("srgb_u16_to_linear_slice", |b| {
        let mut output = vec![0.0f32; BATCH_SIZE];
        b.iter(|| {
            default::srgb_u16_to_linear_slice(black_box(&u16_data), &mut output);
            black_box(&output);
        })
    });

    group.bench_function("linear_to_srgb_u16_slice", |b| {
        let mut output = vec![0u16; BATCH_SIZE];
        b.iter(|| {
            default::linear_to_srgb_u16_slice(black_box(&f32_linear), &mut output);
            black_box(&output);
        })
    });

    group.finish();

    // Restore all tokens
    #[cfg(target_arch = "x86_64")]
    {
        let _ = archmage::X64V3Token::dangerously_disable_token_process_wide(false);
        let _ = archmage::X64V4Token::dangerously_disable_token_process_wide(false);
    }
}

fn bench_tier_v4(c: &mut Criterion) {
    bench_dispatched_at_tier(c, "v4");
}

fn bench_tier_v3(c: &mut Criterion) {
    bench_dispatched_at_tier(c, "v3");
}

fn bench_tier_scalar(c: &mut Criterion) {
    bench_dispatched_at_tier(c, "scalar");
}

criterion_group!(
    benches,
    bench_srgb_to_linear,
    bench_linear_to_srgb,
    bench_roundtrip,
    bench_scaling,
    bench_dispatch_overhead,
    bench_tier_v4,
    bench_tier_v3,
    bench_tier_scalar,
);

criterion_main!(benches);
