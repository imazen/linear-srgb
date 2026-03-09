//! Benchmark: RGBA alpha-preservation strategies for issue #1.
//!
//! Compares approaches for converting RGBA f32 slices while preserving alpha:
//! 1. save_restore_vec — Save all alphas to Vec, convert entire slice, restore
//! 2. save_restore_chunked — Process in 8-element chunks, save 2 alphas each, restore
//! 3. inverse_fixup — Convert everything, then undo alpha with inverse function
//! 4. baseline — Plain srgb_to_linear_slice (no alpha handling, for reference)

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use linear_srgb::default;
use std::hint::black_box;

// ============================================================================
// Test data: RGBA f32, various pixel counts
// ============================================================================

fn create_rgba_srgb(num_pixels: usize) -> Vec<f32> {
    (0..num_pixels * 4)
        .map(|i| {
            if i % 4 == 3 {
                // Alpha: realistic distribution (mostly opaque, some semi-transparent)
                match (i / 4) % 8 {
                    0..=5 => 1.0,
                    6 => 0.5,
                    _ => 0.75,
                }
            } else {
                // RGB: spread across 0..1
                (i % 256) as f32 / 255.0
            }
        })
        .collect()
}

fn create_rgba_linear(num_pixels: usize) -> Vec<f32> {
    let mut data = create_rgba_srgb(num_pixels);
    // Convert RGB channels to linear, leave alpha as-is
    for pixel in data.chunks_exact_mut(4) {
        pixel[0] = default::srgb_to_linear(pixel[0]);
        pixel[1] = default::srgb_to_linear(pixel[1]);
        pixel[2] = default::srgb_to_linear(pixel[2]);
        // pixel[3] stays as linear alpha
    }
    data
}

// ============================================================================
// Approach 1: Save alphas to Vec, bulk convert, restore
// ============================================================================

fn srgb_to_linear_rgba_save_vec(values: &mut [f32]) {
    let alphas: Vec<f32> = values.iter().skip(3).step_by(4).copied().collect();
    default::srgb_to_linear_slice(values);
    for (i, &a) in alphas.iter().enumerate() {
        values[i * 4 + 3] = a;
    }
}

fn linear_to_srgb_rgba_save_vec(values: &mut [f32]) {
    let alphas: Vec<f32> = values.iter().skip(3).step_by(4).copied().collect();
    default::linear_to_srgb_slice(values);
    for (i, &a) in alphas.iter().enumerate() {
        values[i * 4 + 3] = a;
    }
}

fn srgb_to_linear_rgba_chunked_slice(values: &mut [f32]) {
    for pixel in values.chunks_exact_mut(4) {
        let a = pixel[3];
        default::srgb_to_linear_slice(pixel);
        pixel[3] = a;
    }
}

fn linear_to_srgb_rgba_chunked_slice(values: &mut [f32]) {
    for pixel in values.chunks_exact_mut(4) {
        let a = pixel[3];
        default::linear_to_srgb_slice(pixel);
        pixel[3] = a;
    }
}

// ============================================================================
// Approach 3: Convert everything, then undo alpha via inverse function
// ============================================================================

fn srgb_to_linear_rgba_inverse(values: &mut [f32]) {
    default::srgb_to_linear_slice(values);
    // Undo: alpha was srgb_to_linear'd, apply inverse to get back ~original
    for i in (3..values.len()).step_by(4) {
        values[i] = default::linear_to_srgb(values[i]);
    }
}

fn linear_to_srgb_rgba_inverse(values: &mut [f32]) {
    default::linear_to_srgb_slice(values);
    for i in (3..values.len()).step_by(4) {
        values[i] = default::srgb_to_linear(values[i]);
    }
}

// ============================================================================
// Approach 4: Only convert RGB channels, skip alpha entirely (stride-3 scalar)
// ============================================================================

fn srgb_to_linear_rgba_rgb_only(values: &mut [f32]) {
    for pixel in values.chunks_exact_mut(4) {
        pixel[0] = default::srgb_to_linear(pixel[0]);
        pixel[1] = default::srgb_to_linear(pixel[1]);
        pixel[2] = default::srgb_to_linear(pixel[2]);
    }
}

fn linear_to_srgb_rgba_rgb_only(values: &mut [f32]) {
    for pixel in values.chunks_exact_mut(4) {
        pixel[0] = default::linear_to_srgb(pixel[0]);
        pixel[1] = default::linear_to_srgb(pixel[1]);
        pixel[2] = default::linear_to_srgb(pixel[2]);
    }
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_rgba_srgb_to_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgba_s2l");

    let pixel_counts = [64, 256, 1024, 4096, 16384];

    for &num_pixels in &pixel_counts {
        let data = create_rgba_srgb(num_pixels);
        group.throughput(Throughput::Elements(num_pixels as u64));

        // Baseline: no alpha handling
        group.bench_with_input(
            BenchmarkId::new("baseline", num_pixels),
            &data,
            |b, data| {
                let mut buf = data.clone();
                b.iter(|| {
                    buf.copy_from_slice(data);
                    default::srgb_to_linear_slice(black_box(&mut buf));
                })
            },
        );

        // Approach 1: Save to Vec + restore
        group.bench_with_input(
            BenchmarkId::new("save_vec", num_pixels),
            &data,
            |b, data| {
                let mut buf = data.clone();
                b.iter(|| {
                    buf.copy_from_slice(data);
                    srgb_to_linear_rgba_save_vec(black_box(&mut buf));
                })
            },
        );

        // Approach 5: Integrated SIMD (save/restore alpha inside chunk loop)
        group.bench_with_input(
            BenchmarkId::new("integrated_simd", num_pixels),
            &data,
            |b, data| {
                let mut buf = data.clone();
                b.iter(|| {
                    buf.copy_from_slice(data);
                    default::srgb_to_linear_rgba_slice(black_box(&mut buf));
                })
            },
        );

        // Approach 2: Per-pixel via slice dispatch (simulates overhead of many small dispatches)
        group.bench_with_input(
            BenchmarkId::new("per_pixel_slice", num_pixels),
            &data,
            |b, data| {
                let mut buf = data.clone();
                b.iter(|| {
                    buf.copy_from_slice(data);
                    srgb_to_linear_rgba_chunked_slice(black_box(&mut buf));
                })
            },
        );

        // Approach 3: Inverse fixup (convert all, undo alpha)
        group.bench_with_input(
            BenchmarkId::new("inverse_fixup", num_pixels),
            &data,
            |b, data| {
                let mut buf = data.clone();
                b.iter(|| {
                    buf.copy_from_slice(data);
                    srgb_to_linear_rgba_inverse(black_box(&mut buf));
                })
            },
        );

        // Approach 4: RGB-only scalar (no SIMD)
        group.bench_with_input(
            BenchmarkId::new("rgb_only_scalar", num_pixels),
            &data,
            |b, data| {
                let mut buf = data.clone();
                b.iter(|| {
                    buf.copy_from_slice(data);
                    srgb_to_linear_rgba_rgb_only(black_box(&mut buf));
                })
            },
        );
    }

    group.finish();
}

fn bench_rgba_linear_to_srgb(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgba_l2s");

    let pixel_counts = [64, 256, 1024, 4096, 16384];

    for &num_pixels in &pixel_counts {
        let data = create_rgba_linear(num_pixels);
        group.throughput(Throughput::Elements(num_pixels as u64));

        // Baseline
        group.bench_with_input(
            BenchmarkId::new("baseline", num_pixels),
            &data,
            |b, data| {
                let mut buf = data.clone();
                b.iter(|| {
                    buf.copy_from_slice(data);
                    default::linear_to_srgb_slice(black_box(&mut buf));
                })
            },
        );

        // Save/restore Vec
        group.bench_with_input(
            BenchmarkId::new("save_vec", num_pixels),
            &data,
            |b, data| {
                let mut buf = data.clone();
                b.iter(|| {
                    buf.copy_from_slice(data);
                    linear_to_srgb_rgba_save_vec(black_box(&mut buf));
                })
            },
        );

        // Integrated SIMD
        group.bench_with_input(
            BenchmarkId::new("integrated_simd", num_pixels),
            &data,
            |b, data| {
                let mut buf = data.clone();
                b.iter(|| {
                    buf.copy_from_slice(data);
                    default::linear_to_srgb_rgba_slice(black_box(&mut buf));
                })
            },
        );

        // Per-pixel slice dispatch
        group.bench_with_input(
            BenchmarkId::new("per_pixel_slice", num_pixels),
            &data,
            |b, data| {
                let mut buf = data.clone();
                b.iter(|| {
                    buf.copy_from_slice(data);
                    linear_to_srgb_rgba_chunked_slice(black_box(&mut buf));
                })
            },
        );

        // Inverse fixup
        group.bench_with_input(
            BenchmarkId::new("inverse_fixup", num_pixels),
            &data,
            |b, data| {
                let mut buf = data.clone();
                b.iter(|| {
                    buf.copy_from_slice(data);
                    linear_to_srgb_rgba_inverse(black_box(&mut buf));
                })
            },
        );

        // RGB-only scalar
        group.bench_with_input(
            BenchmarkId::new("rgb_only_scalar", num_pixels),
            &data,
            |b, data| {
                let mut buf = data.clone();
                b.iter(|| {
                    buf.copy_from_slice(data);
                    linear_to_srgb_rgba_rgb_only(black_box(&mut buf));
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Fused premultiply vs separate (srgb_to_linear_rgba + manual premul loop)
// ============================================================================

fn premultiply_loop(values: &mut [f32]) {
    for pixel in values.chunks_exact_mut(4) {
        let a = pixel[3];
        pixel[0] *= a;
        pixel[1] *= a;
        pixel[2] *= a;
    }
}

fn unpremultiply_loop(values: &mut [f32]) {
    for pixel in values.chunks_exact_mut(4) {
        let a = pixel[3];
        if a > 0.0 {
            let inv_a = 1.0 / a;
            pixel[0] *= inv_a;
            pixel[1] *= inv_a;
            pixel[2] *= inv_a;
        }
    }
}

fn bench_premultiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("premultiply_s2l");

    let pixel_counts = [64, 256, 1024, 4096, 16384];

    for &num_pixels in &pixel_counts {
        let data = create_rgba_srgb(num_pixels);
        group.throughput(Throughput::Elements(num_pixels as u64));

        // Separate: srgb_to_linear_rgba_slice + premultiply loop
        group.bench_with_input(
            BenchmarkId::new("separate", num_pixels),
            &data,
            |b, data| {
                let mut buf = data.clone();
                b.iter(|| {
                    buf.copy_from_slice(data);
                    default::srgb_to_linear_rgba_slice(black_box(&mut buf));
                    premultiply_loop(black_box(&mut buf));
                })
            },
        );

        // Fused: single pass
        group.bench_with_input(BenchmarkId::new("fused", num_pixels), &data, |b, data| {
            let mut buf = data.clone();
            b.iter(|| {
                buf.copy_from_slice(data);
                default::srgb_to_linear_premultiply_rgba_slice(black_box(&mut buf));
            })
        });
    }
    group.finish();
}

fn bench_unpremultiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("unpremultiply_l2s");

    let pixel_counts = [64, 256, 1024, 4096, 16384];

    for &num_pixels in &pixel_counts {
        // Create premultiplied linear data
        let mut data = create_rgba_srgb(num_pixels);
        default::srgb_to_linear_premultiply_rgba_slice(&mut data);

        group.throughput(Throughput::Elements(num_pixels as u64));

        // Separate: unpremultiply loop + linear_to_srgb_rgba_slice
        group.bench_with_input(
            BenchmarkId::new("separate", num_pixels),
            &data,
            |b, data| {
                let mut buf = data.clone();
                b.iter(|| {
                    buf.copy_from_slice(data);
                    unpremultiply_loop(black_box(&mut buf));
                    default::linear_to_srgb_rgba_slice(black_box(&mut buf));
                })
            },
        );

        // Fused: single pass
        group.bench_with_input(BenchmarkId::new("fused", num_pixels), &data, |b, data| {
            let mut buf = data.clone();
            b.iter(|| {
                buf.copy_from_slice(data);
                default::unpremultiply_linear_to_srgb_rgba_slice(black_box(&mut buf));
            })
        });
    }
    group.finish();
}

// Quick accuracy check for the inverse_fixup approach
fn bench_inverse_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgba_inverse_accuracy");

    // Just report the max alpha error from inverse roundtrip
    let data = create_rgba_srgb(1024);
    let mut buf = data.clone();
    srgb_to_linear_rgba_inverse(&mut buf);

    let mut max_err: f32 = 0.0;
    for (i, (&orig, &conv)) in data.iter().zip(buf.iter()).enumerate() {
        if i % 4 == 3 {
            let err = (orig - conv).abs();
            max_err = max_err.max(err);
        }
    }
    eprintln!(
        "inverse_fixup s2l alpha max error: {:.2e} ({:.1} ULP at alpha=0.5)",
        max_err,
        max_err / f32::EPSILON
    );

    let data = create_rgba_linear(1024);
    let mut buf = data.clone();
    linear_to_srgb_rgba_inverse(&mut buf);

    let mut max_err: f32 = 0.0;
    for (i, (&orig, &conv)) in data.iter().zip(buf.iter()).enumerate() {
        if i % 4 == 3 {
            let err = (orig - conv).abs();
            max_err = max_err.max(err);
        }
    }
    eprintln!(
        "inverse_fixup l2s alpha max error: {:.2e} ({:.1} ULP at alpha=0.5)",
        max_err,
        max_err / f32::EPSILON
    );

    // Dummy benchmark so criterion is happy
    group.bench_function("noop", |b| b.iter(|| black_box(42)));
    group.finish();
}

criterion_group!(
    benches,
    bench_rgba_srgb_to_linear,
    bench_rgba_linear_to_srgb,
    bench_premultiply,
    bench_unpremultiply,
    bench_inverse_accuracy,
);
criterion_main!(benches);
