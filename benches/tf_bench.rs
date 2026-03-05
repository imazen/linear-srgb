//! Benchmarks for all transfer functions across tiers (scalar, x8, x16).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use linear_srgb::tf;
use std::hint::black_box;

const N: usize = 10000;

fn make_encoded() -> Vec<f32> {
    (0..N).map(|i| i as f32 / N as f32).collect()
}

fn make_linear() -> Vec<f32> {
    make_encoded()
        .iter()
        .map(|&v| tf::srgb_to_linear(v))
        .collect()
}

// =============================================================================
// Scalar
// =============================================================================

fn bench_tf_scalar(c: &mut Criterion) {
    let mut g = c.benchmark_group("tf_scalar");
    g.throughput(Throughput::Elements(N as u64));

    let encoded = make_encoded();
    let linear = make_linear();

    g.bench_function("srgb_to_linear", |b| {
        b.iter(|| {
            for &v in &encoded {
                black_box(tf::srgb_to_linear(v));
            }
        })
    });
    g.bench_function("linear_to_srgb", |b| {
        b.iter(|| {
            for &v in &linear {
                black_box(tf::linear_to_srgb(v));
            }
        })
    });
    g.bench_function("bt709_to_linear", |b| {
        b.iter(|| {
            for &v in &encoded {
                black_box(tf::bt709_to_linear(v));
            }
        })
    });
    g.bench_function("linear_to_bt709", |b| {
        b.iter(|| {
            for &v in &linear {
                black_box(tf::linear_to_bt709(v));
            }
        })
    });
    g.bench_function("pq_to_linear", |b| {
        b.iter(|| {
            for &v in &encoded {
                black_box(tf::pq_to_linear(v));
            }
        })
    });
    g.bench_function("linear_to_pq", |b| {
        b.iter(|| {
            for &v in &linear {
                black_box(tf::linear_to_pq(v));
            }
        })
    });
    g.bench_function("hlg_to_linear", |b| {
        b.iter(|| {
            for &v in &encoded {
                black_box(tf::hlg_to_linear(v));
            }
        })
    });
    g.bench_function("linear_to_hlg", |b| {
        b.iter(|| {
            for &v in &linear {
                black_box(tf::linear_to_hlg(v));
            }
        })
    });
    g.finish();
}

// =============================================================================
// x8 rites (AVX2+FMA)
// =============================================================================

#[cfg(target_arch = "x86_64")]
fn bench_tf_x8(c: &mut Criterion) {
    use archmage::SimdToken;
    let Some(token) = archmage::Desktop64::summon() else {
        eprintln!("AVX2+FMA not available, skipping x8 benchmarks");
        return;
    };

    let mut g = c.benchmark_group("tf_x8");
    g.throughput(Throughput::Elements(N as u64));

    let encoded = make_encoded();
    let linear = make_linear();

    macro_rules! bench_x8 {
        ($name:expr, $data:expr, $fn:path) => {
            g.bench_function($name, |b| {
                #[archmage::arcane]
                fn call_slice(token: archmage::Desktop64, values: &mut [f32]) {
                    $fn(token, values);
                }
                let mut buf = $data.clone();
                b.iter(|| {
                    buf.copy_from_slice(&$data);
                    call_slice(token, black_box(&mut buf));
                })
            });
        };
    }

    bench_x8!("srgb_to_linear", encoded, tf::rites_x8::srgb_to_linear_slice_v3);
    bench_x8!("linear_to_srgb", linear, tf::rites_x8::linear_to_srgb_slice_v3);
    bench_x8!("bt709_to_linear", encoded, tf::rites_x8::bt709_to_linear_slice_v3);
    bench_x8!("linear_to_bt709", linear, tf::rites_x8::linear_to_bt709_slice_v3);
    bench_x8!("pq_to_linear", encoded, tf::rites_x8::pq_to_linear_slice_v3);
    bench_x8!("linear_to_pq", linear, tf::rites_x8::linear_to_pq_slice_v3);
    bench_x8!("hlg_to_linear", encoded, tf::rites_x8::hlg_to_linear_slice_v3);
    bench_x8!("linear_to_hlg", linear, tf::rites_x8::linear_to_hlg_slice_v3);
    g.finish();
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_tf_x8(_c: &mut Criterion) {}

// =============================================================================
// x16 rites (AVX-512)
// =============================================================================

#[cfg(target_arch = "x86_64")]
fn bench_tf_x16(c: &mut Criterion) {
    use archmage::SimdToken;
    let Some(token) = archmage::Server64::summon() else {
        eprintln!("AVX-512 not available, skipping x16 benchmarks");
        return;
    };

    let mut g = c.benchmark_group("tf_x16");
    g.throughput(Throughput::Elements(N as u64));

    let encoded = make_encoded();
    let linear = make_linear();

    macro_rules! bench_x16 {
        ($name:expr, $data:expr, $fn:path) => {
            g.bench_function($name, |b| {
                #[archmage::arcane]
                fn call_slice(token: archmage::Server64, values: &mut [f32]) {
                    $fn(token, values);
                }
                let mut buf = $data.clone();
                b.iter(|| {
                    buf.copy_from_slice(&$data);
                    call_slice(token, black_box(&mut buf));
                })
            });
        };
    }

    bench_x16!("srgb_to_linear", encoded, tf::rites_x16::srgb_to_linear_slice_v4);
    bench_x16!("linear_to_srgb", linear, tf::rites_x16::linear_to_srgb_slice_v4);
    bench_x16!("bt709_to_linear", encoded, tf::rites_x16::bt709_to_linear_slice_v4);
    bench_x16!("linear_to_bt709", linear, tf::rites_x16::linear_to_bt709_slice_v4);
    bench_x16!("pq_to_linear", encoded, tf::rites_x16::pq_to_linear_slice_v4);
    bench_x16!("linear_to_pq", linear, tf::rites_x16::linear_to_pq_slice_v4);
    bench_x16!("hlg_to_linear", encoded, tf::rites_x16::hlg_to_linear_slice_v4);
    bench_x16!("linear_to_hlg", linear, tf::rites_x16::linear_to_hlg_slice_v4);
    g.finish();
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_tf_x16(_c: &mut Criterion) {}

criterion_group!(benches, bench_tf_scalar, bench_tf_x8, bench_tf_x16);
criterion_main!(benches);
