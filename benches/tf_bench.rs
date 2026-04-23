//! Benchmarks for all transfer functions across tiers (scalar, x8, x16).

use linear_srgb::tf;
use linear_srgb::tokens;
use std::hint::black_box;
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

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
    let Some(token) = archmage::X64V3Token::summon() else {
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
                fn call_slice(token: archmage::X64V3Token, values: &mut [f32]) {
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

    bench_x8!(
        "srgb_to_linear",
        encoded,
        tokens::x8::tf_srgb_to_linear_slice_v3
    );
    bench_x8!(
        "linear_to_srgb",
        linear,
        tokens::x8::tf_linear_to_srgb_slice_v3
    );
    bench_x8!(
        "bt709_to_linear",
        encoded,
        tokens::x8::bt709_to_linear_slice_v3
    );
    bench_x8!(
        "linear_to_bt709",
        linear,
        tokens::x8::linear_to_bt709_slice_v3
    );
    bench_x8!("pq_to_linear", encoded, tokens::x8::pq_to_linear_slice_v3);
    bench_x8!("linear_to_pq", linear, tokens::x8::linear_to_pq_slice_v3);
    bench_x8!("hlg_to_linear", encoded, tokens::x8::hlg_to_linear_slice_v3);
    bench_x8!("linear_to_hlg", linear, tokens::x8::linear_to_hlg_slice_v3);
    g.finish();
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_tf_x8(_c: &mut Criterion) {}

// =============================================================================
// x16 rites (AVX-512)
// =============================================================================

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn bench_tf_x16(c: &mut Criterion) {
    use archmage::SimdToken;
    let Some(token) = archmage::X64V4Token::summon() else {
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
                fn call_slice(token: archmage::X64V4Token, values: &mut [f32]) {
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

    bench_x16!(
        "srgb_to_linear",
        encoded,
        tokens::x16::tf_srgb_to_linear_slice_v4
    );
    bench_x16!(
        "linear_to_srgb",
        linear,
        tokens::x16::tf_linear_to_srgb_slice_v4
    );
    bench_x16!(
        "bt709_to_linear",
        encoded,
        tokens::x16::bt709_to_linear_slice_v4
    );
    bench_x16!(
        "linear_to_bt709",
        linear,
        tokens::x16::linear_to_bt709_slice_v4
    );
    bench_x16!("pq_to_linear", encoded, tokens::x16::pq_to_linear_slice_v4);
    bench_x16!("linear_to_pq", linear, tokens::x16::linear_to_pq_slice_v4);
    bench_x16!(
        "hlg_to_linear",
        encoded,
        tokens::x16::hlg_to_linear_slice_v4
    );
    bench_x16!("linear_to_hlg", linear, tokens::x16::linear_to_hlg_slice_v4);
    g.finish();
}

#[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
fn bench_tf_x16(_c: &mut Criterion) {}

// =============================================================================
// Tier isolation: x8 rites with V4 (AVX-512) disabled
// =============================================================================
//
// On AVX-512 machines, verify x8 performance is unaffected when V4 is disabled.
// This catches regressions where x8 code accidentally depends on V4 features.

#[cfg(target_arch = "x86_64")]
fn bench_tf_x8_v4_disabled(c: &mut Criterion) {
    use archmage::SimdToken;

    // Disable V4 to ensure x8 runs in pure AVX2+FMA mode
    let _ = archmage::X64V4Token::dangerously_disable_token_process_wide(true);

    let Some(token) = archmage::X64V3Token::summon() else {
        eprintln!("AVX2+FMA not available, skipping x8_v4_disabled benchmarks");
        let _ = archmage::X64V4Token::dangerously_disable_token_process_wide(false);
        return;
    };

    let mut g = c.benchmark_group("tf_x8_v4disabled");
    g.throughput(Throughput::Elements(N as u64));

    let encoded = make_encoded();
    let linear = make_linear();

    macro_rules! bench_x8 {
        ($name:expr, $data:expr, $fn:path) => {
            g.bench_function($name, |b| {
                #[archmage::arcane]
                fn call_slice(token: archmage::X64V3Token, values: &mut [f32]) {
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

    bench_x8!(
        "srgb_to_linear",
        encoded,
        tokens::x8::tf_srgb_to_linear_slice_v3
    );
    bench_x8!(
        "linear_to_srgb",
        linear,
        tokens::x8::tf_linear_to_srgb_slice_v3
    );
    bench_x8!(
        "bt709_to_linear",
        encoded,
        tokens::x8::bt709_to_linear_slice_v3
    );
    bench_x8!(
        "linear_to_bt709",
        linear,
        tokens::x8::linear_to_bt709_slice_v3
    );
    bench_x8!("pq_to_linear", encoded, tokens::x8::pq_to_linear_slice_v3);
    bench_x8!("linear_to_pq", linear, tokens::x8::linear_to_pq_slice_v3);
    bench_x8!("hlg_to_linear", encoded, tokens::x8::hlg_to_linear_slice_v3);
    bench_x8!("linear_to_hlg", linear, tokens::x8::linear_to_hlg_slice_v3);
    g.finish();

    let _ = archmage::X64V4Token::dangerously_disable_token_process_wide(false);
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_tf_x8_v4_disabled(_c: &mut Criterion) {}

// =============================================================================
// Tier isolation: scalar fallback via dispatch disable
// =============================================================================
//
// Disables V3 to force all incant! dispatch to scalar tier. Verifies the
// scalar fallback path through real dispatch machinery — not just direct
// scalar calls. Catches broken dispatch tables and scalar codegen regressions.

#[cfg(target_arch = "x86_64")]
fn bench_tf_scalar_via_dispatch(c: &mut Criterion) {
    if archmage::X64V3Token::dangerously_disable_token_process_wide(true).is_err() {
        eprintln!(
            "Cannot disable V3 (compile-time guaranteed). \
             Build without -Ctarget-cpu=native or enable testable_dispatch."
        );
        return;
    }

    let mut g = c.benchmark_group("tf_scalar_dispatch");
    g.throughput(Throughput::Elements(N as u64));

    let encoded = make_encoded();
    let linear = make_linear();

    // These go through the same scalar path as bench_tf_scalar, but via
    // the dispatch machinery. Perf should be identical to direct scalar calls.
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

    let _ = archmage::X64V3Token::dangerously_disable_token_process_wide(false);
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_tf_scalar_via_dispatch(_c: &mut Criterion) {}

// =============================================================================
// Public slice dispatcher: before (scalar-loop + autoversion) vs after (incant!)
// =============================================================================
//
// Replicates the pre-fix `default::bt709_to_linear_slice` et al. shape
// (scalar for-loop, #[archmage::autoversion]) so we can A/B it against the
// current incant!-dispatched implementation without toggling source trees.
// The `_old` functions are exactly what shipped before issue #10 was fixed.

#[cfg(feature = "transfer")]
mod old_dispatchers {
    use linear_srgb::tf;

    #[archmage::autoversion]
    pub fn bt709_to_linear_slice_old(values: &mut [f32]) {
        for v in values.iter_mut() {
            *v = tf::bt709_to_linear(*v);
        }
    }
    #[archmage::autoversion]
    pub fn linear_to_bt709_slice_old(values: &mut [f32]) {
        for v in values.iter_mut() {
            *v = tf::linear_to_bt709(*v);
        }
    }
    #[archmage::autoversion]
    pub fn pq_to_linear_slice_old(values: &mut [f32]) {
        for v in values.iter_mut() {
            *v = tf::pq_to_linear(*v);
        }
    }
    #[archmage::autoversion]
    pub fn linear_to_pq_slice_old(values: &mut [f32]) {
        for v in values.iter_mut() {
            *v = tf::linear_to_pq(*v);
        }
    }
    #[archmage::autoversion]
    pub fn hlg_to_linear_slice_old(values: &mut [f32]) {
        for v in values.iter_mut() {
            *v = tf::hlg_to_linear(*v);
        }
    }
    #[archmage::autoversion]
    pub fn linear_to_hlg_slice_old(values: &mut [f32]) {
        for v in values.iter_mut() {
            *v = tf::linear_to_hlg(*v);
        }
    }
}

// Scalar RGBA fallback: what a caller *had* to write before the _rgba_slice
// TF variants existed (skipping alpha by hand), so we can measure the gain.
#[cfg(feature = "transfer")]
mod old_rgba_fallbacks {
    use linear_srgb::tf;

    macro_rules! rgba_scalar {
        ($name:ident, $fn:path) => {
            #[archmage::autoversion]
            pub fn $name(values: &mut [f32]) {
                for pixel in values.chunks_exact_mut(4) {
                    pixel[0] = $fn(pixel[0]);
                    pixel[1] = $fn(pixel[1]);
                    pixel[2] = $fn(pixel[2]);
                }
            }
        };
    }

    rgba_scalar!(bt709_to_linear_rgba_old, tf::bt709_to_linear);
    rgba_scalar!(linear_to_bt709_rgba_old, tf::linear_to_bt709);
    rgba_scalar!(pq_to_linear_rgba_old, tf::pq_to_linear);
    rgba_scalar!(linear_to_pq_rgba_old, tf::linear_to_pq);
    rgba_scalar!(hlg_to_linear_rgba_old, tf::hlg_to_linear);
    rgba_scalar!(linear_to_hlg_rgba_old, tf::linear_to_hlg);
}

#[cfg(feature = "transfer")]
fn make_rgba_encoded() -> Vec<f32> {
    // N pixels, alpha at position 3 kept in [0, 1] so the "apply TF to all
    // elements" mistake would corrupt it visibly.
    (0..N)
        .flat_map(|i| {
            let t = i as f32 / N as f32;
            [t, (t * 0.7).min(1.0), (1.0 - t).max(0.0), 0.5_f32]
        })
        .collect()
}

#[cfg(feature = "transfer")]
fn make_rgba_linear() -> Vec<f32> {
    make_rgba_encoded()
        .chunks_exact(4)
        .flat_map(|px| {
            [
                linear_srgb::tf::srgb_to_linear(px[0]),
                linear_srgb::tf::srgb_to_linear(px[1]),
                linear_srgb::tf::srgb_to_linear(px[2]),
                px[3],
            ]
        })
        .collect()
}

#[cfg(feature = "transfer")]
fn bench_tf_public_dispatcher_rgba(c: &mut Criterion) {
    use linear_srgb::default;

    let encoded = make_rgba_encoded();
    let linear = make_rgba_linear();

    macro_rules! ab_rgba {
        ($group:expr, $data:expr, $old:path, $new:path) => {
            let mut g = c.benchmark_group($group);
            g.throughput(Throughput::Elements(N as u64));
            g.bench_function("old_scalar_rgba_loop", |b| {
                let mut buf = $data.clone();
                b.iter(|| {
                    buf.copy_from_slice(&$data);
                    $old(black_box(&mut buf));
                })
            });
            g.bench_function("new_incant_rgba_dispatch", |b| {
                let mut buf = $data.clone();
                b.iter(|| {
                    buf.copy_from_slice(&$data);
                    $new(black_box(&mut buf));
                })
            });
            g.finish();
        };
    }

    ab_rgba!(
        "tf_rgba_bt709_to_linear",
        encoded,
        old_rgba_fallbacks::bt709_to_linear_rgba_old,
        default::bt709_to_linear_rgba_slice
    );
    ab_rgba!(
        "tf_rgba_linear_to_bt709",
        linear,
        old_rgba_fallbacks::linear_to_bt709_rgba_old,
        default::linear_to_bt709_rgba_slice
    );
    ab_rgba!(
        "tf_rgba_pq_to_linear",
        encoded,
        old_rgba_fallbacks::pq_to_linear_rgba_old,
        default::pq_to_linear_rgba_slice
    );
    ab_rgba!(
        "tf_rgba_linear_to_pq",
        linear,
        old_rgba_fallbacks::linear_to_pq_rgba_old,
        default::linear_to_pq_rgba_slice
    );
    ab_rgba!(
        "tf_rgba_hlg_to_linear",
        encoded,
        old_rgba_fallbacks::hlg_to_linear_rgba_old,
        default::hlg_to_linear_rgba_slice
    );
    ab_rgba!(
        "tf_rgba_linear_to_hlg",
        linear,
        old_rgba_fallbacks::linear_to_hlg_rgba_old,
        default::linear_to_hlg_rgba_slice
    );
}

#[cfg(not(feature = "transfer"))]
fn bench_tf_public_dispatcher_rgba(_c: &mut Criterion) {}

#[cfg(feature = "transfer")]
fn bench_tf_public_dispatcher(c: &mut Criterion) {
    use linear_srgb::default;

    let encoded = make_encoded();
    let linear = make_linear();

    macro_rules! ab_bench {
        ($group_name:expr, $bench_name:expr, $data:expr, $old:path, $new:path) => {
            let mut g = c.benchmark_group($group_name);
            g.throughput(Throughput::Elements(N as u64));
            g.bench_function("old_scalar_loop", |b| {
                let mut buf = $data.clone();
                b.iter(|| {
                    buf.copy_from_slice(&$data);
                    $old(black_box(&mut buf));
                })
            });
            g.bench_function("new_incant_dispatch", |b| {
                let mut buf = $data.clone();
                b.iter(|| {
                    buf.copy_from_slice(&$data);
                    $new(black_box(&mut buf));
                })
            });
            let _ = $bench_name;
            g.finish();
        };
    }

    ab_bench!(
        "tf_dispatch_bt709_to_linear",
        "",
        encoded,
        old_dispatchers::bt709_to_linear_slice_old,
        default::bt709_to_linear_slice
    );
    ab_bench!(
        "tf_dispatch_linear_to_bt709",
        "",
        linear,
        old_dispatchers::linear_to_bt709_slice_old,
        default::linear_to_bt709_slice
    );
    ab_bench!(
        "tf_dispatch_pq_to_linear",
        "",
        encoded,
        old_dispatchers::pq_to_linear_slice_old,
        default::pq_to_linear_slice
    );
    ab_bench!(
        "tf_dispatch_linear_to_pq",
        "",
        linear,
        old_dispatchers::linear_to_pq_slice_old,
        default::linear_to_pq_slice
    );
    ab_bench!(
        "tf_dispatch_hlg_to_linear",
        "",
        encoded,
        old_dispatchers::hlg_to_linear_slice_old,
        default::hlg_to_linear_slice
    );
    ab_bench!(
        "tf_dispatch_linear_to_hlg",
        "",
        linear,
        old_dispatchers::linear_to_hlg_slice_old,
        default::linear_to_hlg_slice
    );
}

#[cfg(not(feature = "transfer"))]
fn bench_tf_public_dispatcher(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_tf_scalar,
    bench_tf_x8,
    bench_tf_x16,
    bench_tf_x8_v4_disabled,
    bench_tf_scalar_via_dispatch,
    bench_tf_public_dispatcher,
    bench_tf_public_dispatcher_rgba,
);
criterion_main!(benches);
