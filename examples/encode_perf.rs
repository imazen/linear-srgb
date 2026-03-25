//! Benchmark encode LUT strategies: uniform vs sqrt vs two-range vs polynomial.
//!
//! cargo run --example encode_perf --release --all-features

use std::hint::black_box;
use std::time::Instant;

const N: usize = 15360; // 4K RGBA row (3840 × 4)
const ITERS: usize = 10_000;

fn make_linear(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 / n as f32).powi(2)).collect()
}

// --- Strategy 1: current uniform LUT ---
fn bench_uniform(src: &[f32], lut: &[u16; 65537]) -> Vec<u16> {
    let mut dst = vec![0u16; src.len()];
    for (&l, d) in src.iter().zip(dst.iter_mut()) {
        let idx = (l.clamp(0.0, 1.0) * 65536.0 + 0.5) as usize;
        *d = lut[idx.min(65536)];
    }
    dst
}

// --- Strategy 2: sqrt-indexed LUT ---
fn bench_sqrt(src: &[f32], lut: &[u16]) -> Vec<u16> {
    let n = lut.len() - 1;
    let scale = n as f32;
    let mut dst = vec![0u16; src.len()];
    for (&l, d) in src.iter().zip(dst.iter_mut()) {
        let idx = (l.clamp(0.0, 1.0).sqrt() * scale + 0.5) as usize;
        *d = lut[idx.min(n)];
    }
    dst
}

// --- Strategy 3: two-range (dense toe + uniform main) ---
struct TwoRange {
    toe: Vec<u16>,
    main: Vec<u16>,
    thresh: f32,
    toe_scale: f32,
    main_scale: f32,
}

fn bench_two_range(src: &[f32], tr: &TwoRange) -> Vec<u16> {
    let toe_max = tr.toe.len() - 1;
    let main_max = tr.main.len() - 1;
    let mut dst = vec![0u16; src.len()];
    for (&l, d) in src.iter().zip(dst.iter_mut()) {
        let l = l.clamp(0.0, 1.0);
        *d = if l < tr.thresh {
            let idx = (l * tr.toe_scale + 0.5) as usize;
            tr.toe[idx.min(toe_max)]
        } else {
            let idx = ((l - tr.thresh) * tr.main_scale + 0.5) as usize;
            tr.main[idx.min(main_max)]
        };
    }
    dst
}

// --- Strategy 4: polynomial (no LUT) ---
fn bench_poly(src: &[f32]) -> Vec<u16> {
    let mut dst = vec![0u16; src.len()];
    for (&l, d) in src.iter().zip(dst.iter_mut()) {
        let srgb = linear_srgb::default::linear_to_srgb(l);
        *d = (srgb * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
    }
    dst
}

fn time_it(name: &str, src: &[f32], f: impl Fn(&[f32]) -> Vec<u16>) {
    // Warmup
    for _ in 0..100 {
        black_box(f(black_box(src)));
    }

    let t = Instant::now();
    for _ in 0..ITERS {
        black_box(f(black_box(src)));
    }
    let elapsed = t.elapsed();
    let per_iter = elapsed / ITERS as u32;
    let ops_per_sec = src.len() as f64 * ITERS as f64 / elapsed.as_secs_f64();
    eprintln!(
        "{name:>25}: {per_iter:>8.1?}  ({:.0} Mops/s)",
        ops_per_sec / 1e6
    );
}

fn main() {
    let src = make_linear(N);

    // Build uniform encode LUT (current)
    let uniform_lut = linear_srgb::u16_lut::generate_encode_lut();

    // Build sqrt-indexed LUT (65537 entries)
    let sqrt_n = 65537usize;
    let sqrt_lut: Vec<u16> = (0..sqrt_n)
        .map(|i| {
            let t = i as f32 / (sqrt_n - 1) as f32;
            let linear = t * t; // inverse of sqrt
            let srgb = linear_srgb::default::linear_to_srgb(linear);
            (srgb * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16
        })
        .collect();

    // Build two-range T=0.02 (toe 8192 + main 57345 = 65537 total, 128KB)
    let thresh = 0.02f32;
    let toe_n = 8192usize;
    let main_n = 57345usize;
    let toe_scale = (toe_n - 1) as f32 / thresh;
    let main_scale = (main_n - 1) as f32 / (1.0 - thresh);
    let toe_lut: Vec<u16> = (0..toe_n)
        .map(|i| {
            let linear = i as f32 / toe_scale;
            let srgb = linear_srgb::default::linear_to_srgb(linear);
            (srgb * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16
        })
        .collect();
    let main_lut: Vec<u16> = (0..main_n)
        .map(|i| {
            let linear = thresh + i as f32 / main_scale;
            let srgb = linear_srgb::default::linear_to_srgb(linear);
            (srgb * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16
        })
        .collect();
    let tr = TwoRange {
        toe: toe_lut,
        main: main_lut,
        thresh,
        toe_scale,
        main_scale,
    };

    eprintln!("Encoding {N} elements × {ITERS} iterations:\n");

    time_it("uniform LUT (current)", &src, |s| {
        bench_uniform(s, &uniform_lut)
    });
    time_it("sqrt LUT", &src, |s| bench_sqrt(s, &sqrt_lut));
    time_it("two-range T=.02", &src, |s| bench_two_range(s, &tr));
    time_it("polynomial (no LUT)", &src, bench_poly);
}
