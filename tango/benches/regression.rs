//! Paired-benchmark regression gate for linear-srgb slice dispatchers.
//!
//! Run as a tango-bench binary. Build once against the published baseline,
//! then rebuild with `[patch.crates-io]` pointing at the local WIP and
//! invoke in `compare` mode. See `tango/README.md`.

use std::hint::black_box;

use linear_srgb::default::{
    bt709_to_linear_slice, hlg_to_linear_slice, linear_to_bt709_slice, linear_to_hlg_slice,
    linear_to_pq_slice, pq_to_linear_slice,
};
use tango_bench::{IntoBenchmarks, benchmark_fn, tango_benchmarks, tango_main};

// Sizes chosen to probe three regimes that behave differently in the issue #18
// regression: per-call overhead (256), typical tile-scanline (4096), and
// bandwidth-bound (~1080p width * 4 channels, 8192 floats).
const SIZES: &[usize] = &[256, 4096, 8192];

// Four input distributions so a regression on one branch of the TF kernel
// shows up even if the averaged-mixed case hides it. All map to f32 values
// already in the [0, 1] signal / linear domain used by HLG, BT.709, PQ.
fn dist_uniform(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) / (n as f32)).collect()
}

fn dist_all_small(n: usize) -> Vec<f32> {
    // Forces the quadratic / linear branch on linear_to_hlg, the linear
    // branch on bt709, and the small-poly branch on PQ.
    (0..n).map(|i| 0.0625 * (i as f32) / (n as f32)).collect()
}

fn dist_all_large(n: usize) -> Vec<f32> {
    // Forces the log / sinh / large-poly branch.
    (0..n)
        .map(|i| 0.2 + 0.8 * (i as f32) / (n as f32))
        .collect()
}

fn dist_hdr_luma(n: usize) -> Vec<f32> {
    // Approximates measured HDR luma distribution — ~44% below HLG split.
    (0..n)
        .map(|i| {
            let x = (i as f32) / (n as f32);
            (x * x * x).clamp(0.0, 1.0)
        })
        .collect()
}

struct Case {
    label: &'static str,
    data: Vec<f32>,
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for &n in SIZES {
        out.push(Case {
            label: leak_label(&format!("uniform_{n}")),
            data: dist_uniform(n),
        });
        out.push(Case {
            label: leak_label(&format!("small_{n}")),
            data: dist_all_small(n),
        });
        out.push(Case {
            label: leak_label(&format!("large_{n}")),
            data: dist_all_large(n),
        });
        out.push(Case {
            label: leak_label(&format!("hdr_luma_{n}")),
            data: dist_hdr_luma(n),
        });
    }
    out
}

fn leak_label(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

fn bench_fn(
    prefix: &'static str,
    op: fn(&mut [f32]),
) -> impl IntoIterator<Item = tango_bench::Benchmark> {
    cases().into_iter().map(move |case| {
        let data = case.data.clone();
        let label = leak_label(&format!("{prefix}_{}", case.label));
        benchmark_fn(label, move |b| {
            let mut buf = data.clone();
            let src = data.clone();
            b.iter(move || {
                buf.copy_from_slice(&src);
                op(black_box(&mut buf));
            })
        })
    })
}

fn benchmarks() -> impl IntoBenchmarks {
    let mut out: Vec<tango_bench::Benchmark> = Vec::new();
    out.extend(bench_fn("linear_to_hlg", linear_to_hlg_slice));
    out.extend(bench_fn("hlg_to_linear", hlg_to_linear_slice));
    out.extend(bench_fn("linear_to_bt709", linear_to_bt709_slice));
    out.extend(bench_fn("bt709_to_linear", bt709_to_linear_slice));
    out.extend(bench_fn("linear_to_pq", linear_to_pq_slice));
    out.extend(bench_fn("pq_to_linear", pq_to_linear_slice));
    out
}

tango_benchmarks!(benchmarks());
tango_main!();
