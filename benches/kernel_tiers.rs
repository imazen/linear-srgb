//! Per-kernel NEON-vs-forced-scalar for linear-srgb's batch kernels.
//!
//! This crate has 135 dispatch sites and had no tier benchmark: `benchmarks.rs`
//! and `tf_bench.rs` measure absolute throughput, which cannot reveal a kernel
//! slower than the scalar tier it dispatches away from. That failure mode was
//! real elsewhere in the 2026-07-29 aarch64 sweep — zenquant's palette search
//! was running at 0.58x its own scalar tier, and zenresize's f16 H-filter at
//! 0.94x, both invisible to absolute-throughput benches.
//!
//! NEON is BASELINE on aarch64, so the "scalar" arm is autovectorized too:
//! ~1.00x means LLVM already matched the hand-written path, BELOW 1.00 is a bug.
//!
//! Run: `cargo bench --bench kernel_tiers`
//! Do NOT build with `-C target-cpu=native` (the tier then cannot be disabled).

use zenbench::prelude::*;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(on: bool) -> bool {
    use archmage::SimdToken;
    TierToken::dangerously_disable_token_process_wide(!on).is_ok()
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_on: bool) -> bool {
    false
}

const N: usize = 1 << 20;

fn bench(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!("[kernel_tiers] SIMD tier not toggleable here. Skipping.");
        return;
    }
    set_simd(true);
    eprintln!("[kernel_tiers] comparing {TIER_NAME} vs forced scalar");

    let f32src: &'static [f32] = Box::leak(
        (0..N)
            .map(|i| (i % 1000) as f32 / 1000.0)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    let u8src: &'static [u8] = Box::leak(
        (0..N)
            .map(|i| (i % 251) as u8)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );

    // In-place f32 kernels: the buffer clone is built in with_input so the
    // 4 MB allocation is not inside the timed region.
    macro_rules! inplace {
        ($name:expr, $call:path) => {
            suite.compare($name, |g| {
                g.throughput(Throughput::Bytes((N * 4) as u64));
                for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                    g.bench(arm, move |b| {
                        b.with_input(move || {
                            set_simd(simd);
                            f32src.to_vec()
                        })
                        .run(move |mut v| {
                            $call(&mut v);
                            v
                        })
                    });
                }
            });
        };
    }
    macro_rules! outplace {
        ($name:expr, $bytes:expr, $mk:expr, $call:expr) => {
            suite.compare($name, |g| {
                g.throughput(Throughput::Bytes($bytes as u64));
                for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                    g.bench(arm, move |b| {
                        b.with_input(move || {
                            set_simd(simd);
                            $mk
                        })
                        .run(move |mut o| {
                            $call(&mut o);
                            o
                        })
                    });
                }
            });
        };
    }

    inplace!(
        "srgb_to_linear_slice",
        linear_srgb::default::srgb_to_linear_slice
    );
    inplace!(
        "linear_to_srgb_slice",
        linear_srgb::default::linear_to_srgb_slice
    );
    inplace!(
        "srgb_to_linear_rgba_slice",
        linear_srgb::default::srgb_to_linear_rgba_slice
    );
    inplace!(
        "pq_to_linear_slice",
        linear_srgb::default::pq_to_linear_slice
    );
    inplace!(
        "hlg_to_linear_slice",
        linear_srgb::default::hlg_to_linear_slice
    );
    inplace!(
        "bt709_to_linear_slice",
        linear_srgb::default::bt709_to_linear_slice
    );

    outplace!(
        "srgb_u8_to_linear_slice",
        N,
        vec![0f32; N],
        |o: &mut Vec<f32>| linear_srgb::default::srgb_u8_to_linear_slice(u8src, o)
    );
    outplace!(
        "linear_to_srgb_u8_slice",
        N * 4,
        vec![0u8; N],
        |o: &mut Vec<u8>| linear_srgb::default::linear_to_srgb_u8_slice(f32src, o)
    );
    outplace!(
        "srgb_u8_to_linear_rgba_slice",
        N,
        vec![0f32; N],
        |o: &mut Vec<f32>| linear_srgb::default::srgb_u8_to_linear_rgba_slice(u8src, o)
    );
    outplace!(
        "unpremul_linear_to_srgb_u8_rgba",
        N * 4,
        vec![0u8; N],
        |o: &mut Vec<u8>| linear_srgb::default::unpremultiply_linear_to_srgb_u8_rgba_slice(
            f32src, o
        )
    );

    // ---- the remaining 20 slice kernels ----
    // The first pass covered 10 of the crate's 30 and found one regression in
    // that sample, so the rest are swept too rather than assumed healthy.

    // f32 in-place, no extra parameter.
    inplace!(
        "linear_to_bt709_slice",
        linear_srgb::default::linear_to_bt709_slice
    );
    inplace!(
        "linear_to_hlg_slice",
        linear_srgb::default::linear_to_hlg_slice
    );
    inplace!(
        "linear_to_pq_slice",
        linear_srgb::default::linear_to_pq_slice
    );
    inplace!(
        "linear_to_srgb_rgba_slice",
        linear_srgb::default::linear_to_srgb_rgba_slice
    );
    inplace!(
        "linear_to_srgb_extended_slice",
        linear_srgb::default::linear_to_srgb_extended_slice
    );
    inplace!(
        "srgb_to_linear_extended_slice",
        linear_srgb::default::srgb_to_linear_extended_slice
    );
    inplace!(
        "srgb_to_linear_premultiply_rgba",
        linear_srgb::default::srgb_to_linear_premultiply_rgba_slice
    );
    inplace!(
        "unpremul_linear_to_srgb_rgba",
        linear_srgb::default::unpremultiply_linear_to_srgb_rgba_slice
    );

    // f32 in-place taking a gamma argument.
    macro_rules! inplace_gamma {
        ($name:expr, $call:path) => {
            suite.compare($name, |g| {
                g.throughput(Throughput::Bytes((N * 4) as u64));
                for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                    g.bench(arm, move |b| {
                        b.with_input(move || {
                            set_simd(simd);
                            f32src.to_vec()
                        })
                        .run(move |mut v| {
                            $call(&mut v, 2.2);
                            v
                        })
                    });
                }
            });
        };
    }
    inplace_gamma!(
        "gamma_to_linear_slice",
        linear_srgb::default::gamma_to_linear_slice
    );
    inplace_gamma!(
        "linear_to_gamma_slice",
        linear_srgb::default::linear_to_gamma_slice
    );
    inplace_gamma!(
        "gamma_to_linear_premul_rgba",
        linear_srgb::default::gamma_to_linear_premultiply_rgba_slice
    );
    inplace_gamma!(
        "unpremul_linear_to_gamma_rgba",
        linear_srgb::default::unpremultiply_linear_to_gamma_rgba_slice
    );

    // Out-of-place conversions.
    let u16src: &'static [u16] = Box::leak(
        (0..N)
            .map(|i| (i % 65521) as u16)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );

    outplace!(
        "linear_to_srgb_u8_rgba_slice",
        N * 4,
        vec![0u8; N],
        |o: &mut Vec<u8>| linear_srgb::default::linear_to_srgb_u8_rgba_slice(f32src, o)
    );
    outplace!(
        "linear_to_srgb_u16_slice",
        N * 4,
        vec![0u16; N],
        |o: &mut Vec<u16>| linear_srgb::default::linear_to_srgb_u16_slice(f32src, o)
    );
    outplace!(
        "linear_to_srgb_u16_slice_fast",
        N * 4,
        vec![0u16; N],
        |o: &mut Vec<u16>| linear_srgb::default::linear_to_srgb_u16_slice_fast(f32src, o)
    );
    outplace!(
        "linear_to_srgb_u16_rgba_slice",
        N * 4,
        vec![0u16; N],
        |o: &mut Vec<u16>| linear_srgb::default::linear_to_srgb_u16_rgba_slice(f32src, o)
    );
    outplace!(
        "linear_to_srgb_u16_rgba_fast",
        N * 4,
        vec![0u16; N],
        |o: &mut Vec<u16>| linear_srgb::default::linear_to_srgb_u16_rgba_slice_fast(f32src, o)
    );
    outplace!(
        "srgb_u16_to_linear_slice",
        N * 2,
        vec![0f32; N],
        |o: &mut Vec<f32>| linear_srgb::default::srgb_u16_to_linear_slice(u16src, o)
    );
    outplace!(
        "srgb_u16_to_linear_rgba_slice",
        N * 2,
        vec![0f32; N],
        |o: &mut Vec<f32>| linear_srgb::default::srgb_u16_to_linear_rgba_slice(u16src, o)
    );
    outplace!(
        "srgb_u8_to_linear_premul_rgba",
        N,
        vec![0f32; N],
        |o: &mut Vec<f32>| linear_srgb::default::srgb_u8_to_linear_premultiply_rgba_slice(u8src, o)
    );

    set_simd(true);
}

zenbench::main!(bench);
