//! Versioned `zenbench` suite exercising downstream-used `linear-srgb`
//! functions across:
//! 1. Chained in-register roundtrip latency (`64` dependent roundtrips -> pure
//!    instruction-latency bound, zero memory traffic), and
//! 2. Memory-hierarchy regimes: `L1_256` (`1 KiB`), `L2_16k` (`64 KiB`), and
//!    `DRAM_2M` (`8 MiB`) to separate instruction latency/throughput from
//!    L2/DRAM bandwidth ceilings.

#![allow(deprecated)]

use core::hint::black_box;
use linear_srgb::default as ls;
use zenbench::prelude::*;

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn chain_srgb_rt_v3(t: archmage::X64V3Token, seed: [f32; 8], reps: usize) -> [f32; 8] {
    use linear_srgb::tokens::x8;
    let mut v = seed;
    for _ in 0..reps {
        v = x8::linear_to_srgb_v3(t, x8::srgb_to_linear_v3(t, v));
    }
    v
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn chain_pq_rt_v3(t: archmage::X64V3Token, seed: [f32; 8], reps: usize) -> [f32; 8] {
    use linear_srgb::tokens::x8;
    let mut v = seed;
    for _ in 0..reps {
        v = x8::linear_to_pq_v3(t, x8::pq_to_linear_v3(t, v));
    }
    v
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn chain_bt709_rt_v3(t: archmage::X64V3Token, seed: [f32; 8], reps: usize) -> [f32; 8] {
    use linear_srgb::tokens::x8;
    let mut v = seed;
    for _ in 0..reps {
        v = x8::linear_to_bt709_v3(t, x8::bt709_to_linear_v3(t, v));
    }
    v
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn chain_hlg_rt_v3(t: archmage::X64V3Token, seed: [f32; 8], reps: usize) -> [f32; 8] {
    use linear_srgb::tokens::x8;
    let mut v = seed;
    for _ in 0..reps {
        v = x8::linear_to_hlg_v3(t, x8::hlg_to_linear_v3(t, v));
    }
    v
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn chain_gamma_rt_v3(t: archmage::X64V3Token, seed: [f32; 8], g: f32, reps: usize) -> [f32; 8] {
    use linear_srgb::tokens::x8;
    let mut v = seed;
    for _ in 0..reps {
        v = x8::linear_to_gamma_v3(t, x8::gamma_to_linear_v3(t, v, g), g);
    }
    v
}

fn bench_downstream(suite: &mut Suite) {
    #[cfg(target_arch = "x86_64")]
    if let Some(tok) = <archmage::X64V3Token as archmage::SimdToken>::summon() {
        let reps = 64usize;
        let seed = [0.05f32, 0.15, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95];
        suite.compare("latency_chain64_x8", |g| {
            g.throughput(Throughput::Elements((reps * 8) as u64));
            g.bench("srgb_rt", move |b| {
                b.iter(|| black_box(chain_srgb_rt_v3(tok, black_box(seed), black_box(reps))))
            });
            g.bench("pq_rt", move |b| {
                b.iter(|| black_box(chain_pq_rt_v3(tok, black_box(seed), black_box(reps))))
            });
            g.bench("bt709_rt", move |b| {
                b.iter(|| black_box(chain_bt709_rt_v3(tok, black_box(seed), black_box(reps))))
            });
            g.bench("hlg_rt", move |b| {
                b.iter(|| black_box(chain_hlg_rt_v3(tok, black_box(seed), black_box(reps))))
            });
            g.bench("gamma22_rt", move |b| {
                b.iter(|| {
                    black_box(chain_gamma_rt_v3(
                        tok,
                        black_box(seed),
                        black_box(2.2),
                        black_box(reps),
                    ))
                })
            });
        });
    }

    for &(tag, n) in &[
        ("L1_256", 256usize),
        ("L2_16k", 16_384usize),
        ("DRAM_2M", 2_097_152usize),
    ] {
        let src_f32: Vec<f32> = (0..n)
            .map(|i| 0.025 + ((i % 997) as f32) * (0.95 / 997.0))
            .collect();
        let src_u8: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let src_u16: Vec<u16> = (0..n).map(|i| ((i * 257) & 0xFFFF) as u16).collect();

        suite.compare(format!("{tag}/f32_slices"), |g| {
            g.throughput(Throughput::Bytes((n * 4) as u64));
            macro_rules! arm {
                ($name:expr, $call:ident) => {{
                    let s = src_f32.clone();
                    let mut buf = s.clone();
                    g.bench($name, move |b| {
                        b.iter(|| {
                            buf.copy_from_slice(&s);
                            ls::$call(&mut buf);
                            black_box(buf[0])
                        })
                    });
                }};
            }
            arm!("srgb_to_linear", srgb_to_linear_slice);
            arm!("linear_to_srgb", linear_to_srgb_slice);
            arm!("srgb_to_linear_rgba", srgb_to_linear_rgba_slice);
            arm!("srgb_to_linear_ext", srgb_to_linear_extended_slice);
            arm!("linear_to_srgb_ext", linear_to_srgb_extended_slice);
            arm!("srgb_premul_rgba", srgb_to_linear_premultiply_rgba_slice);
            arm!(
                "unpremul_srgb_rgba",
                unpremultiply_linear_to_srgb_rgba_slice
            );
            arm!("bt709_to_linear", bt709_to_linear_slice);
            arm!("linear_to_bt709", linear_to_bt709_slice);
            arm!("pq_to_linear", pq_to_linear_slice);
            arm!("linear_to_pq", linear_to_pq_slice);
            arm!("hlg_to_linear", hlg_to_linear_slice);
            arm!("linear_to_hlg", linear_to_hlg_slice);
            {
                let s = src_f32.clone();
                let mut buf = s.clone();
                g.bench("gamma22_to_linear", move |b| {
                    b.iter(|| {
                        buf.copy_from_slice(&s);
                        ls::gamma_to_linear_slice(&mut buf, black_box(2.2));
                        black_box(buf[0])
                    })
                });
            }
        });

        suite.compare(format!("{tag}/int_slices"), |g| {
            g.throughput(Throughput::Elements(n as u64));
            {
                let s = src_u8.clone();
                let mut d = vec![0.0f32; n];
                g.bench("srgb_u8_to_linear", move |b| {
                    b.iter(|| {
                        ls::srgb_u8_to_linear_slice(&s, &mut d);
                        black_box(d[0])
                    })
                });
            }
            {
                let s = src_f32.clone();
                let mut d = vec![0u8; n];
                g.bench("linear_to_srgb_u8", move |b| {
                    b.iter(|| {
                        ls::linear_to_srgb_u8_slice(&s, &mut d);
                        black_box(d[0])
                    })
                });
            }
            {
                let s = src_u16.clone();
                let mut d = vec![0.0f32; n];
                g.bench("srgb_u16_to_linear", move |b| {
                    b.iter(|| {
                        ls::srgb_u16_to_linear_slice(&s, &mut d);
                        black_box(d[0])
                    })
                });
            }
            {
                let s = src_f32.clone();
                let mut d = vec![0u16; n];
                g.bench("linear_to_srgb_u16", move |b| {
                    b.iter(|| {
                        ls::linear_to_srgb_u16_slice(&s, &mut d);
                        black_box(d[0])
                    })
                });
            }
            {
                let s = src_f32.clone();
                let mut d = vec![0u16; n];
                g.bench("linear_to_srgb_u16_fast", move |b| {
                    b.iter(|| {
                        ls::linear_to_srgb_u16_slice_fast(&s, &mut d);
                        black_box(d[0])
                    })
                });
            }
        });
    }
}

zenbench::main!(bench_downstream);
