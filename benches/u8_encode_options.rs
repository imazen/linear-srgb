//! In-memory interleaved comparisons, including the fused streaming RGBA path.
#[path = "support/u8_options.rs"]
mod options;
use linear_srgb::default as d;
use std::{hint::black_box, time::Duration};
use zenbench::prelude::*;

fn bench(suite: &mut Suite) {
    let opts: &'static options::Options = Box::leak(Box::new(options::Options::new()));
    for n in [1usize, 7, 8, 64, 1024, 16384] {
        let input: &'static [f32] = Box::leak(
            (0..n)
                .map(|i| ((i * 7349 + 31) % 65536) as f32 / 65535.0)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        suite.compare(format!("encode/{n}"), |g| {
            g.config()
                .min_rounds(30)
                .max_rounds(60)
                .warmup_time(Duration::from_millis(50))
                .max_time(Duration::from_secs(1));
            g.throughput(Throughput::Elements(n as u64));
            for name in [
                "lut12",
                "lut16",
                "corrected",
                "floor_corrected",
                "polynomial",
            ] {
                let mut out = vec![0u8; n];
                g.bench(name, move |b| {
                    b.iter(|| {
                        let src = black_box(input);
                        if name == "lut12" {
                            d::linear_to_srgb_u8_slice(src, &mut out);
                        } else if name == "floor_corrected" {
                            options::floor_corrected_slice(src, &mut out, opts);
                        } else {
                            for (&x, y) in src.iter().zip(&mut out) {
                                *y = match name {
                                    "lut16" => opts.wide(x),
                                    "corrected" => opts.corrected(x),
                                    _ => opts.polynomial(x),
                                };
                            }
                        }
                        black_box(&out);
                    })
                });
            }
        });
        // n is pixels here. zenresize streaming calls this exact public fused
        // function; alternatives use identical unpremultiply/alpha policy.
        let rgba: &'static [f32] = Box::leak(
            input
                .iter()
                .flat_map(|&x| [x * 0.75, x * 0.5, x * 0.25, 0.75])
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        suite.compare(format!("fused_rgba/{n}"), |g| {
            g.config()
                .min_rounds(30)
                .max_rounds(60)
                .warmup_time(Duration::from_millis(50))
                .max_time(Duration::from_secs(1));
            g.throughput(Throughput::Elements(n as u64));
            for name in [
                "lut12",
                "lut16",
                "corrected",
                "floor_corrected",
                "polynomial",
            ] {
                let mut out = vec![0u8; n * 4];
                g.bench(name, move |b| {
                    b.iter(|| {
                        let src = black_box(rgba);
                        if name == "lut12" {
                            d::unpremultiply_linear_to_srgb_u8_rgba_slice(src, &mut out);
                        } else if name == "floor_corrected" {
                            options::floor_corrected_rgba_slice(src, &mut out, opts);
                        } else {
                            for (s, dst) in src
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .zip(out.as_chunks_mut::<4>().0)
                            {
                                let inv = if s[3] > linear_srgb::UNPREMUL_ALPHA_THRESHOLD {
                                    1.0 / s[3]
                                } else {
                                    0.0
                                };
                                for k in 0..3 {
                                    let x = s[k] * inv;
                                    dst[k] = match name {
                                        "lut16" => opts.wide(x),
                                        "corrected" => opts.corrected(x),
                                        "floor_corrected" => opts.floor_corrected(x),
                                        _ => opts.polynomial(x),
                                    };
                                }
                                dst[3] = (s[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                            }
                        }
                        black_box(&out);
                    })
                });
            }
        });
    }
}
zenbench::main!(bench);
