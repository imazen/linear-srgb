//! Experimental encoders; no library API or default changes.
use archmage::prelude::*;
use linear_srgb::default as d;

pub const A: f64 = 0.055_010_718_947_586_6;
pub fn decode(v: f64) -> f64 {
    if v <= 12.92 * 0.003_041_282_560_127_521 {
        v / 12.92
    } else {
        ((v + A) / (1.0 + A)).powf(2.4)
    }
}
pub fn reference(v: f32) -> u8 {
    let v = f64::from(v).clamp(0.0, 1.0);
    let y = if v <= 0.003_041_282_560_127_521 {
        v * 12.92
    } else {
        (1.0 + A) * v.powf(1.0 / 2.4) - A
    };
    (y * 255.0 + 0.5) as u8
}
pub struct Options {
    pub wide: Box<[u8]>,
    pub thresholds: [f32; 255],
    pub lower: Box<[u8]>,
    pub upper: [f32; 256],
}
impl Options {
    pub fn new() -> Self {
        let wide = (0..65536).map(|i| reference(i as f32 / 65535.0)).collect();
        let thresholds = core::array::from_fn(|i| {
            let boundary = decode((i as f64 + 0.5) / 255.0);
            let mut x = boundary as f32;
            // Store the first f32 on the upper side of the mathematical boundary.
            if f64::from(x) < boundary {
                x = x.next_up();
            }
            x
        });
        let lower = (0..4096).map(|i| reference(i as f32 / 4096.0)).collect();
        let upper = core::array::from_fn(|i| {
            if i == 255 {
                f32::INFINITY
            } else {
                thresholds[i]
            }
        });
        Self {
            wide,
            thresholds,
            lower,
            upper,
        }
    }
    #[inline]
    pub fn wide(&self, x: f32) -> u8 {
        self.wide[(x.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize]
    }
    #[inline]
    pub fn corrected(&self, x: f32) -> u8 {
        let q = d::linear_to_srgb_u8(x) as usize;
        if q < 255 && x >= self.thresholds[q] {
            (q + 1) as u8
        } else if q > 0 && x < self.thresholds[q - 1] {
            (q - 1) as u8
        } else {
            q as u8
        }
    }
    #[inline]
    pub fn polynomial(&self, x: f32) -> u8 {
        (f64::from(d::linear_to_srgb(x)) * 255.0 + 0.5) as u8
    }
    #[inline]
    pub fn floor_corrected(&self, x: f32) -> u8 {
        let x = x.clamp(0.0, 1.0);
        let q = self.lower[((x * 4096.0) as usize).min(4095)];
        q + u8::from(x >= self.upper[q as usize])
    }
}

// Each floor-indexed cell spans less than one u8 code even at the toe
// (12.92 * 255 / 4096 < 1). Starting at its lower endpoint needs at most
// one upward threshold comparison, without a downward correction.
#[archmage::magetypes(define(f32x16), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn floor_corrected_tier(
    token: Token,
    input: &[f32],
    output: &mut [u8],
    options: &Options,
    rgba: bool,
) {
    let (src, rem) = input.as_chunks::<16>();
    let (dst, dst_rem) = output.as_chunks_mut::<16>();
    for (s, d) in src.iter().zip(dst) {
        let mut values = *s;
        if rgba {
            for pixel in values.as_chunks_mut::<4>().0 {
                let inv = if pixel[3] > linear_srgb::UNPREMUL_ALPHA_THRESHOLD {
                    1.0 / pixel[3]
                } else {
                    0.0
                };
                for channel in &mut pixel[..3] {
                    *channel *= inv;
                }
            }
        }
        let x = f32x16::from_array_t(token, values)
            .max(f32x16::splat_t(token, 0.0))
            .min(f32x16::splat_t(token, 1.0));
        let index = (x * f32x16::splat_t(token, 4096.0)).to_i32().to_array();
        let q: [u8; 16] = core::array::from_fn(|i| options.lower[(index[i] as usize).min(4095)]);
        let upper = core::array::from_fn(|i| options.upper[q[i] as usize]);
        let correction = x
            .simd_ge(f32x16::from_array_t(token, upper))
            .bitcast_to_i32()
            .to_array();
        for i in 0..16 {
            d[i] = (i32::from(q[i]) - correction[i]) as u8;
        }
        if rgba {
            for i in [3, 7, 11, 15] {
                d[i] = alpha(s[i]);
            }
        }
    }
    if rgba {
        for (s, d) in rem
            .as_chunks::<4>()
            .0
            .iter()
            .zip(dst_rem.as_chunks_mut::<4>().0)
        {
            let inv = if s[3] > linear_srgb::UNPREMUL_ALPHA_THRESHOLD {
                1.0 / s[3]
            } else {
                0.0
            };
            for i in 0..3 {
                d[i] = options.floor_corrected(s[i] * inv);
            }
            d[3] = alpha(s[3]);
        }
    } else {
        for (&s, d) in rem.iter().zip(dst_rem) {
            *d = options.floor_corrected(s);
        }
    }
}

pub fn floor_corrected_slice(input: &[f32], output: &mut [u8], options: &Options) {
    incant!(
        floor_corrected_tier(input, output, options, false),
        [v4, v3, neon, wasm128, scalar]
    );
}

pub fn floor_corrected_rgba_slice(input: &[f32], output: &mut [u8], options: &Options) {
    incant!(
        floor_corrected_tier(input, output, options, true),
        [v4, v3, neon, wasm128, scalar]
    );
}

/// The production RGBA quantizer's widened scale/bias policy.
#[inline(always)]
pub fn alpha(x: f32) -> u8 {
    (f64::from(x).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
