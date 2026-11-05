//! Shared experiment code: native, relaxed scalar and portable scalar arithmetic.
#![allow(dead_code)]
use archmage::{ScalarToken, incant};
use linear_srgb::portable as p;

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Log2,
    Exp2,
    Pow,
}

#[inline(always)]
pub fn one(x: f32, op: Op) -> f32 {
    match op {
        Op::Log2 => p::log2_midp(x),
        Op::Exp2 => p::exp2_midp(x),
        Op::Pow => p::pow_midp(x, 2.4),
    }
}

#[inline(always)]
pub fn portable(input: &[f32], output: &mut [f32], op: Op) {
    for (&x, y) in input.iter().zip(output) {
        *y = one(x, op);
    }
}

#[archmage::magetypes(define(f32x16), v4(cfg(avx512)), v3, neon, scalar)]
fn vector_tier(token: Token, input: &[f32], output: &mut [f32], op: Op) {
    for (src, dst) in input.chunks(16).zip(output.chunks_mut(16)) {
        let x = f32x16::from_array_t(
            token,
            core::array::from_fn(|i| src.get(i).copied().unwrap_or(1.0)),
        );
        let result = match op {
            Op::Log2 => x.log2_midp_unchecked(),
            Op::Exp2 => x.exp2_midp_unchecked(),
            Op::Pow => x.pow_midp_unchecked(2.4),
        }
        .to_array();
        dst.copy_from_slice(&result[..dst.len()]);
    }
}

pub fn native(input: &[f32], output: &mut [f32], op: Op) {
    incant!(vector_tier(input, output, op), [v4, v3, neon, scalar]);
}

pub fn relaxed_scalar(input: &[f32], output: &mut [f32], op: Op) {
    vector_tier_scalar(ScalarToken::from_context(), input, output, op);
}

#[archmage::magetypes(v3, neon, scalar)]
fn portable_tier(_token: Token, input: &[f32], output: &mut [f32], op: Op) {
    portable(input, output, op);
}

pub fn portable_hardware(input: &[f32], output: &mut [f32], op: Op) {
    incant!(portable_tier(input, output, op), [v3, neon, scalar]);
}

pub fn samples(op: Op, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            match op {
                Op::Exp2 => (-126.0 + 254.0 * t) as f32,
                Op::Log2 => 2f64.powf(-120.0 + 240.0 * t) as f32,
                // Keep n*log2(x) inside the normal exp2 construction domain.
                Op::Pow => 2f64.powf(-50.0 + 100.0 * t) as f32,
            }
        })
        .collect()
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn native_v3(token: archmage::X64V3Token, input: &[f32], output: &mut [f32], op: Op) {
    vector_tier_v3(token, input, output, op);
}
