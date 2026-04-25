//! Cross-tier consistency for the f32 RGBA premultiply / unpremultiply families.
//!
//! These four families are the ones Pattern 2 of issue #23 collapsed from 5
//! hand-written tier dispatchers each (V3, V4, NEON, WASM, scalar) into a
//! single `#[magetypes]`-decorated body. The previously-existing tier-match
//! tests covered only the u8 RGBA paths — and a u8 quantization can hide
//! sub-LSB f32 drift between tiers (anything <0.5 LSB rounds to the same
//! u8). These tests close that gap by comparing the f32 outputs directly.
//!
//! Tolerance vs bit-exact: different SIMD ISAs evaluate the same algebraic
//! expression with slightly different rounding (FMA vs separate mul+add
//! being the most common source). 1 ULP at unit magnitude is ~1.2e-7;
//! premultiply (one extra mul) compounds to ~2 ULP; unpremultiply
//! (division by alpha then a transfer-function pow) reaches ~5 ULP near
//! `UNPREMUL_ALPHA_THRESHOLD` because the `1/alpha` factor amplifies
//! upstream error. The bounds below are calibrated to those expectations
//! plus a small safety margin.

// `gamma_to_linear_premultiply_rgba_slice` is deprecated in favor of the
// sRGB-specific variant, but cross-tier consistency still matters for as
// long as it ships — flagging callers, not exempting it from the gate.
#![allow(deprecated)]

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

use linear_srgb::UNPREMUL_ALPHA_THRESHOLD;
use linear_srgb::default::{
    gamma_to_linear_premultiply_rgba_slice, srgb_to_linear_premultiply_rgba_slice,
    unpremultiply_linear_to_gamma_rgba_slice, unpremultiply_linear_to_srgb_rgba_slice,
};

/// Max absolute element-wise difference between two equally-sized slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Build interleaved RGBA pixels with an alpha sweep across the tricky range.
///
/// Includes:
///   - alpha = 0.0 (premultiply → all-zero RGB; unpremultiply skips per threshold)
///   - alpha at and just above `UNPREMUL_ALPHA_THRESHOLD` (the divide branch)
///   - alpha = 0.5, 0.75 (typical compositing values)
///   - alpha = 1.0 (identity-alpha)
///
/// RGB values sweep [0, 1] in 64 steps for each alpha — enough to hit the
/// transfer function's piecewise threshold and the polynomial range.
fn generate_rgba_premultiplied_sweep() -> Vec<f32> {
    let alphas: &[f32] = &[
        0.0,
        UNPREMUL_ALPHA_THRESHOLD * 0.5, // below threshold (zero-out branch in unpremul)
        UNPREMUL_ALPHA_THRESHOLD,       // at threshold
        UNPREMUL_ALPHA_THRESHOLD * 1.5, // just above threshold (worst-case 1/alpha amplification)
        0.01,
        0.1,
        0.25,
        0.5,
        0.75,
        0.999,
        1.0,
    ];
    let rgb_steps: usize = 64;
    let mut out = Vec::with_capacity(rgb_steps * alphas.len() * 4);
    for &a in alphas {
        for i in 0..rgb_steps {
            let v = i as f32 / (rgb_steps - 1) as f32;
            // Premultiplied form: RGB ≤ alpha. Most upstream pipelines maintain
            // this invariant; tests should exercise it.
            let r = v * a;
            let g = (v * 0.5) * a;
            let b = (v * 0.25) * a;
            out.push(r);
            out.push(g);
            out.push(b);
            out.push(a);
        }
    }
    out
}

/// Same sweep but straight-alpha (RGB independent of alpha). Used by the
/// premultiply-input families.
fn generate_rgba_straight_sweep() -> Vec<f32> {
    let alphas: &[f32] = &[0.0, 0.001, 0.1, 0.25, 0.5, 0.75, 0.999, 1.0];
    let rgb_steps: usize = 64;
    let mut out = Vec::with_capacity(rgb_steps * alphas.len() * 4);
    for &a in alphas {
        for i in 0..rgb_steps {
            let v = i as f32 / (rgb_steps - 1) as f32;
            out.push(v);
            out.push(v * 0.5);
            out.push(v * 0.25);
            out.push(a);
        }
    }
    out
}

/// One-ULP-at-unit upper bound for cross-tier f32 drift on a single mul/add
/// chain. Premultiply does `srgb_to_linear(x) * a` — one extra mul on top of
/// the polynomial chain, so allow ~2 ULP at unit magnitude.
const TOL_PREMUL: f32 = 2.5e-7;

/// Unpremultiply does `(x / a)` followed by the transfer function. The
/// division amplifies upstream error by `1/a`. Worst-case is just above
/// `UNPREMUL_ALPHA_THRESHOLD` (a ≈ 1e-3), where 1 ULP can become ~1e-3.
/// Plus the polynomial chain compounds another ~2 ULP. Bound generously.
const TOL_UNPREMUL: f32 = 2e-3;

#[test]
fn srgb_to_linear_premultiply_rgba_all_tiers_within_tolerance() {
    let input = generate_rgba_straight_sweep();
    let mut reference: Option<Vec<f32>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut data = input.clone();
        srgb_to_linear_premultiply_rgba_slice(&mut data);

        // Sanity: alpha is preserved bit-exactly (every 4th element).
        for (i, (a_in, a_out)) in input
            .iter()
            .skip(3)
            .step_by(4)
            .zip(data.iter().skip(3).step_by(4))
            .enumerate()
        {
            assert_eq!(
                a_out.to_bits(),
                a_in.to_bits(),
                "alpha mutated at pixel {i} under '{}'",
                perm.label,
            );
        }

        if let Some(ref ref_data) = reference {
            let diff = max_abs_diff(ref_data, &data);
            assert!(
                diff < TOL_PREMUL,
                "srgb_to_linear_premultiply_rgba_slice under '{}': max_diff={diff} (expected <{TOL_PREMUL})",
                perm.label,
            );
        } else {
            reference = Some(data);
        }
    });
}

#[test]
fn unpremultiply_linear_to_srgb_rgba_all_tiers_within_tolerance() {
    let input = generate_rgba_premultiplied_sweep();
    let mut reference: Option<Vec<f32>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut data = input.clone();
        unpremultiply_linear_to_srgb_rgba_slice(&mut data);

        // Alpha bit-identical even for the below-threshold zero-out branch.
        for (i, (a_in, a_out)) in input
            .iter()
            .skip(3)
            .step_by(4)
            .zip(data.iter().skip(3).step_by(4))
            .enumerate()
        {
            assert_eq!(
                a_out.to_bits(),
                a_in.to_bits(),
                "alpha mutated at pixel {i} under '{}'",
                perm.label,
            );
        }

        if let Some(ref ref_data) = reference {
            let diff = max_abs_diff(ref_data, &data);
            assert!(
                diff < TOL_UNPREMUL,
                "unpremultiply_linear_to_srgb_rgba_slice under '{}': max_diff={diff} (expected <{TOL_UNPREMUL})",
                perm.label,
            );
        } else {
            reference = Some(data);
        }
    });
}

#[test]
fn gamma_to_linear_premultiply_rgba_all_tiers_within_tolerance() {
    let input = generate_rgba_straight_sweep();
    let gamma = 2.2;
    let mut reference: Option<Vec<f32>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut data = input.clone();
        gamma_to_linear_premultiply_rgba_slice(&mut data, gamma);

        for (i, (a_in, a_out)) in input
            .iter()
            .skip(3)
            .step_by(4)
            .zip(data.iter().skip(3).step_by(4))
            .enumerate()
        {
            assert_eq!(
                a_out.to_bits(),
                a_in.to_bits(),
                "alpha mutated at pixel {i} under '{}'",
                perm.label,
            );
        }

        if let Some(ref ref_data) = reference {
            let diff = max_abs_diff(ref_data, &data);
            // pow_midp polynomial is wider-band than sRGB's piecewise; allow a
            // bit more headroom than TOL_PREMUL.
            let tol = 1e-5;
            assert!(
                diff < tol,
                "gamma_to_linear_premultiply_rgba_slice under '{}': max_diff={diff} (expected <{tol})",
                perm.label,
            );
        } else {
            reference = Some(data);
        }
    });
}

#[test]
fn unpremultiply_linear_to_gamma_rgba_all_tiers_within_tolerance() {
    let input = generate_rgba_premultiplied_sweep();
    let gamma = 2.2;
    let mut reference: Option<Vec<f32>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut data = input.clone();
        unpremultiply_linear_to_gamma_rgba_slice(&mut data, gamma);

        for (i, (a_in, a_out)) in input
            .iter()
            .skip(3)
            .step_by(4)
            .zip(data.iter().skip(3).step_by(4))
            .enumerate()
        {
            assert_eq!(
                a_out.to_bits(),
                a_in.to_bits(),
                "alpha mutated at pixel {i} under '{}'",
                perm.label,
            );
        }

        if let Some(ref ref_data) = reference {
            let diff = max_abs_diff(ref_data, &data);
            // Same TOL_UNPREMUL rationale as the sRGB version: 1/alpha
            // amplification near threshold is the dominant error source.
            assert!(
                diff < TOL_UNPREMUL,
                "unpremultiply_linear_to_gamma_rgba_slice under '{}': max_diff={diff} (expected <{TOL_UNPREMUL})",
                perm.label,
            );
        } else {
            reference = Some(data);
        }
    });
}

// ---------------------------------------------------------------------------
// Plain (non-premultiplied) f32 RGBA paths — Pattern 2 also touched these
// (srgb_to_linear_rgba_slice / linear_to_srgb_rgba_slice). The existing
// `srgb_to_linear_f32_all_tiers_within_ulp` test in simd_consistency.rs
// covers the non-RGBA shape; this fills the RGBA-with-alpha-preserved gap.
// ---------------------------------------------------------------------------

#[test]
fn srgb_to_linear_rgba_all_tiers_within_tolerance() {
    use linear_srgb::default::srgb_to_linear_rgba_slice;
    let input = generate_rgba_straight_sweep();
    let mut reference: Option<Vec<f32>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut data = input.clone();
        srgb_to_linear_rgba_slice(&mut data);

        for (i, (a_in, a_out)) in input
            .iter()
            .skip(3)
            .step_by(4)
            .zip(data.iter().skip(3).step_by(4))
            .enumerate()
        {
            assert_eq!(
                a_out.to_bits(),
                a_in.to_bits(),
                "alpha mutated at pixel {i} under '{}'",
                perm.label,
            );
        }

        if let Some(ref ref_data) = reference {
            let diff = max_abs_diff(ref_data, &data);
            // No multiplication by alpha — same tolerance as the plain path.
            let tol = 1e-6;
            assert!(
                diff < tol,
                "srgb_to_linear_rgba_slice under '{}': max_diff={diff} (expected <{tol})",
                perm.label,
            );
        } else {
            reference = Some(data);
        }
    });
}

#[test]
fn linear_to_srgb_rgba_all_tiers_within_tolerance() {
    use linear_srgb::default::linear_to_srgb_rgba_slice;
    let input = generate_rgba_straight_sweep();
    let mut reference: Option<Vec<f32>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut data = input.clone();
        linear_to_srgb_rgba_slice(&mut data);

        for (i, (a_in, a_out)) in input
            .iter()
            .skip(3)
            .step_by(4)
            .zip(data.iter().skip(3).step_by(4))
            .enumerate()
        {
            assert_eq!(
                a_out.to_bits(),
                a_in.to_bits(),
                "alpha mutated at pixel {i} under '{}'",
                perm.label,
            );
        }

        if let Some(ref ref_data) = reference {
            let diff = max_abs_diff(ref_data, &data);
            let tol = 1e-6;
            assert!(
                diff < tol,
                "linear_to_srgb_rgba_slice under '{}': max_diff={diff} (expected <{tol})",
                perm.label,
            );
        } else {
            reference = Some(data);
        }
    });
}
