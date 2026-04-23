//! SIMD tier consistency tests.
//!
//! Runs sRGB↔linear conversions under every archmage SIMD tier permutation
//! (AVX-512, AVX2, SSE, scalar) and verifies all produce identical output.
//! Catches FMA rounding divergence, accumulator ordering bugs, and
//! vectorization correctness issues.

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

use linear_srgb::default::{
    linear_to_srgb_slice, linear_to_srgb_u8_rgba_slice, linear_to_srgb_u8_slice,
    linear_to_srgb_u16_rgba_slice, linear_to_srgb_u16_rgba_slice_fast, linear_to_srgb_u16_slice,
    linear_to_srgb_u16_slice_fast, srgb_to_linear_slice, srgb_u8_to_linear_slice,
    unpremultiply_linear_to_srgb_u8_rgba_slice,
};

/// Hash a byte slice deterministically (FNV-1a).
fn hash_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Hash a float slice as raw bytes.
fn hash_f32(data: &[f32]) -> u64 {
    hash_bytes(bytemuck::cast_slice(data))
}

/// Generate a sweep of u8 values covering all 256 values repeated N times.
fn generate_u8_input(repeats: usize) -> Vec<u8> {
    (0..repeats).flat_map(|_| 0..=255u8).collect()
}

/// Generate a sweep of f32 values in [0, 1] with fine granularity.
fn generate_f32_input(count: usize) -> Vec<f32> {
    (0..count).map(|i| i as f32 / (count - 1) as f32).collect()
}

#[test]
fn srgb_u8_to_linear_all_tiers_match() {
    let input = generate_u8_input(4); // 1024 values
    let mut reference_hash = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut output = vec![0.0f32; input.len()];
        srgb_u8_to_linear_slice(&input, &mut output);
        let h = hash_f32(&output);

        if let Some(ref_h) = reference_hash {
            assert_eq!(
                h, ref_h,
                "srgb_u8_to_linear output differs under '{}'",
                perm.label,
            );
        } else {
            reference_hash = Some(h);
        }
    });
}

#[test]
fn linear_to_srgb_u8_all_tiers_match() {
    let input = generate_f32_input(4096);
    let mut reference_hash = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut output = vec![0u8; input.len()];
        linear_to_srgb_u8_slice(&input, &mut output);
        let h = hash_bytes(&output);

        if let Some(ref_h) = reference_hash {
            assert_eq!(
                h, ref_h,
                "linear_to_srgb_u8 output differs under '{}'",
                perm.label,
            );
        } else {
            reference_hash = Some(h);
        }
    });
}

/// Max absolute u16 difference between two slices.
fn max_u16_diff(a: &[u16], b: &[u16]) -> u16 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

#[test]
fn linear_to_srgb_u16_all_tiers_within_one_lsb() {
    // Each tier uses its own polynomial evaluator (x8 / x16 / NEON / scalar),
    // which can differ by ~1 f32 ULP. When quantized to u16 this occasionally
    // tips a boundary sample by ±1. Matches the cross-tier behavior of
    // `linear_to_srgb_slice` (compared within 1e-6 f32 ULP), not bit-exact.
    let input = generate_f32_input(4096);
    let mut reference: Option<Vec<u16>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut output = vec![0u16; input.len()];
        linear_to_srgb_u16_slice(&input, &mut output);

        if let Some(ref ref_out) = reference {
            let diff = max_u16_diff(ref_out, &output);
            assert!(
                diff <= 1,
                "linear_to_srgb_u16 under '{}': max_u16_diff={diff} (expected <=1)",
                perm.label,
            );
        } else {
            reference = Some(output);
        }
    });
}

#[test]
fn linear_to_srgb_u8_rgba_all_tiers_match() {
    // 1024 pixels with varied alpha so the dedicated alpha path gets exercised.
    let input: Vec<f32> = (0..4096)
        .map(|i| {
            let t = (i / 4) as f32 / 1024.0;
            match i % 4 {
                0 => t,
                1 => (t * 0.7).min(1.0),
                2 => (1.0 - t).max(0.0),
                3 => ((i / 4) as f32 / 1023.0).clamp(0.0, 1.0),
                _ => unreachable!(),
            }
        })
        .collect();
    let mut reference_hash = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut output = vec![0u8; input.len()];
        linear_to_srgb_u8_rgba_slice(&input, &mut output);
        let h = hash_bytes(&output);

        if let Some(ref_h) = reference_hash {
            assert_eq!(
                h, ref_h,
                "linear_to_srgb_u8_rgba output differs under '{}'",
                perm.label,
            );
        } else {
            reference_hash = Some(h);
        }
    });
}

#[test]
fn unpremultiply_linear_to_srgb_u8_all_tiers_match() {
    // Mix of premultiplied pixels with various alphas, including below and
    // above the UNPREMUL_ALPHA_THRESHOLD (1/1024), to exercise both code
    // paths (divide + LUT, and transparent→zero fallback).
    let mut input: Vec<f32> = Vec::with_capacity(4096);
    for i in 0..1024 {
        let alpha = match i % 8 {
            0 => 0.0,          // fully transparent
            1 => 1e-5,         // below threshold
            2 => 1.0 / 1024.0, // at threshold
            3 => 1.5 / 1024.0, // just above threshold
            4 => 0.25,
            5 => 0.5,
            6 => 0.75,
            7 => 1.0,
            _ => unreachable!(),
        };
        let r = (i as f32 / 1023.0) * alpha;
        let g = ((i * 7 % 1024) as f32 / 1023.0) * alpha;
        let b = ((i * 13 % 1024) as f32 / 1023.0) * alpha;
        input.extend_from_slice(&[r, g, b, alpha]);
    }
    let mut reference_hash = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut output = vec![0u8; input.len()];
        unpremultiply_linear_to_srgb_u8_rgba_slice(&input, &mut output);
        let h = hash_bytes(&output);

        if let Some(ref_h) = reference_hash {
            assert_eq!(
                h, ref_h,
                "unpremultiply_linear_to_srgb_u8 output differs under '{}'",
                perm.label,
            );
        } else {
            reference_hash = Some(h);
        }
    });
}

#[test]
fn linear_to_srgb_u16_fast_all_tiers_match() {
    // Sqrt-indexed LUT gives bit-identical output across tiers (same formula,
    // same LUT; the SIMD sqrt/truncate round deterministically to the same
    // integer index).
    let input = generate_f32_input(4096);
    let mut reference_hash = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut output = vec![0u16; input.len()];
        linear_to_srgb_u16_slice_fast(&input, &mut output);
        let h = hash_bytes(bytemuck::cast_slice(&output));

        if let Some(ref_h) = reference_hash {
            assert_eq!(
                h, ref_h,
                "linear_to_srgb_u16_slice_fast output differs under '{}'",
                perm.label,
            );
        } else {
            reference_hash = Some(h);
        }
    });
}

#[test]
fn linear_to_srgb_u16_rgba_fast_all_tiers_match() {
    let input: Vec<f32> = (0..4096)
        .map(|i| {
            let t = (i / 4) as f32 / 1024.0;
            match i % 4 {
                0 => t,
                1 => (t * 0.7).min(1.0),
                2 => (1.0 - t).max(0.0),
                3 => ((i / 4) as f32 / 1023.0).clamp(0.0, 1.0),
                _ => unreachable!(),
            }
        })
        .collect();
    let mut reference_hash = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut output = vec![0u16; input.len()];
        linear_to_srgb_u16_rgba_slice_fast(&input, &mut output);
        let h = hash_bytes(bytemuck::cast_slice(&output));

        if let Some(ref_h) = reference_hash {
            assert_eq!(
                h, ref_h,
                "linear_to_srgb_u16_rgba_slice_fast output differs under '{}'",
                perm.label,
            );
        } else {
            reference_hash = Some(h);
        }
    });
}

#[test]
fn linear_to_srgb_u16_rgba_all_tiers_within_one_lsb() {
    let input: Vec<f32> = (0..4096)
        .map(|i| {
            let t = (i / 4) as f32 / 1024.0;
            match i % 4 {
                0 => t,
                1 => (t * 0.7).min(1.0),
                2 => (1.0 - t).max(0.0),
                3 => ((i / 4) as f32 / 1023.0).clamp(0.0, 1.0),
                _ => unreachable!(),
            }
        })
        .collect();
    let mut reference: Option<Vec<u16>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut output = vec![0u16; input.len()];
        linear_to_srgb_u16_rgba_slice(&input, &mut output);

        if let Some(ref ref_out) = reference {
            // Alpha (every 4th lane) goes through a direct scale so it should
            // be bit-identical; RGB goes through the polynomial → same 1-LSB
            // tier tolerance as the plain slice.
            for (i, (&r, &o)) in ref_out.iter().zip(output.iter()).enumerate() {
                let diff = r.abs_diff(o);
                if i % 4 == 3 {
                    assert_eq!(
                        r, o,
                        "linear_to_srgb_u16_rgba alpha lane differs under '{}' at {i}",
                        perm.label
                    );
                } else {
                    assert!(
                        diff <= 1,
                        "linear_to_srgb_u16_rgba under '{}' at {i}: diff={diff}",
                        perm.label
                    );
                }
            }
        } else {
            reference = Some(output);
        }
    });
}

/// Max absolute difference between two float slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn srgb_to_linear_f32_all_tiers_within_ulp() {
    let input = generate_f32_input(8192);
    let mut reference: Option<Vec<f32>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut data = input.clone();
        srgb_to_linear_slice(&mut data);

        if let Some(ref ref_data) = reference {
            let max_diff = max_abs_diff(ref_data, &data);
            // FMA vs mul+add rounding: expect <=1 ULP at f32 (~1e-7 relative)
            assert!(
                max_diff < 1e-6,
                "srgb_to_linear_f32 under '{}': max_diff={max_diff} (expected <1e-6)",
                perm.label,
            );
        } else {
            reference = Some(data);
        }
    });
}

#[test]
fn linear_to_srgb_f32_all_tiers_within_ulp() {
    let input = generate_f32_input(8192);
    let mut reference: Option<Vec<f32>> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut data = input.clone();
        linear_to_srgb_slice(&mut data);

        if let Some(ref ref_data) = reference {
            let max_diff = max_abs_diff(ref_data, &data);
            assert!(
                max_diff < 1e-6,
                "linear_to_srgb_f32 under '{}': max_diff={max_diff} (expected <1e-6)",
                perm.label,
            );
        } else {
            reference = Some(data);
        }
    });
}

#[test]
fn roundtrip_u8_all_tiers_match() {
    let input = generate_u8_input(4);
    let mut reference_hash = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut linear = vec![0.0f32; input.len()];
        srgb_u8_to_linear_slice(&input, &mut linear);
        let mut roundtripped = vec![0u8; input.len()];
        linear_to_srgb_u8_slice(&linear, &mut roundtripped);
        let h = hash_bytes(&roundtripped);

        if let Some(ref_h) = reference_hash {
            assert_eq!(
                h, ref_h,
                "roundtrip u8 output differs under '{}'",
                perm.label,
            );
        } else {
            reference_hash = Some(h);
        }
    });
}
