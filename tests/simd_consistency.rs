//! SIMD tier consistency tests.
//!
//! Runs sRGB↔linear conversions under every archmage SIMD tier permutation
//! (AVX-512, AVX2, SSE, scalar) and verifies all produce identical output.
//! Catches FMA rounding divergence, accumulator ordering bugs, and
//! vectorization correctness issues.

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

use linear_srgb::default::{
    linear_to_srgb_slice, linear_to_srgb_u8_slice, srgb_to_linear_slice, srgb_u8_to_linear_slice,
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
