//! 16×f32 `#[rite]` transfer function wrappers (AVX-512 on x86-64).
//!
//! sRGB and PQ use `magetypes::simd::v4::f32x16` (only needs `F32x16Backend`).
//! BT.709 and HLG need integer bitcasts for fast_powf/fast_log2f/fast_pow2f,
//! which `F32x16Backend` doesn't provide — so they use 2×x8 via the existing
//! rites_x8 functions (same AVX2+FMA code, just called twice).

#[cfg(target_arch = "x86_64")]
use archmage::rite;

#[cfg(target_arch = "x86_64")]
pub use archmage::Server64;

#[cfg(target_arch = "x86_64")]
use magetypes::simd::v4::f32x16 as mt_f32x16;

// =============================================================================
// sRGB — rational polynomial, no integer ops needed
// =============================================================================

/// Convert 16 sRGB values to linear (rational polynomial, no powf).
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn srgb_to_linear_v4(token: Server64, v: [f32; 16]) -> [f32; 16] {
    use crate::rational_poly::{LINEAR_SCALE, S2L_P, S2L_Q, SRGB_THRESHOLD};

    let v = mt_f32x16::from_array(token, v);
    let threshold = mt_f32x16::splat(token, SRGB_THRESHOLD);
    let inv_12_92 = mt_f32x16::splat(token, LINEAR_SCALE);

    let linear = v * inv_12_92;
    let poly = eval_rational_poly_x16(token, v, S2L_P, S2L_Q);

    let mask = v.simd_le(threshold);
    mt_f32x16::blend(mask, linear, poly).to_array()
}

/// Convert 16 linear values to sRGB (rational polynomial, no powf).
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_srgb_v4(token: Server64, v: [f32; 16]) -> [f32; 16] {
    use crate::rational_poly::{L2S_P, L2S_Q, LINEAR_THRESHOLD, TWELVE_92};

    let v = mt_f32x16::from_array(token, v);
    let threshold = mt_f32x16::splat(token, LINEAR_THRESHOLD);
    let scale = mt_f32x16::splat(token, TWELVE_92);

    let linear = v * scale;
    let s = v.sqrt();
    let poly = eval_rational_poly_x16(token, s, L2S_P, L2S_Q);

    let mask = v.simd_le(threshold);
    mt_f32x16::blend(mask, linear, poly).to_array()
}

// =============================================================================
// PQ — rational polynomial, no integer ops needed
// =============================================================================

/// Convert 16 PQ signal values to linear.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn pq_to_linear_v4(token: Server64, v: [f32; 16]) -> [f32; 16] {
    use super::pq::{PQ_EOTF_P, PQ_EOTF_Q};

    let v = mt_f32x16::from_array(token, v);
    let zero = mt_f32x16::zero(token);
    let a = v.max(zero);
    let x = a.mul_add(a, a); // x = a + a*a
    let result = eval_rational_poly_x16(token, x, PQ_EOTF_P, PQ_EOTF_Q);
    let mask = v.simd_gt(zero);
    (result & mask).to_array()
}

/// Convert 16 linear values to PQ signal.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_pq_v4(token: Server64, v: [f32; 16]) -> [f32; 16] {
    use super::pq::{PQ_INV_P_LARGE, PQ_INV_P_SMALL, PQ_INV_Q_LARGE, PQ_INV_Q_SMALL};

    let v = mt_f32x16::from_array(token, v);
    let zero = mt_f32x16::zero(token);
    let a = v.max(zero);
    let a4 = a.sqrt().sqrt();

    let threshold = mt_f32x16::splat(token, 0.1);
    let large = eval_rational_poly_x16(token, a4, PQ_INV_P_LARGE, PQ_INV_Q_LARGE);
    let small = eval_rational_poly_x16(token, a4, PQ_INV_P_SMALL, PQ_INV_Q_SMALL);

    let mask = a4.simd_lt(threshold);
    let result = mt_f32x16::blend(mask, small, large);

    let pos_mask = v.simd_gt(zero);
    (result & pos_mask).to_array()
}

// =============================================================================
// BT.709 — needs fast_powf, delegate to 2×x8 rites
// =============================================================================

/// Convert 16 BT.709 encoded values to linear.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn bt709_to_linear_v4(token: Server64, v: [f32; 16]) -> [f32; 16] {
    let t3 = token.v3();
    let lo: [f32; 8] = v[..8].try_into().unwrap();
    let hi: [f32; 8] = v[8..].try_into().unwrap();
    let lo = super::rites_x8::bt709_to_linear_v3(t3, lo);
    let hi = super::rites_x8::bt709_to_linear_v3(t3, hi);
    let mut out = [0.0f32; 16];
    out[..8].copy_from_slice(&lo);
    out[8..].copy_from_slice(&hi);
    out
}

/// Convert 16 linear values to BT.709 encoded.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_bt709_v4(token: Server64, v: [f32; 16]) -> [f32; 16] {
    let t3 = token.v3();
    let lo: [f32; 8] = v[..8].try_into().unwrap();
    let hi: [f32; 8] = v[8..].try_into().unwrap();
    let lo = super::rites_x8::linear_to_bt709_v3(t3, lo);
    let hi = super::rites_x8::linear_to_bt709_v3(t3, hi);
    let mut out = [0.0f32; 16];
    out[..8].copy_from_slice(&lo);
    out[8..].copy_from_slice(&hi);
    out
}

// =============================================================================
// HLG — needs fast_pow2f/fast_log2f, delegate to 2×x8 rites
// =============================================================================

/// Convert 16 HLG signal values to linear.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn hlg_to_linear_v4(token: Server64, v: [f32; 16]) -> [f32; 16] {
    let t3 = token.v3();
    let lo: [f32; 8] = v[..8].try_into().unwrap();
    let hi: [f32; 8] = v[8..].try_into().unwrap();
    let lo = super::rites_x8::hlg_to_linear_v3(t3, lo);
    let hi = super::rites_x8::hlg_to_linear_v3(t3, hi);
    let mut out = [0.0f32; 16];
    out[..8].copy_from_slice(&lo);
    out[8..].copy_from_slice(&hi);
    out
}

/// Convert 16 linear values to HLG signal.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_hlg_v4(token: Server64, v: [f32; 16]) -> [f32; 16] {
    let t3 = token.v3();
    let lo: [f32; 8] = v[..8].try_into().unwrap();
    let hi: [f32; 8] = v[8..].try_into().unwrap();
    let lo = super::rites_x8::linear_to_hlg_v3(t3, lo);
    let hi = super::rites_x8::linear_to_hlg_v3(t3, hi);
    let mut out = [0.0f32; 16];
    out[..8].copy_from_slice(&lo);
    out[8..].copy_from_slice(&hi);
    out
}

// =============================================================================
// Slice functions
// =============================================================================

macro_rules! tf_slice_v4 {
    ($name:ident, $rite:ident, $scalar:path) => {
        #[cfg(target_arch = "x86_64")]
        #[rite]
        pub fn $name(token: Server64, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<16>();
            for chunk in chunks {
                *chunk = $rite(token, *chunk);
            }
            for v in remainder {
                *v = $scalar(*v);
            }
        }
    };
}

tf_slice_v4!(
    srgb_to_linear_slice_v4,
    srgb_to_linear_v4,
    super::srgb_to_linear
);
tf_slice_v4!(
    linear_to_srgb_slice_v4,
    linear_to_srgb_v4,
    super::linear_to_srgb
);
tf_slice_v4!(
    bt709_to_linear_slice_v4,
    bt709_to_linear_v4,
    super::bt709_to_linear
);
tf_slice_v4!(
    linear_to_bt709_slice_v4,
    linear_to_bt709_v4,
    super::linear_to_bt709
);
tf_slice_v4!(pq_to_linear_slice_v4, pq_to_linear_v4, super::pq_to_linear);
tf_slice_v4!(linear_to_pq_slice_v4, linear_to_pq_v4, super::linear_to_pq);
tf_slice_v4!(
    hlg_to_linear_slice_v4,
    hlg_to_linear_v4,
    super::hlg_to_linear
);
tf_slice_v4!(
    linear_to_hlg_slice_v4,
    linear_to_hlg_v4,
    super::linear_to_hlg
);

// =============================================================================
// Internal helper — eval_rational_poly_x16 via magetypes
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn eval_rational_poly_x16(
    t: Server64,
    x: magetypes::simd::v4::f32x16,
    p: [f32; 5],
    q: [f32; 5],
) -> magetypes::simd::v4::f32x16 {
    let mut yp = mt_f32x16::splat(t, p[4]);
    yp = yp.mul_add(x, mt_f32x16::splat(t, p[3]));
    yp = yp.mul_add(x, mt_f32x16::splat(t, p[2]));
    yp = yp.mul_add(x, mt_f32x16::splat(t, p[1]));
    yp = yp.mul_add(x, mt_f32x16::splat(t, p[0]));

    let mut yq = mt_f32x16::splat(t, q[4]);
    yq = yq.mul_add(x, mt_f32x16::splat(t, q[3]));
    yq = yq.mul_add(x, mt_f32x16::splat(t, q[2]));
    yq = yq.mul_add(x, mt_f32x16::splat(t, q[1]));
    yq = yq.mul_add(x, mt_f32x16::splat(t, q[0]));

    yp / yq
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;
    use archmage::SimdToken;

    fn get_token() -> Option<Server64> {
        Server64::summon()
    }

    macro_rules! test_x16_tf {
        ($test_name:ident, $x16_fn:path, $scalar_fn:expr, $tol:expr) => {
            #[test]
            fn $test_name() {
                let Some(token) = get_token() else {
                    eprintln!("Skipping: AVX-512 not available");
                    return;
                };

                #[archmage::arcane]
                fn call(token: Server64, v: [f32; 16]) -> [f32; 16] {
                    $x16_fn(token, v)
                }

                let input = [
                    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9,
                    0.95, 1.0,
                ];
                let result = call(token, input);

                for (i, (&got, &inp)) in result.iter().zip(input.iter()).enumerate() {
                    let expected = $scalar_fn(inp);
                    assert!(
                        (got - expected).abs() < $tol,
                        "mismatch at {i}: got {got}, expected {expected} (input={inp})"
                    );
                }
            }
        };
    }

    test_x16_tf!(
        srgb_to_linear_x16,
        srgb_to_linear_v4,
        super::super::srgb_to_linear,
        1e-5
    );
    test_x16_tf!(
        linear_to_srgb_x16,
        linear_to_srgb_v4,
        super::super::linear_to_srgb,
        1e-5
    );
    test_x16_tf!(
        bt709_to_linear_x16,
        bt709_to_linear_v4,
        super::super::bt709_to_linear,
        1e-5
    );
    test_x16_tf!(
        linear_to_bt709_x16,
        linear_to_bt709_v4,
        super::super::linear_to_bt709,
        1e-4
    );
    test_x16_tf!(
        pq_to_linear_x16,
        pq_to_linear_v4,
        super::super::pq_to_linear,
        1e-5
    );
    test_x16_tf!(
        linear_to_pq_x16,
        linear_to_pq_v4,
        super::super::linear_to_pq,
        1e-5
    );
    test_x16_tf!(
        hlg_to_linear_x16,
        hlg_to_linear_v4,
        super::super::hlg_to_linear,
        1e-4
    );
    test_x16_tf!(
        linear_to_hlg_x16,
        linear_to_hlg_v4,
        super::super::linear_to_hlg,
        1e-4
    );

    #[test]
    fn srgb_roundtrip_x16() {
        let Some(token) = get_token() else {
            eprintln!("Skipping: AVX-512 not available");
            return;
        };

        #[archmage::arcane]
        fn call_s2l(token: Server64, v: [f32; 16]) -> [f32; 16] {
            srgb_to_linear_v4(token, v)
        }
        #[archmage::arcane]
        fn call_l2s(token: Server64, v: [f32; 16]) -> [f32; 16] {
            linear_to_srgb_v4(token, v)
        }

        let input = [
            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0,
        ];
        let linear = call_s2l(token, input);
        let roundtrip = call_l2s(token, linear);

        for (i, (&orig, &rt)) in input.iter().zip(roundtrip.iter()).enumerate() {
            assert!(
                (orig - rt).abs() < 1e-4,
                "sRGB roundtrip failed at {i}: {orig} -> {rt}"
            );
        }
    }

    #[test]
    fn slice_roundtrip_x16() {
        let Some(token) = get_token() else {
            eprintln!("Skipping: AVX-512 not available");
            return;
        };

        #[archmage::arcane]
        fn call_s2l(token: Server64, values: &mut [f32]) {
            srgb_to_linear_slice_v4(token, values);
        }
        #[archmage::arcane]
        fn call_l2s(token: Server64, values: &mut [f32]) {
            linear_to_srgb_slice_v4(token, values);
        }

        let mut values: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        let original = values.clone();

        call_s2l(token, &mut values);
        call_l2s(token, &mut values);

        for (i, (&orig, &conv)) in original.iter().zip(values.iter()).enumerate() {
            assert!(
                (orig - conv).abs() < 1e-4,
                "slice roundtrip failed at {i}: {orig} -> {conv}"
            );
        }
    }
}
