//! Transfer function implementations for sRGB, BT.709, PQ, and HLG.
//!
//! This module provides scalar functions for all four transfer curves.
//! SIMD `#[rite]` versions are in [`crate::tokens`] (x4/x8/x16).
//!
//! # Scalar functions
//!
//! | Function | Transfer | Method |
//! |---|---|---|
//! | `srgb_to_linear` | sRGB | rational polynomial |
//! | `linear_to_srgb` | sRGB | rational polynomial |
//! | `bt709_to_linear` | BT.709 | fast_powf approximation |
//! | `linear_to_bt709` | BT.709 | fast_powf approximation |
//! | `pq_to_linear` | PQ ST 2084 | rational polynomial |
//! | `linear_to_pq` | PQ ST 2084 | rational polynomial |
//! | `hlg_to_linear` | HLG ARIB STD-B67 | fast_pow2f approximation |
//! | `linear_to_hlg` | HLG ARIB STD-B67 | fast_log2f approximation |
//!
//! # Accuracy
//!
//! | TF | Max error vs f64 | Within u8? | Within u16? |
//! |---|---|---|---|
//! | sRGB | ~5e-7 | Yes | Yes |
//! | BT.709 forward | ~3e-7 | Yes | Yes |
//! | BT.709 inverse | ~3e-5 (fast_powf) | Yes | Yes (1 LSB = 1.5e-5) |
//! | PQ forward | ~7e-7 | Yes | Yes |
//! | PQ inverse | ~3e-6 | Yes | Yes |
//! | HLG | ~5e-6 | Yes | Yes |

// Submodules with implementations
pub(crate) mod bt709;
pub(crate) mod fast_math;
pub(crate) mod hlg;
pub(crate) mod pq;
pub(crate) mod srgb;

// SIMD rites for TFs are now in `crate::tokens::{x4, x8, x16}` (behind `transfer` feature).

// =============================================================================
// Scalar re-exports
// =============================================================================

/// sRGB EOTF: encoded → linear. Rational polynomial, max error ~5e-7.
///
/// Uses libjxl rational polynomial — no `powf()` calls.
/// Equivalent to `crate::rational_poly::srgb_to_linear_fast`.
#[inline(always)]
pub fn srgb_to_linear(v: f32) -> f32 {
    crate::rational_poly::srgb_to_linear_fast(v)
}

/// sRGB inverse EOTF: linear → encoded. Rational polynomial, max error ~5e-7.
///
/// Uses sqrt + libjxl rational polynomial — no `powf()` calls.
/// Equivalent to `crate::rational_poly::linear_to_srgb_fast`.
#[inline(always)]
pub fn linear_to_srgb(v: f32) -> f32 {
    crate::rational_poly::linear_to_srgb_fast(v)
}

pub use bt709::{bt709_to_linear, linear_to_bt709};
pub use hlg::{hlg_to_linear, linear_to_hlg};
pub use pq::{linear_to_pq, pq_to_linear};

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn srgb_to_linear_f64(v: f64) -> f64 {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    }

    fn srgb_from_linear_f64(v: f64) -> f64 {
        if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    }

    fn pq_to_linear_f64(v: f64) -> f64 {
        if v <= 0.0 {
            return 0.0;
        }
        let m1: f64 = 0.1593017578125;
        let m2: f64 = 78.84375;
        let c1: f64 = 0.8359375;
        let c2: f64 = 18.8515625;
        let c3: f64 = 18.6875;
        let vp = v.powf(1.0 / m2);
        let num = (vp - c1).max(0.0);
        let den = c2 - c3 * vp;
        if den <= 0.0 {
            return 1.0;
        }
        (num / den).powf(1.0 / m1)
    }

    fn pq_from_linear_f64(v: f64) -> f64 {
        if v <= 0.0 {
            return 0.0;
        }
        let m1: f64 = 0.1593017578125;
        let m2: f64 = 78.84375;
        let c1: f64 = 0.8359375;
        let c2: f64 = 18.8515625;
        let c3: f64 = 18.6875;
        let vp = v.powf(m1);
        let num = c1 + c2 * vp;
        let den = 1.0 + c3 * vp;
        (num / den).powf(m2)
    }

    fn bt709_to_linear_f64(v: f64) -> f64 {
        let beta: f64 = 0.018053968510807;
        if v < 4.5 * beta {
            v / 4.5
        } else {
            let alpha: f64 = 0.09929682680944;
            ((v + alpha) / (1.0 + alpha)).powf(1.0 / 0.45)
        }
    }

    fn bt709_from_linear_f64(v: f64) -> f64 {
        let beta: f64 = 0.018053968510807;
        let alpha: f64 = 0.09929682680944;
        if v < beta {
            4.5 * v
        } else {
            (1.0 + alpha) * v.powf(0.45) - alpha
        }
    }

    fn hlg_to_linear_f64(v: f64) -> f64 {
        let a: f64 = 0.17883277;
        let b: f64 = 0.28466892;
        let c: f64 = 0.55991073;
        if v <= 0.0 {
            0.0
        } else if v <= 0.5 {
            (v * v) / 3.0
        } else {
            (((v - c) / a).exp() + b) / 12.0
        }
    }

    fn hlg_from_linear_f64(v: f64) -> f64 {
        let a: f64 = 0.17883277;
        let b: f64 = 0.28466892;
        let c: f64 = 0.55991073;
        if v <= 0.0 {
            0.0
        } else if v <= 1.0 / 12.0 {
            (3.0 * v).sqrt()
        } else {
            a * (12.0 * v - b).ln() + c
        }
    }

    fn max_abs_error(
        fast: impl Fn(f32) -> f32,
        reference: impl Fn(f64) -> f64,
        range: core::ops::RangeInclusive<f32>,
        steps: usize,
    ) -> (f32, f32) {
        let mut max_err: f64 = 0.0;
        let mut worst_input = 0.0f32;
        let lo = *range.start();
        let hi = *range.end();
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let v = lo + (hi - lo) * t;
            let fast_val = fast(v) as f64;
            let ref_val = reference(v as f64);
            let err = (fast_val - ref_val).abs();
            if err > max_err {
                max_err = err;
                worst_input = v;
            }
        }
        (max_err as f32, worst_input)
    }

    #[test]
    fn srgb_to_linear_accuracy() {
        let (err, worst) = max_abs_error(srgb_to_linear, srgb_to_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("sRGB to_linear max error: {err:.2e} at {worst}");
        assert!(
            err < 5e-6,
            "sRGB to_linear error {err:.2e} too high at {worst}"
        );
    }

    #[test]
    fn srgb_from_linear_accuracy() {
        let (err, worst) = max_abs_error(linear_to_srgb, srgb_from_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("sRGB from_linear max error: {err:.2e} at {worst}");
        assert!(
            err < 5e-6,
            "sRGB from_linear error {err:.2e} too high at {worst}"
        );
    }

    #[test]
    fn pq_to_linear_accuracy() {
        let (err, worst) = max_abs_error(pq_to_linear, pq_to_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("PQ to_linear max error: {err:.2e} at {worst}");
        assert!(
            err < 5e-5,
            "PQ to_linear error {err:.2e} too high at {worst}"
        );
    }

    #[test]
    fn pq_from_linear_accuracy() {
        let (err, worst) = max_abs_error(linear_to_pq, pq_from_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("PQ from_linear max error: {err:.2e} at {worst}");
        assert!(
            err < 5e-5,
            "PQ from_linear error {err:.2e} too high at {worst}"
        );
    }

    #[test]
    fn bt709_to_linear_accuracy() {
        let (err, worst) = max_abs_error(bt709_to_linear, bt709_to_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("BT.709 to_linear max error: {err:.2e} at {worst}");
        assert!(
            err < 5e-5,
            "BT.709 to_linear error {err:.2e} too high at {worst}"
        );
    }

    #[test]
    fn bt709_from_linear_accuracy() {
        let (err, worst) =
            max_abs_error(linear_to_bt709, bt709_from_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("BT.709 from_linear max error: {err:.2e} at {worst}");
        assert!(
            err < 5e-4,
            "BT.709 from_linear error {err:.2e} too high at {worst}"
        );
    }

    #[test]
    fn hlg_to_linear_accuracy() {
        let (err, worst) = max_abs_error(hlg_to_linear, hlg_to_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("HLG to_linear max error: {err:.2e} at {worst}");
        assert!(
            err < 5e-4,
            "HLG to_linear error {err:.2e} too high at {worst}"
        );
    }

    #[test]
    fn hlg_from_linear_accuracy() {
        let (err, worst) = max_abs_error(linear_to_hlg, hlg_from_linear_f64, 0.0..=1.0, 100_000);
        eprintln!("HLG from_linear max error: {err:.2e} at {worst}");
        assert!(
            err < 5e-4,
            "HLG from_linear error {err:.2e} too high at {worst}"
        );
    }

    #[test]
    fn srgb_roundtrip() {
        for i in 0..=255 {
            let encoded = i as f32 / 255.0;
            let linear = srgb_to_linear(encoded);
            let back = linear_to_srgb(linear);
            let err = (back - encoded).abs();
            assert!(
                err < 1.0 / 255.0,
                "sRGB roundtrip failed at {i}: {encoded} -> {linear} -> {back} (err={err})"
            );
        }
    }

    #[test]
    fn pq_roundtrip() {
        for i in 50..=1000 {
            let v = i as f32 / 1000.0;
            let linear = pq_to_linear(v);
            let back = linear_to_pq(linear);
            let err = (back - v).abs();
            assert!(
                err < 0.001,
                "PQ roundtrip failed at {v}: -> {linear} -> {back} (err={err})"
            );
        }
    }

    #[test]
    fn hlg_roundtrip() {
        for i in 0..=1000 {
            let v = i as f32 / 1000.0;
            let linear = hlg_to_linear(v);
            let back = linear_to_hlg(linear);
            let err = (back - v).abs();
            assert!(
                err < 0.01,
                "HLG roundtrip failed at {v}: -> {linear} -> {back} (err={err})"
            );
        }
    }

    #[test]
    fn fast_log2f_accuracy() {
        for i in 1..=10000 {
            let x = i as f32 / 10000.0;
            let fast = fast_math::fast_log2f(x);
            let exact = (x as f64).log2() as f32;
            let err = (fast - exact).abs();
            assert!(
                err < 0.01,
                "fast_log2f({x}) = {fast}, expected {exact} (err={err})"
            );
        }
    }

    #[test]
    fn fast_pow2f_accuracy() {
        for i in -100..=100 {
            let x = i as f32 / 10.0;
            let fast = fast_math::fast_pow2f(x);
            let exact = (x as f64).exp2() as f32;
            let rel_err = if exact.abs() > 1e-10 {
                ((fast - exact) / exact).abs()
            } else {
                (fast - exact).abs()
            };
            assert!(
                rel_err < 0.001,
                "fast_pow2f({x}) = {fast}, expected {exact} (rel_err={rel_err})"
            );
        }
    }

    // --- x8 SIMD tests (AVX2) ---

    #[cfg(target_arch = "x86_64")]
    mod x8_tests {
        use super::*;
        use archmage::SimdToken;

        fn get_token() -> Option<archmage::X64V3Token> {
            archmage::X64V3Token::try_new()
        }

        macro_rules! test_x8_tf {
            ($test_name:ident, $x8_fn:path, $scalar_fn:expr, $tol:expr) => {
                #[test]
                fn $test_name() {
                    let Some(token) = get_token() else {
                        eprintln!("Skipping: AVX2+FMA not available");
                        return;
                    };

                    #[archmage::arcane]
                    fn call(token: archmage::X64V3Token, v: [f32; 8]) -> [f32; 8] {
                        $x8_fn(token, v)
                    }

                    let input = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
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

        test_x8_tf!(
            srgb_to_linear_x8,
            crate::tokens::x8::tf_srgb_to_linear_v3,
            srgb_to_linear,
            1e-5
        );
        test_x8_tf!(
            linear_to_srgb_x8,
            crate::tokens::x8::tf_linear_to_srgb_v3,
            linear_to_srgb,
            1e-5
        );
        test_x8_tf!(
            bt709_to_linear_x8,
            crate::tokens::x8::bt709_to_linear_v3,
            bt709_to_linear,
            1e-5
        );
        test_x8_tf!(
            linear_to_bt709_x8,
            crate::tokens::x8::linear_to_bt709_v3,
            linear_to_bt709,
            1e-4
        );
        test_x8_tf!(
            pq_to_linear_x8,
            crate::tokens::x8::pq_to_linear_v3,
            pq_to_linear,
            1e-5
        );
        test_x8_tf!(
            linear_to_pq_x8,
            crate::tokens::x8::linear_to_pq_v3,
            linear_to_pq,
            1e-5
        );
        test_x8_tf!(
            hlg_to_linear_x8,
            crate::tokens::x8::hlg_to_linear_v3,
            hlg_to_linear,
            1e-4
        );
        test_x8_tf!(
            linear_to_hlg_x8,
            crate::tokens::x8::linear_to_hlg_v3,
            linear_to_hlg,
            1e-4
        );
    }
}
