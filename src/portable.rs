//! Scalar companions to magetypes' fused mid-precision vector kernels.
//!
//! Each multiply-add is fused, including when software FMA is needed. With
//! ordinary round-to-nearest floating-point settings, these functions match
//! native FMA vector lanes on their documented domains. They are approximate
//! math, not correctly rounded libm replacements. The magetypes scalar tier
//! uses relaxed multiply-add and can give different bits.
//!
//! Normal positive inputs are required for logarithms. Subnormal inputs,
//! nonfinite values, negative bases and NaN payloads are outside the parity
//! contract. `pow_midp` additionally requires a finite exponent and an
//! intermediate `log2_midp(x) * exponent` in `[-126, 128)` for vector parity.
//! Subnormal exponential results are flushed to zero. Evaluation in
//! `[-126, 127.5)` uses the central degree-6 exponential polynomial; its
//! extrapolation in `[127.5, 128)` has larger error. No uniform pow ULP bound
//! is promised because logarithm error is amplified by the exponent.
//!
//! Software FMA can be costly on baseline x86 and strict WASM. In a hardware
//! FMA target-feature context these inline into hardware operations. See
//! `benchmarks/portable_math_2026-11-05.md` in the repository for measurements.
//!
//! ```
//! use linear_srgb::portable::{exp2_midp, log2_midp, pow_midp};
//! assert_eq!(log2_midp(1.0), 0.0);
//! assert_eq!(exp2_midp(0.0), 1.0);
//! assert_eq!(pow_midp(1.0, 2.4), 1.0);
//! ```

/// Approximate `x.powf(exponent)` using fused vector-compatible scalar math.
///
/// Requires a normal positive base. Vector parity requires the intermediate
/// exponent to lie in `[-126, 128)`. Results below that range become zero;
/// results at or above 128 become infinity. This is not a general IEEE pow.
#[inline(always)]
pub fn pow_midp(x: f32, exponent: f32) -> f32 {
    crate::lanes::pow(x, exponent)
}

/// Approximate base-two logarithm with fused polynomial evaluation.
///
/// Parity is defined for normal positive f32 inputs. Zero returns negative
/// infinity; negative values return NaN; positive infinity returns itself.
/// Subnormal inputs are not normalized and have no accuracy guarantee.
#[inline(always)]
pub fn log2_midp(x: f32) -> f32 {
    crate::lanes::log2(x)
}

/// Approximate base-two exponential with fused polynomial evaluation.
///
/// Vector parity is defined for finite `x` in `[-126, 128)`. Values below
/// -126 return zero, and values at or above 128 return positive infinity.
/// Error increases in `[127.5, 128)`; this function does not construct
/// subnormal results. NaN payloads are not part of the contract.
#[inline(always)]
pub fn exp2_midp(x: f32) -> f32 {
    crate::lanes::exp2(x)
}
