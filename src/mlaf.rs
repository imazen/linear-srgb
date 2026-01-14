//! Fused multiply-add abstraction with compile-time hardware detection.
//!
//! When FMA is available (x86 with FMA feature or ARM64 NEON), uses single-cycle
//! hardware FMA instructions. Otherwise falls back to separate multiply+add.

#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    all(target_arch = "aarch64", target_feature = "neon")
))]
use num_traits::MulAdd;

/// Computes `acc + a * b` using FMA when available.
///
/// # Hardware Detection
/// - x86/x86_64 with FMA feature: uses `_mm_fmadd_*` intrinsics via `MulAdd`
/// - aarch64 with NEON: uses `vfma*` intrinsics via `MulAdd`
/// - Otherwise: falls back to `acc + a * b`
#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    all(target_arch = "aarch64", target_feature = "neon")
))]
#[inline(always)]
pub fn mlaf<T: MulAdd<T, Output = T>>(acc: T, a: T, b: T) -> T {
    MulAdd::mul_add(a, b, acc)
}

/// Computes `acc + a * b` (fallback without hardware FMA).
#[cfg(not(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    all(target_arch = "aarch64", target_feature = "neon")
)))]
#[inline(always)]
pub fn mlaf<T: core::ops::Add<Output = T> + core::ops::Mul<Output = T>>(acc: T, a: T, b: T) -> T {
    acc + a * b
}

/// Computes `acc - a * b` using FMA when available.
#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    all(target_arch = "aarch64", target_feature = "neon")
))]
#[inline(always)]
pub fn neg_mlaf<T: MulAdd<T, Output = T> + core::ops::Neg<Output = T>>(acc: T, a: T, b: T) -> T {
    mlaf(acc, a, -b)
}

/// Computes `acc - a * b` (fallback without hardware FMA).
#[cfg(not(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    all(target_arch = "aarch64", target_feature = "neon")
)))]
#[inline(always)]
pub fn neg_mlaf<
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + core::ops::Neg<Output = T>,
>(
    acc: T,
    a: T,
    b: T,
) -> T {
    acc + a * (-b)
}

/// Computes `a * b + c` (reordered FMA for ergonomics).
///
/// This matches the mathematical notation `a * b + c` more naturally.
#[inline(always)]
pub fn fmla<T>(a: T, b: T, c: T) -> T
where
    T: MlaCompatible,
{
    T::mla(c, a, b)
}

/// Trait for types that support MLA operations.
/// This abstracts over the cfg-dependent mlaf implementations.
pub trait MlaCompatible: Sized {
    fn mla(acc: Self, a: Self, b: Self) -> Self;
}

impl MlaCompatible for f32 {
    #[inline(always)]
    fn mla(acc: Self, a: Self, b: Self) -> Self {
        mlaf(acc, a, b)
    }
}

impl MlaCompatible for f64 {
    #[inline(always)]
    fn mla(acc: Self, a: Self, b: Self) -> Self {
        mlaf(acc, a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlaf_f32() {
        let result = mlaf(1.0f32, 2.0f32, 3.0f32);
        assert!((result - 7.0f32).abs() < 1e-6);
    }

    #[test]
    fn test_mlaf_f64() {
        let result = mlaf(1.0f64, 2.0f64, 3.0f64);
        assert!((result - 7.0f64).abs() < 1e-12);
    }

    #[test]
    fn test_neg_mlaf_f32() {
        let result = neg_mlaf(10.0f32, 2.0f32, 3.0f32);
        assert!((result - 4.0f32).abs() < 1e-6);
    }

    #[test]
    fn test_fmla_f32() {
        let result = fmla(2.0f32, 3.0f32, 1.0f32);
        assert!((result - 7.0f32).abs() < 1e-6);
    }
}
