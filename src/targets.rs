//! SIMD target definitions for multiversion dispatch.
//!
//! Provides macros for consistent SIMD target specification.

// ============================================================================
// x86/x86_64 macros
// ============================================================================

/// Primary SIMD targets for most functions (x86_64 version).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_export]
macro_rules! simd_multiversion {
    ($($item:tt)*) => {
        #[multiversion::multiversion(targets(
            // x86-64-v3 (Haswell 2013+, Zen 2 2019+)
            "x86_64+sse+sse2+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+fxsr",
        ))]
        $($item)*
    };
}

/// Extended SIMD targets (x86_64 version).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_export]
macro_rules! simd_multiversion_extended {
    ($($item:tt)*) => {
        #[multiversion::multiversion(targets(
            // x86-64-v4 with AVX-512 (Skylake-X 2017+, Zen 4 2022+)
            "x86_64+sse+sse2+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+fxsr+avx512f+avx512bw+avx512dq+avx512vl+avx512cd+gfni+vaes+vpclmulqdq",
            // x86-64-v3 (Haswell 2013+)
            "x86_64+sse+sse2+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+fxsr",
        ))]
        $($item)*
    };
}

/// Full SIMD targets (x86_64 version).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_export]
macro_rules! simd_multiversion_full {
    ($($item:tt)*) => {
        #[multiversion::multiversion(targets(
            // x86-64-v4 with AVX-512 (Skylake-X 2017+, Zen 4 2022+)
            "x86_64+sse+sse2+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+fxsr+avx512f+avx512bw+avx512dq+avx512vl+avx512cd+gfni+vaes+vpclmulqdq",
            // x86-64-v3 (Haswell 2013+)
            "x86_64+sse+sse2+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+fxsr",
        ))]
        $($item)*
    };
}

// ============================================================================
// aarch64 macros
// ============================================================================

/// Primary SIMD targets for most functions (aarch64 version).
#[cfg(target_arch = "aarch64")]
#[macro_export]
macro_rules! simd_multiversion {
    ($($item:tt)*) => {
        #[multiversion::multiversion(targets(
            // aarch64 baseline (all ARM64)
            "aarch64+neon+lse+aes+sha2+crc",
        ))]
        $($item)*
    };
}

/// Extended SIMD targets (aarch64 version).
#[cfg(target_arch = "aarch64")]
#[macro_export]
macro_rules! simd_multiversion_extended {
    ($($item:tt)*) => {
        #[multiversion::multiversion(targets(
            // aarch64 with dotprod (A75 2017+, Apple A11+)
            "aarch64+neon+lse+aes+sha2+crc+dotprod+rcpc+fp16+fhm",
            // aarch64 baseline
            "aarch64+neon+lse+aes+sha2+crc",
        ))]
        $($item)*
    };
}

/// Full SIMD targets (aarch64 version).
#[cfg(target_arch = "aarch64")]
#[macro_export]
macro_rules! simd_multiversion_full {
    ($($item:tt)*) => {
        #[multiversion::multiversion(targets(
            // aarch64 with SVE2 (Neoverse V1 2020+, Apple M4 2024+)
            "aarch64+neon+lse+aes+sha2+crc+dotprod+rcpc+fp16+fhm+sve2+sve2-bitperm+i8mm+bf16",
            // aarch64 with sha3/fcma (A76 2018+, Apple M1+)
            "aarch64+neon+lse+aes+sha2+sha3+crc+dotprod+rcpc+fp16+fhm+fcma",
            // aarch64 with dotprod (A75 2017+, Apple A11+)
            "aarch64+neon+lse+aes+sha2+crc+dotprod+rcpc+fp16+fhm",
            // aarch64 baseline
            "aarch64+neon+lse+aes+sha2+crc",
        ))]
        $($item)*
    };
}

// ============================================================================
// Fallback for other architectures (wasm32, etc.)
// ============================================================================

/// Primary SIMD targets (fallback - no multiversion).
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
#[macro_export]
macro_rules! simd_multiversion {
    ($($item:tt)*) => {
        $($item)*
    };
}

/// Extended SIMD targets (fallback - no multiversion).
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
#[macro_export]
macro_rules! simd_multiversion_extended {
    ($($item:tt)*) => {
        $($item)*
    };
}

/// Full SIMD targets (fallback - no multiversion).
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
#[macro_export]
macro_rules! simd_multiversion_full {
    ($($item:tt)*) => {
        $($item)*
    };
}

// Macros are exported at crate root via #[macro_export]
