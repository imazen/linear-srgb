//! 4×f32 `#[rite]` transfer function wrappers (NEON on AArch64, SIMD128 on WebAssembly).
//!
//! Each function converts `[f32; 4]` at the boundary via `from_array`/`to_array`.

use archmage::rite;

#[cfg(target_arch = "aarch64")]
pub use archmage::Arm64;

#[cfg(target_arch = "wasm32")]
pub use archmage::Wasm128Token;

use magetypes::simd::f32x4 as mt_f32x4;

// =============================================================================
// AArch64 NEON
// =============================================================================

macro_rules! neon_rite {
    ($name:ident, $inner:path) => {
        #[cfg(target_arch = "aarch64")]
        #[rite]
        pub fn $name(token: Arm64, v: [f32; 4]) -> [f32; 4] {
            $inner(token, mt_f32x4::from_array(token, v)).to_array()
        }
    };
}

neon_rite!(srgb_to_linear_neon, super::srgb::srgb_to_linear_x4);
neon_rite!(linear_to_srgb_neon, super::srgb::linear_to_srgb_x4);
neon_rite!(bt709_to_linear_neon, super::bt709::bt709_to_linear_x4);
neon_rite!(linear_to_bt709_neon, super::bt709::linear_to_bt709_x4);
neon_rite!(pq_to_linear_neon, super::pq::pq_to_linear_x4);
neon_rite!(linear_to_pq_neon, super::pq::linear_to_pq_x4);
neon_rite!(hlg_to_linear_neon, super::hlg::hlg_to_linear_x4);
neon_rite!(linear_to_hlg_neon, super::hlg::linear_to_hlg_x4);

// Slice functions

macro_rules! neon_slice_rite {
    ($name:ident, $rite:ident, $scalar:path) => {
        #[cfg(target_arch = "aarch64")]
        #[rite]
        pub fn $name(token: Arm64, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = $rite(token, *chunk);
            }
            for v in remainder {
                *v = $scalar(*v);
            }
        }
    };
}

neon_slice_rite!(
    srgb_to_linear_slice_neon,
    srgb_to_linear_neon,
    super::srgb_to_linear
);
neon_slice_rite!(
    linear_to_srgb_slice_neon,
    linear_to_srgb_neon,
    super::linear_to_srgb
);
neon_slice_rite!(
    bt709_to_linear_slice_neon,
    bt709_to_linear_neon,
    super::bt709_to_linear
);
neon_slice_rite!(
    linear_to_bt709_slice_neon,
    linear_to_bt709_neon,
    super::linear_to_bt709
);
neon_slice_rite!(
    pq_to_linear_slice_neon,
    pq_to_linear_neon,
    super::pq_to_linear
);
neon_slice_rite!(
    linear_to_pq_slice_neon,
    linear_to_pq_neon,
    super::linear_to_pq
);
neon_slice_rite!(
    hlg_to_linear_slice_neon,
    hlg_to_linear_neon,
    super::hlg_to_linear
);
neon_slice_rite!(
    linear_to_hlg_slice_neon,
    linear_to_hlg_neon,
    super::linear_to_hlg
);

// =============================================================================
// WebAssembly SIMD128
// =============================================================================

macro_rules! wasm_rite {
    ($name:ident, $inner:path) => {
        #[cfg(target_arch = "wasm32")]
        #[rite]
        pub fn $name(token: Wasm128Token, v: [f32; 4]) -> [f32; 4] {
            $inner(token, mt_f32x4::from_array(token, v)).to_array()
        }
    };
}

wasm_rite!(srgb_to_linear_wasm128, super::srgb::srgb_to_linear_x4);
wasm_rite!(linear_to_srgb_wasm128, super::srgb::linear_to_srgb_x4);
wasm_rite!(bt709_to_linear_wasm128, super::bt709::bt709_to_linear_x4);
wasm_rite!(linear_to_bt709_wasm128, super::bt709::linear_to_bt709_x4);
wasm_rite!(pq_to_linear_wasm128, super::pq::pq_to_linear_x4);
wasm_rite!(linear_to_pq_wasm128, super::pq::linear_to_pq_x4);
wasm_rite!(hlg_to_linear_wasm128, super::hlg::hlg_to_linear_x4);
wasm_rite!(linear_to_hlg_wasm128, super::hlg::linear_to_hlg_x4);

// Slice functions

macro_rules! wasm_slice_rite {
    ($name:ident, $rite:ident, $scalar:path) => {
        #[cfg(target_arch = "wasm32")]
        #[rite]
        pub fn $name(token: Wasm128Token, values: &mut [f32]) {
            let (chunks, remainder) = values.as_chunks_mut::<4>();
            for chunk in chunks {
                *chunk = $rite(token, *chunk);
            }
            for v in remainder {
                *v = $scalar(*v);
            }
        }
    };
}

wasm_slice_rite!(
    srgb_to_linear_slice_wasm128,
    srgb_to_linear_wasm128,
    super::srgb_to_linear
);
wasm_slice_rite!(
    linear_to_srgb_slice_wasm128,
    linear_to_srgb_wasm128,
    super::linear_to_srgb
);
wasm_slice_rite!(
    bt709_to_linear_slice_wasm128,
    bt709_to_linear_wasm128,
    super::bt709_to_linear
);
wasm_slice_rite!(
    linear_to_bt709_slice_wasm128,
    linear_to_bt709_wasm128,
    super::linear_to_bt709
);
wasm_slice_rite!(
    pq_to_linear_slice_wasm128,
    pq_to_linear_wasm128,
    super::pq_to_linear
);
wasm_slice_rite!(
    linear_to_pq_slice_wasm128,
    linear_to_pq_wasm128,
    super::linear_to_pq
);
wasm_slice_rite!(
    hlg_to_linear_slice_wasm128,
    hlg_to_linear_wasm128,
    super::hlg_to_linear
);
wasm_slice_rite!(
    linear_to_hlg_slice_wasm128,
    linear_to_hlg_wasm128,
    super::linear_to_hlg
);
