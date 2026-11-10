entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
    fn srgb_to_linear_v3(values: [f32; 4]) -> [f32; 4]);
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
    fn srgb_to_linear_slice_v3(values: &mut [f32]));
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
    fn linear_to_srgb_v3(values: [f32; 4]) -> [f32; 4]);
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
    fn linear_to_srgb_slice_v3(values: &mut [f32]));
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
    fn srgb_to_linear_extended_v3(values: [f32; 4]) -> [f32; 4]);
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
    fn linear_to_srgb_extended_v3(values: [f32; 4]) -> [f32; 4]);
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
    fn gamma_to_linear_v3(values: [f32; 4], gamma: f32) -> [f32; 4]);
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
    fn gamma_to_linear_slice_v3(values: &mut [f32], gamma: f32));
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
    fn linear_to_gamma_v3(values: [f32; 4], gamma: f32) -> [f32; 4]);
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
    fn linear_to_gamma_slice_v3(values: &mut [f32], gamma: f32));
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn tf_srgb_to_linear_v3(values: [f32; 4]) -> [f32; 4]);
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn tf_srgb_to_linear_slice_v3(values: &mut [f32]));
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn tf_linear_to_srgb_v3(values: [f32; 4]) -> [f32; 4]);
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn tf_linear_to_srgb_slice_v3(values: &mut [f32]));
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn bt709_to_linear_v3(values: [f32; 4]) -> [f32; 4]);
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn bt709_to_linear_slice_v3(values: &mut [f32]));
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn linear_to_bt709_v3(values: [f32; 4]) -> [f32; 4]);
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn linear_to_bt709_slice_v3(values: &mut [f32]));
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn pq_to_linear_v3(values: [f32; 4]) -> [f32; 4]);
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn pq_to_linear_slice_v3(values: &mut [f32]));
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn linear_to_pq_v3(values: [f32; 4]) -> [f32; 4]);
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn linear_to_pq_slice_v3(values: &mut [f32]));
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn hlg_to_linear_v3(values: [f32; 4]) -> [f32; 4]);
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn hlg_to_linear_slice_v3(values: &mut [f32]));
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn linear_to_hlg_v3(values: [f32; 4]) -> [f32; 4]);
entry!(x4, v3, archmage::X64V3Token; #[cfg(target_arch = "x86_64")]
#[cfg(feature = "transfer")]
    fn linear_to_hlg_slice_v3(values: &mut [f32]));
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
    fn srgb_to_linear_neon(values: [f32; 4]) -> [f32; 4]);
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
    fn srgb_to_linear_slice_neon(values: &mut [f32]));
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
    fn linear_to_srgb_neon(values: [f32; 4]) -> [f32; 4]);
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
    fn linear_to_srgb_slice_neon(values: &mut [f32]));
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
    fn srgb_to_linear_extended_neon(values: [f32; 4]) -> [f32; 4]);
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
    fn linear_to_srgb_extended_neon(values: [f32; 4]) -> [f32; 4]);
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
    fn gamma_to_linear_neon(values: [f32; 4], gamma: f32) -> [f32; 4]);
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
    fn gamma_to_linear_slice_neon(values: &mut [f32], gamma: f32));
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
    fn linear_to_gamma_neon(values: [f32; 4], gamma: f32) -> [f32; 4]);
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
    fn linear_to_gamma_slice_neon(values: &mut [f32], gamma: f32));
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn tf_srgb_to_linear_neon(values: [f32; 4]) -> [f32; 4]);
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn tf_srgb_to_linear_slice_neon(values: &mut [f32]));
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn tf_linear_to_srgb_neon(values: [f32; 4]) -> [f32; 4]);
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn tf_linear_to_srgb_slice_neon(values: &mut [f32]));
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn bt709_to_linear_neon(values: [f32; 4]) -> [f32; 4]);
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn bt709_to_linear_slice_neon(values: &mut [f32]));
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn linear_to_bt709_neon(values: [f32; 4]) -> [f32; 4]);
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn linear_to_bt709_slice_neon(values: &mut [f32]));
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn pq_to_linear_neon(values: [f32; 4]) -> [f32; 4]);
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn pq_to_linear_slice_neon(values: &mut [f32]));
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn linear_to_pq_neon(values: [f32; 4]) -> [f32; 4]);
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn linear_to_pq_slice_neon(values: &mut [f32]));
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn hlg_to_linear_neon(values: [f32; 4]) -> [f32; 4]);
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn hlg_to_linear_slice_neon(values: &mut [f32]));
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn linear_to_hlg_neon(values: [f32; 4]) -> [f32; 4]);
entry!(x4, neon, archmage::NeonToken; #[cfg(target_arch = "aarch64")]
#[cfg(feature = "transfer")]
    fn linear_to_hlg_slice_neon(values: &mut [f32]));
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
    fn srgb_to_linear_wasm128(values: [f32; 4]) -> [f32; 4]);
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
    fn srgb_to_linear_slice_wasm128(values: &mut [f32]));
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
    fn linear_to_srgb_wasm128(values: [f32; 4]) -> [f32; 4]);
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
    fn linear_to_srgb_slice_wasm128(values: &mut [f32]));
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
    fn srgb_to_linear_extended_wasm128(values: [f32; 4]) -> [f32; 4]);
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
    fn linear_to_srgb_extended_wasm128(values: [f32; 4]) -> [f32; 4]);
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
    fn gamma_to_linear_wasm128(values: [f32; 4], gamma: f32) -> [f32; 4]);
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
    fn gamma_to_linear_slice_wasm128(values: &mut [f32], gamma: f32));
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
    fn linear_to_gamma_wasm128(values: [f32; 4], gamma: f32) -> [f32; 4]);
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
    fn linear_to_gamma_slice_wasm128(values: &mut [f32], gamma: f32));
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn tf_srgb_to_linear_wasm128(values: [f32; 4]) -> [f32; 4]);
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn tf_srgb_to_linear_slice_wasm128(values: &mut [f32]));
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn tf_linear_to_srgb_wasm128(values: [f32; 4]) -> [f32; 4]);
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn tf_linear_to_srgb_slice_wasm128(values: &mut [f32]));
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn bt709_to_linear_wasm128(values: [f32; 4]) -> [f32; 4]);
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn bt709_to_linear_slice_wasm128(values: &mut [f32]));
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn linear_to_bt709_wasm128(values: [f32; 4]) -> [f32; 4]);
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn linear_to_bt709_slice_wasm128(values: &mut [f32]));
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn pq_to_linear_wasm128(values: [f32; 4]) -> [f32; 4]);
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn pq_to_linear_slice_wasm128(values: &mut [f32]));
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn linear_to_pq_wasm128(values: [f32; 4]) -> [f32; 4]);
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn linear_to_pq_slice_wasm128(values: &mut [f32]));
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn hlg_to_linear_wasm128(values: [f32; 4]) -> [f32; 4]);
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn hlg_to_linear_slice_wasm128(values: &mut [f32]));
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn linear_to_hlg_wasm128(values: [f32; 4]) -> [f32; 4]);
entry!(x4, wasm128, archmage::Wasm128Token; #[cfg(target_arch = "wasm32")]
#[cfg(feature = "transfer")]
    fn linear_to_hlg_slice_wasm128(values: &mut [f32]));
entry!(x4, scalar, archmage::ScalarToken; fn srgb_to_linear_extended_scalar(values: [f32; 4]) -> [f32; 4]);
entry!(x4, scalar, archmage::ScalarToken; fn linear_to_srgb_extended_scalar(values: [f32; 4]) -> [f32; 4]);

// Portable tier completion
include!("portable_x4.rs");
