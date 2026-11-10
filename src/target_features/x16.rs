entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
    fn srgb_to_linear_v4(values: [f32; 16]) -> [f32; 16]);
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
    fn srgb_to_linear_slice_v4(values: &mut [f32]));
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
    fn linear_to_srgb_v4(values: [f32; 16]) -> [f32; 16]);
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
    fn linear_to_srgb_slice_v4(values: &mut [f32]));
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
    fn gamma_to_linear_v4(values: [f32; 16], gamma: f32) -> [f32; 16]);
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
    fn gamma_to_linear_slice_v4(values: &mut [f32], gamma: f32));
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
    fn linear_to_gamma_v4(values: [f32; 16], gamma: f32) -> [f32; 16]);
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
    fn linear_to_gamma_slice_v4(values: &mut [f32], gamma: f32));
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
    fn srgb_u8_to_linear_v4(values: [u8; 16]) -> [f32; 16]);
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
    fn linear_to_srgb_u8_v4(values: [f32; 16]) -> [u8; 16]);
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
    fn srgb_u8_to_linear_slice_v4(input: &[u8], output: &mut [f32]));
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
    fn linear_to_srgb_u8_slice_v4(input: &[f32], output: &mut [u8]));
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn tf_srgb_to_linear_v4(values: [f32; 16]) -> [f32; 16]);
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn tf_srgb_to_linear_slice_v4(values: &mut [f32]));
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn tf_linear_to_srgb_v4(values: [f32; 16]) -> [f32; 16]);
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn tf_linear_to_srgb_slice_v4(values: &mut [f32]));
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn bt709_to_linear_v4(values: [f32; 16]) -> [f32; 16]);
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn bt709_to_linear_slice_v4(values: &mut [f32]));
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn linear_to_bt709_v4(values: [f32; 16]) -> [f32; 16]);
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn linear_to_bt709_slice_v4(values: &mut [f32]));
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn pq_to_linear_v4(values: [f32; 16]) -> [f32; 16]);
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn pq_to_linear_slice_v4(values: &mut [f32]));
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn linear_to_pq_v4(values: [f32; 16]) -> [f32; 16]);
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn linear_to_pq_slice_v4(values: &mut [f32]));
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn hlg_to_linear_v4(values: [f32; 16]) -> [f32; 16]);
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn hlg_to_linear_slice_v4(values: &mut [f32]));
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn linear_to_hlg_v4(values: [f32; 16]) -> [f32; 16]);
entry!(x16, v4, archmage::X64V4Token; #[cfg(feature = "avx512")]
#[cfg(feature = "transfer")]
    fn linear_to_hlg_slice_v4(values: &mut [f32]));

// Portable tier completion
include!("portable_x16.rs");
