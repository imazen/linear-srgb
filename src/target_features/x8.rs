#[cfg(target_arch = "x86_64")]
use crate::tokens::x8 as legacy;

forward!(x8, v3, archmage::X64V3Token;
    fn srgb_to_linear_v3(values: [f32; 8]) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
    fn srgb_to_linear_slice_v3(values: &mut [f32]));
forward!(x8, v3, archmage::X64V3Token;
    fn linear_to_srgb_v3(values: [f32; 8]) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
    fn linear_to_srgb_slice_v3(values: &mut [f32]));
forward!(x8, v3, archmage::X64V3Token;
    fn srgb_to_linear_extended_v3(values: [f32; 8]) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
    fn srgb_to_linear_extended_slice_v3(values: &mut [f32]));
forward!(x8, v3, archmage::X64V3Token;
    fn linear_to_srgb_extended_v3(values: [f32; 8]) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
    fn linear_to_srgb_extended_slice_v3(values: &mut [f32]));
forward!(x8, v3, archmage::X64V3Token;
    fn gamma_to_linear_v3(values: [f32; 8], gamma: f32) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
    fn gamma_to_linear_slice_v3(values: &mut [f32], gamma: f32));
forward!(x8, v3, archmage::X64V3Token;
    fn linear_to_gamma_v3(values: [f32; 8], gamma: f32) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
    fn linear_to_gamma_slice_v3(values: &mut [f32], gamma: f32));
forward!(x8, v3, archmage::X64V3Token;
    fn srgb_u8_to_linear_v3(values: [u8; 8]) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
    fn linear_to_srgb_u8_v3(values: [f32; 8]) -> [u8; 8]);
forward!(x8, v3, archmage::X64V3Token;
    fn srgb_u8_to_linear_slice_v3(input: &[u8], output: &mut [f32]));
forward!(x8, v3, archmage::X64V3Token;
    fn linear_to_srgb_u8_slice_v3(input: &[f32], output: &mut [u8]));
forward!(x8, v3, archmage::X64V3Token;
    fn srgb_u16_to_linear_v3(values: [u16; 8]) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
    fn linear_to_srgb_u16_v3(values: [f32; 8]) -> [u16; 8]);
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn tf_srgb_to_linear_v3(values: [f32; 8]) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn tf_srgb_to_linear_slice_v3(values: &mut [f32]));
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn tf_linear_to_srgb_v3(values: [f32; 8]) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn tf_linear_to_srgb_slice_v3(values: &mut [f32]));
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn bt709_to_linear_v3(values: [f32; 8]) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn bt709_to_linear_slice_v3(values: &mut [f32]));
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn linear_to_bt709_v3(values: [f32; 8]) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn linear_to_bt709_slice_v3(values: &mut [f32]));
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn pq_to_linear_v3(values: [f32; 8]) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn pq_to_linear_slice_v3(values: &mut [f32]));
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn linear_to_pq_v3(values: [f32; 8]) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn linear_to_pq_slice_v3(values: &mut [f32]));
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn hlg_to_linear_v3(values: [f32; 8]) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn hlg_to_linear_slice_v3(values: &mut [f32]));
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn linear_to_hlg_v3(values: [f32; 8]) -> [f32; 8]);
forward!(x8, v3, archmage::X64V3Token;
#[cfg(feature = "transfer")]
    fn linear_to_hlg_slice_v3(values: &mut [f32]));
forward!(x8, scalar, archmage::ScalarToken; #[cfg(target_arch = "x86_64")] fn srgb_u16_to_linear_scalar(values: [u16; 8]) -> [f32; 8]);
forward!(x8, scalar, archmage::ScalarToken; #[cfg(target_arch = "x86_64")] fn linear_to_srgb_u16_scalar(values: [f32; 8]) -> [u16; 8]);

// Portable tier completion
include!("portable_x8.rs");
