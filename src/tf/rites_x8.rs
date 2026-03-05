//! 8×f32 `#[rite]` transfer function wrappers (AVX2+FMA on x86-64).
//!
//! Each function converts `[f32; 8]` at the boundary via `from_array`/`to_array`.
//! Internally uses magetypes `f32x8<Desktop64>` for full SIMD.

#[cfg(target_arch = "x86_64")]
use archmage::rite;

#[cfg(target_arch = "x86_64")]
pub use archmage::Desktop64;

#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8 as mt_f32x8;

// --- sRGB ---

/// Convert 8 sRGB values to linear (rational polynomial, no powf).
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn srgb_to_linear_v3(token: Desktop64, v: [f32; 8]) -> [f32; 8] {
    super::srgb::srgb_to_linear_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

/// Convert 8 linear values to sRGB (rational polynomial, no powf).
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_srgb_v3(token: Desktop64, v: [f32; 8]) -> [f32; 8] {
    super::srgb::linear_to_srgb_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

// --- BT.709 ---

/// Convert 8 BT.709 encoded values to linear.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn bt709_to_linear_v3(token: Desktop64, v: [f32; 8]) -> [f32; 8] {
    super::bt709::bt709_to_linear_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

/// Convert 8 linear values to BT.709 encoded.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_bt709_v3(token: Desktop64, v: [f32; 8]) -> [f32; 8] {
    super::bt709::linear_to_bt709_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

// --- PQ ---

/// Convert 8 PQ signal values to linear.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn pq_to_linear_v3(token: Desktop64, v: [f32; 8]) -> [f32; 8] {
    super::pq::pq_to_linear_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

/// Convert 8 linear values to PQ signal.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_pq_v3(token: Desktop64, v: [f32; 8]) -> [f32; 8] {
    super::pq::linear_to_pq_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

// --- HLG ---

/// Convert 8 HLG signal values to linear.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn hlg_to_linear_v3(token: Desktop64, v: [f32; 8]) -> [f32; 8] {
    super::hlg::hlg_to_linear_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

/// Convert 8 linear values to HLG signal.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_hlg_v3(token: Desktop64, v: [f32; 8]) -> [f32; 8] {
    super::hlg::linear_to_hlg_x8(token, mt_f32x8::from_array(token, v)).to_array()
}

// --- Slice functions ---

/// Convert sRGB f32 values to linear in-place, 8-wide.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn srgb_to_linear_slice_v3(token: Desktop64, values: &mut [f32]) {
    tf_slice_x8(
        values,
        |v| srgb_to_linear_v3(token, v),
        super::srgb_to_linear,
    );
}

/// Convert linear f32 values to sRGB in-place, 8-wide.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_srgb_slice_v3(token: Desktop64, values: &mut [f32]) {
    tf_slice_x8(
        values,
        |v| linear_to_srgb_v3(token, v),
        super::linear_to_srgb,
    );
}

/// Convert BT.709 f32 values to linear in-place, 8-wide.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn bt709_to_linear_slice_v3(token: Desktop64, values: &mut [f32]) {
    tf_slice_x8(
        values,
        |v| bt709_to_linear_v3(token, v),
        super::bt709_to_linear,
    );
}

/// Convert linear f32 values to BT.709 in-place, 8-wide.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_bt709_slice_v3(token: Desktop64, values: &mut [f32]) {
    tf_slice_x8(
        values,
        |v| linear_to_bt709_v3(token, v),
        super::linear_to_bt709,
    );
}

/// Convert PQ f32 values to linear in-place, 8-wide.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn pq_to_linear_slice_v3(token: Desktop64, values: &mut [f32]) {
    tf_slice_x8(values, |v| pq_to_linear_v3(token, v), super::pq_to_linear);
}

/// Convert linear f32 values to PQ in-place, 8-wide.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_pq_slice_v3(token: Desktop64, values: &mut [f32]) {
    tf_slice_x8(values, |v| linear_to_pq_v3(token, v), super::linear_to_pq);
}

/// Convert HLG f32 values to linear in-place, 8-wide.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn hlg_to_linear_slice_v3(token: Desktop64, values: &mut [f32]) {
    tf_slice_x8(values, |v| hlg_to_linear_v3(token, v), super::hlg_to_linear);
}

/// Convert linear f32 values to HLG in-place, 8-wide.
#[cfg(target_arch = "x86_64")]
#[rite]
pub fn linear_to_hlg_slice_v3(token: Desktop64, values: &mut [f32]) {
    tf_slice_x8(values, |v| linear_to_hlg_v3(token, v), super::linear_to_hlg);
}

#[inline(always)]
fn tf_slice_x8(
    values: &mut [f32],
    tf_x8: impl Fn([f32; 8]) -> [f32; 8],
    tf_scalar: fn(f32) -> f32,
) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();
    for chunk in chunks {
        *chunk = tf_x8(*chunk);
    }
    for v in remainder {
        *v = tf_scalar(*v);
    }
}
