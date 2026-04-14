//! Pre-computed lookup tables embedded in the binary.
//!
//! Tables are stored as raw little-endian bytes in `src/data/` and loaded via
//! `include_bytes!`. A `#[repr(C, align(4))]` wrapper ensures the byte arrays
//! are f32-aligned so `bytemuck::cast_slice` never panics.
//!
//! Generated using C0-continuous sRGB constants (moxcms).

#[cfg(not(target_endian = "little"))]
compile_error!(
    "const_luts binary blobs are little-endian; big-endian targets need a byte-swap path or regenerated tables"
);

/// Wrapper that forces 4-byte alignment on an included byte array.
/// `include_bytes!` has no alignment guarantee; this ensures the data
/// can be safely cast to `&[f32]` via `bytemuck::cast_slice`.
#[repr(C, align(4))]
struct Aligned<const N: usize>([u8; N]);

// LinearTable8: sRGB u8 → linear f32 (256 entries, 1 KB)
static LINEAR_TABLE_8_ALIGNED: &Aligned<{ 256 * 4 }> =
    &Aligned(*include_bytes!("data/linear_table_8.bin"));

// EncodeTable12: linear → sRGB f32 (4096 entries, 16 KB)
static ENCODE_TABLE_12_ALIGNED: &Aligned<{ 4096 * 4 }> =
    &Aligned(*include_bytes!("data/encode_table_12.bin"));

// LinearToSrgbU8: linear (12-bit index) → sRGB u8 (4096 entries, 4 KB)
static LINEAR_TO_SRGB_U8_BYTES: &[u8; 4096] = include_bytes!("data/linear_to_srgb_u8.bin");

/// sRGB u8 → linear f32 lookup table (256 entries).
#[inline]
pub(crate) fn linear_table_8() -> &'static [f32; 256] {
    let slice: &[f32] = bytemuck::cast_slice(&LINEAR_TABLE_8_ALIGNED.0);
    slice.try_into().unwrap()
}

/// Linear → sRGB f32 lookup table (4096 entries, 12-bit resolution).
#[inline]
pub(crate) fn encode_table_12() -> &'static [f32; 4096] {
    let slice: &[f32] = bytemuck::cast_slice(&ENCODE_TABLE_12_ALIGNED.0);
    slice.try_into().unwrap()
}

/// Linear (12-bit index) → sRGB u8 lookup table (4096 entries).
#[inline]
pub(crate) fn linear_to_srgb_u8() -> &'static [u8; 4096] {
    LINEAR_TO_SRGB_U8_BYTES
}
