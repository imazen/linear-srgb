//! Lookup table (LUT) based sRGB conversions.
//!
//! Pre-computed tables trade memory for speed. Table sizes:
//! - 8-bit: 256 entries (1KB for f32)
//! - 10-bit: 1024 entries (4KB for f32)
//! - 12-bit: 4096 entries (16KB for f32)
//! - 16-bit: 65536 entries (256KB for f32)

use crate::mlaf::{mlaf, neg_mlaf};
use crate::transfer::{linear_to_srgb_f64, srgb_to_linear_f64};

/// Pre-computed linearization table for sRGB to linear conversion.
pub struct LinearizationTable<const N: usize> {
    table: Box<[f32; N]>,
}

impl<const N: usize> LinearizationTable<N> {
    /// Create a linearization table for the given bit depth.
    ///
    /// N must equal 2^BIT_DEPTH (e.g., N=256 for 8-bit, N=65536 for 16-bit).
    pub fn new() -> Self {
        let mut table = vec![0.0f32; N].into_boxed_slice();
        let max_value = (N - 1) as f64;

        for (i, entry) in table.iter_mut().enumerate() {
            let srgb = i as f64 / max_value;
            *entry = srgb_to_linear_f64(srgb) as f32;
        }

        // Convert boxed slice to boxed array
        let table = unsafe {
            let ptr = Box::into_raw(table) as *mut [f32; N];
            Box::from_raw(ptr)
        };

        Self { table }
    }

    /// Direct lookup (no interpolation) - for exact bit-depth matches.
    #[inline]
    pub fn lookup(&self, index: usize) -> f32 {
        self.table[index.min(N - 1)]
    }

    /// Get the raw table for custom interpolation.
    pub fn as_slice(&self) -> &[f32] {
        &self.table[..]
    }
}

impl<const N: usize> Default for LinearizationTable<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-computed encoding table for linear to sRGB conversion.
pub struct EncodingTable<const N: usize> {
    table: Box<[f32; N]>,
}

impl<const N: usize> EncodingTable<N> {
    /// Create an encoding table for the given resolution.
    ///
    /// Higher N means finer granularity for interpolation.
    pub fn new() -> Self {
        let mut table = vec![0.0f32; N].into_boxed_slice();
        let max_value = (N - 1) as f64;

        for (i, entry) in table.iter_mut().enumerate() {
            let linear = i as f64 / max_value;
            *entry = linear_to_srgb_f64(linear) as f32;
        }

        let table = unsafe {
            let ptr = Box::into_raw(table) as *mut [f32; N];
            Box::from_raw(ptr)
        };

        Self { table }
    }

    /// Direct lookup (no interpolation).
    #[inline]
    pub fn lookup(&self, index: usize) -> f32 {
        self.table[index.min(N - 1)]
    }

    /// Get the raw table for custom interpolation.
    pub fn as_slice(&self) -> &[f32] {
        &self.table[..]
    }
}

impl<const N: usize> Default for EncodingTable<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Linear interpolation in a float LUT.
///
/// `x` should be in [0, 1]. Values outside are clamped.
/// Uses FMA for the interpolation calculation.
#[inline]
pub fn lut_interp_linear_float(x: f32, table: &[f32]) -> f32 {
    let x = x.clamp(0.0, 1.0);
    let value = x * (table.len() - 1) as f32;

    let upper = value.ceil() as usize;
    let lower = value.floor() as usize;

    // Safety: upper and lower are bounded by table.len() - 1 due to clamping
    let tu = table[upper.min(table.len() - 1)];
    let tl = table[lower];

    let diff = upper as f32 - value;

    // result = tl * diff + tu * (1 - diff)
    // Using FMA: neg_mlaf(tu, tu, diff) = tu - tu*diff = tu*(1-diff)
    // Then: mlaf(tu*(1-diff), tl, diff) = tu*(1-diff) + tl*diff
    mlaf(neg_mlaf(tu, tu, diff), tl, diff)
}

/// Linear interpolation in a u16 LUT using fixed-point arithmetic.
///
/// Avoids floating-point entirely for integer-only pipelines.
#[inline]
pub fn lut_interp_linear_u16(input_value: u16, table: &[u16]) -> u16 {
    let table_len = table.len() as u32;
    let mut value: u32 = input_value as u32 * (table_len - 1);

    let upper = value.div_ceil(65535) as usize;
    let lower = (value / 65535) as usize;
    let interp: u32 = value % 65535;

    // Safety: upper and lower are bounded
    let upper = upper.min(table.len() - 1);
    let lower = lower.min(table.len() - 1);

    value = (table[upper] as u32 * interp + table[lower] as u32 * (65535 - interp)) / 65535;
    value as u16
}

/// Pre-computed 8-bit linearization table (256 entries).
pub type LinearTable8 = LinearizationTable<256>;

/// Pre-computed 10-bit linearization table (1024 entries).
pub type LinearTable10 = LinearizationTable<1024>;

/// Pre-computed 12-bit linearization table (4096 entries).
pub type LinearTable12 = LinearizationTable<4096>;

/// Pre-computed 16-bit linearization table (65536 entries).
pub type LinearTable16 = LinearizationTable<65536>;

/// Pre-computed 8-bit encoding table (256 entries).
pub type EncodeTable8 = EncodingTable<256>;

/// Pre-computed 12-bit encoding table (4096 entries).
pub type EncodeTable12 = EncodingTable<4096>;

/// Pre-computed 16-bit encoding table (65536 entries).
pub type EncodeTable16 = EncodingTable<65536>;

/// Converter using pre-computed LUTs for fast batch conversion.
pub struct SrgbConverter {
    linearize_table: LinearTable8,
    encode_table: EncodeTable12,
}

impl SrgbConverter {
    /// Create a new converter with default table sizes.
    ///
    /// Uses 8-bit (256 entry) table for linearization (direct lookup)
    /// and 12-bit (4096 entry) table for encoding (with interpolation).
    pub fn new() -> Self {
        Self {
            linearize_table: LinearTable8::new(),
            encode_table: EncodeTable12::new(),
        }
    }

    /// Convert 8-bit sRGB to linear using direct table lookup.
    #[inline]
    pub fn srgb_u8_to_linear(&self, value: u8) -> f32 {
        self.linearize_table.lookup(value as usize)
    }

    /// Convert linear to sRGB using table interpolation.
    #[inline]
    pub fn linear_to_srgb(&self, linear: f32) -> f32 {
        lut_interp_linear_float(linear, self.encode_table.as_slice())
    }

    /// Convert linear to 8-bit sRGB.
    #[inline]
    pub fn linear_to_srgb_u8(&self, linear: f32) -> u8 {
        (self.linear_to_srgb(linear) * 255.0 + 0.5) as u8
    }

    /// Batch convert sRGB u8 values to linear f32.
    pub fn batch_srgb_to_linear(&self, input: &[u8], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = self.srgb_u8_to_linear(*i);
        }
    }

    /// Batch convert linear f32 values to sRGB u8.
    pub fn batch_linear_to_srgb(&self, input: &[f32], output: &mut [u8]) {
        assert_eq!(input.len(), output.len());
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = self.linear_to_srgb_u8(*i);
        }
    }
}

impl Default for SrgbConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linearization_table_8bit() {
        let table = LinearTable8::new();

        // Check boundaries
        assert_eq!(table.lookup(0), 0.0);
        assert!((table.lookup(255) - 1.0).abs() < 1e-6);

        // Check middle value
        let mid = table.lookup(128);
        assert!(mid > 0.0 && mid < 1.0);
    }

    #[test]
    fn test_encoding_table() {
        let table = EncodeTable12::new();

        assert_eq!(table.lookup(0), 0.0);
        assert!((table.lookup(4095) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lut_interpolation() {
        let table = EncodeTable12::new();

        // Test interpolation at exact points
        let result = lut_interp_linear_float(0.0, table.as_slice());
        assert!((result - 0.0).abs() < 1e-6);

        let result = lut_interp_linear_float(1.0, table.as_slice());
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_converter_roundtrip() {
        let conv = SrgbConverter::new();

        for i in 0..=255u8 {
            let linear = conv.srgb_u8_to_linear(i);
            let back = conv.linear_to_srgb_u8(linear);
            assert!(
                (i as i32 - back as i32).abs() <= 1,
                "Roundtrip failed for {}: {} -> {} -> {}",
                i,
                i,
                linear,
                back
            );
        }
    }

    #[test]
    fn test_lut_vs_direct() {
        use crate::transfer::srgb_to_linear;

        let table = LinearTable8::new();

        // Compare LUT to direct computation
        for i in 0..=255u8 {
            let lut_result = table.lookup(i as usize);
            let direct_result = srgb_to_linear(i as f32 / 255.0);
            assert!(
                (lut_result - direct_result).abs() < 1e-5,
                "Mismatch at {}: LUT={}, direct={}",
                i,
                lut_result,
                direct_result
            );
        }
    }

    #[test]
    fn test_u16_interpolation() {
        // Create a simple linear ramp table
        let table: Vec<u16> = (0..=255).map(|i| (i * 257) as u16).collect();

        // Input 0 should give 0
        assert_eq!(lut_interp_linear_u16(0, &table), 0);

        // Input max should give max
        assert_eq!(lut_interp_linear_u16(65535, &table), 65535);
    }

    #[test]
    fn test_batch_conversion() {
        let conv = SrgbConverter::new();

        let input: Vec<u8> = (0..=255).collect();
        let mut linear = vec![0.0f32; 256];
        let mut back = vec![0u8; 256];

        conv.batch_srgb_to_linear(&input, &mut linear);
        conv.batch_linear_to_srgb(&linear, &mut back);

        for i in 0..256 {
            assert!(
                (input[i] as i32 - back[i] as i32).abs() <= 1,
                "Batch roundtrip failed at {}",
                i
            );
        }
    }
}
