//! Audit u16 roundtrip accuracy: srgb_u16_to_linear → linear_to_srgb_u16.
//! cargo run --example u16_roundtrip_audit --release --all-features

fn main() {
    use linear_srgb::default::{linear_to_srgb_u16, srgb_u16_to_linear};

    let mut max_diff = 0u32;
    let mut worst = 0u16;
    let mut exact = 0u32;
    let mut diff_hist = [0u32; 16];

    for i in 0..=65535u16 {
        let linear = srgb_u16_to_linear(i);
        let back = linear_to_srgb_u16(linear);
        let diff = (i as i32 - back as i32).unsigned_abs();
        if diff > max_diff {
            max_diff = diff;
            worst = i;
        }
        if diff == 0 {
            exact += 1;
        }
        if (diff as usize) < diff_hist.len() {
            diff_hist[diff as usize] += 1;
        }
    }

    eprintln!("u16 roundtrip (SIMD LUT → SIMD LUT):");
    eprintln!("  max diff: {max_diff} at sRGB value {worst}");
    eprintln!(
        "  exact:    {exact}/65536 ({:.1}%)",
        exact as f64 / 65536.0 * 100.0
    );
    for (d, &count) in diff_hist.iter().enumerate() {
        if count > 0 {
            eprintln!("  diff={d}: {count}");
        }
    }

    // Also check encode path consistency: does linear_to_srgb_u16 match
    // the scalar rational polynomial quantized to u16?
    let mut scalar_mismatch = 0u32;
    let mut scalar_max_diff = 0u32;
    for i in 0..=65535u32 {
        let linear = i as f32 / 65535.0;
        let lut_val = linear_to_srgb_u16(linear);
        let poly_val = (linear_srgb::default::linear_to_srgb(linear) * 65535.0 + 0.5)
            .clamp(0.0, 65535.0) as u16;
        let diff = (lut_val as i32 - poly_val as i32).unsigned_abs();
        if diff > 0 {
            scalar_mismatch += 1;
        }
        scalar_max_diff = scalar_max_diff.max(diff);
    }
    eprintln!("\nencode LUT vs scalar polynomial:");
    eprintln!("  mismatches: {scalar_mismatch}/65536");
    eprintln!("  max diff:   {scalar_max_diff}");
}
