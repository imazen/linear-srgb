//! Basic usage of linear-srgb conversions.

use linear_srgb::{linear_to_srgb, srgb_to_linear, SrgbConverter};

fn main() {
    // Direct conversion
    println!("=== Direct Conversion ===");
    let srgb = 0.5_f32;
    let linear = srgb_to_linear(srgb);
    let back = linear_to_srgb(linear);
    println!("sRGB {:.3} -> linear {:.6} -> sRGB {:.3}", srgb, linear, back);

    // 8-bit conversions
    println!("\n=== 8-bit Values ===");
    for val in [0u8, 64, 128, 192, 255] {
        let linear = linear_srgb::srgb_u8_to_linear(val);
        let back = linear_srgb::linear_to_srgb_u8(linear);
        println!(
            "sRGB u8 {:3} -> linear {:.6} -> sRGB u8 {:3}",
            val, linear, back
        );
    }

    // LUT-based conversion (faster for batch processing)
    println!("\n=== LUT-based Conversion ===");
    let conv = SrgbConverter::new();

    let srgb_input: Vec<u8> = (0..=255).collect();
    let mut linear_output = vec![0.0f32; 256];

    conv.batch_srgb_to_linear(&srgb_input, &mut linear_output);

    println!("Converted 256 values using LUT:");
    for (i, &lin) in [0, 64, 128, 192, 255].iter().zip(
        [
            linear_output[0],
            linear_output[64],
            linear_output[128],
            linear_output[192],
            linear_output[255],
        ]
        .iter(),
    ) {
        println!("  sRGB {:3} -> linear {:.6}", i, lin);
    }

    // Verify roundtrip accuracy
    println!("\n=== Roundtrip Accuracy ===");
    let mut max_error = 0.0f32;
    for i in 0..=255u8 {
        let linear = conv.srgb_u8_to_linear(i);
        let back = conv.linear_to_srgb_u8(linear);
        let error = (i as i32 - back as i32).abs();
        if error as f32 > max_error {
            max_error = error as f32;
        }
    }
    println!("Maximum roundtrip error for u8: {} levels", max_error);
}
