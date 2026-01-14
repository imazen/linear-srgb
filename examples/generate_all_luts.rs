//! Generate const LUT arrays for embedding in binary.

fn srgb_to_linear_f64(s: f64) -> f64 {
    const THRESHOLD: f64 = 0.039_293_372_176_817_44;
    const SLOPE: f64 = 1.0 / 12.92;
    const GAMMA: f64 = 2.4;
    const SCALE: f64 = 1.0 / 1.055;
    const OFFSET: f64 = 0.055;

    if s <= THRESHOLD {
        s * SLOPE
    } else {
        ((s + OFFSET) * SCALE).powf(GAMMA)
    }
}

fn linear_to_srgb_f64(l: f64) -> f64 {
    const THRESHOLD: f64 = 0.003_041_282_560_127_521;
    const SLOPE: f64 = 12.92;
    const GAMMA: f64 = 1.0 / 2.4;
    const SCALE: f64 = 1.055;
    const OFFSET: f64 = 0.055;

    if l <= THRESHOLD {
        l * SLOPE
    } else {
        l.powf(GAMMA) * SCALE - OFFSET
    }
}

fn main() {
    // LinearTable8: 256 entries (sRGB u8 -> linear f32)
    println!("// LinearTable8: sRGB u8 → linear f32 (256 entries, 1KB)");
    println!("pub(crate) const LINEAR_TABLE_8: [f32; 256] = [");
    for i in 0..256 {
        let srgb = i as f64 / 255.0;
        let linear = srgb_to_linear_f64(srgb) as f32;
        if i % 4 == 0 {
            print!("    ");
        }
        print!("{:?}_f32, ", linear);
        if i % 4 == 3 {
            println!();
        }
    }
    println!("];");
    println!();

    // EncodeTable12: 4096 entries (linear f32 -> sRGB f32)
    println!("// EncodeTable12: linear → sRGB f32 (4096 entries, 16KB)");
    println!("pub(crate) const ENCODE_TABLE_12: [f32; 4096] = [");
    for i in 0..4096 {
        let linear = i as f64 / 4095.0;
        let srgb = linear_to_srgb_f64(linear) as f32;
        if i % 4 == 0 {
            print!("    ");
        }
        print!("{:?}_f32, ", srgb);
        if i % 4 == 3 {
            println!();
        }
    }
    println!("];");
}
