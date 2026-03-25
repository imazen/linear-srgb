//! Measure u16 LUT initialization time breakdown.
//!
//! Run with: cargo run --example lut_init_time --release --all-features

fn main() {
    use std::hint::black_box;
    use std::time::Instant;

    // Warmup SIMD dispatch
    let mut w = vec![0.5f32; 8];
    linear_srgb::default::srgb_to_linear_slice(&mut w);

    // Breakdown: decode LUT
    let t = Instant::now();
    let v: Vec<f32> = (0..65536u32).map(|i| i as f32 / 65535.0).collect();
    let fill_us = t.elapsed().as_micros();

    let mut v = v;
    let t = Instant::now();
    linear_srgb::default::srgb_to_linear_slice(&mut v);
    let simd_s2l_us = t.elapsed().as_micros();

    let t = Instant::now();
    let _: Box<[f32; 65536]> = v.into_boxed_slice().try_into().ok().unwrap();
    let box_us = t.elapsed().as_micros();

    eprintln!("decode: fill={fill_us}µs  simd_s2l={simd_s2l_us}µs  box={box_us}µs");

    // Breakdown: encode LUT
    let t = Instant::now();
    let mut srgb: Vec<f32> = (0..=65536u32).map(|i| i as f32 / 65536.0).collect();
    let fill2_us = t.elapsed().as_micros();

    let t = Instant::now();
    linear_srgb::default::linear_to_srgb_slice(&mut srgb);
    let simd_l2s_us = t.elapsed().as_micros();

    let t = Instant::now();
    let u: Vec<u16> = srgb
        .iter()
        .map(|&s| (s * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16)
        .collect();
    let quant_us = t.elapsed().as_micros();

    let t = Instant::now();
    let _: Box<[u16; 65537]> = u.into_boxed_slice().try_into().ok().unwrap();
    let box2_us = t.elapsed().as_micros();

    eprintln!(
        "encode: fill={fill2_us}µs  simd_l2s={simd_l2s_us}µs  quant={quant_us}µs  box={box2_us}µs"
    );

    // Full generate functions
    let t = Instant::now();
    let d = linear_srgb::u16_lut::generate_decode_lut();
    let decode_us = t.elapsed().as_micros();

    let t = Instant::now();
    let e = linear_srgb::u16_lut::generate_encode_lut();
    let encode_us = t.elapsed().as_micros();

    eprintln!("---");
    eprintln!("generate_decode_lut: {decode_us}µs");
    eprintln!("generate_encode_lut: {encode_us}µs");
    eprintln!("total: {}µs", decode_us + encode_us);

    black_box(&d);
    black_box(&e);
}
