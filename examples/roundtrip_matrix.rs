//! M×N roundtrip matrix: every decode method × every encode method.
//!
//! cargo run --example roundtrip_matrix --release --all-features

#[allow(clippy::redundant_closure, clippy::type_complexity)]
fn main() {
    // Decode methods: u16 → f32
    let decoders: Vec<(&str, Box<dyn Fn(u16) -> f32>)> = vec![
        (
            "LUT decode",
            Box::new(linear_srgb::default::srgb_u16_to_linear),
        ),
        (
            "poly decode",
            Box::new(|v| linear_srgb::default::srgb_to_linear(v as f32 / 65535.0)),
        ),
        (
            "precise decode",
            Box::new(|v| linear_srgb::precise::srgb_to_linear(v as f32 / 65535.0)),
        ),
    ];

    // Encode methods: f32 → u16
    let encoders: Vec<(&str, Box<dyn Fn(f32) -> u16>)> = vec![
        (
            "poly encode",
            Box::new(linear_srgb::default::linear_to_srgb_u16),
        ),
        (
            "sqrt LUT encode",
            Box::new(linear_srgb::default::linear_to_srgb_u16_fast),
        ),
        (
            "precise encode",
            Box::new(|l| {
                (linear_srgb::precise::linear_to_srgb(l) * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16
            }),
        ),
    ];

    eprintln!(
        "{:>18} × {:<18}  {:>7}  {:>5}  {:>5}  {:>7}",
        "decode", "encode", "exact%", "max±", "±1", "±2+"
    );
    eprintln!("{}", "-".repeat(75));

    for (dec_name, decode) in &decoders {
        for (enc_name, encode) in &encoders {
            let mut max_diff = 0u32;
            let mut exact = 0u32;
            let mut within_1 = 0u32;
            let mut over_1 = 0u32;

            for i in 0..=65535u16 {
                let linear = decode(i);
                let back = encode(linear);
                let diff = (i as i32 - back as i32).unsigned_abs();
                max_diff = max_diff.max(diff);
                match diff {
                    0 => exact += 1,
                    1 => within_1 += 1,
                    _ => over_1 += 1,
                }
            }

            eprintln!(
                "{dec_name:>18} × {enc_name:<18}  {:>6.1}%  {:>4}  {:>5}  {:>7}",
                exact as f64 / 65536.0 * 100.0,
                max_diff,
                within_1,
                over_1
            );
        }
    }
}
