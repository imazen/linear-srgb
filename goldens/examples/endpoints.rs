//! Inspect endpoint differences without dumping a large output grid.
#![forbid(unsafe_code)]
#![allow(deprecated)]

#[cfg(target_arch = "x86_64")]
use archmage::{SimdToken, X64V3Token};

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn probe(token: X64V3Token) {
    use linear_srgb::tokens::{x4, x8};
    for input in [
        0.0f32,
        f32::from_bits(1.0f32.to_bits() - 1),
        1.0,
        1.0001,
        4.0,
        f32::INFINITY,
    ] {
        let d4 = x4::srgb_to_linear_v3(token, [input; 4])[0];
        let e4 = x4::linear_to_srgb_v3(token, [input; 4])[0];
        let d8 = x8::srgb_to_linear_v3(token, [input; 8])[0];
        let e8 = x8::linear_to_srgb_v3(token, [input; 8])[0];
        println!(
            "input={:08x} x4_decode={:08x} x8_decode={:08x} x4_encode={:08x} x8_encode={:08x}",
            input.to_bits(),
            d4.to_bits(),
            d8.to_bits(),
            e4.to_bits(),
            e8.to_bits()
        );
    }
}

fn main() {
    #[cfg(target_arch = "x86_64")]
    probe(X64V3Token::summon().expect("endpoint probe requires AVX2+FMA"));
    #[cfg(not(target_arch = "x86_64"))]
    panic!("this endpoint comparison example requires x86-64");
}
