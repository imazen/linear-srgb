#![cfg(feature = "iec")]
use linear_srgb::iec;

#[test]
fn textbook_junction_uses_the_published_linear_segment() {
    // ICC's sRGB interpretation uses <= 0.04045 for decode and <= 0.0031308
    // for encode, rather than deriving the encode cutoff from 0.04045/12.92.
    let decoded = iec::srgb_to_linear_f64(0.04045);
    assert!(
        (decoded - 0.04045 / 12.92).abs() < 1e-18,
        "decode {decoded}"
    );
    let encoded = iec::linear_to_srgb_f64(0.0031308);
    assert_eq!(encoded, 0.0031308 * 12.92);
    let above = 0.003130802f64;
    let want = 1.055 * above.powf(1.0 / 2.4) - 0.055;
    assert!((iec::linear_to_srgb_f64(above) - want).abs() < 1e-16);
}
