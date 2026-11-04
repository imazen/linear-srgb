//! A value converts to the same bits wherever it sits in a slice.
use linear_srgb::default as d;

fn check(name: &str, f: &dyn Fn(&mut [f32])) {
    let input: Vec<f32> = (0..67).map(|i| (i as f32 * 0.7371).fract()).collect();
    let mut whole = input.clone();
    f(&mut whole);
    for len in 1..input.len() {
        let mut part = input[..len].to_vec();
        f(&mut part);
        let same = part
            .iter()
            .zip(&whole)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "{name} differs at length {len}");
    }
}

#[test]
fn slice_output_does_not_depend_on_position() {
    check("srgb_to_linear", &|v| d::srgb_to_linear_slice(v));
    check("linear_to_srgb", &|v| d::linear_to_srgb_slice(v));
    check("gamma_to_linear", &|v| d::gamma_to_linear_slice(v, 2.2));
    check("linear_to_gamma", &|v| d::linear_to_gamma_slice(v, 2.2));
}

#[cfg(feature = "transfer")]
#[test]
fn transfer_slice_output_does_not_depend_on_position() {
    check("bt709_to_linear", &|v| d::bt709_to_linear_slice(v));
    check("linear_to_bt709", &|v| d::linear_to_bt709_slice(v));
    check("pq_to_linear", &|v| d::pq_to_linear_slice(v));
    check("linear_to_pq", &|v| d::linear_to_pq_slice(v));
    check("hlg_to_linear", &|v| d::hlg_to_linear_slice(v));
    check("linear_to_hlg", &|v| d::linear_to_hlg_slice(v));
}
