//! A value converts to the same bits wherever it sits in a slice.
use linear_srgb::default as d;

/// Issue #32's exact witness must be covered independently of a sampling grid.
#[test]
fn issue_32_srgb_body_and_tail_share_arithmetic() {
    type Conversion = (&'static str, fn(&mut [f32]));
    let conversions: [Conversion; 2] = [
        ("decode", d::srgb_to_linear_slice),
        ("encode", d::linear_to_srgb_slice),
    ];
    for (name, convert) in conversions {
        for input in [0.125_f32, 0.9_f32] {
            let mut whole = [input; 65];
            convert(&mut whole);
            let expected = whole[0].to_bits();
            for len in 0..=65 {
                let mut values = vec![input; len];
                convert(&mut values);
                for (i, value) in values.iter().enumerate() {
                    assert_eq!(
                        value.to_bits(),
                        expected,
                        "issue #32: {name} input={:08x} len={len} index={i}",
                        input.to_bits()
                    );
                }
            }
        }
    }
}

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

/// ST 2084 in f64, the reference for the slices.
#[cfg(feature = "transfer")]
mod st2084 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;
    pub fn eotf(v: f64) -> f64 {
        if v <= 0.0 {
            return 0.0;
        }
        let p = v.powf(1.0 / M2);
        ((p - C1).max(0.0) / (C2 - C3 * p)).powf(1.0 / M1)
    }
    pub fn oetf(l: f64) -> f64 {
        if l <= 0.0 {
            return 0.0;
        }
        let p = l.powf(M1);
        ((C1 + C2 * p) / (1.0 + C3 * p)).powf(M2)
    }
}

/// Across the whole range, near black included, rounded to 16 bits.
#[cfg(feature = "transfer")]
#[test]
fn pq_slices_stay_within_one_u16_code_of_st2084() {
    let code = |v: f64| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as i64;
    let signal: Vec<f32> = (0..=1_000_000).map(|i| i as f32 / 1e6).collect();
    let mut out = signal.clone();
    d::pq_to_linear_slice(&mut out);
    for (x, y) in signal.iter().zip(&out) {
        let want = st2084::eotf(f64::from(*x));
        assert!(
            (code(f64::from(*y)) - code(want)).abs() <= 1,
            "eotf({x}) = {y}, want {want}"
        );
    }
    let linear: Vec<f32> = (0..=1_000_000)
        .map(|i| {
            let t = i as f32 / 1e6;
            t * t * t * t
        })
        .collect();
    let mut out = linear.clone();
    d::linear_to_pq_slice(&mut out);
    for (x, y) in linear.iter().zip(&out) {
        let want = st2084::oetf(f64::from(*x));
        assert!(
            (code(f64::from(*y)) - code(want)).abs() <= 1,
            "oetf({x}) = {y}, want {want}"
        );
    }
}
