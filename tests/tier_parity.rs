//! Every native tier gives the same bits, so a result checked on one machine holds on all.
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use linear_srgb::default as d;

const GAMMA: f32 = 563.0 / 256.0;

fn samples() -> Vec<f32> {
    let mut v: Vec<f32> = (0..=300_000).map(|i| i as f32 / 3e5).collect();
    v.extend((0..=100_000).map(|i| {
        let t = i as f32 / 1e5;
        t * t * t * t
    }));
    v.extend((1..2000).map(|i| -(i as f32) / 2000.0));
    v.extend((1..2000).map(|i| 1.0 + i as f32 / 100.0));
    // Around every branch seam, in both directions.
    for seam in [
        0.02f32,
        0.25,
        1e-7,
        1e-4,
        1.0 / 12.0,
        0.5,
        0.081242,
        0.018053968,
        0.04045,
        0.0031308,
    ] {
        v.extend((0..=2048u32).map(|i| f32::from_bits(seam.to_bits() - 1024 + i)));
    }
    v
}

#[test]
fn every_slice_gives_the_same_bits_at_every_tier() {
    type Slice = (&'static str, fn(&mut [f32]));
    let slices: [Slice; 12] = [
        ("srgb_to_linear", |v| d::srgb_to_linear_slice(v)),
        ("linear_to_srgb", |v| d::linear_to_srgb_slice(v)),
        ("srgb_to_linear_extended", |v| {
            d::srgb_to_linear_extended_slice(v)
        }),
        ("linear_to_srgb_extended", |v| {
            d::linear_to_srgb_extended_slice(v)
        }),
        ("gamma_to_linear", |v| d::gamma_to_linear_slice(v, GAMMA)),
        ("linear_to_gamma", |v| d::linear_to_gamma_slice(v, GAMMA)),
        ("bt709_to_linear", |v| d::bt709_to_linear_slice(v)),
        ("linear_to_bt709", |v| d::linear_to_bt709_slice(v)),
        ("pq_to_linear", |v| d::pq_to_linear_slice(v)),
        ("linear_to_pq", |v| d::linear_to_pq_slice(v)),
        ("hlg_to_linear", |v| d::hlg_to_linear_slice(v)),
        ("linear_to_hlg", |v| d::linear_to_hlg_slice(v)),
    ];
    let input = samples();
    let mut expected: Vec<Vec<u32>> = Vec::new();
    let mut failures = Vec::new();
    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        for (i, (name, f)) in slices.iter().enumerate() {
            let mut out = input.clone();
            f(&mut out);
            let bits: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
            if let Some(want) = expected.get(i) {
                if let Some(k) = (0..bits.len()).find(|&k| bits[k] != want[k]) {
                    let n = bits.iter().zip(want).filter(|(a, b)| a != b).count();
                    failures.push(format!(
                        "{name} at {}: {n} differ, first input {} gives {} against {}",
                        perm.label,
                        input[k],
                        f32::from_bits(bits[k]),
                        f32::from_bits(want[k])
                    ));
                }
            } else {
                expected.push(bits);
            }
        }
    });
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The formula branches meet without a step down.
#[test]
fn slices_do_not_step_down_at_branch_seams() {
    type Seam = (&'static str, fn(&mut [f32]), f32);
    let seams: [Seam; 3] = [
        ("linear_to_hlg", |v| d::linear_to_hlg_slice(v), 1.0 / 12.0),
        ("bt709_to_linear", |v| d::bt709_to_linear_slice(v), 0.081242),
        ("hlg_to_linear", |v| d::hlg_to_linear_slice(v), 0.5),
    ];
    for (name, f, seam) in seams {
        let center = seam.to_bits();
        let input: Vec<f32> = (0..=4096u32)
            .map(|i| f32::from_bits(center - 2048 + i))
            .collect();
        let mut out = input.clone();
        f(&mut out);
        // Only the step across the seam: within a branch, noise is a separate matter.
        let (mut prev, mut biggest_drop) = (f32::NEG_INFINITY, 0u32);
        for v in &out {
            if *v < prev {
                biggest_drop = biggest_drop.max(v.to_bits().abs_diff(prev.to_bits()));
            }
            prev = *v;
        }
        assert!(biggest_drop <= 2, "{name}: {biggest_drop} ULP");
    }
}

/// The 16-bit encoder too, including rows too short for a vector block.
#[test]
fn u16_encode_gives_the_same_codes_at_every_tier_and_length() {
    let input = samples();
    let mut first: Vec<u16> = Vec::new();
    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut whole = vec![0u16; input.len()];
        d::linear_to_srgb_u16_slice(&input, &mut whole);
        for (x, want) in input.iter().zip(&whole).step_by(97) {
            let mut one = [0u16];
            d::linear_to_srgb_u16_slice(&[*x], &mut one);
            assert_eq!(one[0], *want, "{x} alone at {}", perm.label);
        }
        if first.is_empty() {
            first = whole;
        } else {
            assert!(first == whole, "differs at {}", perm.label);
        }
    });
}
