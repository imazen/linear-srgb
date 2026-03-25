//! Compare encode LUT strategies for roundtrip accuracy.
//!
//! cargo run --example encode_lut_strategies --release --all-features

fn main() {
    use linear_srgb::default::srgb_u16_to_linear;

    // Build decode LUT (same for all strategies — it's the source of truth)
    let decode: Vec<f32> = (0..=65535u16).map(srgb_u16_to_linear).collect();

    // Strategy 1: Current — uniform linear index, 65537 entries
    test_strategy(
        "uniform 65537",
        65537,
        |linear| (linear * 65536.0 + 0.5) as usize,
        |i, n| i as f32 / (n - 1) as f32,
        &decode,
    );

    // Strategy 2: uniform, 2× entries
    test_strategy(
        "uniform 131073",
        131073,
        |linear| (linear * 131072.0 + 0.5) as usize,
        |i, n| i as f32 / (n - 1) as f32,
        &decode,
    );

    // Strategy 3: uniform, 4× entries
    test_strategy(
        "uniform 262145",
        262145,
        |linear| (linear * 262144.0 + 0.5) as usize,
        |i, n| i as f32 / (n - 1) as f32,
        &decode,
    );

    // Strategy 4: sqrt-indexed, 65537 entries
    test_strategy(
        "sqrt 65537",
        65537,
        |linear| (linear.sqrt() * 65536.0 + 0.5) as usize,
        |i, n| {
            let t = i as f32 / (n - 1) as f32;
            t * t // inverse of sqrt
        },
        &decode,
    );

    // Strategy 5: sqrt-indexed, 32769 entries (half size!)
    test_strategy(
        "sqrt 32769",
        32769,
        |linear| (linear.sqrt() * 32768.0 + 0.5) as usize,
        |i, n| {
            let t = i as f32 / (n - 1) as f32;
            t * t
        },
        &decode,
    );

    // Strategy 6: cbrt-indexed (cube root), 65537 entries
    test_strategy(
        "cbrt 65537",
        65537,
        |linear| (linear.cbrt() * 65536.0 + 0.5) as usize,
        |i, n| {
            let t = i as f32 / (n - 1) as f32;
            t * t * t
        },
        &decode,
    );

    // Strategy 7: pow(1/2.4) indexed — matches sRGB gamma, should linearize the index
    test_strategy(
        "pow(1/2.4) 65537",
        65537,
        |linear| (linear.powf(1.0 / 2.4) * 65536.0 + 0.5) as usize,
        |i, n| {
            let t = i as f32 / (n - 1) as f32;
            t.powf(2.4)
        },
        &decode,
    );

    // Strategy 8: two-range table — dense toe + coarse main, same 128KB
    eprintln!("\n--- Two-range (dense toe + uniform main, total ~128KB) ---");

    for &(toe_thresh, toe_n, main_n, label) in &[
        (0.01f32, 8192usize, 57345usize, "toe8K+main57K T=.01"),
        (0.02, 8192, 57345, "toe8K+main57K T=.02"),
        (0.01, 16384, 49153, "toe16K+main49K T=.01"),
        (0.005, 8192, 57345, "toe8K+main57K T=.005"),
        (0.01, 4096, 61441, "toe4K+main61K T=.01"),
    ] {
        let toe_scale = (toe_n - 1) as f32 / toe_thresh;
        let main_range = 1.0 - toe_thresh;
        let main_scale = (main_n - 1) as f32 / main_range;

        // Generate toe LUT
        let toe_lut: Vec<u16> = (0..toe_n)
            .map(|i| {
                let linear = i as f32 / toe_scale;
                let srgb = linear_srgb::default::linear_to_srgb(linear);
                (srgb * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16
            })
            .collect();

        // Generate main LUT
        let main_lut: Vec<u16> = (0..main_n)
            .map(|i| {
                let linear = toe_thresh + i as f32 / main_scale;
                let srgb = linear_srgb::default::linear_to_srgb(linear);
                (srgb * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16
            })
            .collect();

        let mut max_diff = 0u32;
        let mut worst = 0u16;
        let mut exact = 0u32;
        let mut diff_hist = [0u32; 16];

        for i in 0..=65535u16 {
            let linear = decode[i as usize];
            let back = if linear < toe_thresh {
                let idx = (linear * toe_scale + 0.5) as usize;
                toe_lut[idx.min(toe_n - 1)]
            } else {
                let idx = ((linear - toe_thresh) * main_scale + 0.5) as usize;
                main_lut[idx.min(main_n - 1)]
            };
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

        let mem_kb = (toe_n + main_n) * 2 / 1024;
        eprint!(
            "{label:>25} ({mem_kb:>3}KB): exact={exact:>5}/65536 ({:>5.1}%) max_diff={max_diff}",
            exact as f64 / 65536.0 * 100.0
        );
        if max_diff > 0 {
            eprint!(" at {worst}");
        }
        eprint!("  [");
        for (d, &count) in diff_hist.iter().enumerate() {
            if count > 0 && d > 0 {
                eprint!(" ±{d}:{count}");
            }
        }
        eprintln!("]");
    }

    // Strategy 9: analytical toe + power-only LUT
    // Linear segment (< 0.003041): exact multiply, no table
    // Power segment: uniform LUT over [threshold, 1.0] only
    eprintln!("\n--- Split strategies (analytical toe + power LUT) ---");

    let lin_thresh: f32 = 0.003_041_282_6;

    for &(power_n, label) in &[
        (65537usize, "split 65537"),
        (32769, "split 32769"),
        (16385, "split 16385"),
    ] {
        let power_range = 1.0 - lin_thresh;
        let power_scale = (power_n - 1) as f32 / power_range;

        // Generate power-segment LUT
        let power_lut: Vec<u16> = (0..power_n)
            .map(|i| {
                let linear = lin_thresh + i as f32 / power_scale;
                let srgb = linear_srgb::default::linear_to_srgb(linear);
                (srgb * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16
            })
            .collect();

        // Test roundtrip
        let mut max_diff = 0u32;
        let mut worst = 0u16;
        let mut exact = 0u32;
        let mut diff_hist = [0u32; 16];

        for i in 0..=65535u16 {
            let linear = decode[i as usize];
            let back = if linear < lin_thresh {
                // Analytical toe: exact
                (linear * 12.92 * 65535.0 + 0.5) as u16
            } else {
                let idx = ((linear - lin_thresh) * power_scale + 0.5) as usize;
                power_lut[idx.min(power_n - 1)]
            };
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

        let mem_kb = power_n * 2 / 1024;
        eprint!(
            "{label:>20} ({mem_kb:>3}KB): exact={exact:>5}/65536 ({:>5.1}%) max_diff={max_diff}",
            exact as f64 / 65536.0 * 100.0
        );
        if max_diff > 0 {
            eprint!(" at {worst}");
        }
        eprint!("  [");
        for (d, &count) in diff_hist.iter().enumerate() {
            if count > 0 && d > 0 {
                eprint!(" ±{d}:{count}");
            }
        }
        eprintln!("]");
    }
}

fn test_strategy(
    name: &str,
    n: usize,
    index_fn: impl Fn(f32) -> usize,
    inverse_fn: impl Fn(usize, usize) -> f32,
    decode: &[f32],
) {
    // Generate encode LUT using the inverse function
    let encode: Vec<u16> = (0..n)
        .map(|i| {
            let linear = inverse_fn(i, n);
            let srgb = linear_srgb::default::linear_to_srgb(linear);
            (srgb * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16
        })
        .collect();

    // Test roundtrip: for each u16 sRGB value, decode then encode
    let mut max_diff = 0u32;
    let mut worst = 0u16;
    let mut exact = 0u32;
    let mut diff_hist = [0u32; 16];

    for i in 0..=65535u16 {
        let linear = decode[i as usize];
        let idx = index_fn(linear).min(n - 1);
        let back = encode[idx];
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

    let mem_kb = n * 2 / 1024;
    eprint!(
        "{name:>20} ({mem_kb:>3}KB): exact={exact:>5}/65536 ({:>5.1}%) max_diff={max_diff}",
        exact as f64 / 65536.0 * 100.0
    );
    if max_diff > 0 {
        eprint!(" at {worst}");
    }
    eprint!("  [");
    for (d, &count) in diff_hist.iter().enumerate() {
        if count > 0 && d > 0 {
            eprint!(" ±{d}:{count}");
        }
    }
    eprintln!("]");
}
