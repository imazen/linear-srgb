//! Shared by the frozen-output test and the explicit capture example.
#![allow(deprecated)]
#![forbid(unsafe_code)]

#[cfg(target_arch = "aarch64")]
use archmage::NeonToken;
#[cfg(target_arch = "wasm32")]
use archmage::Wasm128Token;
#[cfg(target_arch = "x86_64")]
use archmage::X64V3Token;
#[cfg(all(target_arch = "x86_64", feature = "v4"))]
use archmage::X64V4Token;
use linear_srgb::default as d;
use std::collections::BTreeMap;
use std::fmt::Write;

pub const CHUNK: usize = 4096;

macro_rules! token_suite {
    ($entry:ident, $token:ident, $module:path, $width:literal;
        [$($plain:ident),+]; $gamma_decode:ident, $gamma_encode:ident) => {
        #[archmage::arcane]
        fn $entry(token: $token) -> String {
            use $module as api;
            let mut text = String::new();
            let input = inputs();
            $(
                let mut output = Vec::new();
                for chunk in input.as_chunks::<$width>().0 {
                    output.extend(api::$plain(token, *chunk).into_iter().map(bits));
                }
                record(&mut text, concat!(stringify!($module), "/", stringify!($plain)), &output);
            )+
            for gamma in [0.5f32, 1.0, 1.8, 2.2, 2.4] {
                let mut decoded = Vec::new();
                let mut encoded = Vec::new();
                for chunk in input.as_chunks::<$width>().0 {
                    decoded.extend(api::$gamma_decode(token, *chunk, gamma).into_iter().map(bits));
                    encoded.extend(api::$gamma_encode(token, *chunk, gamma).into_iter().map(bits));
                }
                record(&mut text, &format!("{}/{}/{gamma}", stringify!($module), stringify!($gamma_decode)), &decoded);
                record(&mut text, &format!("{}/{}/{gamma}", stringify!($module), stringify!($gamma_encode)), &encoded);
            }
            text
        }
    };
}

#[cfg(target_arch = "x86_64")]
token_suite!(tokens_x4, X64V3Token, linear_srgb::tokens::x4, 4;
    [srgb_to_linear_v3, linear_to_srgb_v3, srgb_to_linear_extended_v3, linear_to_srgb_extended_v3,
     tf_srgb_to_linear_v3, tf_linear_to_srgb_v3, bt709_to_linear_v3, linear_to_bt709_v3,
     pq_to_linear_v3, linear_to_pq_v3, hlg_to_linear_v3, linear_to_hlg_v3];
    gamma_to_linear_v3, linear_to_gamma_v3);
#[cfg(target_arch = "x86_64")]
token_suite!(tokens_x8, X64V3Token, linear_srgb::tokens::x8, 8;
    [srgb_to_linear_v3, linear_to_srgb_v3, srgb_to_linear_extended_v3, linear_to_srgb_extended_v3,
     tf_srgb_to_linear_v3, tf_linear_to_srgb_v3, bt709_to_linear_v3, linear_to_bt709_v3,
     pq_to_linear_v3, linear_to_pq_v3, hlg_to_linear_v3, linear_to_hlg_v3];
    gamma_to_linear_v3, linear_to_gamma_v3);
#[cfg(target_arch = "aarch64")]
token_suite!(tokens_x4, NeonToken, linear_srgb::tokens::x4, 4;
    [srgb_to_linear_neon, linear_to_srgb_neon, srgb_to_linear_extended_neon, linear_to_srgb_extended_neon,
     tf_srgb_to_linear_neon, tf_linear_to_srgb_neon, bt709_to_linear_neon, linear_to_bt709_neon,
     pq_to_linear_neon, linear_to_pq_neon, hlg_to_linear_neon, linear_to_hlg_neon];
    gamma_to_linear_neon, linear_to_gamma_neon);
#[cfg(target_arch = "wasm32")]
token_suite!(tokens_x4, Wasm128Token, linear_srgb::tokens::x4, 4;
    [srgb_to_linear_wasm128, linear_to_srgb_wasm128, srgb_to_linear_extended_wasm128, linear_to_srgb_extended_wasm128,
     tf_srgb_to_linear_wasm128, tf_linear_to_srgb_wasm128, bt709_to_linear_wasm128, linear_to_bt709_wasm128,
     pq_to_linear_wasm128, linear_to_pq_wasm128, hlg_to_linear_wasm128, linear_to_hlg_wasm128];
    gamma_to_linear_wasm128, linear_to_gamma_wasm128);
#[cfg(all(target_arch = "x86_64", feature = "v4"))]
token_suite!(tokens_x16, X64V4Token, linear_srgb::tokens::x16, 16;
    [srgb_to_linear_v4, linear_to_srgb_v4,
     tf_srgb_to_linear_v4, tf_linear_to_srgb_v4, bt709_to_linear_v4, linear_to_bt709_v4,
     pq_to_linear_v4, linear_to_pq_v4, hlg_to_linear_v4, linear_to_hlg_v4];
    gamma_to_linear_v4, linear_to_gamma_v4);

pub fn profile() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => "x86_64",
        ("linux", "aarch64") => "aarch64",
        (_, "wasm32") if cfg!(feature = "precise-portable") => "wasm32-precise",
        (_, "wasm32") => "wasm32-default",
        (os, arch) => {
            panic!("no numerical golden profile for {os}/{arch}; capture that backend explicitly")
        }
    }
}

fn bits(value: f32) -> u32 {
    // NaN payload and sign are not part of the cross-tier numerical contract.
    if value.is_nan() {
        0x7fc0_0000
    } else {
        value.to_bits()
    }
}

fn edges() -> Vec<f32> {
    let mut values = vec![
        -0.0,
        0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0xffc0_1234),
        -4.0,
        -0.1,
        1e-8,
        1e-7,
        0.25,
        1.0,
        1.0001,
        4.0,
    ];
    for seam in [
        0.0030412826f32,
        0.03929337,
        0.0031308,
        0.04045,
        0.02,
        0.018053968,
        0.081242,
        0.5,
        1.0 / 12.0,
        1.0 / 1024.0,
        1.0,
    ] {
        for delta in -3..=3 {
            values.push(f32::from_bits(seam.to_bits().wrapping_add_signed(delta)));
        }
    }
    // Defined integer arithmetic, with no platform transcendental in the inputs.
    let mut state = 0x8a5c_31e7u32;
    while values.len() < 256 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        values.push(f32::from_bits(state));
    }
    values
}

fn inputs() -> Vec<f32> {
    let mut values: Vec<_> = (0..=65535).map(|i| i as f32 / 65535.0).collect();
    values.extend(edges());
    values
}

fn record(text: &mut String, name: &str, words: &[u32]) {
    write!(text, "{name} {}", words.len()).unwrap();
    for chunk in words.chunks(CHUNK) {
        // FNV-1a over explicit little-endian output words; not DefaultHasher.
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in chunk.iter().flat_map(|word| word.to_le_bytes()) {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
        write!(text, " {hash:016x}").unwrap();
    }
    text.push('\n');
}

fn scalar(text: &mut String, name: &str, f: impl Fn(f32) -> u32) {
    record(text, name, &inputs().into_iter().map(f).collect::<Vec<_>>());
}

fn slice(text: &mut String, name: &str, rgba: bool, f: impl Fn(&mut [f32])) {
    let mut values = inputs();
    f(&mut values);
    record(
        text,
        &format!("{name}/grid"),
        &values.into_iter().map(bits).collect::<Vec<_>>(),
    );
    let input = edges();
    let mut tails = Vec::new();
    for pixels in 0..=65 {
        let len = if rgba { pixels * 4 } else { pixels };
        let mut values: Vec<_> = (0..len).map(|i| input[(i + pixels * 7) % 256]).collect();
        f(&mut values);
        tails.push(len as u32);
        tails.extend(values.into_iter().map(bits));
    }
    record(text, &format!("{name}/tails"), &tails);
}

fn encode<T: Copy + Into<u32> + Default>(
    text: &mut String,
    name: &str,
    rgba: bool,
    f: impl Fn(&[f32], &mut [T]),
) {
    let input = inputs();
    let mut output = vec![T::default(); input.len()];
    f(&input, &mut output);
    record(
        text,
        &format!("{name}/grid"),
        &output.into_iter().map(Into::into).collect::<Vec<_>>(),
    );
    let edges = edges();
    let mut tails = Vec::new();
    for pixels in 0..=65 {
        let len = if rgba { pixels * 4 } else { pixels };
        let input: Vec<_> = (0..len).map(|i| edges[(i + pixels * 7) % 256]).collect();
        let mut output = vec![T::default(); len];
        f(&input, &mut output);
        tails.push(len as u32);
        tails.extend(output.into_iter().map(Into::into));
    }
    record(text, &format!("{name}/tails"), &tails);
}

fn decode<T: Copy>(
    text: &mut String,
    name: &str,
    rgba: bool,
    input: &[T],
    f: impl Fn(&[T], &mut [f32]),
) {
    let mut output = vec![0.0; input.len()];
    f(input, &mut output);
    record(
        text,
        &format!("{name}/grid"),
        &output.into_iter().map(bits).collect::<Vec<_>>(),
    );
    let mut tails = Vec::new();
    for pixels in 0..=65 {
        let len = if rgba { pixels * 4 } else { pixels };
        let input: Vec<_> = (0..len)
            .map(|i| input[(i + pixels * 7) % input.len()])
            .collect();
        let mut output = vec![0.0; len];
        f(&input, &mut output);
        tails.push(len as u32);
        tails.extend(output.into_iter().map(bits));
    }
    record(text, &format!("{name}/tails"), &tails);
}

pub fn capture() -> BTreeMap<&'static str, String> {
    let header = "# numerical-goldens-v1: FNV-1a/64; LE u32 words; 4096 words/chunk\n";
    let mut groups = BTreeMap::new();
    let mut text = header.to_owned();
    scalar(&mut text, "default/srgb_to_linear", |v| {
        bits(d::srgb_to_linear(v))
    });
    scalar(&mut text, "default/linear_to_srgb", |v| {
        bits(d::linear_to_srgb(v))
    });
    scalar(&mut text, "default/linear_to_srgb_u8", |v| {
        d::linear_to_srgb_u8(v).into()
    });
    scalar(&mut text, "default/linear_to_srgb_u16", |v| {
        d::linear_to_srgb_u16(v).into()
    });
    scalar(&mut text, "default/linear_to_srgb_u16_fast", |v| {
        d::linear_to_srgb_u16_fast(v).into()
    });
    let converter = d::SrgbConverter::new();
    scalar(&mut text, "lut/linear_to_srgb", |v| {
        bits(converter.linear_to_srgb(v))
    });
    macro_rules! scalar_tf {
        ($($name:ident),+ $(,)?) => {$(
            scalar(&mut text, concat!("tf/", stringify!($name)), |v| bits(d::$name(v)));
        )+};
    }
    scalar_tf!(
        bt709_to_linear,
        linear_to_bt709,
        pq_to_linear,
        linear_to_pq,
        hlg_to_linear,
        linear_to_hlg
    );
    groups.insert("scalar.txt", text);

    let mut text = header.to_owned();
    macro_rules! slices {
        ($text:ident, $rgba:literal; $($name:ident),+ $(,)?) => {$(
            slice(&mut $text, stringify!($name), $rgba, d::$name);
        )+};
    }
    slices!(text, false; srgb_to_linear_slice, linear_to_srgb_slice,
        srgb_to_linear_extended_slice, linear_to_srgb_extended_slice,
        bt709_to_linear_slice, linear_to_bt709_slice, pq_to_linear_slice,
        linear_to_pq_slice, hlg_to_linear_slice, linear_to_hlg_slice);
    for gamma in [0.5f32, 1.0, 1.8, 2.2, 2.4] {
        slice(&mut text, &format!("gamma_to_linear/{gamma}"), false, |v| {
            d::gamma_to_linear_slice(v, gamma)
        });
        slice(&mut text, &format!("linear_to_gamma/{gamma}"), false, |v| {
            d::linear_to_gamma_slice(v, gamma)
        });
    }
    groups.insert("slices.txt", text);

    let mut text = header.to_owned();
    slices!(text, true; srgb_to_linear_rgba_slice, linear_to_srgb_rgba_slice,
        srgb_to_linear_extended_rgba_slice, linear_to_srgb_extended_rgba_slice,
        srgb_to_linear_premultiply_rgba_slice, unpremultiply_linear_to_srgb_rgba_slice,
        bt709_to_linear_rgba_slice, linear_to_bt709_rgba_slice,
        pq_to_linear_rgba_slice, linear_to_pq_rgba_slice,
        hlg_to_linear_rgba_slice, linear_to_hlg_rgba_slice);
    for gamma in [0.5f32, 1.0, 1.8, 2.2, 2.4] {
        slice(
            &mut text,
            &format!("gamma_to_linear_rgba/{gamma}"),
            true,
            |v| d::gamma_to_linear_rgba_slice(v, gamma),
        );
        slice(
            &mut text,
            &format!("linear_to_gamma_rgba/{gamma}"),
            true,
            |v| d::linear_to_gamma_rgba_slice(v, gamma),
        );
        slice(
            &mut text,
            &format!("gamma_to_linear_premultiply/{gamma}"),
            true,
            |v| d::gamma_to_linear_premultiply_rgba_slice(v, gamma),
        );
        slice(
            &mut text,
            &format!("unpremultiply_linear_to_gamma/{gamma}"),
            true,
            |v| d::unpremultiply_linear_to_gamma_rgba_slice(v, gamma),
        );
    }
    groups.insert("rgba.txt", text);

    let mut text = header.to_owned();
    macro_rules! encoders {
        ($rgba:literal; $($name:ident),+ $(,)?) => {$(
            encode(&mut text, stringify!($name), $rgba, d::$name);
        )+};
    }
    encoders!(false; linear_to_srgb_u8_slice, linear_to_srgb_u16_slice, linear_to_srgb_u16_slice_fast);
    encoders!(true; linear_to_srgb_u8_rgba_slice, linear_to_srgb_u16_rgba_slice,
        linear_to_srgb_u16_rgba_slice_fast, unpremultiply_linear_to_srgb_u8_rgba_slice);
    let u8_input: Vec<_> = (0..=65535).map(|i| i as u8).collect();
    let u16_input: Vec<_> = (0..=65535u16).collect();
    decode(
        &mut text,
        "srgb_u8_to_linear_slice",
        false,
        &u8_input,
        d::srgb_u8_to_linear_slice,
    );
    decode(
        &mut text,
        "srgb_u8_to_linear_rgba_slice",
        true,
        &u8_input,
        d::srgb_u8_to_linear_rgba_slice,
    );
    decode(
        &mut text,
        "srgb_u8_to_linear_premultiply_rgba_slice",
        true,
        &u8_input,
        d::srgb_u8_to_linear_premultiply_rgba_slice,
    );
    decode(
        &mut text,
        "srgb_u16_to_linear_slice",
        false,
        &u16_input,
        d::srgb_u16_to_linear_slice,
    );
    decode(
        &mut text,
        "srgb_u16_to_linear_rgba_slice",
        true,
        &u16_input,
        d::srgb_u16_to_linear_rgba_slice,
    );
    record(
        &mut text,
        "scalar/srgb_u16_to_linear",
        &u16_input
            .into_iter()
            .map(|v| bits(d::srgb_u16_to_linear(v)))
            .collect::<Vec<_>>(),
    );
    groups.insert("integers.txt", text);

    let mut text = "# Complete u8 decode: input 0..255, canonical f32 output bits\n".to_owned();
    for value in 0..=255u8 {
        writeln!(text, "{value:03} {:08x}", bits(d::srgb_u8_to_linear(value))).unwrap();
    }
    groups.insert("u8_decode.txt", text);
    use archmage::SimdToken;
    let mut text = header.to_owned();
    #[cfg(target_arch = "x86_64")]
    {
        let token = archmage::X64V3Token::summon()
            .expect("the x86-64 numerical baseline requires AVX2+FMA");
        text.push_str(&tokens_x4(token));
        text.push_str(&tokens_x8(token));
    }
    #[cfg(target_arch = "aarch64")]
    text.push_str(&tokens_x4(
        archmage::NeonToken::summon().expect("NEON required"),
    ));
    #[cfg(target_arch = "wasm32")]
    text.push_str(&tokens_x4(
        archmage::Wasm128Token::summon().expect("SIMD128 required"),
    ));
    groups.insert("tokens.txt", text);
    #[cfg(all(target_arch = "x86_64", feature = "v4"))]
    groups.insert(
        "tokens-v4.txt",
        header.to_owned()
            + &tokens_x16(
                archmage::X64V4Token::summon().expect("v4 capture explicitly requires AVX-512"),
            ),
    );
    groups
}

pub fn compare(name: &str, expected: &str, actual: &str) -> Vec<String> {
    let rows = |text: &str| -> BTreeMap<String, Vec<String>> {
        let mut rows = BTreeMap::new();
        for line in text.lines().filter(|line| !line.starts_with('#')) {
            let mut fields = line.split_whitespace();
            let key = fields.next().expect("empty golden row").to_owned();
            assert!(
                rows.insert(key.clone(), fields.map(str::to_owned).collect())
                    .is_none(),
                "duplicate {name}/{key}"
            );
        }
        rows
    };
    let expected = rows(expected);
    let actual = rows(actual);
    assert!(!expected.is_empty(), "missing golden rows in {name}");
    let mut errors = Vec::new();
    for (key, want) in &expected {
        match actual.get(key) {
            None => errors.push(format!("{name}/{key}: missing case")),
            Some(got) if name == "u8_decode.txt" => {
                if got != want {
                    errors.push(format!(
                        "{name}/{key}: output bits {got:?}, expected {want:?}"
                    ));
                }
            }
            Some(got) => {
                if got.first() != want.first() || got.len() != want.len() {
                    errors.push(format!("{name}/{key}: output/chunk count changed"));
                }
                for (chunk, (a, b)) in got.iter().skip(1).zip(want.iter().skip(1)).enumerate() {
                    if a != b {
                        let len: usize = want[0].parse().expect("invalid golden word count");
                        errors.push(format!(
                            "{name}/{key}: words {}..{} hash {a}, expected {b}",
                            chunk * CHUNK,
                            ((chunk + 1) * CHUNK).min(len)
                        ));
                    }
                }
            }
        }
    }
    for key in actual.keys().filter(|key| !expected.contains_key(*key)) {
        errors.push(format!("{name}/{key}: unexpected case"));
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_bit_change_identifies_the_partial_chunk() {
        let mut words = vec![0u32; CHUNK + 1];
        let mut expected = String::new();
        record(&mut expected, "probe", &words);
        words[CHUNK] ^= 1;
        let mut actual = String::new();
        record(&mut actual, "probe", &words);
        let errors = compare("fixture", &expected, &actual);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("words 4096..4097"));
    }

    #[test]
    fn signed_zero_is_distinct_and_nan_payloads_are_canonical() {
        assert_ne!(bits(0.0), bits(-0.0));
        assert_eq!(
            bits(f32::from_bits(0x7fc0_1234)),
            bits(f32::from_bits(0xffc0_4321))
        );
        let mut positive = String::new();
        let mut negative = String::new();
        record(&mut positive, "zero", &[bits(0.0)]);
        record(&mut negative, "zero", &[bits(-0.0)]);
        assert_eq!(compare("fixture", &positive, &negative).len(), 1);
    }

    #[test]
    fn missing_extra_and_truncated_cases_are_rejected() {
        let mut original = String::new();
        record(&mut original, "original", &[123]);
        let mut extra = String::new();
        record(&mut extra, "extra", &[456]);
        assert_eq!(compare("fixture", &original, &extra).len(), 2);
        assert_eq!(
            compare("fixture", &original, &(original.clone() + &extra)).len(),
            1
        );
        assert!(compare("fixture", &original, "original 1\n")[0].contains("count changed"));
    }
}
