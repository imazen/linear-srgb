#[path = "../benches/support/u8_options.rs"]
mod options;
use linear_srgb::default as d;
fn main() {
    let o = options::Options::new();
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--emit-size") {
        let directory = std::path::Path::new(&args[i + 1]);
        std::fs::create_dir_all(directory).unwrap();
        let table = |values: &[u8]| {
            values
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        let thresholds = |values: &[f32]| {
            values
                .iter()
                .map(|v| format!("f32::from_bits({})", v.to_bits()))
                .collect::<Vec<_>>()
                .join(",")
        };
        for strategy in [
            "lut12",
            "lut16",
            "corrected",
            "floor_corrected",
            "polynomial",
        ] {
            let body=match strategy {
                "lut12" => "linear_srgb::default::linear_to_srgb_u8_slice(input,output);".into(),
                "lut16" => format!("static TABLE:[u8;65536]=[{}];for (&x,y) in input.iter().zip(output) {{*y=TABLE[(x.clamp(0.0,1.0)*65535.0+0.5) as usize];}}",table(&o.wide)),
                "polynomial" => "for (&x,y) in input.iter().zip(output) {*y=(f64::from(linear_srgb::default::linear_to_srgb(x))*255.0+0.5) as u8;}".into(),
                "corrected" => format!("static T:[f32;255]=[{}];for (&x,y) in input.iter().zip(output) {{let q=linear_srgb::default::linear_to_srgb_u8(x) as usize;*y=if q<255 && x>=T[q] {{(q+1) as u8}} else if q>0 && x<T[q-1] {{(q-1) as u8}} else {{q as u8}};}}",thresholds(&o.thresholds)),
                _ => format!("static LOWER:[u8;4096]=[{}];static UPPER:[f32;256]=[{}];let o=crate::options::Options{{wide:Vec::new().into_boxed_slice(),thresholds:[0.0;255],lower:LOWER.to_vec().into_boxed_slice(),upper:UPPER}};crate::options::floor_corrected_slice(input,output,&o);",table(&o.lower),thresholds(&o.upper)),
            };
            let path = directory.join(format!("{strategy}.rs"));
            assert!(!path.exists(), "Choose a fresh size-output directory");
            std::fs::write(
                path,
                format!("pub fn encode(input:&[f32],output:&mut[u8]){{{body}}}\n"),
            )
            .unwrap();
        }
        return;
    }
    let mut bad = [0u64; 5];
    let mut worst = [0u8; 5];
    let full = std::env::args().any(|x| x == "--exhaustive");
    let n = if full {
        0x3f800001u32
    } else {
        (1 << 22) + (1 << 20)
    };
    for i in 0..n {
        let x = if full {
            f32::from_bits(i)
        } else if i < (1 << 22) {
            i as f32 / ((1 << 22) - 1) as f32
        } else {
            10f64.powf(-6.0 + 6.0 * (i - (1 << 22)) as f64 / (1 << 20) as f64) as f32
        };
        let want = options::reference(x);
        let got = [
            d::linear_to_srgb_u8(x),
            o.wide(x),
            o.corrected(x),
            o.polynomial(x),
            o.floor_corrected(x),
        ];
        for k in 0..5 {
            let diff = got[k].abs_diff(want);
            bad[k] += u64::from(diff != 0);
            worst[k] = worst[k].max(diff);
        }
    }
    println!("strategy\tinputs\tdiffer\tmax_codes\ttable_bytes");
    for (k, (name, bytes)) in [
        ("lut12", 4096),
        ("lut16", 65536),
        ("corrected", 5116),
        ("polynomial", 0),
        ("floor_corrected", 5120),
    ]
    .iter()
    .enumerate()
    {
        println!("{name}\t{n}\t{}\t{}\t{bytes}", bad[k], worst[k]);
    }
    // Check all quantization thresholds against powf independently of the sweep.
    for &seam in &o.thresholds {
        for x in [seam.next_down(), seam, seam.next_up()] {
            assert_eq!(o.corrected(x), options::reference(x), "boundary {x}");
            assert_eq!(
                o.floor_corrected(x),
                options::reference(x),
                "floor boundary {x}"
            );
        }
    }
    for x in [
        f32::NAN,
        f32::NEG_INFINITY,
        f32::INFINITY,
        -0.0,
        -1.0,
        0.0,
        1.0,
        2.0,
    ] {
        assert_eq!(o.floor_corrected(x), options::reference(x));
    }
    let rgba = [0.0, 0.1, 0.2, 0.5, 0.01, 0.02, 0.03, 0.75];
    let mut dst = [0u8; 8];
    options::floor_corrected_rgba_slice(&rgba, &mut dst, &o);
    for (s, d) in rgba.as_chunks::<4>().0.iter().zip(dst.as_chunks::<4>().0) {
        for i in 0..3 {
            assert_eq!(d[i], options::reference(s[i] / s[3]));
        }
    }
    let input: Vec<f32> = (0..=65535).map(|i| i as f32 / 65535.0).collect();
    let mut out = vec![0u8; input.len()];
    options::floor_corrected_slice(&input, &mut out, &o);
    for (&x, &y) in input.iter().zip(&out) {
        assert_eq!(y, options::reference(x));
    }
    if full {
        let mut input = vec![0.0f32; 4096];
        let mut output = vec![0u8; 4096];
        let mut bad = 0u64;
        for start in (0..n).step_by(4096) {
            let count = ((n - start) as usize).min(4096);
            for (i, x) in input[..count].iter_mut().enumerate() {
                *x = f32::from_bits(start + i as u32);
            }
            options::floor_corrected_slice(&input[..count], &mut output[..count], &o);
            bad += input[..count]
                .iter()
                .zip(&output[..count])
                .filter(|(x, y)| options::reference(**x) != **y)
                .count() as u64;
        }
        println!("floor_corrected_slice\t{n}\t{bad}\t0\t5120");
        assert_eq!(bad, 0);
    }
}
