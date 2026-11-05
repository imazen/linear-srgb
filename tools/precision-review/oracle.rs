//! Integer-domain and dense-grid checks against f64 formulae.
//! The generated differential driver separately covers the public API snapshots.
use linear_srgb::default as ls;
fn s2l(v: f64) -> f64 { if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } }
fn l2s(v: f64) -> f64 { if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 } }
fn ulp(a: f32, b: f32) -> i64 { (a.to_bits() as i64 - b.to_bits() as i64).abs() }
fn grid() -> Vec<f32> {
    // Uniform in linear, plus log spaced in the darks where sRGB is steepest.
    let mut v: Vec<f32> = (0..(1u32 << 22)).map(|i| i as f32 / ((1u32 << 22) - 1) as f32).collect();
    v.extend((0..(1u32 << 20)).map(|i| (10f64.powf(-6.0 + 6.0 * i as f64 / (1u32 << 20) as f64)) as f32));
    v
}
fn report(name: &str, n: usize, bad: usize, worst: i64, unit: &str, example: String) {
    println!("{name}\t{n}\t{bad}\t{:.4}%\t{worst} {unit}\t{example}", bad as f64 * 100.0 / n as f64);
}
// C0REF: the library's own constants.
const A: f64 = 0.055_010_718_947_586_6;
const LT: f64 = 0.003_041_282_560_127_521;
fn l2s_c0(v: f64) -> f64 { if v <= LT { v * 12.92 } else { (1.0 + A) * v.powf(1.0 / 2.4) - A } }
fn s2l_c0(v: f64) -> f64 { if v <= 12.92 * LT { v / 12.92 } else { ((v + A) / (1.0 + A)).powf(2.4) } }
fn c0_report() {
    let g = grid();
    let uni = 1usize << 22;
    let bad = |pred: &dyn Fn(&f32) -> bool, s: &[f32]| s.iter().filter(|v| pred(v)).count();
    let enc8 = |v: &f32| ls::linear_to_srgb_u8(*v) as i64 != (l2s_c0(*v as f64) * 255.0).round() as i64;
    println!("C0 linear_to_srgb_u8 uniform\t{}\t{}", uni, bad(&enc8, &g[..uni]));
    println!("C0 linear_to_srgb_u8 dark log\t{}\t{}", g.len() - uni, bad(&enc8, &g[uni..]));
    let mut o16 = vec![0u16; g.len()];
    ls::linear_to_srgb_u16_slice(&g, &mut o16);
    let (mut b, mut w) = (0, 0i64);
    for (&v, &o) in g.iter().zip(&o16) { let e = (l2s_c0(v as f64) * 65535.0).round() as i64; if o as i64 != e { b += 1; w = w.max((o as i64 - e).abs()); } }
    println!("C0 linear_to_srgb_u16_slice\t{}\t{b}\t{w} codes", g.len());
    let mut u8bad = (0, 0i64);
    for c in 0..=255u8 { let u = ulp(ls::srgb_u8_to_linear(c), s2l_c0(c as f64 / 255.0) as f32); if u > 0 { u8bad.0 += 1; } u8bad.1 = u8bad.1.max(u); }
    println!("C0 srgb_u8_to_linear\t256\t{}\t{} ulp", u8bad.0, u8bad.1);
    let mut u16bad = (0, 0i64);
    for c in 0..=65535u16 { let u = ulp(ls::srgb_u16_to_linear(c), s2l_c0(c as f64 / 65535.0) as f32); if u > 0 { u16bad.0 += 1; } u16bad.1 = u16bad.1.max(u); }
    println!("C0 srgb_u16_to_linear\t65536\t{}\t{} ulp", u16bad.0, u16bad.1);
}
fn main() {
    c0_report();
    if std::env::var_os("POLICIES").is_some() { policy_report(); hdr_policy_report(); return; }
    if std::env::var_os("C0_ONLY").is_some() { return; }
    // u8 -> linear
    let (mut bad, mut w) = (0, 0);
    for c in 0..=255u8 { let e = s2l(c as f64 / 255.0) as f32; let g = ls::srgb_u8_to_linear(c); let u = ulp(g, e); if u > 0 { bad += 1; } w = w.max(u); }
    report("ls srgb_u8_to_linear", 256, bad, w, "ulp", String::new());
    // Which code is worst, and is that the constants or the table? WORST8
    let c0 = |v: f64| -> f64 { const T: f64 = 0.039285714285714285; if v <= T { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } };
    let mut wc = (0, 0i64, 0.0f64);
    for c in 0..=255u8 { let g = ls::srgb_u8_to_linear(c); let e = s2l(c as f64 / 255.0); let u = ulp(g, e as f32); if u > wc.1 { wc = (c, u, (g as f64 - e) / e.max(1e-12)); } }
    println!("worst u8 vs IEC\tcode {}\t{} ulp\trelative {:.2e}\tvalue {} iec {} c0-threshold-form {}", wc.0, wc.1, wc.2, ls::srgb_u8_to_linear(wc.0), s2l(wc.0 as f64 / 255.0), c0(wc.0 as f64 / 255.0));
    let mut hist = [0usize; 6];
    for c in 0..=255u8 { let u = ulp(ls::srgb_u8_to_linear(c), s2l(c as f64 / 255.0) as f32); hist[match u { 0 => 0, 1 => 1, 2..=8 => 2, 9..=64 => 3, 65..=1024 => 4, _ => 5 }] += 1; }
    println!("u8 ulp histogram 0,1,2-8,9-64,65-1024,more\t{hist:?}");
    println!("code 11 and 12 linear {} {} iec {} {}", ls::srgb_u8_to_linear(11), ls::srgb_u8_to_linear(12), s2l(11.0 / 255.0), s2l(12.0 / 255.0));
    // Uniform-in-linear only, for the u8 encode table.
    let uni: Vec<f32> = (0..(1u32 << 22)).map(|i| i as f32 / ((1u32 << 22) - 1) as f32).collect();
    let bad = uni.iter().filter(|&&v| ls::linear_to_srgb_u8(v) as i64 != (l2s(v as f64) * 255.0).round() as i64).count();
    println!("linear_to_srgb_u8 uniform only\t{}\t{bad}\t{:.4}%", uni.len(), bad as f64 * 100.0 / uni.len() as f64);
    // u16 -> linear
    let (mut bad, mut w) = (0, 0);
    for c in 0..=65535u16 { let e = s2l(c as f64 / 65535.0) as f32; let g = ls::srgb_u16_to_linear(c); let u = ulp(g, e); if u > 0 { bad += 1; } w = w.max(u); }
    report("ls srgb_u16_to_linear", 65536, bad, w, "ulp", String::new());
    let g = grid();
    // linear -> u8, scalar
    let (mut bad, mut w, mut ex) = (0, 0i64, String::new());
    for &v in &g { let e = (l2s(v as f64) * 255.0).round() as i64; let o = ls::linear_to_srgb_u8(v) as i64; if o != e { bad += 1; if (o - e).abs() > w { w = (o - e).abs(); ex = format!("linear {v} gave {o}, exact {e}"); } } }
    report("ls linear_to_srgb_u8", g.len(), bad, w, "codes", ex);
    // linear -> u8, slice at SIMD width
    let mut out = vec![0u8; g.len()];
    ls::linear_to_srgb_u8_slice(&g, &mut out);
    let (mut bad, mut w, mut disagree) = (0, 0i64, 0);
    for (&v, &o) in g.iter().zip(&out) { let e = (l2s(v as f64) * 255.0).round() as i64; if o as i64 != e { bad += 1; w = w.max((o as i64 - e).abs()); } if o != ls::linear_to_srgb_u8(v) { disagree += 1; } }
    report("ls linear_to_srgb_u8_slice", g.len(), bad, w, "codes", format!("disagrees with scalar on {disagree}"));
    // u8 round trip
    let (mut bad, mut w) = (0, 0i64);
    for c in 0..=255u8 { let o = ls::linear_to_srgb_u8(ls::srgb_u8_to_linear(c)); if o != c { bad += 1; w = w.max((o as i64 - c as i64).abs()); } }
    report("u8 -> linear -> u8 scalar", 256, bad, w, "codes", String::new());
    // linear -> u16 slice
    let mut o16 = vec![0u16; g.len()];
    ls::linear_to_srgb_u16_slice(&g, &mut o16);
    let (mut bad, mut w, mut ex) = (0, 0i64, String::new());
    for (&v, &o) in g.iter().zip(&o16) { let e = (l2s(v as f64) * 65535.0).round() as i64; if o as i64 != e { bad += 1; if (o as i64 - e).abs() > w { w = (o as i64 - e).abs(); ex = format!("linear {v} gave {o}, exact {e}"); } } }
    report("ls linear_to_srgb_u16_slice", g.len(), bad, w, "codes", ex);
    // Row length changes u16 encode?
    let row: Vec<f32> = (0..65536).map(|i| i as f32 / 65535.0).collect();
    let mut whole = vec![0u16; row.len()];
    ls::linear_to_srgb_u16_slice(&row, &mut whole);
    let mut split = vec![0u16; row.len()];
    for (a, b) in row.chunks(7).zip(split.chunks_mut(7)) { ls::linear_to_srgb_u16_slice(a, b); }
    report("u16 encode, row split in 7s", row.len(), whole.iter().zip(&split).filter(|(a, b)| a != b).count(), whole.iter().zip(&split).map(|(a, b)| (*a as i64 - *b as i64).abs()).max().unwrap(), "codes", String::new());
    // u16 round trip through linear
    let lin: Vec<f32> = (0..=65535u16).map(ls::srgb_u16_to_linear).collect();
    let mut back = vec![0u16; 65536];
    ls::linear_to_srgb_u16_slice(&lin, &mut back);
    let (mut bad, mut w) = (0, 0i64);
    for (c, &o) in back.iter().enumerate() { if o as usize != c { bad += 1; w = w.max((o as i64 - c as i64).abs()); } }
    report("u16 -> linear -> u16", 65536, bad, w, "codes", String::new());
    // N-bit codes, canonical in u16, through linear and back.
    for n in [8u32, 10, 12, 14] {
        let m = (1u64 << n) - 1;
        let (mut bad, mut w) = (0, 0i64);
        for q in 0..=m {
            let c = ((q * 65535 + m / 2) / m) as u16;
            let o = back[c as usize] as u64;
            let r = (o * m + 32767) / 65535;
            if r != q { bad += 1; w = w.max((r as i64 - q as i64).abs()); }
        }
        report(&format!("{n}-bit code -> u16 -> linear -> u16 -> {n}-bit"), (m + 1) as usize, bad, w, "codes", String::new());
    }
    // sRGB f32 <-> linear f32 slice, in 16-bit codes of the encoded value.
    let mut f = g.clone();
    ls::linear_to_srgb_slice(&mut f);
    let (mut w, mut bad) = (0f64, 0);
    for (&v, &o) in g.iter().zip(&f) { let d = (o as f64 - l2s(v as f64)).abs() * 65535.0; if d > 0.5 { bad += 1; } w = w.max(d); }
    println!("ls linear_to_srgb_slice f32\t{}\t{bad}\t\t{w:.3} u16 codes\tbad means over half a 16-bit code", g.len());
    let mut f: Vec<f32> = (0..1u32 << 22).map(|i| i as f32 / ((1u32 << 22) - 1) as f32).collect();
    let src = f.clone();
    ls::srgb_to_linear_slice(&mut f);
    let (mut w, mut bad) = (0f64, 0);
    for (&v, &o) in src.iter().zip(&f) { let e = s2l(v as f64); let d = (o as f64 - e).abs() / e.max(1e-9); if (o as f64 - e).abs() * 65535.0 > 0.5 { bad += 1; } w = w.max(d); }
    println!("ls srgb_to_linear_slice f32\t{}\t{bad}\t\t{w:.2e} relative\tbad means over half a 16-bit code", src.len());
    // garb
    let all16: Vec<u8> = (0..=65535u16).flat_map(|v| v.to_ne_bytes()).collect();
    for (label, width) in [("row 65536", 65536usize), ("row 1", 1)] {
        let mut o = vec![0u8; 65536];
        for (a, b) in all16.chunks(width * 2).zip(o.chunks_mut(width)) { garb::bytes::convert_u16_to_u8(a, b).unwrap(); }
        let (mut bad, mut w, mut ex) = (0, 0i64, String::new());
        for (c, &g8) in o.iter().enumerate() { let e = ((c + 128) / 257) as i64; if g8 as i64 != e { bad += 1; if (g8 as i64 - e).abs() > w { w = (g8 as i64 - e).abs(); ex = format!("{c} gave {g8}, nearest {e}"); } } }
        report(&format!("garb convert_u16_to_u8 {label}"), 65536, bad, w, "codes", ex);
    }
    let all8: Vec<u8> = (0..=255u8).collect();
    let mut o = vec![0u8; 512];
    garb::bytes::convert_u8_to_u16(&all8, &mut o).unwrap();
    let bad = o.chunks(2).enumerate().filter(|(i, b)| u16::from_ne_bytes([b[0], b[1]]) as usize != i * 257).count();
    report("garb convert_u8_to_u16", 256, bad, 0, "codes", String::new());
    let mut o = vec![0u8; 256 * 4];
    garb::bytes::convert_u8_to_f32(&all8, &mut o).unwrap();
    let w = o.chunks(4).enumerate().map(|(i, b)| ulp(f32::from_ne_bytes(b.try_into().unwrap()), (i as f64 / 255.0) as f32)).max().unwrap();
    let bad = o.chunks(4).enumerate().filter(|(i,b)| ulp(f32::from_ne_bytes((*b).try_into().unwrap()), (*i as f64 / 255.0) as f32) != 0).count();
    report("garb convert_u8_to_f32", 256, bad, w, "ulp", String::new());
    for (label, width) in [("row 65536", 65536usize), ("row 1", 1)] {
        let mut o = vec![0u8; 65536 * 4];
        for (a, b) in all16.chunks(width * 2).zip(o.chunks_mut(width * 4)) { garb::bytes::convert_u16_to_f32(a, b).unwrap(); }
        let mut bad = 0; let mut w = 0;
        for (i, b) in o.chunks(4).enumerate() { let u = ulp(f32::from_ne_bytes(b.try_into().unwrap()), (i as f64 / 65535.0) as f32); if u > 0 { bad += 1; } w = w.max(u); }
        report(&format!("garb convert_u16_to_f32 {label}"), 65536, bad, w, "ulp", String::new());
    }
    // f32 -> integer: on and between the grids.
    let vals: Vec<f32> = (0..(1u32 << 22)).map(|i| i as f32 / ((1u32 << 22) - 1) as f32).collect();
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let mut o = vec![0u8; vals.len()];
    garb::bytes::convert_f32_to_u8(&bytes, &mut o).unwrap();
    let (mut bad, mut w) = (0, 0i64);
    for (&v, &g8) in vals.iter().zip(&o) { let e = (v as f64 * 255.0).round() as i64; if g8 as i64 != e { bad += 1; w = w.max((g8 as i64 - e).abs()); } }
    report("garb convert_f32_to_u8", vals.len(), bad, w, "codes", String::new());
    let mut o = vec![0u8; vals.len() * 2];
    garb::bytes::convert_f32_to_u16(&bytes, &mut o).unwrap();
    let (mut bad, mut w) = (0, 0i64);
    for (&v, b) in vals.iter().zip(o.chunks(2)) { let e = (v as f64 * 65535.0).round() as i64; let g = u16::from_ne_bytes([b[0], b[1]]) as i64; if g != e { bad += 1; w = w.max((g - e).abs()); } }
    report("garb convert_f32_to_u16", vals.len(), bad, w, "codes", String::new());
}

fn policy_report() {
    let input = grid();
    for (name, fun, reference) in [
        ("default s2l scalar", ls::srgb_to_linear as fn(f32)->f32, s2l_c0 as fn(f64)->f64),
        ("default l2s scalar", ls::linear_to_srgb, l2s_c0),
        ("precise s2l scalar", linear_srgb::precise::srgb_to_linear, s2l_c0),
        ("precise l2s scalar", linear_srgb::precise::linear_to_srgb, l2s_c0),
    ] {
        let (mut bad, mut maximum)=(0,0);
        for &x in &input { let error=ulp(fun(x),reference(x as f64) as f32);bad+=usize::from(error!=0);maximum=maximum.max(error); }
        report(name,input.len(),bad,maximum,"ULP",String::new());
    }
    for (name,fun,reference) in [
        ("default s2l slice",ls::srgb_to_linear_slice as fn(&mut[f32]),s2l_c0 as fn(f64)->f64),
        ("default l2s slice",ls::linear_to_srgb_slice,l2s_c0),
        ("extended s2l slice unit range",ls::srgb_to_linear_extended_slice,s2l_c0),
        ("extended l2s slice unit range",ls::linear_to_srgb_extended_slice,l2s_c0),
    ] {
        let mut output=input.clone();fun(&mut output);
        let (mut bad, mut maximum)=(0,0);
        for (&x,&y) in input.iter().zip(&output) {let error=ulp(y,reference(x as f64) as f32);bad+=usize::from(error!=0);maximum=maximum.max(error);}
        report(name,input.len(),bad,maximum,"ULP",String::new());
    }
    let converter=linear_srgb::lut::SrgbConverter::new();
    let (mut bad,mut maximum)=(0,0);
    let mut max_absolute=0.0f64;
    for &x in &input {let expected=l2s_c0(x as f64);let y=converter.linear_to_srgb(x);let error=ulp(y,expected as f32);bad+=usize::from(error!=0);maximum=maximum.max(error);max_absolute=max_absolute.max((y as f64-expected).abs());}
    report("converter interpolated l2s",input.len(),bad,maximum,"ULP",format!("max_absolute={max_absolute:e}"));
    let (mut bad,mut maximum)=(0,0);
    for &x in &input {let y=ls::linear_to_srgb_u16_fast(x) as i64;let e=(l2s_c0(x as f64)*65535.0).round() as i64;bad+=usize::from(y!=e);maximum=maximum.max((y-e).abs());}
    report("default u16 fast sqrt table",input.len(),bad,maximum,"codes",String::new());
}

fn pq_to_linear_f64(v: f64) -> f64 {
    if v <= 0.0 {
        return 0.0;
    }
    let m1: f64 = 0.1593017578125;
    let m2: f64 = 78.84375;
    let c1: f64 = 0.8359375;
    let c2: f64 = 18.8515625;
    let c3: f64 = 18.6875;
    let vp = v.powf(1.0 / m2);
    let num = (vp - c1).max(0.0);
    let den = c2 - c3 * vp;
    if den <= 0.0 {
        return 1.0;
    }
    (num / den).powf(1.0 / m1)
}

fn pq_from_linear_f64(v: f64) -> f64 {
    if v <= 0.0 {
        return 0.0;
    }
    let m1: f64 = 0.1593017578125;
    let m2: f64 = 78.84375;
    let c1: f64 = 0.8359375;
    let c2: f64 = 18.8515625;
    let c3: f64 = 18.6875;
    let vp = v.powf(m1);
    let num = c1 + c2 * vp;
    let den = 1.0 + c3 * vp;
    (num / den).powf(m2)
}

fn bt709_to_linear_f64(v: f64) -> f64 {
    let beta: f64 = 0.018053968510807;
    if v < 4.5 * beta {
        v / 4.5
    } else {
        let alpha: f64 = 0.09929682680944;
        ((v + alpha) / (1.0 + alpha)).powf(1.0 / 0.45)
    }
}

fn bt709_from_linear_f64(v: f64) -> f64 {
    let beta: f64 = 0.018053968510807;
    let alpha: f64 = 0.09929682680944;
    if v < beta {
        4.5 * v
    } else {
        (1.0 + alpha) * v.powf(0.45) - alpha
    }
}

fn hlg_to_linear_f64(v: f64) -> f64 {
    let a: f64 = 0.17883277;
    let b: f64 = 0.28466892;
    let c: f64 = 0.55991073;
    if v <= 0.0 {
        0.0
    } else if v <= 0.5 {
        (v * v) / 3.0
    } else {
        (((v - c) / a).exp() + b) / 12.0
    }
}

fn hlg_from_linear_f64(v: f64) -> f64 {
    let a: f64 = 0.17883277;
    let b: f64 = 0.28466892;
    let c: f64 = 0.55991073;
    if v <= 0.0 {
        0.0
    } else if v <= 1.0 / 12.0 {
        (3.0 * v).sqrt()
    } else {
        a * (12.0 * v - b).ln() + c
    }
}

fn hdr_policy_report() {
    let input=grid();
    for (name,scalar,slice,reference) in [
        ("BT709 decode",ls::bt709_to_linear as fn(f32)->f32,ls::bt709_to_linear_slice as fn(&mut[f32]),bt709_to_linear_f64 as fn(f64)->f64),
        ("BT709 encode",ls::linear_to_bt709,ls::linear_to_bt709_slice,bt709_from_linear_f64),
        ("HLG decode",ls::hlg_to_linear,ls::hlg_to_linear_slice,hlg_to_linear_f64),
        ("HLG encode",ls::linear_to_hlg,ls::linear_to_hlg_slice,hlg_from_linear_f64),
        ("PQ decode",ls::pq_to_linear,ls::pq_to_linear_slice,pq_to_linear_f64),
        ("PQ encode",ls::linear_to_pq,ls::linear_to_pq_slice,pq_from_linear_f64),
    ] {
        let mut output=input.clone();slice(&mut output);
        for (kind, values) in [("scalar",input.iter().map(|&x| scalar(x)).collect::<Vec<_>>()),("slice",output)] {
            let (mut bad,mut maximum)=(0,0);
            let (mut rel,mut code_error)=(0.0f64,0.0f64);
            for (&x,&y) in input.iter().zip(&values) {
                let expected=reference(x as f64);let error=ulp(y,expected as f32);bad+=usize::from(error!=0);maximum=maximum.max(error);
                if expected!=0.0 {rel=rel.max(((y as f64-expected)/expected).abs());}
                code_error=code_error.max((y as f64-expected).abs()*65535.0);
            }
            report(&format!("{name} {kind}"),input.len(),bad,maximum,"ULP",format!("max_relative={rel:e} max_u16_code_error={code_error:e}"));
        }
    }
}
