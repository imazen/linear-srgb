#!/usr/bin/env python3
"""Generate a reproducible public-surface differential driver from API snapshots.

Writes a standalone Cargo package under the caller-supplied output directory.
Unsupported signatures are reported, never counted as checked.
"""
import json
import re
import sys
from extra import adapter
from pathlib import Path

root = Path(__file__).resolve().parents[2]
garb = Path(sys.argv[2]).resolve()
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
(out / "src").mkdir(exist_ok=True)
released = "--released" in sys.argv
dep_ls = 'version = "=0.6.12"' if released else f'path = {json.dumps(str(root))}'
arch_version = "0.9.30" if released else "0.9.31-beta.2"
dep_gb = 'version = "=0.2.8"' if released else f'path = {json.dumps(str(garb))}'
(out / "Cargo.toml").write_text(f'''[package]
name = "precision-review"
version = "0.0.0"
edition = "2024"
publish = false
[workspace]
[[bin]]
name = "oracle"
path = {json.dumps(str(root / "tools/precision-review/oracle.rs"))}
[dependencies]
linear-srgb = {{ {dep_ls}, features = ["transfer", "iec", "alt"] }}
garb = {{ {dep_gb}, features = ["experimental", "rgb", "imgref"] }}
bytemuck = "1.25.2"
rgb = {{ version = "0.8.53", features = ["bytemuck"] }}
imgref = "1.12.3"
archmage = {{ version = "={arch_version}", features = ["testable_dispatch", "avx512"] }}
''')

def api(repo):
    items = set()
    for f in (repo / "docs/public-api").glob("*.txt"):
        items.update(re.sub(r" \[also:.*", "", x) for x in f.read_text().splitlines() if x.startswith("pub fn "))
    return sorted(x for x in items if not x.startswith("pub fn alt::accuracy::UlpStats::"))

def bpp(s):
    if "f32" in s: return 16
    if "1010102" in s: return 4
    if "16" in s: return 8
    if "565" in s or "4444" in s: return 2
    if "gray_alpha" in s: return 2
    if s == "gray": return 1
    if len(s) == 3: return 3
    return 4

base = r'''
#![allow(deprecated)]
use linear_srgb as ls;
use archmage::SimdToken;
use garb::bytes as gb;
use std::path::PathBuf;
thread_local! { static TIER:std::cell::Cell<&'static str>=const{std::cell::Cell::new("hooks")}; static NATIVE:std::cell::RefCell<std::collections::HashMap<String,Vec<u8>>>=std::cell::RefCell::new(std::collections::HashMap::new()); }
fn samples() -> Vec<f32> {
    let mut x: Vec<f32> = (0..=65535).map(|i| i as f32 / 65535.0).collect();
    x.extend((0..65536).map(|i| 10f64.powf(-12.0+12.0*i as f64/65535.0) as f32));
    for v in [0.0f32, 1.0, 0.04045, 0.03929337, 0.0030412826, 0.0031308, 0.02, 0.25, 1e-7, 1e-4, 0.081242, 0.018053968, 0.5, 1.0/12.0, 1.0/1024.0] {
        x.extend((0..=2048u32).filter_map(|i| v.to_bits().checked_add(i)?.checked_sub(1024).map(f32::from_bits)));
    }
    x.extend([-1.0, -0.0, 0.0, 1.0, 1.5, f32::NEG_INFINITY, f32::INFINITY, f32::NAN]);
    while x.len() % 16 != 0 { x.push(0.75); }
    x
}
fn record<T: bytemuck::Pod>(name: &str, data: &[T]) {
    let dir=PathBuf::from(std::env::var_os("AUDIT_BASE").expect("AUDIT_BASE")).join(TIER.with(|t|t.get()));
    std::fs::create_dir_all(&dir).unwrap();
    let path=dir.join(name.replace("::", "_").replace("/", "_")+".bin");
    let bytes=bytemuck::cast_slice(data);
    if TIER.with(|t|t.get())=="native" {
        if let Some(capture)=std::env::var_os("AUDIT_CAPTURE") {
            let path=PathBuf::from(capture);std::fs::create_dir_all(&path).unwrap();
            std::fs::write(path.join(name.replace("::","_").replace("/","_")+".bin"),bytes).unwrap();
        }
    }
    NATIVE.with(|m| {let mut m=m.borrow_mut(); if let Some(old)=m.get(name) {let unit=core::mem::size_of::<T>();println!("tier-parity\t{name}\t{}\t{}",data.len(),old.chunks(unit).zip(bytes.chunks(unit)).filter(|(a,b)|a!=b).count());} else {m.insert(name.into(),bytes.to_vec());}});
    if std::env::var_os("AUDIT_SAVE").is_some() { if path.exists() {assert_eq!(std::fs::read(&path).unwrap(),bytes,"baseline differs: {name}");} else {std::fs::write(path, bytes).unwrap();} }
    else {
        let old=match std::fs::read(path) {Ok(v)=>v,Err(e) if e.kind()==std::io::ErrorKind::NotFound=>{println!("new\t{name}\t{}",data.len());return;},Err(e)=>panic!("{e}")}; assert_eq!(old.len(),bytes.len(),"{name}");
        let unit=core::mem::size_of::<T>();
        let mut bad=0usize; let mut max=0u64; let mut first=usize::MAX;
        for (i,(a,b)) in old.chunks(unit).zip(bytes.chunks(unit)).enumerate() {
            if a!=b { bad+=1; first=first.min(i); let mut aa=[0u8;8];let mut bb=[0u8;8];aa[..unit].copy_from_slice(a);bb[..unit].copy_from_slice(b);max=max.max(u64::from_ne_bytes(aa).abs_diff(u64::from_ne_bytes(bb))); }
        }
        println!("version\t{name}\t{}\t{bad}\t{max}\t{first}",data.len());
    }
}
fn parity<T: PartialEq>(name:&str, a:&[T], b:&[T]) { println!("split\t{name}\t{}\t{}",a.len(),a.iter().zip(b).filter(|(x,y)|x!=y).count()); }
fn bytes_input(n:usize, format:&str)->Vec<u8> {
    if format=="f32" || format.contains("f32") {let x=samples();(0..n/4).flat_map(|i|x[i%x.len()].to_ne_bytes()).collect()}
    else {(0..n).map(|i| ((i/2) >> (8*(i%2))) as u8).collect()}
}
fn main() {
 let x=samples(); let u8s:Vec<u8>=(0..=255).collect(); let u16s:Vec<u16>=(0..=65535).collect();
'''
lines = []
missing = []
token_lines = []
hook_defs = []
for item in api(root):
    m = re.match(r"pub fn ([^(]+)\((.*)\)(?: -> (.*))?$", item)
    if not m:
        missing.append(item); continue
    name, args, ret = m.groups()
    if released and name in {"default::gamma_to_linear_rgba_slice", "default::linear_to_gamma_rgba_slice", "default::linear_to_srgb_extended_rgba_slice", "default::srgb_to_linear_extended_rgba_slice"}:
        continue
    call = "ls::" + name
    if args in ("f32", "f32, f32", "f64", "f64, f64", "u8", "u16") and ret in ("f32", "f64", "u8", "u16", "u32", "u64"):
        typ = args.split(",")[0]
        src = {"u8":"u8s", "u16":"u16s"}.get(typ,"x")
        extra = ", 2.2" if "," in args else ""
        lines.append(f'{{let y:Vec<{ret}>={src}.iter().map(|&v|{call}(v as {typ}{extra})).collect();record("{name}",&y);}}')
    elif args in ("&mut [f32]", "&mut [f32], f32"):
        extra = ", 2.2" if args.endswith(", f32") else ""
        width = 28 if "rgba" in name else 7
        lines.append(f'{{let mut y=x.to_vec();{call}(&mut y{extra}); record("{name}",&y);let mut z=x.to_vec();for c in z.chunks_mut({width}) {{{call}(c{extra});}} parity("{name}",&y.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),&z.iter().map(|v|v.to_bits()).collect::<Vec<_>>());}}')
    elif re.fullmatch(r"&\[(f32|u8|u16)\], &mut \[(f32|u8|u16)\]",args):
        it,ot=re.findall(r"\[(\w+)\]",args)
        src={"f32":"x","u8":"u8s","u16":"u16s"}[it]
        width=28 if "rgba" in name else 7
        lines.append(f'{{let mut y=vec![0 as {ot};{src}.len()];{call}(&{src},&mut y);record("{name}",&y);let mut z=vec![0 as {ot};{src}.len()];for(a,b)in {src}.chunks({width}).zip(z.chunks_mut({width})){{{call}(a,b);}}parity("{name}",bytemuck::cast_slice::<{ot},u8>(&y),bytemuck::cast_slice::<{ot},u8>(&z));}}')
    elif name.startswith("tokens::"):
        # Direct x86 hooks, including array and slice forms.
        tm=re.match(r"archmage::tokens::(?:generated::x86::)?(X64V[34]Token|ScalarToken), (.*)",args)
        if tm:
            token, rest=tm.groups()
            ty="u8" if "[u8;" in rest else "u16" if "[u16;" in rest else "f32"
            src={"f32":"x","u8":"u8s","u16":"u16s"}[ty]
            if rest.startswith("&mut [f32]"):
                body=f'let mut y=x.to_vec();{call}(t,&mut y'+(',2.2' if rest.endswith(', f32') else '')+f');record("{name}",&y);'
            elif rest.startswith("&["):
                it,ot=re.findall(r"\[(\w+)\]",rest)
                src={"f32":"x","u8":"u8s","u16":"u16s"}[it]
                body=f'let mut y=vec![0 as {ot};{src}.len()];{call}(t,{src},&mut y);record("{name}",&y);'
            else:
                ar=re.search(r"\[(\w+); (\d+)\]",rest)
                if not ar: missing.append(item);continue
                width=int(ar[2]);ot= re.search(r"\[(\w+);",ret or "")
                if not ot: missing.append(item);continue
                extra=",2.2" if rest.endswith(", f32") else ""
                body=f'let y:Vec<{ot[1]}>={src}.chunks_exact({width}).flat_map(|v|{call}(t,v.try_into().unwrap(){extra})).collect();record("{name}",&y);'
            idx=len(hook_defs)
            attrs='#[cfg(target_arch="x86_64")]'+(' #[archmage::arcane]' if token!="ScalarToken" else "")
            hook_defs.append(f'{attrs} #[allow(unused_variables)] fn hook{idx}(t:archmage::{token}, x:&[f32],u8s:&[u8],u16s:&[u16]) {{{body}}}')
            token_lines.append(f'#[cfg(target_arch="x86_64")] if let Some(t)=archmage::{token}::summon() {{hook{idx}(t,&x,&u8s,&u16s);}}')
        else: missing.append(item)
    else:
        missing.append(item)

# garb byte entry points include aliases and strided forms. Typed APIs are
# separately identified in the unsupported roster, rather than silently counted.
for item in api(garb):
    m=re.match(r"pub fn bytes::(\w+)\((.*)\) ->",item)
    if not m: missing.append(item);continue
    name,args=m.groups()
    if "&[u16]" in args or "&mut [u16]" in args: missing.append(item);continue
    bare=name.replace("_strided","").replace("_copy","")
    inplace="&[u8]" not in args
    if bare.startswith("convert_"):
        it,ot=bare[8:].split("_to_"); ib={"u8":1,"u16":2,"f32":4}[it];ob={"u8":1,"u16":2,"f32":4}[ot]
    elif "_to_" in bare:
        it,ot=bare.split("_to_",1);ib=bpp(it);ob=bpp(ot.split("_bt")[0].replace("_identity",""))
    else:
        it="f32" if "f32" in bare else "rgba";ib=ob=bpp(it)
    n=65536
    label="garb::bytes::"+name
    if inplace:
        call=f'gb::{name}(&mut y'+(f', 256, 256, {256*ib+ib}' if "strided" in name else '')+').unwrap();'
        length=(256*ib+ib)*256 if "strided" in name else n*ib
        lines.append(f'{{let mut y=bytes_input({length},"{it}");{call}record("{label}",&y);}}')
    else:
        strided="strided" in name
        sl=(256*ib+ib)*256 if strided else n*ib
        dl=(256*ob+ob)*256 if strided else n*ob
        call=f'gb::{name}(&s,&mut y'+(f', 256, 256, {256*ib+ib}, {256*ob+ob}' if strided else '')+').unwrap();'
        lines.append(f'{{let s=bytes_input({sl},"{it}");let mut y=vec![0u8;{dl}];{call}record("{label}",&y);}}')

remaining=[]
for item in missing:
    crate="ls" if item in api(root) else "garb"
    code=adapter(item,crate)
    if code is None: remaining.append(item)
    elif code: lines.append(code)
missing=remaining
for module in ["default", "precise"]:
    for fn in ["srgb_to_linear", "linear_to_srgb"]:
        lines.append(f'{{let y:Vec<f32>=x.iter().map(|&v|ls::{module}::{fn}(v)).collect();record("{module}::{fn}",&y);}}')
for fn in ["fastpow", "srgb_to_linear", "linear_to_srgb_raw", "linear_to_srgb", "uchar_clamp_ff", "linear_to_srgb_u8_fastpow", "linear_to_srgb_lut"]:
    typ="u8" if fn in ["uchar_clamp_ff", "linear_to_srgb_u8_fastpow", "linear_to_srgb_lut"] else "f32"
    extra=",2.2" if fn=="fastpow" else ""
    lines.append(f'{{let y:Vec<{typ}>=x.iter().map(|&v|ls::alt::imageflow::{fn}(v{extra})).collect();record("alt::imageflow::{fn}",&y);}}')

# The scalar/slice workload runs at each actual dispatch tier. Set the parent
# before the child, since archmage enabling a parent re-enables descendants.
tier = r'''
 for tier in ["native", "v3", "scalar"] {
  #[cfg(target_arch="aarch64")] if tier=="v3" {continue;}
  TIER.with(|t|t.set(tier));
  #[cfg(target_arch="x86_64")] { archmage::X64V3Token::dangerously_disable_token_process_wide(tier=="scalar").unwrap(); archmage::X64V4Token::dangerously_disable_token_process_wide(tier!="native").unwrap(); }
  #[cfg(target_arch="aarch64")] { archmage::NeonToken::dangerously_disable_token_process_wide(tier=="scalar").unwrap(); }
  println!("tier\t{tier}");
'''
neon_defs=[]
neon_calls=[]
for definition, invocation in zip(hook_defs,token_lines):
    if "tokens::x4::" in definition and ("X64V3Token" in definition or "ScalarToken" in definition):
        neon_defs.append(definition.replace('target_arch="x86_64"','target_arch="aarch64"').replace("X64V3Token","NeonToken").replace("_v3", "_neon"))
        neon_calls.append(invocation.replace('target_arch="x86_64"','target_arch="aarch64"').replace("X64V3Token","NeonToken"))
hook_defs+=neon_defs
token_lines+=neon_calls
base=base.replace("fn main() {", "\n".join(hook_defs)+"\nfn main() {")
base += "\n".join(token_lines) + tier + "\n".join(lines) + "\n}\n}\n"
base=re.sub(r"let mut s:(Vec<[^;]+);(?=let mut y:)",lambda m:"let s:"+m.group(1)+";",base)
(out/"src/main.rs").write_text(base)
(out/"unsupported.txt").write_text("\n".join(missing)+"\n")
print(f"generated {len(lines)} dispatch/scalar calls and {len(token_lines)} direct hooks; {len(missing)} unsupported signatures")
