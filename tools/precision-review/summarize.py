#!/usr/bin/env python3
"""Compare native captures in sample units; preserve non-finite distinctions."""
import array
import csv
import math
import struct
import sys
from pathlib import Path

before, after, output = map(Path, sys.argv[1:4])
def kind(name):
    if "compare_and_avg" in name or "ulp_distance_f64" in name: return "Q", "u64 diagnostic"
    if "ulp_distance_f32" in name: return "I", "u32 diagnostic"
    if name.startswith("garb_"):
        if "deinterleave" in name or "f32" in name and "_to_u" not in name: return "I", "f32 ULP"
        if "_to_u16" in name or "rgba1010102_to_rgba16" in name: return "H", "u16 code"
        if "_to_rgb565" in name or "_to_rgba4444" in name: return "H", "packed u16 word"
        if "rgba16_to_rgba1010102" in name: return "I", "packed u32 word"
        return "B", "u8 code"
    if "_f64" in name: return "Q", "f64 ULP"
    if "linear_to_srgb_u8" in name or "uchar_clamp" in name or name.endswith("linear_to_srgb_lut") or name.endswith("batch_linear_to_srgb"): return "B", "u8 code"
    if "linear_to_srgb_u16" in name or "interp_linear_u16" in name or "generate_encode_lut" in name: return "H", "u16 code"
    return "I", "f32 ULP"

rows=[]
for path in sorted(after.glob("*.bin")):
    if path.name.startswith("__"): continue
    code,unit=kind(path.stem)
    original=before/path.name
    if not original.exists():
        rows.append([path.stem,unit,path.stat().st_size//array.array(code).itemsize,"new", "", "", ""])
        continue
    old=original.read_bytes();new=path.read_bytes()
    assert len(old)==len(new),path.name
    a=array.array(code);a.frombytes(old)
    if old==new:
        rows.append([path.stem,unit,len(a),0,0,0,""]);continue
    b=array.array(code);b.frombytes(new)
    bad=nonfinite=maximum=0;first=None
    bits=a.itemsize*8;sign=1<<(bits-1);mask=(1<<bits)-1
    for i,(x,y) in enumerate(zip(a,b)):
        if x==y:continue
        bad+=1
        if first is None:first=i
        if "ULP" in unit:
            float_code="f" if bits==32 else "d"
            xx=struct.unpack("<"+float_code,x.to_bytes(a.itemsize,"little"))[0]
            yy=struct.unpack("<"+float_code,y.to_bytes(a.itemsize,"little"))[0]
            if not (math.isfinite(xx) and math.isfinite(yy)):
                nonfinite+=1;continue
            x=(~x)&mask if x&sign else x|sign
            y=(~y)&mask if y&sign else y|sign
        maximum=max(maximum,abs(x-y))
    rows.append([path.stem,unit,len(a),bad,maximum,nonfinite,first])
output.parent.mkdir(parents=True,exist_ok=True)
with output.open("w") as f:
    writer=csv.writer(f,delimiter="\t");writer.writerow(["function","unit","samples","changed","max_finite_delta","nonfinite_changed","first_index"]);writer.writerows(rows)
md=output.with_suffix(".md")
md.write_text("| Function | Unit | Samples | Changed | Max finite change | Nonfinite changed | First index |\n|---|---|---:|---:|---:|---:|---:|\n"+"\n".join("| "+" | ".join(str(x) for x in row)+" |" for row in rows)+"\n")
print(f"{len(rows)} functions; {sum(isinstance(r[3],int) and r[3]>0 for r in rows)} differ; {output}")
