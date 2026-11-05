#!/usr/bin/env python3
"""Read-only census of transfer, color and depth implementations.

Output is deliberately kept outside this repository: a workspace inventory can
include private projects. Classifications are conservative recommendations, not
proof of numerical interchangeability. Each row retains a source excerpt.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

workspace, output = map(Path, sys.argv[1:3])
repos = {}
for parent in [workspace, workspace / "zen"]:
    for path in parent.iterdir():
        if not path.is_dir() or path.is_symlink() or "--" in path.name:
            continue
        if path.name in {"retired", "upstream", "_dbg"}:
            continue
        if (path / ".git").exists():
            repos[str(path.resolve())] = path

patterns = {
    "sRGB": r"0\.04045|0\.055010718|\b(?:fn|function)\s+\w*srgb\w*",
    "BT.709": r"\b(?:fn|function)\s+\w*(?:bt709|rec709)\w*|0\.0992968",
    "PQ": r"\b(?:fn|function)\s+\w*pq\w*|2610\.0\s*/\s*16384",
    "HLG": r"\b(?:fn|function)\s+\w*hlg\w*|0\.17883277",
    "gamma": r"\b(?:fn|function)\s+\w*gamma\w*",
    "cbrt/pow/exp/log": r"\bfn\s+(?:[a-z0-9]+_)*(?:cbrtf?|powf?|pow64|exp|exp2f?|expf|log|log2f?|log10f?|logf|lnf?)(?:_[a-z0-9]+)*\s*(?:<[^>]*>)?\s*\(",
    "matrices/luma": r"0\.2126|0\.2627|0\.412221|0\.412456|\b(?:fn|function)\s+\w*(?:oklab|xyz_to|to_xyz)\w*",
    "depth": r"\b(?:fn|function)\s+\w*(?:u16_to_u8|f32_to_u(?:8|16)|u(?:8|16)_to_f32|scale_bit_depth|expand_\d+)\w*|65535\.0",
}
compiled = {k: re.compile(v, re.I) for k, v in patterns.items()}
any_pattern=re.compile("|".join(patterns.values()),re.I)
rows = []
for repo in repos.values():
    proc = subprocess.run(["rg", "--files", "-g", "*.rs", "-g", "*.wgsl", "-g", "!target/**", "-g", "!**/target/**"], cwd=repo, text=True, capture_output=True)
    origin=subprocess.run(["git","remote","get-url","origin"],cwd=repo,text=True,capture_output=True).stdout.strip()
    if not re.search(r"github.com[:/](?:imazen|lilith|imazen-private)/",origin):
        continue
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    for rel in proc.stdout.splitlines():
        parts = Path(rel).parts
        if any(p in {"target", "retired", "upstream", "reference", "references", "vendor", "vendored", "third_party", "node_modules", ".jj"} or "--" in p for p in parts):
            continue
        if any(p in {"tests", "benches", "examples", "testdata"} for p in parts) or "test" in Path(rel).stem:
            continue
        path = repo / rel
        if path.stat().st_size>600_000: continue
        text = path.read_text(errors="replace").split("#[cfg(test)]",1)[0]
        if not any_pattern.search(text): continue
        hits = [(i, line.strip(), [k for k,p in compiled.items() if p.search(line)]) for i,line in enumerate(text.splitlines(),1)]
        hits = [h for h in hits if h[2]]
        if not hits:
            continue
        kinds = sorted({k for _,_,ks in hits for k in ks})
        reference = any(s in str(path) for s in ["/zenmetrics-math/src/fdlibm", "/zenmetrics-math/src/glibc", "/butteraugli/", "/fast-ssim2/", "/zenjxl-decoder/", "/jxl-encoder/", "/zenjpeg/zenjpeg/src/color/xyb", "/zensim/", "/zensimx/"])
        owner = repo.name in {"linear-srgb", "garb", "archmage"}
        metadata = not any(re.search(r"\bfn\b|\bfunction\b|\bconst\b|\bstatic\b", h[1]) for h in hits) and all(k in {"matrices/luma", "depth"} for k in kinds)
        if metadata:
            cls,rec="not an implementation", "Metadata, caller or coefficient hit; do not count as a second kernel."
        elif reference:
            cls,rec="reference or frozen numerical contract", "Keep its constants, evaluation order and oracle. Sharing requires a separately verified exact policy; no substitution with C0 or approximate math."
        elif owner:
            cls,rec="shared implementation owner", "Keep curve/depth primitives here; do not move generic transcendentals into a transfer crate."
        elif "matrices/luma" in kinds and len(kinds)==1:
            cls,rec="variant / caller policy", "Keep primaries, adaptation and signal-domain luma with the color pipeline; matrix rounding is observable."
        elif "/ultrahdr/" in str(path):
            cls,rec="reference or frozen numerical contract", "Retain the libultrahdr oracle and display-light semantics; normalized transfer helpers alone are not an equivalent replacement."
        else:
            cls,rec="private copy candidate", "Potential shared caller after constants, clamping, rounding and extended-range contract match; no automatic replacement."
        # Snippets are evidence for the classification; no source is modified.
        rows.append(dict(file=str(path), revision=sha, what=", ".join(kinds), classification=cls, recommendation=rec, evidence=[dict(line=i,text=t) for i,t,_ in hits[:8]]))
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(rows,indent=2)+"\n")
table=output.with_suffix(".md")
table.write_text("# Transfer, math and depth inventory\n\nRead-only source census; classifications are recommendations, not parity measurements. Revisions are Git HEAD observations; colocated jj working trees may be newer. Test, benchmark, retired, upstream and vendor trees are excluded. Source excerpts are in the adjacent JSON.\n\n| File | What | Class | Recommendation |\n|---|---|---|---|\n"+"\n".join(f"| `{r['file']}` | {r['what']} | {r['classification']} | {r['recommendation']} |" for r in rows)+"\n")
print(f"{len(repos)} repositories, {len(rows)} candidate files; {output}; {table}")
