#!/usr/bin/env python3
"""Prepare an isolated consumer of exact archived Archmage and current math source.

Preparation only: run Cargo through run-heavy, specifying engine/target explicitly.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("archmage", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
source = args.archmage.resolve()
output = args.output.resolve()
root = Path(__file__).resolve().parents[2]
for name in ["Cargo.toml", "magemath/Cargo.toml", "magetypes/Cargo.toml"]:
    if not (source / name).is_file():
        parser.error(f"missing archived source: {source / name}")
output.mkdir(parents=True, exist_ok=False)
dependencies = {
    "archmage": source,
    "magetypes": source / "magetypes",
    "magemath": source / "magemath",
}
manifest = '[package]\nname = "linear-srgb-magemath-review"\nversion = "0.0.0"\nedition = "2024"\n\n[workspace]\n\n[features]\nprecise-portable = []\n\n[dependencies]\n'
for name, path in dependencies.items():
    features = ["std", "w512"] if name != "archmage" else ["std"]
    manifest += f'{name} = {{ path = {json.dumps(str(path))}, default-features = false, features = {json.dumps(features)} }}\n'
manifest += 'num-traits = "0.2.19"\n\n[[bin]]\nname = "probe"\npath = "main.rs"\n'
(output / "Cargo.toml").write_text(manifest)
legacy = root / "src/tf/fast_math.rs"
probe = Path(__file__).with_suffix(".rs")
(output / "main.rs").write_text(
    f'#[allow(dead_code)]\n#[path = {json.dumps(str(legacy))}]\nmod legacy;\n'
    + probe.read_text().replace("//!", "//")
)
subprocess.run(
    ["cargo", "+1.99.1", "generate-lockfile", "--manifest-path", str(output / "Cargo.toml")],
    check=True,
)
source_identity = source.parent / "SOURCE.json"
metadata = {
    "archmage_source": str(source),
    "source_identity": json.loads(source_identity.read_text()) if source_identity.exists() else None,
    "inputs": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [legacy, probe, Path(__file__).resolve()]
    },
    "lock_sha256": hashlib.sha256((output / "Cargo.lock").read_bytes()).hexdigest(),
    "meaning": "Bitwise compatibility and domain witnesses; no accuracy bound or timing measurement",
}
(output / "SOURCE.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(output)
