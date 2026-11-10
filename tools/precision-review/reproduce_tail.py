#!/usr/bin/env python3
"""Run issue #32 against an explicit committed revision in an isolated archive.

Adapted from imazen/archmage's versioned primitive-contracts reproducer;
see docs/predictable-math.md for source provenance and recorded results.
The original checkout and all committed numerical expectations stay unchanged.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", help="committed source revision, not working-copy contents")
    parser.add_argument("output", type=Path, help="new durable output directory")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "--verify", args.revision + "^{commit}"], cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(["git", "archive", revision], cwd=ROOT)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(source, filter="data")
    probe = output / "probe"
    (probe / "src").mkdir(parents=True)
    (probe / "Cargo.toml").write_text('''[package]
name = "linear-srgb-tail-reproducer"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
linear-srgb = { path = "../source", features = ["std", "avx512"] }
''')
    (probe / "src/main.rs").write_text('''fn main() {
    let mut values = [0.125_f32; 17];
    linear_srgb::default::srgb_to_linear_slice(&mut values);
    let scalar = linear_srgb::default::srgb_to_linear(0.125_f32);
    println!("input=3e000000 body={:08x} tail={:08x} single={:08x}",
        values[0].to_bits(), values[16].to_bits(), scalar.to_bits());
    assert_eq!(values[0].to_bits(), values[16].to_bits(),
        "issue #32: slice chunk/tail mismatch at input 0x3e000000");
}
''')
    env = os.environ.copy()
    env.update(RUSTUP_TOOLCHAIN="1.99.1", CARGO_TARGET_DIR=str(output / "target"),
               TMPDIR=str(Path.home() / "tmp"), RUSTFLAGS="", CARGO_ENCODED_RUSTFLAGS="",
               CARGO_TERM_COLOR="never")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    command = ["cargo", "run", "--release", "--manifest-path", str(probe / "Cargo.toml")]
    rustc = subprocess.check_output(["rustc", "--version"], env=env, text=True).strip()
    metadata = {"source_revision": revision, "source_archive_sha256": hashlib.sha256(archive).hexdigest(),
                "rustc": rustc, "command": command, "input_bits": "3e000000", "length": 17,
                "original_checkout_mutated": False, "native_target_flags": "baseline"}
    metadata_path = output / "RUN.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    with (output / "output.log").open("w") as log:
        process = subprocess.Popen(command, cwd=output, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        result = process.wait()
    metadata["exit_code"] = result
    lock = probe / "Cargo.lock"
    if lock.is_file():
        metadata["lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
        packages = tomllib.loads(lock.read_text())["package"]
        metadata["math_dependencies"] = {p["name"]: p["version"] for p in packages if p["name"] in {"archmage", "archmage-macros", "magetypes", "libm"}}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
