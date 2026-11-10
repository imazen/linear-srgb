#!/usr/bin/env python3
"""Export a complete crate through the pinned anchors in a PR #30 source archive.

Unlike selected-module snapshots, cargo expand receives no ITEM filter.
The source archive is read-only; captures, build cache and metadata use output/.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

TARGETS = ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu", "wasm32-wasip1")
PART_BYTES = 500_000


def verify(directory, selected=None):
    metadata = json.loads((directory / "MANIFEST.json").read_text())
    expected = {(target, preset) for target in TARGETS for preset in ("default", "all-features")}
    observed = [(row["target"], row["preset"]) for row in metadata["captures"]]
    assert len(observed) == len(expected) and set(observed) == expected, "incomplete or duplicated target/profile matrix"
    assembled = None
    for capture in metadata["captures"]:
        chunks = []
        for part in capture["parts"]:
            data = (directory / part["path"]).read_bytes()
            assert len(data) == part["bytes"] <= PART_BYTES, part["path"]
            assert hashlib.sha256(data).hexdigest() == part["sha256"], part["path"]
            chunks.append(data)
        body = b"".join(chunks)
        assert len(body) == capture["bytes"], capture["path"]
        assert hashlib.sha256(body).hexdigest() == capture["sha256"], capture["path"]
        if selected == (capture["target"], capture["preset"]):
            assembled = body
    if selected is not None:
        assert assembled is not None, f"missing capture: {selected}"
    return assembled


def split(body):
    parts, lines, size = [], [], 0
    for line in body.splitlines(keepends=True):
        length = len(line.encode())
        assert length <= PART_BYTES, "one source line exceeds the part limit"
        if lines and size + length > PART_BYTES:
            parts.append("".join(lines).encode())
            lines, size = [], 0
        lines.append(line)
        size += length
    if lines:
        parts.append("".join(lines).encode())
    assert b"".join(parts) == body.encode()
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", help="PR source archive from issue-32-probe, with expand/ anchors")
    parser.add_argument("output", type=Path, nargs="?", help="new durable output directory")
    parser.add_argument("--verify", type=Path, help="verify committed complete-source parts and hashes")
    parser.add_argument("--assemble", nargs=4, metavar=("DIRECTORY", "TARGET", "PRESET", "NEW_FILE"), help="verify and reconstruct one complete expanded Rust file")
    args = parser.parse_args()
    if args.verify is not None or args.assemble is not None:
        if args.source is not None or args.output is not None or (args.verify is not None and args.assemble is not None):
            parser.error("select capture, verify, or assemble")
        if args.verify is not None:
            verify(args.verify)
            print("All complete-crate source parts and hashes verified")
        else:
            directory, target, preset, destination = args.assemble
            body = verify(Path(directory), (target, preset))
            with Path(destination).open("xb") as file:
                file.write(body)
            print(f"Reconstructed {destination}: {len(body)} bytes")
        return
    if args.source is None or args.output is None:
        parser.error("capture requires SOURCE and OUTPUT")
    source = args.source.resolve()
    source_revision = json.loads((source.parent / "RUN.json").read_text())["source_revision"]
    manifests = {"default": source / "expand/Cargo.toml", "all-features": source / "expand/all-features/Cargo.toml"}
    for manifest in manifests.values():
        if not manifest.is_file() or not manifest.with_name("Cargo.lock").is_file():
            raise SystemExit(f"missing pinned expansion anchor: {manifest}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(RUSTUP_TOOLCHAIN="1.99.0", CARGO_TARGET_DIR=str(output / "cargo-target"),
               TMPDIR=str(Path.home() / "tmp"), RUSTFLAGS="", CARGO_ENCODED_RUSTFLAGS="",
               CARGO_TERM_COLOR="never")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    def run(command):
        return subprocess.check_output(command, cwd=source, env=env, text=True)
    rustc = run(["rustc", "--version"]).strip()
    expand = run(["cargo", "expand", "--version"]).strip()
    assert rustc.startswith("rustc 1.99.0 "), rustc
    assert expand == "cargo-expand 1.0.126", expand
    # Reuse the PR's exact feature/provider-resolution guard; do not weaken it.
    spec = importlib.util.spec_from_file_location("expansion_guard", source / "scripts/expand_snapshots.py")
    guard = importlib.util.module_from_spec(spec)
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(guard)
    finally:
        sys.dont_write_bytecode = old_bytecode
    metadata = {"source_revision": source_revision, "rustc": rustc, "cargo_expand": expand,
                "coverage": "complete compiled library; no ITEM filter", "max_part_bytes": PART_BYTES, "captures": []}
    for target in TARGETS:
        for preset, manifest in manifests.items():
            base = ["--locked", "--manifest-path", str(manifest), "--package", "linear-srgb", "--lib", "--target", target]
            features = guard.resolve(manifest, target, preset, env)
            print(f"Typecheck and export whole crate: {target}/{preset}", flush=True)
            subprocess.run(["cargo", "check", *base], cwd=source, env=env, check=True)
            body = run(["cargo", "expand", *base, "--color", "never"])
            assert body.strip(), f"empty crate export: {target}/{preset}"
            path = output / target / f"{preset}.expanded.rs"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
            record = {"target": target, "preset": preset, "path": str(path.relative_to(output)),
                      "bytes": len(body.encode()), "sha256": hashlib.sha256(body.encode()).hexdigest(),
                      "features": features, "parts": [],
                      "lock_sha256": hashlib.sha256(manifest.with_name("Cargo.lock").read_bytes()).hexdigest()}
            for index, data in enumerate(split(body), 1):
                part_path = output / "parts" / target / preset / f"crate.part-{index:02}.expanded.rs"
                part_path.parent.mkdir(parents=True, exist_ok=True)
                part_path.write_bytes(data)
                record["parts"].append({"path": str(part_path.relative_to(output)), "bytes": len(data),
                                        "sha256": hashlib.sha256(data).hexdigest()})
            metadata["captures"].append(record)
            (output / "MANIFEST.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"Wrote {record['path']}: {record['bytes']} bytes, SHA-256 {record['sha256']}", flush=True)
    verify(output)


if __name__ == "__main__":
    main()
