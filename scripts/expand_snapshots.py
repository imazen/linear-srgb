#!/usr/bin/env python3
"""Capture or check the real library's expanded SIMD modules on every target."""

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN = "1.99.0"
EXPAND = "cargo-expand 1.0.126"
PROVIDERS = {"archmage": "0.9.31", "archmage-macros": "0.9.31", "magetypes": "0.9.31"}
TARGETS = ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu", "wasm32-wasip1")
PRESETS = {"default": ROOT / "expand/Cargo.toml", "all-features": ROOT / "expand/all-features/Cargo.toml"}
MODULES = ("target_features", "tokens", "simd", "kernels", "tf")
OUTPUT = ROOT / "docs/expanded"


def environment():
    env = os.environ.copy()
    # A caller's native/relaxed-SIMD flags must not silently change the captures.
    for key in list(env):
        if key in {"RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC", "CARGO_BUILD_TARGET"} or (
            key.startswith("CARGO_TARGET_") and key.endswith("_RUSTFLAGS")
        ):
            del env[key]
    scratch = Path.home() / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    env.update(
        RUSTUP_TOOLCHAIN=TOOLCHAIN,
        CARGO_TARGET_DIR=str(ROOT / "target/expand"),
        CARGO_TERM_COLOR="never",
        TMPDIR=str(scratch),
    )
    return env


def run(args, env):
    return subprocess.run(args, cwd=ROOT, env=env, check=True, text=True, stdout=subprocess.PIPE).stdout


def expected_features(preset):
    features = tomllib.loads((ROOT / "Cargo.toml").read_text())["features"]
    selected = {"default"} if preset == "default" else set(features)
    pending = list(selected)
    while pending:
        for feature in features[pending.pop()]:
            if feature in features and feature not in selected:
                selected.add(feature)
                pending.append(feature)
    return selected


def resolve(manifest, target, preset, env):
    metadata = json.loads(run([
        "cargo", "metadata", "--locked", "--format-version", "1",
        "--manifest-path", str(manifest), "--filter-platform", target,
    ], env))
    library = next(p for p in metadata["packages"] if Path(p["manifest_path"]) == ROOT / "Cargo.toml")
    node = next(n for n in metadata["resolve"]["nodes"] if n["id"] == library["id"])
    actual = set(node["features"])
    expected = expected_features(preset)
    if actual != expected:
        raise ValueError(f"{preset}: expected library features {sorted(expected)}, got {sorted(actual)}; update the resolution anchor")
    for name, version in PROVIDERS.items():
        versions = {p["version"] for p in metadata["packages"] if p["name"] == name}
        if versions != {version}:
            raise ValueError(f"expected {name} {version}, got {versions}")
    return sorted(actual)


def compare(path, expected):
    if not path.is_file():
        return f"Missing expansion: {path.relative_to(ROOT)}"
    actual = path.read_text()
    if actual == expected:
        return None
    diff = difflib.unified_diff(actual.splitlines(), expected.splitlines(), fromfile=str(path), tofile="regenerated", lineterm="")
    # Keep CI readable; the complete new output is available via the update recipe.
    return "Expansion drift:\n" + "\n".join(line for _, line in zip(range(60), diff))


def stale_files(output, targets):
    expected = {output / target / preset / f"{module}.expanded.rs"
                for target in targets for preset in PRESETS for module in MODULES}
    return sorted(path for path in output.rglob("*.rs") if path not in expected and (
        targets == TARGETS or path.relative_to(output).parts[0] in targets
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare without rewriting committed files")
    parser.add_argument("--target", choices=TARGETS, help="capture only this CI matrix target")
    args = parser.parse_args()
    targets = (args.target,) if args.target else TARGETS
    stale = stale_files(OUTPUT, targets)
    if stale:
        raise ValueError("Unexpected expansion files; remove explicitly:\n" + "\n".join(str(p) for p in stale))
    env = environment()
    rustc = run(["rustc", "--version"], env).strip()
    if not rustc.startswith(f"rustc {TOOLCHAIN} "):
        raise ValueError(f"expected Rust {TOOLCHAIN}, got {rustc}")
    expand = run(["cargo", "expand", "--version"], env).strip()
    if expand != EXPAND:
        raise ValueError(f"expected {EXPAND}, got {expand}; install the pinned version")
    errors = []
    for target in targets:
        for preset, manifest in PRESETS.items():
            features = resolve(manifest, target, preset, env)
            base = ["--locked", "--manifest-path", str(manifest), "--package", "linear-srgb", "--lib", "--target", target]
            print(f"Typechecking {target}/{preset}: {', '.join(features)}", flush=True)
            # Expansion is not a compile oracle. Check the exact dependency package first.
            run(["cargo", "check", *base], env)
            lock = hashlib.sha256(manifest.with_name("Cargo.lock").read_bytes()).hexdigest()
            for module in MODULES:
                body = run(["cargo", "expand", *base, "--color", "never", module], env)
                if not body.strip() or "mod " + module + " {" not in body:
                    raise ValueError(f"empty or missing module expansion: {target}/{preset}/{module}")
                if str(ROOT) in body or str(Path.home()) in body:
                    raise ValueError(f"non-portable absolute path in {target}/{preset}/{module}")
                header = (
                    "// @generated by scripts/expand_snapshots.py; do not edit.\n"
                    f"// {rustc}; {expand}\n"
                    f"// Target: {target}; preset: {preset}; module: {module}\n"
                    f"// Library features: {', '.join(features)}\n"
                    f"// Providers: {', '.join(f'{k} {v}' for k, v in PROVIDERS.items())}\n"
                    f"// Resolution lock SHA-256: {lock}\n"
                    "// Baseline target features; RUSTFLAGS overrides cleared.\n\n"
                )
                content = header + body
                if len(content.encode()) > 600_000:
                    raise ValueError(f"{module} exceeds 600 KB; split the capture into child modules")
                path = OUTPUT / target / preset / f"{module}.expanded.rs"
                if args.check:
                    error = compare(path, content)
                    if error:
                        errors.append(error)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content)
                print(f"{'Checked' if args.check else 'Wrote'} {path.relative_to(ROOT)} ({len(content.encode())} bytes)", flush=True)
    if errors:
        raise ValueError("\n\n".join(errors) + "\nRegenerate with just expand-snapshots and review the diff.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
