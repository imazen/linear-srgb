#!/usr/bin/env python3
"""Check each architecture's complete, exact token-free replacement surface."""
from pathlib import Path
import argparse
import re

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "x86_64-unknown-linux-gnu": ("x86_64", "v3", {"x4", "x8"}),
    "aarch64-unknown-linux-gnu": ("aarch64", "neon", {"x4"}),
    "wasm32-wasip1": ("wasm32", "wasm128", {"x4"}),
}
FILES = ["linear-srgb.txt", "linear-srgb.features.txt", "linear-srgb.internal.txt"]


def read_surface(directory: Path, target: str) -> set[str]:
    arch, _, _ = TARGETS[target]
    items = set()
    observed_arches = set()
    for filename in FILES:
        snapshot = directory / filename
        assert snapshot.is_file(), f"Missing {target} snapshot: {snapshot}"
        text = snapshot.read_text()
        observed_arches.update(re.findall(r"^## arch surface: (\w+)", text, re.M))
        for line in text.splitlines():
            if line.startswith("pub mod "):
                items.add(line)
            if not line.startswith("pub fn "):
                continue
            line, _, aliases = line.partition(" [also: ")
            items.add(line)
            if aliases:
                name, signature = line[7:].split("(", 1)
                for alias in aliases.rstrip("]").split(", "):
                    items.add(f"pub fn {alias}::{name.rsplit('::', 1)[1]}({signature}")
    assert observed_arches == {arch}, f"{target}: wrong architecture sections: {observed_arches}"
    return items


def check_surface(items: set[str], target: str) -> int:
    _, tier, widths = TARGETS[target]
    legacy = {line for line in items if line.startswith(("pub fn tokens::x4::", "pub fn tokens::x8::"))}
    assert legacy, f"{target}: no legacy x4/x8 signatures recorded"
    expected = set()
    for line in sorted(legacy):
        replacement = line.replace("pub fn tokens::", "pub fn target_features::", 1)
        replacement, count = re.subn(r"\(archmage::[^,]+, ", "(", replacement, count=1)
        assert count == 1, f"Unrecognized legacy token signature: {line}"
        expected.add(replacement)
    actual = {line for line in items if line.startswith("pub fn target_features::")}
    missing, extra = expected - actual, actual - expected
    assert not missing, f"{target}: missing replacements:\n" + "\n".join(sorted(missing))
    assert not extra, f"{target}: extra public functions:\n" + "\n".join(sorted(extra))
    modules = {line.removeprefix("pub mod target_features::") for line in items if line.startswith("pub mod target_features::")}
    assert modules == widths, f"{target}: expected width modules {widths}, got {modules}"
    required = f"pub fn target_features::x4::srgb_to_linear_{tier}([f32; 4]) -> [f32; 4]"
    assert required in actual, f"{target}: missing native {tier} entry point"
    for line in actual:
        assert "archmage::" not in line and "magetypes::" not in line, line
        suffix = line.split("(", 1)[0].rsplit("_", 1)[1]
        assert suffix in {tier, "scalar"}, f"{target}: unexpected tier: {line}"
    return len(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, help="check only this CI matrix target")
    args = parser.parse_args()
    for target in [args.target] if args.target else TARGETS:
        items = read_surface(ROOT / "docs/public-api" / target, target)
        count = check_surface(items, target)
        print(f"{target}: exactly {count} token-free replacements; no public x16 or extra tier families")


if __name__ == "__main__":
    main()
