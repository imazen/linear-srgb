#!/usr/bin/env python3
"""Compare a probe's output with the captured, pre-refactor output bits."""
from pathlib import Path
import argparse
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
baseline = root / "benchmarks/entry_point_ownership_2026-11-10.md"
pattern = re.compile(r"^(?:(?:native|wasm) )?(x4|x8|x16) (\w+) ([0-9a-f]{16})$", re.M)
expected = {(width, tier): value for width, tier, value in pattern.findall(baseline.read_text())}
actual = pattern.findall(args.output.read_text())
assert actual, f"No fingerprints in {args.output}"
for width, tier, value in actual:
    key = (width, tier)
    assert key in expected, f"No pre-refactor capture for {key}"
    assert value == expected[key], f"Changed bits for {key}: {value} != {expected[key]}"
seen = {(width, tier) for width, tier, _ in actual}
for _, tier in seen:
    for width in ["x4", "x8", "x16"]:
        assert (width, tier) in seen, f"Missing {width}/{tier}"
print(f"All {len(actual)} output fingerprints match the pre-refactor capture")
