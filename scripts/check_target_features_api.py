#!/usr/bin/env python3
"""Check that regenerated snapshots retain token hooks and their token-free peers."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
items = set()
for snapshot in (root / "docs/public-api").glob("linear-srgb*.txt"):
    for line in snapshot.read_text().splitlines():
        if not line.startswith("pub fn "):
            continue
        line, _, aliases = line.partition(" [also: ")
        items.add(line)
        if aliases:
            name, signature = line[7:].split("(", 1)
            for alias in aliases.rstrip("]").split(", "):
                items.add(f"pub fn {alias}::{name.rsplit('::', 1)[1]}({signature}")

legacy = {line for line in items if line.startswith("pub fn tokens::")}
assert legacy, "No legacy signatures found; regenerate snapshots first"
for line in sorted(legacy):
    expected = line.replace("pub fn tokens::", "pub fn target_features::", 1)
    expected, count = re.subn(r"\(archmage::[^,]+, ", "(", expected, count=1)
    assert count == 1, f"Unrecognized legacy token signature: {line}"
    assert expected in items, f"Missing token-free equivalent: {expected}"
for line in items:
    if line.startswith("pub fn target_features::"):
        assert "archmage::" not in line and "magetypes::" not in line, line
print(f"All {len(legacy)} snapshotted token functions have matching token-free signatures")
