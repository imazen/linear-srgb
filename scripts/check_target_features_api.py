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

legacy = {line for line in items if line.startswith(("pub fn tokens::x4::", "pub fn tokens::x8::"))}
assert legacy, "No x4/x8 legacy signatures found; regenerate snapshots first"
expected_items = set()
for line in sorted(legacy):
    expected = line.replace("pub fn tokens::", "pub fn target_features::", 1)
    expected, count = re.subn(r"\(archmage::[^,]+, ", "(", expected, count=1)
    assert count == 1, f"Unrecognized legacy token signature: {line}"
    expected_items.add(expected)
actual = {line for line in items if line.startswith("pub fn target_features::")}
missing = expected_items - actual
extra = actual - expected_items
assert not missing, "Missing replacements:\n" + "\n".join(sorted(missing))
assert not extra, "Unexpected public feature functions:\n" + "\n".join(sorted(extra))
for line in actual:
    assert "archmage::" not in line and "magetypes::" not in line, line
assert not any(line.startswith("pub fn target_features::x16::") for line in actual)
print(f"Exactly {len(legacy)} token-free x4/x8 replacements; no public x16 or extra tier families")
