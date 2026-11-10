#!/usr/bin/env python3
"""Compare frozen ISA profiles; never generate or change expected output."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent / "fixtures"
GROUPS = ("scalar.txt", "slices.txt", "rgba.txt", "integers.txt", "tokens.txt")


def rows(profile):
    result = {}
    for group in GROUPS:
        for line in (ROOT / profile / group).read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            name, count, *hashes = line.split()
            if group == "tokens.txt":
                # Only x4 has corresponding public hooks on all three ISAs.
                if not name.startswith("linear_srgb::tokens::x4/"):
                    continue
                name = re.sub(r"_(?:v3|neon|wasm128)(?=/|$)", "_tier", name)
            key = group + "/" + name
            assert key not in result, f"duplicate case: {key}"
            result[key] = (int(count), hashes)
    return result


def compare(left, right):
    a, b = rows(left), rows(right)
    assert a and a.keys() == b.keys(), "common profile case sets differ"
    grid_total = grid_changed = edge_changed = tail_changed = 0
    for key, (count, hashes) in a.items():
        other_count, other = b[key]
        assert count == other_count and len(hashes) == len(other), key
        if key.endswith("/tails"):
            tail_changed += hashes != other
        else:
            assert count in (65536, 65792), (key, count)
            grid_total += 1
            grid_changed += hashes[:16] != other[:16]
            edge_changed += sum(x != y for x, y in zip(hashes[16:], other[16:]))
    same_u8 = (ROOT / left / "u8_decode.txt").read_bytes() == (ROOT / right / "u8_decode.txt").read_bytes()
    print(f"{left} vs {right}: regular-grid streams {grid_total - grid_changed}/{grid_total} identical; "
          f"changed edge chunks {edge_changed}; changed tail streams {tail_changed}; "
          f"256-value u8 decode {'identical' if same_u8 else 'DIFFERS'}")


if __name__ == "__main__":
    for target in ("aarch64", "wasm32-precise", "wasm32-default"):
        compare("x86_64", target)
