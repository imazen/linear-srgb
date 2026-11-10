#!/usr/bin/env python3
"""Compare dispatcher stub instruction shapes from two rustc --emit=asm captures."""
from pathlib import Path
import argparse
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("before", type=Path)
parser.add_argument("after", type=Path)
parser.add_argument("--require-identical", action="store_true")
args = parser.parse_args()


def read(path):
    source = path.read_text()
    result = {}
    for match in re.finditer(r"^(stub_\w+):", source, re.M):
        name = match.group(1)
        rest = source[match.end():]
        end = re.search(r"^\s*\.size\s+" + name + r"\b", rest, re.M)
        if end is None:
            end = re.search(r"^\s*end_function", rest, re.M)
        assert end is not None, f"Unterminated {name} in {path}"
        operations = []
        for line in rest[:end.start()].splitlines():
            line = line.strip()
            if not line or line.startswith((".", "#", "//")) or line.endswith(":"):
                continue
            opcode = line.split()[0].upper()
            # Match the ARM memory/immediate distinctions in dump-asm.sh.
            if "[" in line:
                opcode += " MEM"
            if "#" in line:
                opcode += " IMM"
            operations.append(opcode)
        result[name] = operations
    assert result, f"No dispatcher stubs in {path}"
    return result


before, after = read(args.before), read(args.after)
assert before.keys() == after.keys(), "Dispatcher stub set changed"
changed = [name for name in before if before[name] != after[name]]
for name in changed:
    print(f"{name}: {len(before[name])} -> {len(after[name])} instructions/tags")
print(f"Identical instruction shapes: {len(before) - len(changed)}/{len(before)}")
if args.require_identical:
    assert not changed, "Dispatcher instruction shapes changed"
