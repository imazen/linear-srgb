#!/usr/bin/env python3
"""Keep deprecated token APIs downstream of the canonical implementation."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]


def code(path):
    source = path.read_text().split("#[cfg(test)]", 1)[0]
    source = re.sub(r"#\[doc[^\n]*", "", source)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


assert "\nmod kernels;" in (root / "src/lib.rs").read_text()
for directory in ["target_features", "kernels"]:
    for path in (root / "src" / directory).glob("*.rs"):
        if directory == "target_features" and path.stem in {"compatibility_tests", "dispatch_tests", "fingerprint"}:
            assert f"#[cfg(test)]\nmod {path.stem};" in (path.parent / "mod.rs").read_text()
            continue
        assert "crate::tokens" not in code(path), f"Legacy dependency in {path}"
for path in (root / "src/tokens").glob("x*.rs"):
    source = code(path)
    assert "crate::target_features::" in source, f"Missing canonical adapter: {path}"
    assert not re.search(r"\bmagetypes::simd\b|\b(?:kernels|fast_math)\b", source), path
    assert not re.search(r"\b(?:for|while|const)\b", source), path
assert "crate::target_features::" in code(root / "src/simd.rs")
print("Canonical entries/private kernels have no dependency on legacy token APIs")
