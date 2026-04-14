#!/usr/bin/env python3
"""
Reproduce the base 4/4 S2L/L2S coefficients in src/rational_poly.rs.

This script shells out to a pinned polyfit example that performs the full
pipeline: Sanathanan-Koerner initialization → Levenberg-Marquardt with Nielsen
damping → deterministic multi-restart → local f32 ULP search with weighted
boundary constraint.

Determinism:
  - polyfit uses no RNG; all perturbations are sin/cos of fixed phases
  - No wall-clock, no hidden seeds
  - Identical inputs → identical output coefficients

Recipe used for the coefficients currently in src/rational_poly.rs:
  S2L: 4/4 rational, domain [threshold_gamma, 1.0], constraint_mode=Both,
       boundary weight w=50000, 8 restarts, relative error weighting,
       f32 ULP local search (radius 3, 5 rounds)
  L2S: 4/4 rational on sqrt(linear), domain [threshold_linear.sqrt(), 1.0],
       constraint_mode=Both, boundary weight w=1000, 8 restarts, relative
       error weighting, f32 ULP local search (radius 3, 5 rounds)

Run from repo root:
  python3 scripts/fit_srgb_fast_polyfit.py

Requires:
  - polyfit crate at ../polyfit (clone https://github.com/lilith/polyfit)
  - cargo, rustc
"""
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Assumes polyfit sibling checkout. Override with POLYFIT_DIR env var.
import os
POLYFIT_DIR = Path(os.environ.get("POLYFIT_DIR", REPO_ROOT.parent / "polyfit"))

if not (POLYFIT_DIR / "Cargo.toml").exists():
    print(f"error: polyfit not found at {POLYFIT_DIR}", file=sys.stderr)
    print("set POLYFIT_DIR=/path/to/polyfit, or clone alongside linear-srgb", file=sys.stderr)
    sys.exit(1)

print(f"Running polyfit's apply_linear_srgb example at {POLYFIT_DIR} ...")
print("This is deterministic: identical input → identical output coefficients.")
print()

result = subprocess.run(
    ["cargo", "run", "--release", "--example", "apply_linear_srgb"],
    cwd=POLYFIT_DIR,
    check=True,
)
sys.exit(result.returncode)
