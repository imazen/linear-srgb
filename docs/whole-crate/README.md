# Complete expanded-crate review exports

[`pr30-f168120c/`](pr30-f168120c/MANIFEST.json) records the **whole compiled
library** from PR #30 commit
[`f168120c`](https://github.com/imazen/linear-srgb/commit/f168120c29a5e3f8e9f71ba0cd17e68ca1d3d4d5).
These captures pass no ITEM filter to `cargo expand --lib`. They include
`default`, `precise`, `scalar`, `rational_poly`, LUTs, crate-root items and
all other modules present in each configuration, rather than only SIMD modules.

| Target | Default bytes | All-feature bytes |
| --- | ---: | ---: |
| x86_64-unknown-linux-gnu | 495,576 | 704,437 |
| aarch64-unknown-linux-gnu | 318,348 | 427,125 |
| wasm32-wasip1 | 313,453 | 423,797 |

The compiler is Rust 1.99.0 and cargo-expand is 1.0.126. The PR's committed
resolution anchors pin macro providers to 0.9.31; their feature/provider guard
also runs before capture. The manifest records resolved library features,
resolution-lock hashes, complete-source SHA-256, and each source part's size
and SHA-256. Global Rust flags are empty; global WASM relaxed-SIMD is disabled.

Source is split only at line boundaries into parts of at most 500,000 bytes.
Concatenating every listed part in order reproduces the **exact full export**;
parts are not independently compilable modules. This keeps the 704,437-byte
configuration within the repository's per-file Rust source limit without
omitting any compiled code. Full unpartitioned files remain with the
[capture artifacts](../predictable-math.md#artifacts-and-reproduction).

```sh
just verify-whole-crate docs/whole-crate/pr30-f168120c
python3 tools/precision-review/expand_all.py --assemble \
  docs/whole-crate/pr30-f168120c x86_64-unknown-linux-gnu all-features \
  "$HOME/tmp/linear-srgb-full-expanded.rs"
```

The assembler refuses an existing destination. The verifier checks every part
and the reconstructed whole. No tests, math implementation or expected output
is changed by assembly or verification.

These are frozen **review exports of the recorded PR revision**. Integrity
verification does not re-expand the current working tree and is not a
freshness check. PR #30's existing selected-module snapshot CI remains a
separate, narrower gate; replacing it with whole-crate comparison is still
needed for complete source-change detection. These default/all-feature
captures also do not cover no-std or globally relaxed-SIMD configurations.
