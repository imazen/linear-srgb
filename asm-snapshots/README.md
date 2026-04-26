# Per-target ASM snapshots

Cross-compiled assembly dumps of the public dispatchers in `examples/asm-stub.rs`,
captured by `cargo asm` and re-checked in CI. Each `stub_<fn>` is
`#[inline(never)] #[no_mangle]` so it survives as its own symbol in the
example binary, with the public dispatcher's per-tier code inlined into it.

**Informational, not a hard CI gate.** The
[`asm-snapshots.yml`](../.github/workflows/asm-snapshots.yml) workflow
re-dumps on every PR and main push, surfaces any diff in the job summary,
and uploads the regenerated files as artifacts. **It does not fail the
build on drift.** ASM is rustc-version-deterministic but not stable across
toolchain bumps, label numbering shifts on cosmetic body reorders, and
`cargo asm`'s output format itself is a moving target — treating drift as
a hard error produced more false positives than real catches.

The committed snapshots are still load-bearing as a "last-known-good"
reference: reviewers can see exactly what changed in codegen during a PR
without running anything locally. For the original Pattern 2 verification
(chunk-size unification on NEON/WASM), the snapshots showed the expected
4× loop unrolling with identical per-pixel ops — the signal landed even
without the hard gate.

## Targets

| Target | What it covers |
|---|---|
| `aarch64-unknown-linux-gnu` | NEON 4-wide path — the high-risk Pattern 2 codegen |
| `wasm32-unknown-unknown` | WASM SIMD128 same shape |

x86_64 V3/V4 are intentionally **not** snapshotted here — their dispatch
goes through separate `__arcane_*_v3 / _v4` symbols (the stub body only
captures the dispatch table on x86_64, not the SIMD code), and their
chunk size doesn't change in Pattern 2 (V3 already 8-wide, V4 already
16-wide). x86_64 verification leans on `cargo test --all-features --release`
+ tango regression bench in `tango/`.

## Regenerating

```bash
scripts/dump-asm.sh                # all targets
scripts/dump-asm.sh aarch64        # one target
```

Then `git diff asm-snapshots/` shows what changed.

## Determinism

ASM is stable for a fixed rustc version + target. Bumping the CI toolchain
will produce drift — that's expected, regenerate the snapshots in the same
PR. The gate's purpose isn't pinning to a specific instruction sequence
forever; it's catching unintended codegen drift introduced by source
refactors against the *same* compiler.

## What "an OK diff" looks like

- Register renaming (`v22` → `v23` etc.) when surrounding code changes.
- Basic block label renumbering (`.LBB12_2` → `.LBB14_2`).
- Reordering of independent operations (LLVM scheduler).

## What "a load-bearing diff" looks like

- New function calls inserted (extra LUT lookups, new intrinsics).
- Loop bounds changing (`subs x8, x8, #16` → `subs x8, x8, #4`).
- Different SIMD widths (`q` register → `s` register on aarch64; `f32x4` → scalar on WASM).
- Vastly different instruction count (>20% line delta).

For Pattern 2 specifically, the expected diff shape is **loop unrolling
expansion** — the 4-wide outer loop becomes 16-wide via 4× the body — but
the per-pixel SIMD instructions should stay in the same form.
