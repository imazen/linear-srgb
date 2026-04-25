# Per-target ASM snapshots

Cross-compiled assembly dumps of the public dispatchers in `examples/asm-stub.rs`,
captured by `cargo asm` and re-checked in CI. Each `stub_<fn>` is
`#[inline(never)] #[no_mangle]` so it survives as its own symbol in the
example binary, with the public dispatcher's per-tier code inlined into it.

This is the **codegen gate** for issue #23 Pattern 2 — the chunk-size
unification (4-wide → 16-wide via `f32x16` polyfill on NEON/WASM) MUST
produce equivalent machine code to the hand-written 4-wide loops it
replaces. ASM diff against committed snapshots is the deterministic answer.

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
