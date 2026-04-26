# Per-target ASM snapshots

Cross-compiled assembly dumps of the public dispatchers in `examples/asm-stub.rs`,
captured by `cargo asm` and re-checked in CI. Each `stub_<fn>` is
`#[inline(never)] #[no_mangle]` so it survives as its own symbol in the
example binary, with the public dispatcher's per-tier code inlined into it.

## Two files per stub

For each `(target, stub)` pair we commit two files:

| File | Role | Gate behavior |
|---|---|---|
| `<stub>.s` | Full assembly listing — registers, immediates, labels, `.cfi_*` directives, section metadata, all of it. Useful for human review when the essence drifts. | **Not asserted.** Diff is rendered into the GitHub job summary and the regenerated files are uploaded as a per-target artifact (`asm-<target>`, 30-day retention). |
| `<stub>.essence` | Opcode-only normalized form. Each line is `MNEMONIC` + optional ` MEM`/` IMM` tags reflecting the operand shape. Stripped of registers, labels, immediate values, addressing-mode offsets, directives. | **Hard gate.** CI runs `git diff --exit-code` on the `.essence` files. Drift here means real codegen change (new instruction types, instruction-count delta, memory-access pattern shift) — not just register renaming or label renumbering. |

The two-tier scheme came out of issue #24. The original single-tier gate
asserted on the full `.s` files and triggered false positives on every
toolchain bump, basic-block label renumbering after a cosmetic body
reorder, or `cargo-show-asm` output-format change. Normalizing to opcode
shape kills those false positives while still catching the codegen
changes we actually want a gate for.

## What the essence form catches

- New instruction types appearing — e.g., a stray `call panic_bounds_check` indicates we lost a fixed-size-array bounds-check elimination.
- Instruction-count change — different unrolling factor, body inlined twice, etc.
- Memory-access pattern change — a `LDR` becoming `LDR MEM IMM` (pre-indexed), or a load disappearing entirely.
- Branch-shape change — new conditional branches, different basic-block count.

## What it lets through

- Register renaming (`v22.4s` → `v23.4s`).
- Basic-block label renumbering (`.LBB12_2` → `.LBB14_5`).
- Immediate constant changes that don't alter semantics (offset reshuffles after a function-prologue reorder).
- Instruction scheduling within an opcode-equivalent sequence.

These are all things the LLVM backend can do across rustc versions
without changing what the code does. The essence diff is invariant to
them; the full `.s` diff isn't.

## Regenerating

```bash
scripts/dump-asm.sh                # all targets
scripts/dump-asm.sh aarch64        # one target (substring match)
```

This rebuilds the example binary, dumps each stub's full assembly via
`cargo asm`, and produces the corresponding `.essence` alongside. Both
files commit together — never regenerate one without the other.

When CI flags an essence drift you believe is intentional (a real
refactor that changes instruction shape), regenerate both, review the
full `.s` diff to sanity-check what changed, then commit.

## Targets

`aarch64-unknown-linux-gnu` (NEON) and `wasm32-unknown-unknown`
(SIMD128) — the two paths where Pattern 2 of issue #23 produced new
codegen via the `f32x16` polyfill (4-wide → 16-wide outer loops via
4× f32x4). x86_64 V3/V4 are intentionally not snapshotted: their
chunk size didn't change in Pattern 2, and on x86_64 the SIMD body
lives in separate `__arcane_*_v3 / _v4` symbols not reachable from the
example stubs.

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
