# linear-srgb — Project Notes

## Current transition

See [docs/SIMD_TRANSITION.md](docs/SIMD_TRANSITION.md) for the branch, issue and TODO inventory checked 2026-11-05. PR #30 adds `target_features` and deprecates the retained `tokens` module; it is open, not a published release. Ordinary code uses `default`; SIMD contexts use arrays/slices without exchanging dependency tokens. Preserve old signatures until an explicitly approved removal.

## Known Bugs

- [Issue 29](https://github.com/imazen/linear-srgb/issues/29): reported LLVM failure on soft-float `x86_64-unknown-none`. Normal-target no-std checks do not resolve this; no new reproduction was run during the documentation inventory.

## Historical assembly drift

The May/June WASM drift and October ARM/WASM drift remain historical reports, not current “every push is red” claims. Both target checks pass on main `1340e864` ([run 41986870385](https://github.com/imazen/linear-srgb/actions/runs/41986870385)) and API PR #30 ([run 41985914165](https://github.com/imazen/linear-srgb/actions/runs/41985914165)). [Issue 28](https://github.com/imazen/linear-srgb/issues/28) retains the old evidence. Do not infer its cause from current green checks, regenerate expected snapshots blindly, or weaken the existing gates.

The original toolchain/codegen observations are retained in [docs/ASM_HISTORY.md](docs/ASM_HISTORY.md).
