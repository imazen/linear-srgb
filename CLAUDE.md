# linear-srgb — Project Notes

## Current transition

See [docs/SIMD_TRANSITION.md](docs/SIMD_TRANSITION.md) for the branch, issue and TODO inventory checked 2026-11-05. PR #30 adds `target_features` and deprecates the retained `tokens` module; it is open, not a published release. Ordinary code uses `default`; SIMD contexts use arrays/slices without exchanging dependency tokens. The owner approved removal of consumer token reexports and token-taking operation surfaces in the next version. Keep the current additive change compatible.

## Known Bugs

- Resolved during the ownership refactor: default WASM scalar HDR entry points and u16 tails used fused arithmetic while SIMD blocks used relaxed arithmetic. `tests/target_features_dispatch.rs` and `tests/remaining_parity.rs` expose the mismatch; `just verify-target-features-wasm-default` runs both without `precise-portable`. The fix aligns evaluation without changing assertions.

- Resolved in the wider-dispatch change: x8/x16 token hooks missed the exact-white endpoint rescue already present at x4. Cross-width dispatch now checks bit equality, including values at and above 1.0; the wider hooks blend those lanes to exactly 1.0.

- [Issue 29](https://github.com/imazen/linear-srgb/issues/29): reported LLVM failure on soft-float `x86_64-unknown-none`. Normal-target no-std checks do not resolve this; no new reproduction was run during the documentation inventory.

## Historical assembly drift

The May/June WASM drift and October ARM/WASM drift remain historical reports, not current “every push is red” claims. Both target checks pass on main `1340e864` ([run 41986870385](https://github.com/imazen/linear-srgb/actions/runs/41986870385)) and API PR #30 ([run 41985914165](https://github.com/imazen/linear-srgb/actions/runs/41985914165)). [Issue 28](https://github.com/imazen/linear-srgb/issues/28) retains the old evidence. Do not infer its cause from current green checks, regenerate expected snapshots blindly, or weaken the existing gates.

The original toolchain/codegen observations are retained in [docs/ASM_HISTORY.md](docs/ASM_HISTORY.md).
