# linear-srgb — Project Notes

## Current transition

See [docs/SIMD_TRANSITION.md](docs/SIMD_TRANSITION.md) for the branch, issue and TODO inventory checked 2026-11-05. PR #30 adds `target_features` and deprecates the retained `tokens` module; it is open, not a published release. Ordinary code uses `default`; SIMD contexts use direct `target_features::x4`/`x8` array/slice replacements without exchanging dependency tokens. `tokens::x16` is deprecated without a direct public replacement; extra internal width/tier helpers stay private. The owner approved removal of consumer token reexports and token-taking operation surfaces in the next version. Keep the current additive change compatible.

## Known Bugs

- Open: default WASM built with `+relaxed-simd` fails existing scalar/SIMD dispatch equality on a fusing Wasmtime engine (originally `tests/target_features_dispatch.rs:46` at numerical source `69eb17ca`; now `src/target_features/dispatch_tests.rs`). `precise-portable` passes with fusion on and off. Reproduce with `just probe-relaxed-wasm true std,transfer`; see the remainder/FMA review in [docs/SIMD_TRANSITION.md](docs/SIMD_TRANSITION.md). Strict-SIMD CI does not cover this configuration.

- Resolved during the ownership refactor: default WASM scalar HDR entry points and u16 tails used fused arithmetic while SIMD blocks used relaxed arithmetic. `src/target_features/dispatch_tests.rs` and `tests/remaining_parity.rs` expose the mismatch; `just verify-target-features-wasm-default` runs both without `precise-portable`. The fix aligns evaluation without changing assertions.

- Resolved in the wider-dispatch change: x8/x16 token hooks missed the exact-white endpoint rescue already present at x4. Cross-width dispatch now checks bit equality, including values at and above 1.0; the wider hooks blend those lanes to exactly 1.0.

- [Issue 29](https://github.com/imazen/linear-srgb/issues/29): reported LLVM failure on soft-float `x86_64-unknown-none`. Normal-target no-std checks do not resolve this; no new reproduction was run during the documentation inventory.

## Historical assembly drift

The May/June WASM drift and October ARM/WASM drift remain historical reports, not current “every push is red” claims. Both target checks pass on main `1340e864` ([run 41986870385](https://github.com/imazen/linear-srgb/actions/runs/41986870385)) and API PR #30 ([run 41985914165](https://github.com/imazen/linear-srgb/actions/runs/41985914165)). [Issue 28](https://github.com/imazen/linear-srgb/issues/28) retains the old evidence. Do not infer its cause from current green checks, regenerate expected snapshots blindly, or weaken the existing gates.

The original toolchain/codegen observations are retained in [docs/ASM_HISTORY.md](docs/ASM_HISTORY.md).
