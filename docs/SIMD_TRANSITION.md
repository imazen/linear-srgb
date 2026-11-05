# SIMD API transition and work inventory

Observed 2026-11-05. Main: `1340e864`; migration: [PR #30](https://github.com/imazen/linear-srgb/pull/30), open, with all checks green at `d8ae85be`. The new API is not in the crates.io 0.6.12 release. Re-resolve branch and issue state before acting on this inventory.

## Calling convention

Ordinary code uses `linear_srgb::default`. Code already compiled for a CPU tier uses `target_features::{x4,x8,x16}`. Keep the function name and omit the old token argument:

```text
old: tokens::x8::srgb_to_linear_v3(token, values)
new: target_features::x8::srgb_to_linear_v3(values)
```

The caller still needs matching target features. A runtime check alone does not establish a native compiler context; use a feature function or an archmage entry point. Native E0133 is a context diagnostic, not an instruction to insert an unsafe block. WASM validates SIMD when loading the module and permits ordinary safe calls. See [the API's calling-context example and complete feature sets](../src/target_features/mod.rs).

Token/vector types stay internal to the new boundary. Keep the deprecated `tokens` signatures for compatibility; no removal version has been approved. Do not restore the public magetypes-vector signatures proposed by old branches. Arrays do not guarantee register-only passing: inspect assembly and measure before claiming overhead savings.

Validation recipes: `just verify-target-features`, `just verify-target-features-native-v4`, `just verify-target-features-portable`, `just api-doc-check`, and `just check-target-features-surface`. Required-tier runs fail when their specified tier is unavailable. Snapshots alone are insufficient: inspect bare token reexports and architecture-gated declarations too.

## Branch inventory

All 14 live remote branches are accounted for below. “Ancestor” is checked against the observed main; divergent history is not proof of missing functionality. Preserve branches until their residual changes have been reviewed.

| Branch / tip | Evidence and disposition |
| --- | --- |
| `main` / `1340e864` | Current source baseline. |
| `api/target-features` / `d8ae85be` | Four migration commits, one main commit behind. Review PR #30; approval/merge and a separately approved release remain. |
| `feat/adobe-rgb-romm-trcs` / `a396300c` | One unique commit adding Adobe/ProPhoto functions. [Issue #7](https://github.com/imazen/linear-srgb/issues/7) was deliberately closed as outside the motivating Ultra HDR work. Preserve as scoped-out work, not an approved implementation backlog. |
| `feat/extended-range-simd` / `95cbcd1e` | [PR #6](https://github.com/imazen/linear-srgb/pull/6) merged; tip has two later documentation/accuracy-test commits not in main. `extended_simd_doc_accuracy_claims` is absent from current tests. Review those residual assertions against today's coefficients before porting; do not import old expected numbers blindly. |
| `feat/pub-tf-x16` / `f6dffe2d` | One unique commit exposes generic magetypes TF kernels and adds gamma kernels. Current array API already covers gamma at x4/x8/x16; retain any needed algorithm internally rather than merging the old public type coupling. No numerical equivalence claim was measured here. |
| `feat/simd-extended-srgb` / `ef5a71c9` | [PR #5](https://github.com/imazen/linear-srgb/pull/5) closed unmerged; five divergent commits. Extended-range operations already exist on main. Compare any residual tests/content before reuse; do not replay the historical implementation wholesale. |
| `feat/simd-u16-decode` / `7ab2e617` | Ancestor; [PR #21](https://github.com/imazen/linear-srgb/pull/21) merged. |
| `perf/simd-extended-magetypes` / `ca95391c` | Ancestor; [PR #17](https://github.com/imazen/linear-srgb/pull/17) merged. |
| `perf/simd-f32-to-int-quantize` / `9a2dcc38` | Ancestor; [PR #13](https://github.com/imazen/linear-srgb/pull/13) merged. |
| `perf/simd-gamma-magetypes` / `b399c180` | Ancestor; [PR #15](https://github.com/imazen/linear-srgb/pull/15) merged. |
| `perf/simd-tf-magetypes` / `ce9b0ce6` | Ancestor; [PR #16](https://github.com/imazen/linear-srgb/pull/16) merged. |
| `perf/tokens-x4-magetypes` / `736b5dea` | Ancestor; [PR #14](https://github.com/imazen/linear-srgb/pull/14) merged. |
| `polyfit-coefficients` / `04f0e1f0` | Ancestor; [PR #8](https://github.com/imazen/linear-srgb/pull/8) merged. |
| `tooling/tango-regression` / `67329f35` | Ancestor; [PR #19](https://github.com/imazen/linear-srgb/pull/19) merged. |

Additional local refs: `abandoned/pulp-rewrite` (`414e59ab`, five unique commits) is historical experimental work, not a migration dependency. `feat/tf-pu-encoding` (`f403168e`) belongs to closed [PR #26](https://github.com/imazen/linear-srgb/pull/26). Its proposed move to `zenpixels-convert::hdr_iqa` was subsequently superseded by the [zenpixels #34 rescope](https://github.com/imazen/zenpixels/issues/34); [zenpixels PR #35](https://github.com/imazen/zenpixels/pull/35) closed unmerged. Do not describe that proposal as a current zenpixels API or resurrect PU in this foundational crate without a new scope decision.

## Issues, TODOs and remaining transitions

- [#29](https://github.com/imazen/linear-srgb/issues/29): open soft-float `x86_64-unknown-none` LLVM failure. Keep it separate from the successful `--no-default-features` check on a normal target. This inventory did not re-run the failing target.
- [#28](https://github.com/imazen/linear-srgb/issues/28): historical assembly drift. Both ASM jobs now pass on [main `1340e864`](https://github.com/imazen/linear-srgb/actions/runs/41986870385) and [PR #30](https://github.com/imazen/linear-srgb/actions/runs/41985914165). Current green checks do not establish the old failure's cause or justify changing expectations.
- Closed #25/#7 represent explicit scope decisions described above. The remaining 13 closed issues are historical completed/resolved work (#1, #2, #3, #4, #9, #10, #11, #12, #18, #20, #22, #23, #24); their closed state is not a fresh verification of every historical claim. Query `gh issue list --state all` when revisiting one.
- `Cargo.toml` still says `TODO: remove in 0.7` for the no-op `unsafe_simd` feature. This is a queued proposal, not approval to remove a public feature or make a breaking release. No other TODO/FIXME/HACK/XXX markers were found in the inspected source, tests, examples, docs and scripts.
- After API review, migrate consumers such as zenresize/zenjpeg only in separately authorized changes. Confirm each resolved dependency includes PR #30 before replacing calls; preserve per-tier numerical behavior and row/stride contracts. An old published version does not have `target_features`.
- Release still needs README/API review, normal release validation and explicit publication approval. This work did not publish, delete branches or weaken tests.
