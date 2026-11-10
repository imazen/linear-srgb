# SIMD API transition and work inventory

Observed 2026-11-05. Main: `1340e864`; migration: [PR #30](https://github.com/imazen/linear-srgb/pull/30), open, with all checks green at `d8ae85be`. The new API is not in the crates.io 0.6.12 release. Re-resolve branch and issue state before acting on this inventory.

## Calling convention

Ordinary code uses `linear_srgb::default`. Code already compiled for a CPU tier uses `target_features::{x4,x8}`. The public surface replaces only `tokens::x4` and `tokens::x8`. `tokens::x16` is deprecated without a direct replacement. Keep the x4/x8 function name and omit the old token argument:

```text
old: tokens::x8::srgb_to_linear_v3(token, values)
new: target_features::x8::srgb_to_linear_v3(values)
```

The caller still needs matching target features. A runtime check alone does not establish a native compiler context; use a feature function or an archmage entry point. Native E0133 is a context diagnostic, not an instruction to insert an unsafe block. WASM validates SIMD when loading the module and permits ordinary safe calls. See [the API's calling-context example and complete feature sets](../src/target_features/mod.rs).

Token/vector types stay internal to the new boundary. Keep deprecated `tokens` signatures in the current additive change. Their removal and removal of token reexports are approved for the next version. Do not restore the public magetypes-vector signatures proposed by old branches. Arrays do not guarantee register-only passing: inspect assembly and measure before claiming overhead savings.

Validation recipes: `just verify-target-features`, `just verify-target-features-native-v4`, `just verify-target-features-portable`, `just api-doc-check`, and `just check-target-features-surface`. Required-tier runs fail when their specified tier is unavailable. Snapshots alone are insufficient: inspect bare token reexports and architecture-gated declarations too.

## Branch inventory

The initial capture contained 14 live remote branches, accounted for below. A later capture also found `api/portable-scalar-math` (`0482d5c4`), preserved after PR #31 was superseded. “Ancestor” is checked against the observed main; divergent history is not proof of missing functionality. Preserve branches until their residual changes have been reviewed.

| Branch / tip | Evidence and disposition |
| --- | --- |
| `main` / `1340e864` | Initial inventory baseline; current precision work through `24a87677` is merged into PR #30. |
| `api/target-features` | PR #30 keeps math crate-private and includes wider f32 dispatch and main's precision fixes through `24a87677`. Review/merge and a separately approved release remain. |
| `feat/adobe-rgb-romm-trcs` / `a396300c` | One unique commit adding Adobe/ProPhoto functions. [Issue #7](https://github.com/imazen/linear-srgb/issues/7) was deliberately closed as outside the motivating Ultra HDR work. Preserve as scoped-out work, not an approved implementation backlog. |
| `feat/extended-range-simd` / `95cbcd1e` | [PR #6](https://github.com/imazen/linear-srgb/pull/6) merged; tip has two later documentation/accuracy-test commits not in main. `extended_simd_doc_accuracy_claims` is absent from current tests. Review those residual assertions against today's coefficients before porting; do not import old expected numbers blindly. |
| `feat/pub-tf-x16` / `f6dffe2d` | One unique commit exposes generic magetypes TF kernels and adds gamma kernels. Private implementations already cover gamma at x4/x8/x16; retain any needed algorithm internally rather than merging the old public type coupling. No numerical equivalence claim was measured here. |
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

## Public replacement scope and private math

The public `target_features::{x4,x8}` functions correspond one-for-one to the
existing token functions, omitting the token argument. `x8` remains x86-64
only. The user explicitly chose no direct replacement for `tokens::x16`.
Wider kernels and fallback functions needed by internal dispatch stay private;
test-only x8 helpers are not compiled into normal library builds. Integer
conversion hooks retain their existing architecture coverage.

The x86 snapshots contain exactly 64 public replacement signatures across
default and optional feature sets (28 x4 and 36 x8), with no additional public
function family. Their default-feature module summary is 34 items (32 functions
and two width modules), down from 123 items in the broader proposal. The
`tokens` summary remains 52 items in that default-feature capture, because its
compatibility paths and token reexports remain. `just check-target-features-surface`
checks exact set equality, rejecting extras as well as missing replacements.

Caller-owned dispatch can combine the available public tier functions with the
caller's own fallback. See `tests/target_features_dispatch.rs`. Numerical
compatibility for every retained entry and wider internal dispatch remains in
`src/target_features/compatibility_tests.rs` and `dispatch_tests.rs`; those
assertions moved into unit tests so private implementations remain covered.

Scalar generic vectors and `mul_add_fused` already exist in published beta.2,
whose midp transcendentals use relaxed evaluation. Keep the log/exp/pow copies
private to this crate: `src/lanes.rs` owns scalar copies and
`src/tf/fast_math.rs` owns vector copies. Their helpers/constants are private or
`pub(crate)`, with no new public math surface. Preserve native fused rounding,
default-WASM relaxed evaluation and the `precise-portable` contract. Exact
parity tests keep the copies aligned.

Use published archmage/magetypes beta.2 under the existing version constraints;
there is no git pin and no dependency on new magetypes methods or a release.
The proposed public linear-srgb companions in
[PR #31](https://github.com/imazen/linear-srgb/pull/31) remain closed. The
shared-math proposal in [archmage PR #140](https://github.com/imazen/archmage/pull/140)
is also superseded by the owner's decision to retain crate-private copies.
Both branches preserve their implementation/measurement evidence. The restored
private kernels retain their numerical policy; the ownership refactor updates
only the eight assembly expectation pairs explicitly approved on 2026-11-10.

## Entry-point ownership (2026-11-10)

`target_features::{x4,x8}` is the public canonical CPU-feature API; internal
wide entries remain private. Its entry points construct
internal proofs with `from_context()` and call private kernels; they do not
perform runtime CPU detection. Deprecated `tokens` functions forward to this
API. Removing that compatibility surface later will not remove numerical code.

`src/kernels/float.rs` defines the shared sRGB and custom-gamma algorithms for
x4/x8/x16. `src/kernels/{x4,x8,x16}.rs` retains private native integer and
transfer implementations and their existing unit coverage. Transfer math in
`src/tf` remains private to the implementation. Tokens and magetypes vectors
are permitted inside these private modules.

Ordinary f32 slice dispatch selects the canonical feature functions. RGBA,
premultiplication and specialized integer loops keep their existing arithmetic
and use private kernels where applicable. Shared slice helpers process full
blocks directly and pad only the final partial block. Scalar slice entry
points retain the original per-sample fused evaluation on native and
`precise-portable` targets. Default WASM scalar entry points and u16 slice
tails use the relaxed generic kernels, matching vector blocks at every row
length. `just verify-target-features-wasm-default` exercises this policy with
the existing dispatch and row-parity assertions. Width-generic native
sRGB/gamma and HDR implementations use their matching backend rather than
requiring every wider entry to compose x4 calls.

`just check-target-feature-ownership` checks the dependency direction.
`just ownership-fingerprint` and `just ownership-fingerprint-check <log>`
compare outputs with captures made before the refactor. The original outputs,
commands and validation limitations are in
[the ownership baseline](../benchmarks/entry_point_ownership_2026-11-10.md).
The legacy/new adapter comparisons remain useful for compatibility; they are
not independent numerical oracles after both APIs share the implementation.


## Remainders and FMA review (2026-11-10)

The ownership refactor moved the existing zero-padded `tail_block` and
`store_tail` helpers from `simd.rs` into private `kernels`; it did not invent
that tail strategy. `map_slice` factors out the same full-block plus padded-tail
loop. The follow-up did change default-WASM u16 tails from scalar-lane evaluation
to padded vector evaluation, and default-WASM scalar f32 slices from per-sample
loops to their array kernels. Native/precise-portable scalar loops retain their
previous evaluation. The private module also extracts and deduplicates math;
that consolidation goes beyond the minimum needed for a token-free public API.

`tf::fast_math::Mla` already delegates to `mul_add_fused` on native and
precise-portable builds, and `mul_add_relaxed` on default WASM. Both calls are
magetypes operations. In a build with `+relaxed-simd`, magetypes' fused WASM
implementation probes the engine's rounding behavior, caches the result and
uses relaxed madd only when it fuses; otherwise it computes fused results in
software. Default WASM's relaxed call does not request that guarantee. Enabling
relaxed SIMD is a module compilation choice, distinct from the engine probe.
Sources: [published magetypes FMA contract](https://docs.rs/magetypes/0.9.31/magetypes/simd/generic/struct.f32x4.html#method.mul_add_fused)
and [the engine probe](https://github.com/imazen/archmage/blob/main/magetypes/src/wasm_fma.rs).

The previous validation used `+simd128`, without `+relaxed-simd`. A new probe
of numerical source `69eb17ca`, using the published beta.2 dependency floor,
executes the existing compatibility, dispatch, row and tier tests with
`+simd128,+relaxed-simd`:

| precise-portable | Wasmtime `cranelift-has-fma` | Selected test command |
| --- | --- | --- |
| off | true | Fails sRGB scalar/SIMD array equality at target_features_dispatch.rs:46; later tier tests are not reached |
| off | false | Passes |
| on | true | Passes |
| on | false | Passes |

Reproduce with `just probe-relaxed-wasm true std,transfer`, or use `false` for
nonfusing engine code generation. Add `,precise-portable` to the feature argument
for the guaranteed-fused policy. The failing assertion remains intact. The
scalar relaxed backend multiplies and adds separately while the SIMD relaxed
backend can fuse, so exact cross-backend parity is not established for default
WASM on fusing engines. This is an open correctness/coverage finding, not a
successful validation of that configuration.

The review adds only a reproduction recipe and documentation. It does not
change numerical policy or undo the user-directed ownership inversion. The
immediate deliverable remains token-free feature-context array/slice APIs,
deprecated compatibility entry points, and verified replacement contracts.
The public surface was subsequently reduced to direct x4/x8 replacements at
the user's request, with no public x16 module.
Further consolidation should keep curve evaluation, buffer traversal and
arithmetic policy separate, with tokens/vector types confined to private code.
A default-fused WASM policy would use the existing magetypes engine probe but
also require software fusion when the engine cannot fuse; that policy and its
cost need a separate decision from public API ownership.

Logs and the four-cell result table are stored under `relaxed-wasm-review/` in
the ownership artifact directory documented in the baseline, with SHA256SUMS.


## Snapshot tooling

`just api-doc`, `just api-doc-check` and `just ownership-api` default to the
installed `nightly-2026-07-13` toolchain, which renders the existing three
const-generic auto-trait expectations correctly with zenutils-apidoc 0.1.1.
The tracking nightly's negative-auto-trait discrepancy described in the
ownership baseline does not occur on this toolchain. Override
`ZEN_API_DOC_TOOLCHAIN` to inspect another nightly; no expected traits were
weakened to update the narrowed API snapshots.


## Per-target API inventories (2026-11-10)

API captures are now independent for x86_64-unknown-linux-gnu,
aarch64-unknown-linux-gnu and wasm32-wasip1. Their default, optional-feature
and hidden surfaces live in target directories under `docs/public-api/`.
The exact replacement check runs per target: 64 x4/x8 replacements on x86-64,
28 x4 replacements on ARM64, and 28 x4 replacements on WASM. These counts
include each target's optional transfer feature additions. The prior 64-entry
count covered x86 only; host snapshots did not inventory NEON/WASM exports.

`just api-doc` regenerates all three and `just api-doc-check` checks all three.
CI explicitly selects each target in a separate matrix job. Guard mutation
checks reject missing/extra functions and a host capture substituted for ARM.
See [the snapshot guide](public-api/README.md) for setup and file meanings.
The dev-only renderer uses a pinned zenutils revision adding actual cross-target
rustdoc builds and target-derived architecture labels; no shipped dependency
or numerical implementation changes for this tooling update.


## Feature contexts and arithmetic policy

Direct feature functions now use `attune`, including the retained compatibility
helpers that establish a feature context. Scalar functions have no feature
context to establish: they are ordinary inlined functions, without
`rite(scalar)`. Generic families that formerly combined `magetypes` and `rite`
use explicit `attune` family selectors. Runtime capability-proof entry points
remain separate from these direct feature functions.

The existing arithmetic choice is centralized in
`tf::fast_math::USE_FUSED_MATH`: native uses guaranteed fusion; default WASM
uses relaxed multiply-add; WASM with `precise-portable` requests guaranteed
fusion. `Mla`, scalar-array/slice evaluation and u16 tails consult this same
constant. Target wrappers and generated helpers no longer carry independent
WASM/precision cfg branches, and the fused scalar implementations remain
available as private helpers on every target.

The guaranteed-fused path delegates engine probing and software fallback to
magetypes. Enabling `precise-portable` does not itself enable WASM relaxed-SIMD
instructions; those remain a compilation target feature. This cleanup preserves
the existing numerical policy, including the separately recorded default
relaxed-SIMD scalar/vector parity finding. It does not make relaxed arithmetic
promise guaranteed fusion.


Some deprecated token functions are proof-taking wrappers callable from ordinary
code and usable as `Fn` callbacks. Those use `attune` proof outputs, with explicit
names preserving the published `_v3`/`_neon`/`_wasm128`/`_scalar` paths. They must
not become direct feature functions merely because their names have a tier
suffix. The token-free replacements intentionally establish direct feature
contexts. Textual signature inventories cannot detect this distinction;
semver checks and the consumer callback compile tests protect it.
