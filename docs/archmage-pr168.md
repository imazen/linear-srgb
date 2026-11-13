# Archmage PR #168: portable and backend-dependent transfer math

Reviewed 2026-11-13 against [PR #168 head
94ed05dc](https://github.com/imazen/archmage/tree/94ed05dcd1f307d2b61dc182cf1e0410a3590f24),
linear-srgb main `fe819899` and unchanged PR #30 head `f168120c`.
PR #168 is open. This is an integration assessment, not a dependency update,
accepted numerical change or throughput benchmark.

## What the upstream source now supplies

The important new contribution in #168 is
[`Wasm128RelaxedFmaToken`](https://github.com/imazen/archmage/blob/94ed05dcd1f307d2b61dc182cf1e0410a3590f24/src/tokens/generated/wasm.rs#L417).
It requires a module compiled with SIMD128 and relaxed SIMD, then screens
all f32/f64 madd lanes with a cached runtime fusion probe. Unlike the ordinary
relaxed-SIMD token, it proves single-rounding behavior. Its Magetypes backend
emits bare relaxed madd for both `mul_add_fused`/`mul_add_portable` and
`mul_add_relaxed`; it removes the per-node fusion probe/fallback branch from
our polynomial kernels when dispatch retains that token. On a nonfusing
engine the token is unavailable, and dispatch must select another backend.
This is an engine arithmetic probe, not runtime detection that can make a
relaxed-SIMD binary load on an engine lacking those instructions.

The head also includes the configurable f32 power/exp/log framework already
present in its base. The complete `docs/POWF.md` has the same blob SHA in base
`bd83f034` and PR head; do not attribute the entire framework to #168.
Its [power contract](https://github.com/imazen/archmage/blob/94ed05dcd1f307d2b61dc182cf1e0410a3590f24/docs/POWF.md)
provides:

- BitExact and BackendDependent policies independent of precision, domain,
  checks and dispatch. Native capable tiers can use the same fused graph;
  backend-dependent scalar/strict WASM can use separate multiply/add.
- Matching typed vector and scalar-with-token entries. Retaining the token
  and const flags keeps body/tail arithmetic consistent, including the
  backend-dependent policy.
- **Full-domain portable f32 HIGHP power**, with a canonical binary64 graph
  rounded to f32, specials, subnormals, overflow and underflow. This closes
  the f32 coverage gap left by the bounded Gamma preset. It does not expose
  general f64 power, promise correct rounding, or preserve current std bits.
- MIDP/LOWP bounded power, exp2 and log2 approximations. Checked calls reject
  a whole vector if any lane violates the domain. AssumeValid removes checks
  only when we establish the same preconditions for every evaluated lane.
- BackendDependent HIGHP power delegates to scalar std `f32::powf` and
  requires std. It is not a vectorized fast-power implementation.

The PR adds cbrt, RGB block helpers and neural integer operations too; the
current production math inventory has no cbrt or neural-dot-product caller.
Primitive sqrt/floor/ceil and fused FMA already exist in Magetypes, including
no-std scalar implementations. Migrating our remaining num-traits calls to
those primitives still requires explicit scalar and evaluation policies.

## Remaining work for the two variants

| Area | Portable variant | Fast backend-dependent variant |
| --- | --- | --- |
| sRGB and extended rational fits | Retain dedicated P/Q fits and division; use one declared fused graph and precision | Retain fits; permit relaxed madd where fusion is unavailable; validate the separate graph |
| Public scalar versus batch sRGB | Choose f64-intermediate or f32 evaluation without weakening monotonicity/accuracy; current graph split remains | Backend dependence need not imply position dependence: use one selected graph for body/tail; scalar API precision remains a separate decision |
| Gamma | Bounded MIDP is a close candidate within its admitted domain; Full HIGHP or an explicit fallback is needed outside it | Backend MIDP can use native FMA or separate arithmetic; same domain/fallback work remains |
| Near-black PQ | Full HIGHP handles tiny bases and PQ exponents that Gamma rejects; validate the whole composed curve and quantization | Gamma preset still cannot admit PQ's exponents; use a supported broader-domain policy/fallback and check error and underflow behavior |
| BT.709/HLG log2 and exp2 | Current P/Q approximations are not identical to configured replacements; retain their graphs with portable primitives or review and validate new fits | The same compatibility decision remains; bounded error/monotonicity must hold for each admitted rounding graph |
| NaNs, signed zero and clamping | Define explicit semantics and implement common masks or integrate the separate primitive-policy work | `min_fast`/`max_fast` intentionally have backend-specific NaN/zero behavior; use them only where those freedoms are acceptable or inputs exclude the difference |
| f64 precise/IEC/gamma and runtime generic LUTs | General portable f64 power is still missing | Existing std/libm f64 power can remain backend dependent; it does not establish matching vector bits |
| std/no-std and quantization | Align division versus reciprocal normalization, LUT generation, conversion rounding and alpha handling | These still need a documented contract; unrelated feature/row-dependent changes are not required by a fast policy |

[PR #153](https://github.com/imazen/archmage/pull/153) remains separate and open:
its configurable min/max, reciprocal and rsqrt families are **not in #168**.
Ordinary `min`/`max` remain backend dependent on exceptional values. A
reciprocal approximation is not a bit-preserving replacement for P/Q division.

One integration detail remains upstream or in our adapter:
[`powf::dispatch`](https://github.com/imazen/archmage/blob/94ed05dcd1f307d2b61dc182cf1e0410a3590f24/magemath/src/powf/dispatch.rs#L24)
recognizes exact base token types, including `Wasm128Token`, rather than every
new subtier. MIDP/LOWP typed Caller still evaluates with the supplied token,
so the new FMA backend works there. However,
[`highp_dispatch::for_token`](https://github.com/imazen/archmage/blob/94ed05dcd1f307d2b61dc182cf1e0410a3590f24/magemath/src/powf/highp_dispatch.rs#L121)
with the new FMA token and Caller falls to the scalar graph; Automatic can
redispatch through the parent and its probe. Preserve the parent proof for
this entry or add explicit subtier selection before assuming packed HIGHP
reuses the new proof. This is a source-level execution-path finding, not a
measured speed penalty.

## Local compatibility and execution checks

`magemath_probe.py` prepares an isolated package with path dependencies on
the exact upstream archive. It includes the actual current
`src/tf/fast_math.rs`; it does not mirror its coefficient tables. The checked-in
probe is development-only and changes neither implementation nor goldens.

The power grid contains 16,384 positive inputs `k/16384`, and five exponents
`0.45`, `1/2.2`, `2.2`, `2.4`, `4`. Each row below has **81,920** power
comparisons and asserts that both new policies' vector results equal their
same-token scalar tails.

| Token / host context | Existing graph versus portable MIDP differences | Backend MIDP versus portable MIDP differences |
| --- | ---: | ---: |
| Native V3 | 0 | 0 |
| Native Scalar Caller | 0 | 11,676 |
| WASM SIMD128, fusing engine | 0 | 0 |
| WASM Relaxed FMA token, fusing engine | 0 | 0 |
| WASM SIMD128, forced nonfusing engine | 11,676 | 11,676 |
| WASM Scalar Caller | 11,676 | 11,676 |

The existing WASM graph uses relaxed arithmetic in this probe; its differences
are not a failure of the new portable policy. The FMA token was present on the
fusing engine and absent when FMA was disabled. Scalar Caller explicitly
prevents autonomous upgrades. A fast wrapper must retain the dispatched token
in its tail rather than substituting ScalarToken.

Across all tested contexts, direct portable LOWP substitution changes
**571/16,384** log2 results on `k/16384`, and **16,336/16,384** exp2 results
on `16*k/16384 - 8`. These compare the actual current P/Q helpers with the new
configured helpers; they are compatibility counts, not mathematical errors.
There is no exposed predecessor-fit selector to make these graphs identical.

The probe also verifies rejection by the Gamma preset and acceptance by Full
HIGHP for `(smallest f32 subnormal, 2.4)`, `(0.5, 1/M1)` and `(0.5, M2)`.
Their Full HIGHP bits are respectively `00000000`, `3c53386b`, `180ea43a`.
The current generic gamma API does not restrict gamma to [0,4], and admits
arbitrarily tiny positive inputs, so the sampled matching grid does not make
Gamma a complete replacement. Speculative inactive lanes in PQ/HLG/BT.709
must be given safe admitted inputs before calling restricted checked math.

## Cost and recommendation

The new source supplies the f32 mathematical building blocks for both policies;
writing another portable pow implementation is unnecessary. The largest
remaining decisions are our scalar/batch precision contract, f64 coverage,
curve-level domain/special handling, and accuracy/monotonicity validation.

Use shared private kernels specialized by const arithmetic policy, retaining
our array/slice public boundaries. Portable selects the fixed graph and
guaranteed fusion; backend-dependent selects relaxed madd and explicitly
bounded approximations. Select the token once for a whole row/operation and
keep it for tails. Avoid adding two independent copies of every curve or
exposing Magemath/Magetypes option types in the public API.
Make the two variants operation-level choices if both must coexist in one
program; Cargo feature unification makes a crate-wide feature unsuitable for
selecting different behavior in different downstream callers.

Native FMA-capable paths need not pay software fusion for the portable graph.
Strict/nonfusing WASM and scalar fallback can pay that cost; #168 removes
per-node FMA checking on a proved fusing WASM tier, not the need for exact
fallback elsewhere. HIGHP changes working precision and packing. No
linear-srgb throughput, compilation-time, code-size or engineering-time cost
was measured, so there is no qualified slowdown ratio or time estimate.

The full-crate exports remain frozen review artifacts. Replacing PR #30's
selected-module freshness gate and extending its configuration coverage are
still needed independently of the math migration. This review does not merge
either PR or select new default public numerical behavior.

## Validation and retained evidence

Rust 1.99.1 release runs pass the upstream `contracts` and `powf_highp`
fixtures on native std, native no-std, strict WASM SIMD128 and relaxed WASM
with engine FMA enabled/disabled. The new token's primitive FMA/dispatch test
also passes both engine choices, including process-wide token suppression.
The final isolated compatibility consumer passes native and both relaxed
engine runs, and Clippy with warnings denied. These are targeted checks, not
an independent rerun of upstream's full CI or exhaustive binary pow domain.
No fresh ARM execution was performed in this review.

Sources, actual lockfiles, source hashes, commands and complete logs are kept
on **i270** at
`/mnt/v/output/linear-srgb/archmage-pr168-2026-11-13/`, checked 2026-11-13.
`REVIEW.json` names authoritative logs and records earlier excluded diagnostic
runs; `final-consumer-validation.log` and `validation-simd-explicit.log` are
the final consumer and SIMD contract/token evidence. WASM recipes unset
`CARGO_ENCODED_RUSTFLAGS` so it cannot override the requested SIMD flags.

Prepare with `just magemath-probe-prepare ARCHMAGE_SOURCE NEW_DIRECTORY`, then
run `just magemath-probe NEW_DIRECTORY TARGET FLAGS RUNNER` through
`run-heavy`. See the [probe instructions](../tools/precision-review/README.md#magemath-policy-integration-probe).
The archived source and lockfiles preserve the actual tested resolution;
re-preparing later may resolve newer transitive dependencies and is a new run.
