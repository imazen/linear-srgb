# Portable, predictable math: scope checked 2026-11-11

The work has three independent contracts: matching outputs across ISAs,
matching outputs across scalar/array/slice/tail entry points, and mathematical
accuracy/monotonicity. A shared dependency, coefficient table or FMA primitive
does not establish all three.

## Issue #32, reproduced on committed source

[Issue #32](https://github.com/imazen/linear-srgb/issues/32) reports a valid
chunk/tail defect at `8900ffc8` (0.6.12). Isolated release builds reproduced
its exact witness, `[0.125_f32; 17]`, without changing source or constraints:

| Committed source | Body bits | Tail bits | Single-value API bits | Row assertion |
| --- | --- | --- | --- | --- |
| Reported `8900ffc8` | `3c6b22d7` | `3c6b22d8` | `3c6b22d8` | Fails |
| Current main `2becf6c2` | `3c6b22d7` | `3c6b22d7` | `3c6b22d8` | Passes |
| PR #30 `f168120c` | `3c6b22d7` | `3c6b22d7` | `3c6b22d8` | Passes |

All used rustc 1.99.1, baseline target flags, and `std,avx512` on the i270
x86-64 host. The old source resolves archmage/magetypes 0.9.31-beta.1; current
main and PR resolve 0.9.31. Every build retains its actual dependency lock.

The existing fix
[`1ce79069`](https://github.com/imazen/linear-srgb/commit/1ce790690df1583bbde3298777dde5548c239daa)
pads tails and evaluates them through the vector kernel. The new exact-witness
regression checks decode and encode at 0.125 and 0.9 for lengths 0..65.
It preserves the existing row contract and does not change numerical source,
accuracy limits or expected goldens. This check does not assert that public
single-value functions use the same graph as a slice.

That separate distinction remains important: `rational_poly` evaluates its
single-value degree-four rational graph with f64 intermediates, explicitly
to avoid f32 monotonicity violations; the SIMD graph uses f32. Choosing one
graph for a future portable policy changes one of those behaviors. Merely
routing the scalar API through current f32 kernels needs monotonicity and
accuracy validation; widening all batch kernels needs a performance study.

## Implementation cost and runtime choices

- **Row consistency:** the reported historical defect already has a main fix.
  Retain the witness and existing row/tier tests; it does not require changing
  the current default graph or accepting weaker assertions.
- **One result across every entry shape:** choose and name the intermediate
  precision, coefficient bits, operation order, fusion, rounding, clamping,
  special-value and domain contracts. Preserve current guarantees while
  validating the chosen graph; do not silently redefine `precise-portable`,
  which currently changes WASM fusion and has no native effect.
- **Cross-ISA results:** native and precise-WASM common grid hashes already
  agree in the [captured baseline](../goldens/README.md#portability-limits),
  but edge/tail differences remain unclassified. Default strict-WASM also
  has regular-grid differences. Shared ISA-independent expected results
  are needed in addition to per-profile drift fixtures.
- **Portable power and other primitives:** the
  [inventory](transcendentals.md) records current Magemath f32 contracts and
  remaining f64 power needs. Bounded profiles need domain adaptation and
  safe speculative inactive lanes. Scalar/tails must retain the same selected
  graph and caller policy as vector blocks. Preserve existing accuracy gates.
  [Archmage PR #153](https://github.com/imazen/archmage/pull/153) additionally
  supplies source contracts for portable f32/f64 min/max and f32 reciprocal/
  rsqrt, including NaN and ordered-zero policies. That can help resolve the
  exceptional-value differences, but does not choose our curve semantics.
  Its [linear-sRGB inventory](https://github.com/imazen/archmage/blob/36af0f3449a73043a32b58161155ed73a9c518ac/docs/LINEAR-SRGB-INVENTORY.md)
  is explicit that the dedicated P/Q fits have no matching generic power
  replacement, the nearby LOWP log2 changes one coefficient bit, and replacing
  `P/Q` with `P * recip(Q)` adds a rounding step even with HIGHP reciprocal.
- **Performance:** existing native f32 FMA paths supply a useful starting
  graph. Strict/nonfusing WASM needs exact software fusion for the same fused
  graph. A universal f64 graph changes packing and arithmetic costs. This
  investigation measured neither candidate's throughput, code size or build
  cost, so it supplies no slowdown ratio or engineering-time estimate.

The immediate low-risk work is coverage, regression witnesses and specifying
the portable contract. The policy choice and performance/accuracy validation
are the substantive implementation work; this assessment changes no policy.

## Complete crate exports

The selected-module `cargo expand` captures omit `scalar`, `rational_poly`,
`default`, `precise`, LUTs and crate-root code. That is a real coverage gap
for this issue. [Complete exports](whole-crate/README.md) now record all
compiled code for the PR's three targets and two feature presets. Complete
source is losslessly partitioned where necessary and verified with whole
and per-part hashes. The current selected-module CI gate still needs to
consume complete exports to detect changes anywhere in the crate.

## Artifacts and reproduction

Source archives, actual lockfiles, compiler identity, commands, stdout and
exit statuses are preserved on **i270** under
`/mnt/v/output/linear-srgb/predictable-math-2026-11-11/`, checked 2026-11-11.
Subdirectories `reported/`, `main/`, and `pr30/` contain the isolated witnesses;
`whole-crate-versioned/` contains complete raw exports, parts and their
manifest. This host/path is not a claim about copies on other hosts.

```sh
just issue-32-check
just issue-32-probe 8900ffc87cd50b249bae19c99d874431e055d888 "$HOME/tmp/new-issue32-old"
just issue-32-probe origin/main "$HOME/tmp/new-issue32-main"
just issue-32-probe origin/api/target-features "$HOME/tmp/new-issue32-pr"
just expand-whole-crate "$HOME/tmp/new-issue32-pr/source" "$HOME/tmp/new-whole-crate"
```

The historical probe intentionally exits nonzero when body/tail bits differ.
The source archive and lock remain available after that failure. The local
probe adapts the
[upstream isolated-build reproducer](https://github.com/imazen/archmage/blob/d8c9ed4d1d59ab1a996f93a37ea84c83fa55e806/benchmarks/primitive_variants_2026-11-10/linear_original.py),
adding an explicit revision, captured dependency metadata and single-value
output. It archives committed source, even if the working copy has unrelated
edits; it does not incorporate or overwrite those edits.
