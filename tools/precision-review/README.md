# Precision driver

## Row precision and complete source review

`just issue-32-check` protects the exact chunk/tail witness from
[issue #32](https://github.com/imazen/linear-srgb/issues/32). To reproduce a
committed source revision in an isolated archive, use
`just issue-32-probe REVISION NEW_DURABLE_DIRECTORY`; a historical mismatch
intentionally exits nonzero and retains its source, lock and stdout.

`just expand-whole-crate ARCHIVED_PR_SOURCE NEW_DIRECTORY` uses the PR's
pinned resolution anchors and feature guard, but exports the entire compiled
library without an ITEM filter. `just verify-whole-crate DIRECTORY` validates
the full six-configuration matrix and lossless source parts. See the
[complete review exports](../../docs/whole-crate/README.md) and
[portability/precision assessment](../../docs/predictable-math.md) for measured
results and coverage limits. These checks do not alter numerical policies or
bless changed output bits.

## Released/current precision driver

Run from the repository root under a resource wrapper:

```sh
run-heavy --mem 16G --jobs 10 -- bash tools/precision-review/run.sh /mnt/v/output/precision-fresh ../garb
```

The output directory must be new. It contains exact dependency locks, generated
source, released sample bytes and current native sample bytes. Do not commit
these bulk artifacts. `generate.py` reads the committed public-API snapshots
including hidden conversion helpers; unsupported signatures are reported.
The oracle compares both versions independently with f64 formulae.

All integer codes are exercised where a one-channel input permits it. Float
inputs combine a uniform grid, dark log grid, junction neighborhoods,
out-of-range values, infinities and NaN. Generic conversion traits use a
representative RGB(A) mapping; named wrappers exercise their own mappings.
This is not exhaustive over every image, exponent, generic table size, or
combination of RGB channel values.

The driver compares native, V3 and forced-scalar dispatch on x86-64, and NEON
and forced-scalar on ARM. It sets parent token state before descendants.
Direct hooks run only where their required native token is available; ARM
uses the x4 NEON hooks. Float row splitting preserves complete RGBA pixels.
NaN sign/payload differences are reported as bit differences. Integer code
error and f32 ULP error must be distinguished from raw byte differences in
the generated tables.

`inventory.py WORKSPACE OUTPUT.json` is a separate read-only source census.
Its class labels are conservative recommendations, not evidence that two
implementations can be substituted. Review source contracts and reference
oracles before consolidating. Keep its output outside a public repository
because the workspace can contain private projects.

`summarize.py BEFORE AFTER OUTPUT.tsv` compares native captures in numeric
units. When AFTER contains `__inputs.bin`, it also reports finite output
changes for original scalar/slice inputs in [0,1]; garb's differently shaped
byte workloads are excluded from that input-domain inference. New probes
have no baseline and remain labeled new. NaN signs/payloads are counted
separately; packed-word changes are not per-channel errors.
