# Public API snapshots

Each target has independent supported, optional-feature and hidden-API captures:

| Directory | Feature entry points | Exact replacements across features |
| --- | --- | --- |
| `x86_64-unknown-linux-gnu/` | x4/x8 V3, plus existing scalar operations | 64 |
| `aarch64-unknown-linux-gnu/` | x4 NEON, plus existing scalar operations | 28 |
| `wasm32-wasip1/` | x4 SIMD128, plus existing scalar operations | 28 |

These are rustdoc captures compiled for each target, not inferred aliases or a
union that could hide a missing platform function. Every target is checked
against its own deprecated x4/x8 signatures. Public x16 replacements and extra
tier families are rejected. A target's three files include the feature delta
and hidden surface as well as its default public surface.

Install the targets for the pinned renderer nightly first:

```sh
rustup target add --toolchain nightly-2026-07-13 \
  aarch64-unknown-linux-gnu wasm32-wasip1 x86_64-unknown-linux-gnu
just api-doc       # regenerate all three
just api-doc-check # check all three, including the exact replacement guards
```

CI runs a separate snapshot-check job for each target. It sets
`ZEN_API_DOC_TARGET`, `ZEN_API_DOC_TOOLCHAIN` and `ZEN_API_DOC=check` explicitly.
The runner requires a target; missing captures and incorrect architecture
labels fail the guard. `scripts/test_target_features_api.py` mutation-checks
missing replacements, extra functions/modules and wrong-platform captures.

The dev-only renderer dependency is pinned to the zenutils source revision
that adds cross-target rustdoc and architecture-label support. It is outside
the shipped dependency graph. Replace that pin with a released version once
the shared renderer change is published.
