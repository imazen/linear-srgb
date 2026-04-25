# API surface snapshots

Frozen, line-by-line dumps of the `pub` items in this crate, captured by
[`cargo-public-api`](https://github.com/cargo-public-api/cargo-public-api).
CI re-runs `cargo public-api` on every PR and `diff`s against the files
here — any added, removed, or changed `pub` item makes the diff non-empty
and fails the build.

This is the **gate** that catches unintentional public-API changes during
internal refactors (issue #23 and beyond). It is intentionally strict:
adding a `pub` item is just as much a "diff" as removing one.

## Files

| File | Feature set | Notes |
|---|---|---|
| `default-features.txt` | `--features std,avx512` | The published default. Most downstream users see exactly this surface. |
| `all-features.txt` | `--all-features` | Maximum surface (`std,avx512,alt,iec,transfer,unsafe_simd`). Catches additions hidden behind feature flags. |

## Regenerating

When you intentionally change the public API, regenerate both files in
the same PR so the diff lands alongside the code:

```bash
cargo public-api --simplified --features "std,avx512" 2>/dev/null > api-snapshots/default-features.txt
cargo public-api --simplified --all-features        2>/dev/null > api-snapshots/all-features.txt
```

`cargo-public-api` internally invokes nightly `rustdoc` to emit JSON, so
you need a nightly toolchain installed (`rustup toolchain install nightly`).
The active toolchain can stay on stable.

## Companion gate: `cargo semver-checks`

The same workflow also runs `cargo semver-checks` against the published
crate on crates.io. `public-api` catches *any* surface change; semver-checks
catches *breaking* surface changes specifically. They overlap on removals
and signature changes; they differ on additions (public-api flags them as
diffs requiring snapshot update; semver-checks accepts them as minor-bump
material). Both must pass.

## Why this matters

Issue #23's premise is "zero public API changes required" while collapsing
~750 lines of duplicated SIMD dispatch code. Without this gate, a stray
`pub` on a helper or a renamed re-export would slip in unnoticed. With
this gate, any such drift surfaces as a snapshot mismatch in code review.
