# Expanded SIMD source

These checked-in `cargo expand` captures show the actual library after macro
expansion. Each target has `default/` and `all-features/` captures of
`target_features`, `tokens`, `simd`, `kernels` and `tf`, including their children.
They expose feature-context entry points, deprecated proof wrappers, dispatch,
vector kernels and transfer math without adding a public API.

Targets match the [public API snapshots](../public-api/README.md): x86-64 Linux,
ARM64 Linux and WASI. Captures use baseline target features: the WASM captures
include SIMD128 target-feature functions but do not enable global `relaxed-simd`.
The all-feature preset includes `precise-portable`; the default preset does not.
Test-only modules and other library modules are outside this capture set.

Like [archmage's expansion tooling](https://github.com/imazen/archmage/blob/91a13739b73f46213ecfe7083c77297c531fc8ad/xtask/site_expansions.py),
the generator pins Rust **1.99.0** and cargo-expand **1.0.126**, typechecks before
expanding, and fails checks on missing, changed or unexpected captures. Headers
record the compiler, target, selected library features, macro-provider versions
and resolution-lock hash. `--check` never rewrites captures.

```sh
rustup toolchain install 1.99.0 --profile minimal --component rustfmt
rustup target add --toolchain 1.99.0 \
  x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu wasm32-wasip1
cargo +1.99.0 install cargo-expand --locked --version 1.0.126
just expand-snapshots       # regenerate, then review the Rust diff
just expand-snapshots-check # typecheck and compare all six target/preset pairs
```

CI checks each target independently with
`python3 scripts/expand_snapshots.py --check --target <triple>`. Python 3.11 or
newer is required. Ambient Rust flags are cleared so a developer's native or
relaxed-SIMD settings cannot silently alter the baseline.

The standalone `expand/` and `expand/all-features/` packages pin archmage,
archmage-macros and magetypes **0.9.31** through committed dependency locks. They
select the real path dependency; neither copies library source. Separate
anchors are needed because Cargo rejects `--features`/`--all-features` when
selecting a dependency package with `--package linear-srgb`. The generator
verifies Cargo's resolved library features against the main manifest, so a
new feature requires updating the all-feature anchor. These packages and the
captures are excluded from the published crate.

When updating a tool or macro provider, update the pins and both resolution
locks, regenerate every target, and review the resulting expansion changes.
These captures complement the signature and assembly snapshots; they do not
prove numerical parity or runtime performance.
