# linear-srgb tango regression bench

Paired-benchmark regression gate that compares the **local WIP** linear-srgb
against the **published crates.io version** (currently `=0.6.11`).

Uses [tango-bench](https://crates.io/crates/tango-bench) so the two versions
run in the *same* process, interleaved microsecond-by-microsecond, and tango
reports a paired A/B percentage per benchmark. This is far more sensitive
than back-to-back cargo bench runs — fractions of a percent are detectable.

## One-time setup

Nothing — just have `cargo export` available (`cargo install cargo-export`).

## Workflow

```bash
# 1. Build the BASELINE binary against the published crate.
#    (Cargo.toml defaults to `linear-srgb = "=0.6.11"` from crates.io.)
cd tango
cargo export target/baseline -- bench --bench regression

# 2. Enable the local WIP by uncommenting the [patch.crates-io] block at
#    the bottom of Cargo.toml:
#        [patch.crates-io]
#        linear-srgb = { path = ".." }
#    (Or: `sed -i 's/^# \[patch/[patch/; s/^# linear-srgb = { path/linear-srgb = { path/' Cargo.toml`.)

# 3. Build + compare against the saved baseline.
cargo bench --bench regression -- compare target/baseline/regression

# 4. Don't forget to re-comment [patch.crates-io] when done (or `git checkout Cargo.toml`).
```

## Interpreting output

Tango prints one line per benchmark:

```
linear_to_hlg_uniform_4096    [ 2.44µs ... 2.55µs ]   +4.5%*
```

- Left = WIP timing, right = baseline timing.
- Sign on the delta is relative to baseline; `*` means the 95% CI excludes zero.
- **Positive = WIP slower = regression.** Negative = WIP faster.

## Benchmarks covered

All six transfer-function slice dispatchers:
- `linear_to_hlg`, `hlg_to_linear`
- `linear_to_bt709`, `bt709_to_linear`
- `linear_to_pq`, `pq_to_linear`

Four input distributions × three sizes (256, 4096, 8192):
- `uniform` — evenly distributed across [0, 1]
- `small` — all values in [0, 1/16], forces the quadratic / linear / small-poly branch
- `large` — all values in [0.2, 1.0], forces the log / sinh / large-poly branch
- `hdr_luma` — cube of uniform, approximates HDR luma distribution (~44% below HLG split)

The size axis is there to catch per-call SIMD setup dominating small slices
(issue #18 reported a 30% regression at 256 that narrowed to 12% at 1080p
as memory bandwidth took over).
