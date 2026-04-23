# TF slice dispatch A/B benchmark — issue #10 fix

**Date:** 2026-04-23
**Commit:** 54e03b4e (pre-bench) → (post-bench with this file committed)
**Host:** AMD Ryzen 9 7950X, Linux WSL2 6.6.87.2, AVX-512 available
**Build:** `cargo bench --features transfer --bench tf_bench`
**Rust:** default toolchain (no `-C target-cpu=native`)
**N:** 10000 f32 values per run
**Bench harness:** zenbench via criterion-compat

## Question

Does the issue #10 fix (wiring `default::{bt709,pq,hlg,...}_*_rgb_slice` through
`incant!` instead of scalar-loop + `#[archmage::autoversion]`) actually
reach the SIMD rites at the expected throughput, or is autoversion's
autovectorizer already close enough to make the change cosmetic?

## Setup

Bench file added `bench_tf_public_dispatcher` with A/B pairs:

- `old_scalar_loop`: replicates the pre-fix function body in the bench file
  (`for v in values { *v = tf::X(*v); }` under `#[archmage::autoversion]`).
- `new_incant_dispatch`: calls the current `default::X_rgb_slice` function,
  which dispatches via `incant!` over `[v4, v3, neon, wasm128, scalar]`.

Baseline numbers included for context:

- `tf_scalar`: per-value scalar calls in a plain loop (no target_feature).
- `tf_x8`: direct call to `tokens::x8::X_slice_v3` inside a manual
  `#[arcane]` wrapper — the ideal "if autoversion did nothing wrong" ceiling.

## Results

All numbers in microseconds per 10000-element run. Lower is better.

| Transfer function   | tf_scalar | old (autoversion'd loop) | **new (incant!)** | tf_x8 rite direct |
|---------------------|----------:|-------------------------:|------------------:|------------------:|
| bt709_to_linear     |    143.4  |                     12.7 |           **5.4** |               5.9 |
| linear_to_bt709     |    135.9  |                     11.5 |           **4.6** |               5.4 |
| pq_to_linear        |    114.7  |                     15.4 |           **3.0** |               3.4 |
| linear_to_pq        |    131.7  |                     27.5 |           **5.2** |               5.5 |
| hlg_to_linear       |     50.8  |                      7.9 |           **3.0** |               3.2 |
| linear_to_hlg       |     51.0  |                      2.6 |           **3.4** |               3.4 |

Speedup (old → new), on this data:

- bt709_to_linear: **2.35×**
- linear_to_bt709: **2.50×**
- pq_to_linear:    **5.13×**
- linear_to_pq:    **5.29×**
- hlg_to_linear:   **2.63×**
- linear_to_hlg:   **0.76×**  (⚠ regression on this data; see below)

## Caveat: linear_to_hlg regression on this benchmark data

The new dispatcher's 3.4µs for `linear_to_hlg` matches the direct `tf_x8`
rite exactly (also 3.4µs), so the dispatch machinery is doing its job.
The "regression" is that the old autoversion'd scalar loop (2.6µs)
*out-ran the x8 rite itself* on this particular data distribution:

- `make_linear()` is `make_encoded().map(srgb_to_linear)` — an sRGB-encoded
  0..1 ramp put through `srgb_to_linear`. The result is linear values
  heavily concentrated near 0 (sRGB EOTF is sublinear near 0).
- In `linear_to_hlg`, the scalar takes the cheap `sqrt(3v)` branch when
  `v ≤ 1/12 ≈ 0.083`. On this data distribution, that branch wins ~30%+
  of the values, and the branch predictor catches the pattern.
- The SIMD rite in `src/tf/hlg.rs:linear_to_hlg_x8` always computes both
  the sqrt and the `fast_log2f_x8` branches and blends. On inputs where
  most lanes would have hit the cheap branch in scalar, the rite wastes
  work on the expensive `fast_log2f_x8` it then throws away.

For realistic linear-light HDR content being encoded to HLG (values
spread across the full 0..1 range, not concentrated near 0), the rite
would win as expected. But on this bench's synthetic input, the branchy
scalar beats the always-both-branches SIMD.

**This is not a dispatcher bug.** The dispatcher correctly reaches the
rite. It's a rite-quality observation worth filing separately: the
linear_to_hlg rite could be improved by branching on `simd_all(mask)`
to skip the expensive branch when no lane needs it, at the cost of
adding a runtime test per chunk.

## RGBA variants (added same day, issue #2 followup)

N=10000 **pixels** (40000 f32 elements). Baseline is the RGBA fallback a
caller would have written before the `_rgba_slice` TF functions existed:
`for px in chunks_exact_mut(4) { px[0..3] = scalar(px[0..3]); }` under
`#[archmage::autoversion]`.

| Transfer function | old RGBA scalar loop | **new RGBA incant!** | speedup |
|-------------------|---------------------:|---------------------:|--------:|
| bt709_to_linear   |                49.6µs|                24.2µs|   2.05× |
| linear_to_bt709   |                44.4µs|                20.3µs|   2.19× |
| pq_to_linear      |                50.8µs|                12.6µs|   4.03× |
| linear_to_pq      |                81.6µs|                21.3µs|   3.83× |
| hlg_to_linear     |                35.1µs|                13.7µs|   2.56× |
| linear_to_hlg     |                16.5µs|                14.5µs|   1.14× |

`linear_to_hlg` RGBA doesn't regress (unlike the plain variant on that
same data distribution) because the alpha save/restore overhead hides the
rite's branchy-data weakness. Every one of the six pairs is a strict win
in RGBA mode.

## Reproduction

```sh
cargo bench --features transfer --bench tf_bench -- tf_dispatch
cargo bench --features transfer --bench tf_bench -- tf_rgba_
cargo bench --features transfer --bench tf_bench -- 'tf_x8\b'
cargo bench --features transfer --bench tf_bench -- tf_scalar
```

Raw zenbench output saved to `/tmp/tf_dispatch_bench.log` and
`/tmp/tf_rgba_bench.log` at run time; not committed because they
contain absolute timestamps.
