# linear-srgb benchmarks — methodology & reproduction

How to run linear-srgb's benchmarks fairly, and how to read the committed result
files. These are **internal** A/B benchmarks: every contender is one of
linear-srgb's own conversion strategies, plus an in-file imageflow-style
reference baseline. No third-party crate is timed, so there are no external
competitor versions to pin — the only dev-dependency is the
[zenbench](https://github.com/imazen/zenbench) harness (`0.1.2`, via its
`criterion-compat` API).

## What is compared

The benches answer "which of our own paths is fastest for this conversion, and
did a code change help or hurt?" The arms are:

- **`scalar_powf`** — the exact `powf`-based scalar curve, per element.
- **rational polynomial** (`srgb_to_linear` / `linear_to_srgb`) — the
  C0-continuous fast scalar, no `powf`.
- **LUT** — `lut8` / `lut12` / `lut16` table lookups (with and without
  interpolation).
- **SIMD slice** — the dispatched `*_slice` functions (AVX-512 / AVX2 / SSE4.1 /
  NEON / WASM128 via archmage `incant!`).
- **`imageflow_*`** — a reference baseline replicating the classic imageflow
  powf / LUT approach, coded directly in the bench file for an honest
  in-process comparison.

## Fairness guarantees

`benches/*.rs` are built so the numbers mean something:

- **Interleaved (paired) measurement.** zenbench runs contenders round-robin,
  not "all of A then all of B," so each sees the same thermal state, turbo
  residency, and OS scheduling — systematic drift cancels in the paired delta.
- **No I/O in the timed region.** Inputs are synthesized into `Vec<u8>` /
  `Vec<u16>` / `Vec<f32>` once, before timing starts. The timed closure only
  calls the conversion. Output is fed to `std::hint::black_box` so it isn't
  optimized away.
- **Single-thread vs single-thread.** linear-srgb has no internal threading (no
  rayon, no thread pool) — every path runs on the calling thread, so all arms
  are inherently apples-to-apples.
- **No `-C target-cpu=native`.** Builds use runtime SIMD dispatch (archmage
  `incant!`), which is what ships. Native builds bake in ISA extensions and give
  misleading numbers. Do not set `RUSTFLAGS=-C target-cpu=native` when
  reproducing.

## Reproduce

```sh
git clone https://github.com/imazen/linear-srgb && cd linear-srgb
git checkout <commit>          # the commit named in the result file you're reproducing

cargo bench --bench benchmarks                    # throughput by type / strategy
cargo bench --features transfer --bench tf_bench  # BT.709 / PQ / HLG dispatch A/B
cargo bench --bench rgba_approach                 # RGBA alpha-handling strategies
```

The three `[[bench]]` targets are `harness = false` (zenbench's criterion-compat
shim); `tf_bench` requires the `transfer` feature. Pass a filter to scope a run,
e.g. `cargo bench --features transfer --bench tf_bench -- tf_dispatch`.

## Result files

Each committed run lands as `benchmarks/<topic>_<YYYY-MM-DD>.md` and **must**
state, in its header: the git commit, the CPU/RAM/OS, `rustc -V`, the exact
command, the element count, and whether `target-cpu` was pinned. Current files:

- **`arm_neoverse_n1_baseline_2026-05-31.md`** — ARM Neoverse-N1 (Ampere Altra,
  Hetzner CAX21) throughput baseline for the f32/u8/u16 slice paths, captured
  both runtime-dispatch and `-C target-cpu=neoverse-n1`; an x86 (Ryzen 9 7950X)
  cross-check; the slice-tail curve fix (tail elements were on the wrong
  transfer curve); and two falsified optimization hypotheses (FDIV→recip,
  small-slice scalar fast-path).
- **`tf_dispatch_2026-04-23.md`** — BT.709 / PQ / HLG slice dispatch A/B (issue
  #10): old `#[autoversion]` scalar loop vs the `incant!` SIMD dispatcher vs the
  direct `tokens::x8` rite, plain and RGBA variants, on a Ryzen 9 7950X.

Do not commit numbers you didn't generate, and don't extrapolate one size or
host to another — measure each. Memory claims need heaptrack / `time -v`, not
estimates.

## Charts (what to plot for which decision)

| Question | Chart |
|----------|-------|
| "Which strategy is fastest?" | horizontal **bar**, sorted by throughput (Mops/s or µs per N) |
| "How does it scale with slice length?" | **line** (x = elements, log); fit `total = α + β·N` and report both the fixed per-call overhead and the per-element slope (the small-N crossover in the ARM file is exactly this α term) |
| "Is the A/B delta real / how noisy?" | **violin** or PDF of per-call times, or zenbench's paired 95% CI |

For new charts, prefer [zenbench](https://github.com/imazen/zenbench) directly —
it does the interleaving and emits a sorted throughput **bar chart**, a
self-contained **SVG** report (`--format=html`), and violin/PDF/regression plots
(plotters.rs).
