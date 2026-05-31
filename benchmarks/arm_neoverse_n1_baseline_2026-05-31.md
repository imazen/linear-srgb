# linear-srgb — ARM Neoverse-N1 baseline + tail-curve fix (2026-05-31)

Hardware: Hetzner CAX21, Ampere Altra (**Neoverse-N1**), 4 cores, 8 GB.
Toolchain: rustc 1.96 (mise), zenbench `criterion-compat`.
Base commit: `8bbc5409` (main, "ci(asm): re-enable hard gate on opcode-essence form").
Bench cmd: `cargo bench --bench benchmarks -- --warm-up-time 1 --measurement-time 3`.
BATCH_SIZE = 10000 elements per call unless noted.

## 1. Baseline throughput (10000-element slices)

Captured BOTH the shipping runtime-dispatch config (`RUSTFLAGS=` unset — what
users actually get, NEON via runtime detection) and the `-C target-cpu=neoverse-n1`
pin. They are within noise of each other because aarch64 already has NEON in the
baseline target, so the dispatched NEON path is identical.

| Bench (f32, 10000 elem) | Neoverse-N1 pin | runtime-dispatch (RUSTFLAGS unset) |
|---|--:|--:|
| `srgb_to_linear/f32_f32/simd_slice` | 13.0 µs | 12.7 µs |
| `linear_to_srgb/f32_f32/simd_slice` | 14.6 µs | 14.1 µs |
| `linear_to_srgb/f32_u8/simd_slice` | 7.7 µs | 7.8 µs |
| `linear_to_srgb/f32_u16/simd_slice` | 15.7 µs | 15.5 µs |
| `srgb_to_linear/u8_f32/simd_lut8_slice` | 3.3 µs | — |

## 2. x86 cross-check (local AMD Ryzen 9 7950X, RUSTFLAGS unset)

| Bench | ARM N1 | x86 7950X | ARM/x86 |
|---|--:|--:|--:|
| f32→f32 srgb_to_linear | 13.0 µs | 1.48 µs | **8.8×** |
| f32→f32 linear_to_srgb | 14.6 µs | 2.17 µs | 6.7× |
| f32→u8 | 7.7 µs | 2.91 µs | 2.6× |
| f32→u16 | 15.7 µs | 2.31 µs | 6.8× |
| u8→f32 (LUT) | 3.3 µs | 1.92 µs | 1.7× |

The ARM/x86 gap is largest exactly on the divide-heavy f32→f32 paths (8.8×) and
smallest on the LUT path (1.7×, no divide). Root cause: the rational-polynomial
kernels do one vector `FDIV` per 4 lanes (4× per 16-lane chunk, visible in the
asm), and Neoverse-N1's vector `FDIV` is non-pipelined (~16-cycle, single divide
unit), whereas Zen4's is pipelined.

## 3. Hypothesis A — replace `yp / yq` with `yp * yq.recip()` — FALSIFIED

Idea: trade the non-pipelined NEON `FDIV` for magetypes' `f32x16::recip()`
(hardware `vrecpeq` estimate + Newton-Raphson), which runs on the two pipelined
FP units.

Result, **falsified on two independent grounds**:

1. **No speedup.** srgb_to_linear 13.0 → 13.3 µs (slightly *worse*),
   linear_to_srgb 14.6 → 14.1 µs (noise). magetypes' NEON `recip` applies
   **two** Newton-Raphson steps (8-bit estimate → ~23-bit), an ~8-op dependency
   chain whose latency matches or exceeds the single `FDIV` it replaces; the
   4-way chunk ILP doesn't recover it.
2. **Correctness regression at the piecewise boundary.** The recip kernel
   returned `1.0` for an input just above the sRGB threshold (≈0.0393) where the
   true value is ≈0.00304 — a 70M-ULP error — while the `FDIV` baseline was a
   clean 4-ULP max vs the f64-evaluated reference. Per the project's
   zero-tolerance precision rule this alone disqualifies it.

Conclusion: FDIV-replacement via the available magetypes reciprocal primitives
is a dead end on N1. A faster-but-correct path would need a hand-tuned **single**
NR step gated to NEON only (the shared magetypes kernel can't branch per-tier
without a separate NEON kernel), and even then the boundary blend needs a guard.
Not pursued — see "next hypothesis" below.

## 4. Hypothesis B — slice tail used the wrong transfer curve — SHIPPED

Finding: the remainder loops in `srgb_to_linear_slice` / `linear_to_srgb_slice`
(+ RGBA variants) converted the trailing up-to-15 elements of any
non-16-multiple slice via `scalar::srgb_to_linear` / `linear_to_srgb` — the
**`powf`-based IEC curve** — while the SIMD chunk loop uses the **C0-continuous
rational polynomial**. Different transfer function, different piecewise
threshold: a real cross-path pixel divergence on unaligned slices (and a `powf`
call per tail element). The aarch64 asm snapshot showed `bl powf` in the tail.

Fix: point the tail at `scalar::srgb_to_linear_fast` / `linear_to_srgb_fast` (the
same polynomial). Tail now matches the SIMD body within 3 ULP (was a different
curve); `bl powf` removed from the snapshots.

Measured tail cost (pure-tail and tail+chunk slices, 2M iters, warmed):

| Slice len | BEFORE (`powf` tail) | AFTER (fast-poly tail) | Speedup |
|---|--:|--:|--:|
| 15 (pure tail, 0 chunks) | 188.8 ns | 69.5 ns | **2.72×** |
| 31 (1 chunk + 15 tail) | 220.6 ns | 86.0 ns | 2.56× |
| 47 (2 chunks + 15 tail) | 244.1 ns | 107.9 ns | 2.26× |

Aligned-size throughput unchanged (10000-elem srgb_to_linear 13.0 → 12.6 µs,
within noise — only the tail changed). All 192 ARM tests pass (the 90 s
exhaustive `brute_force` ULP sweep + cross-tier `simd_consistency` included).

## 5. Next concrete hypothesis (follow-on)

At small aligned sizes the SIMD slice *loses* to scalar on N1 (`s2l_slice/64` =
86.8 ns vs `s2l_scalar/64` = 69.3 ns; `simd_s2l/100` = 127 ns vs `scalar_s2l/100`
= 114 ns). The constant-materialization preamble (~20 `dup`/`mov` to splat the
10 polynomial coefficients + thresholds, seen in the asm) is amortized over too
few chunk iterations. Hypothesis: hoist the splatted-coefficient vectors so they
are built once per call (or const-folded) rather than re-materialized, and/or
add a scalar fast-path for slices below a small threshold (e.g. < 32 elements)
so tiny slices skip the SIMD preamble entirely. Target: close the
SIMD-loses-to-scalar crossover below ~64 elements.

## 6. Iteration #2 — small-slice crossover fix — BOTH candidates FALSIFIED (2026-05-31)

Re-confirmed the crossover, then falsified both proposed fixes by measurement.
Bench cmd: `cargo bench --bench benchmarks -- dispatch_overhead/... --warm-up-time 1
--measurement-time 3`. New bench arms `s2l_scalar_fast` / `l2s_scalar_fast`
added to `benches/benchmarks.rs` (the only commit from this iteration — they
measure the **fast-poly** scalar, `default::{srgb,linear}_to_srgb`, which is the
ONLY correctness-safe path a fast-path branch could take).

### 6.1 Crossover re-confirmed (Neoverse-N1 pin)

`s2l_slice` (SIMD) vs `s2l_scalar` (the **`powf` IEC** scalar the iter-1 bench
used) — SIMD loses up to ~512, crosses over between 512 and 1024:

| N | s2l_slice (SIMD) | s2l_scalar (powf) |
|---|--:|--:|
| 8 | 15.0 | 9.7 |
| 16 | 21.5 | 17.7 |
| 32 | 40.7 | 34.8 |
| 64 | 81.0 | 69.2 |
| 128 | 163.7 | 140.5 |
| 256 | 326.0 | 299.2 |
| 512 | 646.5 | 635.2 |
| 1024 | **1288.0** | 1458.7 |

For `l2s`, SIMD wins at every size (the `powf` l2s scalar is much slower).

### 6.2 Candidate 1 — coefficient-splat hoisting — N/A, LLVM already does it

The aarch64 asm snapshot (`asm-snapshots/.../stub_srgb_to_linear_slice.s`) shows
the ~37-instruction coefficient-splat preamble (`mov`/`movk`/`dup` for the 10
poly coefficients + thresholds into `v2`–`v20`, lines 19–53) is emitted **once,
before the chunk loop `.LBB14_2`** — not re-materialized per chunk. LLVM's LICM
already hoists every loop-invariant splat. There is nothing to restructure; the
splats are paid once per call as the hypothesis wanted. Candidate 1 cannot
produce a delta.

The per-call overhead that *does* exist is intrinsic and unavoidable: the
preamble + a 5-register callee-saved spill (`d8`–`d12`, forced by the 19-vector
register pressure of the splats), run unconditionally whenever there is ≥1 chunk.
That fixed cost is what loses to scalar at small N — but it is not removable by
hoisting (it is already hoisted) and is the price of the vectorized FDIV-amortized
body that *wins* at large N.

### 6.3 Candidate 2 — small-N scalar fast-path — FALSIFIED (no correctness-safe win)

The fast-path may only call `srgb_to_linear_fast` / `linear_to_srgb_fast` (the
C0-continuous rational poly the SIMD body evaluates) — using the `powf` scalar
would re-introduce the exact cross-path curve divergence iter-1 just fixed. The
fast-poly scalar (`s2l_scalar_fast`) measures:

| N | s2l_slice (SIMD) | s2l_scalar_fast | l2s_slice (SIMD) | l2s_scalar_fast |
|---|--:|--:|--:|--:|
| 8 | 16.1 | **14.5** | 38.8 | 37.9 |
| 16 | 22.7 | 28.1 | 23.9 | 80.1 |
| 32 | 40.8 | 55.2 | 48.4 | 162.7 |
| 64 | 81.3 | 109.6 | 90.4 | 322.7 |
| 128 | 161.0 | 226.5 | 187.2 | 671.8 |

The fast-poly scalar beats SIMD **only at N=8 for s2l (14.5 vs 16.1, a 1.6 ns
gap inside run-to-run noise; the runtime-dispatch run had 14.6 vs 15.2 = 0.6 ns)**
and is slower everywhere else; for `l2s` SIMD wins from N=16 up. Root cause
(confirmed in the tail asm, lines 188–199 of the same snapshot): `eval_rational_poly_5`
casts to **f64** for monotonicity and does a non-pipelined **f64 `fdiv`** per
element — on N1 that costs more than the vectorized path that amortizes one
vector FDIV across 4 lanes. The N=8 win is too small and too rare to justify
gating every slice call with a length compare plus the added cross-path surface
area. **No correctness-safe fast-path produces a measurable, robust win.**

Runtime-dispatch (RUSTFLAGS unset) numbers match the pin within noise (aarch64
has NEON in the baseline target, so the dispatched NEON path is the same code).

### 6.4 Verdict + next direction

Iteration #2 is **doc-only**: no `src/` change. The crossover at small N is a
fixed-overhead floor of the vectorized kernel (register spill + coefficient
preamble) that LLVM already minimizes, and the only correctness-preserving
scalar alternative is slower on N1 because of its f64-FDIV.

Next hypotheses worth trying (in priority order):
1. **Reduce register pressure in `mt_srgb_to_linear`** so the `d8`–`d12` spill
   disappears. The 10 coefficient splats + 1.0/zero/threshold consume 13+ vector
   regs simultaneously; interleaving the P and Q Horner chains so fewer splats
   are live at once might let LLVM keep everything in `v0`–`v7` and drop the
   spill. Inspect the asm for whether the spill is gone, then bench small N.
2. **An f32-only (no f64-intermediate) scalar `_fast` variant** gated behind a
   monotonicity re-check — if the f32 rational poly is monotonic enough on N1
   for the slice-tail/fast-path use, it would be much cheaper than the f64 path
   and *could* beat SIMD below ~64. Requires re-running the exhaustive ULP +
   monotonicity sweep on the f32 path; only pursue if §6.1's powf-vs-SIMD gap is
   worth chasing for a real downstream caller.
3. **Accept the floor.** Small in-place f32 slices (< ~64 elem) are not a hot
   path for the crate's consumers (image rows are ≥ width elements, typically
   ≥ 256); the win ceiling here is < 20 ns/call on the rarest case.
