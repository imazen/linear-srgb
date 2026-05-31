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
