# Portable scalar math, 2026-11-05

Source base `0660955b5d367d5fead96151fd7ff8e2926e2db0` (bench body),
with explicit V3 validation added in `781148dd` and audit tooling in `c31dcfe8`.
Commands: `cargo test --test portable_math` with `PORTABLE_MATH_REQUIRE_NATIVE=1`,
`cargo run --release --example portable_math_accuracy`, and
`cargo bench --bench portable_math`. Tests on ARM use `--release`.
Run-heavy: x86 16G/10 jobs; ARM 8G/4 jobs. zenbench 0.1.11 interleaves
preallocated candidates. Exploratory measurements, with noisy/drifting rounds;
these are costs of these whole-loop wrappers, not isolated instruction latency.

The same 1,048,576-point grids on x86-64 and ARM give:

| Operation | Domain | Native lane disagreements | Relaxed scalar disagreements | Max ULP vs f64→f32 |
|---|---|---:|---:|---:|
| log2 | bases 2^-120 .. 2^120, logarithmically spaced | 0 | 1299 | 4 |
| exp2 | exponents -126 .. 128, uniform | 0 | 103518 | 2 below 127.5; 133 at/above 127.5 |
| pow(x, 2.4f32) | bases 2^-50 .. 2^50, logarithmically spaced | 0 | 105833 | 46 |

Ranges exclude the upper endpoint. These maxima describe the grids, not
exhaustive ULP bounds. Portable helpers reuse existing coefficients and fused
operation order. They do not normalize subnormal inputs or construct subnormal
exponentials. Error in pow grows with exponent-amplified logarithm error.
V3, native x86 and NEON tests execute actual hardware tiers; ordinary baseline
CI still verifies the portable functions and their edge policies.

The scalar baseline calls a fused-math helper on this x86 build; the explicit
hardware context lets LLVM inline native FMA and vectorize parts of the loop.
Native-vector cases include dispatch, packing and partial-vector padding.
Single-element inputs are the first point of each grid; those timings should
not be extrapolated to arbitrary tail values.

## Zen 4, Rust 1.99.1

| Operation / samples | Native vector | Relaxed scalar | Portable baseline | Portable FMA context |
|---|---:|---:|---:|---:|
| Log2/1 | 9.65 ±0.05ns | 8.02 ±0.10ns | 7.78 ±0.13ns | 3.21 ±0.01ns |
| Log2/16 | 9.49 ±0.22ns | 11.12 ±0.11ns | 104.68 ±3.04ns | 4.43 ±0.07ns |
| Log2/16384 | 10.0 ±0.0µs | 10.6 ±0.1µs | 111.2 ±2.0µs | 3.6 ±0.0µs |
| Exp2/1 | 9.41 ±0.06ns | 49.71 ±0.22ns | 12.27 ±0.05ns | 3.68 ±0.05ns |
| Exp2/16 | 9.06 ±0.13ns | 52.50 ±0.17ns | 184.40 ±0.48ns | 23.64 ±0.09ns |
| Exp2/16384 | 10.0 ±0.1µs | 52.2 ±0.2µs | 188.1 ±0.5µs | 24.7 ±0.0µs |
| Pow/1 | 12.9 ±0.3ns | 40.5 ±0.4ns | 19.7 ±0.1ns | 7.0 ±0.0ns |
| Pow/16 | 14.7 ±0.2ns | 57.7 ±0.6ns | 282.0 ±1.5ns | 40.0 ±0.3ns |
| Pow/16384 | 12.2 ±0.2µs | 59.1 ±0.6µs | 304.2 ±12.5µs | 39.2 ±0.5µs |

run-heavy: done rc=0 32s | peak-RSS 0.40GiB | min-avail 17263MiB | peak-load 1.47

## Neoverse N1, Rust 1.96.0

| Operation / samples | Native vector | Relaxed scalar | Portable baseline | Portable FMA context |
|---|---:|---:|---:|---:|
| Log2/1 | 25.8 ±0.2ns | 29.1 ±0.3ns | 9.2 ±0.2ns | 9.7 ±0.1ns |
| Log2/16 | 31.7 ±0.2ns | 35.9 ±0.3ns | 31.1 ±0.2ns | 31.5 ±0.1ns |
| Log2/16384 | 24.9 ±0.2µs | 31.3 ±0.4µs | 28.3 ±0.2µs | 28.2 ±0.2µs |
| Exp2/1 | 26.9 ±0.3ns | 108.5 ±0.3ns | 9.3 ±0.1ns | 10.2 ±0.1ns |
| Exp2/16 | 30.0 ±0.4ns | 114.0 ±0.8ns | 42.7 ±0.4ns | 43.4 ±0.2ns |
| Exp2/16384 | 24.5 ±0.4µs | 111.9 ±0.3µs | 40.5 ±0.3µs | 40.5 ±0.2µs |
| Pow/1 | 44.2 ±0.3ns | 85.6 ±0.4ns | 20.2 ±0.3ns | 21.1 ±0.3ns |
| Pow/16 | 47.7 ±0.7ns | 133.0 ±0.8ns | 101.3 ±0.8ns | 101.9 ±1.2ns |
| Pow/16384 | 42.6 ±0.3µs | 134.2 ±0.2µs | 95.4 ±0.3µs | 95.5 ±0.2µs |

run-heavy: done rc=0 84s | peak-RSS 0.57GiB | min-avail 14353MiB | peak-load 0.71

## Recommendation and limits

Expose the three explicitly named scalar approximations as compatibility
companions in this PR. Reuse them for tails only when the surrounding kernel
uses the same coefficients, FMA order and domains. Generic transcendental
ownership should remain in magetypes; a later generic portable-FMA policy
could supersede these wrappers through an additive migration.

Baseline x86 16,384-value costs are 111.2 / 188.1 / 304.2 µs for log2/exp2/pow,
versus 10.6 / 52.2 / 59.1 µs for relaxed scalar. FMA-context costs are
3.6 / 24.7 / 39.2 µs. ARM portable costs are 28.3 / 40.5 / 95.4 µs, versus
31.3 / 111.9 / 134.2 µs relaxed scalar. These are measured whole-loop costs;
baseline helper-call cost must not be presented as an unavoidable SIMD-caller cost.

No lookup tables or new dependencies are added. Linked binary-size deltas
were not measured for this API. Strict WASM and Zen 5 costs were not measured.
ARM Cargo fingerprints record `-C target-cpu=neoverse-n1`; x86 uses the default
portable target. Neither build uses target-cpu=native.

Bt2446A also has a degree-7 output power approximation in its vector body
and libm in its remainder. Replacing its pow/log/exp scalar calls alone would
leave that separate difference. This PR does not migrate consumers or claim
whole-tone-mapper parity.

Raw logs: `~/tmp/precision-logs/portable-{x86-final,arm}.log` on r7900x;
reference binaries and source identities are retained outside git under
`/mnt/v/output/linear-srgb/precision-2026-11-05/`.

## Baseline lowering check

The benchmark executable SHA256 is
`e0b233e01dd02d0610964d4e4125772e8a6305225364a90dd9086de64dc13c5f`.
`objdump -Cd` shows baseline scalar Horner stages calling the local `fmaf`
thunk (GOT relocation0x1f1180→0x1e7c70); the thunk calls compiler-builtins'
cached runtime-selected function. Rust1.99.1's compiler-builtins source has
hardware-FMA/FMA4 and software fallback targets. Thus the measured Zen4
baseline difference must not be labeled pure software-FMA emulation.
The matching V3 function contains packed/scalar `vfmadd` instructions with
those helpers inlined. Times combine helper-call overhead, guard logic and
vectorization; they do not isolate any one factor. The disassembly is retained
outside git as `portable-x86-disassembly.txt` beside the reference artifacts.
