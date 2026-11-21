# `linear-srgb 0.6.13` Math Review, Const-Generic Rail Ablation & 5-Tier Codegen Audit (2026-11-21)

**Host:** `r7900x` (AMD Ryzen 9 7900X, Zen 4, 24 threads, 30.0 GiB RAM, Ubuntu; **not measured on Zen 5**).
**Runner:** `/home/lilith/work/claudehints/scripts/run-heavy` (`peak-RSS = 0.69 GiB` on `ab_vs_0612`, `0.45 GiB` on `ablation`). No `-C target-cpu=native` passed to benchmarks.
**Baseline:** crates.io `linear-srgb = "=0.6.12"`, `garb = "=0.2.8"`, `archmage = "=0.9.22"`, `archmage-macros = "=0.9.22"`, `magetypes = "=0.9.22"`.
**Candidate:** `linear-srgb 0.6.13` (`archmage 0.9.32`, `magetypes 0.9.32`, `magemath 0.1.0`) + `garb 0.2.9`.
**Reproducers:**
- Codegen symbols: `examples/downstream_codegen.rs` (`just asm-v3`, `just asm-v4`, `just asm-neon`, `just asm-wasm128`, `just asm-wasm128-relaxed`, `just asm-all <symbol>`).
- Latency & memory-hierarchy benchmark: `cargo bench --bench downstream_exercised --features transfer`.

---

## 1. Exhaustive `2^32` `f32` & `65,536` `u16` Mathematical Proofs

1. **Post-division `.min(1.0)` on C0 sRGB rational polynomials is 100% dead over all `2^32` `f32` bit patterns:**
   - Exhaustively evaluated across all `4,294,967,296` IEEE-754 `f32` bit patterns under both hardware FMA and separate mul+add rounding:
     - `S2L_P(x) / S2L_Q(x)` maximum on `[0, 1]` is `0.999999642` (`0x3f7ffffa`, 6 ULPs below `1.0`).
     - `L2S_P(√x) / L2S_Q(√x)` maximum on `[0, 1]` is `0.999999583` (`0x3f7ffff9`, 7 ULPs below `1.0`).
     - Removing `(yp / yq).min(one)` (`POST_DIV_MIN_ONE = 0`) produces **`0` bit differences out of `8,589,934,592` evaluations** (`0 / 65,536` `u16` differences), saving `vminps + vcmpunordps + vblendvps` (`3` ops on x86, `3` ops on WASM) and eliminating an AVX2 register-spill slot.
2. **PQ (`ST 2084`) coefficient-blended Horner evaluation (`eval_blended_poly_x*`):**
   - Exhaustively verified across all `1,065,353,217` `f32` values in `[0, 1]` (FMA and non-FMA): blending `[SMALL, LARGE]` coefficients per Horner step (`8` FMAs + `1` divide; `Q_SMALL[4] == Q_LARGE[4] == 1.0` shared without blend) produces **`0` bit differences** vs evaluating all four degree-4 polynomials (`16` FMAs) and blending `p` and `q`.
3. **Custom gamma (`src/tf/math.rs`):**
   - Verified in `magemath 0.1.0` that `GAMMA_OPTIONS.const_exponent(2.4)` and `GAMMA_OPTIONS.const_exponent(1.0 / 2.4)` compile to the exact same `Powf::<GAMMA>` evaluation graph (`clamp_eff = true`), allowing a single `Powf::<GAMMA>` instantiation instead of 3× duplicated inline `Powf` graphs (`18,569 -> 372` / `6,369` WASM instructions).
4. **Golden 64-bit output fingerprints (`target_features::fingerprint::entry_point_fingerprint`):**
   - `0` bit differences across all `18` native hashes (`x4/x8/x16` × `{scalar, v3, v4}` × `{plain, transfer}` over slice lengths `0..=65`) and all `12` WASM default hashes (`scalar` + `wasm128`), plus `probe-relaxed-wasm` with both `fma=true` and `fma=false`.

---

## 2. Const-Generic Rail Ablation (`src/kernels/float.rs::cfg`)

`srgb_to_linear_vector_cfg::<T, CFG>` and `linear_to_srgb_vector_cfg::<T, CFG>` specialize rail enforcement at compile time via bitflags (`CLAMP_LO`, `CLAMP_HI`, `SNAP_ONE`, `POST_DIV_MIN_ONE`):

| `CFG` Profile | Flags Enabled | Chained 64-RT Latency (`X64V3`, `ops/s`) | `L1_256` (`1 KiB`) | `L2_16k` (`64 KiB`) | `DRAM_2M` (`8 MiB`) | Use Case |
|---|---|---:|---:|---:|---:|---|
| `SRGB_LEGACY_POST_MIN` | `CLAMP_LO \| CLAMP_HI \| SNAP_ONE \| POST_DIV_MIN_ONE` | `488 Mops/s` (base) | `19.3 GiB/s` (`49.5 ns`) | `18.1 GiB/s` | `18.5 GiB/s` | 0.6.13-pre baseline |
| `SRGB_CLAMPED` | `CLAMP_LO \| CLAMP_HI \| SNAP_ONE` | **`535 Mops/s` (`-8.8%`)** | **`22.3 GiB/s` (`-14.6%`)** | **`20.3 GiB/s` (`-10.3%`)** | **`21.1 GiB/s` (`-12.4%`)** | Public `f32` contract (`0` bit diffs) |
| `SRGB_UNIT_NONNEG` | `SNAP_ONE` | **`591 Mops/s` (`-17.7%`)** | **`22.5 GiB/s` (`-16.4%`)** | **`21.9 GiB/s` (`-16.4%`)** | **`21.9 GiB/s` (`-15.8%`)** | Normalized `u16 * (1/65535)` decode |
| `SRGB_IN_OPEN_UNIT` | `0` (all rails ablated) | **`608 Mops/s` (`-20.0%`)** | **`25.8 GiB/s` (`-26.7%`)** | **`23.7 GiB/s` (`-23.3%`)** | **`24.0 GiB/s` (`-22.8%`)** | Verified `(0, 1)` interior callers |

---

## 3. Architectural Fixes Applied in `0.6.13`

1. **Closure-free `map_slice_try_inline!` / `map_slice_inline!` macros + `#[cold] #[inline(never)]` fallbacks (`src/kernels/mod.rs`):**
   - Previously, inlined rare `else` scalar fallbacks (`vzeroupper + callq`) inside `bt709`, `hlg`, `pq`, and `extended` loops clobbered all 32 `%zmm` registers across every loop back-edge, forcing 20–24 constant reloads + 384 B stack frames per iteration (`bt709_to_linear_slice` was `+125%` slower).
   - Using `map_slice_try_inline!` macros (avoiding Rust closure `#[target_feature]` loss on `evex512`) keeps every hot SIMD loop 100% call-free with all 12–26 constants hoisted into `%ymm`/`%zmm` registers.
2. **Fixed `+v4x(cfg(avx512))` -> `+v4(cfg(avx512))` on 13 `src/simd.rs` dispatchers:**
   - `f32x16` floating-point kernels require `X64V4Token` (`avx512f/bw/dq/vl`), not `X64V4xToken` (Ice Lake integer extensions). Fixing this cut `-C target-cpu=x86-64-v4` slice codegen by `-102` to `-255` instructions and restored AVX-512 dispatch on Skylake-X/Cascade Lake CPUs.
3. **Eliminated pre-loop `tail_block` `memset`/`memcpy` + `Chain` iterator in `map_slice` and RGBA slices:**
   - Processing full chunks in a flat slice loop and handling `remainder` only when `!remainder.is_empty()` after the loop removed pre-loop PLT calls and stack-frame spills.
4. **Hardware `fast_sqrt_f32` (`vsqrtss` / `fsqrt` / `f32.sqrt` under `std`) + padded SIMD tail in `linear_to_srgb_u16_slice`:**
   - Replaced 65-instruction integer `RSQRT_TAB` software square roots in `std` builds (`ds_scalar_linear_to_pq`: `197 -> 59` instructions; `linear_to_srgb_u16_slice`: `+11.2%` regression -> `-1.0%` faster than `0.6.12`).
5. **Fast clamp (`floor_zero! + min_fast`) + `to_i32_with::<TRUNC>()` in LUT indexers (`u8`, `u16_fast`, `x8`, `x16`):**
   - Reduced clamp + float-to-int from `11` ops/vector to `3` ops/vector with `0` bit differences.

---

## 4. `llvm-mca` Static Pipeline Latency & Reciprocal Throughput (`0.6.12` -> `0.6.13`)

| In-Register Array Kernel | V3 (`znver4`) Insns | V3 Latency (cyc) | V3 RThroughput (cyc) | V4 (`znver4`) Insns | V4 Latency (cyc) | V4 RThroughput (cyc) | NEON (`neoverse-n1`) RThroughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ds_arr_srgb_to_linear` | `31 -> 32` | **`47 -> 45`** | **`7.0 -> 7.0`** | `22 -> 23` | **`58 -> 52`** | **`7.0 -> 7.0`** | **`11.0 -> 11.0`** |
| `ds_arr_linear_to_srgb` | `32 -> 33` | **`61 -> 60`** | **`8.0 -> 8.0`** | `23 -> 24` | **`68 -> 62`** | **`8.0 -> 8.0`** | **`14.0 -> 14.0`** |
| `ds_arr_srgb_u16_to_linear` | **`33 -> 32`** | **`53 -> 50`** | **`8.0 -> 8.0`** | **`24 -> 22`** | **`62 -> 56`** | **`7.5 -> 7.5`** | n/a |
| `ds_arr_linear_to_srgb_u16` | **`41 -> 40`** | **`80 -> 78`** | **`9.0 -> 9.0`** | **`29 -> 28`** | **`95 -> 88`** | **`9.0 -> 9.0`** | n/a |
| `ds_arr_pq_to_linear` | `47 -> 57` | **`50 -> 50`** | `10.5 -> 12.5` | `31 -> 48` | `61 -> 62` | `10.5 -> 12.0` | `18.0 -> 19.5` |
| `ds_arr_linear_to_pq` | `50 -> 61` | `76 -> 81` | **`16.0 -> 13.0`** | `34 -> 51` | `84 -> 93` | `16.0 -> 23.0` | **`28.0 -> 22.0`** |
| `ds_arr_bt709_to_linear` | `56 -> 73` | `90 -> 106` | `12.0 -> 15.0` | `38 -> 53` | `105 -> 142` | `12.0 -> 14.5` | **`27.0 -> 21.0`** |
| `ds_arr_linear_to_bt709` | `55 -> 70` | `80 -> 96` | `12.0 -> 14.5` | `38 -> 52` | `95 -> 131` | `12.0 -> 14.0` | `20.0 -> 21.0` |
| `ds_arr_hlg_to_linear` | `40 -> 49` | `58 -> 61` | `8.0 -> 9.5` | `28 -> 37` | `69 -> 82` | `8.0 -> 9.0` | `12.0 -> 14.5` |
| `ds_arr_linear_to_hlg` | `39 -> 64` | `53 -> 67` | `8.5 -> 12.0` | `28 -> 45` | `82 -> 99` | `8.5 -> 12.0` | `16.0 -> 19.0` |

*(Note: `0.6.12` lacked `NaN` propagation, negative/overflow domain rails, ST 2084 dark-toe precision, and BT.709/HLG C1 seam monotonicity; `0.6.13` enforces all `transfer_rails` invariants via `magemath`.)*

---

## 5. Paired `zenbench` Runtime Results (`0.6.12` vs `0.6.13`, Zen 4 `r7900x`)

| Benchmark | `0.6.13` Pre-Fix vs `0.6.12` | `0.6.13` Final vs `0.6.12` | `0.6.12` Throughput | `0.6.13` Throughput |
|---|---:|---:|---:|---:|
| `lat_chain64_srgb_rt_x8` | `+5.5%..+5.6%` | **`-3.0%..-2.8%`** | `528 Mops/s` | **`540 Mops/s`** |
| `L1_256/srgb_to_linear_slice` | `-0.8%..+0.3%` | **`-0.7%..-0.2%` (~tie)** | `20.0 GiB/s` | **`20.0 GiB/s`** |
| `L1_256/linear_to_srgb_slice` | `-0.0%..+1.0%` | **`-0.3%..+0.0%` (~tie)** | `14.5 GiB/s` | **`14.4 GiB/s`** |
| `L1_256/srgb_to_linear_rgba_slice` | `-75.4%..-74.7%` | **`-80.9%..-80.5%` (5.2×)** | `4.65 GiB/s` | **`24.4 GiB/s`** |
| `DRAM_2M/srgb_to_linear_slice` | `-2.8%..+3.5%` | **`-2.3%..+1.4%` (~tie)** | `17.6 GiB/s` | **`17.6 GiB/s`** |
| `DRAM_2M/linear_to_srgb_slice` | `-3.6%..+2.7%` | **`-1.0%..-0.4%` (~tie)** | `13.1 GiB/s` | **`13.2 GiB/s`** |
| `DRAM_2M/srgb_to_linear_extended_slice` | `-1.6%..+6.6%` | **`-1.9%..+1.1%` (~tie)** | `14.2 GiB/s` | **`14.2 GiB/s`** |
| `DRAM_2M/linear_to_srgb_extended_slice` | `-3.4%..+4.6%` | **`-2.0%..+2.6%` (~tie)** | `11.6 GiB/s` | **`11.6 GiB/s`** |
| `DRAM_2M/srgb_to_linear_rgba_slice` | `-72.4%..-68.5%` | **`-73.0%..-69.1%` (3.5×)** | `4.19 GiB/s` | **`14.5 GiB/s`** |
| `DRAM_2M/srgb_to_linear_premul_rgba` | `-73.7%..-64.4%` | **`-72.8%..-69.3%` (3.5×)** | `4.02 GiB/s` | **`13.9 GiB/s`** |
| `DRAM_2M/unpremul_linear_to_srgb_rgba` | `-17.3%..-11.7%` | **`-18.5%..-14.0%` (1.21×)** | `8.10 GiB/s` | **`9.78 GiB/s`** |
| `DRAM_2M/pq_to_linear_slice` | `+3.1%..+6.1%` | **`-14.2%..-10.1%` (1.13×)** | `11.2 GiB/s` | **`12.7 GiB/s`** |
| `DRAM_2M/linear_to_pq_slice` | `-10.0%..-9.0%` | **`-20.2%..-19.7%` (1.26×)** | `6.81 GiB/s` | **`8.55 GiB/s`** |
| `DRAM_2M/bt709_to_linear_slice` | `+100.9%..+104.9%` | `+20.1%..+26.4%` (domain rails) | `8.84 GiB/s` | `7.17 GiB/s` |
| `DRAM_2M/linear_to_bt709_slice` | `+35.1%..+35.8%` | `+26.5%..+29.2%` (domain rails) | `10.7 GiB/s` | `8.29 GiB/s` |
| `DRAM_2M/hlg_to_linear_slice` | `+6.8%..+8.5%` | `+4.1%..+8.6%` (domain rails) | `15.3 GiB/s` | `14.4 GiB/s` |
| `DRAM_2M/linear_to_hlg_slice` | `+46.2%..+53.0%` | `+40.9%..+42.4%` (domain rails) | `13.9 GiB/s` | `9.83 GiB/s` |
| `DRAM_2M/gamma_to_linear_slice_2.2` | `+13.5%..+14.2%` | `+8.9%..+10.1%` (`NaN` + rails) | `7.86 GiB/s` | `7.14 GiB/s` |
| `DRAM_2M/srgb_u8_to_linear_slice` | `-0.5%..+0.3%` | **`-0.1%..+0.3%` (~tie)** | `5.35 Gelem/s` | **`5.35 Gelem/s`** |
| `DRAM_2M/linear_to_srgb_u8_slice` | `-0.9%..-0.3%` | **`-0.4%..+0.2%` (~tie)** | `3.59 Gelem/s` | **`3.59 Gelem/s`** |
| `DRAM_2M/srgb_u16_to_linear_slice` | `-0.4%..+0.5%` | **`-1.4%..-1.0%`** | `6.07 Gelem/s` | **`6.16 Gelem/s`** |
| `DRAM_2M/linear_to_srgb_u16_slice` | `+9.2%..+10.8%` | **`-1.2%..-0.6%`** | `4.41 Gelem/s` | **`4.42 Gelem/s`** |
| `DRAM_2M/linear_to_srgb_u16_slice_fast` | `-1.6%..+0.9%` | **`-1.3%..+0.1%` (~tie)** | `2.01 Gelem/s` | **`2.01 Gelem/s`** |

---

## 6. Instruction Latency vs Memory Latency / Bandwidth Regime Analysis

1. **In-register fused pipelines (`zenpixels-convert::fast_gamut`, `zenjpeg` linear LUTs):**
   - Zero cache/DRAM traffic inside the pixel kernel; if stages are chained serially, runtime is governed by **critical-path instruction latency** (`45` cycles `srgb_to_linear` + `60` cycles `linear_to_srgb` on Zen 4 V3 = `105` cycles / 8 floats ≈ `1.85 ns/float` measured in `lat_chain64_srgb_rt_x8`), **not** reciprocal throughput (`7–8` cycles). Ablating `POST_DIV_MIN_ONE` + `CLAMP_LO`/`CLAMP_HI` directly shortens the critical dependency chain by `9%–20%`.
2. **Rational-polynomial & transcendental slice sweeps (`srgb`, `extended`, `pq`, `bt709`, `hlg`, `gamma`):**
   - Single-core DDR5 streaming bandwidth on `r7900x` (`118.9 GiB/s` calibrated; `104 GiB/s` achieved in-place by `garb::bytes::rgba_to_bgra_inplace`) is **`5×–14×` higher** than single-core `vdivps`/`vsqrtps`/FMA throughput (`7.1–24.0 GiB/s`).
   - Consequently, `linear-srgb` float slices remain **compute / divider-port bound even at `8 MiB` DRAM** (`L1_256 = 20.0 GiB/s` -> `DRAM_2M = 17.6 GiB/s`, only `-12%` drop from L1 to DRAM!), which is why instruction count and const-generic clamp ablation (`18.5 -> 24.0 GiB/s` at `DRAM_2M`) translate almost 1:1 into `8 MiB` wall-clock speedups.
3. **Scalar-gather LUT slices (`srgb_u8_to_linear_slice`, `linear_to_srgb_u8_slice`, `linear_to_srgb_u16_slice_fast`):**
   - Governed by **L1D cache load-to-use latency (`4–5` cycles) and AGU port saturation** (`16` scalar table loads per SIMD chunk): `u8` (`1 KiB` / `4 KiB` tables) stays 100% L1-resident (`5.42 -> 5.35 Gelem/s` from L1 to DRAM), while `u16_fast` (`128 KiB` table, L2-resident) sees L2 load latency (`2.40 -> 2.01 Gelem/s`).
4. **Pure byte swizzles (`garb::bytes::rgba_to_rgb`, `rgba_to_bgra_inplace`):**
   - `2–4` `pshufb` / `vpshufb` instructions per 64 B (`~1–2` cycles RThroughput) are **compute-bound only in L1/L2** (`86.5–162 GiB/s`) and become **100% DRAM-bandwidth bound at `8 MiB`** (`57.7 GiB/s` out-of-place read+write, `104 GiB/s` in-place).
