# `linear-srgb 0.6.13` Math Review, Const-Generic Rail Ablation, WASM Codegen Shrinkage, Single-Intrinsic `sqrt`, & 5-Tier Codegen Audit (2026-11-21)

**Host:** `r7900x` (AMD Ryzen 9 7900X, Zen 4, 24 threads, 30.0 GiB RAM, Ubuntu; **not measured on Zen 5**).
**Runner:** `/home/lilith/work/claudehints/scripts/run-heavy` (`peak-RSS = 0.69 GiB` on `ab_vs_0612`, `0.65 GiB` on `bench-rail-ablation`, `1.10 GiB` on full CI gate matrix). No `-C target-cpu=native` passed to benchmarks.
**Baseline:** crates.io `linear-srgb = "=0.6.12"`, `garb = "=0.2.8"`, `archmage = "=0.9.22"`, `archmage-macros = "=0.9.22"`, `magetypes = "=0.9.22"`.
**Candidate:** `linear-srgb 0.6.13` (`archmage 0.9.32`, `magetypes 0.9.32`, `magemath 0.1.0`) + `garb 0.2.9`.
**Reproducers:**
- Codegen symbols (including `ds_unclamped_*` and `ds_arr_unclamped_*`): `examples/downstream_codegen.rs` (`just asm-v3`, `just asm-v4`, `just asm-neon`, `just asm-wasm128`, `just asm-wasm128-relaxed`, `just asm-all <symbol>`).
- Latency & memory-hierarchy benchmarks: `cargo bench --bench downstream_exercised --features transfer` and `just bench-rail-ablation`.

---

## 1. Exhaustive `2^32` `f32` & `65,536` `u16` Mathematical Proofs (`0` Bit Differences)

1. **Post-division `.min(1.0)` on C0 sRGB rational polynomials is 100% dead over all `2^32` `f32` bit patterns:**
   - Exhaustively evaluated across all `4,294,967,296` IEEE-754 `f32` bit patterns under both hardware FMA and separate mul+add rounding:
     - `S2L_P(x) / S2L_Q(x)` maximum on `[0, 1]` is `0.999999642` (`0x3f7ffffa`, 6 ULPs below `1.0`).
     - `L2S_P(√x) / L2S_Q(√x)` maximum on `[0, 1]` is `0.999999583` (`0x3f7ffff9`, 7 ULPs below `1.0`).
     - Removing `(yp / yq).min(one)` (`POST_DIV_MIN_ONE = 0`) produces **`0` bit differences out of `8,589,934,592` evaluations** (`0 / 65,536` `u16` differences), saving `vminps + vcmpunordps + vblendvps` (`3` ops on x86, `3` ops on WASM).
2. **PQ (`ST 2084`) coefficient-blended Horner evaluation (`eval_blended_poly_x*`):**
   - Exhaustively verified across all `1,065,353,217` `f32` values in `[0, 1]` (FMA and non-FMA): blending `[SMALL, LARGE]` coefficients per Horner step (`8` FMAs + `1` divide; `Q_SMALL[4] == Q_LARGE[4] == 1.0` shared without blend) produces **`0` bit differences** vs evaluating all four degree-4 polynomials (`16` FMAs).
3. **BT.709 & HLG rail tightening (`0` bit differences on `RAILS = true`):**
   - `BT709_DECODE_FAST` / `BT709_ENCODE_FAST` (`PowfOptions::positive_gamma().reject_subnormals().assume_in_bounds()`) elide `magemath`'s redundant subnormal prescale and 2-sided classifier while `try_bt709_*` checks `input.simd_le(65_536.0).all_true()` (`1` ordered `vcmpleps` catches both `> 65_536.0` and `NaN`).
   - `HLG_EXP_FAST` / `HLG_LOG_FAST` elide dead subnormal/range classifiers; `linear_to_hlg` handles `+inf` inline via `arg.simd_eq(+inf)` blend, making `linear_to_hlg_slice` **100% branchless and fallback-free** even for `RAILS = true`.
4. **Golden 64-bit output fingerprints (`target_features::fingerprint::entry_point_fingerprint`):**
   - `0` bit differences across all `18` native hashes (`x4/x8/x16` × `{scalar, v3, v4}` × `{plain, transfer}` over slice lengths `0..=65`) and all `12` WASM `precise-portable` hashes, plus `just probe-relaxed-wasm true` and `just probe-relaxed-wasm false`.

---

## 2. Catastrophic WASM Codegen Bloat Eliminated (`-64%..-65%` Binary, `-97%..-99%` Per-Function)

| Metric / Symbol | `0.6.12` | `0.6.13` Pre-Fix | `0.6.13` Final (`RAILS=true`) | `0.6.13` (`RAILS=false`) | Net vs `0.6.12` / Pre-Fix |
|---|---:|---:|---:|---:|---|
| Linked `downstream_codegen.wasm` (`+simd128`) | — | `358,621 B` (`351 KB`) | **`128,729 B` (`126 KB`)** | — | **`-229,892 B` (`-64.1%`, `2.79×` smaller)** |
| Linked `downstream_codegen.wasm` (`+relaxed-simd`) | — | `391,433 B` (`383 KB`) | **`136,387 B` (`133 KB`)** | — | **`-255,046 B` (`-65.2%`, `2.87×` smaller)** |
| `ds_bt709_to_linear_slice` (`wasm128`) | `545` | `12,277` | **`164`** | **`147`** | **`-70%` vs `0.6.12` (`75×` smaller vs pre-fix)** |
| `ds_gamma_to_linear_slice` (`wasm128`) | `637` | `12,619` | **`397`** | **`190`** | **`-38%` vs `0.6.12` (`32× / 66×` smaller vs pre-fix)** |
| `ds_hlg_to_linear_slice` (`wasm128`) | `403` | `7,365` | **`123`** | **`105`** | **`-69%` vs `0.6.12` (`60×` smaller vs pre-fix)** |
| `ds_linear_to_hlg_slice` (`wasm128`) | `376` | `5,375` | **`120`** | **`112`** | **`-68%` vs `0.6.12` (`45×` smaller vs pre-fix)** |
| `ds_pq_to_linear_slice` (`wasm128`) | `548` | `932` | **`151`** | **`114`** | **`-72%` vs `0.6.12` (`6.2×` smaller vs pre-fix)** |
| `ds_srgb_to_linear_slice` (`wasm128`) | `340` | `555` | **`97`** | **`86`** | **`-71%` vs `0.6.12` (`5.7×` smaller vs pre-fix)** |
| `ds_srgb_to_linear_rgba_slice` (`wasm128`) | `461` | `565` | **`78`** | — | **`-83%` vs `0.6.12` (`7.2×` smaller vs pre-fix)** |

Root causes fixed:
1. **`MATH_PORTABILITY` (`src/tf/math.rs`):** `magemath` defaults to `Portability::BitExact`, which on WASM without `relaxed-simd` expanded every `fma_f32` into an 18-instruction `f64.promote + f64.mul + f64.add + f64.demote` sequence inside 16-wide (`4 × v128`) unrolled `Powf`/`Expf`/`Logf` loops. Setting `.portability(MATH_PORTABILITY)` (`BitExact` when `USE_FUSED_MATH` is true — native FMA or `precise-portable` — and `BackendDependent` on default WASM) eliminated all `f64` emulation in default WASM SIMD while keeping `precise-portable` 100% bit-exact.
2. **`Powf::<FULL_POW>::scalar_with_token(ScalarToken, ...)` in `fixed_pow`:** Eliminated `Wasm128Token`/`X64V4Token` `highp_simd` binary64 table-gather kernel instantiations on cold out-of-range fallbacks while remaining `#![no_std]` clean.
3. **Native-width slice delegation + `#[cold] #[inline(never)]` `finish_tail_cold` outlining:** `portable_x8` and `portable_x16` now delegate `neon` and `wasm128` slices to native 4-wide `x4` (`1` RGBA pixel per `v128`, `0` tail remainder on RGBA!) and `v3` slices to native 8-wide `x8`, while outlining `< N` partial-chunk tails into cold helpers. Every slice entry point across all 5 tiers (`v3`, `v4`, `neon`, `wasm128`, `wasm128-relaxed`) is now smaller than `0.6.12`!

---

## 3. Single-Intrinsic Hardware `sqrt` Everywhere (`std` + `no_std`, Vector + Scalar)

- `src/mlaf.rs::hw_sqrt_f32` and `Mla::fast_sqrt` lower every `sqrt` (`sRGB`, `Extended sRGB`, `HLG`, `PQ`, `u16`, and `ScalarToken` vectors) to a single hardware instruction in **both** `std` and `#![no_std]` under `#![forbid(unsafe_code)]`:
  - `x86_64`: `vsqrtss` / `sqrtss` (`X64V1Token` `_mm_sqrt_ss`), `vsqrtps` on SIMD vectors (`6` `vsqrtss`/`sqrtss` instructions and **`0` software `sqrt_f32` calls** verified in `--no-default-features --features transfer` `x86_64` assembly).
  - `aarch64`: `fsqrt` (`NeonToken` `vsqrt_f32`).
  - `wasm32`: `f32.sqrt` / `f32x4.sqrt` (`Wasm128Token`).
  - Soft-float (`x86_64-unknown-none`, non-SSE `i686`): IEEE-754 `magetypes::scalar::sqrt_f32`.

---

## 4. Const-Generic Rail Ablation Across All 5 Curve Families (`RAILS = true` vs `RAILS = false`)

Crate-private `*_cfg` kernels (`src/kernels/float.rs`, `src/tf/{math,bt709,hlg,pq}.rs`; exposed to benchmarks/asm via `--cfg linear_srgb_bench_internal` in `src/__bench_rails.rs`) specialize rail enforcement at compile time across all five curve families:

| Curve / Operation | V3 Slice Insns (`0.6.12 -> Railed -> Unclamped`) | WASM128 Slice Insns (`0.6.12 -> Railed -> Unclamped`) | V3 Array Insns (`0.6.12 -> Railed -> Unclamped`) | Chained 64-RT (`X64V3` `Mops/s`) | `L1_256` (`1 KiB` `GiB/s`) | `DRAM_2M` (`8 MiB` `GiB/s`) |
|---|---:|---:|---:|---:|---:|---:|
| `srgb_to_linear` | `158 -> 113 -> 94` | `340 -> 97 -> 86` | `31 -> 32 -> 26` | `543 -> 605` (**`-10.2%`**) | `24.9 -> 27.2` (**`-8.2%`**) | `17.3 -> 18.7` (**`-7.0%`**) |
| `linear_to_srgb` | `161 -> 117 -> 97` | `447 -> 99 -> 88` | `32 -> 33 -> 27` | *(paired above)* | `17.9 -> 15.0` *(divider bound)* | `13.1 -> 13.2` (`~tie`, `vsqrtps+vdivps` port bound) |
| `gamma22_to_linear` | `258 -> 157 -> 115` | `637 -> 255 -> 190` | `69 -> 109 -> 47` | `236 -> 300` (**`-21.3%`**) | `7.31 -> 14.2` (**`-48.4%`, `1.94×`**) | `7.18 -> 11.5` (**`-37.5%`, `1.60×`**) |
| `pq_to_linear` | `173 -> 81 -> 140` | `548 -> 151 -> 114` | `47 -> 65 -> 42` | `407 -> 454` (**`-10.4%`**) | `16.4 -> 18.1` (**`-8.7%`**) | `13.0 -> 16.3` (**`-20.8%`, `1.25×`**) |
| `linear_to_pq` | `171 -> 85 -> 65` | `540 -> 153 -> 116` | `50 -> 69 -> 45` | *(paired above)* | `9.24 -> 10.9` (**`-15.2%`**) | `8.49 -> 8.85` (**`-4.1%`**) |
| `bt709_to_linear` | `332 -> 89 -> 81` | `545 -> 164 -> 147` | `56 -> 71 -> 61` | `254 -> 256` (`-1.0%`) | `8.72 -> 9.97` (**`-13.0%`**) | `8.41 -> 8.39` (`~tie`) |
| `linear_to_bt709` | `332 -> 86 -> 78` | `541 -> 158 -> 141` | `55 -> 68 -> 58` | *(paired above)* | `9.77 -> 11.6` (**`-16.4%`**) | `9.43 -> 9.78` (**`-3.6%`**) |
| `hlg_to_linear` | `270 -> 71 -> 134` | `403 -> 123 -> 105` | `40 -> 52 -> 39` | `416 -> 442` (**`-6.0%`**) | `19.4 -> 16.7` | `15.3 -> 16.2` (**`-5.3%`**) |
| `linear_to_hlg` | `195 -> 68 -> 64` | `376 -> 120 -> 112` | `39 -> 48 -> 44` | *(paired above)* | `12.2 -> 16.2` (**`-23.8%`**) | `12.0 -> 12.8` (**`-7.0%`**) |

---

## 5. `llvm-mca` Static Pipeline Latency & Reciprocal Throughput (`0.6.12` -> `0.6.13` Railed / Unclamped)

| In-Register Array Kernel | V3 (`znver4`) Insns | V3 Latency (cyc) | V3 RThroughput (cyc) | V4 (`znver4`) Insns | V4 Latency (cyc) | V4 RThroughput (cyc) | NEON (`neoverse-n1`) RThroughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| `srgb_to_linear` (`railed` / `unclamped`) | `31 -> 32 / 26` | **`47 -> 45 / 42`** | **`7.0 -> 7.0 / 6.5`** | `22 -> 23 / 17` | **`58 -> 52 / 42`** | **`7.0 -> 7.0 / 6.5`** | **`11.0 -> 11.0 / 11.0`** |
| `linear_to_srgb` (`railed` / `unclamped`) | `32 -> 33 / 27` | **`61 -> 60 / 55`** | **`8.0 -> 8.0 / 8.0`** | `23 -> 24 / 18` | **`68 -> 62 / 57`** | **`8.0 -> 8.0 / 8.0`** | **`14.0 -> 14.0 / 14.0`** |
| `srgb_u16_to_linear` | **`33 -> 32`** | **`53 -> 50`** | **`8.0 -> 8.0`** | **`24 -> 22`** | **`62 -> 56`** | **`7.5 -> 7.5`** | n/a |
| `linear_to_srgb_u16` | **`41 -> 40`** | **`80 -> 78`** | **`9.0 -> 9.0`** | **`29 -> 28`** | **`95 -> 88`** | **`9.0 -> 9.0`** | n/a |
| `gamma_to_linear` (`railed` / `unclamped`) | `69 -> 109 / 47` | `98 -> 109 / 84` | `12.5 -> 21.0 / 9.5` | `53 -> 91 / 34` | **`139 -> 136 / 92`** | `12.5 -> 20.0 / 9.5` | `24.0 -> 33.0 / 13.0` |
| `pq_to_linear` (`railed` / `unclamped`) | `47 -> 65 / 42` | `50 -> 53 / 47` | `10.5 -> 12.5 / 10.5` | `31 -> 56 / 34` | `61 -> 65 / 56` | `10.5 -> 12.5 / 12.0` | `18.0 -> 20.5 / 18.0` |
| `linear_to_pq` (`railed` / `unclamped`) | `50 -> 69 / 45` | `76 -> 84 / 75` | **`16.0 -> 13.5 / 13.0`** | `34 -> 59 / 36` | `84 -> 96 / 86` | `16.0 -> 23.0 / 23.0` | **`28.0 -> 22.0 / 21.0`** |
| `bt709_to_linear` (`railed` / `unclamped`) | `56 -> 71 / 61` | `90 -> 104 / 104` | `12.0 -> 14.0 / 13.0` | `38 -> 53 / 41` | `105 -> 129 / 126` | `12.0 -> 14.0 / 13.0` | **`27.0 -> 20.0 / 20.0`** |
| `linear_to_bt709` (`railed` / `unclamped`) | `55 -> 68 / 58` | `80 -> 95 / 92` | `12.0 -> 13.5 / 12.5` | `38 -> 52 / 40` | `95 -> 118 / 116` | `12.0 -> 13.5 / 12.5` | **`20.0 -> 19.0 / 17.0`** |
| `hlg_to_linear` (`railed` / `unclamped`) | `40 -> 52 / 39` | `58 -> 63 / 60` | `8.0 -> 9.0 / 8.0` | `28 -> 40 / 27` | `69 -> 84 / 68` | `8.0 -> 9.0 / 8.0` | `12.0 -> 14.5 / 11.0` |
| `linear_to_hlg` (`railed` / `unclamped`) | `39 -> 48 / 44` | `53 -> 66 / 63` | `8.5 -> 10.0 / 10.0` | `28 -> 35 / 29` | `82 -> 98 / 90` | `8.5 -> 10.0 / 10.0` | `16.0 -> 16.5 / 16.0` |

---

## 6. Paired `zenbench` Runtime Results (`0.6.12` vs `0.6.13`, Zen 4 `r7900x`)

| Benchmark | `0.6.13` Pre-Fix vs `0.6.12` | `0.6.13` Final (`RAILS=true`) vs `0.6.12` | `0.6.12` Throughput | `0.6.13` (`RAILS=true`) | `0.6.13` (`RAILS=false`) |
|---|---:|---:|---:|---:|---:|
| `lat_chain64_srgb_rt_x8` | `+5.5%..+5.6%` | **`-2.8%..-2.6%`** | `533 Mops/s` | **`548 Mops/s`** | **`605 Mops/s`** |
| `L1_256/srgb_to_linear_slice` | `-0.8%..+0.3%` | **`-0.4%..+0.1%` (~tie)** | `20.0 GiB/s` | **`20.0 GiB/s`** | **`27.2 GiB/s`** |
| `L1_256/linear_to_srgb_slice` | `-0.0%..+1.0%` | **`-0.7%..-0.2%` (~tie)** | `14.6 GiB/s` | **`14.7 GiB/s`** | `15.0 GiB/s` |
| `L1_256/srgb_to_linear_rgba_slice` | `-75.4%..-74.7%` | **`-81.4%..-81.0%` (5.1×)** | `4.74 GiB/s` | **`24.0 GiB/s`** | — |
| `L1_256/hlg_to_linear_slice` | — | **`-13.6%..-10.3%` (1.15×)** | `14.3 GiB/s` | **`16.4 GiB/s`** | `16.7 GiB/s` |
| `DRAM_2M/srgb_to_linear_slice` | `-2.8%..+3.5%` | **`-1.4%..+1.5%` (~tie)** | `17.3 GiB/s` | **`17.3 GiB/s`** | **`18.7 GiB/s`** |
| `DRAM_2M/linear_to_srgb_slice` | `-3.6%..+2.7%` | **`-3.4%..+1.3%` (~tie)** | `13.1 GiB/s` | **`13.2 GiB/s`** | **`13.2 GiB/s`** |
| `DRAM_2M/srgb_to_linear_extended_slice` | `-1.6%..+6.6%` | **`-2.8%..+1.3%` (~tie)** | `14.4 GiB/s` | **`14.5 GiB/s`** | — |
| `DRAM_2M/linear_to_srgb_extended_slice` | `-3.4%..+4.6%` | **`-2.1%..+2.6%` (~tie)** | `11.6 GiB/s` | **`11.6 GiB/s`** | — |
| `DRAM_2M/srgb_to_linear_rgba_slice` | `-72.4%..-68.5%` | **`-73.7%..-69.1%` (3.5×)** | `4.25 GiB/s` | **`14.9 GiB/s`** | — |
| `DRAM_2M/srgb_to_linear_premul_rgba` | `-73.7%..-64.4%` | **`-70.9%..-69.6%` (3.4×)** | `4.11 GiB/s` | **`14.1 GiB/s`** | — |
| `DRAM_2M/unpremul_linear_to_srgb_rgba` | `-17.3%..-11.7%` | **`-18.7%..-14.8%` (1.20×)** | `8.27 GiB/s` | **`9.93 GiB/s`** | — |
| `DRAM_2M/pq_to_linear_slice` | `+3.1%..+6.1%` | **`-15.1%..-10.4%` (1.14×)** | `11.2 GiB/s` | **`12.8 GiB/s`** | **`16.3 GiB/s` (1.46×)** |
| `DRAM_2M/linear_to_pq_slice` | `-10.0%..-9.0%` | **`-21.8%..-18.6%` (1.25×)** | `6.80 GiB/s` | **`8.52 GiB/s`** | **`8.85 GiB/s` (1.30×)** |
| `DRAM_2M/hlg_to_linear_slice` | `+6.8%..+8.5%` | **`-0.0%..+4.5%` (~tie)** | `15.7 GiB/s` | **`15.4 GiB/s`** | **`16.2 GiB/s`** |
| `DRAM_2M/linear_to_hlg_slice` | `+46.2%..+53.0%` | `+12.8%..+16.5%` (`+inf`/NaN/C1 rails) | `14.0 GiB/s` | **`12.2 GiB/s`** | **`12.8 GiB/s`** |
| `DRAM_2M/bt709_to_linear_slice` | `+100.9%..+104.9%` | `+5.5%..+10.8%` (domain/C1 rails) | `9.08 GiB/s` | **`8.40 GiB/s`** | `8.39 GiB/s` |
| `DRAM_2M/linear_to_bt709_slice` | `+35.1%..+35.8%` | `+11.4%..+15.5%` (domain rails) | `10.7 GiB/s` | **`9.36 GiB/s`** | **`9.78 GiB/s`** |
| `DRAM_2M/gamma_to_linear_slice_2.2` | `+13.5%..+14.2%` | `+7.8%..+14.1%` railed / **`-31.9%` unclamped** | `7.83 GiB/s` | `7.06 GiB/s` | **`11.5 GiB/s` (1.47×)** |
| `DRAM_2M/srgb_u8_to_linear_slice` | `-0.5%..+0.3%` | **`-0.4%..+0.0%` (~tie)** | `5.32 Gelem/s` | **`5.34 Gelem/s`** | — |
| `DRAM_2M/linear_to_srgb_u8_slice` | `-0.9%..-0.3%` | **`-0.4%..+0.1%` (~tie)** | `3.60 Gelem/s` | **`3.61 Gelem/s`** | — |
| `DRAM_2M/srgb_u16_to_linear_slice` | `-0.4%..+0.5%` | **`-1.5%..-1.0%`** | `6.09 Gelem/s` | **`6.14 Gelem/s`** | — |
| `DRAM_2M/linear_to_srgb_u16_slice` | `+9.2%..+10.8%` | **`-1.1%..-0.7%`** | `4.39 Gelem/s` | **`4.44 Gelem/s`** | — |
| `DRAM_2M/linear_to_srgb_u16_slice_fast` | `-1.6%..+0.9%` | **`-1.5%..+0.5%` (~tie)** | `1.98 Gelem/s` | **`1.98 Gelem/s`** | — |

---

## 7. Instruction Latency vs Divider Throughput vs Memory Bandwidth Regime Analysis

1. **In-register fused pipelines (`zenpixels-convert::fast_gamut`, `zenjpeg` linear LUTs):**
   - Governed by **critical-path instruction latency** (`45 -> 42` cycles `srgb_to_linear` + `60 -> 55` cycles `linear_to_srgb` on Zen 4 V3), **not** reciprocal throughput (`7–8` cycles). Const-generic rail ablation shortens serial roundtrip chains across every curve family (`sRGB -10.2%`, `PQ -10.4%`, `HLG -6.0%`, `gamma -21.3%`).
2. **Divider-port bound vs ALU/FMA-rail bound slice sweeps:**
   - Single-core DDR5 bandwidth on `r7900x` (`123.9–130.8 GiB/s` calibrated) is `5×–18×` higher than single-core `vdivps`/`vsqrtps`/FMA throughput (`7.1–27.2 GiB/s`), so float slices remain compute-bound even at `8 MiB` DRAM.
   - Where `RThroughput` is pinned by hardware `vsqrtps + vdivps` (`linear_to_srgb`: `8.0` cyc both railed and unclamped), slice throughput stays at the divider ceiling (`13.2 GiB/s`) while latency drops (`-10.2%`). Where rails add ALU/FMA/blend work (`gamma`: `21.0 -> 9.5` cyc RThroughput = **`1.94×` L1 / `1.60×` DRAM**; `pq_to_linear`: `12.5 -> 10.5` cyc = **`1.25×` DRAM**; `linear_to_hlg`: **`1.33×` L1**; `srgb_to_linear`: **`+8%` L1 / `+8%` DRAM**), rail ablation accelerates both latency and `8 MiB` DRAM streaming.
