# linear-srgb 0.6.13 — Plain-English Mental Model (ELI5)

`linear-srgb` does one job: convert pixel numbers between **encoded transfer curves** (sRGB, custom gamma, BT.709, PQ/HDR10, HLG) and **linear light** as fast and accurately as possible, with `#![forbid(unsafe_code)]` and `no_std` support.

It does **not** swap channels (`RGBA↔BGRA` → use `garb`) and does **not** manage image dimensions, strides, ICC profiles, or color primaries (use `zenpixels` / `zenpixels-convert`).

---

## 1. Which module do I import?

| Module | When to use it | Status |
|---|---|---|
| **`linear_srgb::default`** | **99% of callers.** Runtime SIMD dispatch (`AVX-512`, `AVX2`, `NEON`, `WASM SIMD128`, scalar fallback). Rational polynomials for `f32`, LUTs for `u8`/`u16`. | Recommended |
| **`linear_srgb::precise`** | Reference `powf` (`f32`/`f64`) and full-range `f64`-intermediate PQ reference slices (`pq_to_linear_precise_slice`, `linear_to_pq_precise_slice` behind `transfer`). | Recommended for reference/oracle |
| **`linear_srgb::target_features::{x4, x8}`** | When you are **already inside** an `#[archmage::arcane]` / `#[target_feature]` loop and want zero-overhead inlined SIMD on `[f32; 4]` / `[f32; 8]` / `&mut [f32]` without passing archmage tokens across crate boundaries. | Recommended for SIMD authors |
| **`linear_srgb::lut`** | Custom 8/10/12-bit lookup tables (`LinearTable12`, `EncodeTable12`). | Supported |
| **`linear_srgb::iec`** | Original IEC 61966-2-1 textbook sRGB constants (requires `iec` feature). Default uses C0-continuous constants with no junction discontinuity. | Opt-in legacy interop |
| `linear_srgb::tokens`, `linear_srgb::tf` | Old token-taking / separate transfer modules. | **Deprecated in 0.6.13** (removed in 0.7.0) |

---

## 2. Picking the right `default` function in 10 seconds

1. **In-place vs out-of-place (`_into`) for `f32` sRGB:**
   - Already own a `&mut [f32]` you can overwrite? Use `srgb_to_linear_slice(&mut buf)` / `linear_to_srgb_slice(&mut buf)`.
   - Reading from `&[f32]` into a separate `&mut [f32]`? Use **`srgb_to_linear_slice_into(src, dst)`** / **`linear_to_srgb_slice_into(src, dst)`** (plus `_rgba_slice_into`, `_extended_slice_into`, `_extended_rgba_slice_into`). They avoid copying the whole buffer first and preserve alpha in one pass.
   - **Length rule:** every two-slice function (`*_into`, `u8↔f32`, `u16↔f32`) **panics if `input.len() != output.len()`**.
2. **RGBA vs plain slices:**
   - Alpha is **always linear** — never run a plain sRGB function on RGBA data! Use the `_rgba_` variant (`srgb_to_linear_rgba_slice`, `srgb_to_linear_rgba_slice_into`, etc.); it leaves every 4th element untouched (or maps `u8` alpha as `a / 255.0`).
   - **Trailing `< 4` lanes on `_rgba_` slices:** RGBA slice functions operate on complete 4-channel pixels (`chunks_exact(4)`). Any trailing `len % 4` elements beyond the last complete pixel are left untouched (including when `len < 4`).
   - Need premultiplied linear alpha for compositing/resizing? Use the fused single-pass `srgb_to_linear_premultiply_rgba_slice` / `unpremultiply_linear_to_srgb_rgba_slice` (also available in `u8↔f32`).
3. **Clamped `[0, 1]` vs Extended range (`_extended_`):**
   - Clamped functions treat `<= 0.0` and `NaN` as `+0.0` and `>= 1.0` as `1.0`.
   - Extended functions (`srgb_to_linear_extended_*`, `linear_to_srgb_extended_*`) handle values `< 0.0` and `> 1.0` (e.g. Display P3 or BT.2020 gamut conversions), mirror negative signs (CSS Color 4), preserve `-0.0`, and propagate `NaN`.
4. **`u8` / `u16` integer shortcuts:**
   - `u8 → f32`: LUT-based (`srgb_u8_to_linear_slice`). `f32 → u8`: 4096-entry SIMD LUT (`linear_to_srgb_u8_slice`, exact roundtrip on all 256 `u8` codes, ±1 code on filtered intermediates).
   - `u16 → f32`: lazy 65536-entry LUT (`srgb_u16_to_linear_slice`).
   - `f32 → u16`: `linear_to_srgb_u16_slice` is exact roundtrip (polynomial); `linear_to_srgb_u16_slice_fast` uses a sqrt-indexed LUT (~6× faster, max ±1 code).
5. **Custom gamma & HDR transfer curves (`features = ["transfer"]`):**
   - `gamma_to_linear_slice(&mut buf, 2.2)` / `linear_to_gamma_slice(&mut buf, 2.2)` (no linear toe).
   - With `features = ["transfer"]`: `bt709_to_linear*`, `pq_to_linear*` (ST 2084 / HDR10), `hlg_to_linear*` (ARIB STD-B67), plus `_slice` and `_rgba_slice` variants in `linear_srgb::default`.
   - **HDR rail summary:** `PQ` clamps both decode and encode to `[0.0, 1.0]` and propagates `NaN`; `HLG` clamps `<= 0.0` and `NaN` to `+0.0` while allowing `> 1.0` inputs (`hlg_to_linear(1.5) ≈ 16.01`). See [docs/precision.md](precision.md) for the complete rail table.
   - **PQ dark toes & accuracy (0.6.13):** `default::pq_to_linear*` and `default::linear_to_pq*` run 100% in pure SIMD `f32` via `magemath` (zero `f64`, zero `libm`, bit-identical across scalar/AVX2/AVX-512/NEON/WASM). Signals `<= 7.313e-7` (linear `< 3.5e-37`, `< 3.5e-33` nits) flush cleanly to `0.0`. Dark-toe (`v < 0.02`) signal-space roundtrip error is `< 0.004` of a `u16` step; across the full `[0, 1]` range, `default` stays within `~0.34` `u16`-code-units (at most `1` quantized `u16` code) of the `f64` formula (see [docs/precision.md](precision.md)). If you specifically need the `f64::powf` spec formula (including `< 3.5e-37` subnormal/tiny tail values), call `precise::pq_to_linear_precise_slice` / `precise::linear_to_pq_precise_slice` (slice-only).

---

## 3. How `target_features::{x4, x8}` works (and why `tokens` is deprecated)

Previously, callers had to import `linear_srgb::tokens::X64V3Token` and pass it into `tokens::x8::srgb_to_linear_v3(token, arr)`. That tied `linear-srgb`'s public API to a specific `archmage` major version.

In 0.6.13, `target_features::{x4, x8}` takes plain `[f32; N]` / `&mut [f32]` with **no token parameter**:

```rust,ignore
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
{
    use archmage::{SimdToken, X64V3Token, arcane};
    use linear_srgb::target_features::x8;

    #[arcane]
    fn process_rows(_token: X64V3Token, rows: &mut [[f32; 8]]) {
        for row in rows {
            *row = x8::srgb_to_linear_v3(*row);
        }
    }

    if let Some(token) = X64V3Token::summon() {
        process_rows(token, &mut [[0.5; 8]]);
    }
}
```

- Calling `_v3` outside an `#[arcane]` / `#[target_feature]` context is a compile error (`E0133`) — put the call inside your feature-checked loop instead of writing `unsafe`.
- Functions ending in `_scalar` need no feature context.
- **Migrating from `tokens::x16` (AVX-512):** `tokens::x16` is deprecated without a direct public `target_features::x16` replacement; migrate to `linear_srgb::default::*_slice` (which dispatches AVX-512 automatically) or two `target_features::x8` calls.

---

## 4. Feature flags & WASM portability

- **`std`** (default): enables runtime SIMD dispatch (`no_std` builds still work cleanly with `default-features = false`).
- **`avx512`** (default): enables 16-wide AVX-512 dispatch (`X64V4Token`) when the CPU supports it.
- **`transfer`**: enables `bt709`, `pq`, and `hlg` in `default` and `precise`.
- **`precise-portable`**: forces single-rounding software FMA on non-fusing WASM engines so WASM matches native x86-64/AArch64 bit-for-bit. Leave it **off** for compact default WASM builds.
- **`alt`**, **`unsafe_simd`**: deprecated in 0.6.13 (emit warnings when enabled; remove from `Cargo.toml`).
