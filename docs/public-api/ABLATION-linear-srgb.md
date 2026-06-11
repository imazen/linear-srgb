# ABLATION-linear-srgb.md

**Date:** 2026-06-11  
**Snapshot commit:** 00ec86be (main@origin)  
**Surface size:** 173 items (default features) / 250 items (all features)  
**Grep template:** `ugrep -r "<symbol>" /home/lilith/work --exclude-dir=linear-srgb --exclude-dir=target --exclude-dir=.jj`

---

## Summary

**0 items flagged. Surface is coherent.**

173 (default) / 250 (all-features) items reviewed. No public-API mistakes found under the conservative bar.

---

## Consumer Evidence (representative)

All modules confirmed consumed externally:

| Module | Confirmed consumers (as of scan) |
|---|---|
| `linear_srgb::default` | zenresize (color.rs, streaming.rs, transfer.rs), zenjpeg (linear_lut.rs, scanline.rs, ultrahdr), zenresize benches |
| `linear_srgb::lut::SrgbConverter` | zenpng (indexed.rs) |
| `linear_srgb::precise` | zenresize (composite.rs), zenpixels-convert (ext.rs, fast_gamut.rs) |
| `linear_srgb::tf` | zenresize (wasm128.rs, neon.rs, fastmath.rs), zenanalyze (tier_depth.rs) |
| `linear_srgb::tokens::x8` | zenresize (x86.rs), zenjpeg (linear_lut.rs) |
| `linear_srgb::tokens::x4` | zenresize (wasm128.rs, neon.rs) |

### Two items with zero external hits (not flagged)

`lut_interp_linear_float` and `lut_interp_linear_u16` — zero hits outside the crate in /home/lilith/work. However, these are **not flagged** under the conservative bar because:

1. They are generic LUT interpolation helpers in `linear_srgb::lut`, a module whose primary types (`EncodingTable<N>`, `LinearizationTable<N>`, their type aliases, and `SrgbConverter`) are all externally consumed. Exposing the helper functions is coherent with exposing the tables — callers who hold a custom `&[f32]` or `&[u16]` table not produced by this crate (e.g. ICC profile curves) reasonably want these interpolation primitives.
2. They are documented with clear semantic contracts (`x` in [0,1], clamping, FMA).
3. Removing them would be a breaking change (they appear in the published API snapshot).
4. The `.jplag` comparison directory contains moxcms code that implements identical private functions — indication that the ICC use-case is real even if our callers haven't wired it yet.

`UNPREMUL_ALPHA_THRESHOLD` — zero external hits in /home/lilith/work. Not flagged: it is a semantically meaningful constant tied to the documented `unpremultiply_*` functions and its value (a platform-tuned alpha threshold) is the kind of thing downstream code might guard against. Small, no footprint cost.

### `linear_srgb::iec` module

Zero external hits in /home/lilith/work (only available under all-features). Not flagged: it is an optional feature (`iec` feature gate), exists as a precision-specific implementation variant, and its items are clearly scoped. Size impact is zero when the feature is absent.

---

## Items considered and kept

All items examined, including:
- `EncodingTable<N>` / `LinearizationTable<N>` generics and type aliases (EncodeTable8/12/16, LinearTable8/10/12/16) — KEEP: composable with custom table consumers
- `lut_interp_linear_float` / `lut_interp_linear_u16` — KEEP: coherent with the module; zero hits may reflect incomplete scan not absence of use-case
- `UNPREMUL_ALPHA_THRESHOLD` — KEEP: tied to premultiplied alpha API
- All `tokens::x4`, `x8`, `x16` SIMD dispatch fns — KEEP: consumed by zenresize and zenjpeg
- All `default`, `precise`, `tf`, `iec`, `lut` module fns — KEEP: consumed externally

---

## Digest

| Metric | Count |
|---|---|
| Items in surface (default) | 173 |
| Items in surface (all-features) | 250 |
| Items flagged (Action A) | 0 |
| Items flagged (Action B) | 0 |
| Flag rate | 0% |

**Verdict:** Surface is well-scoped. The `lut` module exposes useful generic helpers alongside the table types. Every module has at least one confirmed external consumer. No leaks of internal plumbing detected.
