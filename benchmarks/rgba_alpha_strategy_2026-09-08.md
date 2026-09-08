# RGBA alpha preservation: fused vs restore-after

Measured 2026-09-08 to answer whether the `*_rgba_slice` family is worth
switching to for consumers that currently transform every channel and then put
alpha back.

## Why

`zenpixels-convert/src/convert_kernels.rs` has 12 f32→f32 transfer kernels
shaped as: copy src→dst, run the plain (alpha-unaware) slice transform over
**all** channels, then `restore_alpha_f32_f32` — a scalar per-pixel pass that
rewrites the alpha lane from the untouched source row. The `*_rgba_slice`
functions do the same job in one pass, saving and restoring alpha inside the
SIMD chunk loop.

The pre-existing `save_vec` case overstates the cost of the restore-after
approach because it allocates a `Vec` for the alphas. Real consumers still hold
the source row, so `restore_from_src` was added to `benches/rgba_approach.rs`
to measure the shape they actually use.

## Numbers

sRGB→linear, RGBA f32, `cargo bench --bench rgba_approach -- --group=rgba_s2l`.
Apple Silicon, darwin arm64, NEON tier, no `target-cpu=native`.

| pixels | restore_from_src | fused `srgb_to_linear_rgba_slice` | speedup | fused vs alpha-unaware baseline |
|-------:|-----------------:|----------------------------------:|--------:|--------------------------------:|
|     64 |          90.0 ns |                           79.7 ns |   1.13x |                          1.024x |
|    256 |         379.9 ns |                          347.0 ns |   1.09x |                          1.028x |
|   1024 |        1508.8 ns |                         1373.2 ns |   1.10x |                          1.017x |
|   4096 |        6369.9 ns |                         5502.0 ns |   1.16x |                          1.019x |
|  16384 |       27961.7 ns |                        23452.8 ns |   1.19x |                          1.009x |

`save_vec` (allocating variant, for reference): 122.0 / 499.3 / 1804.2 /
7843.2 / 31880.4 ns — 1.36-1.53x slower than fused, which is why it should not
be used to size this decision.

## Findings

- **Switching to the fused form is worth ~1.09-1.19x** on the transfer kernel,
  growing with row length as the extra pass stops fitting in cache.
- **Fused alpha preservation is nearly free**: within 1-3 % of the
  alpha-unaware `srgb_to_linear_slice` baseline, and the gap *narrows* with
  size (2.4 % at 64 px, 0.9 % at 16384 px). There is no reason to avoid the
  RGBA variants on cost grounds.
- The win is a removed pass, not better math — both forms produce identical
  RGB output and identical alpha.

## Not measured

- x86 AVX2 / AVX-512 tiers. This is a NEON-only measurement; do not
  extrapolate the ratio to the `v3`/`v4` tiers.
- 3-channel (RGB, no alpha) layouts, where `restore_alpha_f32_f32` early-returns
  and the two shapes are equivalent.
