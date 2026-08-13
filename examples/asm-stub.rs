//! ASM-snapshot stubs.
//!
//! Each `stub_*` is `#[inline(never)] #[no_mangle]` so it survives as its own
//! symbol in the cross-compiled binary, with the public dispatcher's
//! per-tier code inlined into it. `scripts/dump-asm.sh` extracts each
//! stub's body via `cargo asm` and writes it to `asm-snapshots/<target>/`.
//!
//! When refactoring `src/simd.rs` (issue #23 Pattern 2), the snapshot diff
//! is the gate that proves NEON/WASM/V3/V4 codegen stayed equivalent. ASM
//! is target-deterministic on a fixed rustc version.
//!
//! NOT a runtime example — the `main` does nothing useful, it just exists
//! so the binary links.

use std::hint::black_box;

#[cfg(feature = "transfer")]
use linear_srgb::default;

// ---------- sRGB ↔ linear (clamped, plain + RGBA) ----------

#[inline(never)]
#[unsafe(no_mangle)]
pub fn stub_srgb_to_linear_slice(buf: &mut [f32]) {
    linear_srgb::default::srgb_to_linear_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn stub_srgb_to_linear_rgba_slice(buf: &mut [f32]) {
    linear_srgb::default::srgb_to_linear_rgba_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn stub_linear_to_srgb_slice(buf: &mut [f32]) {
    linear_srgb::default::linear_to_srgb_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn stub_linear_to_srgb_rgba_slice(buf: &mut [f32]) {
    linear_srgb::default::linear_to_srgb_rgba_slice(buf);
}

// ---------- sRGB ↔ linear (extended, sign-preserving CSS Color 4) ----------

#[inline(never)]
#[unsafe(no_mangle)]
pub fn stub_srgb_to_linear_extended_slice(buf: &mut [f32]) {
    linear_srgb::default::srgb_to_linear_extended_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn stub_linear_to_srgb_extended_slice(buf: &mut [f32]) {
    linear_srgb::default::linear_to_srgb_extended_slice(buf);
}

// ---------- Premultiply / unpremultiply families ----------

#[inline(never)]
#[unsafe(no_mangle)]
pub fn stub_srgb_to_linear_premultiply_rgba_slice(buf: &mut [f32]) {
    linear_srgb::default::srgb_to_linear_premultiply_rgba_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn stub_unpremultiply_linear_to_srgb_rgba_slice(buf: &mut [f32]) {
    linear_srgb::default::unpremultiply_linear_to_srgb_rgba_slice(buf);
}

#[inline(never)]
#[unsafe(no_mangle)]
#[allow(deprecated)]
pub fn stub_gamma_to_linear_premultiply_rgba_slice(buf: &mut [f32], gamma: f32) {
    linear_srgb::default::gamma_to_linear_premultiply_rgba_slice(buf, gamma);
}

#[inline(never)]
#[unsafe(no_mangle)]
#[allow(deprecated)]
pub fn stub_unpremultiply_linear_to_gamma_rgba_slice(buf: &mut [f32], gamma: f32) {
    linear_srgb::default::unpremultiply_linear_to_gamma_rgba_slice(buf, gamma);
}

// ---------- Custom gamma (already magetypes-unified — stable reference) ----------

#[inline(never)]
#[unsafe(no_mangle)]
pub fn stub_gamma_to_linear_slice(buf: &mut [f32], gamma: f32) {
    linear_srgb::default::gamma_to_linear_slice(buf, gamma);
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn stub_linear_to_gamma_slice(buf: &mut [f32], gamma: f32) {
    linear_srgb::default::linear_to_gamma_slice(buf, gamma);
}

fn main() {
    // Touch each stub through black_box so the linker can't drop them and
    // the optimizer can't hoist anything across the boundary.
    let mut buf = vec![0.5f32; 64];
    stub_srgb_to_linear_slice(black_box(&mut buf));
    stub_srgb_to_linear_rgba_slice(black_box(&mut buf));
    stub_linear_to_srgb_slice(black_box(&mut buf));
    stub_linear_to_srgb_rgba_slice(black_box(&mut buf));
    stub_srgb_to_linear_extended_slice(black_box(&mut buf));
    stub_linear_to_srgb_extended_slice(black_box(&mut buf));
    stub_srgb_to_linear_premultiply_rgba_slice(black_box(&mut buf));
    stub_unpremultiply_linear_to_srgb_rgba_slice(black_box(&mut buf));
    stub_gamma_to_linear_premultiply_rgba_slice(black_box(&mut buf), black_box(2.2));
    stub_unpremultiply_linear_to_gamma_rgba_slice(black_box(&mut buf), black_box(2.2));
    stub_gamma_to_linear_slice(black_box(&mut buf), black_box(2.2));
    stub_linear_to_gamma_slice(black_box(&mut buf), black_box(2.2));
    let _ = black_box(buf);
    let _ = black_box(default::srgb_to_linear);
}
