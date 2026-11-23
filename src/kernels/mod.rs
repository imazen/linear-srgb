//! Private numerical kernels shared by target-feature entry points and runtime dispatch.
pub(crate) mod float;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
pub(crate) mod x16;
pub(crate) mod x4;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
pub(crate) mod x8;

/// A slice tail padded to one vector block, or `None` for an empty tail.
///
/// Tails run through the same kernel as the body, so a value converts to the same bits
/// wherever it sits in a slice and however a row is split.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn tail_block<const N: usize>(tail: &[f32]) -> Option<[f32; N]> {
    if tail.is_empty() {
        return None;
    }
    let mut block = [0.0; N];
    block[..tail.len()].copy_from_slice(tail);
    Some(block)
}

/// Apply one array kernel to full blocks and one padded tail (scalar helper).
#[inline(always)]
pub(crate) fn map_slice<const N: usize>(
    values: &mut [f32],
    mut convert: impl FnMut([f32; N]) -> [f32; N],
) {
    let (chunks, remainder) = values.as_chunks_mut::<N>();
    for chunk in chunks {
        *chunk = convert(*chunk);
    }
    if !remainder.is_empty() {
        let mut block = [0.0; N];
        block[..remainder.len()].copy_from_slice(remainder);
        let converted = convert(block);
        remainder.copy_from_slice(&converted[..remainder.len()]);
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[cold]
#[inline(never)]
pub(crate) fn finish_tail_cold<C: Copy, const N: usize>(
    remainder: &mut [f32],
    ctx: C,
    convert: fn(C, [f32; N]) -> [f32; N],
) {
    const { assert!(N.is_power_of_two()) };
    let n = remainder.len() & (N - 1);
    let rem = &mut remainder[..n];
    let mut block = [0.0; N];
    block[..n].copy_from_slice(rem);
    let converted = convert(ctx, block);
    rem.copy_from_slice(&converted[..n]);
}

/// Closure-free slice loop macro so the loop body stays directly inside the
/// enclosing `#[target_feature]` / `#[arcane]` / `#[attune]` function (Rust
/// closures and nested `fn` items do not inherit caller `#[target_feature]`,
/// e.g. `avx2,fma` / `evex512` on x86-64).
///
/// On non-x86_64 targets (`aarch64`, `wasm32`, where SIMD target features are
/// compile-time global), the `< N` remainder is outlined into `finish_tail_cold`
/// to keep binary size small. On `x86_64`, the single padded remainder block
/// stays inside the caller's `#[target_feature]` context so it executes native
/// AVX2/AVX-512 without SSE2 `__simd_inner_*` shims.
macro_rules! map_slice_inline {
    ($values:expr, $width:literal, $ctx:expr, |$chunk:ident| $body:expr, $tail_fn:expr) => {{
        let (chunks, remainder) = ($values).as_chunks_mut::<$width>();
        for slot in chunks {
            let $chunk: [f32; $width] = *slot;
            *slot = $body;
        }
        if !remainder.is_empty() {
            #[cfg(target_arch = "x86_64")]
            {
                let _ = &$ctx;
                let mut block = [0.0; $width];
                block[..remainder.len()].copy_from_slice(remainder);
                let $chunk: [f32; $width] = block;
                let converted = $body;
                remainder.copy_from_slice(&converted[..remainder.len()]);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                $crate::kernels::finish_tail_cold(remainder, $ctx, $tail_fn);
            }
        }
    }};
    ($values:expr, $width:literal, |$chunk:ident| $body:expr) => {{
        let (chunks, remainder) = ($values).as_chunks_mut::<$width>();
        for slot in chunks {
            let $chunk: [f32; $width] = *slot;
            *slot = $body;
        }
        if !remainder.is_empty() {
            let mut block = [0.0; $width];
            block[..remainder.len()].copy_from_slice(remainder);
            let $chunk: [f32; $width] = block;
            let converted = $body;
            remainder.copy_from_slice(&converted[..remainder.len()]);
        }
    }};
}
pub(crate) use map_slice_inline;

/// Closure-free out-of-place (`input -> output`) slice loop macro preserving
/// caller `#[target_feature]`, with a padded tail block for position-independent
/// rounding.
macro_rules! map_slice_into_inline {
    ($input:expr, $output:expr, $width:literal, |$chunk:ident| $body:expr) => {{
        assert_eq!(($input).len(), ($output).len());
        let (in_chunks, in_rem) = ($input).as_chunks::<$width>();
        let (out_chunks, out_rem) = ($output).as_chunks_mut::<$width>();
        for (src_slot, dst_slot) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            let $chunk: [f32; $width] = *src_slot;
            *dst_slot = $body;
        }
        if !in_rem.is_empty() {
            let mut block = [0.0; $width];
            block[..in_rem.len()].copy_from_slice(in_rem);
            let $chunk: [f32; $width] = block;
            let converted = $body;
            out_rem.copy_from_slice(&converted[..out_rem.len()]);
        }
    }};
}
pub(crate) use map_slice_into_inline;

#[cfg(not(target_arch = "x86_64"))]
#[cold]
#[inline(never)]
pub(crate) fn finish_rgba_tail_cold<C: Copy>(
    remainder: &mut [f32],
    ctx: C,
    convert: fn(C, &mut [f32; 16]),
) {
    let n = remainder.len() & 12;
    let rem = &mut remainder[..n];
    let mut block = [0.0; 16];
    block[..n].copy_from_slice(rem);
    convert(ctx, &mut block);
    rem.copy_from_slice(&block[..n]);
}

/// Closure-free 16-lane RGBA slice loop macro (preserves caller `#[target_feature]`).
macro_rules! map_rgba_slice_inline {
    ($values:expr, $ctx:expr, |$chunk:ident| $body:block, $tail_fn:expr) => {{
        let (chunks, remainder) = ($values).as_chunks_mut::<16>();
        for $chunk in chunks {
            $body
        }
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        if !remainder.is_empty() {
            #[cfg(target_arch = "x86_64")]
            {
                let _ = &$ctx;
                let mut block = [0.0; 16];
                block[..remainder.len()].copy_from_slice(remainder);
                {
                    let $chunk = &mut block;
                    $body
                }
                remainder.copy_from_slice(&block[..remainder.len()]);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                $crate::kernels::finish_rgba_tail_cold(remainder, $ctx, $tail_fn);
            }
        }
    }};
    ($values:expr, |$chunk:ident| $body:block) => {{
        let (chunks, remainder) = ($values).as_chunks_mut::<16>();
        for $chunk in chunks {
            $body
        }
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        if !remainder.is_empty() {
            let mut block = [0.0; 16];
            block[..remainder.len()].copy_from_slice(remainder);
            {
                let $chunk = &mut block;
                $body
            }
            remainder.copy_from_slice(&block[..remainder.len()]);
        }
    }};
}
pub(crate) use map_rgba_slice_inline;

/// Closure-free out-of-place 16-lane RGBA slice loop macro preserving caller
/// `#[target_feature]`. Trailing elements that don't form a complete RGBA pixel
/// are ignored (matching in-place `*_rgba_slice` and `*_u8_rgba_slice`).
macro_rules! map_rgba_slice_into_inline {
    ($input:expr, $output:expr, |$src_chunk:ident, $dst_chunk:ident| $body:block) => {{
        assert_eq!(($input).len(), ($output).len());
        let (in_chunks, in_rem) = ($input).as_chunks::<16>();
        let (out_chunks, out_rem) = ($output).as_chunks_mut::<16>();
        for ($src_chunk, $dst_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
            $body
        }
        let in_rem = in_rem.as_chunks::<4>().0.as_flattened();
        let out_rem = out_rem.as_chunks_mut::<4>().0.as_flattened_mut();
        if !in_rem.is_empty() {
            let mut src_block = [0.0; 16];
            let mut dst_block = [0.0; 16];
            src_block[..in_rem.len()].copy_from_slice(in_rem);
            {
                let $src_chunk = &src_block;
                let $dst_chunk = &mut dst_block;
                $body
            }
            out_rem.copy_from_slice(&dst_block[..out_rem.len()]);
        }
    }};
}
pub(crate) use map_rgba_slice_into_inline;

/// Keep scalar HDR arrays on the same arithmetic policy as their SIMD peers.
#[cfg(feature = "transfer")]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn scalar_array<const N: usize>(
    values: [f32; N],
    vector: impl FnOnce([f32; N]) -> [f32; N],
    fused: impl FnMut(f32) -> f32,
) -> [f32; N] {
    if crate::tf::polynomial::USE_FUSED_MATH {
        values.map(fused)
    } else {
        vector(values)
    }
}

/// Preserve per-sample fused scalar loops and relaxed-WASM padded vector tails.
#[inline(always)]
pub(crate) fn scalar_slice<const N: usize>(
    values: &mut [f32],
    vector: impl FnMut([f32; N]) -> [f32; N],
    mut fused: impl FnMut(f32) -> f32,
) {
    if crate::tf::polynomial::USE_FUSED_MATH {
        for value in values {
            *value = fused(*value);
        }
    } else {
        map_slice::<N>(values, vector);
    }
}
