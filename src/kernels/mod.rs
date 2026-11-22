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

#[cold]
#[inline(never)]
pub(crate) fn finish_tail_cold<C: Copy, const N: usize>(
    remainder: &mut [f32],
    ctx: C,
    convert: fn(C, [f32; N]) -> [f32; N],
) {
    let mut block = [0.0; N];
    block[..remainder.len()].copy_from_slice(remainder);
    let converted = convert(ctx, block);
    remainder.copy_from_slice(&converted[..remainder.len()]);
}

/// Closure-free slice loop macro so the loop body stays directly inside the
/// enclosing `#[target_feature]` / `#[arcane]` / `#[attune]` function (Rust
/// closures do not inherit caller `#[target_feature]`, e.g. `evex512` on V4).
///
/// When given `($values, $width, $ctx, |$chunk| $body, $tail_fn)`, the `< N`
/// remainder block is outlined into `finish_tail_cold` so the hot function
/// contains only a single copy of the SIMD loop body and no stack tail buffer.
macro_rules! map_slice_inline {
    ($values:expr, $width:literal, $ctx:expr, |$chunk:ident| $body:expr, $tail_fn:expr) => {{
        let (chunks, remainder) = ($values).as_chunks_mut::<$width>();
        for slot in chunks {
            let $chunk: [f32; $width] = *slot;
            *slot = $body;
        }
        if !remainder.is_empty() {
            $crate::kernels::finish_tail_cold(remainder, $ctx, $tail_fn);
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

#[cold]
#[inline(never)]
pub(crate) fn finish_rgba_tail_cold<C: Copy>(
    remainder: &mut [f32],
    ctx: C,
    convert: fn(C, &mut [f32; 16]),
) {
    let mut block = [0.0; 16];
    block[..remainder.len()].copy_from_slice(remainder);
    convert(ctx, &mut block);
    remainder.copy_from_slice(&block[..remainder.len()]);
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
            $crate::kernels::finish_rgba_tail_cold(remainder, $ctx, $tail_fn);
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

#[cold]
#[inline(never)]
pub(crate) fn finish_slice_cold<C: Copy, const N: usize>(
    rest_chunks: &mut [[f32; N]],
    remainder: &mut [f32],
    ctx: C,
    slow: fn(C, [f32; N]) -> [f32; N],
) {
    for chunk in rest_chunks {
        *chunk = slow(ctx, *chunk);
    }
    if !remainder.is_empty() {
        let mut block = [0.0; N];
        block[..remainder.len()].copy_from_slice(remainder);
        let converted = slow(ctx, block);
        remainder.copy_from_slice(&converted[..remainder.len()]);
    }
}

/// Closure-free call-free SIMD slice fast-path macro with cold `fn` pointer
/// fallback when any chunk contains an exceptional (`NaN` / out-of-fit) lane
/// or a `< N` trailing remainder.
macro_rules! map_slice_try_inline {
    ($values:expr, $width:literal, $ctx:expr, |$chunk:ident| $fast:expr, $slow_fn:expr) => {{
        let (chunks, remainder) = ($values).as_chunks_mut::<$width>();
        let mut idx = 0;
        while idx < chunks.len() {
            let $chunk: [f32; $width] = chunks[idx];
            let Some(converted) = ($fast) else {
                $crate::kernels::finish_slice_cold(&mut chunks[idx..], remainder, $ctx, $slow_fn);
                return;
            };
            chunks[idx] = converted;
            idx += 1;
        }
        if !remainder.is_empty() {
            $crate::kernels::finish_slice_cold(&mut [], remainder, $ctx, $slow_fn);
        }
    }};
}
pub(crate) use map_slice_try_inline;

#[cold]
#[inline(never)]
pub(crate) fn finish_rgba_slice_cold<C: Copy>(
    rest_chunks: &mut [[f32; 16]],
    remainder: &mut [f32],
    ctx: C,
    slow: fn(C, &mut [f32; 16]),
) {
    for chunk in rest_chunks {
        slow(ctx, chunk);
    }
    if !remainder.is_empty() {
        let mut block = [0.0; 16];
        block[..remainder.len()].copy_from_slice(remainder);
        slow(ctx, &mut block);
        remainder.copy_from_slice(&block[..remainder.len()]);
    }
}

/// Closure-free RGBA slice fast-path macro with cold `fn` pointer fallback.
macro_rules! map_rgba_slice_try_inline {
    ($values:expr, $ctx:expr, |$chunk:ident| $fast:expr, $slow_fn:expr) => {{
        let (chunks, remainder) = ($values).as_chunks_mut::<16>();
        let remainder = remainder.as_chunks_mut::<4>().0.as_flattened_mut();
        let mut idx = 0;
        while idx < chunks.len() {
            let ok = {
                let $chunk = &mut chunks[idx];
                $fast
            };
            if !ok {
                $crate::kernels::finish_rgba_slice_cold(
                    &mut chunks[idx..],
                    remainder,
                    $ctx,
                    $slow_fn,
                );
                return;
            }
            idx += 1;
        }
        if !remainder.is_empty() {
            $crate::kernels::finish_rgba_slice_cold(&mut [], remainder, $ctx, $slow_fn);
        }
    }};
}
pub(crate) use map_rgba_slice_try_inline;

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
