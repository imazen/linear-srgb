//! Private numerical kernels shared by target-feature entry points and runtime dispatch.
pub(crate) mod float;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub(crate) mod x16;
pub(crate) mod x4;
#[cfg(target_arch = "x86_64")]
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

/// Writes a converted [`tail_block`] back.
#[inline(always)]
pub(crate) fn store_tail<const N: usize>(tail: &mut [f32], block: Option<&[f32; N]>) {
    if let Some(block) = block {
        tail.copy_from_slice(&block[..tail.len()]);
    }
}

/// Apply one array kernel to full blocks and one padded tail.
#[inline(always)]
pub(crate) fn map_slice<const N: usize>(
    values: &mut [f32],
    mut convert: impl FnMut([f32; N]) -> [f32; N],
) {
    let (chunks, remainder) = values.as_chunks_mut::<N>();
    let mut tail = tail_block::<N>(remainder);
    for chunk in chunks.iter_mut().chain(tail.as_mut()) {
        *chunk = convert(*chunk);
    }
    store_tail(remainder, tail.as_ref());
}

/// Keep scalar HDR arrays on the same arithmetic policy as their SIMD peers.
#[cfg(feature = "transfer")]
#[inline(always)]
pub(crate) fn scalar_array<const N: usize>(
    values: [f32; N],
    vector: impl FnOnce([f32; N]) -> [f32; N],
    fused: impl FnMut(f32) -> f32,
) -> [f32; N] {
    if crate::tf::fast_math::USE_FUSED_MATH {
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
    if crate::tf::fast_math::USE_FUSED_MATH {
        for value in values {
            *value = fused(*value);
        }
    } else {
        map_slice::<N>(values, vector);
    }
}
