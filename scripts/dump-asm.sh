#!/usr/bin/env bash
#
# Dump per-target ASM snapshots of the public dispatchers in `examples/asm-stub.rs`.
#
# Output: `asm-snapshots/<target>/<stub_name>.s` — one file per stub per target.
# CI re-runs this and `git diff --exit-code asm-snapshots/` to catch codegen
# regressions during the issue #23 Pattern 2 refactor (chunk-size unification
# on NEON/WASM via `f32x16` polyfill must produce equivalent machine code to
# the hand-written 4-wide loops).
#
# Usage:
#   scripts/dump-asm.sh                # all targets
#   scripts/dump-asm.sh aarch64        # one target (matches first column below)
#
# Targets are scoped to where Pattern 2 carries the highest codegen risk —
# the chunk-size unification (4-wide → 16-wide via `f32x16` polyfill) on:
#   - aarch64-unknown-linux-gnu : NEON 4-wide → 4× f32x4 unrolled
#   - wasm32-unknown-unknown    : WASM SIMD128 same shape
#
# x86_64 V3/V4 already use 8-/16-wide chunks natively; their dispatch goes
# through separate `__arcane_*` symbols (not the stub body). x86_64 verification
# leans on `cargo test --all-features --release` + local tango bench.
#
# Determinism note: ASM is stable for a given rustc version + target.
# Toolchain bumps will require regenerating snapshots — the diff is the signal.

set -euo pipefail

cd "$(dirname "$0")/.."

# Stubs declared in examples/asm-stub.rs. Order = file order = snapshot order.
STUBS=(
    stub_srgb_to_linear_slice
    stub_srgb_to_linear_rgba_slice
    stub_linear_to_srgb_slice
    stub_linear_to_srgb_rgba_slice
    stub_srgb_to_linear_extended_slice
    stub_linear_to_srgb_extended_slice
    stub_srgb_to_linear_premultiply_rgba_slice
    stub_unpremultiply_linear_to_srgb_rgba_slice
    stub_gamma_to_linear_premultiply_rgba_slice
    stub_unpremultiply_linear_to_gamma_rgba_slice
    stub_gamma_to_linear_slice
    stub_linear_to_gamma_slice
)

# Each entry: <target-triple>|<feature-set>|<extra-rustflags>|<extra-cargo-asm-args>
TARGETS=(
    "aarch64-unknown-linux-gnu|transfer||"
    "wasm32-unknown-unknown|transfer|-C target-feature=+simd128|--wasm"
)

filter="${1:-}"

for entry in "${TARGETS[@]}"; do
    IFS='|' read -r target features rustflags asmargs <<< "$entry"

    if [[ -n "$filter" && "$target" != *"$filter"* ]]; then
        continue
    fi

    echo "==> $target (features=$features)"
    out_dir="asm-snapshots/$target"
    mkdir -p "$out_dir"

    # Pre-build once so cargo-asm doesn't recompile per stub. Errors from this
    # step (missing cross linker, feature mismatch, etc.) are surfaced to the
    # CI log — silencing them once cost a debugging round.
    RUSTFLAGS="$rustflags" cargo build --release \
        --example asm-stub \
        --features "$features" \
        --target "$target"

    for stub in "${STUBS[@]}"; do
        out_file="$out_dir/${stub}.s"
        # shellcheck disable=SC2086
        RUSTFLAGS="$rustflags" cargo asm \
            --example asm-stub \
            --features "$features" \
            --target "$target" \
            $asmargs \
            "$stub" \
            2>/dev/null > "$out_file"
        # cargo-asm sometimes prepends an empty line; trim leading blanks.
        sed -i '/./,$!d' "$out_file"
        printf '  %s -> %s (%d lines)\n' "$stub" "$out_file" "$(wc -l < "$out_file")"
    done
done

echo ""
echo "Done. Diff:"
git status -s asm-snapshots/ || true
