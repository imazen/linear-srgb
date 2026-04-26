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
# For each (target, stub) pair we emit TWO files:
#   1. `<stub>.s`        — full assembly listing, for human review
#   2. `<stub>.essence`  — normalized opcode-only sequence, for the CI gate
#
# The `.essence` file strips registers, immediates, addressing-mode offsets,
# basic-block labels, .cfi_* directives, and section/symbol metadata. What's
# left is the instruction *shape* — same length and mnemonic sequence
# whether LLVM picks v22 or v23, .LBB12_2 or .LBB14_5, #16 or #-160. That's
# the signal we want the gate to assert (issue #24): "is the codegen still
# the same algorithm at the same instruction count?", not "are the registers
# spelled identically?". Toolchain bumps and cosmetic body reorders no
# longer trip the gate.
#
# Pin the toolchain in `.github/workflows/asm-snapshots.yml` so the .s
# files are also stable across CI runs (used for diff review, even though
# the gate doesn't assert on them).

set -euo pipefail

cd "$(dirname "$0")/.."

# Distill a full assembly listing to an opcode-essence sequence. Reads the
# raw .s on stdin, writes the normalized form to stdout. Per-target rules:
#
#   aarch64:
#     - skip directive/label/empty lines (`.section`, `.cfi_*`, `label:`, ...)
#     - keep only the mnemonic (first whitespace-separated token), uppercased
#     - tag memory accesses with " MEM" (operand contained `[` ) so a load
#       turning into a non-load shows up
#     - tag immediate-only instructions with " IMM" (operand contained `#`)
#       so e.g. `add x0, x1, x2` and `add x0, x1, #N` differ
#
#   wasm:
#     - skip `.section`, `.globl`, `.type`, `.size`, `.functype`, `.local`
#     - keep only the mnemonic, uppercased; constants/locals stay as their
#       opcode (`i32.const 5` → `I32.CONST`, `local.get 1` → `LOCAL.GET`)
#
# What this catches: new instruction types appearing, instruction-count
# changes, memory-access pattern changes, branch shape changes.
# What this lets through: register renaming, label renumbering, immediate
# constant changes, instruction scheduling within an opcode-equivalent
# sequence.
normalize_asm_essence() {
    local target="$1"
    case "$target" in
    aarch64-*)
        awk '
        # Skip directives, labels, empty lines, and pure-comment lines.
        /^[[:space:]]*$/ { next }
        /^[[:space:]]*\./ { next }
        /^[[:space:]]*[A-Za-z_.][A-Za-z0-9_.:]*:[[:space:]]*$/ { next }
        /^[[:space:]]*\/\// { next }
        {
            # Strip leading whitespace.
            sub(/^[[:space:]]+/, "")
            # Drop trailing comments.
            sub(/[[:space:]]*\/\/.*$/, "")
            # Mnemonic is the first whitespace-separated token.
            n = split($0, tok, /[[:space:]]+/)
            mnem = toupper(tok[1])
            if (mnem == "") next
            # Reassemble the operand list to inspect its shape.
            ops = ""
            for (i = 2; i <= n; i++) ops = ops " " tok[i]
            tag = ""
            if (ops ~ /\[/) tag = tag " MEM"
            if (ops ~ /#/) tag = tag " IMM"
            print mnem tag
        }
        '
        ;;
    wasm32-*)
        awk '
        /^[[:space:]]*$/ { next }
        /^[[:space:]]*\./ { next }
        /^[[:space:]]*#/ { next }
        /^[[:space:]]*[A-Za-z_.][A-Za-z0-9_.:]*:[[:space:]]*$/ { next }
        {
            sub(/^[[:space:]]+/, "")
            sub(/[[:space:]]*\/\/.*$/, "")
            n = split($0, tok, /[[:space:]]+/)
            mnem = toupper(tok[1])
            if (mnem == "") next
            print mnem
        }
        '
        ;;
    *)
        echo "normalize_asm_essence: unknown target '$target'" >&2
        return 1
        ;;
    esac
}

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
        essence_file="$out_dir/${stub}.essence"
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

        # Distill to opcode essence. See header comment for rationale.
        normalize_asm_essence "$target" < "$out_file" > "$essence_file"

        printf '  %s -> %s (%d lines, .essence %d lines)\n' \
            "$stub" "$out_file" \
            "$(wc -l < "$out_file")" \
            "$(wc -l < "$essence_file")"
    done
done

echo ""
echo "Done. Diff:"
git status -s asm-snapshots/ || true
