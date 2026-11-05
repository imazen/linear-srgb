#!/usr/bin/env bash
set -euo pipefail
directory=$1
printf 'strategy\tfile_bytes\n'
for strategy in lut12 lut16 corrected floor_corrected polynomial; do
    ENCODE_SIZE_SOURCE="$directory/$strategy.rs" RUSTFLAGS="--cfg encode_size" \
        cargo build --release --example u8_encode_size
    test ! -e "$directory/$strategy-binary"
    cp target/release/examples/u8_encode_size "$directory/$strategy-binary"
    printf '%s\t' "$strategy"
    stat -c '%s' "$directory/$strategy-binary"
    size "$directory/$strategy-binary"
    sha256sum "$directory/$strategy-binary"
done
