#!/usr/bin/env bash
# Run under a resource wrapper. OUTPUT must be a fresh, external directory.
set -euo pipefail
output=$1
garb_repo=${2:-../garb}
mkdir "$output"
output=$(cd "$output" && pwd)
root=$(cd "$(dirname "$0")/../.." && pwd)
garb_repo=$(cd "$garb_repo" && pwd)
python3 "$root/tools/precision-review/generate.py" "$output/released" "$garb_repo" --released
python3 "$root/tools/precision-review/generate.py" "$output/current" "$garb_repo"
export CARGO_TARGET_DIR="$output/target"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
AUDIT_BASE="$output/baseline" AUDIT_SAVE=1 \
    cargo run --release --manifest-path "$output/released/Cargo.toml" --bin precision-review
AUDIT_BASE="$output/baseline" AUDIT_CAPTURE="$output/current-samples" \
    cargo run --release --manifest-path "$output/current/Cargo.toml" --bin precision-review
cargo run --release --manifest-path "$output/released/Cargo.toml" --bin oracle
cargo run --release --manifest-path "$output/current/Cargo.toml" --bin oracle
