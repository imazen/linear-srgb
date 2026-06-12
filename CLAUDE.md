# linear-srgb — Project Notes

## Known Bugs

- **"ASM snapshots" workflow red on every push since 2026-05-31** (8+ consecutive
  failures through 2026-06-11). Job `ASM snapshot diff (informational)
  (wasm32-unknown-unknown)`, step `Assert essence unchanged (hard gate)`: the
  committed wasm32 ASM snapshots no longer match output under the runner's
  rustc (1.96.0 as of run 27375675758) — diff hunks around `I32.AND` /
  `LOCAL.TEE` show both shrinking and growing regions, i.e. upstream codegen
  drift, not a single added bounds check. The aarch64 matrix leg passes.
  Unrelated to the 2026-06-11 api-doc migration (1bcf6e52 touched ci.yml +
  api-surface.yml only; failures predate it by 11 days). Fix requires
  re-blessing the wasm snapshots after a human reviews the new codegen for
  regressions — do NOT regen blindly; the gate exists to catch exactly that.

  **2026-06-12 characterization (local repro, rustc 1.96.0):** the essence
  delta is NOT benign reshuffling. Per slice stub the f32 polynomial branch
  tree shrinks (−42 F32.CONST, −22 F32.LT, −16 BR_IF, −13 BLOCK) and is
  replaced by +48 CALL — including 7× `call fma` per slice stub — plus
  +72 F64.CONST and +8 each of F64.PROMOTE_F32 / F64.DIV / F64.ADD /
  F32.DEMOTE_F64. I.e. f32 `mul_add` paths (`src/mlaf.rs` `MulAdd`) now
  lower to software libm fma with f64 arithmetic on wasm32 — a real
  per-element perf regression, worst in hot conversion loops. Re-blessing
  the snapshots as-is would enshrine it. Likely fix: target-gate mlaf to
  plain mul+add on ISAs without hardware FMA (wasm32 SIMD128 has none),
  then regen + re-bless. The aarch64 leg is unaffected (native FMLA).
  Repro: `scripts/dump-asm.sh wasm32-unknown-unknown && git diff --stat`.
