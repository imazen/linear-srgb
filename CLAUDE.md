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
