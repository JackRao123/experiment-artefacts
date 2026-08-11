# F2 main-rebase notes (follow-up deliverable for the ship PR)

`f2_main_rebase.patch` ports `f2.patch` (validated against `0e0b65a6`) onto
main (`ef4ea4a8`). helmholtz's third-review confirmed the logic maps 1:1; this
is the mechanical port. Per-review fixes included: the mandatory boot WARNING
for dp_size>1 + `BT_F2_PHANTOM_PARTITIONS=0` (review point 1).

## Symbol mapping (0e0b65a6 → main)

| 0e0b65a6 (api/) | main (backends/megatron_bridge/ + packing/) |
|---|---|
| `api/packing.py: PackedMicrobatches` | `packing/types.py: PackedMicrobatches` (Generic) — same trailing field |
| `api/packing.py: build_phantom_thd_cp_partition` | `backends/megatron_bridge/thd_cp.py` — identical body |
| `api/megatron_controller.py: _phantom_partitions_enabled / _dp_max_partition_count` | `backends/megatron_bridge/loss.py` (next to `_dp_reduce_sum`) |
| `api/megatron_controller.py: _pack_thd_cp_microbatches` (controller method) | `backends/megatron_bridge/packer.py: MegatronBridgePacker.pack_thd_cp_microbatches` |
| `api/megatron_controller.py: _run_forward_backward` loop | `backends/megatron_bridge/training_runner.py: _run_forward_backward` loop |
| boot warning in controller `__init__` | runner `__init__` (same mpu-derived sizes) |

## Design adjustments forced by the main layout

1. **DIP for the collective.** On main the packer owns partitioning, so the
   DP all-reduce is INJECTED: `pack_thd_cp_microbatches(...,
   template_data=..., dp_partition_count_equalizer=...)` — the runner passes
   `_dp_max_partition_count` when `dp_size > 1 and _phantom_partitions_enabled()`,
   else `None` (inert). The packer stays collective-free and unit-testable
   (codebase convention: loose coupling via DIP).
2. **Route-array walk guard.** Main's packer walks `routed_experts_arrays`
   alongside partitions via `datum_offset`. Phantom partitions (strict suffix)
   own no datums: they take `routed_arrays=None` and do NOT advance the walk.
   (Phantom `routed_experts` is None ⇒ the runner's replay branch stays
   inactive for them.)
3. **Batch type.** `MegatronBridgeBatch` replaces `Batch`; field set
   unchanged for our purposes.

## Test status

- 0e0b65a6 patch: 9/9 Mac CPU tests green + 39/39 existing packing/CE/CP-THD
  tests green (see f2.patch's test file).
- Main rebase: compiles clean; the phantom builder verified via direct-module
  load (packs, masks correct). NOTE: main's `backends/__init__.py` imports
  megatron eagerly, so `thd_cp` is NOT Mac-importable through the package
  path — the rebased unit tests run in CI/on-box, not on Mac. Port the test
  file's Section A against `thd_cp` + `packing/types` and Section B against
  `training_runner` when landing the ship PR.

## Validation

Tonight's on-box validation (ONBOX_VALIDATION_F2.md) runs against 0e0b65a6 =
what the box runs. The main rebase is for the ship PR; its on-box evidence
carries over logically (same algorithm), but CI + a fresh DP2 smoke on main
are the ship gates.
