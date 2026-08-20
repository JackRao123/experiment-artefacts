# W1b blockK OOM evidence note — grothendieck, 2026-08-14 ~04:4x CDT

Companion to the poller CSVs in this folder. Box-side trainer_srun.log files
for the two K=21 boots were OVERWRITTEN by later boots (start_trainer.sh
truncates); this note preserves the verbatim numbers captured while they
lived, plus the K=25 peak line. All box times UTC; Mac = CDT.

## Boot #1 — blockK21, trainer-internal warmup OOM (no BT_SKIP_WARMUP)

Config: trainer_pp2cp8ep8_131k_blockK21.json (recompute {full, block, 21}).
Env: BT_SAVE_STATE_SYNC=1, BT_PROFILE_RANKS=0,8. Slurm job 19.
Died 07:42:33, ~14 min into boot, immediately after weight load, inside
`controller.run_startup_warmup()` (backend.py run_startup_warmup — the
seq-64 warmup datum PADDED to the full 131k buffer = full-shape M=1
fwd+bwd on unsettled weight-load transients).

OOM site: rank15 (node 1 = stage 1), forward path
transformer_block → recompute.checkpointed_forward → transformer_layer
→ _forward_mlp → moe_layer.routed_experts_compute → experts.forward →
bias_act_func (`intermediate_parallel.to(original_dtype)`).

Verbatim torch error:
> torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 872.00 MiB.
> GPU 7 has a total capacity of 267.69 GiB of which 305.19 MiB is free.
> Including non-PyTorch memory, this process has 266.76 GiB memory in use.
> Process 14960 has 618.00 MiB memory in use. Of the allocated memory
> 260.33 GiB is allocated by PyTorch, and 323.82 MiB is reserved by PyTorch
> but unallocated.

## Boot #2 — blockK21 + BT_SKIP_WARMUP=1, bench warmup0 OOM

Healthy in ~6 min (warm cache, warmup skipped). d2 bench
(`run_bench2c.sh w1b-blockK21-131k-d2 --seq-len 131072 --num-gpus 16
--datums 2 --repeats 2 --canary-json w1a-131k-d2.json`) died in warmup0 —
no window completed. lovelace (supervisor) readings at death: node
smw3h6mp-0001 (stage 1) GPUs at 266.3–266.7 GiB reserved trying to
allocate 1.18–2.36 GiB; rank15 died first (same rank class as boot #1);
rank14 at 266.6. Surviving curve: `w1b-blockK21-131k-d2_mem/*.csv`
(this folder). Verdict (lovelace, definitive): blockK21 @131k OOMs on the
first full-shape fwd+bwd EVEN POST-SETTLE.

## Boot #3 — blockK25 + BT_SKIP_WARMUP=1 (lovelace driving)

Config: trainer_pp2cp8ep8_131k_blockK25.json (num_layers 25).
d2 warmup0 COMPLETED then the run was terminated (no mains, no result
json):
`[w1b-blockK25-131k-d2 warmup0] fb=193.4s (42 tok/s/GPU) optim=6.4s
loss=12.324204057342968 gn=0.41854017972946167` — canary IN BAND
(loss 12.2–12.4 ✓, gn 0.36–0.49 ✓).

Trainer peak line (current trainer_srun.log, dispatcher_worker, optim_step
op c1d5a48e6aec4e5f8c1a0800b1610e8a, step=1.0):
`peak_reserved_bytes=275802750976 peak_allocated_bytes=271916234752
device_total_bytes=287428771840`
= **peak reserved 256.83 GiB / allocated 253.25 GiB / capacity 267.69 GiB**
— SURVIVED but over the 255 GiB fit bar by ~1.8 GiB. d16 projection off
the W1a d2→d16 base growth (+25 GiB) lands ~282 = unreachable.

## Constants cross-check (pauli's correction, for joining)

- pauli fitted: base 147 + 2×(21×0.19 + 19×2.94) = 266.7 = measured
  266.3–266.7 (boot #2, I=2) → S_eager = 2.94 GiB/layer/mb (e−c = 2.745).
- K=25 prediction was 244.6; measured warmup0 peak reserved 256.8 —
  residual +12.2 GiB ≈ the optimizer-state materialization pauli flagged
  (the 266.5 reference predated it; the 256.8 reading includes it).
