# SHIM SMOKE — 1-node B200 protocol (lebesgue, for bohr, 2026-08-13)

Opportunistic de-risking for W2 (fermi-ordered, hatch (a)). Validates the
contract shim + the overlap+LoRA landmine fix + the DSA×executor interaction
on hardware using a cut-down random-init GLM-5.2 on ONE 8×B200 node — so a
boot/contract/backward failure surfaces on free B200 hours instead of burning
a B300 mission window. **This is NOT the W2 parity gate** (that stays on the
B300 box); it proves boot + contract + backward, not numerics/perf/memory at
mission scale.

## The snapshot (cut-down GLM-5.2, random-init bf16)

- Build on the box (needs the GLM-5.2-FP8 HF cache entry for config+tokenizer;
  ~3-4 min, CPU-only):
  `python3 tools/build_shim_smoke_snapshot.py --output-dir /root/lps1062/glm52_smoke`
  (script: `pp2cp8ep8/tools/build_shim_smoke_snapshot.py`; if the box has no HF
  cache, it pulls only config.json + tokenizer — small.)
- Preserved per-layer dims (hidden 6144; kv_lora 512 / kv-lora-out 576;
  q_lora 2048; qk_nope 192 / qk_rope 64 / v 256; 64 heads; indexer
  head_dim 128 / n_heads 32 / topk 2048 / freq 4 / skip-offset 3; expert
  hidden 2048; dense MLP 12288; vocab 154880) — so per-layer activation shapes
  match the real model (this is what makes the dial mem-probe bonus useful).
- Cut: 78→12 layers (smallest VPP2-legal N: chunk starts must be ≤3 or ≡3
  mod 4; N=8 can't field 4 chunks), 256→32 routed experts (snapshot stays
  ~22 GiB, builds in minutes; per-token MoE activation is topk=8-driven,
  unaffected by the count).
- New layout entry `_GLM52_DSA_PIPELINE_LAYOUTS[(12,2,2)]` = [2+emb, 4, 4,
  2+loss] (chunk starts 1/3/7/11, all DSA-legal) — on the shim branch, unit
  tested (`test_glm52_dsa_smoke_pp2_vpp2_layout` + the generic DSA-constraint
  sweep).

## Smoke protocol

1. Tree: shim branch `jackrao/lps-1062-overlap-contract-shim` @ e13de4d7
   (submodule chain carries mcore b37c01f2e = the landmine fix) + fixed wheel
   (cudnn-frontend 1.27.0 — the DSA cudnn backend needs it).
2. Config: `pp2cp8ep8/configs/trainer_shim_smoke_1node.json` (PP2/VPP2/CP2/EP2,
   seqlen 8192, LoRA 32 — LoRA is load-bearing: the landmine is the
   frozen-embedding grad root). Env: `BT_SKIP_WARMUP=1` (VPP2 auto-skip covers
   it; belt-and-braces).
3. Boot to READY, then a d2-class canary (2 datums/window, a few steps).
4. **PASS bar (pre-registered, honest for random init):** boot reaches READY;
   the first flag-ON step completes — no `return_schedule_plan` TypeError
   (contract), no stage-0-backward RuntimeError (landmine); the log carries
   "combined-1F1B schedule-plan protocol engaged"; CE ≈ ln(154880) ≈ 11.95 at
   step 0 and DECREASING across steps (= computes + trains). STOP on any
   contract/executor error → lebesgue; on the stage-0-backward RuntimeError →
   jacobi+fermi (fix-insufficient, per the pre-registered routing in
   OVERLAP_AB_DESIGN.md §5/§7).
5. **What PASS proves:** the executor contract, the landmine fix, and the
   DSA×executor interaction work on hardware. **What it does NOT prove:** the
   W2 parity gate (per-token logprob bars), perf, or 131k memory — all stay on
   the B300 box.

## BONUS (fermi's (4)): BT_DIAL_MEM_PROBE per-layer S_eager

With per-layer dims preserved, a `BT_DIAL_MEM_PROBE` run on this snapshot
approximates the real per-layer eager activation sizes (jacobi's dial-K
input). Caveats to mark in the output table: (a) expert-count sharding — 32
vs 256 experts changes the expert-activation *distribution across ranks* but
not per-token volume (topk=8 preserved); (b) routing distribution is coarser
over 32 experts; (c) random-init weights → routing is near-uniform, so
per-expert load is flatter than a trained model's. Per-layer S_eager (the
dial's sizing input) is seq×hidden-driven and transfers; read the table as
per-layer, scale by seqlen.
