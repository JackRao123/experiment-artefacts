# GC threshold validation (LPS-1065) — simulated B200 on B300

Prompt: `prompts/d.md`. Box: `tj-qvg98rq` (4 nodes × 8 B300, 275040 MiB ≈ 268.6 GiB/GPU;
nvidia-smi name "L20D" = B300 shim). Trainer: main @ `061947ac` (worktree on box).
Config: golden GLM-5.2-FP8 B200 256k — TP1/PP1/EP32/CP32/ETP1, 32 GPUs
(`trainer-config-glm52-b200-256k.json`), LoRA rank 32, `zai-org/GLM-5.2-FP8` from team HF cache.

Driver: `tools/profile_driver.py` from `lps_1062_perf` (reused read-only), run on the
rank-0 node against `127.0.0.1:8001`. Per arm: `--seq-len 262144 --datums 1 --num-gpus 32
--lora-rank 32 --control-repeats 7 --memory-profile --runtime-profile`.

NVML: `nvml_poller.py` (pynvml direct, same counters as /debug/nvml) at 10 Hz on all 4
nodes, whole run duration. Files: `nvml/<arm>/<host>.jsonl`.

## Arms

| arm | PYTORCH_CUDA_ALLOC_CONF | intent |
|---|---|---|
| off | `expandable_segments:True` | control |
| on | `...,garbage_collection_threshold:0.63` | simulate B200@0.95 trigger point (~170 GiB) |
| stress35 | `...,garbage_collection_threshold:0.35` | force the GC to fire every step (mechanism stress test) |

0.63 = 0.95 × 179/268.6 (B200 usable ≈179 GiB → trigger ≈170 GiB; /268.6 → 0.633).

## Box gotchas encountered (for the next run)

- The ssh leader (`tj-qvg98rq`) was slurm **node_rank 2**, not rank 0. The trainer
  HTTP API (port 8001) binds on slurm rank 0 only (node `b300-1-4wprtzyj-0003` =
  `tj-qvg98rq-3`). `wait_trainer_health.sh` on the ssh leader polls 127.0.0.1:8001 and
  can never succeed — run it on the rank-0 node (find it via `squeue -o "%N"` first
  entry). `ss` is not installed; check `/proc/net/tcp` for `:1F41` (8001) instead.
- An idle healthy trainer shows **100% GPU util** — the `_peer_loop` NCCL broadcast
  busy-waits on GPU. 100% util ≠ working; frozen log + fixed memory + no listener =
  check the right node before assuming a hang.
- Profile dumps land in `/tmp/checkpoints/profiles/...` on the rank-0 node with NO
  per-run/arm separation — later runs overwrite earlier ones. Copy off `/tmp`
  immediately after each arm.

## Results

### Driver (control windows, tok/s/GPU)

| window | off | on | stress35 |
|---|---:|---:|---:|
| warmup | 104.1 | 92.6 | 92.7 |
| c0 | 254.0 | 265.3 | 258.6 |
| c1 | 292.9 | 284.3 | 292.3 |
| c2 | 282.2 | 289.4 | 297.2 |
| c3 | 277.0 | 305.2 | 315.7 |
| c4 | 306.8 | 263.9 | 318.0 |
| c5 | 301.7 | 294.2 | 309.6 |
| c6 | 305.3 | 307.4 | 319.1 |
| mean controls | 288.6 | 287.1 | 301.5 |
| memory-profile fb | 29.7s | 30.4s | 29.0s |
| runtime-profile fb | 29.2s | 28.5s | 26.8s |

Driver headlines: off 287, on 286, stress35 300 tok/s/GPU. The stress35 edge is
boot-order/box-drift variance, NOT a GC effect (the GC never fired — below).

### Memory

- NVML whole-GPU peak per node (GiB): off 138.0/125.7/123.1/132.3,
  on 138.2/126.1/123.2/131.9, stress35 138.2/125.9/123.5/131.7. Max across all arms: 138.2.
- Torch rank-0 snapshot: off reserved 111.6 / live-end 82.6 GiB;
  on reserved 112.9 / live-end 82.6; stress35 reserved 112.8 / live-end 82.6.
  Pickles confirm arm separation via allocator settings (gc_thresh 0.0 / 0.63 / 0.35).
- Out-of-pool footprint (NVML − reserved) ≈ 26 GiB at CP32/EP32 (bigger than the 5–13 GiB
  seen on the 131k PP2 runs — more parallel groups → more NCCL channels).

### The decisive negative: the GC never fired in ANY arm

Allocator event histories from the three rank-0 pickles are **byte-for-byte identical**:
320,447 alloc / free_requested / free_completed events each, zero segment-release /
decommit events, including stress35 whose trigger (0.35 × 268.6 = 94 GiB) sat 19 GiB
BELOW the measured reserved peak (112.8 GiB) for the entire profiled step.

Conclusion about the mechanism (this PyTorch build): `garbage_collection_threshold`
does not sweep on cache hits. In steady state every allocation is served from the stash
(320k/320k hits), no sweep ever runs, and the knob is a strict no-op no matter how far
reserved sits above the threshold. It only engages on the pressure path — a cache miss
that would otherwise grow reserved — i.e., exactly the LPS-1065 regime. The Aug-24
"thrash when placed below the working set" model does not materialize: there is no
free/re-fetch churn because there are no sweeps without misses.

## Verdicts vs d.md success criteria

- TPS parity ±3%: **PASS** (on/off means 287.1 vs 288.6, −0.5%; stress35 301.5 is within
  boot-to-boot drift; per-window jitter comparable across arms, no progressive
  degradation in any arm).
- NVML peak ≤175 GiB: **PASS, trivially** — all arms peak ≈138 GiB. The golden B200 256k
  config (4 nodes, CP32/EP32) peaks at ~77% of a B200's ~179 GiB usable. It never rode
  the B200 ceiling; the LPS-1065 crash regime (178/179) was a different, tighter shape.
- Doesn't churn: **PASS by mechanism** — the knob provably did nothing at steady state,
  even deliberately mis-placed at 0.35. No sweeps → no churn → identical TPS/memory.
- "It works" (protection): **NOT VALIDATED** — untestable at this shape on this card.
  The 0.63 "simulate B200" premise was voided by the config's real peak (138 GiB, not
  ~179). Protection requires a shape whose allocator keeps missing near the card ceiling
  (e.g. the 2-node B300 256k config that peaked ~263 GiB in the Aug session) — a
  follow-up run, out of d.md's scope.

## Net assessment for PR #1181

- The delivery mechanism works end-to-end: env var → launch.sh → torchrun → allocator
  (verified in the on-arm pickle's allocator_settings).
- Setting the knob fleet-wide is safe: at comfortable shapes it is a proven no-op
  (zero allocator events, zero TPS effect, zero memory effect — measured, not argued).
- The LPS-1065 protection claim remains reasoned, not measured. If we want it measured:
  run the tight 2-node B300 shape with and without the threshold and watch whether the
  sweep bends the NVML climb / prevents the starvation OOM.

## Artifacts

- `driver/gc-{off,on,stress35}-256k.json` + `.log` — full per-window data.
- `driver/gc-off-256k-profiles.json` + `.log` — off-arm profile re-run (see incident
  below); its TPS is NOT comparable to the main off-arm run — use only its trace/pickle.
- `nvml/<arm>/<host>.jsonl` — 10 Hz pynvml whole-GPU memory, all 4 nodes, full duration.
- `pickles/gc-<arm>-256k.memory.rank0.pickle` — torch allocator snapshots (rank 0).
- `traces/gc-<arm>-256k.pt.trace.json.gz` — kineto traces (gzip ~57 MB each).
- `trainer-config-glm52-b200-256k.json` — exact trainer config used for all arms.

## Incident: off-arm profile artifacts overwritten, re-captured

The trainer's profile dumps land in `/tmp/checkpoints/profiles/...` on the rank-0 node
with no per-run separation — the ON arm's dumps overwrote the OFF arm's trace+pickle
before they were copied off /tmp. The off-arm profiles were re-captured as
`gc-off-256k-profiles` (minimal 1-control re-run). In future runs, copy dumps off /tmp
immediately after each arm.
