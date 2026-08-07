# LPS-1062 — GLM-5.2 B300 baseline perf profile

**Artifacts live locally on Jack's laptop (not committed — the 995 MB kineto trace exceeds GitHub's 100 MB blob limit):**

```
/Users/jackrao/perf_profiles/lps-1062/
```

## What's there

```
lps-1062/
├── trainer-config.json          # exact trainer config (golden B300: TP1/PP1/EP16/CP16, 256k, LoRA r32)
├── trainer-server-config.json
├── profile_driver.py            # synthetic-data benchmark + profiler driver (HTTP)
├── devbox-up.log                # devbox q480z53 provisioning log
└── glm52-b300-s256k/
    ├── REPORT.md                # full writeup (numbers, bottlenecks, artifact index)
    ├── results.json             # raw window timings + profiler responses
    ├── mfu_calc.py              # reproducible MFU math (745B total / 42.1B active params)
    ├── trainer_srun.log         # trainer boot log
    ├── profile_driver.log
    ├── node0/*.pt.trace.json    # rank-0 kineto trace, one full step (995 MB, open in Perfetto)
    ├── node0/memory.rank0.pickle
    └── node1/memory.rank8.pickle
```

## Headline (2026-08-06, devbox q480z53, 2×8 B300 ali, trainers_main @ 0e0b65a6)

- **446 tok/s/GPU** (7,134 tok/s), **73.5 s** per 524k-token step, optim 0.06 s
- **MFU ≈ 7.5–9.4%** — ~3× off the 20–30% target
- **NCCL = 65% of step**; EP all-to-all `SendRecv` alone = 59% (uniform ~34 ms/call ≈ 47 GB/s → RoCE-bandwidth-bound)
- 26,684 `aten::nonzero` GPU syncs/step (35 s CPU) — kills comm/compute overlap
- 8 vocab-shaped FP32 SIMT GEMMs/step (~1.85 s) — CE/LM-head path off tensor cores
- Memory: 131 GB persistent, 172 GB alloc peak, **260 GB reserved / 275 GB** (95%)
